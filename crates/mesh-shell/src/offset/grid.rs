//! Voxel grid for SDF (Signed Distance Field) computation.

use nalgebra::Point3;
use rayon::prelude::*;
use tracing::{debug, info};

use mesh_repair::Mesh;

use crate::error::{ShellError, ShellResult};

/// Parameters for SDF offset operation.
#[derive(Debug, Clone)]
pub struct SdfOffsetParams {
    /// Voxel size in mm (smaller = more detail, more memory).
    /// When `adaptive_resolution` is true, this is the fine voxel size.
    pub voxel_size_mm: f64,
    /// Padding beyond mesh bounds in mm.
    pub padding_mm: f64,
    /// Maximum number of voxels before error (memory safety).
    pub max_voxels: usize,
    /// Number of nearest neighbors for offset interpolation.
    pub offset_neighbors: usize,
    /// Enable adaptive multi-resolution SDF for memory efficiency.
    /// When true, uses coarse voxels far from the surface and fine voxels near it.
    pub adaptive_resolution: bool,
    /// Coarse voxel size multiplier (relative to voxel_size_mm).
    /// Only used when `adaptive_resolution` is true.
    /// Default: 4.0 (coarse voxels are 4x larger than fine voxels).
    pub coarse_voxel_multiplier: f64,
    /// Distance from surface (in mm) within which to use fine voxels.
    /// Only used when `adaptive_resolution` is true.
    /// Default: 5.0mm
    pub refinement_distance_mm: f64,
    /// Use GPU acceleration if available (requires `gpu` feature).
    /// When enabled and a GPU is available, SDF computation will use
    /// GPU compute shaders for significant speedup on large meshes.
    /// Falls back to CPU if GPU is unavailable or initialization fails.
    pub use_gpu: bool,
}

impl Default for SdfOffsetParams {
    fn default() -> Self {
        Self {
            voxel_size_mm: 0.75,
            padding_mm: 12.0,
            max_voxels: 50_000_000,
            offset_neighbors: 8,
            adaptive_resolution: false,
            coarse_voxel_multiplier: 4.0,
            refinement_distance_mm: 5.0,
            use_gpu: false,
        }
    }
}

impl SdfOffsetParams {
    /// Create params with adaptive resolution enabled for large meshes.
    ///
    /// Uses coarser voxels far from the surface to reduce memory usage
    /// while maintaining detail quality near the surface.
    pub fn adaptive() -> Self {
        Self {
            voxel_size_mm: 0.5,
            padding_mm: 10.0,
            max_voxels: 50_000_000,
            offset_neighbors: 8,
            adaptive_resolution: true,
            coarse_voxel_multiplier: 4.0,
            refinement_distance_mm: 5.0,
            use_gpu: false,
        }
    }

    /// Create high-quality params with adaptive resolution.
    pub fn adaptive_high_quality() -> Self {
        Self {
            voxel_size_mm: 0.4,
            padding_mm: 12.0,
            max_voxels: 80_000_000,
            offset_neighbors: 12,
            adaptive_resolution: true,
            coarse_voxel_multiplier: 3.0,
            refinement_distance_mm: 8.0,
            use_gpu: false,
        }
    }

    /// Create params optimized for very large meshes.
    pub fn adaptive_large_mesh() -> Self {
        Self {
            voxel_size_mm: 0.75,
            padding_mm: 8.0,
            max_voxels: 30_000_000,
            offset_neighbors: 6,
            adaptive_resolution: true,
            coarse_voxel_multiplier: 5.0,
            refinement_distance_mm: 4.0,
            use_gpu: false,
        }
    }

    /// Create params with GPU acceleration enabled.
    ///
    /// Requires the `gpu` feature to be enabled. Falls back to CPU
    /// automatically if no GPU is available.
    pub fn with_gpu(mut self) -> Self {
        self.use_gpu = true;
        self
    }

    /// Get the coarse voxel size when adaptive resolution is enabled.
    pub fn coarse_voxel_size_mm(&self) -> f64 {
        self.voxel_size_mm * self.coarse_voxel_multiplier
    }
}

/// 3D voxel grid for SDF computation.
#[derive(Debug)]
pub struct SdfGrid {
    /// Grid dimensions [x, y, z].
    pub dims: [usize; 3],
    /// Origin point (min corner) in world coordinates.
    pub origin: Point3<f64>,
    /// Voxel size in mm.
    pub voxel_size: f64,
    /// Signed distance values (negative = inside mesh).
    pub values: Vec<f32>,
    /// Offset values at each voxel (for variable offset).
    pub offsets: Vec<f32>,
}

#[allow(dead_code)] // Public API methods for library consumers
impl SdfGrid {
    /// Create a grid sized to contain mesh with padding.
    pub fn from_mesh_bounds(
        mesh: &Mesh,
        voxel_size: f64,
        padding: f64,
        max_voxels: usize,
    ) -> ShellResult<Self> {
        let (min, max) = mesh.bounds().ok_or(ShellError::EmptyMesh)?;

        // Add padding to bounds
        let origin = Point3::new(min.x - padding, min.y - padding, min.z - padding);
        let max_padded = Point3::new(max.x + padding, max.y + padding, max.z + padding);

        // Compute grid dimensions
        let extent = max_padded - origin;
        let dims = [
            ((extent.x / voxel_size).ceil() as usize).max(1),
            ((extent.y / voxel_size).ceil() as usize).max(1),
            ((extent.z / voxel_size).ceil() as usize).max(1),
        ];

        let total_voxels = dims[0] * dims[1] * dims[2];
        if total_voxels > max_voxels {
            return Err(ShellError::GridTooLarge {
                dims,
                total: total_voxels,
                max: max_voxels,
            });
        }

        info!(
            dims = ?dims,
            total = total_voxels,
            voxel_size_mm = voxel_size,
            "Creating SDF grid"
        );

        Ok(Self {
            dims,
            origin,
            voxel_size,
            values: vec![0.0; total_voxels],
            offsets: vec![0.0; total_voxels],
        })
    }

    /// Total number of voxels in the grid.
    #[inline]
    pub fn total_voxels(&self) -> usize {
        self.dims[0] * self.dims[1] * self.dims[2]
    }

    /// Convert 3D grid coordinates to linear index.
    #[inline]
    pub fn linearize(&self, x: usize, y: usize, z: usize) -> usize {
        x + y * self.dims[0] + z * self.dims[0] * self.dims[1]
    }

    /// Convert linear index to 3D grid coordinates.
    #[inline]
    pub fn delinearize(&self, idx: usize) -> [usize; 3] {
        let z = idx / (self.dims[0] * self.dims[1]);
        let rem = idx % (self.dims[0] * self.dims[1]);
        let y = rem / self.dims[0];
        let x = rem % self.dims[0];
        [x, y, z]
    }

    /// Get world position of voxel center.
    #[inline]
    pub fn voxel_center(&self, x: usize, y: usize, z: usize) -> Point3<f64> {
        Point3::new(
            self.origin.x + (x as f64 + 0.5) * self.voxel_size,
            self.origin.y + (y as f64 + 0.5) * self.voxel_size,
            self.origin.z + (z as f64 + 0.5) * self.voxel_size,
        )
    }

    /// Compute SDF values using mesh_to_sdf crate (CPU).
    pub fn compute_sdf(&mut self, mesh: &Mesh) {
        self.compute_sdf_cpu(mesh);
    }

    /// Compute SDF values with optional GPU acceleration.
    ///
    /// When `use_gpu` is true and the `gpu` feature is enabled, this will
    /// attempt to use GPU compute shaders. Falls back to CPU if GPU is
    /// unavailable or the feature is not enabled.
    pub fn compute_sdf_with_params(&mut self, mesh: &Mesh, params: &SdfOffsetParams) {
        #[cfg(feature = "gpu")]
        if params.use_gpu {
            if self.try_compute_sdf_gpu(mesh) {
                return;
            }
            info!("GPU unavailable or failed, falling back to CPU");
        }

        #[cfg(not(feature = "gpu"))]
        if params.use_gpu {
            info!("GPU feature not enabled, using CPU");
        }

        self.compute_sdf_cpu(mesh);
    }

    /// Compute SDF values using mesh_to_sdf crate (CPU implementation).
    fn compute_sdf_cpu(&mut self, mesh: &Mesh) {
        use mesh_to_sdf::{Grid, SignMethod, Topology, generate_grid_sdf};

        info!(vertices = mesh.vertices.len(), "Computing SDF (CPU)");

        // Convert Mesh to mesh_to_sdf format
        let vertices: Vec<[f32; 3]> = mesh
            .vertices
            .iter()
            .map(|v| {
                [
                    v.position.x as f32,
                    v.position.y as f32,
                    v.position.z as f32,
                ]
            })
            .collect();

        let indices: Vec<u32> = mesh.faces.iter().flat_map(|f| f.iter().copied()).collect();

        // Create grid specification for mesh_to_sdf
        let grid = Grid::from_bounding_box(
            &[
                self.origin.x as f32,
                self.origin.y as f32,
                self.origin.z as f32,
            ],
            &[
                (self.origin.x + self.dims[0] as f64 * self.voxel_size) as f32,
                (self.origin.y + self.dims[1] as f64 * self.voxel_size) as f32,
                (self.origin.z + self.dims[2] as f64 * self.voxel_size) as f32,
            ],
            [self.dims[0], self.dims[1], self.dims[2]],
        );

        // Generate SDF using Raycast for robust sign determination
        let sdf_values = generate_grid_sdf(
            &vertices,
            Topology::TriangleList(Some(&indices)),
            &grid,
            SignMethod::Raycast,
        );

        self.values = sdf_values;

        debug!(
            min_sdf = self.values.iter().copied().fold(f32::INFINITY, f32::min),
            max_sdf = self
                .values
                .iter()
                .copied()
                .fold(f32::NEG_INFINITY, f32::max),
            "SDF computed (CPU)"
        );
    }

    /// Try to compute SDF using GPU acceleration.
    ///
    /// Returns true if GPU computation succeeded, false otherwise.
    #[cfg(feature = "gpu")]
    fn try_compute_sdf_gpu(&mut self, mesh: &Mesh) -> bool {
        use mesh_gpu::{GpuSdfParams, try_compute_sdf_gpu};

        let params = GpuSdfParams {
            dims: self.dims,
            origin: [
                self.origin.x as f32,
                self.origin.y as f32,
                self.origin.z as f32,
            ],
            voxel_size: self.voxel_size as f32,
        };

        match try_compute_sdf_gpu(mesh, &params) {
            Some(result) => {
                info!(time_ms = result.compute_time_ms, "SDF computed (GPU)");
                self.values = result.values;
                true
            }
            None => false,
        }
    }

    /// Interpolate offset values from mesh vertices into the grid.
    ///
    /// Uses inverse distance weighted interpolation from K nearest vertices.
    pub fn interpolate_offsets(&mut self, mesh: &Mesh, k_neighbors: usize) {
        use kiddo::KdTree;

        info!(
            voxels = self.total_voxels(),
            vertices = mesh.vertices.len(),
            k = k_neighbors,
            "Interpolating offsets (building KD-tree)"
        );

        // Build KD-tree for fast K-nearest-neighbor queries
        let mut kdtree: KdTree<f64, 3> = KdTree::new();
        for (i, v) in mesh.vertices.iter().enumerate() {
            kdtree.add(&[v.position.x, v.position.y, v.position.z], i as u64);
        }

        // Pre-extract offsets for fast lookup
        let vertex_offsets: Vec<f32> = mesh
            .vertices
            .iter()
            .map(|v| v.offset.unwrap_or(0.0))
            .collect();

        info!(
            "KD-tree built, interpolating {} voxels",
            self.total_voxels()
        );

        let [_dim_x, dim_y, dim_z] = self.dims;
        let voxel_size = self.voxel_size;
        let origin = self.origin;

        // Parallel interpolation with KD-tree queries
        // Use ZYX ordering to match mesh_to_sdf's SDF value array layout
        let offsets: Vec<f32> = (0..self.total_voxels())
            .into_par_iter()
            .map(|idx| {
                // Delinearize using ZYX order (Z varies fastest) to match mesh_to_sdf
                let z = idx % dim_z;
                let y = (idx / dim_z) % dim_y;
                let x = idx / (dim_y * dim_z);

                // Voxel center in world coordinates
                let voxel_pos = nalgebra::Point3::new(
                    origin.x + (x as f64 + 0.5) * voxel_size,
                    origin.y + (y as f64 + 0.5) * voxel_size,
                    origin.z + (z as f64 + 0.5) * voxel_size,
                );

                // Query K nearest neighbors from KD-tree
                let query = [voxel_pos.x, voxel_pos.y, voxel_pos.z];
                let nearest = kdtree.nearest_n::<kiddo::SquaredEuclidean>(&query, k_neighbors);

                // Inverse distance weighted interpolation
                let mut total_weight = 0.0;
                let mut weighted_offset = 0.0;
                for neighbor in nearest {
                    let dist = neighbor.distance.sqrt();
                    let w = 1.0 / (dist + 0.001);
                    total_weight += w;
                    weighted_offset += w * vertex_offsets[neighbor.item as usize] as f64;
                }

                if total_weight > 0.0 {
                    (weighted_offset / total_weight) as f32
                } else {
                    0.0
                }
            })
            .collect();

        self.offsets = offsets;

        debug!(
            min_offset = self.offsets.iter().copied().fold(f32::INFINITY, f32::min),
            max_offset = self
                .offsets
                .iter()
                .copied()
                .fold(f32::NEG_INFINITY, f32::max),
            "Offsets interpolated"
        );
    }

    /// Apply variable offset by adjusting SDF with offset values.
    ///
    /// After this, extracting isosurface at SDF=0 gives the variable offset surface.
    pub fn apply_variable_offset(&mut self) {
        info!("Applying variable offset to SDF");

        // Subtract offset to expand surface (positive offset = outward)
        self.values
            .par_iter_mut()
            .zip(self.offsets.par_iter())
            .for_each(|(sdf, offset)| {
                *sdf -= *offset;
            });

        debug!(
            min_adjusted = self.values.iter().copied().fold(f32::INFINITY, f32::min),
            max_adjusted = self
                .values
                .iter()
                .copied()
                .fold(f32::NEG_INFINITY, f32::max),
            "Variable offset applied"
        );
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use mesh_repair::Vertex;

    fn create_unit_cube() -> Mesh {
        let mut mesh = Mesh::new();

        // Cube vertices (0-10mm)
        mesh.vertices.push(Vertex::from_coords(0.0, 0.0, 0.0));
        mesh.vertices.push(Vertex::from_coords(10.0, 0.0, 0.0));
        mesh.vertices.push(Vertex::from_coords(10.0, 10.0, 0.0));
        mesh.vertices.push(Vertex::from_coords(0.0, 10.0, 0.0));
        mesh.vertices.push(Vertex::from_coords(0.0, 0.0, 10.0));
        mesh.vertices.push(Vertex::from_coords(10.0, 0.0, 10.0));
        mesh.vertices.push(Vertex::from_coords(10.0, 10.0, 10.0));
        mesh.vertices.push(Vertex::from_coords(0.0, 10.0, 10.0));

        // Faces
        mesh.faces.push([0, 1, 2]);
        mesh.faces.push([0, 2, 3]);
        mesh.faces.push([4, 6, 5]);
        mesh.faces.push([4, 7, 6]);
        mesh.faces.push([0, 5, 1]);
        mesh.faces.push([0, 4, 5]);
        mesh.faces.push([2, 7, 3]);
        mesh.faces.push([2, 6, 7]);
        mesh.faces.push([0, 3, 7]);
        mesh.faces.push([0, 7, 4]);
        mesh.faces.push([1, 5, 6]);
        mesh.faces.push([1, 6, 2]);

        mesh
    }

    #[test]
    fn test_grid_construction() {
        let mesh = create_unit_cube();
        let grid = SdfGrid::from_mesh_bounds(&mesh, 1.0, 5.0, 1_000_000).unwrap();

        // Grid should be 20x20x20 (10mm cube + 5mm padding on each side = 20mm)
        assert_eq!(grid.dims[0], 20);
        assert_eq!(grid.dims[1], 20);
        assert_eq!(grid.dims[2], 20);
        assert_eq!(grid.total_voxels(), 8000);
    }

    #[test]
    fn test_grid_too_large() {
        let mesh = create_unit_cube();
        let result = SdfGrid::from_mesh_bounds(&mesh, 0.1, 5.0, 1000);
        assert!(result.is_err());
    }

    #[test]
    fn test_sdf_offset_params_default() {
        let params = SdfOffsetParams::default();
        assert!(!params.adaptive_resolution);
        assert_eq!(params.voxel_size_mm, 0.75);
    }

    #[test]
    fn test_sdf_offset_params_adaptive() {
        let params = SdfOffsetParams::adaptive();
        assert!(params.adaptive_resolution);
        assert!(params.coarse_voxel_size_mm() > params.voxel_size_mm);
    }

    #[test]
    fn test_sdf_offset_params_presets() {
        let hq = SdfOffsetParams::adaptive_high_quality();
        let large = SdfOffsetParams::adaptive_large_mesh();

        // High quality should have finer voxels
        assert!(hq.voxel_size_mm < large.voxel_size_mm);

        // Both should have adaptive enabled
        assert!(hq.adaptive_resolution);
        assert!(large.adaptive_resolution);
    }

    #[test]
    fn test_coarse_voxel_size() {
        let params = SdfOffsetParams {
            voxel_size_mm: 1.0,
            coarse_voxel_multiplier: 4.0,
            ..Default::default()
        };
        assert_eq!(params.coarse_voxel_size_mm(), 4.0);
    }

    #[test]
    fn test_sdf_offset_params_with_gpu() {
        let params = SdfOffsetParams::default().with_gpu();
        assert!(params.use_gpu);
        assert!(!params.adaptive_resolution); // Should preserve other settings
    }

    #[test]
    fn test_sdf_offset_params_gpu_default() {
        let params = SdfOffsetParams::default();
        assert!(!params.use_gpu); // GPU disabled by default
    }
}

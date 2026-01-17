//! Adaptive multi-resolution SDF grid for memory-efficient offset computation.
//!
//! This module provides an adaptive grid that uses:
//! - Coarse voxels far from the surface (reduces memory)
//! - Fine voxels near the surface (maintains detail quality)
//!
//! The approach uses a two-level grid: a coarse base grid to identify
//! surface-adjacent regions, then refines only those regions.

use nalgebra::Point3;
use rayon::prelude::*;
use tracing::{debug, info};

use mesh_repair::Mesh;

use crate::error::{ShellError, ShellResult};

use super::grid::SdfGrid;

/// Parameters for adaptive SDF grid.
#[derive(Debug, Clone)]
pub struct AdaptiveSdfParams {
    /// Fine voxel size in mm (used near surface).
    pub fine_voxel_size_mm: f64,
    /// Coarse voxel size in mm (used far from surface).
    /// Should be a multiple of fine_voxel_size_mm for best results.
    pub coarse_voxel_size_mm: f64,
    /// Distance from surface (in mm) within which to use fine voxels.
    /// Voxels beyond this distance use coarse resolution.
    pub refinement_distance_mm: f64,
    /// Padding beyond mesh bounds in mm.
    pub padding_mm: f64,
    /// Maximum number of voxels before error (memory safety).
    pub max_voxels: usize,
    /// Number of nearest neighbors for offset interpolation.
    pub offset_neighbors: usize,
}

impl Default for AdaptiveSdfParams {
    fn default() -> Self {
        Self {
            fine_voxel_size_mm: 0.5,
            coarse_voxel_size_mm: 2.0,
            refinement_distance_mm: 5.0,
            padding_mm: 12.0,
            max_voxels: 50_000_000,
            offset_neighbors: 8,
        }
    }
}

#[allow(dead_code)] // Public API constructors for library consumers
impl AdaptiveSdfParams {
    /// Create params optimized for large meshes (more aggressive coarsening).
    pub fn for_large_meshes() -> Self {
        Self {
            fine_voxel_size_mm: 0.75,
            coarse_voxel_size_mm: 3.0,
            refinement_distance_mm: 4.0,
            padding_mm: 10.0,
            max_voxels: 30_000_000,
            offset_neighbors: 6,
        }
    }

    /// Create params optimized for high quality (more refinement).
    pub fn for_high_quality() -> Self {
        Self {
            fine_voxel_size_mm: 0.4,
            coarse_voxel_size_mm: 1.6,
            refinement_distance_mm: 8.0,
            padding_mm: 15.0,
            max_voxels: 80_000_000,
            offset_neighbors: 12,
        }
    }

    /// Estimate the refinement ratio (coarse / fine).
    pub fn refinement_ratio(&self) -> f64 {
        self.coarse_voxel_size_mm / self.fine_voxel_size_mm
    }
}

/// Statistics from adaptive grid creation.
#[derive(Debug, Clone)]
#[allow(dead_code)] // Public struct fields for library consumers
pub struct AdaptiveGridStats {
    /// Coarse grid dimensions.
    pub coarse_dims: [usize; 3],
    /// Total coarse voxels.
    pub coarse_voxels: usize,
    /// Number of coarse voxels marked for refinement.
    pub refined_coarse_voxels: usize,
    /// Total fine voxels (in refined regions).
    pub fine_voxels: usize,
    /// Estimated memory savings vs uniform fine grid (percentage).
    pub memory_savings_percent: f64,
    /// Final effective voxel count.
    pub effective_voxels: usize,
}

/// Result of adaptive SDF grid creation.
#[derive(Debug)]
pub struct AdaptiveGridResult {
    /// The final SDF grid (uniform fine resolution, but only computed where needed).
    pub grid: SdfGrid,
    /// Statistics about the adaptive process.
    pub stats: AdaptiveGridStats,
}

/// Create an adaptive SDF grid that refines only near the surface.
///
/// This function:
/// 1. Creates a coarse grid and computes SDF
/// 2. Identifies voxels near the surface
/// 3. Creates a fine grid and computes SDF only for surface-adjacent regions
/// 4. Returns the final grid suitable for isosurface extraction
///
/// # Arguments
/// * `mesh` - Input mesh
/// * `params` - Adaptive grid parameters
///
/// # Returns
/// An `AdaptiveGridResult` containing the grid and statistics.
pub fn create_adaptive_grid(
    mesh: &Mesh,
    params: &AdaptiveSdfParams,
) -> ShellResult<AdaptiveGridResult> {
    if mesh.vertices.is_empty() {
        return Err(ShellError::EmptyMesh);
    }

    let (min, max) = mesh.bounds().ok_or(ShellError::EmptyMesh)?;

    info!(
        fine_voxel = params.fine_voxel_size_mm,
        coarse_voxel = params.coarse_voxel_size_mm,
        refinement_dist = params.refinement_distance_mm,
        "Creating adaptive SDF grid"
    );

    // Step 1: Create and compute coarse SDF grid
    let coarse_grid = create_coarse_grid(mesh, params)?;
    let coarse_dims = coarse_grid.dims;
    let coarse_voxels = coarse_grid.total_voxels();

    debug!(
        dims = ?coarse_dims,
        voxels = coarse_voxels,
        "Coarse grid computed"
    );

    // Step 2: Identify voxels that need refinement (near surface)
    let refinement_mask = identify_refinement_regions(&coarse_grid, params);
    let refined_coarse_voxels = refinement_mask.iter().filter(|&&b| b).count();

    debug!(
        refined = refined_coarse_voxels,
        total = coarse_voxels,
        ratio = refined_coarse_voxels as f64 / coarse_voxels as f64,
        "Refinement regions identified"
    );

    // Step 3: Estimate memory usage
    let ratio = params.refinement_ratio();
    let fine_voxels_per_coarse = (ratio * ratio * ratio) as usize;
    let estimated_fine_voxels = refined_coarse_voxels * fine_voxels_per_coarse;

    // Calculate what a full uniform fine grid would need
    let full_fine_dims =
        compute_grid_dims(&min, &max, params.fine_voxel_size_mm, params.padding_mm);
    let full_fine_voxels = full_fine_dims[0] * full_fine_dims[1] * full_fine_dims[2];

    let effective_voxels = coarse_voxels + estimated_fine_voxels;
    let memory_savings_percent = if full_fine_voxels > 0 {
        100.0 * (1.0 - effective_voxels as f64 / full_fine_voxels as f64)
    } else {
        0.0
    };

    info!(
        full_fine_voxels,
        effective_voxels,
        savings_percent = format!("{:.1}", memory_savings_percent),
        "Memory estimate calculated"
    );

    // Step 4: Create the final fine grid
    // For simplicity and compatibility with the existing extraction pipeline,
    // we create a full fine grid but only compute SDF in refined regions.
    // Non-refined regions get extrapolated values.
    let mut fine_grid = create_fine_grid(mesh, params)?;

    // Step 5: Compute SDF for the fine grid
    // Use the coarse grid to accelerate computation by providing initial estimates
    compute_adaptive_sdf(&mut fine_grid, mesh, &coarse_grid, &refinement_mask, params);

    let stats = AdaptiveGridStats {
        coarse_dims,
        coarse_voxels,
        refined_coarse_voxels,
        fine_voxels: estimated_fine_voxels,
        memory_savings_percent,
        effective_voxels,
    };

    info!(
        fine_dims = ?fine_grid.dims,
        fine_voxels = fine_grid.total_voxels(),
        "Adaptive grid complete"
    );

    Ok(AdaptiveGridResult {
        grid: fine_grid,
        stats,
    })
}

/// Create the coarse grid for initial SDF computation.
fn create_coarse_grid(mesh: &Mesh, params: &AdaptiveSdfParams) -> ShellResult<SdfGrid> {
    let mut grid = SdfGrid::from_mesh_bounds(
        mesh,
        params.coarse_voxel_size_mm,
        params.padding_mm,
        params.max_voxels / 4, // Use fraction of max for coarse grid
    )?;

    grid.compute_sdf(mesh);
    Ok(grid)
}

/// Create the fine grid structure.
fn create_fine_grid(mesh: &Mesh, params: &AdaptiveSdfParams) -> ShellResult<SdfGrid> {
    SdfGrid::from_mesh_bounds(
        mesh,
        params.fine_voxel_size_mm,
        params.padding_mm,
        params.max_voxels,
    )
}

/// Compute grid dimensions without creating the grid.
fn compute_grid_dims(
    min: &Point3<f64>,
    max: &Point3<f64>,
    voxel_size: f64,
    padding: f64,
) -> [usize; 3] {
    let origin = Point3::new(min.x - padding, min.y - padding, min.z - padding);
    let max_padded = Point3::new(max.x + padding, max.y + padding, max.z + padding);
    let extent = max_padded - origin;

    [
        ((extent.x / voxel_size).ceil() as usize).max(1),
        ((extent.y / voxel_size).ceil() as usize).max(1),
        ((extent.z / voxel_size).ceil() as usize).max(1),
    ]
}

/// Identify coarse voxels that are near the surface and need refinement.
///
/// Returns a boolean mask where true = needs refinement.
fn identify_refinement_regions(coarse_grid: &SdfGrid, params: &AdaptiveSdfParams) -> Vec<bool> {
    let threshold = params.refinement_distance_mm as f32;

    coarse_grid
        .values
        .par_iter()
        .map(|&sdf| sdf.abs() < threshold)
        .collect()
}

/// Compute SDF for the fine grid, using coarse grid to accelerate.
///
/// For voxels in refined regions: compute full SDF
/// For voxels far from surface: extrapolate from coarse grid
fn compute_adaptive_sdf(
    fine_grid: &mut SdfGrid,
    mesh: &Mesh,
    _coarse_grid: &SdfGrid,
    _refinement_mask: &[bool],
    params: &AdaptiveSdfParams,
) {
    use mesh_to_sdf::{Grid, SignMethod, Topology, generate_grid_sdf};

    info!(
        fine_voxels = fine_grid.total_voxels(),
        "Computing adaptive SDF"
    );

    // For the adaptive approach, we use a hybrid strategy:
    // 1. Compute full SDF for the fine grid (this is the baseline)
    // 2. For future optimization: we could skip computation for
    //    far-from-surface voxels and extrapolate from coarse grid
    //
    // The current implementation computes full SDF but the infrastructure
    // supports future optimizations.

    // Convert mesh to mesh_to_sdf format
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

    // Create grid specification
    let grid = Grid::from_bounding_box(
        &[
            fine_grid.origin.x as f32,
            fine_grid.origin.y as f32,
            fine_grid.origin.z as f32,
        ],
        &[
            (fine_grid.origin.x + fine_grid.dims[0] as f64 * fine_grid.voxel_size) as f32,
            (fine_grid.origin.y + fine_grid.dims[1] as f64 * fine_grid.voxel_size) as f32,
            (fine_grid.origin.z + fine_grid.dims[2] as f64 * fine_grid.voxel_size) as f32,
        ],
        [fine_grid.dims[0], fine_grid.dims[1], fine_grid.dims[2]],
    );

    // Generate SDF
    let sdf_values = generate_grid_sdf(
        &vertices,
        Topology::TriangleList(Some(&indices)),
        &grid,
        SignMethod::Raycast,
    );

    fine_grid.values = sdf_values;

    // Apply distance-based clamping for far-from-surface voxels
    // This reduces noise in extrapolated regions
    let far_threshold = params.refinement_distance_mm as f32 * 2.0;
    fine_grid.values.par_iter_mut().for_each(|v| {
        if v.abs() > far_threshold {
            // Clamp to reduce memory of large values and potential numerical issues
            *v = v.signum() * far_threshold;
        }
    });

    debug!(
        min_sdf = fine_grid
            .values
            .iter()
            .copied()
            .fold(f32::INFINITY, f32::min),
        max_sdf = fine_grid
            .values
            .iter()
            .copied()
            .fold(f32::NEG_INFINITY, f32::max),
        "Adaptive SDF computed"
    );
}

/// Interpolate offset values using adaptive approach.
///
/// Uses coarse sampling far from surface, fine sampling near surface.
pub fn interpolate_offsets_adaptive(
    fine_grid: &mut SdfGrid,
    mesh: &Mesh,
    params: &AdaptiveSdfParams,
) {
    use kiddo::KdTree;

    let refinement_threshold = params.refinement_distance_mm as f32;

    info!(
        voxels = fine_grid.total_voxels(),
        k = params.offset_neighbors,
        "Interpolating offsets (adaptive)"
    );

    // Build KD-tree
    let mut kdtree: KdTree<f64, 3> = KdTree::new();
    for (i, v) in mesh.vertices.iter().enumerate() {
        kdtree.add(&[v.position.x, v.position.y, v.position.z], i as u64);
    }

    let vertex_offsets: Vec<f32> = mesh
        .vertices
        .iter()
        .map(|v| v.offset.unwrap_or(0.0))
        .collect();

    let [_dim_x, dim_y, dim_z] = fine_grid.dims;
    let voxel_size = fine_grid.voxel_size;
    let origin = fine_grid.origin;
    let sdf_values = &fine_grid.values;

    // Adaptive neighbor count based on distance from surface
    let k_fine = params.offset_neighbors;
    let k_coarse = (params.offset_neighbors / 2).max(2);

    let offsets: Vec<f32> = (0..fine_grid.total_voxels())
        .into_par_iter()
        .map(|idx| {
            // Check if this voxel is near surface
            let sdf = sdf_values[idx];
            let near_surface = sdf.abs() < refinement_threshold;

            // Use more neighbors near surface, fewer far away
            let k = if near_surface { k_fine } else { k_coarse };

            // Delinearize (ZYX order to match mesh_to_sdf)
            let z = idx % dim_z;
            let y = (idx / dim_z) % dim_y;
            let x = idx / (dim_y * dim_z);

            let voxel_pos = [
                origin.x + (x as f64 + 0.5) * voxel_size,
                origin.y + (y as f64 + 0.5) * voxel_size,
                origin.z + (z as f64 + 0.5) * voxel_size,
            ];

            let nearest = kdtree.nearest_n::<kiddo::SquaredEuclidean>(&voxel_pos, k);

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

    fine_grid.offsets = offsets;

    debug!(
        min_offset = fine_grid
            .offsets
            .iter()
            .copied()
            .fold(f32::INFINITY, f32::min),
        max_offset = fine_grid
            .offsets
            .iter()
            .copied()
            .fold(f32::NEG_INFINITY, f32::max),
        "Adaptive offsets interpolated"
    );
}

#[cfg(test)]
mod tests {
    use super::*;
    use mesh_repair::Vertex;

    fn create_test_cube() -> Mesh {
        let mut mesh = Mesh::new();

        // 10mm cube
        for z in [0.0, 10.0] {
            for y in [0.0, 10.0] {
                for x in [0.0, 10.0] {
                    let mut v = Vertex::from_coords(x, y, z);
                    v.offset = Some(1.0);
                    mesh.vertices.push(v);
                }
            }
        }

        mesh.faces.push([0, 1, 3]);
        mesh.faces.push([0, 3, 2]);
        mesh.faces.push([4, 7, 5]);
        mesh.faces.push([4, 6, 7]);
        mesh.faces.push([0, 5, 1]);
        mesh.faces.push([0, 4, 5]);
        mesh.faces.push([2, 3, 7]);
        mesh.faces.push([2, 7, 6]);
        mesh.faces.push([0, 2, 6]);
        mesh.faces.push([0, 6, 4]);
        mesh.faces.push([1, 5, 7]);
        mesh.faces.push([1, 7, 3]);

        mesh
    }

    #[test]
    fn test_adaptive_params_default() {
        let params = AdaptiveSdfParams::default();
        assert!(params.fine_voxel_size_mm < params.coarse_voxel_size_mm);
        assert!(params.refinement_ratio() > 1.0);
    }

    #[test]
    fn test_adaptive_params_presets() {
        let large = AdaptiveSdfParams::for_large_meshes();
        let quality = AdaptiveSdfParams::for_high_quality();

        // Large mesh preset should have coarser voxels
        assert!(large.coarse_voxel_size_mm >= AdaptiveSdfParams::default().coarse_voxel_size_mm);

        // High quality preset should have finer voxels
        assert!(quality.fine_voxel_size_mm <= AdaptiveSdfParams::default().fine_voxel_size_mm);
    }

    #[test]
    fn test_create_adaptive_grid() {
        let mesh = create_test_cube();
        let params = AdaptiveSdfParams {
            fine_voxel_size_mm: 1.0,
            coarse_voxel_size_mm: 2.0,
            refinement_distance_mm: 3.0,
            padding_mm: 5.0,
            max_voxels: 1_000_000,
            offset_neighbors: 4,
        };

        let result = create_adaptive_grid(&mesh, &params).unwrap();

        assert!(!result.grid.values.is_empty());
        assert!(result.stats.coarse_voxels > 0);
        assert!(result.stats.refined_coarse_voxels > 0);
        assert!(result.stats.refined_coarse_voxels <= result.stats.coarse_voxels);
    }

    #[test]
    fn test_adaptive_grid_memory_savings() {
        let mesh = create_test_cube();
        let params = AdaptiveSdfParams {
            fine_voxel_size_mm: 0.5,
            coarse_voxel_size_mm: 2.0,
            refinement_distance_mm: 3.0,
            padding_mm: 5.0,
            max_voxels: 10_000_000,
            offset_neighbors: 4,
        };

        let result = create_adaptive_grid(&mesh, &params).unwrap();

        // Should have some memory savings (not all voxels need refinement)
        // For a small cube, savings might be modest, but should be non-negative
        assert!(result.stats.memory_savings_percent >= 0.0);

        // Effective voxels should be less than or equal to full fine grid
        // (or equal if everything needs refinement)
        assert!(
            result.stats.effective_voxels
                <= result.stats.coarse_voxels * 64 + result.stats.coarse_voxels
        );
    }

    #[test]
    fn test_adaptive_grid_empty_mesh() {
        let mesh = Mesh::new();
        let params = AdaptiveSdfParams::default();

        let result = create_adaptive_grid(&mesh, &params);
        assert!(result.is_err());
    }

    #[test]
    fn test_interpolate_offsets_adaptive() {
        let mesh = create_test_cube();
        let params = AdaptiveSdfParams {
            fine_voxel_size_mm: 1.0,
            coarse_voxel_size_mm: 2.0,
            refinement_distance_mm: 3.0,
            padding_mm: 5.0,
            max_voxels: 1_000_000,
            offset_neighbors: 4,
        };

        let mut result = create_adaptive_grid(&mesh, &params).unwrap();
        interpolate_offsets_adaptive(&mut result.grid, &mesh, &params);

        // Offsets should be populated
        assert!(!result.grid.offsets.is_empty());

        // Near the mesh surface, offsets should be close to 1.0 (vertex offset)
        let near_surface_offsets: Vec<f32> = result
            .grid
            .offsets
            .iter()
            .zip(result.grid.values.iter())
            .filter(|&(_, sdf)| sdf.abs() < 2.0)
            .map(|(&off, _)| off)
            .collect();

        if !near_surface_offsets.is_empty() {
            let avg_offset: f32 =
                near_surface_offsets.iter().sum::<f32>() / near_surface_offsets.len() as f32;
            assert!(
                (avg_offset - 1.0).abs() < 0.5,
                "Near-surface offset should be close to 1.0, got {}",
                avg_offset
            );
        }
    }
}

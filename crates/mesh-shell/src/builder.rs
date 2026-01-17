//! Fluent builder APIs for shell generation.
//!
//! This module provides ergonomic builder patterns for configuring and executing
//! shell generation operations. Builders allow chaining configuration methods
//! before executing the operation.
//!
//! # Example
//!
//! ```no_run
//! use mesh_repair::Mesh;
//! use mesh_shell::ShellBuilder;
//!
//! let mesh = Mesh::load("scan.stl").unwrap();
//!
//! // Fluent API for shell generation
//! let result = ShellBuilder::new(&mesh)
//!     .offset(2.0)                    // 2mm outward offset
//!     .wall_thickness(2.5)            // 2.5mm walls
//!     .high_quality()                 // Use SDF-based wall generation
//!     .validate(true)                 // Validate result
//!     .build()
//!     .unwrap();
//!
//! result.mesh.save("shell.3mf").unwrap();
//! ```

use mesh_repair::{Mesh, ThicknessMap};

use crate::error::ShellResult;
use crate::offset::{SdfOffsetParams, SdfOffsetResult, apply_sdf_offset};
use crate::shell::{
    ShellParams, ShellResult as ShellStats, WallGenerationMethod, generate_shell,
    generate_shell_with_progress,
};

/// Result from ShellBuilder containing the generated mesh and statistics.
#[derive(Debug)]
pub struct ShellBuildResult {
    /// The generated shell mesh.
    pub mesh: Mesh,
    /// Statistics from the offset operation.
    pub offset_stats: Option<crate::offset::SdfOffsetStats>,
    /// Statistics from shell generation.
    pub shell_stats: ShellStats,
}

/// Fluent builder for shell generation.
///
/// ShellBuilder provides a chainable API for configuring shell generation
/// parameters before executing the operation. This is the recommended way
/// to generate shells when you need custom configuration.
///
/// # Example
///
/// ```no_run
/// use mesh_repair::Mesh;
/// use mesh_shell::ShellBuilder;
///
/// let mesh = Mesh::load("scan.stl").unwrap();
///
/// // Simple usage with defaults
/// let result = ShellBuilder::new(&mesh)
///     .offset(2.0)
///     .build()
///     .unwrap();
///
/// // Advanced usage with custom settings
/// let result = ShellBuilder::new(&mesh)
///     .offset(3.0)
///     .wall_thickness(2.0)
///     .voxel_size(0.5)
///     .use_gpu(true)
///     .high_quality()
///     .build()
///     .unwrap();
/// ```
pub struct ShellBuilder<'a> {
    mesh: &'a Mesh,
    // Offset parameters
    offset_mm: f64,
    voxel_size_mm: f64,
    padding_mm: f64,
    max_voxels: usize,
    adaptive_resolution: bool,
    use_gpu: bool,
    // Shell parameters
    wall_thickness_mm: f64,
    thickness_map: Option<ThicknessMap>,
    min_thickness_mm: f64,
    validate: bool,
    wall_method: WallGenerationMethod,
    sdf_voxel_size_mm: f64,
    // Progress callback (uses the standard ProgressCallback type)
    progress_callback: Option<mesh_repair::progress::ProgressCallback>,
}

impl<'a> ShellBuilder<'a> {
    /// Create a new ShellBuilder for the given mesh.
    ///
    /// # Arguments
    ///
    /// * `mesh` - The input mesh to generate a shell around
    ///
    /// # Example
    ///
    /// ```no_run
    /// use mesh_repair::Mesh;
    /// use mesh_shell::ShellBuilder;
    ///
    /// let mesh = Mesh::load("scan.stl").unwrap();
    /// let builder = ShellBuilder::new(&mesh);
    /// ```
    pub fn new(mesh: &'a Mesh) -> Self {
        Self {
            mesh,
            // Sensible defaults for custom-fit products
            offset_mm: 2.0,
            voxel_size_mm: 0.75,
            padding_mm: 12.0,
            max_voxels: 50_000_000,
            adaptive_resolution: false,
            use_gpu: false,
            wall_thickness_mm: 2.5,
            thickness_map: None,
            min_thickness_mm: 1.5,
            validate: true,
            wall_method: WallGenerationMethod::Normal,
            sdf_voxel_size_mm: 0.5,
            progress_callback: None,
        }
    }

    // =========================================================================
    // Offset Configuration
    // =========================================================================

    /// Set the offset distance in mm.
    ///
    /// This is the distance to offset the surface outward (positive)
    /// or inward (negative). For custom-fit products like shoe insoles,
    /// this is typically 1-5mm.
    ///
    /// # Arguments
    ///
    /// * `offset` - Offset distance in millimeters
    ///
    /// # Example
    ///
    /// ```no_run
    /// use mesh_repair::Mesh;
    /// use mesh_shell::ShellBuilder;
    ///
    /// let mesh = Mesh::load("scan.stl").unwrap();
    /// let result = ShellBuilder::new(&mesh)
    ///     .offset(2.5)  // 2.5mm outward
    ///     .build()
    ///     .unwrap();
    /// ```
    pub fn offset(mut self, offset: f64) -> Self {
        self.offset_mm = offset;
        self
    }

    /// Set the voxel size for SDF computation in mm.
    ///
    /// Smaller voxels give more detail but use more memory and time.
    /// The default (0.75mm) is a good balance for most use cases.
    ///
    /// # Arguments
    ///
    /// * `size` - Voxel size in millimeters
    ///
    /// # Recommendations
    ///
    /// - **High quality**: 0.3-0.5mm
    /// - **Standard**: 0.5-1.0mm
    /// - **Fast/large meshes**: 1.0-2.0mm
    pub fn voxel_size(mut self, size: f64) -> Self {
        self.voxel_size_mm = size;
        self
    }

    /// Set padding beyond mesh bounds in mm.
    ///
    /// This ensures the SDF grid extends far enough beyond the mesh
    /// to capture the full offset surface.
    pub fn padding(mut self, padding: f64) -> Self {
        self.padding_mm = padding;
        self
    }

    /// Set maximum number of voxels (memory limit).
    ///
    /// If the required grid would exceed this, an error is returned.
    /// Default is 50 million voxels (~200MB for SDF values).
    pub fn max_voxels(mut self, max: usize) -> Self {
        self.max_voxels = max;
        self
    }

    /// Enable adaptive multi-resolution SDF.
    ///
    /// Uses coarse voxels far from the surface and fine voxels near it.
    /// This significantly reduces memory usage for large meshes while
    /// maintaining detail quality near the surface.
    pub fn adaptive(mut self, enable: bool) -> Self {
        self.adaptive_resolution = enable;
        self
    }

    /// Enable GPU acceleration for SDF computation.
    ///
    /// When enabled and a GPU is available, SDF computation uses
    /// GPU compute shaders for significant speedup (3-68x faster
    /// for small-medium meshes).
    ///
    /// Falls back to CPU if GPU is unavailable.
    pub fn use_gpu(mut self, enable: bool) -> Self {
        self.use_gpu = enable;
        self
    }

    // =========================================================================
    // Wall/Shell Configuration
    // =========================================================================

    /// Set uniform wall thickness in mm.
    ///
    /// This is the thickness of the shell walls. For 3D printing,
    /// typical values are 1.5-4mm depending on the application.
    ///
    /// # Arguments
    ///
    /// * `thickness` - Wall thickness in millimeters
    pub fn wall_thickness(mut self, thickness: f64) -> Self {
        self.wall_thickness_mm = thickness;
        self
    }

    /// Set variable wall thickness using a thickness map.
    ///
    /// This allows different wall thicknesses in different regions
    /// (e.g., thick heel cup, thin arch in a shoe insole).
    ///
    /// # Arguments
    ///
    /// * `map` - ThicknessMap with per-vertex or per-region thickness values
    pub fn thickness_map(mut self, map: ThicknessMap) -> Self {
        self.thickness_map = Some(map);
        self
    }

    /// Set minimum acceptable wall thickness in mm.
    ///
    /// Used during validation to flag walls that are too thin
    /// for reliable 3D printing.
    pub fn min_thickness(mut self, thickness: f64) -> Self {
        self.min_thickness_mm = thickness;
        self
    }

    /// Enable or disable post-generation validation.
    ///
    /// When enabled, the generated shell is validated for
    /// manifoldness, wall thickness, and other quality metrics.
    pub fn validate(mut self, enable: bool) -> Self {
        self.validate = enable;
        self
    }

    /// Use normal-based wall generation (fast but less accurate).
    ///
    /// Each vertex is offset along its normal. Fast, but wall thickness
    /// may vary at corners (thinner at convex, thicker at concave).
    pub fn fast_walls(mut self) -> Self {
        self.wall_method = WallGenerationMethod::Normal;
        self
    }

    /// Use SDF-based wall generation (slower but consistent thickness).
    ///
    /// Computes a signed distance field and extracts an isosurface.
    /// This ensures consistent wall thickness regardless of curvature.
    pub fn sdf_walls(mut self) -> Self {
        self.wall_method = WallGenerationMethod::Sdf;
        self
    }

    // =========================================================================
    // Presets
    // =========================================================================

    /// Apply high-quality preset settings.
    ///
    /// Uses SDF-based wall generation with fine voxel resolution
    /// for consistent wall thickness and smooth surfaces.
    pub fn high_quality(mut self) -> Self {
        self.wall_method = WallGenerationMethod::Sdf;
        self.sdf_voxel_size_mm = 0.3;
        self.voxel_size_mm = 0.5;
        self.validate = true;
        self
    }

    /// Apply fast preset settings.
    ///
    /// Uses normal-based wall generation with coarser resolution.
    /// Good for quick previews or when speed is more important than quality.
    pub fn fast(mut self) -> Self {
        self.wall_method = WallGenerationMethod::Normal;
        self.voxel_size_mm = 1.0;
        self.validate = false;
        self
    }

    /// Apply settings optimized for large meshes.
    ///
    /// Uses adaptive resolution and larger voxels to handle
    /// meshes with hundreds of thousands of triangles.
    pub fn large_mesh(mut self) -> Self {
        self.adaptive_resolution = true;
        self.voxel_size_mm = 0.75;
        self.max_voxels = 30_000_000;
        self
    }

    // =========================================================================
    // Progress Reporting
    // =========================================================================

    /// Set a progress callback.
    ///
    /// The callback receives:
    /// - `progress`: 0.0-1.0 completion percentage
    /// - `stage`: Description of current stage
    ///
    /// Return `false` from the callback to cancel the operation.
    ///
    /// # Example
    ///
    /// ```no_run
    /// use mesh_repair::Mesh;
    /// use mesh_repair::progress::ProgressCallback;
    /// use mesh_shell::ShellBuilder;
    ///
    /// let mesh = Mesh::load("scan.stl").unwrap();
    ///
    /// let callback: ProgressCallback = Box::new(|progress| {
    ///     println!("{}%: {}", progress.percent(), progress.message);
    ///     true // continue
    /// });
    ///
    /// let result = ShellBuilder::new(&mesh)
    ///     .offset(2.0)
    ///     .with_progress(callback)
    ///     .build();
    /// ```
    pub fn with_progress(mut self, callback: mesh_repair::progress::ProgressCallback) -> Self {
        self.progress_callback = Some(callback);
        self
    }

    // =========================================================================
    // Build
    // =========================================================================

    /// Build the shell with the configured parameters.
    ///
    /// This executes the full shell generation pipeline:
    /// 1. Apply SDF offset to create inner surface
    /// 2. Generate outer surface with walls
    /// 3. Create rim connecting inner and outer
    /// 4. Validate result (if enabled)
    ///
    /// # Returns
    ///
    /// A `ShellBuildResult` containing the generated mesh and statistics.
    ///
    /// # Errors
    ///
    /// Returns an error if:
    /// - The mesh is empty or invalid
    /// - The voxel grid would exceed memory limits
    /// - Shell generation fails for any reason
    pub fn build(self) -> ShellResult<ShellBuildResult> {
        // Prepare the mesh with offset values
        let mut mesh = self.mesh.clone();
        for v in &mut mesh.vertices {
            v.offset = Some(self.offset_mm as f32);
        }

        // Build SDF offset params
        let offset_params = SdfOffsetParams {
            voxel_size_mm: self.voxel_size_mm,
            padding_mm: self.padding_mm,
            max_voxels: self.max_voxels,
            offset_neighbors: 8,
            adaptive_resolution: self.adaptive_resolution,
            coarse_voxel_multiplier: 4.0,
            refinement_distance_mm: 5.0,
            use_gpu: self.use_gpu,
        };

        // Apply SDF offset
        let offset_result = apply_sdf_offset(&mesh, &offset_params)?;
        let inner_shell = offset_result.mesh;
        let offset_stats = Some(offset_result.stats);

        // Build shell params
        let shell_params = ShellParams {
            wall_thickness_mm: self.wall_thickness_mm,
            thickness_map: self.thickness_map,
            min_thickness_mm: self.min_thickness_mm,
            validate_after_generation: self.validate,
            wall_generation_method: self.wall_method,
            sdf_voxel_size_mm: self.sdf_voxel_size_mm,
            sdf_max_voxels: self.max_voxels,
        };

        // Generate shell (with or without progress)
        let (shell_mesh, shell_stats) = if let Some(ref callback) = self.progress_callback {
            generate_shell_with_progress(&inner_shell, &shell_params, Some(callback))
        } else {
            generate_shell(&inner_shell, &shell_params)
        };

        Ok(ShellBuildResult {
            mesh: shell_mesh,
            offset_stats,
            shell_stats,
        })
    }

    /// Build only the offset surface (no walls).
    ///
    /// This is useful when you want the inner surface without
    /// generating a full shell with walls.
    ///
    /// # Returns
    ///
    /// The offset result containing the inner surface mesh.
    pub fn build_offset_only(self) -> ShellResult<SdfOffsetResult> {
        // Prepare the mesh with offset values
        let mut mesh = self.mesh.clone();
        for v in &mut mesh.vertices {
            v.offset = Some(self.offset_mm as f32);
        }

        // Build SDF offset params
        let offset_params = SdfOffsetParams {
            voxel_size_mm: self.voxel_size_mm,
            padding_mm: self.padding_mm,
            max_voxels: self.max_voxels,
            offset_neighbors: 8,
            adaptive_resolution: self.adaptive_resolution,
            coarse_voxel_multiplier: 4.0,
            refinement_distance_mm: 5.0,
            use_gpu: self.use_gpu,
        };

        apply_sdf_offset(&mesh, &offset_params)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use mesh_repair::Vertex;

    fn create_test_cube() -> Mesh {
        let mut mesh = Mesh::new();

        // 8 vertices of a unit cube
        mesh.vertices = vec![
            Vertex::from_coords(0.0, 0.0, 0.0),
            Vertex::from_coords(10.0, 0.0, 0.0),
            Vertex::from_coords(10.0, 10.0, 0.0),
            Vertex::from_coords(0.0, 10.0, 0.0),
            Vertex::from_coords(0.0, 0.0, 10.0),
            Vertex::from_coords(10.0, 0.0, 10.0),
            Vertex::from_coords(10.0, 10.0, 10.0),
            Vertex::from_coords(0.0, 10.0, 10.0),
        ];

        // 12 triangles (2 per face)
        mesh.faces = vec![
            // Bottom
            [0, 2, 1],
            [0, 3, 2],
            // Top
            [4, 5, 6],
            [4, 6, 7],
            // Front
            [0, 1, 5],
            [0, 5, 4],
            // Back
            [2, 3, 7],
            [2, 7, 6],
            // Left
            [0, 4, 7],
            [0, 7, 3],
            // Right
            [1, 2, 6],
            [1, 6, 5],
        ];

        mesh
    }

    #[test]
    fn test_builder_defaults() {
        let mesh = create_test_cube();
        let builder = ShellBuilder::new(&mesh);

        // Check defaults
        assert!((builder.offset_mm - 2.0).abs() < 1e-6);
        assert!((builder.wall_thickness_mm - 2.5).abs() < 1e-6);
        assert!(builder.validate);
        assert!(!builder.use_gpu);
    }

    #[test]
    fn test_builder_chaining() {
        let mesh = create_test_cube();
        let builder = ShellBuilder::new(&mesh)
            .offset(3.0)
            .wall_thickness(2.0)
            .voxel_size(0.5)
            .use_gpu(true)
            .high_quality()
            .validate(false);

        assert!((builder.offset_mm - 3.0).abs() < 1e-6);
        assert!((builder.wall_thickness_mm - 2.0).abs() < 1e-6);
        assert!((builder.voxel_size_mm - 0.5).abs() < 1e-6);
        assert!(builder.use_gpu);
        assert!(!builder.validate);
        assert_eq!(builder.wall_method, WallGenerationMethod::Sdf);
    }

    #[test]
    fn test_fast_preset() {
        let mesh = create_test_cube();
        let builder = ShellBuilder::new(&mesh).fast();

        assert_eq!(builder.wall_method, WallGenerationMethod::Normal);
        assert!(!builder.validate);
    }

    #[test]
    fn test_large_mesh_preset() {
        let mesh = create_test_cube();
        let builder = ShellBuilder::new(&mesh).large_mesh();

        assert!(builder.adaptive_resolution);
        assert_eq!(builder.max_voxels, 30_000_000);
    }
}

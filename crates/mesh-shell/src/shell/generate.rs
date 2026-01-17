//! Shell generation algorithm.
//!
//! Generates a printable shell from the inner surface.

use tracing::{debug, info, warn};

use mesh_repair::{Mesh, ThicknessMap, compute_vertex_normals};

use super::rim::{generate_rim, generate_rim_for_sdf_shell};
use super::validation::{ShellValidationResult, validate_shell};
use crate::offset::extract::extract_isosurface;
use crate::offset::grid::SdfGrid;

/// Method for generating the outer surface of the shell.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum WallGenerationMethod {
    /// Normal-based offset (fast, but may have inconsistent thickness at corners).
    ///
    /// Each vertex is offset along its normal by the wall thickness.
    /// Pros: Fast, preserves vertex correspondence with inner surface.
    /// Cons: Wall thickness varies at corners (thinner at convex, thicker at concave).
    #[default]
    Normal,

    /// SDF-based offset (robust, consistent wall thickness).
    ///
    /// Computes a signed distance field and extracts an isosurface at the
    /// desired wall thickness distance. This ensures consistent wall thickness
    /// regardless of surface curvature.
    /// Pros: Consistent wall thickness, handles concave regions correctly.
    /// Cons: Slower, may change vertex count, requires additional memory.
    Sdf,
}

impl std::fmt::Display for WallGenerationMethod {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            WallGenerationMethod::Normal => write!(f, "normal"),
            WallGenerationMethod::Sdf => write!(f, "sdf"),
        }
    }
}

/// Parameters for shell generation.
#[derive(Debug, Clone)]
pub struct ShellParams {
    /// Uniform wall thickness in mm.
    /// Used when `thickness_map` is None.
    pub wall_thickness_mm: f64,
    /// Variable wall thickness map.
    /// When set, per-vertex thickness values override `wall_thickness_mm`.
    /// This enables different wall thicknesses in different regions (e.g., thick heel, thin arch).
    pub thickness_map: Option<ThicknessMap>,
    /// Minimum acceptable wall thickness.
    pub min_thickness_mm: f64,
    /// Whether to validate the shell after generation.
    pub validate_after_generation: bool,
    /// Method for generating the outer surface.
    pub wall_generation_method: WallGenerationMethod,
    /// Voxel size for SDF-based wall generation (mm).
    /// Smaller values give more detail but use more memory.
    /// Only used when `wall_generation_method` is `Sdf`.
    pub sdf_voxel_size_mm: f64,
    /// Maximum voxels for SDF grid (memory limit).
    /// Only used when `wall_generation_method` is `Sdf`.
    pub sdf_max_voxels: usize,
}

impl Default for ShellParams {
    fn default() -> Self {
        Self {
            wall_thickness_mm: 2.5,
            thickness_map: None,
            min_thickness_mm: 1.5,
            validate_after_generation: true,
            wall_generation_method: WallGenerationMethod::Normal,
            sdf_voxel_size_mm: 0.5,
            sdf_max_voxels: 50_000_000,
        }
    }
}

impl ShellParams {
    /// Set the thickness map for variable wall thickness.
    ///
    /// # Example
    ///
    /// ```
    /// use mesh_shell::ShellParams;
    /// use mesh_repair::ThicknessMap;
    ///
    /// let thickness_map = ThicknessMap::new(2.0); // 2mm default
    /// let params = ShellParams::default().with_thickness_map(thickness_map);
    /// ```
    pub fn with_thickness_map(mut self, map: ThicknessMap) -> Self {
        self.thickness_map = Some(map);
        self
    }

    /// Create params with a uniform thickness map.
    ///
    /// This is equivalent to setting `wall_thickness_mm`, but using the
    /// thickness map infrastructure.
    pub fn with_uniform_thickness(mut self, thickness: f64) -> Self {
        self.thickness_map = Some(ThicknessMap::uniform(thickness));
        self.wall_thickness_mm = thickness;
        self
    }

    /// Create params optimized for high-quality output with consistent wall thickness.
    ///
    /// Uses SDF-based wall generation for consistent thickness at corners.
    pub fn high_quality() -> Self {
        Self {
            wall_generation_method: WallGenerationMethod::Sdf,
            sdf_voxel_size_mm: 0.3,
            ..Default::default()
        }
    }

    /// Create params optimized for fast generation.
    ///
    /// Uses normal-based offset which is faster but may have inconsistent thickness.
    pub fn fast() -> Self {
        Self {
            wall_generation_method: WallGenerationMethod::Normal,
            validate_after_generation: false,
            ..Default::default()
        }
    }

    /// Get the wall thickness for a specific vertex index.
    ///
    /// If a thickness map is set, uses the per-vertex value.
    /// Otherwise, returns the uniform `wall_thickness_mm`.
    pub fn get_vertex_thickness(&self, vertex_index: u32) -> f64 {
        self.thickness_map
            .as_ref()
            .map(|m| m.get_vertex_thickness(vertex_index))
            .unwrap_or(self.wall_thickness_mm)
    }
}

/// Result of shell generation.
#[derive(Debug)]
pub struct ShellResult {
    /// Number of inner surface vertices.
    pub inner_vertex_count: usize,
    /// Number of outer surface vertices.
    pub outer_vertex_count: usize,
    /// Number of rim faces generated.
    pub rim_face_count: usize,
    /// Total face count.
    pub total_face_count: usize,
    /// Boundary loop size (number of edges).
    pub boundary_size: usize,
    /// Validation result (if validation was performed).
    pub validation: Option<ShellValidationResult>,
    /// Wall generation method used.
    pub wall_method: WallGenerationMethod,
    /// Whether variable thickness was used.
    pub variable_thickness: bool,
}

/// Generate a printable shell from the inner surface.
///
/// Creates outer surface using the configured method (normal or SDF-based),
/// then connects inner and outer at boundaries with a rim.
///
/// # Arguments
/// * `inner_shell` - The inner surface mesh (from offset stage)
/// * `params` - Shell generation parameters
///
/// # Returns
/// A tuple of (shell mesh, generation result).
pub fn generate_shell(inner_shell: &Mesh, params: &ShellParams) -> (Mesh, ShellResult) {
    let has_variable_thickness = params.thickness_map.is_some();

    if has_variable_thickness {
        info!(
            "Generating shell with variable thickness (default={:.2}mm), method={}",
            params.wall_thickness_mm, params.wall_generation_method
        );
    } else {
        info!(
            "Generating shell with thickness={:.2}mm, method={}",
            params.wall_thickness_mm, params.wall_generation_method
        );
    }

    match params.wall_generation_method {
        WallGenerationMethod::Normal => generate_shell_normal(inner_shell, params),
        WallGenerationMethod::Sdf => generate_shell_sdf(inner_shell, params),
    }
}

/// Generate shell using normal-based offset (original fast method).
fn generate_shell_normal(inner_shell: &Mesh, params: &ShellParams) -> (Mesh, ShellResult) {
    let n = inner_shell.vertices.len();
    let mut shell = Mesh::new();

    // Step 1: Copy inner vertices and ensure normals
    let mut inner_with_normals = inner_shell.clone();
    compute_vertex_normals(&mut inner_with_normals);

    // Step 2: Generate outer vertices by offsetting along normals
    // Copy inner vertices first
    for vertex in &inner_with_normals.vertices {
        // Inner vertex (copy directly)
        shell.vertices.push(vertex.clone());
    }

    // Generate outer vertices with per-vertex thickness
    for (i, vertex) in inner_with_normals.vertices.iter().enumerate() {
        // Get thickness for this vertex (uses thickness map if available)
        let thickness = params.get_vertex_thickness(i as u32);

        // Outer vertex (offset by wall thickness)
        let normal = vertex
            .normal
            .unwrap_or_else(|| nalgebra::Vector3::new(0.0, 0.0, 1.0));
        let outer_pos = vertex.position + normal * thickness;

        let mut outer_vertex = vertex.clone();
        outer_vertex.position = outer_pos;
        // Keep normal for outer surface (points outward)
        outer_vertex.normal = Some(normal);

        shell.vertices.push(outer_vertex);
    }

    debug!("Generated {} inner + {} outer vertices", n, n);

    // Step 3: Copy inner faces (reversed winding so normal points inward)
    for face in &inner_shell.faces {
        // Reverse winding so normal points inward
        shell.faces.push([face[0], face[2], face[1]]);
    }

    // Step 4: Generate outer faces with offset indices (original winding for outward normals)
    for face in &inner_shell.faces {
        let n32 = n as u32;
        shell
            .faces
            .push([face[0] + n32, face[1] + n32, face[2] + n32]);
    }

    let inner_face_count = inner_shell.faces.len();
    debug!(
        "Added {} inner + {} outer faces",
        inner_face_count, inner_face_count
    );

    // Step 5: Find boundary edges and generate rim
    let (rim_faces, boundary_size) = generate_rim(&inner_with_normals, n);

    let rim_face_count = rim_faces.len();
    for face in rim_faces {
        shell.faces.push(face);
    }

    info!(
        "Shell generation complete: {} vertices, {} faces",
        shell.vertices.len(),
        shell.faces.len()
    );

    // Optionally validate the generated shell
    let validation = if params.validate_after_generation {
        let validation_result = validate_shell(&shell);
        if !validation_result.is_printable() {
            warn!(
                "Generated shell has {} validation issue(s)",
                validation_result.issue_count()
            );
        }
        Some(validation_result)
    } else {
        None
    };

    let result = ShellResult {
        inner_vertex_count: n,
        outer_vertex_count: n,
        rim_face_count,
        total_face_count: shell.faces.len(),
        boundary_size,
        validation,
        wall_method: WallGenerationMethod::Normal,
        variable_thickness: params.thickness_map.is_some(),
    };

    (shell, result)
}

/// Generate shell using SDF-based offset for consistent wall thickness.
///
/// Note: Variable thickness (ThicknessMap) is not fully supported with SDF method.
/// The SDF method uses uniform wall thickness for consistent geometry.
/// For variable thickness, use `WallGenerationMethod::Normal`.
fn generate_shell_sdf(inner_shell: &Mesh, params: &ShellParams) -> (Mesh, ShellResult) {
    let inner_vertex_count = inner_shell.vertices.len();

    // Warn if using variable thickness with SDF (not fully supported)
    if params.thickness_map.is_some() {
        warn!(
            "Variable thickness (ThicknessMap) is not fully supported with SDF wall generation. \
             Using uniform thickness={:.2}mm. Consider using WallGenerationMethod::Normal for variable thickness.",
            params.wall_thickness_mm
        );
    }

    // Step 1: Ensure inner mesh has normals
    let mut inner_with_normals = inner_shell.clone();
    compute_vertex_normals(&mut inner_with_normals);

    // Step 2: Create SDF grid for the inner surface
    let padding = params.wall_thickness_mm + params.sdf_voxel_size_mm * 3.0;
    let grid_result = SdfGrid::from_mesh_bounds(
        &inner_with_normals,
        params.sdf_voxel_size_mm,
        padding,
        params.sdf_max_voxels,
    );

    let mut grid = match grid_result {
        Ok(g) => g,
        Err(e) => {
            warn!(
                "SDF grid creation failed: {:?}, falling back to normal method",
                e
            );
            return generate_shell_normal(inner_shell, params);
        }
    };

    info!(
        dims = ?grid.dims,
        total_voxels = grid.total_voxels(),
        "Created SDF grid for wall generation"
    );

    // Step 3: Compute SDF of inner surface
    grid.compute_sdf(&inner_with_normals);

    // Step 4: Offset the SDF by wall thickness to get outer surface
    // Adding positive offset shifts isosurface outward
    for val in &mut grid.values {
        *val -= params.wall_thickness_mm as f32;
    }

    debug!("Applied wall thickness offset to SDF");

    // Step 5: Extract outer surface from offset SDF
    let outer_mesh = match extract_isosurface(&grid) {
        Ok(m) => m,
        Err(e) => {
            warn!(
                "Isosurface extraction failed: {:?}, falling back to normal method",
                e
            );
            return generate_shell_normal(inner_shell, params);
        }
    };

    let outer_vertex_count = outer_mesh.vertices.len();
    debug!(
        "Extracted outer surface: {} vertices, {} faces",
        outer_vertex_count,
        outer_mesh.faces.len()
    );

    // Step 6: Combine inner and outer surfaces into shell
    let mut shell = Mesh::new();

    // Add inner vertices
    for vertex in &inner_with_normals.vertices {
        shell.vertices.push(vertex.clone());
    }

    // Add outer vertices (offset by inner count)
    let inner_count = inner_with_normals.vertices.len() as u32;
    for vertex in &outer_mesh.vertices {
        shell.vertices.push(vertex.clone());
    }

    // Add inner faces (reversed winding so normal points inward)
    for face in &inner_with_normals.faces {
        shell.faces.push([face[0], face[2], face[1]]);
    }

    // Add outer faces (keep original winding, offset indices)
    for face in &outer_mesh.faces {
        shell.faces.push([
            face[0] + inner_count,
            face[1] + inner_count,
            face[2] + inner_count,
        ]);
    }

    // Step 7: Generate rim connecting inner and outer boundaries
    let (rim_faces, boundary_size) =
        generate_rim_for_sdf_shell(&inner_with_normals, &outer_mesh, inner_count as usize);

    let rim_face_count = rim_faces.len();
    for face in rim_faces {
        shell.faces.push(face);
    }

    info!(
        "SDF shell generation complete: {} vertices, {} faces (rim: {})",
        shell.vertices.len(),
        shell.faces.len(),
        rim_face_count
    );

    // Optionally validate the generated shell
    let validation = if params.validate_after_generation {
        let validation_result = validate_shell(&shell);
        if !validation_result.is_printable() {
            warn!(
                "Generated shell has {} validation issue(s)",
                validation_result.issue_count()
            );
        }
        Some(validation_result)
    } else {
        None
    };

    let result = ShellResult {
        inner_vertex_count,
        outer_vertex_count,
        rim_face_count,
        total_face_count: shell.faces.len(),
        boundary_size,
        validation,
        wall_method: WallGenerationMethod::Sdf,
        variable_thickness: false, // SDF doesn't support variable thickness
    };

    (shell, result)
}

/// Generate a shell without automatic validation.
///
/// This is equivalent to calling `generate_shell` with `validate_after_generation = false`.
pub fn generate_shell_no_validation(
    inner_shell: &Mesh,
    params: &ShellParams,
) -> (Mesh, ShellResult) {
    let mut params = params.clone();
    params.validate_after_generation = false;
    generate_shell(inner_shell, &params)
}

/// Generate a printable shell with progress reporting.
///
/// This is a progress-reporting variant of [`generate_shell`] that allows tracking
/// the shell generation progress and supports cancellation via the progress callback.
///
/// The shell generation proceeds through these phases:
/// 1. Vertex normal computation
/// 2. Outer surface generation (normal offset or SDF)
/// 3. Inner/outer face creation
/// 4. Rim generation to connect boundaries
/// 5. Optional validation
///
/// # Arguments
/// * `inner_shell` - The inner surface mesh (from offset stage)
/// * `params` - Shell generation parameters
/// * `callback` - Optional progress callback. Returns `false` to request cancellation.
///
/// # Returns
/// A tuple of (shell mesh, generation result).
/// If cancelled via callback, returns the partial shell.
///
/// # Example
/// ```ignore
/// use mesh_shell::{generate_shell_with_progress, ShellParams};
/// use mesh_repair::progress::ProgressCallback;
///
/// let callback: ProgressCallback = Box::new(|progress| {
///     println!("{}% - {}", progress.percent(), progress.message);
///     true // Continue
/// });
///
/// let (shell, result) = generate_shell_with_progress(&inner_mesh, &ShellParams::default(), Some(&callback));
/// ```
pub fn generate_shell_with_progress(
    inner_shell: &Mesh,
    params: &ShellParams,
    callback: Option<&mesh_repair::progress::ProgressCallback>,
) -> (Mesh, ShellResult) {
    use mesh_repair::progress::ProgressTracker;

    let has_variable_thickness = params.thickness_map.is_some();

    if has_variable_thickness {
        info!(
            "Generating shell with variable thickness (default={:.2}mm), method={}",
            params.wall_thickness_mm, params.wall_generation_method
        );
    } else {
        info!(
            "Generating shell with thickness={:.2}mm, method={}",
            params.wall_thickness_mm, params.wall_generation_method
        );
    }

    // Total phases: normal computation (10%), outer generation (40%), faces (20%), rim (20%), validation (10%)
    let tracker = ProgressTracker::new(100);

    // Phase 1: Start and compute normals
    tracker.set(5);
    if !tracker.maybe_callback(callback, "Computing vertex normals".to_string()) {
        return empty_shell_result(params);
    }

    let n = inner_shell.vertices.len();
    let mut inner_with_normals = inner_shell.clone();
    compute_vertex_normals(&mut inner_with_normals);

    // Dispatch based on wall generation method
    match params.wall_generation_method {
        WallGenerationMethod::Normal => generate_shell_normal_with_progress(
            inner_shell,
            params,
            &inner_with_normals,
            n,
            &tracker,
            callback,
        ),
        WallGenerationMethod::Sdf => generate_shell_sdf_with_progress(
            inner_shell,
            params,
            &inner_with_normals,
            &tracker,
            callback,
        ),
    }
}

/// Helper to create an empty shell result for early cancellation
fn empty_shell_result(params: &ShellParams) -> (Mesh, ShellResult) {
    (
        Mesh::new(),
        ShellResult {
            inner_vertex_count: 0,
            outer_vertex_count: 0,
            rim_face_count: 0,
            total_face_count: 0,
            boundary_size: 0,
            validation: None,
            wall_method: params.wall_generation_method,
            variable_thickness: params.thickness_map.is_some(),
        },
    )
}

/// Generate shell using normal-based offset with progress reporting.
fn generate_shell_normal_with_progress(
    inner_shell: &Mesh,
    params: &ShellParams,
    inner_with_normals: &Mesh,
    n: usize,
    tracker: &mesh_repair::progress::ProgressTracker,
    callback: Option<&mesh_repair::progress::ProgressCallback>,
) -> (Mesh, ShellResult) {
    let mut shell = Mesh::new();

    // Phase 2: Copy inner vertices
    tracker.set(10);
    if !tracker.maybe_callback(callback, "Copying inner vertices".to_string()) {
        return empty_shell_result(params);
    }

    for vertex in &inner_with_normals.vertices {
        shell.vertices.push(vertex.clone());
    }

    // Phase 3: Generate outer vertices by offsetting along normals
    tracker.set(30);
    if !tracker.maybe_callback(callback, "Generating outer surface vertices".to_string()) {
        return empty_shell_result(params);
    }

    for (i, vertex) in inner_with_normals.vertices.iter().enumerate() {
        let thickness = params.get_vertex_thickness(i as u32);
        let normal = vertex
            .normal
            .unwrap_or_else(|| nalgebra::Vector3::new(0.0, 0.0, 1.0));
        let outer_pos = vertex.position + normal * thickness;

        let mut outer_vertex = vertex.clone();
        outer_vertex.position = outer_pos;
        outer_vertex.normal = Some(normal);

        shell.vertices.push(outer_vertex);
    }

    debug!("Generated {} inner + {} outer vertices", n, n);

    // Phase 4: Create inner and outer faces
    tracker.set(50);
    if !tracker.maybe_callback(callback, "Creating inner and outer faces".to_string()) {
        return empty_shell_result(params);
    }

    // Inner faces (reversed winding so normal points inward)
    for face in &inner_shell.faces {
        shell.faces.push([face[0], face[2], face[1]]);
    }

    // Outer faces with offset indices (original winding for outward normals)
    for face in &inner_shell.faces {
        let n32 = n as u32;
        shell
            .faces
            .push([face[0] + n32, face[1] + n32, face[2] + n32]);
    }

    let inner_face_count = inner_shell.faces.len();
    debug!(
        "Added {} inner + {} outer faces",
        inner_face_count, inner_face_count
    );

    // Phase 5: Generate rim
    tracker.set(70);
    if !tracker.maybe_callback(callback, "Generating rim to connect boundaries".to_string()) {
        return empty_shell_result(params);
    }

    let (rim_faces, boundary_size) = generate_rim(inner_with_normals, n);

    let rim_face_count = rim_faces.len();
    for face in rim_faces {
        shell.faces.push(face);
    }

    info!(
        "Shell generation complete: {} vertices, {} faces",
        shell.vertices.len(),
        shell.faces.len()
    );

    // Phase 6: Optional validation
    tracker.set(90);
    let validation = if params.validate_after_generation {
        if !tracker.maybe_callback(callback, "Validating shell".to_string()) {
            return (
                shell.clone(),
                ShellResult {
                    inner_vertex_count: n,
                    outer_vertex_count: n,
                    rim_face_count,
                    total_face_count: shell.faces.len(),
                    boundary_size,
                    validation: None,
                    wall_method: WallGenerationMethod::Normal,
                    variable_thickness: params.thickness_map.is_some(),
                },
            );
        }

        let validation_result = validate_shell(&shell);
        if !validation_result.is_printable() {
            warn!(
                "Generated shell has {} validation issue(s)",
                validation_result.issue_count()
            );
        }
        Some(validation_result)
    } else {
        None
    };

    tracker.set(100);
    let _ = tracker.maybe_callback(callback, "Shell generation complete".to_string());

    let result = ShellResult {
        inner_vertex_count: n,
        outer_vertex_count: n,
        rim_face_count,
        total_face_count: shell.faces.len(),
        boundary_size,
        validation,
        wall_method: WallGenerationMethod::Normal,
        variable_thickness: params.thickness_map.is_some(),
    };

    (shell, result)
}

/// Generate shell using SDF-based offset with progress reporting.
fn generate_shell_sdf_with_progress(
    inner_shell: &Mesh,
    params: &ShellParams,
    inner_with_normals: &Mesh,
    tracker: &mesh_repair::progress::ProgressTracker,
    callback: Option<&mesh_repair::progress::ProgressCallback>,
) -> (Mesh, ShellResult) {
    let inner_vertex_count = inner_with_normals.vertices.len();

    // Warn if using variable thickness with SDF (not fully supported)
    if params.thickness_map.is_some() {
        warn!(
            "Variable thickness (ThicknessMap) is not fully supported with SDF wall generation. \
             Using uniform thickness={:.2}mm. Consider using WallGenerationMethod::Normal for variable thickness.",
            params.wall_thickness_mm
        );
    }

    // Phase 2: Create SDF grid
    tracker.set(20);
    if !tracker.maybe_callback(callback, "Creating SDF grid".to_string()) {
        return empty_shell_result(params);
    }

    let padding = params.wall_thickness_mm + params.sdf_voxel_size_mm * 3.0;
    let grid_result = SdfGrid::from_mesh_bounds(
        inner_with_normals,
        params.sdf_voxel_size_mm,
        padding,
        params.sdf_max_voxels,
    );

    let mut grid = match grid_result {
        Ok(g) => g,
        Err(e) => {
            warn!(
                "SDF grid creation failed: {:?}, falling back to normal method",
                e
            );
            return generate_shell_normal(inner_shell, params);
        }
    };

    info!(
        dims = ?grid.dims,
        total_voxels = grid.total_voxels(),
        "Created SDF grid for wall generation"
    );

    // Phase 3: Compute SDF
    tracker.set(40);
    if !tracker.maybe_callback(callback, "Computing signed distance field".to_string()) {
        return empty_shell_result(params);
    }

    grid.compute_sdf(inner_with_normals);

    // Phase 4: Offset SDF by wall thickness
    tracker.set(50);
    if !tracker.maybe_callback(callback, "Applying wall thickness offset".to_string()) {
        return empty_shell_result(params);
    }

    for val in &mut grid.values {
        *val -= params.wall_thickness_mm as f32;
    }

    debug!("Applied wall thickness offset to SDF");

    // Phase 5: Extract outer surface from offset SDF
    tracker.set(60);
    if !tracker.maybe_callback(callback, "Extracting outer surface isosurface".to_string()) {
        return empty_shell_result(params);
    }

    let outer_mesh = match extract_isosurface(&grid) {
        Ok(m) => m,
        Err(e) => {
            warn!(
                "Isosurface extraction failed: {:?}, falling back to normal method",
                e
            );
            return generate_shell_normal(inner_shell, params);
        }
    };

    let outer_vertex_count = outer_mesh.vertices.len();
    debug!(
        "Extracted outer surface: {} vertices, {} faces",
        outer_vertex_count,
        outer_mesh.faces.len()
    );

    // Phase 6: Combine inner and outer surfaces
    tracker.set(70);
    if !tracker.maybe_callback(callback, "Combining inner and outer surfaces".to_string()) {
        return empty_shell_result(params);
    }

    let mut shell = Mesh::new();

    // Add inner vertices
    for vertex in &inner_with_normals.vertices {
        shell.vertices.push(vertex.clone());
    }

    // Add outer vertices (offset by inner count)
    let inner_count = inner_with_normals.vertices.len() as u32;
    for vertex in &outer_mesh.vertices {
        shell.vertices.push(vertex.clone());
    }

    // Add inner faces (reversed winding so normal points inward)
    for face in &inner_with_normals.faces {
        shell.faces.push([face[0], face[2], face[1]]);
    }

    // Add outer faces (keep original winding, offset indices)
    for face in &outer_mesh.faces {
        shell.faces.push([
            face[0] + inner_count,
            face[1] + inner_count,
            face[2] + inner_count,
        ]);
    }

    // Phase 7: Generate rim connecting inner and outer boundaries
    tracker.set(80);
    if !tracker.maybe_callback(callback, "Generating rim to connect boundaries".to_string()) {
        return empty_shell_result(params);
    }

    let (rim_faces, boundary_size) =
        generate_rim_for_sdf_shell(inner_with_normals, &outer_mesh, inner_count as usize);

    let rim_face_count = rim_faces.len();
    for face in rim_faces {
        shell.faces.push(face);
    }

    info!(
        "SDF shell generation complete: {} vertices, {} faces (rim: {})",
        shell.vertices.len(),
        shell.faces.len(),
        rim_face_count
    );

    // Phase 8: Optional validation
    tracker.set(90);
    let validation = if params.validate_after_generation {
        if !tracker.maybe_callback(callback, "Validating shell".to_string()) {
            return (
                shell.clone(),
                ShellResult {
                    inner_vertex_count,
                    outer_vertex_count,
                    rim_face_count,
                    total_face_count: shell.faces.len(),
                    boundary_size,
                    validation: None,
                    wall_method: WallGenerationMethod::Sdf,
                    variable_thickness: false,
                },
            );
        }

        let validation_result = validate_shell(&shell);
        if !validation_result.is_printable() {
            warn!(
                "Generated shell has {} validation issue(s)",
                validation_result.issue_count()
            );
        }
        Some(validation_result)
    } else {
        None
    };

    tracker.set(100);
    let _ = tracker.maybe_callback(callback, "Shell generation complete".to_string());

    let result = ShellResult {
        inner_vertex_count,
        outer_vertex_count,
        rim_face_count,
        total_face_count: shell.faces.len(),
        boundary_size,
        validation,
        wall_method: WallGenerationMethod::Sdf,
        variable_thickness: false,
    };

    (shell, result)
}

#[cfg(test)]
mod tests {
    use super::*;
    use mesh_repair::Vertex;

    fn create_open_box() -> Mesh {
        // A box open on top (5 faces instead of 6)
        let mut mesh = Mesh::new();

        // Bottom corners
        mesh.vertices.push(Vertex::from_coords(0.0, 0.0, 0.0));
        mesh.vertices.push(Vertex::from_coords(10.0, 0.0, 0.0));
        mesh.vertices.push(Vertex::from_coords(10.0, 10.0, 0.0));
        mesh.vertices.push(Vertex::from_coords(0.0, 10.0, 0.0));
        // Top corners
        mesh.vertices.push(Vertex::from_coords(0.0, 0.0, 10.0));
        mesh.vertices.push(Vertex::from_coords(10.0, 0.0, 10.0));
        mesh.vertices.push(Vertex::from_coords(10.0, 10.0, 10.0));
        mesh.vertices.push(Vertex::from_coords(0.0, 10.0, 10.0));

        // Bottom (2 triangles)
        mesh.faces.push([0, 2, 1]);
        mesh.faces.push([0, 3, 2]);
        // Front
        mesh.faces.push([0, 1, 5]);
        mesh.faces.push([0, 5, 4]);
        // Back
        mesh.faces.push([2, 3, 7]);
        mesh.faces.push([2, 7, 6]);
        // Left
        mesh.faces.push([0, 4, 7]);
        mesh.faces.push([0, 7, 3]);
        // Right
        mesh.faces.push([1, 2, 6]);
        mesh.faces.push([1, 6, 5]);
        // Top is OPEN - boundary is 4-5-6-7

        mesh
    }

    #[test]
    fn test_shell_params_default() {
        let params = ShellParams::default();
        assert_eq!(params.wall_thickness_mm, 2.5);
        assert_eq!(params.min_thickness_mm, 1.5);
        assert!(params.validate_after_generation);
        assert_eq!(params.wall_generation_method, WallGenerationMethod::Normal);
    }

    #[test]
    fn test_shell_params_high_quality() {
        let params = ShellParams::high_quality();
        assert_eq!(params.wall_generation_method, WallGenerationMethod::Sdf);
        assert!(params.sdf_voxel_size_mm < 0.5);
    }

    #[test]
    fn test_shell_params_fast() {
        let params = ShellParams::fast();
        assert_eq!(params.wall_generation_method, WallGenerationMethod::Normal);
        assert!(!params.validate_after_generation);
    }

    #[test]
    fn test_wall_generation_method_display() {
        assert_eq!(format!("{}", WallGenerationMethod::Normal), "normal");
        assert_eq!(format!("{}", WallGenerationMethod::Sdf), "sdf");
    }

    #[test]
    fn test_generate_shell_doubles_vertices() {
        let inner = create_open_box();
        let params = ShellParams::default();

        let (shell, result) = generate_shell(&inner, &params);

        // Should have 2x vertices (inner + outer) for normal method
        assert_eq!(shell.vertices.len(), inner.vertices.len() * 2);
        assert_eq!(result.inner_vertex_count, inner.vertices.len());
        assert_eq!(result.outer_vertex_count, inner.vertices.len());
        assert_eq!(result.wall_method, WallGenerationMethod::Normal);
    }

    #[test]
    fn test_shell_has_more_faces() {
        let inner = create_open_box();
        let params = ShellParams::default();

        let (shell, result) = generate_shell(&inner, &params);

        // Should have inner + outer + rim faces
        assert!(shell.faces.len() > inner.faces.len() * 2);
        assert!(result.rim_face_count > 0);
    }

    #[test]
    fn test_generate_shell_sdf_method() {
        let inner = create_open_box();
        let params = ShellParams {
            wall_generation_method: WallGenerationMethod::Sdf,
            sdf_voxel_size_mm: 1.0, // Coarse for fast test
            validate_after_generation: false,
            ..Default::default()
        };

        let (shell, result) = generate_shell(&inner, &params);

        // Should produce a valid mesh
        assert!(!shell.vertices.is_empty());
        assert!(!shell.faces.is_empty());
        assert_eq!(result.wall_method, WallGenerationMethod::Sdf);

        // Inner vertex count should match
        assert_eq!(result.inner_vertex_count, inner.vertices.len());

        // Outer vertex count may differ from inner (SDF remeshes)
        assert!(result.outer_vertex_count > 0);
    }

    #[test]
    fn test_sdf_produces_larger_outer_surface() {
        let inner = create_open_box();
        let wall_thickness = 2.0;

        let params = ShellParams {
            wall_thickness_mm: wall_thickness,
            wall_generation_method: WallGenerationMethod::Sdf,
            sdf_voxel_size_mm: 0.5,
            validate_after_generation: false,
            ..Default::default()
        };

        let (shell, _result) = generate_shell(&inner, &params);

        // Get bounds of inner and combined shell
        let inner_bounds = inner.bounds().unwrap();
        let shell_bounds = shell.bounds().unwrap();

        // Shell should be larger due to wall thickness
        let inner_extent = inner_bounds.1 - inner_bounds.0;
        let shell_extent = shell_bounds.1 - shell_bounds.0;

        // Shell should be ~2*wall_thickness larger in each dimension
        assert!(
            shell_extent.x > inner_extent.x,
            "Shell should be wider: {} vs {}",
            shell_extent.x,
            inner_extent.x
        );
        assert!(
            shell_extent.y > inner_extent.y,
            "Shell should be deeper: {} vs {}",
            shell_extent.y,
            inner_extent.y
        );
    }

    #[test]
    fn test_variable_thickness_params() {
        let mut thickness_map = ThicknessMap::new(2.0);
        thickness_map.set_vertex_thickness(0, 3.0);
        thickness_map.set_vertex_thickness(1, 1.5);

        let params = ShellParams::default().with_thickness_map(thickness_map);

        assert!(params.thickness_map.is_some());
        assert_eq!(params.get_vertex_thickness(0), 3.0);
        assert_eq!(params.get_vertex_thickness(1), 1.5);
        assert_eq!(params.get_vertex_thickness(2), 2.0); // Default
    }

    #[test]
    fn test_variable_thickness_shell_generation() {
        let inner = create_open_box();

        // Create thickness map: bottom vertices thin (1mm), top vertices thick (3mm)
        let mut thickness_map = ThicknessMap::new(2.0);
        // Bottom vertices (0-3) are thin
        for i in 0..4 {
            thickness_map.set_vertex_thickness(i, 1.0);
        }
        // Top vertices (4-7) are thick
        for i in 4..8 {
            thickness_map.set_vertex_thickness(i, 3.0);
        }

        let params = ShellParams {
            wall_generation_method: WallGenerationMethod::Normal,
            validate_after_generation: false,
            ..ShellParams::default()
        }
        .with_thickness_map(thickness_map);

        let (shell, result) = generate_shell(&inner, &params);

        // Check that variable thickness was reported
        assert!(result.variable_thickness);

        // Verify the shell has correct structure
        assert_eq!(shell.vertices.len(), inner.vertices.len() * 2);

        // Check that bottom outer vertices are offset less than top outer vertices
        let inner_vertex_count = inner.vertices.len();

        // Outer vertex 0 (bottom) should be offset ~1mm from inner vertex 0
        let inner_v0 = shell.vertices[0].position;
        let outer_v0 = shell.vertices[inner_vertex_count].position;
        let offset_0 = (outer_v0 - inner_v0).norm();

        // Outer vertex 4 (top) should be offset ~3mm from inner vertex 4
        let inner_v4 = shell.vertices[4].position;
        let outer_v4 = shell.vertices[inner_vertex_count + 4].position;
        let offset_4 = (outer_v4 - inner_v4).norm();

        assert!(
            offset_4 > offset_0,
            "Top vertices should have larger offset: {} vs {}",
            offset_4,
            offset_0
        );

        // The offsets should be close to their target values (within tolerance due to normal direction)
        assert!(
            offset_0 < 2.0,
            "Bottom offset should be around 1mm: {}",
            offset_0
        );
        assert!(
            offset_4 > 2.0,
            "Top offset should be around 3mm: {}",
            offset_4
        );
    }

    #[test]
    fn test_uniform_thickness_via_map() {
        let inner = create_open_box();

        // Use uniform thickness via map (should behave same as default)
        let params = ShellParams::default().with_uniform_thickness(2.5);

        let (_shell, result) = generate_shell(&inner, &params);

        assert!(result.variable_thickness); // Still counts as using thickness map
        assert_eq!(params.wall_thickness_mm, 2.5);
    }

    #[test]
    fn test_get_vertex_thickness_without_map() {
        let params = ShellParams {
            wall_thickness_mm: 3.0,
            ..Default::default()
        };

        // Without a thickness map, should return the uniform wall thickness
        assert_eq!(params.get_vertex_thickness(0), 3.0);
        assert_eq!(params.get_vertex_thickness(100), 3.0);
    }
}

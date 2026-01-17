//! Mesh morphing and deformation algorithms.
//!
//! This module provides tools for deforming meshes to fit target shapes,
//! including Free-Form Deformation (FFD) with control lattices and
//! Radial Basis Function (RBF) morphing.
//!
//! # Use Cases
//!
//! - Morphing a template shoe last to fit a customer's foot scan
//! - Deforming a helmet liner to match head measurements
//! - Creating parametric variations of product designs
//!
//! # Example
//!
//! ```
//! use mesh_repair::{Mesh, Vertex};
//! use mesh_repair::morph::{MorphParams, MorphResult, Constraint};
//! use nalgebra::Point3;
//!
//! // Create a simple mesh
//! let mut mesh = Mesh::new();
//! mesh.vertices.push(Vertex::from_coords(0.0, 0.0, 0.0));
//! mesh.vertices.push(Vertex::from_coords(1.0, 0.0, 0.0));
//! mesh.vertices.push(Vertex::from_coords(0.5, 1.0, 0.0));
//! mesh.vertices.push(Vertex::from_coords(0.5, 0.5, 1.0));
//! mesh.faces.push([0, 1, 2]);
//! mesh.faces.push([0, 2, 3]);
//! mesh.faces.push([0, 3, 1]);
//! mesh.faces.push([1, 3, 2]);
//!
//! // Define control point constraints
//! let constraints = vec![
//!     Constraint::point(Point3::new(0.0, 0.0, 0.0), Point3::new(0.0, 0.0, 0.5)),
//!     Constraint::point(Point3::new(1.0, 0.0, 0.0), Point3::new(1.0, 0.0, 0.5)),
//! ];
//!
//! // Morph the mesh using RBF
//! let params = MorphParams::rbf().with_constraints(constraints);
//! let result = mesh_repair::morph::morph_mesh(&mesh, &params).unwrap();
//!
//! println!("Max displacement: {:.3} mm", result.max_displacement);
//! ```

use crate::{Mesh, MeshError, MeshResult};
use nalgebra::{DMatrix, DVector, Point3, Vector3};
use std::collections::HashSet;

/// Parameters for mesh morphing operations.
#[derive(Debug, Clone)]
pub struct MorphParams {
    /// The morphing algorithm to use.
    pub algorithm: MorphAlgorithm,

    /// Control point constraints (source → target mappings).
    pub constraints: Vec<Constraint>,

    /// Optional region mask - only vertices in this set will be deformed.
    /// If None, all vertices are affected.
    pub region_mask: Option<HashSet<u32>>,

    /// Smoothness parameter for RBF (higher = smoother deformation).
    /// Default: 1.0
    pub smoothness: f64,

    /// Number of control points per dimension for FFD lattice.
    /// Default: (4, 4, 4)
    pub ffd_resolution: (usize, usize, usize),
}

impl Default for MorphParams {
    fn default() -> Self {
        Self {
            algorithm: MorphAlgorithm::Rbf(RbfKernel::ThinPlateSpline),
            constraints: Vec::new(),
            region_mask: None,
            smoothness: 1.0,
            ffd_resolution: (4, 4, 4),
        }
    }
}

impl MorphParams {
    /// Create params for RBF morphing with thin-plate spline kernel.
    pub fn rbf() -> Self {
        Self {
            algorithm: MorphAlgorithm::Rbf(RbfKernel::ThinPlateSpline),
            ..Default::default()
        }
    }

    /// Create params for RBF morphing with Gaussian kernel.
    pub fn rbf_gaussian() -> Self {
        Self {
            algorithm: MorphAlgorithm::Rbf(RbfKernel::Gaussian),
            ..Default::default()
        }
    }

    /// Create params for RBF morphing with multiquadric kernel.
    pub fn rbf_multiquadric() -> Self {
        Self {
            algorithm: MorphAlgorithm::Rbf(RbfKernel::Multiquadric),
            ..Default::default()
        }
    }

    /// Create params for Free-Form Deformation.
    pub fn ffd() -> Self {
        Self {
            algorithm: MorphAlgorithm::Ffd,
            ..Default::default()
        }
    }

    /// Create params for FFD with custom lattice resolution.
    pub fn ffd_with_resolution(nx: usize, ny: usize, nz: usize) -> Self {
        Self {
            algorithm: MorphAlgorithm::Ffd,
            ffd_resolution: (nx, ny, nz),
            ..Default::default()
        }
    }

    /// Add control point constraints.
    pub fn with_constraints(mut self, constraints: Vec<Constraint>) -> Self {
        self.constraints = constraints;
        self
    }

    /// Set the region mask for partial deformation.
    pub fn with_region(mut self, vertex_indices: HashSet<u32>) -> Self {
        self.region_mask = Some(vertex_indices);
        self
    }

    /// Set the smoothness parameter (RBF only).
    pub fn with_smoothness(mut self, smoothness: f64) -> Self {
        self.smoothness = smoothness;
        self
    }

    /// Set the FFD lattice resolution.
    pub fn with_ffd_resolution(mut self, nx: usize, ny: usize, nz: usize) -> Self {
        self.ffd_resolution = (nx, ny, nz);
        self
    }
}

/// The morphing algorithm to use.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum MorphAlgorithm {
    /// Radial Basis Function morphing with the specified kernel.
    Rbf(RbfKernel),
    /// Free-Form Deformation with a control lattice.
    Ffd,
}

/// Kernel function for RBF morphing.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum RbfKernel {
    /// Thin-plate spline: r² log(r)
    /// Good for smooth, natural deformations.
    ThinPlateSpline,

    /// Gaussian: exp(-r²/σ²)
    /// Provides local deformation with controllable support.
    Gaussian,

    /// Multiquadric: sqrt(r² + c²)
    /// Good balance between local and global deformation.
    Multiquadric,

    /// Inverse multiquadric: 1/sqrt(r² + c²)
    /// Strong local effect, diminishes quickly with distance.
    InverseMultiquadric,
}

/// A control point constraint for morphing.
#[derive(Debug, Clone)]
pub struct Constraint {
    /// Source position (where the point is in the original mesh).
    pub source: Point3<f64>,

    /// Target position (where the point should move to).
    pub target: Point3<f64>,

    /// Optional weight for this constraint (default 1.0).
    pub weight: f64,
}

impl Constraint {
    /// Create a point-to-point constraint.
    pub fn point(source: Point3<f64>, target: Point3<f64>) -> Self {
        Self {
            source,
            target,
            weight: 1.0,
        }
    }

    /// Create a weighted constraint.
    pub fn weighted(source: Point3<f64>, target: Point3<f64>, weight: f64) -> Self {
        Self {
            source,
            target,
            weight,
        }
    }

    /// Create a constraint from a displacement vector.
    pub fn displacement(source: Point3<f64>, displacement: Vector3<f64>) -> Self {
        Self {
            source,
            target: source + displacement,
            weight: 1.0,
        }
    }

    /// Get the displacement vector for this constraint.
    pub fn displacement_vector(&self) -> Vector3<f64> {
        self.target - self.source
    }
}

/// Result of a morphing operation.
#[derive(Debug, Clone)]
pub struct MorphResult {
    /// The morphed mesh.
    pub mesh: Mesh,

    /// Number of vertices that were modified.
    pub vertices_modified: usize,

    /// Maximum displacement of any vertex.
    pub max_displacement: f64,

    /// Average displacement across all modified vertices.
    pub average_displacement: f64,

    /// Maximum local stretch (ratio of deformed to original edge length).
    pub max_stretch: f64,

    /// Maximum local compression (ratio of original to deformed edge length).
    pub max_compression: f64,

    /// Volume change ratio (new_volume / old_volume).
    pub volume_ratio: f64,
}

impl MorphResult {
    /// Check if the deformation introduced significant distortion.
    ///
    /// Returns true if stretch or compression exceeds the threshold.
    pub fn has_significant_distortion(&self, threshold: f64) -> bool {
        self.max_stretch > 1.0 + threshold || self.max_compression > 1.0 + threshold
    }

    /// Check if the volume changed significantly.
    pub fn has_significant_volume_change(&self, threshold: f64) -> bool {
        (self.volume_ratio - 1.0).abs() > threshold
    }
}

/// Morph a mesh according to the given parameters.
///
/// # Arguments
///
/// * `mesh` - The input mesh to morph
/// * `params` - Morphing parameters including algorithm and constraints
///
/// # Returns
///
/// A `MorphResult` containing the morphed mesh and quality metrics.
///
/// # Errors
///
/// Returns an error if:
/// - The mesh is empty
/// - No constraints are provided
/// - The constraint system is degenerate (e.g., all constraints at same point)
pub fn morph_mesh(mesh: &Mesh, params: &MorphParams) -> MeshResult<MorphResult> {
    if mesh.is_empty() {
        return Err(MeshError::EmptyMesh {
            details: "Cannot morph an empty mesh".to_string(),
        });
    }

    if params.constraints.is_empty() {
        return Err(MeshError::RepairFailed {
            details: "No constraints provided for morphing".to_string(),
        });
    }

    match params.algorithm {
        MorphAlgorithm::Rbf(kernel) => morph_rbf(mesh, params, kernel),
        MorphAlgorithm::Ffd => morph_ffd(mesh, params),
    }
}

/// RBF morphing implementation.
fn morph_rbf(mesh: &Mesh, params: &MorphParams, kernel: RbfKernel) -> MeshResult<MorphResult> {
    let n = params.constraints.len();

    // Build the RBF interpolation matrix
    // For thin-plate splines, we add polynomial terms for affine transformation
    let use_polynomial = matches!(kernel, RbfKernel::ThinPlateSpline);
    let matrix_size = if use_polynomial { n + 4 } else { n };

    let mut matrix = DMatrix::zeros(matrix_size, matrix_size);

    // Fill the RBF part of the matrix
    for i in 0..n {
        for j in 0..n {
            let r = (params.constraints[i].source - params.constraints[j].source).norm();
            let value =
                evaluate_kernel(kernel, r, params.smoothness) * params.constraints[i].weight;
            matrix[(i, j)] = value;
        }
        // Add small regularization to diagonal for numerical stability
        matrix[(i, i)] += 1e-8;
    }

    // Add polynomial terms for thin-plate spline
    if use_polynomial {
        for i in 0..n {
            let p = &params.constraints[i].source;
            matrix[(i, n)] = 1.0;
            matrix[(i, n + 1)] = p.x;
            matrix[(i, n + 2)] = p.y;
            matrix[(i, n + 3)] = p.z;

            matrix[(n, i)] = 1.0;
            matrix[(n + 1, i)] = p.x;
            matrix[(n + 2, i)] = p.y;
            matrix[(n + 3, i)] = p.z;
        }
        // Small regularization for polynomial constraints
        for i in n..matrix_size {
            matrix[(i, i)] += 1e-10;
        }
    }

    // Solve for coefficients for each coordinate - use SVD for better numerical stability
    let svd = matrix.svd(true, true);

    // Build target vectors
    let mut dx = DVector::zeros(matrix_size);
    let mut dy = DVector::zeros(matrix_size);
    let mut dz = DVector::zeros(matrix_size);

    for i in 0..n {
        let d = params.constraints[i].displacement_vector();
        dx[i] = d.x;
        dy[i] = d.y;
        dz[i] = d.z;
    }

    // Solve the systems using SVD pseudoinverse for better numerical stability
    let epsilon = 1e-10;
    let wx = svd
        .solve(&dx, epsilon)
        .map_err(|_| MeshError::RepairFailed {
            details: "Failed to solve RBF system (degenerate constraint configuration)".to_string(),
        })?;
    let wy = svd
        .solve(&dy, epsilon)
        .map_err(|_| MeshError::RepairFailed {
            details: "Failed to solve RBF system (degenerate constraint configuration)".to_string(),
        })?;
    let wz = svd
        .solve(&dz, epsilon)
        .map_err(|_| MeshError::RepairFailed {
            details: "Failed to solve RBF system (degenerate constraint configuration)".to_string(),
        })?;

    // Calculate original volume for comparison
    let original_volume = mesh.volume();

    // Apply the deformation to each vertex
    let mut morphed = mesh.clone();
    let mut vertices_modified = 0;
    let mut max_displacement = 0.0f64;
    let mut total_displacement = 0.0f64;

    for (idx, vertex) in morphed.vertices.iter_mut().enumerate() {
        // Check region mask
        if let Some(ref mask) = params.region_mask
            && !mask.contains(&(idx as u32))
        {
            continue;
        }

        let p = vertex.position;

        // Evaluate the RBF at this point
        let mut disp_x = 0.0;
        let mut disp_y = 0.0;
        let mut disp_z = 0.0;

        for i in 0..n {
            let r = (p - params.constraints[i].source).norm();
            let phi = evaluate_kernel(kernel, r, params.smoothness);
            disp_x += wx[i] * phi;
            disp_y += wy[i] * phi;
            disp_z += wz[i] * phi;
        }

        // Add polynomial terms for thin-plate spline
        if use_polynomial {
            disp_x += wx[n] + wx[n + 1] * p.x + wx[n + 2] * p.y + wx[n + 3] * p.z;
            disp_y += wy[n] + wy[n + 1] * p.x + wy[n + 2] * p.y + wy[n + 3] * p.z;
            disp_z += wz[n] + wz[n + 1] * p.x + wz[n + 2] * p.y + wz[n + 3] * p.z;
        }

        // Apply displacement
        vertex.position = Point3::new(p.x + disp_x, p.y + disp_y, p.z + disp_z);

        let displacement = Vector3::new(disp_x, disp_y, disp_z).norm();
        max_displacement = max_displacement.max(displacement);
        total_displacement += displacement;
        vertices_modified += 1;
    }

    let average_displacement = if vertices_modified > 0 {
        total_displacement / vertices_modified as f64
    } else {
        0.0
    };

    // Calculate stretch/compression metrics
    let (max_stretch, max_compression) = calculate_distortion_metrics(mesh, &morphed);

    // Calculate new volume
    let new_volume = morphed.volume();
    let volume_ratio = if original_volume > 0.0 {
        new_volume / original_volume
    } else {
        1.0
    };

    Ok(MorphResult {
        mesh: morphed,
        vertices_modified,
        max_displacement,
        average_displacement,
        max_stretch,
        max_compression,
        volume_ratio,
    })
}

/// Evaluate an RBF kernel at distance r.
fn evaluate_kernel(kernel: RbfKernel, r: f64, smoothness: f64) -> f64 {
    match kernel {
        RbfKernel::ThinPlateSpline => {
            if r < 1e-10 {
                0.0
            } else {
                r * r * r.ln()
            }
        }
        RbfKernel::Gaussian => {
            let sigma = smoothness;
            (-r * r / (sigma * sigma)).exp()
        }
        RbfKernel::Multiquadric => {
            let c = smoothness;
            (r * r + c * c).sqrt()
        }
        RbfKernel::InverseMultiquadric => {
            let c = smoothness;
            1.0 / (r * r + c * c).sqrt()
        }
    }
}

/// FFD morphing implementation.
fn morph_ffd(mesh: &Mesh, params: &MorphParams) -> MeshResult<MorphResult> {
    let (nx, ny, nz) = params.ffd_resolution;

    if nx < 2 || ny < 2 || nz < 2 {
        return Err(MeshError::RepairFailed {
            details: "FFD resolution must be at least 2x2x2".to_string(),
        });
    }

    // Get mesh bounds to define the lattice
    let (min, max) = mesh.bounds().ok_or(MeshError::EmptyMesh {
        details: "Cannot get bounds of empty mesh".to_string(),
    })?;

    // Add small padding to avoid boundary issues
    let padding = 0.01;
    let range = max - min;
    let lattice_min = Point3::new(
        min.x - padding * range.x,
        min.y - padding * range.y,
        min.z - padding * range.z,
    );
    let lattice_max = Point3::new(
        max.x + padding * range.x,
        max.y + padding * range.y,
        max.z + padding * range.z,
    );
    let lattice_size = lattice_max - lattice_min;

    // Initialize control points on a regular grid
    let mut control_points: Vec<Vec<Vec<Point3<f64>>>> = Vec::with_capacity(nx);
    for i in 0..nx {
        let mut plane = Vec::with_capacity(ny);
        for j in 0..ny {
            let mut row = Vec::with_capacity(nz);
            for k in 0..nz {
                let u = i as f64 / (nx - 1) as f64;
                let v = j as f64 / (ny - 1) as f64;
                let w = k as f64 / (nz - 1) as f64;
                row.push(Point3::new(
                    lattice_min.x + u * lattice_size.x,
                    lattice_min.y + v * lattice_size.y,
                    lattice_min.z + w * lattice_size.z,
                ));
            }
            plane.push(row);
        }
        control_points.push(plane);
    }

    // Apply constraints to nearest control points
    for constraint in &params.constraints {
        // Find lattice coordinates
        let u = (constraint.source.x - lattice_min.x) / lattice_size.x;
        let v = (constraint.source.y - lattice_min.y) / lattice_size.y;
        let w = (constraint.source.z - lattice_min.z) / lattice_size.z;

        // Get nearest control point indices
        let i = ((u * (nx - 1) as f64).round() as usize).min(nx - 1);
        let j = ((v * (ny - 1) as f64).round() as usize).min(ny - 1);
        let k = ((w * (nz - 1) as f64).round() as usize).min(nz - 1);

        // Displace the control point
        let disp = constraint.displacement_vector() * constraint.weight;
        control_points[i][j][k] += disp;

        // Optionally influence nearby control points for smoother deformation
        // (using a simple falloff)
        let influence_radius = 1; // Influence neighboring cells
        for di in 0..=influence_radius * 2 {
            for dj in 0..=influence_radius * 2 {
                for dk in 0..=influence_radius * 2 {
                    let ni = (i + di).saturating_sub(influence_radius);
                    let nj = (j + dj).saturating_sub(influence_radius);
                    let nk = (k + dk).saturating_sub(influence_radius);

                    if ni < nx && nj < ny && nk < nz && (ni != i || nj != j || nk != k) {
                        let dist = ((ni as f64 - i as f64).powi(2)
                            + (nj as f64 - j as f64).powi(2)
                            + (nk as f64 - k as f64).powi(2))
                        .sqrt();
                        let falloff = (1.0 - dist / (influence_radius as f64 + 1.0)).max(0.0);
                        control_points[ni][nj][nk] += disp * falloff * 0.5;
                    }
                }
            }
        }
    }

    // Calculate original volume
    let original_volume = mesh.volume();

    // Deform mesh vertices using trilinear interpolation
    let mut morphed = mesh.clone();
    let mut vertices_modified = 0;
    let mut max_displacement = 0.0f64;
    let mut total_displacement = 0.0f64;

    for (idx, vertex) in morphed.vertices.iter_mut().enumerate() {
        // Check region mask
        if let Some(ref mask) = params.region_mask
            && !mask.contains(&(idx as u32))
        {
            continue;
        }

        let p = vertex.position;

        // Convert to lattice coordinates [0, 1]
        let u = ((p.x - lattice_min.x) / lattice_size.x).clamp(0.0, 1.0);
        let v = ((p.y - lattice_min.y) / lattice_size.y).clamp(0.0, 1.0);
        let w = ((p.z - lattice_min.z) / lattice_size.z).clamp(0.0, 1.0);

        // Evaluate the deformed position using Bernstein polynomials
        let new_pos = evaluate_ffd(&control_points, nx, ny, nz, u, v, w);

        let displacement = (new_pos - p).norm();
        max_displacement = max_displacement.max(displacement);
        total_displacement += displacement;
        vertices_modified += 1;

        vertex.position = new_pos;
    }

    let average_displacement = if vertices_modified > 0 {
        total_displacement / vertices_modified as f64
    } else {
        0.0
    };

    // Calculate stretch/compression metrics
    let (max_stretch, max_compression) = calculate_distortion_metrics(mesh, &morphed);

    // Calculate new volume
    let new_volume = morphed.volume();
    let volume_ratio = if original_volume > 0.0 {
        new_volume / original_volume
    } else {
        1.0
    };

    Ok(MorphResult {
        mesh: morphed,
        vertices_modified,
        max_displacement,
        average_displacement,
        max_stretch,
        max_compression,
        volume_ratio,
    })
}

/// Evaluate FFD at parameter coordinates (u, v, w) using Bernstein polynomials.
#[allow(clippy::needless_range_loop)] // 3D lattice indexing is clearer with explicit indices
fn evaluate_ffd(
    control_points: &[Vec<Vec<Point3<f64>>>],
    nx: usize,
    ny: usize,
    nz: usize,
    u: f64,
    v: f64,
    w: f64,
) -> Point3<f64> {
    let mut result = Point3::new(0.0, 0.0, 0.0);

    for i in 0..nx {
        for j in 0..ny {
            for k in 0..nz {
                let bi = bernstein(nx - 1, i, u);
                let bj = bernstein(ny - 1, j, v);
                let bk = bernstein(nz - 1, k, w);
                let weight = bi * bj * bk;
                result += control_points[i][j][k].coords * weight;
            }
        }
    }

    result
}

/// Evaluate Bernstein polynomial B_{i,n}(t).
fn bernstein(n: usize, i: usize, t: f64) -> f64 {
    binomial(n, i) as f64 * t.powi(i as i32) * (1.0 - t).powi((n - i) as i32)
}

/// Calculate binomial coefficient C(n, k).
fn binomial(n: usize, k: usize) -> usize {
    if k > n {
        return 0;
    }
    if k == 0 || k == n {
        return 1;
    }

    // Use the smaller k to minimize iterations
    let k = k.min(n - k);
    let mut result = 1usize;

    for i in 0..k {
        result = result * (n - i) / (i + 1);
    }

    result
}

/// Calculate distortion metrics by comparing edge lengths.
fn calculate_distortion_metrics(original: &Mesh, deformed: &Mesh) -> (f64, f64) {
    let mut max_stretch = 1.0f64;
    let mut max_compression = 1.0f64;

    for face in &original.faces {
        for i in 0..3 {
            let j = (i + 1) % 3;
            let v0_orig = original.vertices[face[i] as usize].position;
            let v1_orig = original.vertices[face[j] as usize].position;
            let v0_def = deformed.vertices[face[i] as usize].position;
            let v1_def = deformed.vertices[face[j] as usize].position;

            let orig_len = (v1_orig - v0_orig).norm();
            let def_len = (v1_def - v0_def).norm();

            if orig_len > 1e-10 {
                let ratio = def_len / orig_len;
                if ratio > 1.0 {
                    max_stretch = max_stretch.max(ratio);
                } else if ratio < 1.0 && ratio > 1e-10 {
                    max_compression = max_compression.max(1.0 / ratio);
                }
            }
        }
    }

    (max_stretch, max_compression)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::Vertex;

    fn create_test_tetrahedron() -> Mesh {
        let mut mesh = Mesh::new();
        mesh.vertices.push(Vertex::from_coords(0.0, 0.0, 0.0));
        mesh.vertices.push(Vertex::from_coords(10.0, 0.0, 0.0));
        mesh.vertices.push(Vertex::from_coords(5.0, 8.66, 0.0));
        mesh.vertices.push(Vertex::from_coords(5.0, 2.89, 8.16));
        mesh.faces.push([0, 2, 1]); // Bottom
        mesh.faces.push([0, 1, 3]); // Front
        mesh.faces.push([1, 2, 3]); // Right
        mesh.faces.push([2, 0, 3]); // Left
        mesh
    }

    fn create_test_cube() -> Mesh {
        let mut mesh = Mesh::new();
        // 8 vertices of a unit cube
        mesh.vertices.push(Vertex::from_coords(0.0, 0.0, 0.0));
        mesh.vertices.push(Vertex::from_coords(10.0, 0.0, 0.0));
        mesh.vertices.push(Vertex::from_coords(10.0, 10.0, 0.0));
        mesh.vertices.push(Vertex::from_coords(0.0, 10.0, 0.0));
        mesh.vertices.push(Vertex::from_coords(0.0, 0.0, 10.0));
        mesh.vertices.push(Vertex::from_coords(10.0, 0.0, 10.0));
        mesh.vertices.push(Vertex::from_coords(10.0, 10.0, 10.0));
        mesh.vertices.push(Vertex::from_coords(0.0, 10.0, 10.0));

        // 12 triangles (2 per face)
        mesh.faces.push([0, 2, 1]); // Bottom
        mesh.faces.push([0, 3, 2]);
        mesh.faces.push([4, 5, 6]); // Top
        mesh.faces.push([4, 6, 7]);
        mesh.faces.push([0, 1, 5]); // Front
        mesh.faces.push([0, 5, 4]);
        mesh.faces.push([2, 3, 7]); // Back
        mesh.faces.push([2, 7, 6]);
        mesh.faces.push([0, 4, 7]); // Left
        mesh.faces.push([0, 7, 3]);
        mesh.faces.push([1, 2, 6]); // Right
        mesh.faces.push([1, 6, 5]);
        mesh
    }

    #[test]
    fn test_rbf_identity_morph() {
        let mesh = create_test_tetrahedron();

        // Identity constraints (target == source) - need at least 3 well-separated points
        let constraints = vec![
            Constraint::point(Point3::new(0.0, 0.0, 0.0), Point3::new(0.0, 0.0, 0.0)),
            Constraint::point(Point3::new(10.0, 0.0, 0.0), Point3::new(10.0, 0.0, 0.0)),
            Constraint::point(Point3::new(5.0, 8.66, 0.0), Point3::new(5.0, 8.66, 0.0)),
            Constraint::point(Point3::new(5.0, 2.89, 8.16), Point3::new(5.0, 2.89, 8.16)),
        ];

        let params = MorphParams::rbf().with_constraints(constraints);
        let result = morph_mesh(&mesh, &params).unwrap();

        // With identity constraints, displacement should be minimal
        assert!(
            result.max_displacement < 0.1,
            "Max displacement: {}",
            result.max_displacement
        );
    }

    #[test]
    fn test_rbf_translation() {
        let mesh = create_test_tetrahedron();

        // Translate by (5, 0, 0) using constraints at all vertices
        let constraints = vec![
            Constraint::displacement(Point3::new(0.0, 0.0, 0.0), Vector3::new(5.0, 0.0, 0.0)),
            Constraint::displacement(Point3::new(10.0, 0.0, 0.0), Vector3::new(5.0, 0.0, 0.0)),
            Constraint::displacement(Point3::new(5.0, 8.66, 0.0), Vector3::new(5.0, 0.0, 0.0)),
            Constraint::displacement(Point3::new(5.0, 2.89, 8.16), Vector3::new(5.0, 0.0, 0.0)),
        ];

        let params = MorphParams::rbf().with_constraints(constraints);
        let result = morph_mesh(&mesh, &params).unwrap();

        // All vertices should be displaced by approximately 5mm in X
        for (i, vertex) in result.mesh.vertices.iter().enumerate() {
            let orig = &mesh.vertices[i];
            let dx = vertex.position.x - orig.position.x;
            assert!(
                (dx - 5.0).abs() < 0.1,
                "Vertex {} X displacement: {}",
                i,
                dx
            );
        }

        // Volume should be approximately preserved
        assert!(
            (result.volume_ratio - 1.0).abs() < 0.01,
            "Volume ratio: {}",
            result.volume_ratio
        );
    }

    #[test]
    fn test_rbf_local_deformation() {
        let mesh = create_test_cube();

        // Deform just one corner
        let constraints = vec![Constraint::displacement(
            Point3::new(0.0, 0.0, 0.0),
            Vector3::new(0.0, 0.0, 5.0),
        )];

        let params = MorphParams::rbf_gaussian()
            .with_constraints(constraints)
            .with_smoothness(5.0);
        let result = morph_mesh(&mesh, &params).unwrap();

        // The corner at origin should be displaced most
        let corner_disp = (result.mesh.vertices[0].position.z - mesh.vertices[0].position.z).abs();
        assert!(
            corner_disp > 4.0,
            "Corner displacement should be significant: {}",
            corner_disp
        );

        // Far corner should be displaced less
        let far_corner_disp =
            (result.mesh.vertices[6].position.z - mesh.vertices[6].position.z).abs();
        assert!(
            far_corner_disp < corner_disp,
            "Far corner should be displaced less: {}",
            far_corner_disp
        );
    }

    #[test]
    fn test_ffd_identity() {
        let mesh = create_test_cube();

        // No constraints = identity transformation
        let constraints = vec![
            Constraint::point(Point3::new(5.0, 5.0, 5.0), Point3::new(5.0, 5.0, 5.0)), // Center, no change
        ];

        let params = MorphParams::ffd().with_constraints(constraints);
        let result = morph_mesh(&mesh, &params).unwrap();

        // Displacement should be small
        assert!(result.max_displacement < 2.0);
    }

    #[test]
    fn test_ffd_scaling() {
        let mesh = create_test_cube();

        // Scale the mesh by moving control points outward
        let constraints = vec![
            Constraint::displacement(Point3::new(0.0, 0.0, 0.0), Vector3::new(-2.0, -2.0, -2.0)),
            Constraint::displacement(Point3::new(10.0, 10.0, 10.0), Vector3::new(2.0, 2.0, 2.0)),
        ];

        let params = MorphParams::ffd_with_resolution(3, 3, 3).with_constraints(constraints);
        let result = morph_mesh(&mesh, &params).unwrap();

        // Volume should increase (scaled up)
        assert!(
            result.volume_ratio > 1.0,
            "Volume should increase: {}",
            result.volume_ratio
        );
    }

    #[test]
    fn test_region_mask() {
        let mesh = create_test_cube();

        // Only affect vertices 0-3 (bottom face)
        let mut region: HashSet<u32> = HashSet::new();
        region.insert(0);
        region.insert(1);
        region.insert(2);
        region.insert(3);

        // Need multiple well-separated constraints for numerical stability
        let constraints = vec![
            Constraint::displacement(Point3::new(0.0, 0.0, 0.0), Vector3::new(0.0, 0.0, -2.0)),
            Constraint::displacement(Point3::new(10.0, 0.0, 0.0), Vector3::new(0.0, 0.0, -2.0)),
            Constraint::displacement(Point3::new(10.0, 10.0, 0.0), Vector3::new(0.0, 0.0, -2.0)),
            Constraint::displacement(Point3::new(0.0, 10.0, 0.0), Vector3::new(0.0, 0.0, -2.0)),
        ];

        let params = MorphParams::rbf()
            .with_constraints(constraints)
            .with_region(region);
        let result = morph_mesh(&mesh, &params).unwrap();

        assert_eq!(result.vertices_modified, 4);

        // Top vertices should not be displaced
        for i in 4..8 {
            let orig = mesh.vertices[i].position;
            let new = result.mesh.vertices[i].position;
            assert!(
                (orig - new).norm() < 1e-10,
                "Top vertex {} should not move",
                i
            );
        }
    }

    #[test]
    fn test_empty_mesh_error() {
        let mesh = Mesh::new();
        let constraints = vec![Constraint::point(
            Point3::new(0.0, 0.0, 0.0),
            Point3::new(1.0, 0.0, 0.0),
        )];
        let params = MorphParams::rbf().with_constraints(constraints);

        assert!(matches!(
            morph_mesh(&mesh, &params),
            Err(MeshError::EmptyMesh { .. })
        ));
    }

    #[test]
    fn test_no_constraints_error() {
        let mesh = create_test_tetrahedron();
        let params = MorphParams::rbf();

        assert!(matches!(
            morph_mesh(&mesh, &params),
            Err(MeshError::RepairFailed { .. })
        ));
    }

    #[test]
    fn test_distortion_metrics() {
        let mesh = create_test_cube();

        // Create a significant stretch in X direction
        let constraints = vec![
            Constraint::displacement(Point3::new(10.0, 0.0, 0.0), Vector3::new(5.0, 0.0, 0.0)),
            Constraint::displacement(Point3::new(10.0, 10.0, 0.0), Vector3::new(5.0, 0.0, 0.0)),
            Constraint::displacement(Point3::new(10.0, 0.0, 10.0), Vector3::new(5.0, 0.0, 0.0)),
            Constraint::displacement(Point3::new(10.0, 10.0, 10.0), Vector3::new(5.0, 0.0, 0.0)),
        ];

        let params = MorphParams::rbf().with_constraints(constraints);
        let result = morph_mesh(&mesh, &params).unwrap();

        // Should have stretch > 1
        assert!(
            result.max_stretch > 1.0,
            "Should have stretch: {}",
            result.max_stretch
        );
        assert!(result.has_significant_distortion(0.1));
    }

    #[test]
    fn test_weighted_constraints() {
        let mesh = create_test_tetrahedron();

        // Multiple well-separated constraints for numerical stability
        let constraints = vec![
            Constraint::weighted(Point3::new(0.0, 0.0, 0.0), Point3::new(0.0, 0.0, 10.0), 2.0),
            Constraint::weighted(
                Point3::new(10.0, 0.0, 0.0),
                Point3::new(10.0, 0.0, 5.0),
                1.0,
            ),
            Constraint::weighted(
                Point3::new(5.0, 8.66, 0.0),
                Point3::new(5.0, 8.66, 3.0),
                1.0,
            ),
        ];

        let params = MorphParams::rbf().with_constraints(constraints);
        let result = morph_mesh(&mesh, &params).unwrap();

        // Higher weighted constraint should have more influence
        let v0_z = result.mesh.vertices[0].position.z;
        let v1_z = result.mesh.vertices[1].position.z;

        // Vertex 0 has higher weight target at z=10, vertex 1 at z=5
        // Expect v0 to be displaced more than v1
        assert!(v0_z > v1_z, "v0_z={} should be > v1_z={}", v0_z, v1_z);
    }

    #[test]
    fn test_different_kernels() {
        let mesh = create_test_cube();
        // Multiple well-separated constraints for numerical stability
        let constraints = vec![
            Constraint::displacement(Point3::new(5.0, 5.0, 10.0), Vector3::new(0.0, 0.0, 5.0)),
            Constraint::displacement(Point3::new(0.0, 0.0, 10.0), Vector3::new(0.0, 0.0, 3.0)),
            Constraint::displacement(Point3::new(10.0, 10.0, 10.0), Vector3::new(0.0, 0.0, 3.0)),
        ];

        // Test all kernel types
        for kernel in [
            RbfKernel::ThinPlateSpline,
            RbfKernel::Gaussian,
            RbfKernel::Multiquadric,
            RbfKernel::InverseMultiquadric,
        ] {
            let params = MorphParams {
                algorithm: MorphAlgorithm::Rbf(kernel),
                constraints: constraints.clone(),
                smoothness: 5.0,
                ..Default::default()
            };
            let result = morph_mesh(&mesh, &params).unwrap();
            assert!(
                result.max_displacement > 0.0,
                "Kernel {:?} should produce displacement",
                kernel
            );
        }
    }

    #[test]
    fn test_bernstein_basis() {
        // B_{0,2}(0.5) = (1-t)^2 = 0.25
        assert!((bernstein(2, 0, 0.5) - 0.25).abs() < 1e-10);
        // B_{1,2}(0.5) = 2t(1-t) = 0.5
        assert!((bernstein(2, 1, 0.5) - 0.5).abs() < 1e-10);
        // B_{2,2}(0.5) = t^2 = 0.25
        assert!((bernstein(2, 2, 0.5) - 0.25).abs() < 1e-10);

        // Sum should be 1
        let sum: f64 = (0..=3).map(|i| bernstein(3, i, 0.3)).sum();
        assert!((sum - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_binomial_coefficients() {
        assert_eq!(binomial(0, 0), 1);
        assert_eq!(binomial(4, 0), 1);
        assert_eq!(binomial(4, 4), 1);
        assert_eq!(binomial(4, 2), 6);
        assert_eq!(binomial(5, 2), 10);
        assert_eq!(binomial(10, 5), 252);
    }
}

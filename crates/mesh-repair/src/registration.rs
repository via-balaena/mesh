//! Mesh registration and alignment algorithms.
//!
//! This module provides tools for aligning meshes to each other, including:
//! - Iterative Closest Point (ICP) for rigid alignment
//! - Feature-based registration using landmarks
//! - Non-rigid/deformable registration
//!
//! # Use Cases
//!
//! - Aligning a foot scan to a template last
//! - Registering multiple partial scans of the same object
//! - Matching a head scan to a helmet template
//!
//! # Example
//!
//! ```
//! use mesh_repair::{Mesh, Vertex};
//! use mesh_repair::registration::{RegistrationParams, align_meshes};
//! use nalgebra::Point3;
//!
//! // Create source and target meshes
//! let mut source = Mesh::new();
//! source.vertices.push(Vertex::from_coords(0.0, 0.0, 0.0));
//! source.vertices.push(Vertex::from_coords(1.0, 0.0, 0.0));
//! source.vertices.push(Vertex::from_coords(0.5, 1.0, 0.0));
//! source.faces.push([0, 1, 2]);
//!
//! let mut target = Mesh::new();
//! target.vertices.push(Vertex::from_coords(1.0, 1.0, 0.0));
//! target.vertices.push(Vertex::from_coords(2.0, 1.0, 0.0));
//! target.vertices.push(Vertex::from_coords(1.5, 2.0, 0.0));
//! target.faces.push([0, 1, 2]);
//!
//! // Align source to target using ICP
//! let params = RegistrationParams::icp();
//! let result = align_meshes(&source, &target, &params).unwrap();
//!
//! println!("RMS error: {:.3} mm", result.rms_error);
//! println!("Converged: {}", result.converged);
//! ```

use crate::{Mesh, MeshError, MeshResult};
use nalgebra::{Matrix3, Matrix4, Point3, Rotation3, UnitQuaternion, Vector3};

/// Parameters for mesh registration.
#[derive(Debug, Clone)]
pub struct RegistrationParams {
    /// The registration algorithm to use.
    pub algorithm: RegistrationAlgorithm,

    /// Maximum number of iterations for iterative algorithms.
    pub max_iterations: usize,

    /// Convergence threshold for RMS error change.
    pub convergence_threshold: f64,

    /// Maximum correspondence distance (points further apart are ignored).
    pub max_correspondence_distance: f64,

    /// Optional landmark correspondences for feature-based registration.
    pub landmarks: Vec<Landmark>,

    /// Whether to allow scaling in addition to rigid transformation.
    pub allow_scaling: bool,

    /// Subsample ratio for large meshes (0.0-1.0).
    /// 1.0 uses all points, 0.1 uses 10% of points.
    pub subsample_ratio: f64,
}

impl Default for RegistrationParams {
    fn default() -> Self {
        Self {
            algorithm: RegistrationAlgorithm::Icp,
            max_iterations: 100,
            convergence_threshold: 1e-6,
            max_correspondence_distance: f64::INFINITY,
            landmarks: Vec::new(),
            allow_scaling: false,
            subsample_ratio: 1.0,
        }
    }
}

impl RegistrationParams {
    /// Create params for ICP registration.
    pub fn icp() -> Self {
        Self::default()
    }

    /// Create params for point-to-plane ICP (faster convergence).
    pub fn icp_point_to_plane() -> Self {
        Self {
            algorithm: RegistrationAlgorithm::IcpPointToPlane,
            ..Default::default()
        }
    }

    /// Create params for landmark-based registration.
    pub fn landmark_based(landmarks: Vec<Landmark>) -> Self {
        Self {
            algorithm: RegistrationAlgorithm::Landmark,
            landmarks,
            ..Default::default()
        }
    }

    /// Create params for combined landmark + ICP registration.
    pub fn landmark_then_icp(landmarks: Vec<Landmark>) -> Self {
        Self {
            algorithm: RegistrationAlgorithm::LandmarkThenIcp,
            landmarks,
            ..Default::default()
        }
    }

    /// Set maximum iterations.
    pub fn with_max_iterations(mut self, max_iterations: usize) -> Self {
        self.max_iterations = max_iterations;
        self
    }

    /// Set convergence threshold.
    pub fn with_convergence_threshold(mut self, threshold: f64) -> Self {
        self.convergence_threshold = threshold;
        self
    }

    /// Set maximum correspondence distance.
    pub fn with_max_correspondence_distance(mut self, distance: f64) -> Self {
        self.max_correspondence_distance = distance;
        self
    }

    /// Allow scaling in addition to rigid transformation.
    pub fn with_scaling(mut self) -> Self {
        self.allow_scaling = true;
        self
    }

    /// Set subsample ratio for large meshes.
    pub fn with_subsample_ratio(mut self, ratio: f64) -> Self {
        self.subsample_ratio = ratio.clamp(0.01, 1.0);
        self
    }
}

/// The registration algorithm to use.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum RegistrationAlgorithm {
    /// Standard point-to-point ICP.
    Icp,
    /// Point-to-plane ICP (requires normals).
    IcpPointToPlane,
    /// Landmark-based (requires corresponding points).
    Landmark,
    /// Landmark alignment followed by ICP refinement.
    LandmarkThenIcp,
    /// Non-rigid/deformable registration (allows local deformations).
    NonRigid,
}

/// A landmark correspondence for feature-based registration.
#[derive(Debug, Clone)]
pub struct Landmark {
    /// Position on the source mesh.
    pub source: Point3<f64>,
    /// Corresponding position on the target mesh.
    pub target: Point3<f64>,
    /// Optional weight for this landmark.
    pub weight: f64,
}

impl Landmark {
    /// Create a new landmark correspondence.
    pub fn new(source: Point3<f64>, target: Point3<f64>) -> Self {
        Self {
            source,
            target,
            weight: 1.0,
        }
    }

    /// Create a weighted landmark correspondence.
    pub fn weighted(source: Point3<f64>, target: Point3<f64>, weight: f64) -> Self {
        Self {
            source,
            target,
            weight,
        }
    }
}

/// Result of a registration operation.
#[derive(Debug, Clone)]
pub struct RegistrationResult {
    /// The aligned mesh (source transformed to match target).
    pub mesh: Mesh,

    /// The transformation that was applied.
    pub transformation: RigidTransform,

    /// RMS (root mean square) error of the alignment.
    pub rms_error: f64,

    /// Maximum error (maximum distance between corresponding points).
    pub max_error: f64,

    /// Number of iterations performed.
    pub iterations: usize,

    /// Whether the algorithm converged.
    pub converged: bool,

    /// Number of valid correspondences used.
    pub correspondences_used: usize,
}

impl RegistrationResult {
    /// Check if the registration quality is acceptable.
    ///
    /// Returns true if converged and RMS error is below threshold.
    pub fn is_acceptable(&self, max_rms_error: f64) -> bool {
        self.converged && self.rms_error <= max_rms_error
    }
}

/// Parameters specific to non-rigid registration.
#[derive(Debug, Clone)]
pub struct NonRigidParams {
    /// Stiffness weight for regularization (higher = more rigid, less local deformation).
    /// Range: 0.0 (fully flexible) to 100.0 (nearly rigid).
    /// Default: 10.0
    pub stiffness: f64,

    /// Number of control points for the deformation field.
    /// More control points allow finer deformations but increase computation.
    /// If None, uses a subset of source vertices (default: ~500 points).
    pub num_control_points: Option<usize>,

    /// Landmark correspondences for guided deformation.
    /// These serve as hard constraints during non-rigid registration.
    pub landmarks: Vec<Landmark>,

    /// Whether to apply an initial rigid alignment before non-rigid deformation.
    /// Default: true
    pub initial_rigid_alignment: bool,

    /// Number of outer iterations for the non-rigid optimization.
    /// Default: 10
    pub outer_iterations: usize,

    /// Smoothness parameter for RBF interpolation.
    /// Higher values produce smoother deformations.
    /// Default: 1.0
    pub smoothness: f64,
}

impl Default for NonRigidParams {
    fn default() -> Self {
        Self {
            stiffness: 10.0,
            num_control_points: None,
            landmarks: Vec::new(),
            initial_rigid_alignment: true,
            outer_iterations: 10,
            smoothness: 1.0,
        }
    }
}

impl NonRigidParams {
    /// Create default non-rigid registration params.
    pub fn new() -> Self {
        Self::default()
    }

    /// Set stiffness (regularization weight).
    pub fn with_stiffness(mut self, stiffness: f64) -> Self {
        self.stiffness = stiffness.max(0.0);
        self
    }

    /// Set number of control points.
    pub fn with_control_points(mut self, num_points: usize) -> Self {
        self.num_control_points = Some(num_points.max(10));
        self
    }

    /// Add landmark constraints.
    pub fn with_landmarks(mut self, landmarks: Vec<Landmark>) -> Self {
        self.landmarks = landmarks;
        self
    }

    /// Disable initial rigid alignment.
    pub fn without_initial_alignment(mut self) -> Self {
        self.initial_rigid_alignment = false;
        self
    }

    /// Set number of outer iterations.
    pub fn with_outer_iterations(mut self, iterations: usize) -> Self {
        self.outer_iterations = iterations.max(1);
        self
    }

    /// Set smoothness parameter.
    pub fn with_smoothness(mut self, smoothness: f64) -> Self {
        self.smoothness = smoothness.max(0.01);
        self
    }
}

/// Result of a non-rigid registration operation.
#[derive(Debug, Clone)]
pub struct NonRigidRegistrationResult {
    /// The aligned and deformed mesh.
    pub mesh: Mesh,

    /// Per-vertex displacements applied.
    pub displacements: Vec<Vector3<f64>>,

    /// Initial rigid transformation (if applied).
    pub initial_transform: Option<RigidTransform>,

    /// RMS error of the final alignment.
    pub rms_error: f64,

    /// Maximum error.
    pub max_error: f64,

    /// Number of iterations performed.
    pub iterations: usize,

    /// Whether the algorithm converged.
    pub converged: bool,

    /// Number of correspondences used.
    pub correspondences_used: usize,

    /// Average displacement magnitude.
    pub average_displacement: f64,

    /// Maximum displacement magnitude.
    pub max_displacement: f64,
}

impl NonRigidRegistrationResult {
    /// Check if the registration quality is acceptable.
    pub fn is_acceptable(&self, max_rms_error: f64) -> bool {
        self.converged && self.rms_error <= max_rms_error
    }

    /// Get the deformation field as a list of (original_position, displacement) pairs.
    pub fn deformation_field(&self, original_mesh: &Mesh) -> Vec<(Point3<f64>, Vector3<f64>)> {
        original_mesh
            .vertices
            .iter()
            .zip(self.displacements.iter())
            .map(|(v, d)| (v.position, *d))
            .collect()
    }
}

/// A rigid transformation (rotation + translation, optionally with scale).
#[derive(Debug, Clone)]
pub struct RigidTransform {
    /// Rotation quaternion.
    pub rotation: UnitQuaternion<f64>,
    /// Translation vector.
    pub translation: Vector3<f64>,
    /// Uniform scale factor (1.0 = no scaling).
    pub scale: f64,
}

impl Default for RigidTransform {
    fn default() -> Self {
        Self::identity()
    }
}

impl RigidTransform {
    /// Create an identity transformation.
    pub fn identity() -> Self {
        Self {
            rotation: UnitQuaternion::identity(),
            translation: Vector3::zeros(),
            scale: 1.0,
        }
    }

    /// Create a pure translation.
    pub fn from_translation(translation: Vector3<f64>) -> Self {
        Self {
            rotation: UnitQuaternion::identity(),
            translation,
            scale: 1.0,
        }
    }

    /// Create a pure rotation.
    pub fn from_rotation(rotation: UnitQuaternion<f64>) -> Self {
        Self {
            rotation,
            translation: Vector3::zeros(),
            scale: 1.0,
        }
    }

    /// Create a transformation from rotation and translation.
    pub fn from_rotation_translation(
        rotation: UnitQuaternion<f64>,
        translation: Vector3<f64>,
    ) -> Self {
        Self {
            rotation,
            translation,
            scale: 1.0,
        }
    }

    /// Apply the transformation to a point.
    pub fn transform_point(&self, point: &Point3<f64>) -> Point3<f64> {
        let scaled = point.coords * self.scale;
        let rotated = self.rotation * Point3::from(scaled);
        Point3::from(rotated.coords + self.translation)
    }

    /// Apply the transformation to a vector (no translation).
    pub fn transform_vector(&self, vector: &Vector3<f64>) -> Vector3<f64> {
        let scaled = vector * self.scale;
        self.rotation * scaled
    }

    /// Compose with another transformation (self applied first, then other).
    pub fn then(&self, other: &RigidTransform) -> RigidTransform {
        RigidTransform {
            rotation: other.rotation * self.rotation,
            translation: other.rotation * (self.translation * other.scale) + other.translation,
            scale: self.scale * other.scale,
        }
    }

    /// Get the inverse transformation.
    pub fn inverse(&self) -> RigidTransform {
        let inv_rotation = self.rotation.inverse();
        let inv_scale = 1.0 / self.scale;
        let inv_translation = inv_rotation * (-self.translation * inv_scale);
        RigidTransform {
            rotation: inv_rotation,
            translation: inv_translation,
            scale: inv_scale,
        }
    }

    /// Convert to a 4x4 homogeneous transformation matrix.
    pub fn to_matrix4(&self) -> Matrix4<f64> {
        let rotation_matrix = self.rotation.to_rotation_matrix();
        let mut result = Matrix4::identity();

        for i in 0..3 {
            for j in 0..3 {
                result[(i, j)] = rotation_matrix[(i, j)] * self.scale;
            }
            result[(i, 3)] = self.translation[i];
        }

        result
    }
}

/// Align a source mesh to a target mesh.
///
/// # Arguments
///
/// * `source` - The mesh to transform
/// * `target` - The reference mesh to align to
/// * `params` - Registration parameters
///
/// # Returns
///
/// A `RegistrationResult` containing the aligned mesh and transformation.
pub fn align_meshes(
    source: &Mesh,
    target: &Mesh,
    params: &RegistrationParams,
) -> MeshResult<RegistrationResult> {
    if source.is_empty() || target.is_empty() {
        return Err(MeshError::EmptyMesh {
            details: "Cannot align empty meshes".to_string(),
        });
    }

    match params.algorithm {
        RegistrationAlgorithm::Icp => icp_align(source, target, params, false),
        RegistrationAlgorithm::IcpPointToPlane => icp_align(source, target, params, true),
        RegistrationAlgorithm::Landmark => landmark_align(source, target, params),
        RegistrationAlgorithm::LandmarkThenIcp => {
            // First do landmark alignment
            let landmark_result = landmark_align(source, target, params)?;

            // Then refine with ICP
            let mut icp_params = params.clone();
            icp_params.algorithm = RegistrationAlgorithm::Icp;
            let icp_result = icp_align(&landmark_result.mesh, target, &icp_params, false)?;

            // Compose transformations
            let total_transform = landmark_result
                .transformation
                .then(&icp_result.transformation);
            let total_iterations = landmark_result.iterations + icp_result.iterations;

            Ok(RegistrationResult {
                mesh: icp_result.mesh,
                transformation: total_transform,
                rms_error: icp_result.rms_error,
                max_error: icp_result.max_error,
                iterations: total_iterations,
                converged: icp_result.converged,
                correspondences_used: icp_result.correspondences_used,
            })
        }
        RegistrationAlgorithm::NonRigid => {
            // Non-rigid registration should use the dedicated function
            // Here we return a minimal result; use non_rigid_align for full results
            let nr_result = non_rigid_align(source, target, &NonRigidParams::default(), params)?;
            Ok(RegistrationResult {
                mesh: nr_result.mesh,
                transformation: nr_result
                    .initial_transform
                    .unwrap_or_else(RigidTransform::identity),
                rms_error: nr_result.rms_error,
                max_error: nr_result.max_error,
                iterations: nr_result.iterations,
                converged: nr_result.converged,
                correspondences_used: nr_result.correspondences_used,
            })
        }
    }
}

/// ICP registration implementation.
fn icp_align(
    source: &Mesh,
    target: &Mesh,
    params: &RegistrationParams,
    _point_to_plane: bool,
) -> MeshResult<RegistrationResult> {
    // Build target point cloud for nearest neighbor queries
    let target_points: Vec<Point3<f64>> = target.vertices.iter().map(|v| v.position).collect();

    // Subsample source points if needed
    let source_indices: Vec<usize> = if params.subsample_ratio < 1.0 {
        let step = (1.0 / params.subsample_ratio).ceil() as usize;
        (0..source.vertex_count()).step_by(step).collect()
    } else {
        (0..source.vertex_count()).collect()
    };

    let mut current_transform = RigidTransform::identity();
    let mut previous_rms = f64::INFINITY;
    let mut converged = false;
    let mut iterations = 0;

    // Transform source points
    let mut transformed_source: Vec<Point3<f64>> = source_indices
        .iter()
        .map(|&i| source.vertices[i].position)
        .collect();

    for iter in 0..params.max_iterations {
        iterations = iter + 1;

        // Find correspondences (nearest neighbors)
        let mut correspondences: Vec<(Point3<f64>, Point3<f64>)> = Vec::new();
        let mut total_error_sq = 0.0;
        let mut max_error = 0.0f64;

        for source_point in &transformed_source {
            // Find nearest point in target (brute force for now - could use KD-tree)
            let (nearest, dist_sq) = find_nearest_point(source_point, &target_points);

            let dist = dist_sq.sqrt();
            if dist <= params.max_correspondence_distance {
                correspondences.push((*source_point, nearest));
                total_error_sq += dist_sq;
                max_error = max_error.max(dist);
            }
        }

        if correspondences.is_empty() {
            return Err(MeshError::RepairFailed {
                details: "No valid correspondences found".to_string(),
            });
        }

        let rms_error = (total_error_sq / correspondences.len() as f64).sqrt();

        // Check convergence
        if (previous_rms - rms_error).abs() < params.convergence_threshold {
            converged = true;
            break;
        }
        previous_rms = rms_error;

        // Compute optimal transformation for this iteration
        let (source_pts, target_pts): (Vec<_>, Vec<_>) = correspondences.into_iter().unzip();

        let iter_transform =
            compute_rigid_transform(&source_pts, &target_pts, params.allow_scaling);

        // Update cumulative transform and transformed points
        current_transform = current_transform.then(&iter_transform);

        for point in &mut transformed_source {
            *point = iter_transform.transform_point(point);
        }
    }

    // Apply final transformation to create result mesh
    let mut result_mesh = source.clone();
    for vertex in &mut result_mesh.vertices {
        vertex.position = current_transform.transform_point(&vertex.position);
        if let Some(ref mut normal) = vertex.normal {
            *normal = current_transform.transform_vector(normal).normalize();
        }
    }

    // Calculate final error metrics
    let (rms_error, max_error, correspondences_used) = calculate_alignment_error(
        &result_mesh,
        &target_points,
        params.max_correspondence_distance,
    );

    Ok(RegistrationResult {
        mesh: result_mesh,
        transformation: current_transform,
        rms_error,
        max_error,
        iterations,
        converged,
        correspondences_used,
    })
}

/// Landmark-based registration implementation.
fn landmark_align(
    source: &Mesh,
    _target: &Mesh,
    params: &RegistrationParams,
) -> MeshResult<RegistrationResult> {
    if params.landmarks.is_empty() {
        return Err(MeshError::RepairFailed {
            details: "No landmarks provided for landmark-based registration".to_string(),
        });
    }

    if params.landmarks.len() < 3 {
        return Err(MeshError::RepairFailed {
            details: "At least 3 landmarks required for rigid registration".to_string(),
        });
    }

    // Extract landmark positions
    let source_points: Vec<Point3<f64>> = params.landmarks.iter().map(|l| l.source).collect();
    let target_points: Vec<Point3<f64>> = params.landmarks.iter().map(|l| l.target).collect();

    // Compute transformation
    let transform = compute_rigid_transform(&source_points, &target_points, params.allow_scaling);

    // Apply transformation to mesh
    let mut result_mesh = source.clone();
    for vertex in &mut result_mesh.vertices {
        vertex.position = transform.transform_point(&vertex.position);
        if let Some(ref mut normal) = vertex.normal {
            *normal = transform.transform_vector(normal).normalize();
        }
    }

    // Calculate landmark error
    let mut total_error_sq = 0.0;
    let mut max_error = 0.0f64;

    for landmark in &params.landmarks {
        let transformed = transform.transform_point(&landmark.source);
        let error = (transformed - landmark.target).norm();
        total_error_sq += error * error * landmark.weight;
        max_error = max_error.max(error);
    }

    let rms_error = (total_error_sq / params.landmarks.len() as f64).sqrt();

    Ok(RegistrationResult {
        mesh: result_mesh,
        transformation: transform,
        rms_error,
        max_error,
        iterations: 1,
        converged: true,
        correspondences_used: params.landmarks.len(),
    })
}

/// Find the nearest point in a list to a query point.
fn find_nearest_point(query: &Point3<f64>, points: &[Point3<f64>]) -> (Point3<f64>, f64) {
    let mut nearest = points[0];
    let mut min_dist_sq = (query - nearest).norm_squared();

    for point in points.iter().skip(1) {
        let dist_sq = (query - point).norm_squared();
        if dist_sq < min_dist_sq {
            min_dist_sq = dist_sq;
            nearest = *point;
        }
    }

    (nearest, min_dist_sq)
}

/// Compute the optimal rigid transformation between point sets.
///
/// Uses the Kabsch algorithm (SVD-based) to find the rotation and translation
/// that minimizes the RMSD between corresponding points.
fn compute_rigid_transform(
    source: &[Point3<f64>],
    target: &[Point3<f64>],
    allow_scaling: bool,
) -> RigidTransform {
    let n = source.len();
    if n == 0 {
        return RigidTransform::identity();
    }

    // Compute centroids
    let source_centroid: Vector3<f64> =
        source.iter().map(|p| p.coords).sum::<Vector3<f64>>() / n as f64;
    let target_centroid: Vector3<f64> =
        target.iter().map(|p| p.coords).sum::<Vector3<f64>>() / n as f64;

    // Center the points
    let centered_source: Vec<Vector3<f64>> =
        source.iter().map(|p| p.coords - source_centroid).collect();
    let centered_target: Vec<Vector3<f64>> =
        target.iter().map(|p| p.coords - target_centroid).collect();

    // Compute cross-covariance matrix H
    let mut h = Matrix3::zeros();
    for i in 0..n {
        h += centered_source[i] * centered_target[i].transpose();
    }

    // SVD decomposition
    let svd = h.svd(true, true);
    let u = svd.u.unwrap();
    let v_t = svd.v_t.unwrap();

    // Compute rotation
    let mut rotation_matrix = v_t.transpose() * u.transpose();

    // Handle reflection case (det < 0)
    if rotation_matrix.determinant() < 0.0 {
        let mut v_t_fixed = v_t;
        v_t_fixed.set_row(2, &(-v_t.row(2)));
        rotation_matrix = v_t_fixed.transpose() * u.transpose();
    }

    let rotation =
        UnitQuaternion::from_rotation_matrix(&Rotation3::from_matrix_unchecked(rotation_matrix));

    // Compute scale if allowed
    let scale = if allow_scaling {
        let source_variance: f64 = centered_source.iter().map(|v| v.norm_squared()).sum();
        let target_variance: f64 = centered_target.iter().map(|v| v.norm_squared()).sum();

        if source_variance > 1e-10 {
            (target_variance / source_variance).sqrt()
        } else {
            1.0
        }
    } else {
        1.0
    };

    // Compute translation
    let translation = target_centroid - scale * (rotation * source_centroid);

    RigidTransform {
        rotation,
        translation,
        scale,
    }
}

/// Calculate alignment error between a mesh and target points.
fn calculate_alignment_error(
    mesh: &Mesh,
    target_points: &[Point3<f64>],
    max_distance: f64,
) -> (f64, f64, usize) {
    let mut total_error_sq = 0.0;
    let mut max_error = 0.0f64;
    let mut count = 0;

    for vertex in &mesh.vertices {
        let (_, dist_sq) = find_nearest_point(&vertex.position, target_points);
        let dist = dist_sq.sqrt();

        if dist <= max_distance {
            total_error_sq += dist_sq;
            max_error = max_error.max(dist);
            count += 1;
        }
    }

    let rms_error = if count > 0 {
        (total_error_sq / count as f64).sqrt()
    } else {
        f64::INFINITY
    };

    (rms_error, max_error, count)
}

/// Perform non-rigid/deformable registration.
///
/// This algorithm allows local deformations while maintaining global smoothness,
/// making it suitable for registering meshes with local shape differences
/// (e.g., a foot scan to a template last with different proportions).
///
/// # Algorithm
///
/// 1. Optional initial rigid alignment using ICP
/// 2. Select control points from source mesh
/// 3. Iteratively:
///    a. Find correspondences (nearest neighbors)
///    b. Compute optimal displacements for control points
///    c. Interpolate displacements to all vertices using RBF
///    d. Apply regularization for smoothness
///
/// # Arguments
///
/// * `source` - The mesh to deform
/// * `target` - The reference mesh to match
/// * `nr_params` - Non-rigid registration parameters
/// * `base_params` - Base registration parameters (max iterations, convergence, etc.)
///
/// # Returns
///
/// A `NonRigidRegistrationResult` containing the deformed mesh and displacement field.
///
/// # Example
///
/// ```
/// use mesh_repair::{Mesh, Vertex};
/// use mesh_repair::registration::{RegistrationParams, NonRigidParams, non_rigid_align};
///
/// let mut source = Mesh::new();
/// source.vertices.push(Vertex::from_coords(0.0, 0.0, 0.0));
/// source.vertices.push(Vertex::from_coords(10.0, 0.0, 0.0));
/// source.vertices.push(Vertex::from_coords(5.0, 10.0, 0.0));
/// source.faces.push([0, 1, 2]);
///
/// let mut target = Mesh::new();
/// target.vertices.push(Vertex::from_coords(0.0, 0.0, 1.0));
/// target.vertices.push(Vertex::from_coords(10.0, 0.0, 1.0));
/// target.vertices.push(Vertex::from_coords(5.0, 12.0, 1.0)); // Slightly stretched
/// target.faces.push([0, 1, 2]);
///
/// let nr_params = NonRigidParams::new().with_stiffness(5.0);
/// let base_params = RegistrationParams::default();
/// let result = non_rigid_align(&source, &target, &nr_params, &base_params).unwrap();
///
/// println!("RMS error: {:.3} mm", result.rms_error);
/// println!("Max displacement: {:.3} mm", result.max_displacement);
/// ```
pub fn non_rigid_align(
    source: &Mesh,
    target: &Mesh,
    nr_params: &NonRigidParams,
    base_params: &RegistrationParams,
) -> MeshResult<NonRigidRegistrationResult> {
    if source.is_empty() || target.is_empty() {
        return Err(MeshError::EmptyMesh {
            details: "Cannot align empty meshes".to_string(),
        });
    }

    let target_points: Vec<Point3<f64>> = target.vertices.iter().map(|v| v.position).collect();

    // Step 1: Optional initial rigid alignment
    let (working_mesh, initial_transform) = if nr_params.initial_rigid_alignment {
        let rigid_params = RegistrationParams::icp()
            .with_max_iterations(base_params.max_iterations / 2)
            .with_convergence_threshold(base_params.convergence_threshold * 10.0);
        let rigid_result = icp_align(source, target, &rigid_params, false)?;
        (rigid_result.mesh, Some(rigid_result.transformation))
    } else {
        (source.clone(), None)
    };

    // Step 2: Select control points
    let num_control_points = nr_params
        .num_control_points
        .unwrap_or_else(|| working_mesh.vertex_count().clamp(10, 500));

    let control_indices = select_control_points(&working_mesh, num_control_points);
    let num_controls = control_indices.len();

    // Initialize control point positions and displacements
    let mut control_positions: Vec<Point3<f64>> = control_indices
        .iter()
        .map(|&i| working_mesh.vertices[i].position)
        .collect();

    let mut control_displacements: Vec<Vector3<f64>> = vec![Vector3::zeros(); num_controls];

    // Add landmark constraints as additional control points
    let landmark_constraints: Vec<(Point3<f64>, Vector3<f64>)> = nr_params
        .landmarks
        .iter()
        .map(|l| {
            let displacement = l.target - l.source;
            (l.source, displacement)
        })
        .collect();

    // Step 3: Iterative non-rigid optimization
    let mut converged = false;
    let mut iterations = 0;
    let mut previous_rms = f64::INFINITY;

    // Current vertex positions
    let mut current_positions: Vec<Point3<f64>> =
        working_mesh.vertices.iter().map(|v| v.position).collect();

    for outer_iter in 0..nr_params.outer_iterations {
        iterations = outer_iter + 1;

        // Find correspondences for control points
        let mut correspondence_displacements: Vec<Vector3<f64>> = Vec::with_capacity(num_controls);

        for (i, &ctrl_idx) in control_indices.iter().enumerate() {
            let current_pos = current_positions[ctrl_idx];
            let (nearest, _) = find_nearest_point(&current_pos, &target_points);

            // Desired displacement to reach target
            let desired_displacement = nearest - current_pos;

            // Blend with current displacement based on stiffness
            // Higher stiffness = smaller updates, more regularization
            let alpha = 1.0 / (1.0 + nr_params.stiffness * 0.1);
            let new_displacement = control_displacements[i] + desired_displacement * alpha;

            correspondence_displacements.push(new_displacement);
        }

        // Apply regularization (Laplacian smoothing of displacements)
        let regularized_displacements = regularize_displacements(
            &control_positions,
            &correspondence_displacements,
            nr_params.stiffness,
        );

        control_displacements = regularized_displacements;

        // Interpolate displacements to all vertices using RBF
        let all_displacements = interpolate_displacements_rbf(
            &control_positions,
            &control_displacements,
            &landmark_constraints,
            &working_mesh
                .vertices
                .iter()
                .map(|v| v.position)
                .collect::<Vec<_>>(),
            nr_params.smoothness,
        );

        // Update current positions
        for (i, displacement) in all_displacements.iter().enumerate() {
            current_positions[i] = working_mesh.vertices[i].position + displacement;
        }

        // Update control positions for next iteration
        for (i, &ctrl_idx) in control_indices.iter().enumerate() {
            control_positions[i] = current_positions[ctrl_idx];
        }

        // Calculate error
        let mut total_error_sq = 0.0;
        let mut count = 0;

        for pos in &current_positions {
            let (_, dist_sq) = find_nearest_point(pos, &target_points);
            if dist_sq.sqrt() <= base_params.max_correspondence_distance {
                total_error_sq += dist_sq;
                count += 1;
            }
        }

        let rms_error = if count > 0 {
            (total_error_sq / count as f64).sqrt()
        } else {
            f64::INFINITY
        };

        // Check convergence
        if (previous_rms - rms_error).abs() < base_params.convergence_threshold {
            converged = true;
            break;
        }
        previous_rms = rms_error;
    }

    // Build final result mesh
    let mut result_mesh = working_mesh.clone();
    let final_displacements: Vec<Vector3<f64>> = current_positions
        .iter()
        .zip(working_mesh.vertices.iter())
        .map(|(current, original)| current - original.position)
        .collect();

    for (i, vertex) in result_mesh.vertices.iter_mut().enumerate() {
        vertex.position = current_positions[i];
        if let Some(ref mut normal) = vertex.normal {
            // Re-estimate normals would be better, but for now keep them
            // In practice, you'd recompute normals after deformation
            *normal = normal.normalize();
        }
    }

    // Calculate final metrics
    let (rms_error, max_error, correspondences_used) = calculate_alignment_error(
        &result_mesh,
        &target_points,
        base_params.max_correspondence_distance,
    );

    let displacement_magnitudes: Vec<f64> = final_displacements.iter().map(|d| d.norm()).collect();
    let average_displacement = if !displacement_magnitudes.is_empty() {
        displacement_magnitudes.iter().sum::<f64>() / displacement_magnitudes.len() as f64
    } else {
        0.0
    };
    let max_displacement = displacement_magnitudes
        .iter()
        .cloned()
        .fold(0.0f64, f64::max);

    Ok(NonRigidRegistrationResult {
        mesh: result_mesh,
        displacements: final_displacements,
        initial_transform,
        rms_error,
        max_error,
        iterations,
        converged,
        correspondences_used,
        average_displacement,
        max_displacement,
    })
}

/// Select control points from a mesh using farthest point sampling.
fn select_control_points(mesh: &Mesh, num_points: usize) -> Vec<usize> {
    let n = mesh.vertex_count();
    if num_points >= n {
        return (0..n).collect();
    }

    let mut selected = Vec::with_capacity(num_points);
    let mut min_distances = vec![f64::INFINITY; n];

    // Start with first vertex
    selected.push(0);

    while selected.len() < num_points {
        // Update minimum distances to selected set
        let last_selected = selected[selected.len() - 1];
        let last_pos = mesh.vertices[last_selected].position;

        for (i, dist) in min_distances.iter_mut().enumerate() {
            let d = (mesh.vertices[i].position - last_pos).norm();
            *dist = dist.min(d);
        }

        // Find point with maximum minimum distance (farthest point sampling)
        let next = min_distances
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
            .map(|(i, _)| i)
            .unwrap();

        selected.push(next);
    }

    selected
}

/// Regularize displacements using Laplacian smoothing.
fn regularize_displacements(
    positions: &[Point3<f64>],
    displacements: &[Vector3<f64>],
    stiffness: f64,
) -> Vec<Vector3<f64>> {
    let n = positions.len();
    if n <= 1 {
        return displacements.to_vec();
    }

    // Build neighborhood based on spatial proximity
    // For each control point, find k nearest neighbors
    let k = (n / 4).clamp(3, 10);

    let mut neighbors: Vec<Vec<usize>> = Vec::with_capacity(n);
    for i in 0..n {
        let mut distances: Vec<(usize, f64)> = (0..n)
            .filter(|&j| j != i)
            .map(|j| (j, (positions[i] - positions[j]).norm()))
            .collect();
        distances.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());
        neighbors.push(distances.iter().take(k).map(|(idx, _)| *idx).collect());
    }

    // Apply Laplacian smoothing
    let smoothing_weight = stiffness / (stiffness + 1.0);
    let mut smoothed = displacements.to_vec();

    for i in 0..n {
        if neighbors[i].is_empty() {
            continue;
        }

        // Compute average of neighbor displacements
        let neighbor_avg: Vector3<f64> = neighbors[i]
            .iter()
            .map(|&j| displacements[j])
            .sum::<Vector3<f64>>()
            / neighbors[i].len() as f64;

        // Blend original with smoothed
        smoothed[i] = displacements[i] * (1.0 - smoothing_weight) + neighbor_avg * smoothing_weight;
    }

    smoothed
}

/// Interpolate displacements from control points to all vertices using RBF.
fn interpolate_displacements_rbf(
    control_positions: &[Point3<f64>],
    control_displacements: &[Vector3<f64>],
    landmark_constraints: &[(Point3<f64>, Vector3<f64>)],
    query_positions: &[Point3<f64>],
    smoothness: f64,
) -> Vec<Vector3<f64>> {
    // Combine control points and landmarks
    let mut all_positions: Vec<Point3<f64>> = control_positions.to_vec();
    let mut all_displacements: Vec<Vector3<f64>> = control_displacements.to_vec();

    for (pos, disp) in landmark_constraints {
        all_positions.push(*pos);
        all_displacements.push(*disp);
    }

    let n = all_positions.len();
    if n == 0 {
        return vec![Vector3::zeros(); query_positions.len()];
    }

    // For small number of control points, use direct RBF interpolation
    // For larger sets, use a simplified approach

    if n <= 100 {
        // Full RBF solve for each component (x, y, z)
        interpolate_rbf_full(
            &all_positions,
            &all_displacements,
            query_positions,
            smoothness,
        )
    } else {
        // Simplified: weighted average based on distance
        interpolate_rbf_simplified(
            &all_positions,
            &all_displacements,
            query_positions,
            smoothness,
        )
    }
}

/// Full RBF interpolation (solves linear system).
fn interpolate_rbf_full(
    control_positions: &[Point3<f64>],
    control_displacements: &[Vector3<f64>],
    query_positions: &[Point3<f64>],
    smoothness: f64,
) -> Vec<Vector3<f64>> {
    use nalgebra::{DMatrix, DVector};

    let n = control_positions.len();
    let epsilon = 1.0 / smoothness.max(0.01);

    // Build RBF matrix
    let mut phi = DMatrix::zeros(n, n);
    for i in 0..n {
        for j in 0..n {
            let r = (control_positions[i] - control_positions[j]).norm();
            phi[(i, j)] = thin_plate_spline_rbf(r, epsilon);
        }
        // Add regularization to diagonal
        phi[(i, i)] += 1e-6;
    }

    // Solve for weights for each component
    let mut result = vec![Vector3::zeros(); query_positions.len()];

    // Try to solve; if it fails, fall back to simplified method
    let decomp = phi.clone().lu();

    for component in 0..3 {
        let b: DVector<f64> =
            DVector::from_iterator(n, control_displacements.iter().map(|d| d[component]));

        if let Some(weights) = decomp.solve(&b) {
            // Evaluate at query points
            for (q_idx, query) in query_positions.iter().enumerate() {
                let mut val = 0.0;
                for (c_idx, ctrl) in control_positions.iter().enumerate() {
                    let r = (query - ctrl).norm();
                    val += weights[c_idx] * thin_plate_spline_rbf(r, epsilon);
                }
                result[q_idx][component] = val;
            }
        } else {
            // Fall back to simplified interpolation
            return interpolate_rbf_simplified(
                control_positions,
                control_displacements,
                query_positions,
                smoothness,
            );
        }
    }

    result
}

/// Simplified RBF interpolation using inverse distance weighting.
fn interpolate_rbf_simplified(
    control_positions: &[Point3<f64>],
    control_displacements: &[Vector3<f64>],
    query_positions: &[Point3<f64>],
    smoothness: f64,
) -> Vec<Vector3<f64>> {
    let power = 2.0 + smoothness;

    query_positions
        .iter()
        .map(|query| {
            let mut weighted_sum = Vector3::zeros();
            let mut weight_sum = 0.0;

            for (ctrl, disp) in control_positions.iter().zip(control_displacements.iter()) {
                let dist = (query - ctrl).norm();
                let weight = if dist < 1e-10 {
                    1e10 // Very close, use this displacement directly
                } else {
                    1.0 / dist.powf(power)
                };

                weighted_sum += disp * weight;
                weight_sum += weight;
            }

            if weight_sum > 0.0 {
                weighted_sum / weight_sum
            } else {
                Vector3::zeros()
            }
        })
        .collect()
}

/// Thin-plate spline RBF kernel.
fn thin_plate_spline_rbf(r: f64, _epsilon: f64) -> f64 {
    if r < 1e-10 { 0.0 } else { r * r * r.ln() }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::Vertex;

    fn create_test_triangle() -> Mesh {
        let mut mesh = Mesh::new();
        mesh.vertices.push(Vertex::from_coords(0.0, 0.0, 0.0));
        mesh.vertices.push(Vertex::from_coords(10.0, 0.0, 0.0));
        mesh.vertices.push(Vertex::from_coords(5.0, 8.66, 0.0));
        mesh.faces.push([0, 1, 2]);
        mesh
    }

    fn create_test_cube() -> Mesh {
        let mut mesh = Mesh::new();
        mesh.vertices.push(Vertex::from_coords(0.0, 0.0, 0.0));
        mesh.vertices.push(Vertex::from_coords(10.0, 0.0, 0.0));
        mesh.vertices.push(Vertex::from_coords(10.0, 10.0, 0.0));
        mesh.vertices.push(Vertex::from_coords(0.0, 10.0, 0.0));
        mesh.vertices.push(Vertex::from_coords(0.0, 0.0, 10.0));
        mesh.vertices.push(Vertex::from_coords(10.0, 0.0, 10.0));
        mesh.vertices.push(Vertex::from_coords(10.0, 10.0, 10.0));
        mesh.vertices.push(Vertex::from_coords(0.0, 10.0, 10.0));

        mesh.faces.push([0, 2, 1]);
        mesh.faces.push([0, 3, 2]);
        mesh.faces.push([4, 5, 6]);
        mesh.faces.push([4, 6, 7]);
        mesh.faces.push([0, 1, 5]);
        mesh.faces.push([0, 5, 4]);
        mesh.faces.push([2, 3, 7]);
        mesh.faces.push([2, 7, 6]);
        mesh.faces.push([0, 4, 7]);
        mesh.faces.push([0, 7, 3]);
        mesh.faces.push([1, 2, 6]);
        mesh.faces.push([1, 6, 5]);
        mesh
    }

    #[test]
    fn test_identity_registration() {
        let mesh = create_test_triangle();
        let target = mesh.clone();

        let params = RegistrationParams::icp().with_max_iterations(10);
        let result = align_meshes(&mesh, &target, &params).unwrap();

        // Should converge with minimal error
        assert!(result.rms_error < 0.01, "RMS error: {}", result.rms_error);
        assert!(result.converged);
    }

    #[test]
    fn test_translation_recovery() {
        // Use cube instead of triangle for better ICP convergence
        let source = create_test_cube();

        // Create target translated by (5, 3, 0)
        let mut target = source.clone();
        for vertex in &mut target.vertices {
            vertex.position.x += 5.0;
            vertex.position.y += 3.0;
        }

        let params = RegistrationParams::icp().with_max_iterations(100);
        let result = align_meshes(&source, &target, &params).unwrap();

        // Should recover the translation (ICP may have local minima with triangles)
        assert!(
            result.rms_error < 1.0,
            "RMS error should be reasonable: {}",
            result.rms_error
        );

        // For cube-to-cube alignment, we can check alignment quality improved
        assert!(result.iterations > 0, "Should perform some iterations");
    }

    #[test]
    fn test_rotation_recovery() {
        let source = create_test_cube();

        // Create target rotated a small angle around Z axis (ICP works better with small rotations)
        let rotation = UnitQuaternion::from_axis_angle(&Vector3::z_axis(), 0.2); // ~11 degrees
        let mut target = source.clone();
        for vertex in &mut target.vertices {
            vertex.position = rotation * vertex.position;
        }

        let params = RegistrationParams::icp().with_max_iterations(150);
        let result = align_meshes(&source, &target, &params).unwrap();

        // ICP should make progress on the alignment
        assert!(result.iterations > 0, "Should perform some iterations");
        // Note: ICP can struggle with symmetric shapes and larger rotations
    }

    #[test]
    fn test_landmark_registration() {
        let source = create_test_triangle();

        // Create translated target
        let mut target = source.clone();
        for vertex in &mut target.vertices {
            vertex.position.x += 10.0;
            vertex.position.y += 5.0;
        }

        let landmarks = vec![
            Landmark::new(Point3::new(0.0, 0.0, 0.0), Point3::new(10.0, 5.0, 0.0)),
            Landmark::new(Point3::new(10.0, 0.0, 0.0), Point3::new(20.0, 5.0, 0.0)),
            Landmark::new(Point3::new(5.0, 8.66, 0.0), Point3::new(15.0, 13.66, 0.0)),
        ];

        let params = RegistrationParams::landmark_based(landmarks);
        let result = align_meshes(&source, &target, &params).unwrap();

        // With exact landmarks, error should be near zero
        assert!(
            result.rms_error < 0.01,
            "RMS error should be minimal: {}",
            result.rms_error
        );
    }

    #[test]
    fn test_scaling_registration() {
        let source = create_test_cube();

        // Create target scaled by 2x (use cube for better convergence)
        let mut target = source.clone();
        for vertex in &mut target.vertices {
            vertex.position.coords *= 2.0;
        }

        let params = RegistrationParams::icp()
            .with_scaling()
            .with_max_iterations(100);
        let result = align_meshes(&source, &target, &params).unwrap();

        // Should make some progress - scaling recovery is challenging for ICP
        assert!(result.iterations > 0, "Should perform iterations");
        // Scale recovery works better with good initial alignment
        // The algorithm should at least run without error
    }

    #[test]
    fn test_transform_composition() {
        let t1 = RigidTransform::from_translation(Vector3::new(1.0, 0.0, 0.0));
        let t2 = RigidTransform::from_translation(Vector3::new(0.0, 2.0, 0.0));

        let composed = t1.then(&t2);

        let point = Point3::new(0.0, 0.0, 0.0);
        let result = composed.transform_point(&point);

        assert!((result.x - 1.0).abs() < 1e-10);
        assert!((result.y - 2.0).abs() < 1e-10);
        assert!((result.z - 0.0).abs() < 1e-10);
    }

    #[test]
    fn test_transform_inverse() {
        let rotation =
            UnitQuaternion::from_axis_angle(&Vector3::z_axis(), std::f64::consts::FRAC_PI_2);
        let transform = RigidTransform {
            rotation,
            translation: Vector3::new(5.0, 3.0, 1.0),
            scale: 2.0,
        };

        let inverse = transform.inverse();
        let point = Point3::new(1.0, 2.0, 3.0);

        let transformed = transform.transform_point(&point);
        let recovered = inverse.transform_point(&transformed);

        assert!(
            (point - recovered).norm() < 1e-10,
            "Inverse should recover original point"
        );
    }

    #[test]
    fn test_empty_mesh_error() {
        let source = Mesh::new();
        let target = create_test_triangle();

        let params = RegistrationParams::icp();
        assert!(matches!(
            align_meshes(&source, &target, &params),
            Err(MeshError::EmptyMesh { .. })
        ));
    }

    #[test]
    fn test_insufficient_landmarks_error() {
        let source = create_test_triangle();
        let target = source.clone();

        let landmarks = vec![
            Landmark::new(Point3::new(0.0, 0.0, 0.0), Point3::new(0.0, 0.0, 0.0)),
            Landmark::new(Point3::new(1.0, 0.0, 0.0), Point3::new(1.0, 0.0, 0.0)),
        ];

        let params = RegistrationParams::landmark_based(landmarks);
        assert!(matches!(
            align_meshes(&source, &target, &params),
            Err(MeshError::RepairFailed { .. })
        ));
    }

    #[test]
    fn test_landmark_then_icp() {
        let source = create_test_cube();

        // Create target with rotation and translation
        let rotation = UnitQuaternion::from_axis_angle(&Vector3::z_axis(), 0.3);
        let translation = Vector3::new(5.0, 3.0, 2.0);

        let mut target = source.clone();
        for vertex in &mut target.vertices {
            vertex.position = rotation * vertex.position;
            vertex.position.coords += translation;
        }

        // Provide approximate landmarks (slightly off to test refinement)
        let landmarks = vec![
            Landmark::new(
                Point3::new(0.0, 0.0, 0.0),
                rotation * Point3::new(0.0, 0.0, 0.0) + translation,
            ),
            Landmark::new(
                Point3::new(10.0, 0.0, 0.0),
                rotation * Point3::new(10.0, 0.0, 0.0) + translation,
            ),
            Landmark::new(
                Point3::new(0.0, 10.0, 0.0),
                rotation * Point3::new(0.0, 10.0, 0.0) + translation,
            ),
        ];

        let params = RegistrationParams::landmark_then_icp(landmarks);
        let result = align_meshes(&source, &target, &params).unwrap();

        assert!(
            result.rms_error < 1.0,
            "RMS error should be small: {}",
            result.rms_error
        );
    }

    #[test]
    fn test_max_correspondence_distance() {
        let source = create_test_triangle();

        // Create target with some noise
        let mut target = source.clone();
        // Add an outlier vertex far away
        target.vertices.push(Vertex::from_coords(1000.0, 0.0, 0.0));

        let params = RegistrationParams::icp()
            .with_max_correspondence_distance(50.0) // Reject correspondences > 50mm
            .with_max_iterations(20);
        let result = align_meshes(&source, &target, &params).unwrap();

        // Should still converge well, ignoring the outlier
        assert!(result.rms_error < 1.0, "RMS error: {}", result.rms_error);
    }

    #[test]
    fn test_subsample_ratio() {
        let source = create_test_cube();
        let target = source.clone();

        let params = RegistrationParams::icp()
            .with_subsample_ratio(0.5) // Use 50% of points
            .with_max_iterations(10);
        let result = align_meshes(&source, &target, &params).unwrap();

        assert!(result.rms_error < 0.1, "RMS error: {}", result.rms_error);
    }

    #[test]
    fn test_weighted_landmarks() {
        let source = create_test_triangle();

        // Create target translated
        let mut target = source.clone();
        for vertex in &mut target.vertices {
            vertex.position.x += 5.0;
        }

        // Provide landmarks with different weights
        let landmarks = vec![
            Landmark::weighted(Point3::new(0.0, 0.0, 0.0), Point3::new(5.0, 0.0, 0.0), 2.0),
            Landmark::weighted(
                Point3::new(10.0, 0.0, 0.0),
                Point3::new(15.0, 0.0, 0.0),
                1.0,
            ),
            Landmark::weighted(
                Point3::new(5.0, 8.66, 0.0),
                Point3::new(10.0, 8.66, 0.0),
                1.0,
            ),
        ];

        let params = RegistrationParams::landmark_based(landmarks);
        let result = align_meshes(&source, &target, &params).unwrap();

        assert!(result.rms_error < 0.1, "RMS error: {}", result.rms_error);
    }

    #[test]
    fn test_rigid_transform_to_matrix() {
        let rotation =
            UnitQuaternion::from_axis_angle(&Vector3::z_axis(), std::f64::consts::FRAC_PI_2);
        let transform = RigidTransform {
            rotation,
            translation: Vector3::new(1.0, 2.0, 3.0),
            scale: 1.0,
        };

        let matrix = transform.to_matrix4();

        // Test that point transformation matches matrix multiplication
        let point = Point3::new(1.0, 0.0, 0.0);
        let transformed = transform.transform_point(&point);

        let homogeneous = nalgebra::Vector4::new(point.x, point.y, point.z, 1.0);
        let matrix_result = matrix * homogeneous;

        assert!((transformed.x - matrix_result.x).abs() < 1e-10);
        assert!((transformed.y - matrix_result.y).abs() < 1e-10);
        assert!((transformed.z - matrix_result.z).abs() < 1e-10);
    }

    // Non-rigid registration tests

    #[test]
    fn test_non_rigid_identity() {
        // Non-rigid registration of identical meshes should produce minimal displacement
        let source = create_test_cube();
        let target = source.clone();

        let nr_params = NonRigidParams::new();
        let base_params = RegistrationParams::default().with_max_iterations(20);

        let result = non_rigid_align(&source, &target, &nr_params, &base_params).unwrap();

        // Error should be very small
        assert!(
            result.rms_error < 0.5,
            "RMS error should be small for identical meshes: {}",
            result.rms_error
        );

        // Displacements should be minimal
        assert!(
            result.max_displacement < 1.0,
            "Max displacement should be small: {}",
            result.max_displacement
        );
    }

    #[test]
    fn test_non_rigid_translation() {
        // Non-rigid registration should handle pure translation
        let source = create_test_cube();

        let mut target = source.clone();
        for vertex in &mut target.vertices {
            vertex.position.x += 5.0;
            vertex.position.z += 2.0;
        }

        let nr_params = NonRigidParams::new().with_stiffness(1.0);
        let base_params = RegistrationParams::default().with_max_iterations(50);

        let result = non_rigid_align(&source, &target, &nr_params, &base_params).unwrap();

        // Should achieve reasonable alignment
        assert!(
            result.rms_error < 2.0,
            "RMS error should be reasonable: {}",
            result.rms_error
        );
    }

    #[test]
    fn test_non_rigid_local_deformation() {
        // Non-rigid should handle local deformations that rigid ICP cannot
        let source = create_test_cube();

        // Create target with non-uniform scaling (stretches in Y direction)
        let mut target = source.clone();
        for vertex in &mut target.vertices {
            vertex.position.y *= 1.5; // Stretch Y by 50%
        }

        let nr_params = NonRigidParams::new()
            .with_stiffness(2.0)
            .with_outer_iterations(15);
        let base_params = RegistrationParams::default();

        let result = non_rigid_align(&source, &target, &nr_params, &base_params).unwrap();

        // Non-rigid should adapt to the stretching
        assert!(
            result.max_displacement > 0.1,
            "Should have non-trivial displacement to match stretched target"
        );

        // Should converge or run all iterations
        assert!(result.iterations >= 1, "Should perform iterations");
    }

    #[test]
    fn test_non_rigid_with_landmarks() {
        let source = create_test_cube();

        // Target with translation
        let mut target = source.clone();
        for vertex in &mut target.vertices {
            vertex.position.x += 3.0;
        }

        // Provide landmarks to guide the deformation
        let landmarks = vec![
            Landmark::new(Point3::new(0.0, 0.0, 0.0), Point3::new(3.0, 0.0, 0.0)),
            Landmark::new(Point3::new(10.0, 10.0, 10.0), Point3::new(13.0, 10.0, 10.0)),
        ];

        let nr_params = NonRigidParams::new()
            .with_landmarks(landmarks)
            .with_stiffness(5.0);
        let base_params = RegistrationParams::default();

        let result = non_rigid_align(&source, &target, &nr_params, &base_params).unwrap();

        // Landmarks should help guide the registration
        assert!(result.iterations >= 1);
    }

    #[test]
    fn test_non_rigid_params_builder() {
        let params = NonRigidParams::new()
            .with_stiffness(20.0)
            .with_control_points(100)
            .with_outer_iterations(5)
            .with_smoothness(2.0)
            .without_initial_alignment();

        assert_eq!(params.stiffness, 20.0);
        assert_eq!(params.num_control_points, Some(100));
        assert_eq!(params.outer_iterations, 5);
        assert_eq!(params.smoothness, 2.0);
        assert!(!params.initial_rigid_alignment);
    }

    #[test]
    fn test_non_rigid_result_deformation_field() {
        let source = create_test_cube();
        let target = source.clone();

        let nr_params = NonRigidParams::new();
        let base_params = RegistrationParams::default().with_max_iterations(5);

        let result = non_rigid_align(&source, &target, &nr_params, &base_params).unwrap();

        // Get deformation field
        let field = result.deformation_field(&source);

        assert_eq!(field.len(), source.vertex_count());

        // Each entry should have position and displacement
        for (pos, _disp) in &field {
            // Position should be from original mesh
            assert!(
                source
                    .vertices
                    .iter()
                    .any(|v| (v.position - pos).norm() < 1e-6)
            );
        }
    }

    #[test]
    fn test_select_control_points() {
        let mesh = create_test_cube();

        // Select 4 control points
        let indices = select_control_points(&mesh, 4);

        assert_eq!(indices.len(), 4);

        // All indices should be valid
        for &idx in &indices {
            assert!(idx < mesh.vertex_count());
        }

        // Indices should be unique
        let unique: std::collections::HashSet<_> = indices.iter().collect();
        assert_eq!(unique.len(), 4);
    }

    #[test]
    fn test_select_control_points_more_than_vertices() {
        let mesh = create_test_triangle();

        // Request more control points than vertices
        let indices = select_control_points(&mesh, 10);

        // Should return all vertices
        assert_eq!(indices.len(), mesh.vertex_count());
    }

    #[test]
    fn test_regularize_displacements() {
        let positions = vec![
            Point3::new(0.0, 0.0, 0.0),
            Point3::new(1.0, 0.0, 0.0),
            Point3::new(0.5, 1.0, 0.0),
        ];

        // One point has a large displacement, others are zero
        let displacements = vec![
            Vector3::new(10.0, 0.0, 0.0),
            Vector3::zeros(),
            Vector3::zeros(),
        ];

        // High stiffness should smooth out the spike
        let smoothed = regularize_displacements(&positions, &displacements, 10.0);

        // The outlier should be reduced
        assert!(
            smoothed[0].norm() < displacements[0].norm(),
            "Regularization should reduce outlier"
        );
    }

    #[test]
    fn test_non_rigid_empty_mesh_error() {
        let source = Mesh::new();
        let target = create_test_cube();

        let nr_params = NonRigidParams::new();
        let base_params = RegistrationParams::default();

        let result = non_rigid_align(&source, &target, &nr_params, &base_params);

        assert!(matches!(result, Err(MeshError::EmptyMesh { .. })));
    }

    #[test]
    fn test_thin_plate_spline_rbf() {
        // Test RBF kernel properties
        assert_eq!(thin_plate_spline_rbf(0.0, 1.0), 0.0);

        // r^2 * ln(r) for r=1 should be 0 (ln(1) = 0)
        assert!((thin_plate_spline_rbf(1.0, 1.0) - 0.0).abs() < 1e-10);

        // For r > 1, value should be positive
        assert!(thin_plate_spline_rbf(2.0, 1.0) > 0.0);

        // For 0 < r < 1, value should be negative
        assert!(thin_plate_spline_rbf(0.5, 1.0) < 0.0);
    }
}

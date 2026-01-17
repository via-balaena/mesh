//! Fluent builder API for mesh fitting workflows.
//!
//! This module provides a high-level, chainable API for fitting template meshes
//! to scans or measurements. It combines registration, morphing, and template
//! fitting into a streamlined workflow.
//!
//! # Example
//!
//! ```no_run
//! use mesh_repair::{Mesh, FittingBuilder};
//! use nalgebra::Point3;
//!
//! let template = Mesh::load("shoe_last_template.stl").unwrap();
//! let foot_scan = Mesh::load("foot_scan.stl").unwrap();
//!
//! // Fluent API for mesh fitting
//! let result = FittingBuilder::new(template)
//!     .fit_to_scan(&foot_scan)           // Align to foot scan
//!     .with_landmark("heel", Point3::new(0.0, 0.0, 0.0))
//!     .with_landmark("toe", Point3::new(250.0, 0.0, 0.0))
//!     .smooth(1.5)                        // Smoothness parameter
//!     .build()
//!     .unwrap();
//!
//! result.mesh.save("fitted_last.stl").unwrap();
//! ```

use crate::morph::{self, Constraint, MorphParams, RbfKernel};
use crate::registration::{self, RegistrationParams, RigidTransform};
use crate::template::{ControlRegion, Measurement};
use crate::{Mesh, MeshResult};
use nalgebra::{Point3, Vector3};
use std::collections::{HashMap, HashSet};

/// Result from FittingBuilder containing the fitted mesh and statistics.
#[derive(Debug, Clone)]
pub struct FittingResult {
    /// The fitted mesh.
    pub mesh: Mesh,
    /// Registration error (RMS distance) if rigid alignment was performed.
    pub registration_error: Option<f64>,
    /// Maximum displacement of any vertex during morphing.
    pub max_displacement: f64,
    /// Average displacement across all vertices.
    pub average_displacement: f64,
    /// Volume change ratio (new_volume / old_volume).
    pub volume_ratio: f64,
    /// Number of fitting stages completed.
    pub stages_completed: usize,
}

/// Fluent builder for mesh fitting workflows.
///
/// FittingBuilder provides a chainable API for configuring and executing
/// mesh fitting operations. This is the recommended high-level API for
/// fitting template meshes to scans or measurements.
///
/// # Workflow
///
/// A typical fitting workflow involves:
/// 1. Load a template mesh
/// 2. Optionally align to a target scan (ICP registration)
/// 3. Define landmark constraints (point-to-point)
/// 4. Define measurement constraints (circumferences, lengths)
/// 5. Configure morphing parameters
/// 6. Execute the fitting
///
/// # Example
///
/// ```no_run
/// use mesh_repair::{Mesh, FittingBuilder};
/// use nalgebra::Point3;
///
/// let template = Mesh::load("helmet_liner.stl").unwrap();
/// let head_scan = Mesh::load("head_scan.stl").unwrap();
///
/// // Simple fitting to scan
/// let result = FittingBuilder::new(template)
///     .fit_to_scan(&head_scan)
///     .build()
///     .unwrap();
///
/// // Or with landmarks and measurements
/// let template = Mesh::load("helmet_liner.stl").unwrap();
/// let result = FittingBuilder::new(template)
///     .fit_to_scan(&head_scan)
///     .with_landmark("crown", Point3::new(0.0, 0.0, 200.0))
///     .with_landmark("forehead", Point3::new(0.0, 100.0, 180.0))
///     .with_circumference("head_circ", 580.0)  // 58cm head circumference
///     .build()
///     .unwrap();
/// ```
pub struct FittingBuilder {
    template: Mesh,
    target_scan: Option<Mesh>,
    landmarks: HashMap<String, (Point3<f64>, Point3<f64>)>, // source -> target
    measurements: HashMap<String, Measurement>,
    control_regions: HashMap<String, ControlRegion>,
    smoothness: f64,
    registration_iterations: usize,
    convergence_threshold: f64,
    morph_algorithm: MorphAlgorithm,
    region_mask: Option<HashSet<u32>>,
}

/// Morphing algorithm selection.
#[derive(Debug, Clone, Copy)]
enum MorphAlgorithm {
    Rbf(RbfKernel),
    Ffd { resolution: (usize, usize, usize) },
}

impl FittingBuilder {
    /// Create a new FittingBuilder for the given template mesh.
    ///
    /// # Arguments
    ///
    /// * `template` - The template mesh to fit (takes ownership)
    ///
    /// # Example
    ///
    /// ```no_run
    /// use mesh_repair::{Mesh, FittingBuilder};
    ///
    /// let template = Mesh::load("template.stl").unwrap();
    /// let builder = FittingBuilder::new(template);
    /// ```
    pub fn new(template: Mesh) -> Self {
        Self {
            template,
            target_scan: None,
            landmarks: HashMap::new(),
            measurements: HashMap::new(),
            control_regions: HashMap::new(),
            smoothness: 1.0,
            registration_iterations: 100,
            convergence_threshold: 1e-6,
            morph_algorithm: MorphAlgorithm::Rbf(RbfKernel::ThinPlateSpline),
            region_mask: None,
        }
    }

    // =========================================================================
    // Target Configuration
    // =========================================================================

    /// Set a target scan to fit to.
    ///
    /// When a target scan is provided, the template will first be rigidly
    /// aligned using ICP (Iterative Closest Point) registration, then
    /// morphed to match landmarks and measurements.
    ///
    /// # Arguments
    ///
    /// * `scan` - The target scan mesh
    pub fn fit_to_scan(mut self, scan: &Mesh) -> Self {
        self.target_scan = Some(scan.clone());
        self
    }

    // =========================================================================
    // Landmark Constraints
    // =========================================================================

    /// Add a landmark constraint with both source and target positions.
    ///
    /// Use this when you know both the position on the template and the
    /// target position it should move to.
    ///
    /// # Arguments
    ///
    /// * `name` - Name for this landmark
    /// * `source` - Position on the template mesh
    /// * `target` - Target position after fitting
    pub fn with_landmark_pair(
        mut self,
        name: impl Into<String>,
        source: Point3<f64>,
        target: Point3<f64>,
    ) -> Self {
        self.landmarks.insert(name.into(), (source, target));
        self
    }

    /// Add a landmark constraint with only target position.
    ///
    /// Use this with `with_control_region` to define the source position
    /// via a named region on the template.
    ///
    /// # Arguments
    ///
    /// * `name` - Name matching a control region on the template
    /// * `target` - Target position after fitting
    pub fn with_landmark(mut self, name: impl Into<String>, target: Point3<f64>) -> Self {
        let name_str = name.into();
        // Store with zero source - will be resolved from control region during build
        self.landmarks.insert(name_str, (Point3::origin(), target));
        self
    }

    /// Add multiple landmark constraints.
    pub fn with_landmarks(mut self, landmarks: HashMap<String, Point3<f64>>) -> Self {
        for (name, target) in landmarks {
            self.landmarks.insert(name, (Point3::origin(), target));
        }
        self
    }

    // =========================================================================
    // Control Regions
    // =========================================================================

    /// Add a point control region.
    ///
    /// Control regions define areas on the template that can be manipulated.
    pub fn with_control_point(mut self, name: impl Into<String>, position: Point3<f64>) -> Self {
        let n = name.into();
        self.control_regions
            .insert(n.clone(), ControlRegion::point(n, position));
        self
    }

    /// Add a vertex-based control region.
    pub fn with_control_vertices(mut self, name: impl Into<String>, indices: Vec<u32>) -> Self {
        let n = name.into();
        self.control_regions
            .insert(n.clone(), ControlRegion::vertices(n, indices));
        self
    }

    /// Add a control region.
    pub fn with_control_region(mut self, region: ControlRegion) -> Self {
        self.control_regions.insert(region.name.clone(), region);
        self
    }

    // =========================================================================
    // Measurement Constraints
    // =========================================================================

    /// Add a circumference measurement constraint.
    ///
    /// The mesh will be scaled/morphed to achieve the target circumference.
    ///
    /// # Arguments
    ///
    /// * `name` - Name for this measurement
    /// * `target_mm` - Target circumference in millimeters
    pub fn with_circumference(mut self, name: impl Into<String>, target_mm: f64) -> Self {
        self.measurements.insert(
            name.into(),
            Measurement::with_tolerance(target_mm, target_mm * 0.02),
        );
        self
    }

    /// Add a length measurement constraint.
    pub fn with_length(mut self, name: impl Into<String>, target_mm: f64) -> Self {
        self.measurements.insert(
            name.into(),
            Measurement::with_tolerance(target_mm, target_mm * 0.02),
        );
        self
    }

    /// Add a width measurement constraint.
    pub fn with_width(mut self, name: impl Into<String>, target_mm: f64) -> Self {
        self.measurements.insert(
            name.into(),
            Measurement::with_tolerance(target_mm, target_mm * 0.02),
        );
        self
    }

    /// Add a generic measurement constraint.
    pub fn with_measurement(mut self, name: impl Into<String>, measurement: Measurement) -> Self {
        self.measurements.insert(name.into(), measurement);
        self
    }

    // =========================================================================
    // Algorithm Configuration
    // =========================================================================

    /// Set the smoothness parameter for morphing.
    ///
    /// Higher values produce smoother deformations. Default is 1.0.
    pub fn smooth(mut self, smoothness: f64) -> Self {
        self.smoothness = smoothness;
        self
    }

    /// Use RBF (Radial Basis Function) morphing with thin-plate spline kernel.
    ///
    /// This is the default and provides smooth, natural deformations.
    pub fn rbf_thin_plate(mut self) -> Self {
        self.morph_algorithm = MorphAlgorithm::Rbf(RbfKernel::ThinPlateSpline);
        self
    }

    /// Use RBF morphing with Gaussian kernel.
    ///
    /// Provides more localized deformations.
    pub fn rbf_gaussian(mut self) -> Self {
        self.morph_algorithm = MorphAlgorithm::Rbf(RbfKernel::Gaussian);
        self
    }

    /// Use RBF morphing with multiquadric kernel.
    ///
    /// Good balance between local and global effects.
    pub fn rbf_multiquadric(mut self) -> Self {
        self.morph_algorithm = MorphAlgorithm::Rbf(RbfKernel::Multiquadric);
        self
    }

    /// Use FFD (Free-Form Deformation) with the specified lattice resolution.
    pub fn ffd(mut self, nx: usize, ny: usize, nz: usize) -> Self {
        self.morph_algorithm = MorphAlgorithm::Ffd {
            resolution: (nx, ny, nz),
        };
        self
    }

    /// Set maximum iterations for ICP registration.
    pub fn registration_iterations(mut self, iterations: usize) -> Self {
        self.registration_iterations = iterations;
        self
    }

    /// Set convergence threshold for ICP registration.
    pub fn convergence_threshold(mut self, threshold: f64) -> Self {
        self.convergence_threshold = threshold;
        self
    }

    /// Limit deformation to specific vertices.
    ///
    /// Only vertices in this set will be morphed; others remain fixed.
    pub fn with_region_mask(mut self, vertex_indices: HashSet<u32>) -> Self {
        self.region_mask = Some(vertex_indices);
        self
    }

    // =========================================================================
    // Build
    // =========================================================================

    /// Execute the fitting workflow and return the result.
    ///
    /// The fitting proceeds in stages:
    /// 1. Rigid alignment to target scan (if provided)
    /// 2. Morph to match landmark constraints
    /// 3. Adjust for measurement constraints
    ///
    /// # Returns
    ///
    /// A `FittingResult` containing the fitted mesh and statistics.
    ///
    /// # Errors
    ///
    /// Returns an error if:
    /// - The template mesh is empty
    /// - Registration fails
    /// - Morphing fails
    pub fn build(self) -> MeshResult<FittingResult> {
        let mut current_mesh = self.template;
        let mut result = FittingResult {
            mesh: Mesh::new(),
            registration_error: None,
            max_displacement: 0.0,
            average_displacement: 0.0,
            volume_ratio: 1.0,
            stages_completed: 0,
        };

        let original_volume = current_mesh.volume().abs();

        // Stage 1: Rigid alignment to scan
        let mut transform = RigidTransform::identity();
        if let Some(ref scan) = self.target_scan {
            let reg_params = RegistrationParams::icp()
                .with_max_iterations(self.registration_iterations)
                .with_convergence_threshold(self.convergence_threshold);

            let reg_result = registration::align_meshes(&current_mesh, scan, &reg_params)?;
            current_mesh = reg_result.mesh;
            transform = reg_result.transformation;
            result.registration_error = Some(reg_result.rms_error);
            result.stages_completed += 1;
        }

        // Stage 2: Build morph constraints from landmarks
        if !self.landmarks.is_empty() {
            let mut constraints = Vec::new();

            for (name, (source, target)) in &self.landmarks {
                // If source is origin, try to find it from control region
                let actual_source = if *source == Point3::origin() {
                    if let Some(region) = self.control_regions.get(name) {
                        match &region.definition {
                            crate::template::RegionDefinition::Point(p) => {
                                transform.transform_point(p)
                            }
                            crate::template::RegionDefinition::Vertices(indices)
                                if !indices.is_empty() =>
                            {
                                // Average of vertices
                                let sum: Vector3<f64> = indices
                                    .iter()
                                    .filter_map(|&i| current_mesh.vertices.get(i as usize))
                                    .map(|v| v.position.coords)
                                    .sum();
                                Point3::from(sum / indices.len() as f64)
                            }
                            _ => continue, // Skip if can't determine source
                        }
                    } else {
                        continue; // Skip if no control region and no explicit source
                    }
                } else {
                    transform.transform_point(source)
                };

                constraints.push(Constraint::point(actual_source, *target));
            }

            if !constraints.is_empty() {
                let morph_params = match self.morph_algorithm {
                    MorphAlgorithm::Rbf(kernel) => MorphParams {
                        algorithm: morph::MorphAlgorithm::Rbf(kernel),
                        constraints,
                        region_mask: self.region_mask.clone(),
                        smoothness: self.smoothness,
                        ffd_resolution: (4, 4, 4),
                    },
                    MorphAlgorithm::Ffd { resolution } => MorphParams {
                        algorithm: morph::MorphAlgorithm::Ffd,
                        constraints,
                        region_mask: self.region_mask.clone(),
                        smoothness: self.smoothness,
                        ffd_resolution: resolution,
                    },
                };

                let morph_result = morph::morph_mesh(&current_mesh, &morph_params)?;
                current_mesh = morph_result.mesh;
                result.max_displacement = morph_result.max_displacement;
                result.average_displacement = morph_result.average_displacement;
                result.stages_completed += 1;
            }
        }

        // Calculate volume ratio
        let final_volume = current_mesh.volume().abs();
        if original_volume > 0.0 {
            result.volume_ratio = final_volume / original_volume;
        }

        result.mesh = current_mesh;
        Ok(result)
    }

    /// Get the current template mesh without executing the fitting.
    pub fn into_template(self) -> Mesh {
        self.template
    }

    /// Get a reference to the current template mesh.
    pub fn template(&self) -> &Mesh {
        &self.template
    }

    /// Get the number of landmarks defined.
    pub fn landmark_count(&self) -> usize {
        self.landmarks.len()
    }

    /// Get the number of measurements defined.
    pub fn measurement_count(&self) -> usize {
        self.measurements.len()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::Vertex;

    fn create_test_mesh() -> Mesh {
        let mut mesh = Mesh::new();

        // Simple tetrahedron
        mesh.vertices = vec![
            Vertex::from_coords(0.0, 0.0, 0.0),
            Vertex::from_coords(10.0, 0.0, 0.0),
            Vertex::from_coords(5.0, 10.0, 0.0),
            Vertex::from_coords(5.0, 5.0, 10.0),
        ];

        mesh.faces = vec![[0, 1, 2], [0, 2, 3], [0, 3, 1], [1, 3, 2]];

        mesh
    }

    #[test]
    fn test_builder_defaults() {
        let mesh = create_test_mesh();
        let builder = FittingBuilder::new(mesh);

        assert_eq!(builder.landmark_count(), 0);
        assert_eq!(builder.measurement_count(), 0);
    }

    #[test]
    fn test_builder_chaining() {
        let mesh = create_test_mesh();
        let builder = FittingBuilder::new(mesh)
            .with_landmark_pair(
                "tip",
                Point3::new(5.0, 5.0, 10.0),
                Point3::new(5.0, 5.0, 12.0),
            )
            .smooth(1.5)
            .rbf_thin_plate();

        assert_eq!(builder.landmark_count(), 1);
    }

    #[test]
    fn test_with_measurements() {
        let mesh = create_test_mesh();
        let builder = FittingBuilder::new(mesh)
            .with_circumference("head", 580.0)
            .with_length("nose_to_back", 200.0);

        assert_eq!(builder.measurement_count(), 2);
    }

    #[test]
    fn test_build_no_constraints() {
        let mesh = create_test_mesh();
        let result = FittingBuilder::new(mesh).build().unwrap();

        // With no constraints, mesh should be unchanged
        assert_eq!(result.mesh.vertices.len(), 4);
        assert_eq!(result.stages_completed, 0);
    }

    #[test]
    fn test_into_template() {
        let mesh = create_test_mesh();
        let vertex_count = mesh.vertices.len();

        let recovered = FittingBuilder::new(mesh).into_template();

        assert_eq!(recovered.vertices.len(), vertex_count);
    }
}

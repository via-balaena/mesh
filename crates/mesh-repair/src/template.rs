//! Template-based mesh fitting.
//!
//! This module provides tools for fitting template meshes to scans or measurements,
//! enabling parametric customization of product designs.
//!
//! # Use Cases
//!
//! - Fitting a shoe last template to a foot scan
//! - Adapting a helmet liner to head measurements
//! - Creating size variations of a product template
//!
//! # Example
//!
//! ```
//! use mesh_repair::{Mesh, Vertex};
//! use mesh_repair::template::{FitTemplate, FitParams, ControlRegion};
//! use nalgebra::Point3;
//!
//! // Create a template mesh
//! let mut template_mesh = Mesh::new();
//! template_mesh.vertices.push(Vertex::from_coords(0.0, 0.0, 0.0));
//! template_mesh.vertices.push(Vertex::from_coords(10.0, 0.0, 0.0));
//! template_mesh.vertices.push(Vertex::from_coords(5.0, 10.0, 0.0));
//! template_mesh.faces.push([0, 1, 2]);
//!
//! // Create template with control regions
//! let template = FitTemplate::new(template_mesh)
//!     .with_control_region(ControlRegion::point("tip", Point3::new(5.0, 10.0, 0.0)));
//!
//! // Fit to target measurements
//! let params = FitParams::default()
//!     .with_landmark_target("tip", Point3::new(5.0, 12.0, 0.0));
//!
//! let result = template.fit(&params).unwrap();
//! println!("Fit error: {:.3} mm", result.fit_error);
//! ```

use crate::morph::{self, Constraint, MorphParams};
use crate::registration::{self, RegistrationParams, RigidTransform};
use crate::{Mesh, MeshError, MeshResult};
use nalgebra::{Point3, Vector3};
use std::collections::{HashMap, HashSet};

/// A template mesh with control regions for parametric fitting.
#[derive(Debug, Clone)]
pub struct FitTemplate {
    /// The base template mesh.
    pub mesh: Mesh,

    /// Named control regions for manipulation.
    pub control_regions: HashMap<String, ControlRegion>,

    /// Default fitting parameters.
    pub default_params: FitParams,
}

impl FitTemplate {
    /// Create a new template from a mesh.
    pub fn new(mesh: Mesh) -> Self {
        Self {
            mesh,
            control_regions: HashMap::new(),
            default_params: FitParams::default(),
        }
    }

    /// Add a control region to the template.
    pub fn with_control_region(mut self, region: ControlRegion) -> Self {
        self.control_regions.insert(region.name.clone(), region);
        self
    }

    /// Add multiple control regions.
    pub fn with_control_regions(mut self, regions: Vec<ControlRegion>) -> Self {
        for region in regions {
            self.control_regions.insert(region.name.clone(), region);
        }
        self
    }

    /// Set default fitting parameters.
    pub fn with_default_params(mut self, params: FitParams) -> Self {
        self.default_params = params;
        self
    }

    /// Get a control region by name.
    pub fn get_region(&self, name: &str) -> Option<&ControlRegion> {
        self.control_regions.get(name)
    }

    /// Get the position of a landmark control region.
    pub fn get_landmark_position(&self, name: &str) -> Option<Point3<f64>> {
        self.control_regions
            .get(name)
            .and_then(|r| match &r.definition {
                RegionDefinition::Point(p) => Some(*p),
                RegionDefinition::Vertices(indices) if indices.len() == 1 => self
                    .mesh
                    .vertices
                    .get(indices[0] as usize)
                    .map(|v| v.position),
                _ => None,
            })
    }

    /// List all control region names.
    pub fn region_names(&self) -> Vec<&str> {
        self.control_regions.keys().map(|s| s.as_str()).collect()
    }

    /// Fit the template using the given parameters.
    ///
    /// This performs a multi-stage fitting process:
    /// 1. Rigid alignment (if a target scan is provided)
    /// 2. Landmark-based deformation
    /// 3. Measurement-based adjustment
    pub fn fit(&self, params: &FitParams) -> MeshResult<FitResult> {
        if self.mesh.is_empty() {
            return Err(MeshError::EmptyMesh {
                details: "Cannot fit an empty template mesh".to_string(),
            });
        }

        let mut current_mesh = self.mesh.clone();
        let mut total_transform = RigidTransform::identity();
        let mut stages_completed = Vec::new();

        // Stage 1: Rigid alignment to scan (if provided)
        if let Some(ref scan) = params.target_scan {
            let reg_params = RegistrationParams::icp()
                .with_max_iterations(params.registration_iterations)
                .with_convergence_threshold(params.convergence_threshold);

            let reg_result = registration::align_meshes(&current_mesh, scan, &reg_params)?;
            current_mesh = reg_result.mesh;
            total_transform = reg_result.transformation;
            stages_completed.push(FitStage::RigidAlignment {
                rms_error: reg_result.rms_error,
            });
        }

        // Stage 2: Landmark-based deformation
        if !params.landmark_targets.is_empty() {
            let mut constraints = Vec::new();

            for (name, target) in &params.landmark_targets {
                if let Some(region) = self.control_regions.get(name) {
                    // Get current position (after rigid alignment)
                    let source = match &region.definition {
                        RegionDefinition::Point(p) => total_transform.transform_point(p),
                        RegionDefinition::Vertices(indices) if !indices.is_empty() => {
                            // Average position of vertices
                            let sum: Vector3<f64> = indices
                                .iter()
                                .filter_map(|&i| current_mesh.vertices.get(i as usize))
                                .map(|v| v.position.coords)
                                .sum();
                            Point3::from(sum / indices.len() as f64)
                        }
                        _ => continue,
                    };

                    constraints.push(Constraint::weighted(source, *target, region.weight));
                }
            }

            if !constraints.is_empty() {
                let morph_params = MorphParams::rbf()
                    .with_constraints(constraints)
                    .with_smoothness(params.smoothness);

                let morph_result = morph::morph_mesh(&current_mesh, &morph_params)?;
                current_mesh = morph_result.mesh;
                stages_completed.push(FitStage::LandmarkDeformation {
                    constraints_applied: params.landmark_targets.len(),
                    max_displacement: morph_result.max_displacement,
                });
            }
        }

        // Stage 3: Measurement-based adjustment
        if !params.measurement_targets.is_empty() {
            for (name, measurement) in &params.measurement_targets {
                if let Some(region) = self.control_regions.get(name) {
                    current_mesh =
                        apply_measurement_constraint(&current_mesh, region, measurement)?;
                }
            }
            stages_completed.push(FitStage::MeasurementAdjustment {
                measurements_applied: params.measurement_targets.len(),
            });
        }

        // Calculate fit quality metrics
        let fit_error = calculate_fit_error(&current_mesh, params, &self.control_regions);

        Ok(FitResult {
            mesh: current_mesh,
            fit_error,
            stages: stages_completed,
            transform: total_transform,
        })
    }

    /// Fit the template to a target scan.
    ///
    /// This is a convenience method that combines registration and morphing.
    pub fn fit_to_scan(&self, scan: &Mesh) -> MeshResult<FitResult> {
        let params = FitParams::default().with_target_scan(scan.clone());
        self.fit(&params)
    }

    /// Fit the template to target measurements only.
    pub fn fit_to_measurements(
        &self,
        measurements: HashMap<String, Measurement>,
    ) -> MeshResult<FitResult> {
        let params = FitParams::default().with_measurements(measurements);
        self.fit(&params)
    }
}

/// A control region on a template mesh.
#[derive(Debug, Clone)]
pub struct ControlRegion {
    /// Unique name for this region (e.g., "heel", "toe_tip", "ankle").
    pub name: String,

    /// How this region is defined.
    pub definition: RegionDefinition,

    /// Weight for this region in fitting operations.
    pub weight: f64,

    /// Whether this region should be preserved (not deformed).
    pub preserve: bool,
}

impl ControlRegion {
    /// Create a point-based control region (single landmark).
    pub fn point(name: impl Into<String>, position: Point3<f64>) -> Self {
        Self {
            name: name.into(),
            definition: RegionDefinition::Point(position),
            weight: 1.0,
            preserve: false,
        }
    }

    /// Create a vertex-based control region.
    pub fn vertices(name: impl Into<String>, indices: Vec<u32>) -> Self {
        Self {
            name: name.into(),
            definition: RegionDefinition::Vertices(indices),
            weight: 1.0,
            preserve: false,
        }
    }

    /// Create a face-based control region.
    pub fn faces(name: impl Into<String>, indices: Vec<u32>) -> Self {
        Self {
            name: name.into(),
            definition: RegionDefinition::Faces(indices),
            weight: 1.0,
            preserve: false,
        }
    }

    /// Create a spatial bounds region (box).
    pub fn bounds(name: impl Into<String>, min: Point3<f64>, max: Point3<f64>) -> Self {
        Self {
            name: name.into(),
            definition: RegionDefinition::Bounds { min, max },
            weight: 1.0,
            preserve: false,
        }
    }

    /// Create a spherical region.
    pub fn sphere(name: impl Into<String>, center: Point3<f64>, radius: f64) -> Self {
        Self {
            name: name.into(),
            definition: RegionDefinition::Sphere { center, radius },
            weight: 1.0,
            preserve: false,
        }
    }

    /// Create a cylindrical region.
    pub fn cylinder(
        name: impl Into<String>,
        axis_start: Point3<f64>,
        axis_end: Point3<f64>,
        radius: f64,
    ) -> Self {
        Self {
            name: name.into(),
            definition: RegionDefinition::Cylinder {
                axis_start,
                axis_end,
                radius,
            },
            weight: 1.0,
            preserve: false,
        }
    }

    /// Create a measurement region (for circumference, etc.).
    pub fn measurement(
        name: impl Into<String>,
        measurement_type: MeasurementType,
        plane_origin: Point3<f64>,
        plane_normal: Vector3<f64>,
    ) -> Self {
        Self {
            name: name.into(),
            definition: RegionDefinition::MeasurementPlane {
                measurement_type,
                origin: plane_origin,
                normal: plane_normal.normalize(),
            },
            weight: 1.0,
            preserve: false,
        }
    }

    /// Set the weight for this region.
    pub fn with_weight(mut self, weight: f64) -> Self {
        self.weight = weight;
        self
    }

    /// Mark this region as preserved (not deformed).
    pub fn preserved(mut self) -> Self {
        self.preserve = true;
        self
    }

    /// Get the vertex indices that belong to this region.
    pub fn get_vertex_indices(&self, mesh: &Mesh) -> HashSet<u32> {
        match &self.definition {
            RegionDefinition::Point(_) => HashSet::new(),
            RegionDefinition::Vertices(indices) => indices.iter().copied().collect(),
            RegionDefinition::Faces(face_indices) => {
                let mut vertices = HashSet::new();
                for &fi in face_indices {
                    if let Some(face) = mesh.faces.get(fi as usize) {
                        vertices.insert(face[0]);
                        vertices.insert(face[1]);
                        vertices.insert(face[2]);
                    }
                }
                vertices
            }
            RegionDefinition::Bounds { min, max } => mesh
                .vertices
                .iter()
                .enumerate()
                .filter(|(_, v)| {
                    v.position.x >= min.x
                        && v.position.x <= max.x
                        && v.position.y >= min.y
                        && v.position.y <= max.y
                        && v.position.z >= min.z
                        && v.position.z <= max.z
                })
                .map(|(i, _)| i as u32)
                .collect(),
            RegionDefinition::Sphere { center, radius } => mesh
                .vertices
                .iter()
                .enumerate()
                .filter(|(_, v)| (v.position - center).norm() <= *radius)
                .map(|(i, _)| i as u32)
                .collect(),
            RegionDefinition::Cylinder {
                axis_start,
                axis_end,
                radius,
            } => {
                let axis = axis_end - axis_start;
                let axis_len_sq = axis.norm_squared();
                if axis_len_sq < 1e-10 {
                    return HashSet::new();
                }

                mesh.vertices
                    .iter()
                    .enumerate()
                    .filter(|(_, v)| {
                        let to_point = v.position - axis_start;
                        let t = to_point.dot(&axis) / axis_len_sq;
                        if !(0.0..=1.0).contains(&t) {
                            return false;
                        }
                        let projection = axis_start + axis * t;
                        (v.position - projection).norm() <= *radius
                    })
                    .map(|(i, _)| i as u32)
                    .collect()
            }
            RegionDefinition::MeasurementPlane { origin, normal, .. } => {
                // Get vertices near the measurement plane
                let tolerance = 5.0; // mm
                mesh.vertices
                    .iter()
                    .enumerate()
                    .filter(|(_, v)| {
                        let dist = (v.position - origin).dot(normal).abs();
                        dist <= tolerance
                    })
                    .map(|(i, _)| i as u32)
                    .collect()
            }
        }
    }
}

/// How a control region is defined.
#[derive(Debug, Clone)]
pub enum RegionDefinition {
    /// A single point (landmark).
    Point(Point3<f64>),

    /// A set of vertex indices.
    Vertices(Vec<u32>),

    /// A set of face indices.
    Faces(Vec<u32>),

    /// An axis-aligned bounding box.
    Bounds { min: Point3<f64>, max: Point3<f64> },

    /// A sphere.
    Sphere { center: Point3<f64>, radius: f64 },

    /// A cylinder (for limbs, handles, etc.).
    Cylinder {
        axis_start: Point3<f64>,
        axis_end: Point3<f64>,
        radius: f64,
    },

    /// A measurement plane (for circumferences, widths, etc.).
    MeasurementPlane {
        measurement_type: MeasurementType,
        origin: Point3<f64>,
        normal: Vector3<f64>,
    },
}

/// Types of measurements that can be constrained.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum MeasurementType {
    /// Circumference around a cross-section.
    Circumference,
    /// Width (extent in a direction).
    Width,
    /// Height (extent in vertical direction).
    Height,
    /// Depth (extent in a direction).
    Depth,
}

/// A measurement value with optional tolerance.
#[derive(Debug, Clone)]
pub struct Measurement {
    /// The target value.
    pub value: f64,
    /// Tolerance for the measurement (default: 1mm).
    pub tolerance: f64,
    /// Whether this is a minimum (true) or exact (false) constraint.
    pub is_minimum: bool,
}

impl Measurement {
    /// Create an exact measurement constraint.
    pub fn exact(value: f64) -> Self {
        Self {
            value,
            tolerance: 1.0,
            is_minimum: false,
        }
    }

    /// Create a measurement with tolerance.
    pub fn with_tolerance(value: f64, tolerance: f64) -> Self {
        Self {
            value,
            tolerance,
            is_minimum: false,
        }
    }

    /// Create a minimum measurement constraint.
    pub fn minimum(value: f64) -> Self {
        Self {
            value,
            tolerance: 1.0,
            is_minimum: true,
        }
    }
}

/// Parameters for template fitting.
#[derive(Debug, Clone, Default)]
pub struct FitParams {
    /// Target scan to fit to (optional).
    pub target_scan: Option<Mesh>,

    /// Target positions for landmark regions.
    pub landmark_targets: HashMap<String, Point3<f64>>,

    /// Target measurements for measurement regions.
    pub measurement_targets: HashMap<String, Measurement>,

    /// Smoothness parameter for morphing (higher = smoother).
    pub smoothness: f64,

    /// Maximum iterations for registration.
    pub registration_iterations: usize,

    /// Convergence threshold for registration.
    pub convergence_threshold: f64,
}

impl FitParams {
    /// Create default fitting parameters.
    pub fn new() -> Self {
        Self {
            target_scan: None,
            landmark_targets: HashMap::new(),
            measurement_targets: HashMap::new(),
            smoothness: 1.0,
            registration_iterations: 100,
            convergence_threshold: 1e-6,
        }
    }

    /// Set the target scan.
    pub fn with_target_scan(mut self, scan: Mesh) -> Self {
        self.target_scan = Some(scan);
        self
    }

    /// Add a landmark target.
    pub fn with_landmark_target(mut self, name: impl Into<String>, target: Point3<f64>) -> Self {
        self.landmark_targets.insert(name.into(), target);
        self
    }

    /// Add multiple landmark targets.
    pub fn with_landmark_targets(mut self, targets: HashMap<String, Point3<f64>>) -> Self {
        self.landmark_targets.extend(targets);
        self
    }

    /// Add a measurement target.
    pub fn with_measurement(mut self, name: impl Into<String>, measurement: Measurement) -> Self {
        self.measurement_targets.insert(name.into(), measurement);
        self
    }

    /// Add multiple measurement targets.
    pub fn with_measurements(mut self, measurements: HashMap<String, Measurement>) -> Self {
        self.measurement_targets.extend(measurements);
        self
    }

    /// Set the smoothness parameter.
    pub fn with_smoothness(mut self, smoothness: f64) -> Self {
        self.smoothness = smoothness;
        self
    }

    /// Set the registration iterations.
    pub fn with_registration_iterations(mut self, iterations: usize) -> Self {
        self.registration_iterations = iterations;
        self
    }
}

/// Result of a template fitting operation.
#[derive(Debug, Clone)]
pub struct FitResult {
    /// The fitted mesh.
    pub mesh: Mesh,

    /// Overall fit error (RMS distance at control points).
    pub fit_error: f64,

    /// Stages that were completed.
    pub stages: Vec<FitStage>,

    /// The rigid transformation applied.
    pub transform: RigidTransform,
}

impl FitResult {
    /// Check if the fit is acceptable.
    pub fn is_acceptable(&self, max_error: f64) -> bool {
        self.fit_error <= max_error
    }
}

/// A stage in the fitting process.
#[derive(Debug, Clone)]
pub enum FitStage {
    /// Rigid alignment stage.
    RigidAlignment { rms_error: f64 },

    /// Landmark-based deformation stage.
    LandmarkDeformation {
        constraints_applied: usize,
        max_displacement: f64,
    },

    /// Measurement-based adjustment stage.
    MeasurementAdjustment { measurements_applied: usize },
}

/// Apply a measurement constraint to a mesh.
fn apply_measurement_constraint(
    mesh: &Mesh,
    region: &ControlRegion,
    measurement: &Measurement,
) -> MeshResult<Mesh> {
    let RegionDefinition::MeasurementPlane {
        measurement_type,
        origin,
        normal,
    } = &region.definition
    else {
        return Ok(mesh.clone());
    };

    // Get vertices in the measurement region
    let vertex_indices = region.get_vertex_indices(mesh);
    if vertex_indices.is_empty() {
        return Ok(mesh.clone());
    }

    // Calculate current measurement
    let current_value = match measurement_type {
        MeasurementType::Circumference => {
            // Approximate circumference from vertices near the plane
            // This is a simplified calculation
            let region_vertices: Vec<Point3<f64>> = vertex_indices
                .iter()
                .filter_map(|&i| mesh.vertices.get(i as usize))
                .map(|v| v.position)
                .collect();

            if region_vertices.len() < 3 {
                return Ok(mesh.clone());
            }

            // Project vertices onto plane and compute perimeter
            let projected: Vec<Point3<f64>> = region_vertices
                .iter()
                .map(|p| {
                    let dist = (p - origin).dot(normal);
                    Point3::from(p.coords - dist * normal)
                })
                .collect();

            // Very rough circumference estimate using bounding box
            let centroid: Vector3<f64> =
                projected.iter().map(|p| p.coords).sum::<Vector3<f64>>() / projected.len() as f64;
            let avg_radius = projected
                .iter()
                .map(|p| (p.coords - centroid).norm())
                .sum::<f64>()
                / projected.len() as f64;

            2.0 * std::f64::consts::PI * avg_radius
        }
        MeasurementType::Width | MeasurementType::Depth => {
            // Compute extent perpendicular to normal
            let region_vertices: Vec<Point3<f64>> = vertex_indices
                .iter()
                .filter_map(|&i| mesh.vertices.get(i as usize))
                .map(|v| v.position)
                .collect();

            if region_vertices.is_empty() {
                return Ok(mesh.clone());
            }

            // Project onto plane
            let projected: Vec<Point3<f64>> = region_vertices
                .iter()
                .map(|p| {
                    let dist = (p - origin).dot(normal);
                    Point3::from(p.coords - dist * normal)
                })
                .collect();

            // Compute extent in an arbitrary direction perpendicular to normal
            let perpendicular = if normal.x.abs() < 0.9 {
                normal.cross(&Vector3::x()).normalize()
            } else {
                normal.cross(&Vector3::y()).normalize()
            };

            let projections: Vec<f64> = projected
                .iter()
                .map(|p| p.coords.dot(&perpendicular))
                .collect();
            let min = projections.iter().copied().fold(f64::INFINITY, f64::min);
            let max = projections
                .iter()
                .copied()
                .fold(f64::NEG_INFINITY, f64::max);

            max - min
        }
        MeasurementType::Height => {
            // Compute extent in the normal direction
            let projections: Vec<f64> = vertex_indices
                .iter()
                .filter_map(|&i| mesh.vertices.get(i as usize))
                .map(|v| (v.position - origin).dot(normal))
                .collect();

            if projections.is_empty() {
                return Ok(mesh.clone());
            }

            let min = projections.iter().copied().fold(f64::INFINITY, f64::min);
            let max = projections
                .iter()
                .copied()
                .fold(f64::NEG_INFINITY, f64::max);

            max - min
        }
    };

    // Check if adjustment is needed
    let target_value = measurement.value;
    let diff = target_value - current_value;

    if diff.abs() <= measurement.tolerance
        || (measurement.is_minimum && current_value >= target_value)
    {
        return Ok(mesh.clone());
    }

    // Calculate scale factor
    let scale_factor = if current_value > 1e-6 {
        target_value / current_value
    } else {
        1.0
    };

    // Apply scaling to vertices in the region
    let mut result = mesh.clone();
    let centroid = {
        let sum: Vector3<f64> = vertex_indices
            .iter()
            .filter_map(|&i| mesh.vertices.get(i as usize))
            .map(|v| v.position.coords)
            .sum();
        Point3::from(sum / vertex_indices.len() as f64)
    };

    for &idx in &vertex_indices {
        if let Some(vertex) = result.vertices.get_mut(idx as usize) {
            let offset = vertex.position - centroid;
            // Scale perpendicular to the measurement direction
            let along_normal = offset.dot(normal) * normal;
            let perpendicular = offset - along_normal;
            let scaled_perpendicular = perpendicular * scale_factor;
            vertex.position = centroid + along_normal + scaled_perpendicular;
        }
    }

    Ok(result)
}

/// Calculate the overall fit error.
fn calculate_fit_error(
    mesh: &Mesh,
    params: &FitParams,
    regions: &HashMap<String, ControlRegion>,
) -> f64 {
    let mut total_error_sq = 0.0;
    let mut count = 0;

    // Error from landmark targets
    for (name, target) in &params.landmark_targets {
        if let Some(region) = regions.get(name) {
            let current = match &region.definition {
                RegionDefinition::Point(p) => *p,
                RegionDefinition::Vertices(indices) if !indices.is_empty() => {
                    let sum: Vector3<f64> = indices
                        .iter()
                        .filter_map(|&i| mesh.vertices.get(i as usize))
                        .map(|v| v.position.coords)
                        .sum();
                    Point3::from(sum / indices.len() as f64)
                }
                _ => continue,
            };

            let error = (current - target).norm();
            total_error_sq += error * error * region.weight;
            count += 1;
        }
    }

    if count > 0 {
        (total_error_sq / count as f64).sqrt()
    } else {
        0.0
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::Vertex;

    fn create_test_mesh() -> Mesh {
        let mut mesh = Mesh::new();
        // Create a simple box-like mesh
        mesh.vertices.push(Vertex::from_coords(0.0, 0.0, 0.0));
        mesh.vertices.push(Vertex::from_coords(10.0, 0.0, 0.0));
        mesh.vertices.push(Vertex::from_coords(10.0, 10.0, 0.0));
        mesh.vertices.push(Vertex::from_coords(0.0, 10.0, 0.0));
        mesh.vertices.push(Vertex::from_coords(5.0, 5.0, 10.0)); // Apex

        mesh.faces.push([0, 1, 4]);
        mesh.faces.push([1, 2, 4]);
        mesh.faces.push([2, 3, 4]);
        mesh.faces.push([3, 0, 4]);
        mesh.faces.push([0, 3, 2]);
        mesh.faces.push([0, 2, 1]);
        mesh
    }

    #[test]
    fn test_template_creation() {
        let mesh = create_test_mesh();
        let template = FitTemplate::new(mesh)
            .with_control_region(ControlRegion::point("apex", Point3::new(5.0, 5.0, 10.0)))
            .with_control_region(ControlRegion::vertices("base", vec![0, 1, 2, 3]));

        assert_eq!(template.control_regions.len(), 2);
        assert!(template.get_region("apex").is_some());
        assert!(template.get_region("base").is_some());
    }

    #[test]
    fn test_landmark_position() {
        let mesh = create_test_mesh();
        let template = FitTemplate::new(mesh)
            .with_control_region(ControlRegion::point("apex", Point3::new(5.0, 5.0, 10.0)));

        let pos = template.get_landmark_position("apex").unwrap();
        assert!((pos.x - 5.0).abs() < 1e-10);
        assert!((pos.y - 5.0).abs() < 1e-10);
        assert!((pos.z - 10.0).abs() < 1e-10);
    }

    #[test]
    fn test_landmark_fitting() {
        let mesh = create_test_mesh();
        let template = FitTemplate::new(mesh)
            .with_control_region(ControlRegion::point("apex", Point3::new(5.0, 5.0, 10.0)))
            .with_control_region(ControlRegion::point("base1", Point3::new(0.0, 0.0, 0.0)))
            .with_control_region(ControlRegion::point("base2", Point3::new(10.0, 0.0, 0.0)))
            .with_control_region(ControlRegion::point("base3", Point3::new(10.0, 10.0, 0.0)));

        // Fit to move the apex higher - need multiple landmarks for numerical stability
        let params = FitParams::default()
            .with_landmark_target("apex", Point3::new(5.0, 5.0, 15.0))
            .with_landmark_target("base1", Point3::new(0.0, 0.0, 0.0))
            .with_landmark_target("base2", Point3::new(10.0, 0.0, 0.0))
            .with_landmark_target("base3", Point3::new(10.0, 10.0, 0.0));

        let result = template.fit(&params).unwrap();

        // The apex vertex should have moved upward
        let apex = result.mesh.vertices[4].position;
        assert!(apex.z > 10.0, "Apex should have moved up: z={}", apex.z);
    }

    #[test]
    fn test_region_vertex_indices_bounds() {
        let mesh = create_test_mesh();

        // Create a bounds region that includes the base vertices
        let region = ControlRegion::bounds(
            "lower_half",
            Point3::new(-1.0, -1.0, -1.0),
            Point3::new(11.0, 11.0, 5.0),
        );

        let indices = region.get_vertex_indices(&mesh);
        assert_eq!(indices.len(), 4); // Should include vertices 0-3 (base)
        assert!(indices.contains(&0));
        assert!(indices.contains(&1));
        assert!(indices.contains(&2));
        assert!(indices.contains(&3));
        assert!(!indices.contains(&4)); // Apex is at z=10, above the bounds
    }

    #[test]
    fn test_region_vertex_indices_sphere() {
        let mesh = create_test_mesh();

        // Create a sphere region centered on the apex
        let region = ControlRegion::sphere("near_apex", Point3::new(5.0, 5.0, 10.0), 3.0);

        let indices = region.get_vertex_indices(&mesh);
        assert!(indices.contains(&4)); // Apex should be included
    }

    #[test]
    fn test_fit_to_scan() {
        let template_mesh = create_test_mesh();
        let template = FitTemplate::new(template_mesh.clone());

        // Create a "scan" that's the same mesh translated
        let mut scan = template_mesh.clone();
        for vertex in &mut scan.vertices {
            vertex.position.x += 5.0;
        }

        let result = template.fit_to_scan(&scan).unwrap();

        // After fitting, the mesh should be closer to the scan
        assert!(!result.stages.is_empty());
    }

    #[test]
    fn test_fit_stages() {
        let mesh = create_test_mesh();
        let template = FitTemplate::new(mesh.clone())
            .with_control_region(ControlRegion::point("apex", Point3::new(5.0, 5.0, 10.0)))
            .with_control_region(ControlRegion::point("base1", Point3::new(0.0, 0.0, 0.0)))
            .with_control_region(ControlRegion::point("base2", Point3::new(10.0, 0.0, 0.0)))
            .with_control_region(ControlRegion::point("base3", Point3::new(10.0, 10.0, 0.0)));

        let mut scan = mesh.clone();
        for vertex in &mut scan.vertices {
            vertex.position.x += 2.0;
        }

        // Need multiple well-separated landmarks for RBF stability
        let params = FitParams::default()
            .with_target_scan(scan)
            .with_landmark_target("apex", Point3::new(7.0, 5.0, 12.0))
            .with_landmark_target("base1", Point3::new(2.0, 0.0, 0.0))
            .with_landmark_target("base2", Point3::new(12.0, 0.0, 0.0))
            .with_landmark_target("base3", Point3::new(12.0, 10.0, 0.0));

        let result = template.fit(&params).unwrap();

        // Should have both rigid alignment and landmark deformation stages
        assert!(result.stages.len() >= 2);
    }

    #[test]
    fn test_empty_template_error() {
        let mesh = Mesh::new();
        let template = FitTemplate::new(mesh);

        let params = FitParams::default();
        assert!(matches!(
            template.fit(&params),
            Err(MeshError::EmptyMesh { .. })
        ));
    }

    #[test]
    fn test_measurement_exact() {
        let m = Measurement::exact(100.0);
        assert!((m.value - 100.0).abs() < 1e-10);
        assert!(!m.is_minimum);
    }

    #[test]
    fn test_measurement_minimum() {
        let m = Measurement::minimum(50.0);
        assert!(m.is_minimum);
    }

    #[test]
    fn test_control_region_weights() {
        let region = ControlRegion::point("test", Point3::new(0.0, 0.0, 0.0)).with_weight(2.5);
        assert!((region.weight - 2.5).abs() < 1e-10);
    }

    #[test]
    fn test_preserved_region() {
        let region = ControlRegion::point("test", Point3::new(0.0, 0.0, 0.0)).preserved();
        assert!(region.preserve);
    }

    #[test]
    fn test_cylinder_region() {
        let mesh = create_test_mesh();

        // Create a cylinder that should include vertices near the center
        // The test mesh has: (0,0,0), (10,0,0), (10,10,0), (0,10,0), (5,5,10)
        // Center is approximately (5,5,5) - use a larger radius to catch vertices
        let region = ControlRegion::cylinder(
            "vertical",
            Point3::new(5.0, 5.0, 0.0),
            Point3::new(5.0, 5.0, 10.0),
            10.0, // Larger radius to include corner vertices
        );

        let indices = region.get_vertex_indices(&mesh);
        // With radius 10, should include apex at (5,5,10) and potentially others
        assert!(
            !indices.is_empty(),
            "Should find at least some vertices in cylinder"
        );
    }

    #[test]
    fn test_region_names() {
        let mesh = create_test_mesh();
        let template = FitTemplate::new(mesh)
            .with_control_region(ControlRegion::point("a", Point3::origin()))
            .with_control_region(ControlRegion::point("b", Point3::origin()))
            .with_control_region(ControlRegion::point("c", Point3::origin()));

        let names = template.region_names();
        assert_eq!(names.len(), 3);
    }

    #[test]
    fn test_fit_params_builder() {
        let params = FitParams::new()
            .with_landmark_target("heel", Point3::new(0.0, 0.0, 0.0))
            .with_landmark_target("toe", Point3::new(100.0, 0.0, 0.0))
            .with_smoothness(2.0)
            .with_registration_iterations(50);

        assert_eq!(params.landmark_targets.len(), 2);
        assert!((params.smoothness - 2.0).abs() < 1e-10);
        assert_eq!(params.registration_iterations, 50);
    }

    #[test]
    fn test_fit_result_acceptable() {
        let mesh = create_test_mesh();
        let template = FitTemplate::new(mesh);

        let params = FitParams::default();
        let result = template.fit(&params).unwrap();

        assert!(result.is_acceptable(1.0)); // With no constraints, error should be 0
    }
}

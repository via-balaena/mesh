//! Mesh validation and reporting.

use nalgebra::Point3;
use tracing::{debug, info, warn};

use crate::Mesh;
use crate::adjacency::MeshAdjacency;
use crate::error::{MeshError, MeshResult, ValidationIssue};

/// Validation report for a mesh.
#[derive(Debug, Clone)]
pub struct MeshReport {
    /// Whether the mesh has no boundary edges.
    pub is_watertight: bool,

    /// Whether all edges have at most 2 adjacent faces.
    pub is_manifold: bool,

    /// Number of boundary edges (edges with 1 adjacent face).
    pub boundary_edge_count: usize,

    /// Number of non-manifold edges (edges with >2 adjacent faces).
    pub non_manifold_edge_count: usize,

    /// Total vertex count.
    pub vertex_count: usize,

    /// Total face count.
    pub face_count: usize,

    /// Bounding box as (min_corner, max_corner).
    pub bounds: Option<(Point3<f64>, Point3<f64>)>,

    /// Dimensions (x, y, z).
    pub dimensions: Option<(f64, f64, f64)>,

    /// Signed volume of the mesh (positive = outward normals, negative = inside-out).
    /// Only meaningful for closed (watertight) meshes.
    pub signed_volume: f64,

    /// Absolute volume of the mesh.
    pub volume: f64,

    /// Total surface area of the mesh.
    pub surface_area: f64,

    /// Whether the mesh appears to be inside-out (negative signed volume).
    pub is_inside_out: bool,

    /// Number of connected components.
    pub component_count: usize,
}

impl MeshReport {
    /// Check if mesh passes basic validity checks.
    pub fn is_valid(&self) -> bool {
        self.vertex_count > 0 && self.face_count > 0
    }

    /// Check if mesh is suitable for 3D printing.
    ///
    /// A printable mesh must be:
    /// - Watertight (no boundary edges)
    /// - Manifold (no edge shared by more than 2 faces)
    /// - Not inside-out (normals point outward)
    pub fn is_printable(&self) -> bool {
        self.is_watertight && self.is_manifold && !self.is_inside_out
    }

    /// Check if mesh has correct normal orientation (not inside-out).
    pub fn has_correct_orientation(&self) -> bool {
        !self.is_inside_out
    }
}

impl std::fmt::Display for MeshReport {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        writeln!(f, "Mesh Report:")?;
        writeln!(f, "  Vertices: {}", self.vertex_count)?;
        writeln!(f, "  Faces: {}", self.face_count)?;
        writeln!(f, "  Components: {}", self.component_count)?;

        if let Some((min, max)) = &self.bounds {
            writeln!(
                f,
                "  Bounds: [{:.1}, {:.1}, {:.1}] to [{:.1}, {:.1}, {:.1}]",
                min.x, min.y, min.z, max.x, max.y, max.z
            )?;
        }

        if let Some((dx, dy, dz)) = &self.dimensions {
            writeln!(f, "  Dimensions: {:.1} x {:.1} x {:.1}", dx, dy, dz)?;
        }

        writeln!(f, "  Surface Area: {:.2}", self.surface_area)?;
        writeln!(
            f,
            "  Volume: {:.2} (signed: {:.2})",
            self.volume, self.signed_volume
        )?;

        writeln!(
            f,
            "  Watertight: {} (boundary edges: {})",
            if self.is_watertight { "yes" } else { "NO" },
            self.boundary_edge_count
        )?;

        writeln!(
            f,
            "  Manifold: {} (non-manifold edges: {})",
            if self.is_manifold { "yes" } else { "NO" },
            self.non_manifold_edge_count
        )?;

        writeln!(
            f,
            "  Orientation: {}",
            if self.is_inside_out {
                "INSIDE-OUT"
            } else {
                "correct"
            }
        )?;

        writeln!(
            f,
            "  Printable: {}",
            if self.is_printable() { "yes" } else { "NO" }
        )?;

        Ok(())
    }
}

/// Validate a mesh and return a report.
pub fn validate_mesh(mesh: &Mesh) -> MeshReport {
    let adjacency = MeshAdjacency::build(&mesh.faces);

    let boundary_edge_count = adjacency.boundary_edge_count();
    let non_manifold_edge_count = adjacency.non_manifold_edge_count();

    let bounds = mesh.bounds();
    let dimensions = bounds.map(|(min, max)| (max.x - min.x, max.y - min.y, max.z - min.z));

    // Compute volume
    let signed_volume = mesh.signed_volume();
    let volume = signed_volume.abs();
    let is_inside_out = signed_volume < 0.0;

    // Compute surface area
    let surface_area = mesh.surface_area();

    // Count connected components
    let component_count = crate::components::find_connected_components(mesh).component_count;

    let report = MeshReport {
        is_watertight: boundary_edge_count == 0,
        is_manifold: non_manifold_edge_count == 0,
        boundary_edge_count,
        non_manifold_edge_count,
        vertex_count: mesh.vertex_count(),
        face_count: mesh.face_count(),
        bounds,
        dimensions,
        signed_volume,
        volume,
        surface_area,
        is_inside_out,
        component_count,
    };

    // Log warnings
    if !report.is_watertight {
        warn!(
            "Mesh is not watertight: {} boundary edges",
            boundary_edge_count
        );
    }

    if !report.is_manifold {
        warn!(
            "Mesh is not manifold: {} non-manifold edges",
            non_manifold_edge_count
        );
    }

    if report.is_inside_out && report.is_watertight {
        warn!("Mesh appears to be inside-out (negative signed volume)");
    }

    debug!("{}", report);

    report
}

/// Log a summary of mesh validation.
pub fn log_validation(report: &MeshReport) {
    info!(
        "Mesh: {} verts, {} faces, {}x{}x{}",
        report.vertex_count,
        report.face_count,
        report
            .dimensions
            .map(|d| format!("{:.1}", d.0))
            .unwrap_or_default(),
        report
            .dimensions
            .map(|d| format!("{:.1}", d.1))
            .unwrap_or_default(),
        report
            .dimensions
            .map(|d| format!("{:.1}", d.2))
            .unwrap_or_default(),
    );

    if report.is_printable() {
        info!("Mesh is watertight and manifold (printable)");
    } else {
        if !report.is_watertight {
            warn!(
                "Not watertight: {} boundary edges",
                report.boundary_edge_count
            );
        }
        if !report.is_manifold {
            warn!(
                "Not manifold: {} non-manifold edges",
                report.non_manifold_edge_count
            );
        }
    }
}

/// Options for mesh data validation.
#[derive(Debug, Clone)]
pub struct ValidationOptions {
    /// Whether to reject the mesh on finding invalid data (default: true).
    /// If false, issues are collected but validation continues.
    pub reject_on_invalid: bool,
    /// Maximum number of issues to collect before stopping (default: 100).
    pub max_issues: usize,
}

impl Default for ValidationOptions {
    fn default() -> Self {
        Self {
            reject_on_invalid: true,
            max_issues: 100,
        }
    }
}

impl ValidationOptions {
    /// Create options that collect all issues without rejecting.
    pub fn collect_all() -> Self {
        Self {
            reject_on_invalid: false,
            max_issues: 1000,
        }
    }
}

/// Result of mesh data validation.
#[derive(Debug, Clone)]
pub struct DataValidationResult {
    /// List of issues found during validation.
    pub issues: Vec<ValidationIssue>,
    /// Number of invalid vertex indices found.
    pub invalid_index_count: usize,
    /// Number of NaN coordinates found.
    pub nan_count: usize,
    /// Number of infinite coordinates found.
    pub infinity_count: usize,
}

impl DataValidationResult {
    /// Check if validation passed with no issues.
    pub fn is_valid(&self) -> bool {
        self.issues.is_empty()
    }

    /// Get total number of issues found.
    pub fn issue_count(&self) -> usize {
        self.issues.len()
    }
}

impl std::fmt::Display for DataValidationResult {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        if self.is_valid() {
            write!(f, "Data validation passed: no issues found")
        } else {
            writeln!(f, "Data validation found {} issue(s):", self.issue_count())?;
            if self.invalid_index_count > 0 {
                writeln!(f, "  - {} invalid vertex indices", self.invalid_index_count)?;
            }
            if self.nan_count > 0 {
                writeln!(f, "  - {} NaN coordinates", self.nan_count)?;
            }
            if self.infinity_count > 0 {
                writeln!(f, "  - {} infinite coordinates", self.infinity_count)?;
            }
            Ok(())
        }
    }
}

/// Validate mesh data for invalid indices and coordinates.
///
/// This function checks:
/// - Face indices are within vertex bounds
/// - Vertex coordinates are not NaN
/// - Vertex coordinates are not infinite
///
/// # Arguments
/// * `mesh` - The mesh to validate
/// * `options` - Validation options controlling behavior
///
/// # Returns
/// - `Ok(DataValidationResult)` - Validation completed (check `is_valid()` for result)
/// - `Err(MeshError)` - Validation failed (only when `reject_on_invalid` is true and issues found)
///
/// # Example
/// ```
/// use mesh_repair::{Mesh, validate::{validate_mesh_data, ValidationOptions}};
///
/// let mesh = Mesh::new();
/// let result = validate_mesh_data(&mesh, &ValidationOptions::default());
/// match result {
///     Ok(validation) => {
///         if validation.is_valid() {
///             println!("Mesh data is valid");
///         } else {
///             println!("Found {} issues", validation.issue_count());
///         }
///     }
///     Err(e) => println!("Validation failed: {}", e),
/// }
/// ```
pub fn validate_mesh_data(
    mesh: &Mesh,
    options: &ValidationOptions,
) -> MeshResult<DataValidationResult> {
    let mut issues = Vec::new();
    let mut invalid_index_count = 0;
    let mut nan_count = 0;
    let mut infinity_count = 0;

    let vertex_count = mesh.vertices.len();

    // Check vertex coordinates for NaN and Infinity
    for (vertex_idx, vertex) in mesh.vertices.iter().enumerate() {
        if issues.len() >= options.max_issues {
            break;
        }

        let coords = [
            ("x", vertex.position.x),
            ("y", vertex.position.y),
            ("z", vertex.position.z),
        ];

        for (coord_name, value) in coords {
            if value.is_nan() {
                nan_count += 1;
                issues.push(ValidationIssue::NaNCoordinate {
                    vertex_index: vertex_idx,
                    coordinate: coord_name,
                });

                if options.reject_on_invalid {
                    return Err(MeshError::InvalidCoordinate {
                        vertex_index: vertex_idx,
                        coordinate: coord_name,
                        value,
                    });
                }
            } else if value.is_infinite() {
                infinity_count += 1;
                issues.push(ValidationIssue::InfiniteCoordinate {
                    vertex_index: vertex_idx,
                    coordinate: coord_name,
                    value,
                });

                if options.reject_on_invalid {
                    return Err(MeshError::InvalidCoordinate {
                        vertex_index: vertex_idx,
                        coordinate: coord_name,
                        value,
                    });
                }
            }
        }
    }

    // Check face indices are within bounds
    for (face_idx, face) in mesh.faces.iter().enumerate() {
        if issues.len() >= options.max_issues {
            break;
        }

        for &vertex_idx in face {
            if vertex_idx as usize >= vertex_count {
                invalid_index_count += 1;
                issues.push(ValidationIssue::InvalidVertexIndex {
                    face_index: face_idx,
                    vertex_index: vertex_idx,
                    vertex_count,
                });

                if options.reject_on_invalid {
                    return Err(MeshError::InvalidVertexIndex {
                        face_index: face_idx,
                        vertex_index: vertex_idx,
                        vertex_count,
                    });
                }
            }
        }
    }

    if !issues.is_empty() {
        warn!(
            "Mesh data validation found {} issue(s): {} invalid indices, {} NaN, {} Inf",
            issues.len(),
            invalid_index_count,
            nan_count,
            infinity_count
        );
    } else {
        debug!("Mesh data validation passed");
    }

    Ok(DataValidationResult {
        issues,
        invalid_index_count,
        nan_count,
        infinity_count,
    })
}

/// Validate mesh data with default options (rejects on first error).
///
/// This is a convenience wrapper around `validate_mesh_data` with default options.
pub fn validate_mesh_data_strict(mesh: &Mesh) -> MeshResult<()> {
    validate_mesh_data(mesh, &ValidationOptions::default())?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::Vertex;

    fn tetrahedron() -> Mesh {
        // Tetrahedron with outward-facing normals (positive signed volume)
        let mut mesh = Mesh::new();
        mesh.vertices.push(Vertex::from_coords(0.0, 0.0, 0.0)); // 0
        mesh.vertices.push(Vertex::from_coords(1.0, 0.0, 0.0)); // 1
        mesh.vertices.push(Vertex::from_coords(0.5, 0.866025, 0.0)); // 2 (approx sqrt(3)/2)
        mesh.vertices
            .push(Vertex::from_coords(0.5, 0.288675, 0.816497)); // 3 (apex)

        // Faces with outward normals (CCW from outside)
        mesh.faces.push([0, 2, 1]); // Bottom face
        mesh.faces.push([0, 1, 3]); // Front face
        mesh.faces.push([1, 2, 3]); // Right face
        mesh.faces.push([2, 0, 3]); // Left face

        mesh
    }

    fn single_triangle() -> Mesh {
        let mut mesh = Mesh::new();
        mesh.vertices.push(Vertex::from_coords(0.0, 0.0, 0.0));
        mesh.vertices.push(Vertex::from_coords(1.0, 0.0, 0.0));
        mesh.vertices.push(Vertex::from_coords(0.0, 1.0, 0.0));
        mesh.faces.push([0, 1, 2]);
        mesh
    }

    #[test]
    fn test_validate_watertight_mesh() {
        let mesh = tetrahedron();
        let report = validate_mesh(&mesh);

        assert!(report.is_valid());
        assert!(report.is_watertight);
        assert!(report.is_manifold);
        assert!(report.is_printable());
        assert_eq!(report.boundary_edge_count, 0);
        assert_eq!(report.non_manifold_edge_count, 0);
        assert_eq!(report.component_count, 1);
        assert!(report.volume > 0.0);
        assert!(report.surface_area > 0.0);
        assert!(!report.is_inside_out);
    }

    #[test]
    fn test_validate_open_mesh() {
        let mesh = single_triangle();
        let report = validate_mesh(&mesh);

        assert!(report.is_valid());
        assert!(!report.is_watertight); // Has boundary edges
        assert!(report.is_manifold); // No edge has >2 faces (manifold allows boundaries)
        assert!(!report.is_printable()); // Not printable because not watertight
        assert_eq!(report.boundary_edge_count, 3);
        assert_eq!(report.component_count, 1);
        assert!(report.surface_area > 0.0);
    }

    #[test]
    fn test_validate_inside_out_mesh() {
        // Create a tetrahedron with inverted winding (inside-out)
        // Same vertices as tetrahedron(), but with flipped faces
        let mut mesh = Mesh::new();
        mesh.vertices.push(Vertex::from_coords(0.0, 0.0, 0.0)); // 0
        mesh.vertices.push(Vertex::from_coords(1.0, 0.0, 0.0)); // 1
        mesh.vertices.push(Vertex::from_coords(0.5, 0.866025, 0.0)); // 2
        mesh.vertices
            .push(Vertex::from_coords(0.5, 0.288675, 0.816497)); // 3

        // Inverted winding (swap indices to reverse normal direction)
        mesh.faces.push([0, 1, 2]); // Bottom face - inverted
        mesh.faces.push([0, 3, 1]); // Front face - inverted
        mesh.faces.push([1, 3, 2]); // Right face - inverted
        mesh.faces.push([2, 3, 0]); // Left face - inverted

        let report = validate_mesh(&mesh);

        assert!(report.is_watertight);
        assert!(report.is_manifold);
        assert!(report.is_inside_out);
        assert!(!report.is_printable()); // Inside-out meshes are not printable
        assert!(report.signed_volume < 0.0);
    }

    #[test]
    fn test_report_display() {
        let mesh = tetrahedron();
        let report = validate_mesh(&mesh);
        let output = format!("{}", report);

        assert!(output.contains("Vertices: 4"));
        assert!(output.contains("Faces: 4"));
        assert!(output.contains("Components: 1"));
        assert!(output.contains("Watertight: yes"));
        assert!(output.contains("Surface Area:"));
        assert!(output.contains("Volume:"));
        assert!(output.contains("Orientation: correct"));
    }

    // ==================== Data Validation Tests ====================

    #[test]
    fn test_validate_valid_mesh_data() {
        let mesh = tetrahedron();
        let result = validate_mesh_data(&mesh, &ValidationOptions::default()).unwrap();

        assert!(result.is_valid());
        assert_eq!(result.issue_count(), 0);
        assert_eq!(result.invalid_index_count, 0);
        assert_eq!(result.nan_count, 0);
        assert_eq!(result.infinity_count, 0);
    }

    #[test]
    fn test_validate_invalid_vertex_index_strict() {
        let mut mesh = Mesh::new();
        mesh.vertices.push(Vertex::from_coords(0.0, 0.0, 0.0));
        mesh.vertices.push(Vertex::from_coords(1.0, 0.0, 0.0));
        mesh.vertices.push(Vertex::from_coords(0.0, 1.0, 0.0));
        // Face references vertex 10, but mesh only has 3 vertices
        mesh.faces.push([0, 1, 10]);

        let result = validate_mesh_data(&mesh, &ValidationOptions::default());
        assert!(result.is_err());

        match result.unwrap_err() {
            MeshError::InvalidVertexIndex {
                face_index,
                vertex_index,
                vertex_count,
            } => {
                assert_eq!(face_index, 0);
                assert_eq!(vertex_index, 10);
                assert_eq!(vertex_count, 3);
            }
            e => panic!("Expected InvalidVertexIndex error, got: {:?}", e),
        }
    }

    #[test]
    fn test_validate_invalid_vertex_index_collect() {
        let mut mesh = Mesh::new();
        mesh.vertices.push(Vertex::from_coords(0.0, 0.0, 0.0));
        mesh.vertices.push(Vertex::from_coords(1.0, 0.0, 0.0));
        mesh.vertices.push(Vertex::from_coords(0.0, 1.0, 0.0));
        // Multiple faces with invalid indices
        mesh.faces.push([0, 1, 10]);
        mesh.faces.push([0, 20, 2]);
        mesh.faces.push([30, 1, 2]);

        let result = validate_mesh_data(&mesh, &ValidationOptions::collect_all()).unwrap();

        assert!(!result.is_valid());
        assert_eq!(result.invalid_index_count, 3);
        assert_eq!(result.issue_count(), 3);
    }

    #[test]
    fn test_validate_nan_coordinate_strict() {
        let mut mesh = Mesh::new();
        mesh.vertices.push(Vertex::from_coords(f64::NAN, 0.0, 0.0));
        mesh.vertices.push(Vertex::from_coords(1.0, 0.0, 0.0));
        mesh.vertices.push(Vertex::from_coords(0.0, 1.0, 0.0));
        mesh.faces.push([0, 1, 2]);

        let result = validate_mesh_data(&mesh, &ValidationOptions::default());
        assert!(result.is_err());

        match result.unwrap_err() {
            MeshError::InvalidCoordinate {
                vertex_index,
                coordinate,
                value,
            } => {
                assert_eq!(vertex_index, 0);
                assert_eq!(coordinate, "x");
                assert!(value.is_nan());
            }
            e => panic!("Expected InvalidCoordinate error, got: {:?}", e),
        }
    }

    #[test]
    fn test_validate_nan_coordinate_collect() {
        let mut mesh = Mesh::new();
        mesh.vertices
            .push(Vertex::from_coords(f64::NAN, f64::NAN, 0.0));
        mesh.vertices.push(Vertex::from_coords(1.0, f64::NAN, 0.0));
        mesh.vertices.push(Vertex::from_coords(0.0, 1.0, 0.0));
        mesh.faces.push([0, 1, 2]);

        let result = validate_mesh_data(&mesh, &ValidationOptions::collect_all()).unwrap();

        assert!(!result.is_valid());
        assert_eq!(result.nan_count, 3); // 2 in first vertex, 1 in second
    }

    #[test]
    fn test_validate_infinity_coordinate_strict() {
        let mut mesh = Mesh::new();
        mesh.vertices
            .push(Vertex::from_coords(f64::INFINITY, 0.0, 0.0));
        mesh.vertices.push(Vertex::from_coords(1.0, 0.0, 0.0));
        mesh.vertices.push(Vertex::from_coords(0.0, 1.0, 0.0));
        mesh.faces.push([0, 1, 2]);

        let result = validate_mesh_data(&mesh, &ValidationOptions::default());
        assert!(result.is_err());

        match result.unwrap_err() {
            MeshError::InvalidCoordinate {
                vertex_index,
                coordinate,
                value,
            } => {
                assert_eq!(vertex_index, 0);
                assert_eq!(coordinate, "x");
                assert!(value.is_infinite());
            }
            e => panic!("Expected InvalidCoordinate error, got: {:?}", e),
        }
    }

    #[test]
    fn test_validate_negative_infinity_coordinate() {
        let mut mesh = Mesh::new();
        mesh.vertices
            .push(Vertex::from_coords(0.0, f64::NEG_INFINITY, 0.0));
        mesh.vertices.push(Vertex::from_coords(1.0, 0.0, 0.0));
        mesh.vertices.push(Vertex::from_coords(0.0, 1.0, 0.0));
        mesh.faces.push([0, 1, 2]);

        let result = validate_mesh_data(&mesh, &ValidationOptions::collect_all()).unwrap();

        assert!(!result.is_valid());
        assert_eq!(result.infinity_count, 1);
    }

    #[test]
    fn test_validate_multiple_issues_types() {
        let mut mesh = Mesh::new();
        mesh.vertices.push(Vertex::from_coords(f64::NAN, 0.0, 0.0)); // NaN
        mesh.vertices
            .push(Vertex::from_coords(1.0, f64::INFINITY, 0.0)); // Infinity
        mesh.vertices.push(Vertex::from_coords(0.0, 1.0, 0.0)); // Valid
        mesh.faces.push([0, 1, 99]); // Invalid index

        let result = validate_mesh_data(&mesh, &ValidationOptions::collect_all()).unwrap();

        assert!(!result.is_valid());
        assert_eq!(result.nan_count, 1);
        assert_eq!(result.infinity_count, 1);
        assert_eq!(result.invalid_index_count, 1);
        assert_eq!(result.issue_count(), 3);
    }

    #[test]
    fn test_validate_empty_mesh() {
        let mesh = Mesh::new();
        let result = validate_mesh_data(&mesh, &ValidationOptions::default()).unwrap();

        // Empty mesh has no issues (no vertices or faces to validate)
        assert!(result.is_valid());
    }

    #[test]
    fn test_validation_options_max_issues() {
        let mut mesh = Mesh::new();
        // Create a mesh with many invalid indices
        for i in 0..10 {
            mesh.vertices.push(Vertex::from_coords(i as f64, 0.0, 0.0));
        }
        // Add 20 faces with invalid indices
        for _ in 0..20 {
            mesh.faces.push([0, 1, 100]); // Invalid index
        }

        let options = ValidationOptions {
            reject_on_invalid: false,
            max_issues: 5,
        };
        let result = validate_mesh_data(&mesh, &options).unwrap();

        // Should stop at max_issues
        assert_eq!(result.issue_count(), 5);
    }

    #[test]
    fn test_validate_mesh_data_strict_passes() {
        let mesh = tetrahedron();
        assert!(validate_mesh_data_strict(&mesh).is_ok());
    }

    #[test]
    fn test_validate_mesh_data_strict_fails() {
        let mut mesh = Mesh::new();
        mesh.vertices.push(Vertex::from_coords(0.0, 0.0, 0.0));
        mesh.faces.push([0, 1, 2]); // Invalid indices

        assert!(validate_mesh_data_strict(&mesh).is_err());
    }

    #[test]
    fn test_data_validation_result_display() {
        let mut mesh = Mesh::new();
        mesh.vertices.push(Vertex::from_coords(f64::NAN, 0.0, 0.0));
        mesh.vertices.push(Vertex::from_coords(1.0, 0.0, 0.0));
        mesh.faces.push([0, 1, 99]);

        let result = validate_mesh_data(&mesh, &ValidationOptions::collect_all()).unwrap();
        let output = format!("{}", result);

        assert!(output.contains("issue"));
        assert!(output.contains("invalid vertex indices"));
        assert!(output.contains("NaN"));
    }

    #[test]
    fn test_validation_issue_display() {
        let issue = ValidationIssue::InvalidVertexIndex {
            face_index: 5,
            vertex_index: 100,
            vertex_count: 50,
        };
        let output = format!("{}", issue);
        assert!(output.contains("face 5"));
        assert!(output.contains("vertex 100"));
        assert!(output.contains("50 vertices"));

        let issue = ValidationIssue::NaNCoordinate {
            vertex_index: 3,
            coordinate: "y",
        };
        let output = format!("{}", issue);
        assert!(output.contains("vertex 3"));
        assert!(output.contains("NaN"));
        assert!(output.contains("y"));

        let issue = ValidationIssue::InfiniteCoordinate {
            vertex_index: 7,
            coordinate: "z",
            value: f64::INFINITY,
        };
        let output = format!("{}", issue);
        assert!(output.contains("vertex 7"));
        assert!(output.contains("infinite"));
        assert!(output.contains("z"));
    }
}

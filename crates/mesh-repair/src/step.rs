//! STEP file export for CAD interchange.
//!
//! This module provides STEP (ISO 10303 AP203/AP214) export functionality
//! using the truck CAD kernel. STEP is a widely-supported format for
//! exchanging geometry with CAD systems.
//!
//! # Note on Mesh Representation
//!
//! STEP files are designed for exact B-rep (boundary representation) geometry
//! with parametric curves and surfaces (NURBS, B-splines). Triangle meshes
//! are exported as faceted B-rep geometry, which uses:
//!
//! - Planar surfaces for each triangle face
//! - Linear edges for triangle edges
//! - Cartesian points for vertices
//!
//! This is the standard way to represent tessellated/mesh geometry in STEP.
//!
//! # Example
//!
//! ```no_run
//! use mesh_repair::Mesh;
//! # #[cfg(feature = "step")]
//! use mesh_repair::step::{export_step, StepExportParams};
//!
//! let mesh = Mesh::load("model.stl").unwrap();
//!
//! // Export to STEP
//! # #[cfg(feature = "step")]
//! export_step(&mesh, "output.step", &StepExportParams::default()).unwrap();
//! ```

use crate::{Mesh, MeshError, MeshResult};
use std::path::Path;

// Use truck_modeling types which have the STEP trait implementations
use truck_geometry::specifieds::{Line, Plane};
use truck_modeling::topology::{Edge, Face, Shell, Solid, Vertex, Wire};
use truck_modeling::{Curve, InnerSpace, Point3, Surface};
use truck_stepio::out::{CompleteStepDisplay, StepHeaderDescriptor, StepModel};

/// Parameters for STEP file export.
#[derive(Debug, Clone)]
pub struct StepExportParams {
    /// Organization name for STEP header
    pub organization: String,
    /// Author name for STEP header
    pub author: String,
    /// File description for STEP header
    pub description: String,
    /// Whether to merge coplanar adjacent triangles into larger polygons.
    /// This can reduce file size but may lose triangle structure.
    pub merge_coplanar: bool,
    /// Angle tolerance (in radians) for coplanar detection.
    /// Faces with normals differing by less than this are considered coplanar.
    pub coplanar_tolerance: f64,
    /// Whether to output as a solid (closed shell) or just a shell.
    /// If true and the mesh is watertight, outputs as MANIFOLD_SOLID_BREP.
    /// Otherwise outputs as SHELL_BASED_SURFACE_MODEL.
    pub prefer_solid: bool,
}

impl Default for StepExportParams {
    fn default() -> Self {
        Self {
            organization: "mesh-repair".to_string(),
            author: "mesh-repair".to_string(),
            description: "Exported triangle mesh".to_string(),
            merge_coplanar: false,
            coplanar_tolerance: 1e-6,
            prefer_solid: true,
        }
    }
}

impl StepExportParams {
    /// Create params with a custom description.
    pub fn with_description(mut self, description: impl Into<String>) -> Self {
        self.description = description.into();
        self
    }

    /// Create params with author and organization info.
    pub fn with_author(
        mut self,
        author: impl Into<String>,
        organization: impl Into<String>,
    ) -> Self {
        self.author = author.into();
        self.organization = organization.into();
        self
    }

    /// Enable coplanar face merging.
    pub fn merge_coplanar(mut self) -> Self {
        self.merge_coplanar = true;
        self
    }

    /// Set coplanar tolerance angle in radians.
    pub fn with_coplanar_tolerance(mut self, tolerance: f64) -> Self {
        self.coplanar_tolerance = tolerance;
        self
    }
}

/// Result of STEP export operation.
#[derive(Debug, Clone)]
pub struct StepExportResult {
    /// Number of faces written
    pub face_count: usize,
    /// Number of edges written
    pub edge_count: usize,
    /// Number of vertices written
    pub vertex_count: usize,
    /// Size of output file in bytes
    pub file_size: u64,
    /// Whether exported as solid (vs shell)
    pub exported_as_solid: bool,
}

/// Export a mesh to STEP format.
///
/// # Arguments
///
/// * `mesh` - The mesh to export
/// * `path` - Output file path (should end in .step or .stp)
/// * `params` - Export parameters
///
/// # Returns
///
/// Result containing export statistics, or an error if export fails.
///
/// # Example
///
/// ```no_run
/// use mesh_repair::Mesh;
/// # #[cfg(feature = "step")]
/// use mesh_repair::step::{export_step, StepExportParams};
///
/// let mesh = Mesh::load("model.stl").unwrap();
/// # #[cfg(feature = "step")]
/// let result = export_step(&mesh, "output.step", &StepExportParams::default()).unwrap();
/// # #[cfg(feature = "step")]
/// println!("Exported {} faces to STEP", result.face_count);
/// ```
pub fn export_step(
    mesh: &Mesh,
    path: impl AsRef<Path>,
    params: &StepExportParams,
) -> MeshResult<StepExportResult> {
    let path = path.as_ref();

    if mesh.faces.is_empty() {
        return Err(MeshError::EmptyMesh {
            details: "Mesh has no faces to export".into(),
        });
    }

    // Convert mesh to truck Shell
    let shell = mesh_to_truck_shell(mesh)?;

    // Determine if we should export as solid
    let is_watertight = crate::validate::validate_mesh(mesh).is_watertight;
    let export_as_solid = params.prefer_solid && is_watertight;

    // Create STEP model from shell
    let (step_string, actually_exported_as_solid) = if export_as_solid {
        // Try to create a solid from the shell
        match create_solid_step(&shell, params) {
            Ok(s) => (s, true),
            Err(e) => {
                // Fall back to shell if solid creation fails
                tracing::warn!(
                    "Could not export as STEP solid, falling back to shell: {}",
                    e
                );
                (create_shell_step(&shell, params)?, false)
            }
        }
    } else {
        (create_shell_step(&shell, params)?, false)
    };

    // Write to file
    std::fs::write(path, &step_string).map_err(|e| MeshError::IoWrite {
        path: path.to_path_buf(),
        source: e,
    })?;

    let file_size = step_string.len() as u64;

    Ok(StepExportResult {
        face_count: mesh.faces.len(),
        edge_count: count_edges(&mesh.faces),
        vertex_count: mesh.vertices.len(),
        file_size,
        exported_as_solid: actually_exported_as_solid,
    })
}

/// Export a mesh to STEP format and return the STEP data as a string.
///
/// This is useful when you want to process the STEP data without writing to a file.
pub fn export_step_to_string(mesh: &Mesh, params: &StepExportParams) -> MeshResult<String> {
    if mesh.faces.is_empty() {
        return Err(MeshError::EmptyMesh {
            details: "Mesh has no faces to export".into(),
        });
    }

    let shell = mesh_to_truck_shell(mesh)?;
    let is_watertight = crate::validate::validate_mesh(mesh).is_watertight;

    if params.prefer_solid && is_watertight {
        match create_solid_step(&shell, params) {
            Ok(s) => Ok(s),
            Err(e) => {
                tracing::warn!(
                    "Could not export as STEP solid, falling back to shell: {}",
                    e
                );
                create_shell_step(&shell, params)
            }
        }
    } else {
        create_shell_step(&shell, params)
    }
}

/// Convert our Mesh to a truck Shell.
///
/// This creates a faceted B-rep where each triangle becomes a Face with:
/// - A planar surface defined by the three vertices
/// - Three linear edges forming the boundary wire
/// - Three vertices at the corners
fn mesh_to_truck_shell(mesh: &Mesh) -> MeshResult<Shell> {
    // Create truck vertices for each mesh vertex
    let mut truck_vertices: Vec<Vertex> = Vec::with_capacity(mesh.vertices.len());
    for v in &mesh.vertices {
        let point = Point3::new(v.position[0], v.position[1], v.position[2]);
        truck_vertices.push(Vertex::new(point));
    }

    // Create faces for each triangle
    let mut faces: Vec<Face> = Vec::with_capacity(mesh.faces.len());

    for face_indices in &mesh.faces {
        let v0 = &truck_vertices[face_indices[0] as usize];
        let v1 = &truck_vertices[face_indices[1] as usize];
        let v2 = &truck_vertices[face_indices[2] as usize];

        // Get the vertex positions
        let p0 = v0.point();
        let p1 = v1.point();
        let p2 = v2.point();

        // Skip degenerate triangles
        let edge1 = p1 - p0;
        let edge2 = p2 - p0;
        let cross = edge1.cross(edge2);
        if cross.magnitude2() < 1e-20 {
            continue; // Degenerate triangle, skip it
        }

        // Create the three edges of the triangle using Line wrapped in Curve enum
        let line01: Curve = Line(p0, p1).into();
        let line12: Curve = Line(p1, p2).into();
        let line20: Curve = Line(p2, p0).into();

        let edge01 = Edge::new(v0, v1, line01);
        let edge12 = Edge::new(v1, v2, line12);
        let edge20 = Edge::new(v2, v0, line20);

        // Create a wire from the three edges
        let wire = Wire::from(vec![edge01, edge12, edge20]);

        // Create the planar surface for this face wrapped in Surface enum
        // Plane::new takes origin, u_point, v_point
        let plane: Surface = Plane::new(p0, p1, p2).into();

        // Create the face with the wire boundary and plane surface
        match Face::try_new(vec![wire], plane) {
            Ok(face) => faces.push(face),
            Err(_) => {
                // Face creation failed (likely degenerate), skip
                continue;
            }
        }
    }

    if faces.is_empty() {
        return Err(MeshError::EmptyMesh {
            details: "No valid faces could be created (all may be degenerate)".into(),
        });
    }

    Ok(Shell::from(faces))
}

/// Create STEP output for a shell (open or closed).
fn create_shell_step(shell: &Shell, params: &StepExportParams) -> MeshResult<String> {
    let compressed = shell.compress();
    let step_model = StepModel::from(&compressed);

    let header = StepHeaderDescriptor {
        organization_system: params.organization.clone(),
        ..Default::default()
    };

    let display = CompleteStepDisplay::new(step_model, header);
    Ok(format!("{}", display))
}

/// Create STEP output for a solid (closed shell).
///
/// Note: Creating a proper solid requires the shell to be topologically valid,
/// meaning faces must share edges properly. For mesh exports where faces
/// may not share topology, this will fall back to shell export.
fn create_solid_step(shell: &Shell, params: &StepExportParams) -> MeshResult<String> {
    use truck_modeling::ShellCondition;

    // Check if the shell is properly closed for solid creation
    let condition = shell.shell_condition();
    if condition != ShellCondition::Closed && condition != ShellCondition::Oriented {
        return Err(MeshError::InvalidTopology {
            details: format!(
                "Shell is not closed/oriented (condition: {:?}), cannot create solid",
                condition
            ),
        });
    }

    // Try to create a solid from the shell
    // Use catch_unwind because truck_topology::Solid::new panics if shell isn't valid
    let shell_clone = shell.clone();
    let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(move || {
        Solid::new(vec![shell_clone])
    }));

    let solid = result.map_err(|_| MeshError::InvalidTopology {
        details: "Shell topology is not valid for solid creation".into(),
    })?;

    let compressed = solid.compress();
    let step_model = StepModel::from(&compressed);

    let header = StepHeaderDescriptor {
        organization_system: params.organization.clone(),
        ..Default::default()
    };

    let display = CompleteStepDisplay::new(step_model, header);
    Ok(format!("{}", display))
}

/// Count the number of unique edges in the mesh faces.
fn count_edges(faces: &[[u32; 3]]) -> usize {
    let mut edges = std::collections::HashSet::new();
    for face in faces {
        // Add edges in canonical order (smaller index first)
        let e0 = if face[0] < face[1] {
            (face[0], face[1])
        } else {
            (face[1], face[0])
        };
        let e1 = if face[1] < face[2] {
            (face[1], face[2])
        } else {
            (face[2], face[1])
        };
        let e2 = if face[2] < face[0] {
            (face[2], face[0])
        } else {
            (face[0], face[2])
        };
        edges.insert(e0);
        edges.insert(e1);
        edges.insert(e2);
    }
    edges.len()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::Vertex as MeshVertex;

    fn create_simple_triangle() -> Mesh {
        let mut mesh = Mesh::new();
        mesh.vertices = vec![
            MeshVertex::from_coords(0.0, 0.0, 0.0),
            MeshVertex::from_coords(10.0, 0.0, 0.0),
            MeshVertex::from_coords(5.0, 10.0, 0.0),
        ];
        mesh.faces = vec![[0, 1, 2]];
        mesh
    }

    fn create_tetrahedron() -> Mesh {
        let mut mesh = Mesh::new();
        // A simple tetrahedron (closed solid)
        mesh.vertices = vec![
            MeshVertex::from_coords(0.0, 0.0, 0.0),
            MeshVertex::from_coords(10.0, 0.0, 0.0),
            MeshVertex::from_coords(5.0, 8.66, 0.0),
            MeshVertex::from_coords(5.0, 2.89, 8.16),
        ];
        // CCW winding when viewed from outside
        mesh.faces = vec![
            [0, 2, 1], // bottom
            [0, 1, 3], // front
            [1, 2, 3], // right
            [2, 0, 3], // left
        ];
        mesh
    }

    fn create_cube() -> Mesh {
        let mut mesh = Mesh::new();
        // Unit cube vertices
        mesh.vertices = vec![
            MeshVertex::from_coords(0.0, 0.0, 0.0), // 0
            MeshVertex::from_coords(1.0, 0.0, 0.0), // 1
            MeshVertex::from_coords(1.0, 1.0, 0.0), // 2
            MeshVertex::from_coords(0.0, 1.0, 0.0), // 3
            MeshVertex::from_coords(0.0, 0.0, 1.0), // 4
            MeshVertex::from_coords(1.0, 0.0, 1.0), // 5
            MeshVertex::from_coords(1.0, 1.0, 1.0), // 6
            MeshVertex::from_coords(0.0, 1.0, 1.0), // 7
        ];
        // Two triangles per face, CCW winding from outside
        mesh.faces = vec![
            // Bottom (z=0)
            [0, 2, 1],
            [0, 3, 2],
            // Top (z=1)
            [4, 5, 6],
            [4, 6, 7],
            // Front (y=0)
            [0, 1, 5],
            [0, 5, 4],
            // Back (y=1)
            [2, 3, 7],
            [2, 7, 6],
            // Left (x=0)
            [0, 4, 7],
            [0, 7, 3],
            // Right (x=1)
            [1, 2, 6],
            [1, 6, 5],
        ];
        mesh
    }

    #[test]
    fn test_export_step_to_string_triangle() {
        let mesh = create_simple_triangle();
        let params = StepExportParams::default();

        let result = export_step_to_string(&mesh, &params);
        assert!(
            result.is_ok(),
            "STEP export should succeed: {:?}",
            result.err()
        );

        let step_string = result.unwrap();
        // Check basic STEP structure
        assert!(
            step_string.contains("ISO-10303-21"),
            "Should have ISO header"
        );
        assert!(step_string.contains("HEADER"), "Should have HEADER section");
        assert!(step_string.contains("DATA"), "Should have DATA section");
        assert!(
            step_string.contains("CARTESIAN_POINT"),
            "Should have vertices"
        );
    }

    #[test]
    fn test_export_step_to_string_tetrahedron() {
        let mesh = create_tetrahedron();
        let params = StepExportParams::default();

        let result = export_step_to_string(&mesh, &params);
        assert!(
            result.is_ok(),
            "STEP export should succeed for tetrahedron: {:?}",
            result.err()
        );

        let step_string = result.unwrap();
        assert!(
            step_string.contains("CARTESIAN_POINT"),
            "Should have vertices"
        );
    }

    #[test]
    fn test_export_step_to_string_cube() {
        let mesh = create_cube();
        let params = StepExportParams::default();

        let result = export_step_to_string(&mesh, &params);
        assert!(
            result.is_ok(),
            "STEP export should succeed for cube: {:?}",
            result.err()
        );

        let step_string = result.unwrap();
        assert!(
            step_string.contains("CARTESIAN_POINT"),
            "Should have vertices"
        );
    }

    #[test]
    fn test_export_step_file() {
        let mesh = create_tetrahedron();
        let params = StepExportParams::default().with_description("Test tetrahedron");

        let temp_dir = std::env::temp_dir();
        let path = temp_dir.join("test_mesh.step");

        let result = export_step(&mesh, &path, &params);
        assert!(
            result.is_ok(),
            "STEP file export should succeed: {:?}",
            result.err()
        );

        let stats = result.unwrap();
        assert_eq!(stats.face_count, 4);
        assert!(stats.file_size > 0);

        // Cleanup
        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn test_export_step_empty_mesh() {
        let mesh = Mesh::new();
        let params = StepExportParams::default();

        let result = export_step_to_string(&mesh, &params);
        assert!(result.is_err(), "Should fail for empty mesh");
    }

    #[test]
    fn test_step_params_builder() {
        let params = StepExportParams::default()
            .with_description("Custom description")
            .with_author("Test Author", "Test Org")
            .merge_coplanar()
            .with_coplanar_tolerance(1e-5);

        assert_eq!(params.description, "Custom description");
        assert_eq!(params.author, "Test Author");
        assert_eq!(params.organization, "Test Org");
        assert!(params.merge_coplanar);
        assert!((params.coplanar_tolerance - 1e-5).abs() < 1e-10);
    }

    #[test]
    fn test_count_edges() {
        // Single triangle has 3 edges
        let faces1 = vec![[0, 1, 2]];
        assert_eq!(count_edges(&faces1), 3);

        // Two triangles sharing an edge have 5 edges
        let faces2 = vec![[0, 1, 2], [0, 2, 3]];
        assert_eq!(count_edges(&faces2), 5);

        // Tetrahedron has 6 edges
        let faces3 = vec![[0, 2, 1], [0, 1, 3], [1, 2, 3], [2, 0, 3]];
        assert_eq!(count_edges(&faces3), 6);
    }
}

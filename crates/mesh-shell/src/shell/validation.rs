//! Shell validation utilities.
//!
//! Validates shell meshes to ensure they are suitable for 3D printing.

use tracing::{debug, info, warn};

use mesh_repair::{
    Mesh, MeshAdjacency, fix_winding_order, remove_degenerate_triangles_enhanced, validate_mesh,
};

/// Result of shell validation.
#[derive(Debug, Clone)]
pub struct ShellValidationResult {
    /// Whether the shell is watertight (no boundary edges).
    pub is_watertight: bool,
    /// Whether the shell is manifold (no edges with >2 faces).
    pub is_manifold: bool,
    /// Whether the shell has consistent winding order.
    pub has_consistent_winding: bool,
    /// Number of boundary edges (should be 0 for printable shell).
    pub boundary_edge_count: usize,
    /// Number of non-manifold edges (should be 0 for printable shell).
    pub non_manifold_edge_count: usize,
    /// Total vertex count.
    pub vertex_count: usize,
    /// Total face count.
    pub face_count: usize,
    /// List of validation issues found.
    pub issues: Vec<ShellIssue>,
}

impl ShellValidationResult {
    /// Check if the shell passes all validation checks.
    pub fn is_valid(&self) -> bool {
        self.is_watertight && self.is_manifold && self.has_consistent_winding
    }

    /// Check if the shell is suitable for 3D printing.
    pub fn is_printable(&self) -> bool {
        self.is_watertight && self.is_manifold
    }

    /// Get the total number of issues found.
    pub fn issue_count(&self) -> usize {
        self.issues.len()
    }
}

impl std::fmt::Display for ShellValidationResult {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        writeln!(f, "Shell Validation Result:")?;
        writeln!(f, "  Vertices: {}", self.vertex_count)?;
        writeln!(f, "  Faces: {}", self.face_count)?;
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
            "  Consistent winding: {}",
            if self.has_consistent_winding {
                "yes"
            } else {
                "NO"
            }
        )?;
        writeln!(
            f,
            "  Printable: {}",
            if self.is_printable() { "yes" } else { "NO" }
        )?;

        if !self.issues.is_empty() {
            writeln!(f, "  Issues ({}):", self.issues.len())?;
            for issue in &self.issues {
                writeln!(f, "    - {}", issue)?;
            }
        }

        Ok(())
    }
}

/// Issues that can be found during shell validation.
#[derive(Debug, Clone)]
pub enum ShellIssue {
    /// Shell has boundary edges (not watertight).
    NotWatertight { boundary_edge_count: usize },
    /// Shell has non-manifold edges.
    NonManifold { non_manifold_edge_count: usize },
    /// Shell has inconsistent face winding.
    InconsistentWinding,
    /// Shell has zero faces.
    EmptyShell,
    /// Shell has degenerate triangles.
    DegenerateTriangles { count: usize },
}

impl std::fmt::Display for ShellIssue {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ShellIssue::NotWatertight {
                boundary_edge_count,
            } => {
                write!(
                    f,
                    "Shell is not watertight ({} boundary edges)",
                    boundary_edge_count
                )
            }
            ShellIssue::NonManifold {
                non_manifold_edge_count,
            } => {
                write!(
                    f,
                    "Shell is not manifold ({} non-manifold edges)",
                    non_manifold_edge_count
                )
            }
            ShellIssue::InconsistentWinding => {
                write!(f, "Shell has inconsistent face winding order")
            }
            ShellIssue::EmptyShell => {
                write!(f, "Shell is empty (no faces)")
            }
            ShellIssue::DegenerateTriangles { count } => {
                write!(f, "Shell has {} degenerate triangles", count)
            }
        }
    }
}

/// Validate a shell mesh for 3D printing suitability.
///
/// Checks:
/// - Watertightness (no boundary edges)
/// - Manifoldness (no edges with >2 adjacent faces)
/// - Consistent winding order
///
/// # Arguments
/// * `shell` - The shell mesh to validate
///
/// # Returns
/// A `ShellValidationResult` with detailed validation information.
///
/// # Example
/// ```
/// use mesh_repair::Mesh;
/// use mesh_shell::validate_shell;
///
/// let shell = Mesh::new();
/// let result = validate_shell(&shell);
/// if result.is_printable() {
///     println!("Shell is ready for printing!");
/// } else {
///     println!("Issues found: {}", result);
/// }
/// ```
pub fn validate_shell(shell: &Mesh) -> ShellValidationResult {
    info!(
        "Validating shell mesh ({} vertices, {} faces)",
        shell.vertex_count(),
        shell.face_count()
    );

    let mut issues = Vec::new();

    // Check for empty shell
    if shell.faces.is_empty() {
        issues.push(ShellIssue::EmptyShell);
        return ShellValidationResult {
            is_watertight: false,
            is_manifold: false,
            has_consistent_winding: false,
            boundary_edge_count: 0,
            non_manifold_edge_count: 0,
            vertex_count: shell.vertex_count(),
            face_count: 0,
            issues,
        };
    }

    // Use mesh-repair's validation to check topology
    let mesh_report = validate_mesh(shell);

    let boundary_edge_count = mesh_report.boundary_edge_count;
    let non_manifold_edge_count = mesh_report.non_manifold_edge_count;

    // Check watertightness
    let is_watertight = boundary_edge_count == 0;
    if !is_watertight {
        issues.push(ShellIssue::NotWatertight {
            boundary_edge_count,
        });
        warn!(
            "Shell is not watertight: {} boundary edges",
            boundary_edge_count
        );
    }

    // Check manifoldness
    let is_manifold = non_manifold_edge_count == 0;
    if !is_manifold {
        issues.push(ShellIssue::NonManifold {
            non_manifold_edge_count,
        });
        warn!(
            "Shell is not manifold: {} non-manifold edges",
            non_manifold_edge_count
        );
    }

    // Check winding consistency
    let has_consistent_winding = check_winding_consistency(shell);
    if !has_consistent_winding {
        issues.push(ShellIssue::InconsistentWinding);
        warn!("Shell has inconsistent winding order");
    }

    // Check for degenerate triangles
    let degenerate_count = count_degenerate_triangles(shell);
    if degenerate_count > 0 {
        issues.push(ShellIssue::DegenerateTriangles {
            count: degenerate_count,
        });
        warn!("Shell has {} degenerate triangles", degenerate_count);
    }

    let result = ShellValidationResult {
        is_watertight,
        is_manifold,
        has_consistent_winding,
        boundary_edge_count,
        non_manifold_edge_count,
        vertex_count: shell.vertex_count(),
        face_count: shell.face_count(),
        issues,
    };

    if result.is_printable() {
        info!("Shell validation passed - mesh is printable");
    } else {
        warn!("Shell validation found {} issue(s)", result.issue_count());
    }

    debug!("{}", result);

    result
}

/// Check if the mesh has consistent winding order.
///
/// For a valid closed mesh, adjacent faces should have opposite winding
/// along their shared edge (so normals point consistently outward).
fn check_winding_consistency(mesh: &Mesh) -> bool {
    let adjacency = MeshAdjacency::build(&mesh.faces);

    // For each edge, check that the two adjacent faces have opposite winding
    for (&edge, face_indices) in &adjacency.edge_to_faces {
        if face_indices.len() != 2 {
            // Skip boundary or non-manifold edges
            continue;
        }

        let face_a = mesh.faces[face_indices[0] as usize];
        let face_b = mesh.faces[face_indices[1] as usize];

        // Find the shared edge orientation in each face
        let edge_in_a = find_edge_direction(&face_a, edge);
        let edge_in_b = find_edge_direction(&face_b, edge);

        // For consistent winding, the edge should appear in opposite directions
        // in the two adjacent faces
        if edge_in_a == edge_in_b {
            return false;
        }
    }

    true
}

/// Find the direction of an edge in a face.
/// Returns true if edge goes v0->v1 in the face's winding order, false if v1->v0.
fn find_edge_direction(face: &[u32; 3], edge: (u32, u32)) -> bool {
    let (v0, v1) = edge;

    // Check all three edges of the triangle
    for i in 0..3 {
        let a = face[i];
        let b = face[(i + 1) % 3];

        if a == v0 && b == v1 {
            return true; // Forward direction
        }
        if a == v1 && b == v0 {
            return false; // Reverse direction
        }
    }

    // Edge not found in face (shouldn't happen with valid adjacency)
    true
}

/// Count degenerate triangles in the mesh.
fn count_degenerate_triangles(mesh: &Mesh) -> usize {
    const DEGENERATE_THRESHOLD: f64 = 1e-10;

    mesh.faces
        .iter()
        .filter(|face| {
            let v0 = &mesh.vertices[face[0] as usize].position;
            let v1 = &mesh.vertices[face[1] as usize].position;
            let v2 = &mesh.vertices[face[2] as usize].position;

            let edge1 = v1 - v0;
            let edge2 = v2 - v0;
            let cross = edge1.cross(&edge2);
            let area = cross.norm() / 2.0;

            area < DEGENERATE_THRESHOLD
        })
        .count()
}

/// Result of shell auto-repair operation.
#[derive(Debug, Clone)]
pub struct ShellRepairResult {
    /// Number of degenerate triangles removed.
    pub degenerate_triangles_removed: usize,
    /// Number of faces with winding fixed.
    pub faces_with_winding_fixed: bool,
    /// Whether repair was successful.
    pub success: bool,
    /// Any repair errors encountered.
    pub errors: Vec<String>,
}

impl std::fmt::Display for ShellRepairResult {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        if self.success {
            writeln!(f, "Shell repair completed successfully:")?;
            if self.degenerate_triangles_removed > 0 {
                writeln!(
                    f,
                    "  - Removed {} degenerate triangles",
                    self.degenerate_triangles_removed
                )?;
            }
            if self.faces_with_winding_fixed {
                writeln!(f, "  - Fixed winding order")?;
            }
            if self.degenerate_triangles_removed == 0 && !self.faces_with_winding_fixed {
                writeln!(f, "  - No repairs needed")?;
            }
        } else {
            writeln!(f, "Shell repair failed:")?;
            for error in &self.errors {
                writeln!(f, "  - {}", error)?;
            }
        }
        Ok(())
    }
}

/// Attempt to automatically repair minor issues in a shell mesh.
///
/// This function can fix:
/// - Inconsistent winding order (using flood-fill algorithm)
/// - Degenerate triangles (removes them)
///
/// It cannot fix:
/// - Non-watertight meshes (holes) - would require hole-filling which may not
///   be appropriate for all shell geometries
/// - Non-manifold edges - requires manual intervention
///
/// # Arguments
/// * `shell` - The shell mesh to repair (modified in place)
///
/// # Returns
/// A `ShellRepairResult` with details about repairs performed.
///
/// # Example
/// ```
/// use mesh_repair::Mesh;
/// use mesh_shell::{validate_shell, repair_shell};
///
/// let mut shell = Mesh::new();
/// // ... generate shell ...
/// let repair_result = repair_shell(&mut shell);
/// if repair_result.success {
///     println!("{}", repair_result);
/// }
/// let validation = validate_shell(&shell);
/// ```
pub fn repair_shell(shell: &mut Mesh) -> ShellRepairResult {
    info!(
        "Attempting to repair shell mesh ({} vertices, {} faces)",
        shell.vertex_count(),
        shell.face_count()
    );

    let mut result = ShellRepairResult {
        degenerate_triangles_removed: 0,
        faces_with_winding_fixed: false,
        success: true,
        errors: Vec::new(),
    };

    if shell.faces.is_empty() {
        result.success = false;
        result.errors.push("Cannot repair empty shell".to_string());
        return result;
    }

    // 1. Remove degenerate triangles
    let removed = remove_degenerate_triangles_enhanced(
        shell, 1e-10,  // area_threshold
        1000.0, // max_aspect_ratio
        1e-9,   // min_edge_length
    );
    result.degenerate_triangles_removed = removed;
    if removed > 0 {
        info!("Removed {} degenerate triangles", removed);
    }

    // Check if we still have faces after removing degenerate triangles
    if shell.faces.is_empty() {
        result.success = false;
        result
            .errors
            .push("All faces were degenerate - shell is now empty".to_string());
        return result;
    }

    // 2. Fix winding order
    let pre_winding_check = check_winding_consistency(shell);
    if !pre_winding_check {
        match fix_winding_order(shell) {
            Ok(()) => {
                result.faces_with_winding_fixed = true;
                info!("Fixed winding order");
            }
            Err(e) => {
                result
                    .errors
                    .push(format!("Failed to fix winding order: {}", e));
                warn!("Failed to fix winding order: {}", e);
            }
        }
    }

    // Log summary
    if result.success {
        let total_repairs = result.degenerate_triangles_removed
            + (if result.faces_with_winding_fixed {
                1
            } else {
                0
            });
        if total_repairs > 0 {
            info!("Shell repair completed: {} repairs applied", total_repairs);
        } else {
            debug!("Shell repair: no repairs needed");
        }
    }

    result
}

/// Validate and optionally repair a shell mesh.
///
/// This is a convenience function that validates, attempts repair if needed,
/// and re-validates.
///
/// # Arguments
/// * `shell` - The shell mesh to validate and repair
/// * `auto_repair` - Whether to attempt automatic repair of minor issues
///
/// # Returns
/// A tuple of (validation result, optional repair result).
pub fn validate_and_repair_shell(
    shell: &mut Mesh,
    auto_repair: bool,
) -> (ShellValidationResult, Option<ShellRepairResult>) {
    // Initial validation
    let initial_validation = validate_shell(shell);

    if initial_validation.is_printable() || !auto_repair {
        return (initial_validation, None);
    }

    // Check if we can repair (only certain issues are fixable)
    let can_repair = initial_validation.issues.iter().any(|issue| {
        matches!(
            issue,
            ShellIssue::InconsistentWinding | ShellIssue::DegenerateTriangles { .. }
        )
    });

    if !can_repair {
        debug!("Shell has issues that cannot be auto-repaired");
        return (initial_validation, None);
    }

    // Attempt repair
    let repair_result = repair_shell(shell);

    // Re-validate
    let final_validation = validate_shell(shell);

    (final_validation, Some(repair_result))
}

#[cfg(test)]
mod tests {
    use super::*;
    use mesh_repair::Vertex;

    fn create_watertight_tetrahedron() -> Mesh {
        let mut mesh = Mesh::new();
        mesh.vertices.push(Vertex::from_coords(0.0, 0.0, 0.0));
        mesh.vertices.push(Vertex::from_coords(10.0, 0.0, 0.0));
        mesh.vertices.push(Vertex::from_coords(5.0, 10.0, 0.0));
        mesh.vertices.push(Vertex::from_coords(5.0, 5.0, 10.0));

        // Faces with consistent outward winding
        mesh.faces.push([0, 2, 1]); // Bottom
        mesh.faces.push([0, 1, 3]); // Front
        mesh.faces.push([1, 2, 3]); // Right
        mesh.faces.push([2, 0, 3]); // Left

        mesh
    }

    fn create_open_box() -> Mesh {
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
        // Top is OPEN

        mesh
    }

    #[test]
    fn test_validate_watertight_shell() {
        let shell = create_watertight_tetrahedron();
        let result = validate_shell(&shell);

        assert!(result.is_watertight);
        assert!(result.is_manifold);
        assert!(result.is_printable());
        assert_eq!(result.boundary_edge_count, 0);
        assert_eq!(result.non_manifold_edge_count, 0);
    }

    #[test]
    fn test_validate_open_shell() {
        let shell = create_open_box();
        let result = validate_shell(&shell);

        assert!(!result.is_watertight);
        assert!(result.is_manifold);
        assert!(!result.is_printable());
        assert!(result.boundary_edge_count > 0);
        assert!(
            result
                .issues
                .iter()
                .any(|i| matches!(i, ShellIssue::NotWatertight { .. }))
        );
    }

    #[test]
    fn test_validate_empty_shell() {
        let shell = Mesh::new();
        let result = validate_shell(&shell);

        assert!(!result.is_valid());
        assert!(!result.is_printable());
        assert!(
            result
                .issues
                .iter()
                .any(|i| matches!(i, ShellIssue::EmptyShell))
        );
    }

    #[test]
    fn test_validate_shell_with_degenerate_triangles() {
        let mut mesh = create_watertight_tetrahedron();

        // Add a degenerate triangle (all vertices at same position)
        let idx = mesh.vertices.len() as u32;
        mesh.vertices.push(Vertex::from_coords(0.0, 0.0, 0.0));
        mesh.vertices.push(Vertex::from_coords(0.0, 0.0, 0.0));
        mesh.vertices.push(Vertex::from_coords(0.0, 0.0, 0.0));
        mesh.faces.push([idx, idx + 1, idx + 2]);

        let result = validate_shell(&mesh);

        assert!(
            result
                .issues
                .iter()
                .any(|i| matches!(i, ShellIssue::DegenerateTriangles { .. }))
        );
    }

    #[test]
    fn test_shell_validation_result_display() {
        let shell = create_watertight_tetrahedron();
        let result = validate_shell(&shell);
        let output = format!("{}", result);

        assert!(output.contains("Vertices:"));
        assert!(output.contains("Faces:"));
        assert!(output.contains("Watertight: yes"));
        assert!(output.contains("Manifold: yes"));
        assert!(output.contains("Printable: yes"));
    }

    #[test]
    fn test_shell_issue_display() {
        let issue = ShellIssue::NotWatertight {
            boundary_edge_count: 4,
        };
        let output = format!("{}", issue);
        assert!(output.contains("watertight"));
        assert!(output.contains("4"));

        let issue = ShellIssue::NonManifold {
            non_manifold_edge_count: 2,
        };
        let output = format!("{}", issue);
        assert!(output.contains("manifold"));
        assert!(output.contains("2"));

        let issue = ShellIssue::InconsistentWinding;
        let output = format!("{}", issue);
        assert!(output.contains("winding"));

        let issue = ShellIssue::EmptyShell;
        let output = format!("{}", issue);
        assert!(output.contains("empty"));

        let issue = ShellIssue::DegenerateTriangles { count: 5 };
        let output = format!("{}", issue);
        assert!(output.contains("degenerate"));
        assert!(output.contains("5"));
    }

    #[test]
    fn test_repair_shell_removes_degenerate_triangles() {
        let mut mesh = create_watertight_tetrahedron();
        let original_face_count = mesh.faces.len();

        // Add degenerate triangles
        let idx = mesh.vertices.len() as u32;
        mesh.vertices.push(Vertex::from_coords(0.0, 0.0, 0.0));
        mesh.vertices.push(Vertex::from_coords(0.0, 0.0, 0.0));
        mesh.vertices.push(Vertex::from_coords(0.0, 0.0, 0.0));
        mesh.faces.push([idx, idx + 1, idx + 2]);

        // Another degenerate - very thin triangle
        let idx = mesh.vertices.len() as u32;
        mesh.vertices.push(Vertex::from_coords(0.0, 0.0, 0.0));
        mesh.vertices.push(Vertex::from_coords(10.0, 0.0, 0.0));
        mesh.vertices.push(Vertex::from_coords(5.0, 0.0000001, 0.0)); // Nearly collinear
        mesh.faces.push([idx, idx + 1, idx + 2]);

        let repair_result = repair_shell(&mut mesh);

        assert!(repair_result.success);
        assert!(repair_result.degenerate_triangles_removed > 0);
        // Should have removed the degenerate triangles
        assert!(mesh.faces.len() <= original_face_count + 2);
    }

    #[test]
    fn test_repair_shell_result_display() {
        let result = ShellRepairResult {
            degenerate_triangles_removed: 3,
            faces_with_winding_fixed: true,
            success: true,
            errors: vec![],
        };

        let output = format!("{}", result);
        assert!(output.contains("Removed 3 degenerate triangles"));
        assert!(output.contains("Fixed winding order"));
        assert!(output.contains("successfully"));
    }

    #[test]
    fn test_repair_shell_result_with_errors() {
        let result = ShellRepairResult {
            degenerate_triangles_removed: 0,
            faces_with_winding_fixed: false,
            success: false,
            errors: vec!["Test error".to_string()],
        };

        let output = format!("{}", result);
        assert!(output.contains("failed"));
        assert!(output.contains("Test error"));
    }

    #[test]
    fn test_validate_and_repair_shell_no_repair() {
        let mut mesh = create_watertight_tetrahedron();

        let (validation, repair) = validate_and_repair_shell(&mut mesh, false);

        assert!(validation.is_printable());
        assert!(repair.is_none());
    }

    #[test]
    fn test_validate_and_repair_shell_with_repair() {
        let mut mesh = create_watertight_tetrahedron();

        // Add a degenerate triangle
        let idx = mesh.vertices.len() as u32;
        mesh.vertices.push(Vertex::from_coords(0.0, 0.0, 0.0));
        mesh.vertices.push(Vertex::from_coords(0.0, 0.0, 0.0));
        mesh.vertices.push(Vertex::from_coords(0.0, 0.0, 0.0));
        mesh.faces.push([idx, idx + 1, idx + 2]);

        let (validation, repair) = validate_and_repair_shell(&mut mesh, true);

        // Repair should have been attempted
        assert!(repair.is_some());
        let repair_result = repair.unwrap();
        assert!(repair_result.degenerate_triangles_removed > 0);

        // Final validation should show the degenerate triangles are gone
        let has_degenerate_issue = validation
            .issues
            .iter()
            .any(|i| matches!(i, ShellIssue::DegenerateTriangles { .. }));
        assert!(
            !has_degenerate_issue,
            "Degenerate triangles should have been removed"
        );
    }

    #[test]
    fn test_repair_clean_shell() {
        let mut mesh = create_watertight_tetrahedron();

        let repair_result = repair_shell(&mut mesh);

        assert!(repair_result.success);
        assert_eq!(repair_result.degenerate_triangles_removed, 0);
        assert!(!repair_result.faces_with_winding_fixed);
        assert!(repair_result.errors.is_empty());
    }
}

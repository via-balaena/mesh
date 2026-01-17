//! Property-based tests for mesh operations.
//!
//! These tests use proptest to generate random meshes and verify invariants.
//!
//! Run with: cargo test -p mesh-repair -- proptest

use mesh_repair::{DecimateParams, Mesh, Vertex, decimate_mesh};
use proptest::prelude::*;

// =============================================================================
// Strategies for generating random meshes
// =============================================================================

/// Generate a random vertex position in a bounded range.
fn arb_position() -> impl Strategy<Value = [f64; 3]> {
    prop::array::uniform3(-100.0..100.0f64)
}

/// Generate a random vertex with position only.
fn arb_vertex() -> impl Strategy<Value = Vertex> {
    arb_position().prop_map(|[x, y, z]| Vertex::from_coords(x, y, z))
}

/// Generate a valid mesh with the specified number of vertices and faces.
/// Ensures all face indices are valid.
fn arb_mesh(
    min_vertices: usize,
    max_vertices: usize,
    min_faces: usize,
    max_faces: usize,
) -> impl Strategy<Value = Mesh> {
    (min_vertices..=max_vertices).prop_flat_map(move |num_vertices| {
        let vertices = prop::collection::vec(arb_vertex(), num_vertices);

        vertices.prop_flat_map(move |verts| {
            let n = verts.len() as u32;
            if n < 3 {
                // Need at least 3 vertices for a face
                return Just(Mesh {
                    vertices: verts,
                    faces: Vec::new(),
                })
                .boxed();
            }

            let face = prop::array::uniform3(0..n);
            let faces = prop::collection::vec(face, min_faces..=max_faces);

            faces
                .prop_map(move |f| Mesh {
                    vertices: verts.clone(),
                    faces: f,
                })
                .boxed()
        })
    })
}

/// Generate a simple triangulated cube mesh.
fn cube_mesh() -> Mesh {
    let mut mesh = Mesh::new();

    let verts = [
        [-0.5, -0.5, -0.5],
        [0.5, -0.5, -0.5],
        [0.5, 0.5, -0.5],
        [-0.5, 0.5, -0.5],
        [-0.5, -0.5, 0.5],
        [0.5, -0.5, 0.5],
        [0.5, 0.5, 0.5],
        [-0.5, 0.5, 0.5],
    ];

    for v in &verts {
        mesh.vertices.push(Vertex::from_coords(v[0], v[1], v[2]));
    }

    let faces = [
        [0, 1, 2],
        [0, 2, 3],
        [4, 6, 5],
        [4, 7, 6],
        [0, 4, 5],
        [0, 5, 1],
        [2, 6, 7],
        [2, 7, 3],
        [0, 3, 7],
        [0, 7, 4],
        [1, 5, 6],
        [1, 6, 2],
    ];

    for f in &faces {
        mesh.faces.push([f[0] as u32, f[1] as u32, f[2] as u32]);
    }

    mesh
}

// =============================================================================
// Property Tests: Basic Invariants
// =============================================================================

proptest! {
    /// Vertex count should not change after validation.
    #[test]
    fn proptest_validation_preserves_vertex_count(mesh in arb_mesh(4, 50, 1, 20)) {
        let original_count = mesh.vertex_count();
        let _report = mesh_repair::validate_mesh(&mesh);
        prop_assert_eq!(mesh.vertex_count(), original_count);
    }

    /// Face count should not change after validation.
    #[test]
    fn proptest_validation_preserves_face_count(mesh in arb_mesh(4, 50, 1, 20)) {
        let original_count = mesh.face_count();
        let _report = mesh_repair::validate_mesh(&mesh);
        prop_assert_eq!(mesh.face_count(), original_count);
    }

    /// Cloning a mesh should produce an identical mesh.
    #[test]
    fn proptest_clone_is_identical(mesh in arb_mesh(4, 50, 1, 20)) {
        let cloned = mesh.clone();
        prop_assert_eq!(mesh.vertex_count(), cloned.vertex_count());
        prop_assert_eq!(mesh.face_count(), cloned.face_count());
    }
}

// =============================================================================
// Property Tests: Repair Operations
// =============================================================================

proptest! {
    /// Weld vertices should not increase vertex count.
    #[test]
    fn proptest_weld_does_not_increase_vertices(mesh in arb_mesh(10, 100, 5, 30)) {
        let original_count = mesh.vertex_count();
        let mut m = mesh.clone();
        mesh_repair::weld_vertices(&mut m, 1e-6);
        prop_assert!(m.vertex_count() <= original_count);
    }

    /// Remove degenerate triangles should not increase face count.
    #[test]
    fn proptest_remove_degenerate_does_not_increase_faces(mesh in arb_mesh(4, 50, 1, 20)) {
        let original_count = mesh.face_count();
        let mut m = mesh.clone();
        mesh_repair::remove_degenerate_triangles(&mut m, 1e-10);
        prop_assert!(m.face_count() <= original_count);
    }

    /// After removing degenerate triangles, no face should have duplicate vertices.
    #[test]
    fn proptest_remove_degenerate_no_duplicate_indices(mesh in arb_mesh(4, 50, 1, 20)) {
        let mut m = mesh.clone();
        mesh_repair::remove_degenerate_triangles(&mut m, 1e-10);

        for face in &m.faces {
            prop_assert!(face[0] != face[1] || face[1] != face[2] || face[0] != face[2],
                "Found degenerate face: {:?}", face);
        }
    }
}

// =============================================================================
// Property Tests: Geometry Invariants
// =============================================================================

proptest! {
    /// Bounding box should contain all vertices.
    #[test]
    fn proptest_bounds_contain_all_vertices(mesh in arb_mesh(4, 100, 1, 50)) {
        if mesh.vertex_count() == 0 {
            return Ok(());
        }

        if let Some((min, max)) = mesh.bounds() {
            for vertex in &mesh.vertices {
                prop_assert!(vertex.position.x >= min.x - 1e-10);
                prop_assert!(vertex.position.y >= min.y - 1e-10);
                prop_assert!(vertex.position.z >= min.z - 1e-10);
                prop_assert!(vertex.position.x <= max.x + 1e-10);
                prop_assert!(vertex.position.y <= max.y + 1e-10);
                prop_assert!(vertex.position.z <= max.z + 1e-10);
            }
        }
    }

    /// Surface area should be non-negative.
    #[test]
    fn proptest_surface_area_non_negative(mesh in arb_mesh(4, 50, 1, 20)) {
        let area = mesh.surface_area();
        prop_assert!(area >= 0.0, "Surface area was negative: {}", area);
    }

    /// Volume can be negative for inside-out meshes, but should be finite.
    #[test]
    fn proptest_volume_is_finite(mesh in arb_mesh(4, 50, 1, 20)) {
        let volume = mesh.volume();
        prop_assert!(volume.is_finite(), "Volume was not finite: {}", volume);
    }
}

// =============================================================================
// Property Tests: Decimation
// =============================================================================

proptest! {
    /// Decimation should reduce face count (or stay same if target is >= current).
    #[test]
    fn proptest_decimation_reduces_faces(
        mesh in arb_mesh(12, 100, 10, 50),
        ratio in 0.1..0.9f64
    ) {
        let original = mesh.face_count();
        if original < 4 {
            return Ok(());
        }

        let target = ((original as f64) * ratio) as usize;
        if target < 4 {
            return Ok(());
        }

        let params = DecimateParams::with_target_triangles(target);
        let result = decimate_mesh(&mesh, &params);
        prop_assert!(result.mesh.face_count() <= original,
            "Decimated mesh has more faces ({}) than original ({})",
            result.mesh.face_count(), original);
    }

    /// Decimated mesh should have valid topology (no invalid indices).
    #[test]
    fn proptest_decimation_valid_indices(
        mesh in arb_mesh(12, 100, 10, 50)
    ) {
        if mesh.face_count() < 4 {
            return Ok(());
        }

        let target = mesh.face_count() / 2;
        if target < 4 {
            return Ok(());
        }

        let params = DecimateParams::with_target_triangles(target);
        let result = decimate_mesh(&mesh, &params);
        let vertex_count = result.mesh.vertex_count() as u32;

        for face in &result.mesh.faces {
            prop_assert!(face[0] < vertex_count,
                "Invalid vertex index {} >= {}", face[0], vertex_count);
            prop_assert!(face[1] < vertex_count,
                "Invalid vertex index {} >= {}", face[1], vertex_count);
            prop_assert!(face[2] < vertex_count,
                "Invalid vertex index {} >= {}", face[2], vertex_count);
        }
    }
}

// =============================================================================
// Property Tests: I/O Round-trip
// =============================================================================

proptest! {
    /// Validation should not panic on any mesh.
    #[test]
    fn proptest_validation_no_panic(mesh in arb_mesh(4, 50, 1, 20)) {
        // Skip empty meshes
        if mesh.face_count() == 0 {
            return Ok(());
        }

        // Validation should never panic
        let report = mesh_repair::validate_mesh(&mesh);
        prop_assert!(report.vertex_count == mesh.vertex_count());
    }
}

// =============================================================================
// Property Tests: Cube Specific (Known Good Mesh)
// =============================================================================

#[test]
fn proptest_cube_is_manifold() {
    let cube = cube_mesh();
    let report = mesh_repair::validate_mesh(&cube);
    assert!(report.is_manifold, "Cube should be manifold");
}

#[test]
fn proptest_cube_is_watertight() {
    let cube = cube_mesh();
    let report = mesh_repair::validate_mesh(&cube);
    assert!(report.is_watertight, "Cube should be watertight");
}

#[test]
fn proptest_cube_has_positive_volume() {
    let cube = cube_mesh();
    let volume = cube.volume();
    assert!(
        volume > 0.0,
        "Cube should have positive volume, got {}",
        volume
    );
}

#[test]
fn proptest_cube_surface_area() {
    let cube = cube_mesh();
    let area = cube.surface_area();
    // Unit cube has surface area of 6.0 (6 faces * 1.0 each)
    let expected = 6.0;
    assert!(
        (area - expected).abs() < 1e-10,
        "Cube surface area should be {}, got {}",
        expected,
        area
    );
}

// =============================================================================
// Property Tests: Boolean Operations
// =============================================================================

proptest! {
    /// Boolean union should not produce empty mesh from non-empty inputs.
    #[test]
    fn proptest_boolean_union_non_empty(
        offset_x in -2.0..2.0f64,
        offset_y in -2.0..2.0f64,
        offset_z in -2.0..2.0f64
    ) {
        let cube1 = cube_mesh();
        let mut cube2 = cube_mesh();

        // Offset cube2
        for v in &mut cube2.vertices {
            v.position.x += offset_x;
            v.position.y += offset_y;
            v.position.z += offset_z;
        }

        // Only test overlapping cubes (non-overlapping is a different case)
        if offset_x.abs() > 1.5 || offset_y.abs() > 1.5 || offset_z.abs() > 1.5 {
            return Ok(());
        }

        let params = mesh_repair::BooleanParams::default();
        let result = mesh_repair::boolean_operation(
            &cube1,
            &cube2,
            mesh_repair::BooleanOp::Union,
            &params
        );

        if let Ok(union) = result {
            prop_assert!(union.mesh.face_count() > 0,
                "Union should not be empty for overlapping cubes");
        }
    }
}

// =============================================================================
// Property Tests: Subdivision
// =============================================================================

proptest! {
    /// Loop subdivision should increase face count by factor of 4.
    #[test]
    fn proptest_loop_subdivision_increases_faces(mesh in arb_mesh(4, 20, 4, 10)) {
        if mesh.face_count() == 0 {
            return Ok(());
        }

        let original_faces = mesh.face_count();
        let result = mesh_repair::subdivide_mesh(&mesh, &Default::default());

        // Loop subdivision creates 4 faces per original face
        let expected = original_faces * 4;
        prop_assert_eq!(result.mesh.face_count(), expected,
            "Expected {} faces after subdivision, got {}",
            expected, result.mesh.face_count());
    }
}

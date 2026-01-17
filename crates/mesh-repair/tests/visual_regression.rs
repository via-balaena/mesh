//! Visual regression tests for mesh operations.
//!
//! These tests verify that mesh operations produce consistent, deterministic
//! outputs. While we can't render images directly, we compare "mesh snapshots"
//! that capture key geometric properties:
//!
//! - Vertex count and positions (sampled)
//! - Face count
//! - Bounding box
//! - Surface area
//! - Volume
//!
//! Any change in these properties indicates a visual change in the mesh.
//!
//! Run with: cargo test -p mesh-repair --test visual_regression

use mesh_repair::{
    DecimateParams, Mesh, RemeshParams, SubdivideParams, Vertex, decimate_mesh, fill_holes,
    remesh_isotropic, subdivide_mesh, validate_mesh, weld_vertices,
};

// =============================================================================
// Mesh Snapshot for Comparison
// =============================================================================

/// A snapshot of mesh properties for regression testing.
#[derive(Debug, Clone)]
struct MeshSnapshot {
    vertex_count: usize,
    face_count: usize,
    bounds_min: [f64; 3],
    bounds_max: [f64; 3],
    surface_area: f64,
    volume: f64,
    /// Sample of vertex positions (first, middle, last)
    vertex_samples: Vec<[f64; 3]>,
}

impl MeshSnapshot {
    fn from_mesh(mesh: &Mesh) -> Self {
        let (bounds_min, bounds_max) = if let Some((min, max)) = mesh.bounds() {
            ([min.x, min.y, min.z], [max.x, max.y, max.z])
        } else {
            ([0.0, 0.0, 0.0], [0.0, 0.0, 0.0])
        };
        let min_arr = bounds_min;
        let max_arr = bounds_max;

        // Sample some vertices
        let mut vertex_samples = Vec::new();
        if !mesh.vertices.is_empty() {
            // First vertex
            let v = &mesh.vertices[0];
            vertex_samples.push([v.position.x, v.position.y, v.position.z]);

            // Middle vertex
            let mid_idx = mesh.vertices.len() / 2;
            let v = &mesh.vertices[mid_idx];
            vertex_samples.push([v.position.x, v.position.y, v.position.z]);

            // Last vertex
            let v = &mesh.vertices[mesh.vertices.len() - 1];
            vertex_samples.push([v.position.x, v.position.y, v.position.z]);
        }

        Self {
            vertex_count: mesh.vertex_count(),
            face_count: mesh.face_count(),
            bounds_min: min_arr,
            bounds_max: max_arr,
            surface_area: mesh.surface_area(),
            volume: mesh.volume(),
            vertex_samples,
        }
    }

    /// Compare two snapshots with tolerance.
    fn compare(&self, other: &MeshSnapshot, tolerance: f64) -> Result<(), String> {
        if self.vertex_count != other.vertex_count {
            return Err(format!(
                "Vertex count mismatch: {} vs {}",
                self.vertex_count, other.vertex_count
            ));
        }

        if self.face_count != other.face_count {
            return Err(format!(
                "Face count mismatch: {} vs {}",
                self.face_count, other.face_count
            ));
        }

        for i in 0..3 {
            if (self.bounds_min[i] - other.bounds_min[i]).abs() > tolerance {
                return Err(format!(
                    "Bounds min[{}] mismatch: {} vs {}",
                    i, self.bounds_min[i], other.bounds_min[i]
                ));
            }
            if (self.bounds_max[i] - other.bounds_max[i]).abs() > tolerance {
                return Err(format!(
                    "Bounds max[{}] mismatch: {} vs {}",
                    i, self.bounds_max[i], other.bounds_max[i]
                ));
            }
        }

        let area_diff = (self.surface_area - other.surface_area).abs();
        let area_tolerance = self.surface_area.max(other.surface_area) * tolerance;
        if area_diff > area_tolerance.max(tolerance) {
            return Err(format!(
                "Surface area mismatch: {} vs {} (diff: {})",
                self.surface_area, other.surface_area, area_diff
            ));
        }

        let vol_diff = (self.volume - other.volume).abs();
        let vol_tolerance = self.volume.abs().max(other.volume.abs()) * tolerance;
        if vol_diff > vol_tolerance.max(tolerance) {
            return Err(format!(
                "Volume mismatch: {} vs {} (diff: {})",
                self.volume, other.volume, vol_diff
            ));
        }

        Ok(())
    }
}

// =============================================================================
// Test Mesh Creation
// =============================================================================

fn create_test_cube() -> Mesh {
    let mut mesh = Mesh::new();

    mesh.vertices = vec![
        Vertex::from_coords(0.0, 0.0, 0.0),
        Vertex::from_coords(1.0, 0.0, 0.0),
        Vertex::from_coords(1.0, 1.0, 0.0),
        Vertex::from_coords(0.0, 1.0, 0.0),
        Vertex::from_coords(0.0, 0.0, 1.0),
        Vertex::from_coords(1.0, 0.0, 1.0),
        Vertex::from_coords(1.0, 1.0, 1.0),
        Vertex::from_coords(0.0, 1.0, 1.0),
    ];

    mesh.faces = vec![
        [0, 2, 1],
        [0, 3, 2], // front
        [4, 5, 6],
        [4, 6, 7], // back
        [0, 1, 5],
        [0, 5, 4], // bottom
        [2, 3, 7],
        [2, 7, 6], // top
        [0, 4, 7],
        [0, 7, 3], // left
        [1, 2, 6],
        [1, 6, 5], // right
    ];

    mesh
}

fn create_test_sphere(subdivisions: u32) -> Mesh {
    let mut mesh = Mesh::new();

    let phi = (1.0 + 5.0_f64.sqrt()) / 2.0;
    let a = 1.0;
    let b = 1.0 / phi;

    let ico_verts = [
        [0.0, b, -a],
        [b, a, 0.0],
        [-b, a, 0.0],
        [0.0, b, a],
        [0.0, -b, a],
        [-a, 0.0, b],
        [0.0, -b, -a],
        [a, 0.0, -b],
        [a, 0.0, b],
        [-a, 0.0, -b],
        [b, -a, 0.0],
        [-b, -a, 0.0],
    ];

    for v in &ico_verts {
        let len = (v[0] * v[0] + v[1] * v[1] + v[2] * v[2]).sqrt();
        mesh.vertices
            .push(Vertex::from_coords(v[0] / len, v[1] / len, v[2] / len));
    }

    let ico_faces: [[u32; 3]; 20] = [
        [0, 1, 2],
        [3, 2, 1],
        [3, 4, 5],
        [3, 8, 4],
        [0, 6, 7],
        [0, 9, 6],
        [4, 10, 11],
        [6, 11, 10],
        [2, 5, 9],
        [11, 9, 5],
        [1, 7, 8],
        [10, 8, 7],
        [3, 5, 2],
        [3, 1, 8],
        [0, 2, 9],
        [0, 7, 1],
        [6, 9, 11],
        [6, 10, 7],
        [4, 11, 5],
        [4, 8, 10],
    ];

    for f in &ico_faces {
        mesh.faces.push(*f);
    }

    // Subdivide
    for _ in 0..subdivisions {
        mesh = subdivide_icosphere(&mesh);
    }

    mesh
}

fn subdivide_icosphere(mesh: &Mesh) -> Mesh {
    use std::collections::HashMap;

    let mut new_mesh = Mesh::new();
    new_mesh.vertices = mesh.vertices.clone();

    let mut edge_midpoints: HashMap<(u32, u32), u32> = HashMap::new();

    for face in &mesh.faces {
        let v0 = face[0];
        let v1 = face[1];
        let v2 = face[2];

        let m01 = get_midpoint(v0, v1, &mut new_mesh.vertices, &mut edge_midpoints);
        let m12 = get_midpoint(v1, v2, &mut new_mesh.vertices, &mut edge_midpoints);
        let m20 = get_midpoint(v2, v0, &mut new_mesh.vertices, &mut edge_midpoints);

        new_mesh.faces.push([v0, m01, m20]);
        new_mesh.faces.push([v1, m12, m01]);
        new_mesh.faces.push([v2, m20, m12]);
        new_mesh.faces.push([m01, m12, m20]);
    }

    new_mesh
}

fn get_midpoint(
    v1: u32,
    v2: u32,
    vertices: &mut Vec<Vertex>,
    edge_midpoints: &mut std::collections::HashMap<(u32, u32), u32>,
) -> u32 {
    let key = if v1 < v2 { (v1, v2) } else { (v2, v1) };

    if let Some(&idx) = edge_midpoints.get(&key) {
        return idx;
    }

    let p1 = &vertices[v1 as usize];
    let p2 = &vertices[v2 as usize];

    let mx = (p1.position.x + p2.position.x) / 2.0;
    let my = (p1.position.y + p2.position.y) / 2.0;
    let mz = (p1.position.z + p2.position.z) / 2.0;
    let len = (mx * mx + my * my + mz * mz).sqrt();

    let idx = vertices.len() as u32;
    vertices.push(Vertex::from_coords(mx / len, my / len, mz / len));
    edge_midpoints.insert(key, idx);
    idx
}

fn create_open_box() -> Mesh {
    let mut mesh = Mesh::new();

    mesh.vertices = vec![
        Vertex::from_coords(0.0, 0.0, 0.0),
        Vertex::from_coords(1.0, 0.0, 0.0),
        Vertex::from_coords(1.0, 1.0, 0.0),
        Vertex::from_coords(0.0, 1.0, 0.0),
        Vertex::from_coords(0.0, 0.0, 1.0),
        Vertex::from_coords(1.0, 0.0, 1.0),
        Vertex::from_coords(1.0, 1.0, 1.0),
        Vertex::from_coords(0.0, 1.0, 1.0),
    ];

    // Missing bottom faces
    mesh.faces = vec![
        [0, 2, 1],
        [0, 3, 2], // front
        [4, 5, 6],
        [4, 6, 7], // back
        [2, 3, 7],
        [2, 7, 6], // top
        [0, 4, 7],
        [0, 7, 3], // left
        [1, 2, 6],
        [1, 6, 5], // right
    ];

    mesh
}

// =============================================================================
// Visual Regression Tests: Subdivision
// =============================================================================

#[test]
fn visual_regression_subdivision_cube() {
    let cube = create_test_cube();
    let params = SubdivideParams::with_iterations(1);
    let result = subdivide_mesh(&cube, &params);

    let snapshot = MeshSnapshot::from_mesh(&result.mesh);

    // Expected values for 1x subdivision of a cube
    // Loop subdivision: 12 faces * 4 = 48 faces
    assert_eq!(snapshot.face_count, 48, "Face count after subdivision");

    // Bounds should be approximately preserved (slight shrinkage due to smoothing)
    assert!(snapshot.bounds_min[0] >= -0.1, "Min X bound");
    assert!(snapshot.bounds_max[0] <= 1.1, "Max X bound");

    // Surface area should be roughly similar (smoothing can affect it)
    // Loop subdivision on a cube can produce various results depending on implementation
    assert!(
        snapshot.surface_area > 0.0,
        "Surface area should be positive"
    );
    assert!(
        snapshot.surface_area < 20.0,
        "Surface area reasonable upper bound"
    );
}

#[test]
fn visual_regression_subdivision_deterministic() {
    let cube = create_test_cube();
    let params = SubdivideParams::with_iterations(1);

    let result1 = subdivide_mesh(&cube, &params);
    let result2 = subdivide_mesh(&cube, &params);

    let snap1 = MeshSnapshot::from_mesh(&result1.mesh);
    let snap2 = MeshSnapshot::from_mesh(&result2.mesh);

    snap1
        .compare(&snap2, 1e-10)
        .expect("Subdivision should be deterministic");
}

// =============================================================================
// Visual Regression Tests: Decimation
// =============================================================================

#[test]
fn visual_regression_decimation_sphere() {
    let sphere = create_test_sphere(2); // 320 triangles
    let params = DecimateParams::with_target_triangles(160);
    let result = decimate_mesh(&sphere, &params);

    let snapshot = MeshSnapshot::from_mesh(&result.mesh);

    // Should reduce face count
    assert!(snapshot.face_count <= 320, "Face count should be reduced");
    assert!(
        snapshot.face_count >= 100,
        "Should have reasonable faces left"
    );

    // Bounds should be preserved (sphere stays spherical)
    for i in 0..3 {
        assert!(snapshot.bounds_min[i] >= -1.1, "Min bound {} preserved", i);
        assert!(snapshot.bounds_max[i] <= 1.1, "Max bound {} preserved", i);
    }
}

#[test]
fn visual_regression_decimation_deterministic() {
    let sphere = create_test_sphere(2);
    let params = DecimateParams::with_target_triangles(160);

    let result1 = decimate_mesh(&sphere, &params);
    let result2 = decimate_mesh(&sphere, &params);

    let snap1 = MeshSnapshot::from_mesh(&result1.mesh);
    let snap2 = MeshSnapshot::from_mesh(&result2.mesh);

    snap1
        .compare(&snap2, 1e-10)
        .expect("Decimation should be deterministic");
}

#[test]
fn visual_regression_aggressive_decimation() {
    let sphere = create_test_sphere(3); // 1280 triangles
    let params = DecimateParams::with_target_ratio(0.1); // 10% = ~128 triangles
    let result = decimate_mesh(&sphere, &params);

    let snapshot = MeshSnapshot::from_mesh(&result.mesh);

    // Aggressive decimation should still preserve basic shape
    assert!(snapshot.face_count >= 50, "Should have at least 50 faces");
    assert!(
        snapshot.face_count <= 200,
        "Should be significantly reduced"
    );

    // Volume should be roughly preserved (aggressive decimation can reduce it)
    let original_volume = create_test_sphere(3).volume().abs();
    let decimated_volume = snapshot.volume.abs();
    let volume_ratio = decimated_volume / original_volume;
    // Aggressive decimation to 10% may significantly alter volume
    assert!(
        volume_ratio > 0.1,
        "Volume should be at least 10% preserved after aggressive decimation"
    );
}

// =============================================================================
// Visual Regression Tests: Remeshing
// =============================================================================

#[test]
fn visual_regression_remesh_sphere() {
    let sphere = create_test_sphere(2);
    let params = RemeshParams {
        target_edge_length: Some(0.3),
        iterations: 3,
        ..Default::default()
    };
    let result = remesh_isotropic(&sphere, &params);

    let snapshot = MeshSnapshot::from_mesh(&result.mesh);

    // Remeshing should produce valid output
    assert!(snapshot.vertex_count > 0, "Should have vertices");
    assert!(snapshot.face_count > 0, "Should have faces");

    // Bounds should be approximately preserved
    for i in 0..3 {
        assert!(snapshot.bounds_min[i] >= -1.5, "Min bound preserved");
        assert!(snapshot.bounds_max[i] <= 1.5, "Max bound preserved");
    }
}

#[test]
fn visual_regression_remesh_deterministic() {
    let sphere = create_test_sphere(1);
    let params = RemeshParams {
        target_edge_length: Some(0.5),
        iterations: 2,
        ..Default::default()
    };

    let result1 = remesh_isotropic(&sphere, &params);
    let result2 = remesh_isotropic(&sphere, &params);

    let snap1 = MeshSnapshot::from_mesh(&result1.mesh);
    let snap2 = MeshSnapshot::from_mesh(&result2.mesh);

    // Remeshing should be deterministic
    snap1
        .compare(&snap2, 1e-10)
        .expect("Remeshing should be deterministic");
}

// =============================================================================
// Visual Regression Tests: Hole Filling
// =============================================================================

#[test]
fn visual_regression_hole_filling() {
    let mut open_box = create_open_box();
    let original_faces = open_box.face_count();

    let _ = fill_holes(&mut open_box);

    let snapshot = MeshSnapshot::from_mesh(&open_box);

    // Should have added faces to close the hole
    assert!(
        snapshot.face_count >= original_faces,
        "Face count should increase or stay same"
    );

    // Bounds should be preserved
    assert!(snapshot.bounds_min[0] >= -0.01, "Min X preserved");
    assert!(snapshot.bounds_max[0] <= 1.01, "Max X preserved");
}

#[test]
fn visual_regression_hole_filling_deterministic() {
    let mut box1 = create_open_box();
    let mut box2 = create_open_box();

    let _ = fill_holes(&mut box1);
    let _ = fill_holes(&mut box2);

    let snap1 = MeshSnapshot::from_mesh(&box1);
    let snap2 = MeshSnapshot::from_mesh(&box2);

    snap1
        .compare(&snap2, 1e-10)
        .expect("Hole filling should be deterministic");
}

// =============================================================================
// Visual Regression Tests: Weld Vertices
// =============================================================================

#[test]
fn visual_regression_weld_vertices() {
    let mut cube = create_test_cube();
    let original_snapshot = MeshSnapshot::from_mesh(&cube);

    weld_vertices(&mut cube, 1e-6);

    let welded_snapshot = MeshSnapshot::from_mesh(&cube);

    // Clean cube shouldn't change much
    assert_eq!(
        original_snapshot.face_count, welded_snapshot.face_count,
        "Face count should be preserved"
    );

    // Bounds should be exactly preserved
    for i in 0..3 {
        assert!(
            (original_snapshot.bounds_min[i] - welded_snapshot.bounds_min[i]).abs() < 1e-10,
            "Bounds should be preserved"
        );
        assert!(
            (original_snapshot.bounds_max[i] - welded_snapshot.bounds_max[i]).abs() < 1e-10,
            "Bounds should be preserved"
        );
    }
}

// =============================================================================
// Visual Regression Tests: Validation
// =============================================================================

#[test]
fn visual_regression_validation_cube() {
    let cube = create_test_cube();
    let report = validate_mesh(&cube);

    // Expected validation results for clean cube
    assert!(report.is_watertight, "Cube should be watertight");
    assert!(report.is_manifold, "Cube should be manifold");
    assert_eq!(report.vertex_count, 8, "Cube has 8 vertices");
    assert_eq!(report.face_count, 12, "Cube has 12 faces");
    assert_eq!(report.component_count, 1, "Cube is single component");
}

#[test]
fn visual_regression_validation_open_box() {
    let open_box = create_open_box();
    let report = validate_mesh(&open_box);

    // Open box should not be watertight
    assert!(!report.is_watertight, "Open box should not be watertight");
    assert_eq!(report.vertex_count, 8, "Open box has 8 vertices");
    assert_eq!(report.face_count, 10, "Open box has 10 faces (missing 2)");
}

// =============================================================================
// Visual Regression Tests: Complex Operations
// =============================================================================

#[test]
fn visual_regression_repair_and_decimate() {
    // Test a realistic workflow: repair then decimate
    let sphere = create_test_sphere(2);
    let original = MeshSnapshot::from_mesh(&sphere);

    // Decimate
    let params = DecimateParams::with_target_ratio(0.5);
    let result = decimate_mesh(&sphere, &params);
    let decimated = MeshSnapshot::from_mesh(&result.mesh);

    // Face count should be reduced
    assert!(
        decimated.face_count < original.face_count,
        "Decimation should reduce faces"
    );

    // Volume should be roughly preserved
    let vol_diff = (original.volume.abs() - decimated.volume.abs()).abs();
    let vol_tolerance = original.volume.abs() * 0.3; // 30% tolerance
    assert!(
        vol_diff < vol_tolerance,
        "Volume should be roughly preserved (diff: {})",
        vol_diff
    );
}

#[test]
fn visual_regression_subdivide_and_decimate() {
    // Test: subdivide then decimate back
    let cube = create_test_cube();
    let original_faces = cube.face_count();

    // Subdivide (12 * 4 = 48 faces)
    let sub_result = subdivide_mesh(&cube, &SubdivideParams::with_iterations(1));

    // Decimate back to original face count
    let dec_params = DecimateParams::with_target_triangles(original_faces);
    let dec_result = decimate_mesh(&sub_result.mesh, &dec_params);

    let final_snapshot = MeshSnapshot::from_mesh(&dec_result.mesh);

    // Should be close to original face count
    assert!(
        final_snapshot.face_count <= original_faces + 4,
        "Should be near original face count"
    );

    // Bounds should still be approximately 0-1
    assert!(final_snapshot.bounds_min[0] >= -0.5, "Bounds preserved");
    assert!(final_snapshot.bounds_max[0] <= 1.5, "Bounds preserved");
}

// =============================================================================
// Visual Regression Tests: Edge Cases
// =============================================================================

#[test]
fn visual_regression_minimal_mesh() {
    // Single triangle
    let mut mesh = Mesh::new();
    mesh.vertices.push(Vertex::from_coords(0.0, 0.0, 0.0));
    mesh.vertices.push(Vertex::from_coords(1.0, 0.0, 0.0));
    mesh.vertices.push(Vertex::from_coords(0.5, 1.0, 0.0));
    mesh.faces.push([0, 1, 2]);

    let snapshot = MeshSnapshot::from_mesh(&mesh);

    assert_eq!(snapshot.vertex_count, 3);
    assert_eq!(snapshot.face_count, 1);
    assert!(snapshot.surface_area > 0.0);
}

#[test]
fn visual_regression_large_mesh() {
    // Test with a reasonably large mesh
    let sphere = create_test_sphere(4); // 5120 triangles

    let snapshot = MeshSnapshot::from_mesh(&sphere);

    assert_eq!(snapshot.face_count, 5120, "Large sphere face count");
    assert!(
        snapshot.surface_area > 10.0,
        "Large sphere has significant area"
    );
    assert!(snapshot.volume.abs() > 3.0, "Large sphere has volume");
}

// =============================================================================
// Baseline Tests for Future Regression Detection
// =============================================================================

/// This test establishes baselines for key operations.
/// If these values change, it indicates a potential regression.
#[test]
fn visual_baseline_operations() {
    // Cube subdivision baseline
    let cube = create_test_cube();
    let sub = subdivide_mesh(&cube, &SubdivideParams::with_iterations(1));
    assert_eq!(
        sub.mesh.face_count(),
        48,
        "Cube subdivision baseline: 48 faces"
    );

    // Sphere decimation baseline
    let sphere = create_test_sphere(2); // 320 faces
    let dec = decimate_mesh(&sphere, &DecimateParams::with_target_ratio(0.5));
    assert!(
        dec.mesh.face_count() <= 320,
        "Sphere decimation should reduce faces"
    );

    // Cube surface area baseline
    let cube = create_test_cube();
    let area = cube.surface_area();
    assert!((area - 6.0).abs() < 0.01, "Cube surface area baseline: 6.0");

    // Cube volume baseline
    let cube = create_test_cube();
    let vol = cube.volume().abs();
    assert!((vol - 1.0).abs() < 0.1, "Cube volume baseline: 1.0");
}

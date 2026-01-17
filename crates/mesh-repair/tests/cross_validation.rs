//! Cross-validation tests for mesh operations.
//!
//! These tests verify mesh-repair outputs are compatible with other tools
//! by testing file format compliance, geometric properties, and invariants.
//!
//! For full external validation, generated files can be imported into:
//! - MeshLab (https://www.meshlab.net/)
//! - Blender (https://www.blender.org/)
//! - FreeCAD (https://www.freecadweb.org/)
//!
//! Run with: cargo test -p mesh-repair --test cross_validation

use mesh_repair::{
    DecimateParams, Mesh, RemeshParams, SubdivideParams, Vertex, decimate_mesh, load_mesh,
    remesh_isotropic, save_mesh, subdivide_mesh, validate_mesh,
};

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
        [0, 3, 2],
        [4, 5, 6],
        [4, 6, 7],
        [0, 1, 5],
        [0, 5, 4],
        [2, 3, 7],
        [2, 7, 6],
        [0, 4, 7],
        [0, 7, 3],
        [1, 2, 6],
        [1, 6, 5],
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

// =============================================================================
// STL Format Compliance (MeshLab-compatible)
// =============================================================================

#[test]
fn cross_validate_stl_ascii_format() {
    let mesh = create_test_cube();
    let temp_dir = std::env::temp_dir();
    let path = temp_dir.join("cross_validate_cube.stl");

    // Save as STL (may be binary or ASCII depending on implementation)
    save_mesh(&mesh, &path).expect("Should save STL");

    // Read raw bytes to check format
    let bytes = std::fs::read(&path).expect("Should read file");

    // Binary STL: 80-byte header + 4-byte triangle count + (50 bytes per triangle)
    // ASCII STL: starts with "solid"
    let is_ascii = bytes.starts_with(b"solid") && !bytes.iter().take(80).any(|&b| b == 0); // ASCII shouldn't have null bytes in header

    if is_ascii {
        let content = String::from_utf8_lossy(&bytes);
        // ASCII STL must start with "solid"
        assert!(
            content.starts_with("solid"),
            "ASCII STL must start with 'solid'"
        );

        // Must contain facet definitions
        assert!(content.contains("facet normal"), "Must have facet normals");
        assert!(content.contains("outer loop"), "Must have outer loop");
        assert!(content.contains("vertex"), "Must have vertices");
        assert!(content.contains("endloop"), "Must have endloop");
        assert!(content.contains("endfacet"), "Must have endfacet");
        assert!(content.contains("endsolid"), "Must end with endsolid");

        // Should have 12 facets for a cube
        let facet_count = content.matches("facet normal").count();
        assert_eq!(facet_count, 12, "Cube should have 12 facets");
    } else {
        // Binary STL validation
        assert!(bytes.len() >= 84, "Binary STL must have header + count");

        // Read triangle count (bytes 80-83, little-endian u32)
        let triangle_count = u32::from_le_bytes([bytes[80], bytes[81], bytes[82], bytes[83]]);
        assert_eq!(triangle_count, 12, "Cube should have 12 triangles");

        // Each triangle: 12 bytes normal + 36 bytes vertices + 2 bytes attribute = 50 bytes
        let expected_size = 84 + (triangle_count as usize * 50);
        assert_eq!(bytes.len(), expected_size, "Binary STL size should match");
    }

    // Cleanup
    let _ = std::fs::remove_file(&path);
}

#[test]
fn cross_validate_stl_normal_vectors() {
    let mesh = create_test_cube();
    let temp_dir = std::env::temp_dir();
    let path = temp_dir.join("cross_validate_normals.stl");

    save_mesh(&mesh, &path).expect("Should save STL");
    let bytes = std::fs::read(&path).expect("Should read file");

    // Check if ASCII or binary
    let is_ascii = bytes.starts_with(b"solid") && !bytes.iter().take(80).any(|&b| b == 0);

    if is_ascii {
        let content = String::from_utf8_lossy(&bytes);
        // Parse and validate normal vectors from ASCII
        for line in content.lines() {
            if line.trim().starts_with("facet normal") {
                let parts: Vec<&str> = line.split_whitespace().collect();
                if parts.len() >= 5 {
                    let nx: f64 = parts[2].parse().unwrap_or(0.0);
                    let ny: f64 = parts[3].parse().unwrap_or(0.0);
                    let nz: f64 = parts[4].parse().unwrap_or(0.0);

                    // Normal should be unit length (with tolerance)
                    let len = (nx * nx + ny * ny + nz * nz).sqrt();
                    assert!(
                        (len - 1.0).abs() < 0.01 || len < 0.01, // Allow zero normals
                        "Normal should be unit length, got {}",
                        len
                    );
                }
            }
        }
    } else {
        // Binary STL: validate normals from binary data
        // Skip 80-byte header and 4-byte count
        let triangle_count =
            u32::from_le_bytes([bytes[80], bytes[81], bytes[82], bytes[83]]) as usize;

        for i in 0..triangle_count {
            let offset = 84 + i * 50; // Each triangle is 50 bytes
            // Normal is first 12 bytes (3 f32s)
            let nx = f32::from_le_bytes([
                bytes[offset],
                bytes[offset + 1],
                bytes[offset + 2],
                bytes[offset + 3],
            ]);
            let ny = f32::from_le_bytes([
                bytes[offset + 4],
                bytes[offset + 5],
                bytes[offset + 6],
                bytes[offset + 7],
            ]);
            let nz = f32::from_le_bytes([
                bytes[offset + 8],
                bytes[offset + 9],
                bytes[offset + 10],
                bytes[offset + 11],
            ]);

            let len = (nx * nx + ny * ny + nz * nz).sqrt();
            assert!(
                (len - 1.0).abs() < 0.01 || len < 0.01, // Allow zero normals
                "Normal should be unit length, got {} for triangle {}",
                len,
                i
            );
        }
    }

    let _ = std::fs::remove_file(&path);
}

// =============================================================================
// OBJ Format Compliance (Blender/MeshLab-compatible)
// =============================================================================

#[test]
fn cross_validate_obj_format() {
    let mesh = create_test_cube();
    let temp_dir = std::env::temp_dir();
    let path = temp_dir.join("cross_validate_cube.obj");

    save_mesh(&mesh, &path).expect("Should save OBJ");
    let content = std::fs::read_to_string(&path).unwrap();

    // Count vertices and faces
    let vertex_count = content.lines().filter(|l| l.starts_with("v ")).count();
    let face_count = content.lines().filter(|l| l.starts_with("f ")).count();

    assert_eq!(vertex_count, 8, "Cube should have 8 vertices");
    assert_eq!(face_count, 12, "Cube should have 12 faces");

    // Verify vertex format: "v x y z"
    for line in content.lines().filter(|l| l.starts_with("v ")) {
        let parts: Vec<&str> = line.split_whitespace().collect();
        assert_eq!(parts.len(), 4, "Vertex line should have 4 parts");
        assert!(parts[0] == "v", "Should start with 'v'");
        assert!(parts[1].parse::<f64>().is_ok(), "X should be numeric");
        assert!(parts[2].parse::<f64>().is_ok(), "Y should be numeric");
        assert!(parts[3].parse::<f64>().is_ok(), "Z should be numeric");
    }

    // Verify face format: "f v1 v2 v3"
    for line in content.lines().filter(|l| l.starts_with("f ")) {
        let parts: Vec<&str> = line.split_whitespace().collect();
        assert!(parts.len() >= 4, "Face line should have at least 4 parts");
        assert!(parts[0] == "f", "Should start with 'f'");

        // Verify indices are valid (1-indexed in OBJ)
        for idx_str in &parts[1..] {
            // Handle "v/vt/vn" format
            let idx: usize = idx_str.split('/').next().unwrap().parse().unwrap();
            assert!(idx >= 1 && idx <= vertex_count, "Index should be valid");
        }
    }

    let _ = std::fs::remove_file(&path);
}

// =============================================================================
// PLY Format Compliance
// =============================================================================

#[test]
fn cross_validate_ply_format() {
    let mesh = create_test_cube();
    let temp_dir = std::env::temp_dir();
    let path = temp_dir.join("cross_validate_cube.ply");

    save_mesh(&mesh, &path).expect("Should save PLY");
    let content = std::fs::read_to_string(&path).unwrap();

    // PLY must start with magic number
    assert!(content.starts_with("ply"), "PLY must start with 'ply'");

    // Must have format declaration
    assert!(
        content.contains("format ascii") || content.contains("format binary"),
        "Must have format declaration"
    );

    // Must declare vertex element
    assert!(
        content.contains("element vertex 8"),
        "Must declare 8 vertices"
    );

    // Must declare face element
    assert!(content.contains("element face 12"), "Must declare 12 faces");

    // Must have end_header
    assert!(content.contains("end_header"), "Must have end_header");

    let _ = std::fs::remove_file(&path);
}

// =============================================================================
// 3MF Format Compliance
// =============================================================================

#[test]
fn cross_validate_3mf_format() {
    let mesh = create_test_cube();
    let temp_dir = std::env::temp_dir();
    let path = temp_dir.join("cross_validate_cube.3mf");

    save_mesh(&mesh, &path).expect("Should save 3MF");

    // 3MF is a ZIP file
    let file = std::fs::File::open(&path).unwrap();
    let archive = zip::ZipArchive::new(file);
    assert!(archive.is_ok(), "3MF should be valid ZIP");

    let mut archive = archive.unwrap();

    // Must contain [Content_Types].xml
    assert!(
        archive.by_name("[Content_Types].xml").is_ok(),
        "Must have Content_Types"
    );

    // Must contain model file
    let model_exists = archive.by_name("3D/3dmodel.model").is_ok();
    assert!(model_exists, "Must have 3D model");

    let _ = std::fs::remove_file(&path);
}

// =============================================================================
// Geometric Invariant Tests (Cross-tool validation)
// =============================================================================

#[test]
fn cross_validate_volume_preservation() {
    let cube = create_test_cube();
    let original_volume = cube.volume().abs();

    // Volume should be 1.0 for unit cube
    assert!(
        (original_volume - 1.0).abs() < 0.1,
        "Unit cube volume should be ~1.0"
    );

    // After subdivision, volume changes due to Loop subdivision smoothing
    // Loop subdivision smooths corners, which reduces volume for convex shapes
    let sub = subdivide_mesh(&cube, &SubdivideParams::with_iterations(1));
    let sub_volume = sub.mesh.volume().abs();
    let volume_ratio = sub_volume / original_volume;
    // Volume should be within reasonable range (40-100% of original)
    // Loop subdivision on a cube typically reduces volume by ~50% due to corner smoothing
    assert!(
        volume_ratio > 0.4 && volume_ratio < 1.1,
        "Volume ratio after subdivision: {} (expected 0.4-1.1)",
        volume_ratio
    );
}

#[test]
fn cross_validate_surface_area() {
    let cube = create_test_cube();
    let area = cube.surface_area();

    // Unit cube surface area = 6
    assert!(
        (area - 6.0).abs() < 0.01,
        "Unit cube surface area should be 6.0, got {}",
        area
    );
}

#[test]
fn cross_validate_bounds() {
    let cube = create_test_cube();
    let (min, max) = cube.bounds().expect("Should have bounds");

    // Unit cube from (0,0,0) to (1,1,1)
    assert!((min.x - 0.0).abs() < 1e-10, "Min X");
    assert!((min.y - 0.0).abs() < 1e-10, "Min Y");
    assert!((min.z - 0.0).abs() < 1e-10, "Min Z");
    assert!((max.x - 1.0).abs() < 1e-10, "Max X");
    assert!((max.y - 1.0).abs() < 1e-10, "Max Y");
    assert!((max.z - 1.0).abs() < 1e-10, "Max Z");
}

#[test]
fn cross_validate_euler_characteristic() {
    // For a closed manifold mesh: V - E + F = 2 (Euler formula)
    let cube = create_test_cube();

    let v = cube.vertex_count();
    let f = cube.face_count();
    // For triangle mesh: E = 3F/2 (each edge shared by 2 faces)
    let e = (3 * f) / 2;

    let euler = v as i32 - e as i32 + f as i32;
    assert_eq!(euler, 2, "Euler characteristic for closed mesh should be 2");
}

#[test]
fn cross_validate_sphere_euler() {
    let sphere = create_test_sphere(2);

    let v = sphere.vertex_count();
    let f = sphere.face_count();
    let e = (3 * f) / 2;

    let euler = v as i32 - e as i32 + f as i32;
    assert_eq!(euler, 2, "Sphere Euler characteristic should be 2");
}

// =============================================================================
// Operation Consistency Tests
// =============================================================================

#[test]
fn cross_validate_decimation_consistency() {
    let sphere = create_test_sphere(3); // 1280 triangles
    let original_volume = sphere.volume().abs();

    // Decimate to 50%
    let params = DecimateParams::with_target_ratio(0.5);
    let result = decimate_mesh(&sphere, &params);
    let decimated_volume = result.mesh.volume().abs();

    // Volume should be approximately preserved (within 30%)
    let volume_ratio = decimated_volume / original_volume;
    assert!(
        volume_ratio > 0.7 && volume_ratio < 1.3,
        "Decimation should preserve volume (ratio: {})",
        volume_ratio
    );
}

#[test]
fn cross_validate_io_roundtrip_all_formats() {
    let original = create_test_sphere(1);
    let temp_dir = std::env::temp_dir();

    // Test STL roundtrip
    let stl_path = temp_dir.join("roundtrip.stl");
    save_mesh(&original, &stl_path).unwrap();
    let stl_loaded = load_mesh(&stl_path).unwrap();
    assert_eq!(original.face_count(), stl_loaded.face_count(), "STL faces");

    // Test OBJ roundtrip
    let obj_path = temp_dir.join("roundtrip.obj");
    save_mesh(&original, &obj_path).unwrap();
    let obj_loaded = load_mesh(&obj_path).unwrap();
    assert_eq!(original.face_count(), obj_loaded.face_count(), "OBJ faces");

    // Test PLY roundtrip
    let ply_path = temp_dir.join("roundtrip.ply");
    save_mesh(&original, &ply_path).unwrap();
    let ply_loaded = load_mesh(&ply_path).unwrap();
    assert_eq!(original.face_count(), ply_loaded.face_count(), "PLY faces");

    // Test 3MF roundtrip
    let threemf_path = temp_dir.join("roundtrip.3mf");
    save_mesh(&original, &threemf_path).unwrap();
    let threemf_loaded = load_mesh(&threemf_path).unwrap();
    assert_eq!(
        original.face_count(),
        threemf_loaded.face_count(),
        "3MF faces"
    );

    // Cleanup
    let _ = std::fs::remove_file(&stl_path);
    let _ = std::fs::remove_file(&obj_path);
    let _ = std::fs::remove_file(&ply_path);
    let _ = std::fs::remove_file(&threemf_path);
}

// =============================================================================
// Manifold Validation (matches MeshLab's manifold check)
// =============================================================================

#[test]
fn cross_validate_manifold_cube() {
    let cube = create_test_cube();
    let report = validate_mesh(&cube);

    // Clean cube should be manifold
    assert!(report.is_manifold, "Cube should be manifold");
    assert!(report.is_watertight, "Cube should be watertight");
    assert_eq!(
        report.boundary_edge_count, 0,
        "Cube should have no boundary edges"
    );
    assert_eq!(
        report.non_manifold_edge_count, 0,
        "Cube should have no non-manifold edges"
    );
}

#[test]
fn cross_validate_manifold_sphere() {
    let sphere = create_test_sphere(2);
    let report = validate_mesh(&sphere);

    assert!(report.is_manifold, "Sphere should be manifold");
    assert!(report.is_watertight, "Sphere should be watertight");
}

// =============================================================================
// External Validation Helper
// =============================================================================

/// Generate test meshes for external validation.
///
/// Run this test to generate files that can be imported into:
/// - MeshLab: Filter > Quality Measure and Topology Checks
/// - Blender: Import, then use Mesh > Check All
/// - FreeCAD: Part > Check Geometry
#[test]
#[ignore] // Run manually with: cargo test -p mesh-repair --test cross_validation generate_validation_files -- --ignored
fn generate_validation_files() {
    let output_dir = std::env::temp_dir().join("mesh_validation_files");
    std::fs::create_dir_all(&output_dir).unwrap();

    println!("\nGenerating validation files in: {:?}", output_dir);

    // 1. Clean cube
    let cube = create_test_cube();
    save_mesh(&cube, &output_dir.join("clean_cube.stl")).unwrap();
    save_mesh(&cube, &output_dir.join("clean_cube.obj")).unwrap();
    save_mesh(&cube, &output_dir.join("clean_cube.ply")).unwrap();
    save_mesh(&cube, &output_dir.join("clean_cube.3mf")).unwrap();
    println!("  Created: clean_cube.{{stl,obj,ply,3mf}}");

    // 2. Subdivided sphere
    let sphere = create_test_sphere(3);
    save_mesh(&sphere, &output_dir.join("sphere_1280tri.stl")).unwrap();
    println!("  Created: sphere_1280tri.stl");

    // 3. Decimated sphere
    let dec_params = DecimateParams::with_target_ratio(0.5);
    let decimated = decimate_mesh(&sphere, &dec_params);
    save_mesh(&decimated.mesh, &output_dir.join("sphere_decimated.stl")).unwrap();
    println!("  Created: sphere_decimated.stl");

    // 4. Remeshed sphere
    let remesh_params = RemeshParams {
        target_edge_length: Some(0.2),
        iterations: 3,
        ..Default::default()
    };
    let remeshed = remesh_isotropic(&sphere, &remesh_params);
    save_mesh(&remeshed.mesh, &output_dir.join("sphere_remeshed.stl")).unwrap();
    println!("  Created: sphere_remeshed.stl");

    // 5. Subdivided cube (for smoothness test)
    let sub_cube = subdivide_mesh(&cube, &SubdivideParams::with_iterations(2));
    save_mesh(&sub_cube.mesh, &output_dir.join("cube_subdivided.stl")).unwrap();
    println!("  Created: cube_subdivided.stl");

    println!("\nValidation instructions:");
    println!("1. Open each file in MeshLab");
    println!(
        "2. Run: Filters > Quality Measure and Topology Checks > Compute Topological Measures"
    );
    println!("3. Verify: 'Mesh is two-manifold' = Yes");
    println!("4. Verify: 'Mesh is watertight' = Yes (for closed meshes)");
    println!("\nFiles location: {:?}", output_dir);
}

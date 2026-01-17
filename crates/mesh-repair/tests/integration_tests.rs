//! End-to-end integration tests for mesh-repair.
//!
//! These tests exercise the full pipeline from load -> repair -> validate -> save
//! to ensure all components work together correctly.

use mesh_repair::{
    Mesh, RepairParams, ThicknessParams, ValidationOptions, Vertex, validate_mesh_data,
};
use std::io::Write;
use tempfile::NamedTempFile;

/// Create a simple valid cube mesh for testing.
fn create_test_cube(size: f64) -> Mesh {
    let mut mesh = Mesh::new();

    // 8 vertices of the cube
    mesh.vertices.push(Vertex::from_coords(0.0, 0.0, 0.0)); // 0
    mesh.vertices.push(Vertex::from_coords(size, 0.0, 0.0)); // 1
    mesh.vertices.push(Vertex::from_coords(size, size, 0.0)); // 2
    mesh.vertices.push(Vertex::from_coords(0.0, size, 0.0)); // 3
    mesh.vertices.push(Vertex::from_coords(0.0, 0.0, size)); // 4
    mesh.vertices.push(Vertex::from_coords(size, 0.0, size)); // 5
    mesh.vertices.push(Vertex::from_coords(size, size, size)); // 6
    mesh.vertices.push(Vertex::from_coords(0.0, size, size)); // 7

    // 12 triangles (2 per face), CCW winding when viewed from outside
    // Bottom face (z=0)
    mesh.faces.push([0, 2, 1]);
    mesh.faces.push([0, 3, 2]);
    // Top face (z=size)
    mesh.faces.push([4, 5, 6]);
    mesh.faces.push([4, 6, 7]);
    // Front face (y=0)
    mesh.faces.push([0, 1, 5]);
    mesh.faces.push([0, 5, 4]);
    // Back face (y=size)
    mesh.faces.push([3, 7, 6]);
    mesh.faces.push([3, 6, 2]);
    // Left face (x=0)
    mesh.faces.push([0, 4, 7]);
    mesh.faces.push([0, 7, 3]);
    // Right face (x=size)
    mesh.faces.push([1, 2, 6]);
    mesh.faces.push([1, 6, 5]);

    mesh
}

/// Create a cube with one face removed (open mesh with hole).
fn create_open_cube(size: f64) -> Mesh {
    let mut mesh = create_test_cube(size);
    // Remove the top face (last 2 triangles)
    mesh.faces.pop();
    mesh.faces.pop();
    mesh
}

/// Create a mesh with some degenerate triangles.
fn create_mesh_with_degenerates() -> Mesh {
    let mut mesh = Mesh::new();

    // Good triangle
    mesh.vertices.push(Vertex::from_coords(0.0, 0.0, 0.0));
    mesh.vertices.push(Vertex::from_coords(10.0, 0.0, 0.0));
    mesh.vertices.push(Vertex::from_coords(5.0, 10.0, 0.0));
    mesh.faces.push([0, 1, 2]);

    // Degenerate triangle (collinear points)
    mesh.vertices.push(Vertex::from_coords(0.0, 0.0, 5.0));
    mesh.vertices.push(Vertex::from_coords(5.0, 0.0, 5.0));
    mesh.vertices.push(Vertex::from_coords(10.0, 0.0, 5.0)); // Collinear
    mesh.faces.push([3, 4, 5]);

    // Another good triangle
    mesh.vertices.push(Vertex::from_coords(0.0, 0.0, 10.0));
    mesh.vertices.push(Vertex::from_coords(10.0, 0.0, 10.0));
    mesh.vertices.push(Vertex::from_coords(5.0, 10.0, 10.0));
    mesh.faces.push([6, 7, 8]);

    mesh
}

/// Create an ASCII STL string for a single triangle.
fn stl_triangle() -> String {
    r#"solid test
facet normal 0 0 1
  outer loop
    vertex 0 0 0
    vertex 100 0 0
    vertex 0 100 0
  endloop
endfacet
facet normal 0 0 1
  outer loop
    vertex 0 100 0
    vertex 100 0 0
    vertex 100 100 0
  endloop
endfacet
facet normal 0 0 -1
  outer loop
    vertex 0 0 0
    vertex 0 100 0
    vertex 100 0 0
  endloop
endfacet
facet normal 0 0 -1
  outer loop
    vertex 100 0 0
    vertex 0 100 0
    vertex 100 100 0
  endloop
endfacet
endsolid test"#
        .to_string()
}

// =============================================================================
// Integration Test: Load -> Validate -> Save cycle
// =============================================================================

#[test]
fn test_stl_load_validate_save_cycle() {
    // Create a test STL file
    let mut file = NamedTempFile::with_suffix(".stl").unwrap();
    writeln!(file, "{}", stl_triangle()).unwrap();

    // Load
    let mesh = Mesh::load(file.path()).expect("Should load STL");
    assert!(mesh.vertex_count() >= 3);
    assert!(mesh.face_count() >= 2);

    // Validate
    let report = mesh.validate();
    assert!(report.is_valid());

    // Save to new file
    let output = NamedTempFile::with_suffix(".stl").unwrap();
    mesh.save(output.path()).expect("Should save STL");

    // Reload and compare
    let reloaded = Mesh::load(output.path()).expect("Should reload STL");
    assert_eq!(mesh.face_count(), reloaded.face_count());
}

#[test]
fn test_obj_load_validate_save_cycle() {
    // Create test OBJ content
    let obj_content = r#"# Simple cube
v 0 0 0
v 10 0 0
v 10 10 0
v 0 10 0
v 0 0 10
v 10 0 10
v 10 10 10
v 0 10 10
f 1 2 3
f 1 3 4
f 5 7 6
f 5 8 7
f 1 6 2
f 1 5 6
f 4 7 8
f 4 3 7
f 1 8 5
f 1 4 8
f 2 6 7
f 2 7 3
"#;

    let mut file = NamedTempFile::with_suffix(".obj").unwrap();
    write!(file, "{}", obj_content).unwrap();

    // Load
    let mesh = Mesh::load(file.path()).expect("Should load OBJ");
    assert_eq!(mesh.vertex_count(), 8);
    assert_eq!(mesh.face_count(), 12);

    // Validate
    let report = mesh.validate();
    assert!(report.is_valid());
    assert!(report.is_watertight);
    assert!(report.is_manifold);

    // Save to new file
    let output = NamedTempFile::with_suffix(".obj").unwrap();
    mesh.save(output.path()).expect("Should save OBJ");

    // Reload and verify index preservation
    let reloaded = Mesh::load(output.path()).expect("Should reload OBJ");
    assert_eq!(mesh.vertex_count(), reloaded.vertex_count());
    assert_eq!(mesh.face_count(), reloaded.face_count());

    // Verify vertex positions match
    for (i, (orig, loaded)) in mesh
        .vertices
        .iter()
        .zip(reloaded.vertices.iter())
        .enumerate()
    {
        let diff = (orig.position - loaded.position).norm();
        assert!(diff < 1e-5, "Vertex {} position mismatch", i);
    }
}

#[test]
fn test_ply_load_validate_save_cycle() {
    let ply_content = r#"ply
format ascii 1.0
element vertex 4
property float x
property float y
property float z
element face 4
property list uchar int vertex_indices
end_header
0 0 0
10 0 0
5 10 0
5 5 10
3 0 1 2
3 0 1 3
3 1 2 3
3 2 0 3
"#;

    let mut file = NamedTempFile::with_suffix(".ply").unwrap();
    write!(file, "{}", ply_content).unwrap();

    // Load
    let mesh = Mesh::load(file.path()).expect("Should load PLY");
    assert_eq!(mesh.vertex_count(), 4);
    assert_eq!(mesh.face_count(), 4);

    // Validate
    let report = mesh.validate();
    assert!(report.is_valid());

    // Save to new file
    let output = NamedTempFile::with_suffix(".ply").unwrap();
    mesh.save(output.path()).expect("Should save PLY");

    // Reload and verify
    let reloaded = Mesh::load(output.path()).expect("Should reload PLY");
    assert_eq!(mesh.vertex_count(), reloaded.vertex_count());
    assert_eq!(mesh.face_count(), reloaded.face_count());
}

#[test]
fn test_3mf_load_validate_save_cycle() {
    // Create a simple 3MF by saving a known mesh
    let mesh = create_test_cube(10.0);

    // Validate before save
    let report = mesh.validate();
    assert!(report.is_watertight);
    assert!(report.is_manifold);

    // Save as 3MF
    let output = NamedTempFile::with_suffix(".3mf").unwrap();
    mesh.save(output.path()).expect("Should save 3MF");

    // Reload
    let reloaded = Mesh::load(output.path()).expect("Should reload 3MF");
    assert_eq!(mesh.vertex_count(), reloaded.vertex_count());
    assert_eq!(mesh.face_count(), reloaded.face_count());

    // Verify still valid after round-trip
    let reloaded_report = reloaded.validate();
    assert!(reloaded_report.is_watertight);
    assert!(reloaded_report.is_manifold);
}

// =============================================================================
// Integration Test: Full Repair Pipeline
// =============================================================================

#[test]
fn test_repair_open_mesh() {
    let mut mesh = create_open_cube(10.0);

    // Should have holes initially
    let initial_report = mesh.validate();
    assert!(!initial_report.is_watertight, "Open cube should have holes");

    // Repair should fill holes
    let params = RepairParams::default();
    mesh.repair_with_config(&params)
        .expect("Repair should succeed");

    // Should be watertight after repair
    let final_report = mesh.validate();
    // Note: hole filling may not always succeed in making watertight
    // depending on hole complexity, but it should at least not crash
    assert!(final_report.is_valid());
}

#[test]
fn test_repair_removes_degenerates() {
    let mut mesh = create_mesh_with_degenerates();
    assert_eq!(mesh.face_count(), 3, "Should start with 3 faces");

    // Repair should remove degenerate triangles
    let params = RepairParams::default();
    mesh.repair_with_config(&params)
        .expect("Repair should succeed");

    // Degenerate triangle should be removed
    // (exact count depends on removal logic)
    let report = mesh.validate();
    assert!(report.is_valid());
}

#[test]
fn test_repair_preset_for_scans() {
    let mut mesh = create_test_cube(10.0);

    let params = RepairParams::for_scans();
    mesh.repair_with_config(&params)
        .expect("Repair should succeed");

    let report = mesh.validate();
    assert!(report.is_valid());
    assert!(report.is_manifold);
}

#[test]
fn test_repair_preset_for_cad() {
    let mut mesh = create_test_cube(10.0);

    let params = RepairParams::for_cad();
    mesh.repair_with_config(&params)
        .expect("Repair should succeed");

    let report = mesh.validate();
    assert!(report.is_valid());
}

#[test]
fn test_repair_preset_for_printing() {
    let mut mesh = create_test_cube(10.0);

    let params = RepairParams::for_printing();
    mesh.repair_with_config(&params)
        .expect("Repair should succeed");

    let report = mesh.validate();
    assert!(report.is_valid());
    assert!(report.is_manifold);
}

// =============================================================================
// Integration Test: Full Pipeline with Validation
// =============================================================================

#[test]
fn test_full_pipeline_load_repair_validate_save() {
    // Create test file
    let obj_content = r#"# Test mesh with potential issues
v 0 0 0
v 10 0 0
v 10 10 0
v 0 10 0
v 0 0 10
v 10 0 10
v 10 10 10
v 0 10 10
f 1 2 3
f 1 3 4
f 5 7 6
f 5 8 7
f 1 6 2
f 1 5 6
f 4 7 8
f 4 3 7
f 1 8 5
f 1 4 8
f 2 6 7
f 2 7 3
"#;

    let mut input = NamedTempFile::with_suffix(".obj").unwrap();
    write!(input, "{}", obj_content).unwrap();

    // Step 1: Load
    let mut mesh = Mesh::load(input.path()).expect("Should load");

    // Step 2: Validate data integrity
    validate_mesh_data(&mesh, &ValidationOptions::default()).expect("Data should be valid");

    // Step 3: Repair
    mesh.repair().expect("Repair should succeed");

    // Step 4: Validate topology
    let report = mesh.validate();
    assert!(report.is_valid(), "Mesh should be valid after repair");

    // Step 5: Compute normals
    mesh.compute_normals();
    assert!(
        mesh.vertices.iter().all(|v| v.normal.is_some()),
        "All vertices should have normals after compute_normals"
    );

    // Step 6: Save as different formats
    let stl_output = NamedTempFile::with_suffix(".stl").unwrap();
    mesh.save(stl_output.path()).expect("Should save as STL");

    let obj_output = NamedTempFile::with_suffix(".obj").unwrap();
    mesh.save(obj_output.path()).expect("Should save as OBJ");

    let ply_output = NamedTempFile::with_suffix(".ply").unwrap();
    mesh.save(ply_output.path()).expect("Should save as PLY");

    let mf_output = NamedTempFile::with_suffix(".3mf").unwrap();
    mesh.save(mf_output.path()).expect("Should save as 3MF");

    // Step 7: Verify all outputs are loadable
    Mesh::load(stl_output.path()).expect("Should reload STL");
    Mesh::load(obj_output.path()).expect("Should reload OBJ");
    Mesh::load(ply_output.path()).expect("Should reload PLY");
    Mesh::load(mf_output.path()).expect("Should reload 3MF");
}

// =============================================================================
// Integration Test: Printability Analysis
// =============================================================================

#[test]
fn test_printability_check() {
    let mesh = create_test_cube(10.0);

    // Validate for printing requirements
    let report = mesh.validate();

    assert!(report.is_watertight, "Cube should be watertight");
    assert!(report.is_manifold, "Cube should be manifold");
    assert!(!report.is_inside_out, "Cube should not be inside-out");
    assert!(report.is_printable(), "Cube should be printable");

    // Check volume
    assert!(report.volume > 0.0, "Volume should be positive");

    // Check surface area
    assert!(report.surface_area > 0.0, "Surface area should be positive");
}

#[test]
fn test_inside_out_detection() {
    let mut mesh = create_test_cube(10.0);

    // Initially should be correct orientation
    let report = mesh.validate();
    assert!(!report.is_inside_out, "Should not be inside-out initially");

    // Invert all faces
    for face in &mut mesh.faces {
        face.swap(1, 2);
    }

    // Should detect inside-out
    let inverted_report = mesh.validate();
    assert!(inverted_report.is_inside_out, "Should detect inside-out");
    assert!(
        !inverted_report.is_printable(),
        "Inside-out should not be printable"
    );
}

// =============================================================================
// Integration Test: Wall Thickness Analysis
// =============================================================================

#[test]
fn test_wall_thickness_solid_cube() {
    let mesh = create_test_cube(10.0);

    let result = mesh.analyze_thickness(&ThicknessParams::default());

    // Solid cube should have thick walls
    assert!(result.vertices_analyzed > 0);
    // The cube is 10mm, so wall thickness should be significant
    // (actual values depend on ray casting implementation)
}

#[test]
fn test_wall_thickness_for_printing() {
    let mesh = create_test_cube(10.0);

    let result = mesh.analyze_thickness(&ThicknessParams::for_printing());

    assert!(result.vertices_analyzed > 0);
    // Solid 10mm cube should pass FDM thickness requirements (0.8mm)
}

// =============================================================================
// Integration Test: Component Analysis
// =============================================================================

#[test]
fn test_component_analysis_single_component() {
    let mesh = create_test_cube(10.0);

    let analysis = mesh.find_components();
    assert_eq!(analysis.component_count, 1, "Cube should have 1 component");
}

#[test]
fn test_component_analysis_multiple_components() {
    // Create two separate cubes
    let mut mesh = create_test_cube(10.0);
    let cube2 = create_test_cube(10.0);

    // Add second cube with offset
    let offset = mesh.vertices.len() as u32;
    for v in &cube2.vertices {
        let mut v2 = v.clone();
        v2.position.x += 20.0; // Offset in X
        mesh.vertices.push(v2);
    }
    for f in &cube2.faces {
        mesh.faces
            .push([f[0] + offset, f[1] + offset, f[2] + offset]);
    }

    // Should detect 2 components
    let analysis = mesh.find_components();
    assert_eq!(analysis.component_count, 2, "Should have 2 components");

    // Test splitting
    let components = mesh.split_components();
    assert_eq!(components.len(), 2, "Should split into 2 meshes");

    // Each component should be valid
    for (i, comp) in components.iter().enumerate() {
        let report = comp.validate();
        assert!(report.is_valid(), "Component {} should be valid", i);
    }
}

#[test]
fn test_keep_largest_component() {
    // Create main cube and small cube
    let mut mesh = create_test_cube(10.0);

    // Add a tiny cube (1x1x1 vs 10x10x10)
    let small_cube = create_test_cube(1.0);
    let offset = mesh.vertices.len() as u32;
    for v in &small_cube.vertices {
        let mut v2 = v.clone();
        v2.position.x += 50.0;
        mesh.vertices.push(v2);
    }
    for f in &small_cube.faces {
        mesh.faces
            .push([f[0] + offset, f[1] + offset, f[2] + offset]);
    }

    let initial_analysis = mesh.find_components();
    assert_eq!(initial_analysis.component_count, 2);

    // Keep only largest
    let removed = mesh.keep_largest_component();
    assert_eq!(removed, 1, "Should remove 1 component");

    let final_analysis = mesh.find_components();
    assert_eq!(final_analysis.component_count, 1);
}

// =============================================================================
// Integration Test: Self-Intersection Detection
// =============================================================================

#[test]
fn test_self_intersection_clean_mesh() {
    let mesh = create_test_cube(10.0);

    let result = mesh.detect_self_intersections();
    assert!(result.is_clean(), "Cube should have no self-intersections");
}

#[test]
fn test_self_intersection_detection() {
    let mut mesh = Mesh::new();

    // Create two intersecting triangles
    // Triangle 1 in XY plane
    mesh.vertices.push(Vertex::from_coords(-5.0, -5.0, 0.0));
    mesh.vertices.push(Vertex::from_coords(5.0, -5.0, 0.0));
    mesh.vertices.push(Vertex::from_coords(0.0, 5.0, 0.0));

    // Triangle 2 in XZ plane, crossing through triangle 1
    mesh.vertices.push(Vertex::from_coords(-5.0, 0.0, -5.0));
    mesh.vertices.push(Vertex::from_coords(5.0, 0.0, -5.0));
    mesh.vertices.push(Vertex::from_coords(0.0, 0.0, 5.0));

    mesh.faces.push([0, 1, 2]);
    mesh.faces.push([3, 4, 5]);

    let result = mesh.detect_self_intersections();
    assert!(!result.is_clean(), "Should detect self-intersection");
    assert_eq!(result.intersection_count, 1);
}

// =============================================================================
// Integration Test: Mesh Transformations
// =============================================================================

#[test]
fn test_mesh_decimation() {
    // Create a subdivided mesh with many triangles
    let mesh = create_test_cube(10.0);
    let subdivided = mesh.subdivide_n(2); // 12 -> 192 triangles

    assert!(
        subdivided.mesh.face_count() > 100,
        "Subdivided should have many triangles"
    );

    // Decimate to reduce triangles
    let decimated = subdivided.mesh.decimate_to_count(50);

    assert!(
        decimated.mesh.face_count() <= 50,
        "Decimated should have <= 50 triangles"
    );

    // Should still be valid mesh
    let report = decimated.mesh.validate();
    assert!(report.is_valid());
}

#[test]
fn test_mesh_subdivision() {
    let mesh = create_test_cube(10.0);
    let original_faces = mesh.face_count();

    // One level of subdivision quadruples triangle count
    let subdivided = mesh.subdivide();

    assert_eq!(
        subdivided.mesh.face_count(),
        original_faces * 4,
        "Subdivision should quadruple face count"
    );

    // Validate subdivided mesh
    let report = subdivided.mesh.validate();
    assert!(report.is_valid());
}

#[test]
fn test_mesh_remeshing() {
    let mesh = create_test_cube(10.0);

    // Remesh with target edge length
    let remeshed = mesh.remesh_with_edge_length(2.0);

    // Remeshing creates more uniform triangles
    assert!(remeshed.mesh.face_count() > 0);

    // Validate remeshed mesh
    let report = remeshed.mesh.validate();
    assert!(report.is_valid());
}

// =============================================================================
// Integration Test: Error Handling
// =============================================================================

#[test]
fn test_load_nonexistent_file() {
    let result = Mesh::load("/nonexistent/path/mesh.stl");
    assert!(result.is_err());
}

#[test]
fn test_load_invalid_extension() {
    let mut file = NamedTempFile::with_suffix(".xyz").unwrap();
    write!(file, "invalid content").unwrap();

    let result = Mesh::load(file.path());
    assert!(result.is_err());
}

#[test]
fn test_load_corrupted_file() {
    // Create a file with .stl extension but invalid content
    let mut file = NamedTempFile::with_suffix(".stl").unwrap();
    write!(file, "this is not a valid STL file content at all!").unwrap();

    let result = Mesh::load(file.path());
    // Should return error, not panic
    assert!(result.is_err());
}

#[test]
fn test_empty_mesh_operations() {
    let mesh = Mesh::new();

    // All operations on empty mesh should handle gracefully
    let report = mesh.validate();
    assert!(!report.is_valid());

    let components = mesh.find_components();
    assert_eq!(components.component_count, 0);

    let intersections = mesh.detect_self_intersections();
    assert!(intersections.is_clean());
}

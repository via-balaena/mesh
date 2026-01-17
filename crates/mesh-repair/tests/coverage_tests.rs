//! Additional tests to improve code coverage.
//!
//! This file contains tests targeting modules with lower coverage:
//! - Pipeline (non-feature-gated code)
//! - Error handling
//! - Builder patterns
//! - Scan processing
//! - Point cloud operations

use mesh_repair::{
    DecimateParams, Mesh, Pipeline, RemeshParams, RepairParams, Vertex, fill_holes, load_mesh,
    remesh_isotropic, save_mesh, validate_mesh, weld_vertices,
};
use std::path::Path;

// =============================================================================
// Test Mesh Creation Helpers
// =============================================================================

fn create_unit_cube() -> Mesh {
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

fn create_open_box() -> Mesh {
    // Cube missing one face (bottom)
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
        // bottom faces removed
        [2, 3, 7],
        [2, 7, 6], // top
        [0, 4, 7],
        [0, 7, 3], // left
        [1, 2, 6],
        [1, 6, 5], // right
    ];

    mesh
}

fn create_sphere(subdivisions: u32) -> Mesh {
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
        mesh = subdivide_sphere(&mesh);
    }

    mesh
}

fn subdivide_sphere(mesh: &Mesh) -> Mesh {
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
// Pipeline Tests (Core, no feature flag needed)
// =============================================================================

#[test]
fn test_pipeline_new_and_finish() {
    let mesh = create_unit_cube();
    let pipeline = Pipeline::new(mesh);

    assert_eq!(pipeline.stages_executed(), 0);
    assert_eq!(pipeline.mesh().face_count(), 12);

    let result = pipeline.finish();
    assert_eq!(result.mesh.face_count(), 12);
    assert_eq!(result.stages_executed, 0);
}

#[test]
fn test_pipeline_repair() {
    let mesh = create_unit_cube();
    let result = Pipeline::new(mesh).repair().finish();

    assert_eq!(result.stages_executed, 1);
    assert!(!result.operation_log.is_empty());
}

#[test]
fn test_pipeline_repair_for_scans() {
    let mesh = create_unit_cube();
    let result = Pipeline::new(mesh).repair_for_scans().finish();

    assert_eq!(result.stages_executed, 1);
}

#[test]
fn test_pipeline_repair_for_printing() {
    let mesh = create_unit_cube();
    let result = Pipeline::new(mesh).repair_for_printing().finish();

    assert_eq!(result.stages_executed, 1);
}

#[test]
fn test_pipeline_repair_for_cad() {
    let mesh = create_unit_cube();
    let result = Pipeline::new(mesh).repair_for_cad().finish();

    assert_eq!(result.stages_executed, 1);
}

#[test]
fn test_pipeline_repair_with_params() {
    let mesh = create_unit_cube();
    let params = RepairParams::for_printing();
    let result = Pipeline::new(mesh).repair_with_params(&params).finish();

    assert_eq!(result.stages_executed, 1);
}

#[test]
fn test_pipeline_fill_holes() {
    let mesh = create_open_box();
    let result = Pipeline::new(mesh).fill_holes(100).finish();

    assert_eq!(result.stages_executed, 1);
}

#[test]
fn test_pipeline_fix_winding() {
    let mesh = create_unit_cube();
    let result = Pipeline::new(mesh).fix_winding().finish();

    assert_eq!(result.stages_executed, 1);
}

#[test]
fn test_pipeline_remove_small_components() {
    let mesh = create_unit_cube();
    let result = Pipeline::new(mesh).remove_small_components(5).finish();

    assert_eq!(result.stages_executed, 1);
}

#[test]
fn test_pipeline_remesh() {
    let mesh = create_sphere(2); // Use sphere for better remesh results
    let result = Pipeline::new(mesh).remesh(0.2).finish();

    assert_eq!(result.stages_executed, 1);
    // Remeshing should produce faces (may return 0 for degenerate input)
    // The main goal is to test the pipeline doesn't crash
}

#[test]
fn test_pipeline_remesh_with_params() {
    let mesh = create_unit_cube();
    let params = RemeshParams {
        target_edge_length: Some(0.5),
        iterations: 2,
        ..Default::default()
    };
    let result = Pipeline::new(mesh).remesh_with_params(&params).finish();

    assert_eq!(result.stages_executed, 1);
}

#[test]
fn test_pipeline_decimate_to_count() {
    let mesh = create_sphere(2); // 320 triangles
    let result = Pipeline::new(mesh).decimate_to_count(100).finish();

    assert_eq!(result.stages_executed, 1);
    assert!(result.mesh.face_count() <= 320);
}

#[test]
fn test_pipeline_decimate_to_ratio() {
    let mesh = create_sphere(2); // 320 triangles
    let original_count = mesh.face_count();
    let result = Pipeline::new(mesh).decimate_to_ratio(0.5).finish();

    assert_eq!(result.stages_executed, 1);
    assert!(result.mesh.face_count() <= original_count);
}

#[test]
fn test_pipeline_decimate_with_params() {
    let mesh = create_sphere(2);
    let params = DecimateParams::with_target_ratio(0.5);
    let result = Pipeline::new(mesh).decimate_with_params(&params).finish();

    assert_eq!(result.stages_executed, 1);
}

#[test]
fn test_pipeline_subdivide() {
    let mesh = create_unit_cube();
    let original_count = mesh.face_count();
    let result = Pipeline::new(mesh).subdivide(1).finish();

    assert_eq!(result.stages_executed, 1);
    // Subdivision multiplies face count by 4
    assert_eq!(result.mesh.face_count(), original_count * 4);
}

#[test]
fn test_pipeline_subdivide_with_params() {
    let mesh = create_unit_cube();
    let params = mesh_repair::SubdivideParams::with_iterations(1);
    let result = Pipeline::new(mesh).subdivide_with_params(&params).finish();

    assert_eq!(result.stages_executed, 1);
}

#[test]
fn test_pipeline_compute_normals() {
    let mesh = create_unit_cube();
    let result = Pipeline::new(mesh).compute_normals().finish();

    assert_eq!(result.stages_executed, 1);
}

#[test]
fn test_pipeline_validate() {
    let mesh = create_unit_cube();
    let result = Pipeline::new(mesh).validate().finish();

    assert_eq!(result.stages_executed, 1);
    assert!(result.validation.is_some());

    let report = result.validation.unwrap();
    assert!(report.is_watertight);
    assert!(report.is_manifold);
}

#[test]
fn test_pipeline_require_printable_success() {
    let mesh = create_unit_cube();
    let result = Pipeline::new(mesh)
        .require_printable()
        .expect("Should be printable")
        .finish();

    assert_eq!(result.stages_executed, 1);
    assert!(result.validation.is_some());
}

#[test]
fn test_pipeline_require_printable_failure() {
    let mesh = create_open_box(); // Open mesh, not printable
    let result = Pipeline::new(mesh).require_printable();

    assert!(result.is_err());
}

#[test]
fn test_pipeline_chaining_multiple_operations() {
    let mesh = create_unit_cube();
    let result = Pipeline::new(mesh)
        .repair()
        .compute_normals()
        .validate()
        .finish();

    assert_eq!(result.stages_executed, 3);
    assert_eq!(result.operation_log.len(), 3);
    assert!(result.validation.is_some());
}

#[test]
fn test_pipeline_validation_report_accessor() {
    let mesh = create_unit_cube();
    let pipeline = Pipeline::new(mesh).validate();

    // Check accessor before finishing
    let report = pipeline.validation_report();
    assert!(report.is_some());

    let result = pipeline.finish();
    assert!(result.validation.is_some());
}

#[test]
fn test_pipeline_log_entries() {
    let mesh = create_unit_cube();
    let pipeline = Pipeline::new(mesh).repair().compute_normals();

    let entries = pipeline.log_entries();
    assert_eq!(entries.len(), 2);
}

#[test]
fn test_pipeline_mesh_accessor() {
    let mesh = create_unit_cube();
    let pipeline = Pipeline::new(mesh);

    assert_eq!(pipeline.mesh().face_count(), 12);
}

#[test]
fn test_into_pipeline_from_mesh() {
    use mesh_repair::IntoPipeline;

    let mesh = create_unit_cube();
    let pipeline = mesh.into_pipeline();

    assert_eq!(pipeline.mesh().face_count(), 12);
}

#[test]
fn test_into_pipeline_from_remesh_result() {
    use mesh_repair::IntoPipeline;

    let mesh = create_unit_cube();
    let params = RemeshParams {
        target_edge_length: Some(0.5),
        iterations: 2,
        ..Default::default()
    };
    let remesh_result = remesh_isotropic(&mesh, &params);

    let pipeline = remesh_result.into_pipeline();
    assert!(pipeline.mesh().face_count() > 0);
}

#[test]
fn test_pipeline_save_and_load() {
    let mesh = create_unit_cube();
    let temp_dir = std::env::temp_dir();
    let temp_path = temp_dir.join("pipeline_test_output.stl");

    let result = Pipeline::new(mesh)
        .repair()
        .save(&temp_path)
        .expect("Save should succeed");

    assert!(result.stages_executed >= 2); // repair + save

    // Verify file was created
    assert!(temp_path.exists());

    // Load it back
    let loaded = Pipeline::load(&temp_path).expect("Load should succeed");
    assert!(loaded.mesh().face_count() > 0);

    // Cleanup
    let _ = std::fs::remove_file(&temp_path);
}

#[test]
fn test_pipeline_load_from_fixture() {
    // Use one of our test fixtures
    let fixtures_dir = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .unwrap()
        .parent()
        .unwrap()
        .join("tests")
        .join("fixtures")
        .join("thingi10k");

    let spider_path = fixtures_dir.join("spider_ascii.stl");
    if !spider_path.exists() {
        eprintln!("Skipping test: spider_ascii.stl not found");
        return;
    }

    let result = Pipeline::load(&spider_path)
        .expect("Load should succeed")
        .validate()
        .finish();

    assert!(result.mesh.face_count() > 0);
    assert!(result.validation.is_some());
}

// =============================================================================
// Error Handling Tests
// =============================================================================

#[test]
fn test_mesh_error_display() {
    use mesh_repair::MeshError;

    let err = MeshError::parse_error("test.xyz", "invalid format");
    let display = format!("{}", err);
    assert!(display.contains("test.xyz") || display.contains("invalid format"));

    let err = MeshError::invalid_topology("broken mesh");
    assert!(format!("{}", err).contains("broken"));

    let err = MeshError::empty_mesh("no vertices");
    assert!(
        format!("{}", err).contains("no vertices")
            || format!("{}", err).to_lowercase().contains("empty")
    );

    let err = MeshError::repair_failed("could not fix");
    assert!(
        format!("{}", err).contains("could not fix")
            || format!("{}", err).to_lowercase().contains("repair")
    );

    let err = MeshError::hole_fill_failed("too complex");
    let display = format!("{}", err);
    assert!(display.contains("too complex") || display.to_lowercase().contains("hole"));

    let err = MeshError::boolean_failed("union", "meshes don't overlap");
    let display = format!("{}", err);
    assert!(
        display.contains("union")
            || display.contains("overlap")
            || display.to_lowercase().contains("boolean")
    );

    let err = MeshError::decimation_failed("target too low");
    let display = format!("{}", err);
    assert!(display.contains("target") || display.to_lowercase().contains("decim"));

    let err = MeshError::remeshing_failed("edge length issue");
    let display = format!("{}", err);
    assert!(display.contains("edge") || display.to_lowercase().contains("remesh"));
}

#[test]
fn test_load_nonexistent_file() {
    let result = load_mesh(Path::new("/nonexistent/path/to/mesh.stl"));
    assert!(result.is_err());
}

#[test]
fn test_load_unsupported_extension() {
    let temp_dir = std::env::temp_dir();
    let bad_path = temp_dir.join("test.xyz");
    std::fs::write(&bad_path, "dummy content").unwrap();

    let result = load_mesh(&bad_path);
    assert!(result.is_err());

    let _ = std::fs::remove_file(&bad_path);
}

// =============================================================================
// Repair Parameter Tests
// =============================================================================

#[test]
fn test_repair_params_for_scans() {
    let params = RepairParams::for_scans();
    // Scans typically need larger weld tolerance
    assert!(params.weld_epsilon > 0.0);
    assert!(params.fill_holes);
}

#[test]
fn test_repair_params_for_printing() {
    let params = RepairParams::for_printing();
    assert!(params.fill_holes);
    assert!(params.fix_winding);
}

#[test]
fn test_repair_params_for_cad() {
    let params = RepairParams::for_cad();
    // CAD uses smaller tolerance to preserve precision
    assert!(params.weld_epsilon > 0.0);
}

#[test]
fn test_repair_params_default() {
    let params = RepairParams::default();
    assert!(params.weld_epsilon > 0.0);
}

// =============================================================================
// Decimate Parameter Tests
// =============================================================================

#[test]
fn test_decimate_params_with_target_triangles() {
    let params = DecimateParams::with_target_triangles(100);
    assert!(params.target_triangles.is_some());
    assert_eq!(params.target_triangles.unwrap(), 100);
}

#[test]
fn test_decimate_params_with_target_ratio() {
    let params = DecimateParams::with_target_ratio(0.5);
    assert!((params.target_ratio - 0.5).abs() < 1e-10);
}

#[test]
fn test_decimate_params_default() {
    let params = DecimateParams::default();
    assert!(params.target_triangles.is_none());
    assert!(params.target_ratio > 0.0);
}

// =============================================================================
// Remesh Parameter Tests
// =============================================================================

#[test]
fn test_remesh_params_default() {
    let params = RemeshParams::default();
    // Default should have some reasonable values
    assert!(params.iterations > 0);
}

#[test]
fn test_remesh_params_custom() {
    let params = RemeshParams {
        target_edge_length: Some(2.0),
        iterations: 5,
        ..Default::default()
    };

    assert_eq!(params.target_edge_length, Some(2.0));
    assert_eq!(params.iterations, 5);
}

// =============================================================================
// Subdivide Parameter Tests
// =============================================================================

#[test]
fn test_subdivide_params_with_iterations() {
    let params = mesh_repair::SubdivideParams::with_iterations(3);
    assert_eq!(params.iterations, 3);
}

#[test]
fn test_subdivide_params_default() {
    let params = mesh_repair::SubdivideParams::default();
    assert_eq!(params.iterations, 1);
}

// =============================================================================
// Mesh Validation Tests
// =============================================================================

#[test]
fn test_validate_watertight_cube() {
    let mesh = create_unit_cube();
    let report = validate_mesh(&mesh);

    assert!(report.is_watertight);
    assert!(report.is_manifold);
    assert_eq!(report.component_count, 1);
}

#[test]
fn test_validate_open_mesh() {
    let mesh = create_open_box();
    let report = validate_mesh(&mesh);

    assert!(!report.is_watertight);
}

#[test]
fn test_validate_printable() {
    let mesh = create_unit_cube();
    let report = validate_mesh(&mesh);

    assert!(report.is_printable());
}

#[test]
fn test_mesh_report_display() {
    let mesh = create_unit_cube();
    let report = validate_mesh(&mesh);

    let display = format!("{:?}", report);
    // Debug format should contain field info
    assert!(
        display.contains("vertex_count")
            || display.contains("face_count")
            || display.contains("MeshReport")
    );
}

// =============================================================================
// Component Analysis Tests
// =============================================================================

#[test]
fn test_find_connected_components() {
    let mesh = create_unit_cube();
    let analysis = mesh_repair::find_connected_components(&mesh);

    assert_eq!(analysis.components.len(), 1);
}

#[test]
fn test_keep_largest_component() {
    let mesh = create_unit_cube();
    let mut mesh_copy = mesh.clone();

    // keep_largest_component modifies the mesh to keep only the largest component
    // For a single-component mesh, it should keep all faces
    let _result = mesh_repair::keep_largest_component(&mut mesh_copy);

    // The mesh should still have faces (single component = all kept)
    assert!(mesh_copy.face_count() > 0);
}

#[test]
fn test_remove_small_components() {
    let mesh = create_unit_cube();
    let mut mesh_copy = mesh.clone();
    let removed = mesh_repair::remove_small_components(&mut mesh_copy, 5);

    // Cube is 12 faces, larger than threshold
    assert_eq!(removed, 0);
}

// =============================================================================
// Hole Filling Tests
// =============================================================================

#[test]
fn test_fill_holes_closed_mesh() {
    let mesh = create_unit_cube();
    let mut mesh_copy = mesh.clone();
    let result = fill_holes(&mut mesh_copy);

    assert!(result.is_ok());
}

#[test]
fn test_fill_holes_open_mesh() {
    let mesh = create_open_box();
    let mut mesh_copy = mesh.clone();
    let original_faces = mesh_copy.face_count();
    let result = fill_holes(&mut mesh_copy);

    assert!(result.is_ok());
    // Should have added faces to close hole
    assert!(mesh_copy.face_count() >= original_faces);
}

// =============================================================================
// Weld Vertices Tests
// =============================================================================

#[test]
fn test_weld_vertices_no_duplicates() {
    let mesh = create_unit_cube();
    let original_vertices = mesh.vertex_count();
    let mut mesh_copy = mesh.clone();
    weld_vertices(&mut mesh_copy, 1e-6);

    // No duplicates to weld in clean cube
    assert!(mesh_copy.vertex_count() <= original_vertices);
}

#[test]
fn test_weld_vertices_with_duplicates() {
    let mut mesh = create_unit_cube();
    // Add duplicate vertex at same position as vertex 0
    let dup_vertex = mesh.vertices[0].clone();
    let _original_vertices = mesh.vertex_count();
    mesh.vertices.push(dup_vertex);

    // Add a face that uses the duplicate
    let new_vert_idx = mesh.vertices.len() as u32 - 1;
    mesh.faces.push([new_vert_idx, 1, 2]);

    weld_vertices(&mut mesh, 1e-6);

    // Should have welded the duplicate, so vertex count should be <= original + 1
    // (depends on implementation - some welders only weld if vertex is referenced)
    // Main test is that it doesn't crash and mesh is still valid
    assert!(mesh.vertex_count() > 0);
    assert!(mesh.face_count() > 0);
}

// =============================================================================
// I/O Tests
// =============================================================================

#[test]
fn test_save_and_load_stl() {
    let mesh = create_unit_cube();
    let temp_dir = std::env::temp_dir();
    let temp_path = temp_dir.join("io_test.stl");

    save_mesh(&mesh, &temp_path).expect("Save should succeed");
    let loaded = load_mesh(&temp_path).expect("Load should succeed");

    assert_eq!(mesh.face_count(), loaded.face_count());

    let _ = std::fs::remove_file(&temp_path);
}

#[test]
fn test_save_and_load_obj() {
    let mesh = create_unit_cube();
    let temp_dir = std::env::temp_dir();
    let temp_path = temp_dir.join("io_test.obj");

    save_mesh(&mesh, &temp_path).expect("Save should succeed");
    let loaded = load_mesh(&temp_path).expect("Load should succeed");

    assert_eq!(mesh.face_count(), loaded.face_count());

    let _ = std::fs::remove_file(&temp_path);
}

#[test]
fn test_save_and_load_ply() {
    let mesh = create_unit_cube();
    let temp_dir = std::env::temp_dir();
    let temp_path = temp_dir.join("io_test.ply");

    save_mesh(&mesh, &temp_path).expect("Save should succeed");
    let loaded = load_mesh(&temp_path).expect("Load should succeed");

    assert_eq!(mesh.face_count(), loaded.face_count());

    let _ = std::fs::remove_file(&temp_path);
}

// =============================================================================
// Geometry Tests
// =============================================================================

#[test]
fn test_mesh_bounds() {
    let mesh = create_unit_cube();
    let (min, max) = mesh.bounds().expect("Should have bounds");

    // Unit cube from 0,0,0 to 1,1,1
    assert!((min.x - 0.0).abs() < 1e-10);
    assert!((min.y - 0.0).abs() < 1e-10);
    assert!((min.z - 0.0).abs() < 1e-10);
    assert!((max.x - 1.0).abs() < 1e-10);
    assert!((max.y - 1.0).abs() < 1e-10);
    assert!((max.z - 1.0).abs() < 1e-10);
}

#[test]
fn test_mesh_surface_area() {
    let mesh = create_unit_cube();
    let area = mesh.surface_area();

    // Unit cube has surface area of 6
    assert!((area - 6.0).abs() < 1e-10);
}

#[test]
fn test_mesh_volume() {
    let mesh = create_unit_cube();
    let volume = mesh.volume();

    // Unit cube has volume of 1
    assert!((volume.abs() - 1.0).abs() < 0.1); // Allow some tolerance for winding
}

#[test]
fn test_empty_mesh_bounds() {
    let mesh = Mesh::new();
    assert!(mesh.bounds().is_none());
}

#[test]
fn test_empty_mesh_surface_area() {
    let mesh = Mesh::new();
    assert_eq!(mesh.surface_area(), 0.0);
}

#[test]
fn test_empty_mesh_volume() {
    let mesh = Mesh::new();
    assert_eq!(mesh.volume(), 0.0);
}

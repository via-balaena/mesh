//! Conformance tests using real-world 3D printing models.
//!
//! These tests use sample meshes to verify mesh-repair handles real-world
//! inputs correctly. The test fixtures simulate common characteristics found
//! in the Thingi10K dataset:
//! - Multiple disconnected components
//! - Various polygon counts (100s to 1000s of triangles)
//! - Both ASCII and binary STL formats
//!
//! To run: cargo test -p mesh-repair thingi10k
//!
//! # Adding More Test Models
//!
//! To expand testing coverage, download additional models from:
//! - Thingi10K dataset: https://github.com/Thingi10K/Thingi10K
//! - Thingiverse: https://www.thingiverse.com
//!
//! Place STL files in: tests/fixtures/thingi10k/

use mesh_repair::{
    DecimateParams, RemeshParams, decimate_mesh, fill_holes, load_mesh, remesh_isotropic,
    save_mesh, validate_mesh, weld_vertices,
};
use std::path::PathBuf;

/// Get the path to test fixtures directory.
fn fixtures_dir() -> PathBuf {
    let manifest_dir = env!("CARGO_MANIFEST_DIR");
    PathBuf::from(manifest_dir)
        .parent()
        .unwrap()
        .parent()
        .unwrap()
        .join("tests")
        .join("fixtures")
        .join("thingi10k")
}

/// Get all STL files in the fixtures directory.
fn get_test_files() -> Vec<PathBuf> {
    let dir = fixtures_dir();
    if !dir.exists() {
        return Vec::new();
    }

    std::fs::read_dir(&dir)
        .unwrap()
        .filter_map(|entry| {
            let path = entry.ok()?.path();
            if path.extension().map(|e| e == "stl").unwrap_or(false) {
                Some(path)
            } else {
                None
            }
        })
        .collect()
}

// =============================================================================
// Loading Tests
// =============================================================================

#[test]
fn test_load_all_fixtures() {
    let files = get_test_files();
    if files.is_empty() {
        eprintln!("Warning: No test fixtures found in {:?}", fixtures_dir());
        return;
    }

    for file in &files {
        let result = load_mesh(file);
        assert!(
            result.is_ok(),
            "Failed to load {}: {:?}",
            file.display(),
            result.err()
        );

        let mesh = result.unwrap();
        assert!(
            mesh.vertex_count() > 0,
            "Mesh {} has no vertices",
            file.display()
        );
        assert!(
            mesh.face_count() > 0,
            "Mesh {} has no faces",
            file.display()
        );
    }
}

#[test]
fn test_ascii_stl_spider() {
    let path = fixtures_dir().join("spider_ascii.stl");
    if !path.exists() {
        eprintln!("Skipping test: spider_ascii.stl not found");
        return;
    }

    let mesh = load_mesh(&path).expect("Failed to load spider_ascii.stl");

    // Verify expected characteristics
    assert!(
        mesh.vertex_count() >= 700,
        "Spider should have ~722 vertices"
    );
    assert!(mesh.face_count() >= 1300, "Spider should have ~1312 faces");

    // Should have multiple components (spider body parts)
    let report = validate_mesh(&mesh);
    assert!(
        report.component_count >= 10,
        "Spider should have multiple components (body parts)"
    );
}

#[test]
fn test_binary_stl_colored() {
    let path = fixtures_dir().join("colored.stl");
    if !path.exists() {
        eprintln!("Skipping test: colored.stl not found");
        return;
    }

    let mesh = load_mesh(&path).expect("Failed to load colored.stl");

    // Verify expected characteristics
    assert!(
        mesh.vertex_count() >= 1000,
        "Colored mesh should have ~1080 vertices"
    );
    assert!(
        mesh.face_count() >= 2000,
        "Colored mesh should have ~2156 faces"
    );

    // Should be a single connected component
    let report = validate_mesh(&mesh);
    assert_eq!(report.component_count, 1, "Should be single component");
}

// =============================================================================
// Validation Tests
// =============================================================================

#[test]
fn test_validate_all_fixtures() {
    let files = get_test_files();
    if files.is_empty() {
        return;
    }

    for file in &files {
        let mesh = load_mesh(file).expect("Failed to load mesh");
        let report = validate_mesh(&mesh);

        // Basic sanity checks - validation shouldn't crash
        assert_eq!(
            report.vertex_count,
            mesh.vertex_count(),
            "Vertex count mismatch for {}",
            file.display()
        );
        assert_eq!(
            report.face_count,
            mesh.face_count(),
            "Face count mismatch for {}",
            file.display()
        );

        // All test fixtures should have positive surface area
        assert!(
            mesh.surface_area() > 0.0,
            "Surface area should be positive for {}",
            file.display()
        );
    }
}

#[test]
fn test_multi_component_mesh() {
    // Test mesh with multiple disconnected components (spider has 18)
    let path = fixtures_dir().join("spider_ascii.stl");
    if !path.exists() {
        return;
    }

    let mesh = load_mesh(&path).unwrap();
    let report = validate_mesh(&mesh);

    // Multi-component meshes are common in 3D printing (assembled parts)
    assert!(
        report.component_count > 1,
        "Expected multiple components in spider mesh"
    );

    // Components should be detectable
    let analysis = mesh_repair::find_connected_components(&mesh);
    assert_eq!(
        analysis.components.len(),
        report.component_count,
        "Component count should match"
    );
}

// =============================================================================
// Repair Operations
// =============================================================================

#[test]
fn test_weld_vertices_all_fixtures() {
    let files = get_test_files();
    if files.is_empty() {
        return;
    }

    for file in &files {
        let mesh = load_mesh(file).expect("Failed to load mesh");
        let original_vertices = mesh.vertex_count();

        let mut welded = mesh.clone();
        weld_vertices(&mut welded, 1e-6);

        // Welding should not increase vertex count
        assert!(
            welded.vertex_count() <= original_vertices,
            "Weld should not increase vertices for {}",
            file.display()
        );

        // Should still be valid mesh
        let report = validate_mesh(&welded);
        assert_eq!(report.vertex_count, welded.vertex_count());
    }
}

#[test]
fn test_hole_filling() {
    let files = get_test_files();
    if files.is_empty() {
        return;
    }

    for file in &files {
        let mesh = load_mesh(file).expect("Failed to load mesh");

        let mut repaired = mesh.clone();
        let result = fill_holes(&mut repaired);

        // Hole filling should succeed (even if no holes to fill)
        assert!(
            result.is_ok(),
            "Hole filling failed for {}: {:?}",
            file.display(),
            result.err()
        );

        // Face count should stay same or increase (filled holes)
        assert!(
            repaired.face_count() >= mesh.face_count(),
            "Hole filling should not remove faces for {}",
            file.display()
        );
    }
}

// =============================================================================
// Decimation Tests
// =============================================================================

#[test]
fn test_decimate_50_percent() {
    let files = get_test_files();
    if files.is_empty() {
        return;
    }

    for file in &files {
        let mesh = load_mesh(file).expect("Failed to load mesh");
        let original_faces = mesh.face_count();

        if original_faces < 100 {
            continue; // Skip very small meshes
        }

        let target = original_faces / 2;
        let params = DecimateParams::with_target_triangles(target);
        let result = decimate_mesh(&mesh, &params);

        // Decimation should reduce face count
        assert!(
            result.mesh.face_count() <= original_faces,
            "Decimation should reduce faces for {}",
            file.display()
        );

        // Should maintain valid topology
        let report = validate_mesh(&result.mesh);
        assert!(
            report.vertex_count > 0,
            "Decimated mesh should have vertices"
        );
    }
}

#[test]
fn test_aggressive_decimation() {
    // Test 90% reduction on larger meshes
    let path = fixtures_dir().join("spider_ascii.stl");
    if !path.exists() {
        return;
    }

    let mesh = load_mesh(&path).unwrap();
    let original_faces = mesh.face_count();
    let target = original_faces / 10; // 90% reduction

    let params = DecimateParams::with_target_triangles(target);
    let result = decimate_mesh(&mesh, &params);

    // Should significantly reduce complexity
    assert!(
        result.mesh.face_count() < original_faces / 2,
        "Aggressive decimation should significantly reduce faces"
    );

    // Should still be a valid mesh
    let report = validate_mesh(&result.mesh);
    assert!(report.vertex_count > 0);
    assert!(report.face_count > 0);
}

// =============================================================================
// Remeshing Tests
// =============================================================================

#[test]
fn test_remesh_small_mesh() {
    let path = fixtures_dir().join("slotted_disk.stl");
    if !path.exists() {
        return;
    }

    let mesh = load_mesh(&path).unwrap();

    let params = RemeshParams {
        target_edge_length: Some(0.5),
        iterations: 2,
        ..Default::default()
    };

    let result = remesh_isotropic(&mesh, &params);

    // Remeshing should produce valid output
    assert!(
        result.mesh.vertex_count() > 0,
        "Remeshed mesh should have vertices"
    );
    assert!(
        result.mesh.face_count() > 0,
        "Remeshed mesh should have faces"
    );
}

// =============================================================================
// Round-trip I/O Tests
// =============================================================================

#[test]
fn test_io_roundtrip_stl() {
    let files = get_test_files();
    if files.is_empty() {
        return;
    }

    let temp_dir = std::env::temp_dir();

    for file in &files {
        let mesh = load_mesh(file).expect("Failed to load mesh");

        // Save to temp file
        let temp_path = temp_dir.join(format!(
            "roundtrip_{}.stl",
            file.file_stem().unwrap().to_string_lossy()
        ));
        save_mesh(&mesh, &temp_path).expect("Failed to save mesh");

        // Reload
        let reloaded = load_mesh(&temp_path).expect("Failed to reload mesh");

        // Should preserve geometry
        assert_eq!(
            mesh.face_count(),
            reloaded.face_count(),
            "Face count should be preserved for {}",
            file.display()
        );

        // Cleanup
        let _ = std::fs::remove_file(&temp_path);
    }
}

#[test]
fn test_io_roundtrip_obj() {
    let path = fixtures_dir().join("colored.stl");
    if !path.exists() {
        return;
    }

    let mesh = load_mesh(&path).unwrap();
    let temp_dir = std::env::temp_dir();
    let temp_path = temp_dir.join("roundtrip_test.obj");

    // Save as OBJ
    save_mesh(&mesh, &temp_path).expect("Failed to save as OBJ");

    // Reload
    let reloaded = load_mesh(&temp_path).expect("Failed to reload OBJ");

    // OBJ should preserve face count
    assert_eq!(
        mesh.face_count(),
        reloaded.face_count(),
        "Face count should be preserved"
    );

    // Cleanup
    let _ = std::fs::remove_file(&temp_path);
}

// =============================================================================
// Geometry Invariant Tests
// =============================================================================

#[test]
fn test_bounds_finite() {
    let files = get_test_files();
    if files.is_empty() {
        return;
    }

    for file in &files {
        let mesh = load_mesh(file).expect("Failed to load mesh");

        if let Some((min, max)) = mesh.bounds() {
            // All coordinates should be finite
            assert!(min.x.is_finite(), "Min X not finite for {}", file.display());
            assert!(min.y.is_finite(), "Min Y not finite for {}", file.display());
            assert!(min.z.is_finite(), "Min Z not finite for {}", file.display());
            assert!(max.x.is_finite(), "Max X not finite for {}", file.display());
            assert!(max.y.is_finite(), "Max Y not finite for {}", file.display());
            assert!(max.z.is_finite(), "Max Z not finite for {}", file.display());

            // Max should be >= min
            assert!(max.x >= min.x, "Invalid X bounds for {}", file.display());
            assert!(max.y >= min.y, "Invalid Y bounds for {}", file.display());
            assert!(max.z >= min.z, "Invalid Z bounds for {}", file.display());
        }
    }
}

#[test]
fn test_surface_area_positive() {
    let files = get_test_files();
    if files.is_empty() {
        return;
    }

    for file in &files {
        let mesh = load_mesh(file).expect("Failed to load mesh");
        let area = mesh.surface_area();

        assert!(
            area > 0.0,
            "Surface area should be positive for {}",
            file.display()
        );
        assert!(
            area.is_finite(),
            "Surface area should be finite for {}",
            file.display()
        );
    }
}

#[test]
fn test_volume_finite() {
    let files = get_test_files();
    if files.is_empty() {
        return;
    }

    for file in &files {
        let mesh = load_mesh(file).expect("Failed to load mesh");
        let volume = mesh.volume();

        // Volume may be negative for inside-out meshes, but should be finite
        assert!(
            volume.is_finite(),
            "Volume should be finite for {}",
            file.display()
        );
    }
}

// =============================================================================
// Performance Regression Tests
// =============================================================================

#[test]
fn test_validation_performance() {
    // Ensure validation completes in reasonable time for medium meshes
    let path = fixtures_dir().join("spider_ascii.stl");
    if !path.exists() {
        return;
    }

    let mesh = load_mesh(&path).unwrap();

    let start = std::time::Instant::now();
    for _ in 0..10 {
        let _ = validate_mesh(&mesh);
    }
    let elapsed = start.elapsed();

    // 10 validations of ~1300 face mesh should complete in < 1 second
    assert!(
        elapsed.as_secs_f64() < 1.0,
        "Validation too slow: {:?} for 10 iterations",
        elapsed
    );
}

#[test]
fn test_decimation_performance() {
    let path = fixtures_dir().join("colored.stl");
    if !path.exists() {
        return;
    }

    let mesh = load_mesh(&path).unwrap();
    let target = mesh.face_count() / 2;
    let params = DecimateParams::with_target_triangles(target);

    let start = std::time::Instant::now();
    let _ = decimate_mesh(&mesh, &params);
    let elapsed = start.elapsed();

    // 50% decimation of ~2000 face mesh should complete in < 2 seconds
    assert!(
        elapsed.as_secs_f64() < 2.0,
        "Decimation too slow: {:?}",
        elapsed
    );
}

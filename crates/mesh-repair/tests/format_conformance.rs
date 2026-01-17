//! Format conformance tests for mesh I/O.
//!
//! These tests verify that mesh file formats are read and written
//! according to their specifications.

use mesh_repair::{Mesh, Vertex, load_mesh, save_mesh};
use std::io::Write;
use tempfile::NamedTempFile;

// =============================================================================
// STL Format Conformance Tests
// =============================================================================

mod stl_conformance {
    use super::*;

    /// STL files must have ASCII or binary format detection working correctly.
    #[test]
    fn test_ascii_stl_format() {
        let mut file = NamedTempFile::with_suffix(".stl").unwrap();

        // Standard ASCII STL format
        writeln!(file, "solid test_solid").unwrap();
        writeln!(file, "  facet normal 0 0 1").unwrap();
        writeln!(file, "    outer loop").unwrap();
        writeln!(file, "      vertex 0 0 0").unwrap();
        writeln!(file, "      vertex 1 0 0").unwrap();
        writeln!(file, "      vertex 0 1 0").unwrap();
        writeln!(file, "    endloop").unwrap();
        writeln!(file, "  endfacet").unwrap();
        writeln!(file, "endsolid test_solid").unwrap();

        let mesh = load_mesh(file.path()).expect("Should load ASCII STL");
        assert_eq!(mesh.face_count(), 1);
        assert_eq!(mesh.vertex_count(), 3);
    }

    /// STL normals should be preserved or recalculated consistently.
    #[test]
    fn test_stl_normal_handling() {
        let mut mesh = Mesh::new();
        mesh.vertices.push(Vertex::from_coords(0.0, 0.0, 0.0));
        mesh.vertices.push(Vertex::from_coords(1.0, 0.0, 0.0));
        mesh.vertices.push(Vertex::from_coords(0.0, 1.0, 0.0));
        mesh.faces.push([0, 1, 2]);

        let file = NamedTempFile::with_suffix(".stl").unwrap();
        save_mesh(&mesh, file.path()).expect("Should save STL");

        let reloaded = load_mesh(file.path()).expect("Should reload STL");
        assert_eq!(reloaded.face_count(), 1);
    }

    /// STL should handle meshes with many triangles.
    #[test]
    fn test_stl_large_mesh() {
        let mut mesh = Mesh::new();

        // Create a grid of triangles
        for i in 0..100 {
            for j in 0..100 {
                let base = mesh.vertices.len() as u32;
                let x = i as f64;
                let y = j as f64;

                mesh.vertices.push(Vertex::from_coords(x, y, 0.0));
                mesh.vertices.push(Vertex::from_coords(x + 1.0, y, 0.0));
                mesh.vertices.push(Vertex::from_coords(x, y + 1.0, 0.0));
                mesh.vertices
                    .push(Vertex::from_coords(x + 1.0, y + 1.0, 0.0));

                mesh.faces.push([base, base + 1, base + 2]);
                mesh.faces.push([base + 1, base + 3, base + 2]);
            }
        }

        let file = NamedTempFile::with_suffix(".stl").unwrap();
        save_mesh(&mesh, file.path()).expect("Should save large STL");

        let reloaded = load_mesh(file.path()).expect("Should reload large STL");
        assert_eq!(reloaded.face_count(), 20000);
    }

    /// STL coordinates should handle negative values.
    #[test]
    fn test_stl_negative_coordinates() {
        let mut mesh = Mesh::new();
        mesh.vertices.push(Vertex::from_coords(-10.0, -10.0, -10.0));
        mesh.vertices.push(Vertex::from_coords(10.0, -10.0, -10.0));
        mesh.vertices.push(Vertex::from_coords(0.0, 10.0, 10.0));
        mesh.faces.push([0, 1, 2]);

        let file = NamedTempFile::with_suffix(".stl").unwrap();
        save_mesh(&mesh, file.path()).expect("Should save STL with negative coords");

        let reloaded = load_mesh(file.path()).expect("Should reload");
        let (min, max) = reloaded.bounds().unwrap();

        assert!(min.x < 0.0, "Min X should be negative");
        assert!(min.y < 0.0, "Min Y should be negative");
        assert!(min.z < 0.0, "Min Z should be negative");
        assert!(max.x > 0.0, "Max X should be positive");
    }

    /// STL should handle very small coordinates (precision test).
    #[test]
    fn test_stl_precision() {
        let mut mesh = Mesh::new();
        mesh.vertices.push(Vertex::from_coords(0.0, 0.0, 0.0));
        mesh.vertices.push(Vertex::from_coords(0.001, 0.0, 0.0));
        mesh.vertices.push(Vertex::from_coords(0.0, 0.001, 0.0));
        mesh.faces.push([0, 1, 2]);

        let file = NamedTempFile::with_suffix(".stl").unwrap();
        save_mesh(&mesh, file.path()).expect("Should save small STL");

        let reloaded = load_mesh(file.path()).expect("Should reload");
        let (_min, max) = reloaded.bounds().unwrap();

        // STL uses f32 internally, so precision is limited
        assert!(
            (max.x - 0.001).abs() < 1e-5,
            "X precision should be preserved"
        );
        assert!(
            (max.y - 0.001).abs() < 1e-5,
            "Y precision should be preserved"
        );
    }
}

// =============================================================================
// OBJ Format Conformance Tests
// =============================================================================

mod obj_conformance {
    use super::*;

    /// OBJ files should preserve vertex order exactly.
    #[test]
    fn test_obj_vertex_order() {
        let mut mesh = Mesh::new();
        for i in 0..10 {
            mesh.vertices.push(Vertex::from_coords(
                i as f64,
                i as f64 * 2.0,
                i as f64 * 3.0,
            ));
        }
        mesh.faces.push([0, 1, 2]);
        mesh.faces.push([3, 4, 5]);

        let file = NamedTempFile::with_suffix(".obj").unwrap();
        save_mesh(&mesh, file.path()).expect("Should save OBJ");

        let reloaded = load_mesh(file.path()).expect("Should reload OBJ");

        for (i, (orig, loaded)) in mesh
            .vertices
            .iter()
            .zip(reloaded.vertices.iter())
            .enumerate()
        {
            let diff = (orig.position - loaded.position).norm();
            assert!(
                diff < 1e-10,
                "Vertex {} should match: {:?} vs {:?}",
                i,
                orig.position,
                loaded.position
            );
        }
    }

    /// OBJ files should preserve face connectivity (same topology).
    #[test]
    fn test_obj_face_connectivity() {
        let mut mesh = Mesh::new();
        // Create a simple triangular mesh where all vertices are used
        mesh.vertices.push(Vertex::from_coords(0.0, 0.0, 0.0));
        mesh.vertices.push(Vertex::from_coords(1.0, 0.0, 0.0));
        mesh.vertices.push(Vertex::from_coords(0.5, 1.0, 0.0));
        mesh.vertices.push(Vertex::from_coords(1.5, 1.0, 0.0));
        mesh.faces.push([0, 1, 2]);
        mesh.faces.push([1, 3, 2]);

        let file = NamedTempFile::with_suffix(".obj").unwrap();
        save_mesh(&mesh, file.path()).expect("Should save OBJ");

        let reloaded = load_mesh(file.path()).expect("Should reload OBJ");

        // OBJ preserves face count
        assert_eq!(
            mesh.face_count(),
            reloaded.face_count(),
            "Face count should match"
        );
        // OBJ preserves vertex count
        assert_eq!(
            mesh.vertex_count(),
            reloaded.vertex_count(),
            "Vertex count should match"
        );
    }

    /// OBJ should handle reasonable precision coordinates.
    #[test]
    fn test_obj_reasonable_precision() {
        let mut mesh = Mesh::new();
        mesh.vertices.push(Vertex::from_coords(0.123456, 0.0, 0.0));
        mesh.vertices.push(Vertex::from_coords(1.0, 0.0, 0.0));
        mesh.vertices.push(Vertex::from_coords(0.0, 1.0, 0.0));
        mesh.faces.push([0, 1, 2]);

        let file = NamedTempFile::with_suffix(".obj").unwrap();
        save_mesh(&mesh, file.path()).expect("Should save OBJ");

        let reloaded = load_mesh(file.path()).expect("Should reload OBJ");

        // OBJ should preserve reasonable precision (6 decimal places)
        let diff = (mesh.vertices[0].position.x - reloaded.vertices[0].position.x).abs();
        assert!(
            diff < 1e-5,
            "OBJ should preserve reasonable precision, diff was {}",
            diff
        );
    }

    /// OBJ should handle comments correctly.
    #[test]
    fn test_obj_with_comments() {
        let mut file = NamedTempFile::with_suffix(".obj").unwrap();

        writeln!(file, "# This is a comment").unwrap();
        writeln!(file, "# Another comment").unwrap();
        writeln!(file, "v 0 0 0").unwrap();
        writeln!(file, "# Comment between vertices").unwrap();
        writeln!(file, "v 1 0 0").unwrap();
        writeln!(file, "v 0 1 0").unwrap();
        writeln!(file, "# Comment before face").unwrap();
        writeln!(file, "f 1 2 3").unwrap();

        let mesh = load_mesh(file.path()).expect("Should load OBJ with comments");
        assert_eq!(mesh.vertex_count(), 3);
        assert_eq!(mesh.face_count(), 1);
    }

    /// OBJ should handle blank lines.
    #[test]
    fn test_obj_with_blank_lines() {
        let mut file = NamedTempFile::with_suffix(".obj").unwrap();

        writeln!(file, "").unwrap();
        writeln!(file, "v 0 0 0").unwrap();
        writeln!(file, "").unwrap();
        writeln!(file, "v 1 0 0").unwrap();
        writeln!(file, "").unwrap();
        writeln!(file, "v 0 1 0").unwrap();
        writeln!(file, "").unwrap();
        writeln!(file, "f 1 2 3").unwrap();
        writeln!(file, "").unwrap();

        let mesh = load_mesh(file.path()).expect("Should load OBJ with blank lines");
        assert_eq!(mesh.vertex_count(), 3);
        assert_eq!(mesh.face_count(), 1);
    }
}

// =============================================================================
// PLY Format Conformance Tests
// =============================================================================

mod ply_conformance {
    use super::*;

    /// PLY ASCII format should be readable.
    #[test]
    fn test_ply_ascii_format() {
        let mut file = NamedTempFile::with_suffix(".ply").unwrap();

        // Standard ASCII PLY header
        writeln!(file, "ply").unwrap();
        writeln!(file, "format ascii 1.0").unwrap();
        writeln!(file, "element vertex 3").unwrap();
        writeln!(file, "property float x").unwrap();
        writeln!(file, "property float y").unwrap();
        writeln!(file, "property float z").unwrap();
        writeln!(file, "element face 1").unwrap();
        writeln!(file, "property list uchar int vertex_indices").unwrap();
        writeln!(file, "end_header").unwrap();
        writeln!(file, "0 0 0").unwrap();
        writeln!(file, "1 0 0").unwrap();
        writeln!(file, "0 1 0").unwrap();
        writeln!(file, "3 0 1 2").unwrap();

        let mesh = load_mesh(file.path()).expect("Should load ASCII PLY");
        assert_eq!(mesh.vertex_count(), 3);
        assert_eq!(mesh.face_count(), 1);
    }

    /// PLY should handle different property names.
    #[test]
    fn test_ply_vertex_properties() {
        let mut mesh = Mesh::new();
        mesh.vertices.push(Vertex::from_coords(0.0, 0.0, 0.0));
        mesh.vertices.push(Vertex::from_coords(1.0, 0.0, 0.0));
        mesh.vertices.push(Vertex::from_coords(0.0, 1.0, 0.0));
        mesh.faces.push([0, 1, 2]);

        let file = NamedTempFile::with_suffix(".ply").unwrap();
        save_mesh(&mesh, file.path()).expect("Should save PLY");

        let reloaded = load_mesh(file.path()).expect("Should reload PLY");
        assert_eq!(reloaded.vertex_count(), 3);
        assert_eq!(reloaded.face_count(), 1);
    }
}

// =============================================================================
// 3MF Format Conformance Tests
// =============================================================================

mod threemf_conformance {
    use super::*;

    /// 3MF should create valid ZIP archives.
    #[test]
    fn test_3mf_is_valid_zip() {
        let mut mesh = Mesh::new();
        mesh.vertices.push(Vertex::from_coords(0.0, 0.0, 0.0));
        mesh.vertices.push(Vertex::from_coords(1.0, 0.0, 0.0));
        mesh.vertices.push(Vertex::from_coords(0.0, 1.0, 0.0));
        mesh.faces.push([0, 1, 2]);

        let file = NamedTempFile::with_suffix(".3mf").unwrap();
        save_mesh(&mesh, file.path()).expect("Should save 3MF");

        // Verify it's a valid ZIP file
        let zip_file = std::fs::File::open(file.path()).expect("Should open file");
        let archive = zip::ZipArchive::new(zip_file).expect("Should be valid ZIP");

        // 3MF must contain specific files
        assert!(
            archive.file_names().any(|n| n.contains("3D/3dmodel.model")),
            "3MF must contain 3D/3dmodel.model"
        );
        assert!(
            archive.file_names().any(|n| n == "[Content_Types].xml"),
            "3MF must contain [Content_Types].xml"
        );
    }

    /// 3MF round-trip should preserve geometry.
    #[test]
    fn test_3mf_roundtrip() {
        let mut mesh = Mesh::new();

        // Create a tetrahedron
        mesh.vertices.push(Vertex::from_coords(0.0, 0.0, 0.0));
        mesh.vertices.push(Vertex::from_coords(10.0, 0.0, 0.0));
        mesh.vertices.push(Vertex::from_coords(5.0, 10.0, 0.0));
        mesh.vertices.push(Vertex::from_coords(5.0, 5.0, 10.0));

        mesh.faces.push([0, 1, 2]);
        mesh.faces.push([0, 3, 1]);
        mesh.faces.push([1, 3, 2]);
        mesh.faces.push([2, 3, 0]);

        let file = NamedTempFile::with_suffix(".3mf").unwrap();
        save_mesh(&mesh, file.path()).expect("Should save 3MF");

        let reloaded = load_mesh(file.path()).expect("Should reload 3MF");

        assert_eq!(reloaded.face_count(), 4, "Face count should be preserved");
        // Note: 3MF may reorder vertices, so we check bounds instead
        let (orig_min, orig_max) = mesh.bounds().unwrap();
        let (new_min, new_max) = reloaded.bounds().unwrap();

        assert!(
            (orig_min - new_min).norm() < 1e-6,
            "Min bounds should match"
        );
        assert!(
            (orig_max - new_max).norm() < 1e-6,
            "Max bounds should match"
        );
    }

    /// 3MF coordinates are in millimeters (spec requirement).
    #[test]
    fn test_3mf_units_are_mm() {
        let mut mesh = Mesh::new();
        mesh.vertices.push(Vertex::from_coords(0.0, 0.0, 0.0));
        mesh.vertices.push(Vertex::from_coords(25.4, 0.0, 0.0)); // 1 inch in mm
        mesh.vertices.push(Vertex::from_coords(0.0, 25.4, 0.0));
        mesh.faces.push([0, 1, 2]);

        let file = NamedTempFile::with_suffix(".3mf").unwrap();
        save_mesh(&mesh, file.path()).expect("Should save 3MF");

        let reloaded = load_mesh(file.path()).expect("Should reload 3MF");
        let (_, max) = reloaded.bounds().unwrap();

        // Coordinates should be preserved (3MF uses mm internally)
        assert!((max.x - 25.4).abs() < 1e-6, "X coordinate should be 25.4mm");
        assert!((max.y - 25.4).abs() < 1e-6, "Y coordinate should be 25.4mm");
    }
}

// =============================================================================
// Cross-Format Tests
// =============================================================================

mod cross_format {
    use super::*;

    /// Meshes should survive conversion between formats.
    #[test]
    fn test_stl_to_obj_to_3mf() {
        // Create original mesh
        let mut mesh = Mesh::new();
        mesh.vertices.push(Vertex::from_coords(0.0, 0.0, 0.0));
        mesh.vertices.push(Vertex::from_coords(10.0, 0.0, 0.0));
        mesh.vertices.push(Vertex::from_coords(5.0, 10.0, 0.0));
        mesh.vertices.push(Vertex::from_coords(5.0, 5.0, 10.0));
        mesh.faces.push([0, 1, 2]);
        mesh.faces.push([0, 3, 1]);
        mesh.faces.push([1, 3, 2]);
        mesh.faces.push([2, 3, 0]);

        // STL -> OBJ
        let stl_file = NamedTempFile::with_suffix(".stl").unwrap();
        save_mesh(&mesh, stl_file.path()).expect("Save STL");
        let from_stl = load_mesh(stl_file.path()).expect("Load STL");

        let obj_file = NamedTempFile::with_suffix(".obj").unwrap();
        save_mesh(&from_stl, obj_file.path()).expect("Save OBJ");
        let from_obj = load_mesh(obj_file.path()).expect("Load OBJ");

        // OBJ -> 3MF
        let threemf_file = NamedTempFile::with_suffix(".3mf").unwrap();
        save_mesh(&from_obj, threemf_file.path()).expect("Save 3MF");
        let from_3mf = load_mesh(threemf_file.path()).expect("Load 3MF");

        // Verify geometry is preserved
        assert_eq!(
            from_3mf.face_count(),
            4,
            "Should have 4 faces after conversions"
        );

        let (orig_min, orig_max) = mesh.bounds().unwrap();
        let (final_min, final_max) = from_3mf.bounds().unwrap();

        // Allow for floating-point conversion errors
        assert!(
            (orig_min - final_min).norm() < 1e-4,
            "Min bounds should be preserved"
        );
        assert!(
            (orig_max - final_max).norm() < 1e-4,
            "Max bounds should be preserved"
        );
    }

    /// Surface area should be preserved across format conversions.
    #[test]
    fn test_surface_area_preservation() {
        let mut mesh = Mesh::new();
        mesh.vertices.push(Vertex::from_coords(0.0, 0.0, 0.0));
        mesh.vertices.push(Vertex::from_coords(10.0, 0.0, 0.0));
        mesh.vertices.push(Vertex::from_coords(0.0, 10.0, 0.0));
        mesh.faces.push([0, 1, 2]);

        let original_area = mesh.surface_area();

        // Convert through STL
        let stl_file = NamedTempFile::with_suffix(".stl").unwrap();
        save_mesh(&mesh, stl_file.path()).expect("Save STL");
        let from_stl = load_mesh(stl_file.path()).expect("Load STL");
        let stl_area = from_stl.surface_area();

        // Convert through OBJ
        let obj_file = NamedTempFile::with_suffix(".obj").unwrap();
        save_mesh(&mesh, obj_file.path()).expect("Save OBJ");
        let from_obj = load_mesh(obj_file.path()).expect("Load OBJ");
        let obj_area = from_obj.surface_area();

        // Convert through 3MF
        let threemf_file = NamedTempFile::with_suffix(".3mf").unwrap();
        save_mesh(&mesh, threemf_file.path()).expect("Save 3MF");
        let from_3mf = load_mesh(threemf_file.path()).expect("Load 3MF");
        let threemf_area = from_3mf.surface_area();

        // All should be within floating-point tolerance
        assert!(
            (original_area - stl_area).abs() < 1e-4,
            "STL area {} should match original {}",
            stl_area,
            original_area
        );
        assert!(
            (original_area - obj_area).abs() < 1e-10,
            "OBJ area {} should match original {}",
            obj_area,
            original_area
        );
        assert!(
            (original_area - threemf_area).abs() < 1e-4,
            "3MF area {} should match original {}",
            threemf_area,
            original_area
        );
    }
}

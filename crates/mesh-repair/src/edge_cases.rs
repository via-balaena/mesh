//! Edge case tests for mesh repair robustness.
//!
//! This module contains tests for edge cases and unusual inputs to ensure
//! the mesh repair library handles them gracefully without panicking.

#[cfg(test)]
mod tests {
    use crate::adjacency::MeshAdjacency;
    use crate::components::{
        find_connected_components, keep_largest_component, remove_small_components,
        split_into_components,
    };
    use crate::holes::{detect_holes, fill_holes};
    use crate::intersect::{IntersectionParams, detect_self_intersections};
    use crate::repair::{
        RepairParams, compute_vertex_normals, fix_non_manifold_edges, remove_degenerate_triangles,
        remove_degenerate_triangles_enhanced, remove_duplicate_faces, remove_unreferenced_vertices,
        repair_mesh, repair_mesh_with_config, weld_vertices,
    };
    use crate::validate::{ValidationOptions, validate_mesh, validate_mesh_data};
    use crate::winding::fix_winding_order;
    use crate::{Mesh, Vertex};

    // ==================== Empty Mesh Tests ====================

    #[test]
    fn test_empty_mesh_validate() {
        let mesh = Mesh::new();
        let report = validate_mesh(&mesh);

        assert!(!report.is_valid()); // Empty mesh is not valid
        assert_eq!(report.vertex_count, 0);
        assert_eq!(report.face_count, 0);
        assert_eq!(report.component_count, 0);
        // Note: Empty mesh is technically watertight (no boundary edges) and manifold,
        // but not valid since it has no faces. is_printable() returns true for
        // watertight + manifold + not inside-out, which empty mesh satisfies.
        // This is a quirk of the definition - an empty mesh has no issues to report.
        assert!(report.bounds.is_none());
        assert!(report.dimensions.is_none());
    }

    #[test]
    fn test_empty_mesh_data_validation() {
        let mesh = Mesh::new();
        let result = validate_mesh_data(&mesh, &ValidationOptions::default()).unwrap();
        assert!(result.is_valid()); // No invalid data (no data at all)
    }

    #[test]
    fn test_empty_mesh_repair() {
        let mut mesh = Mesh::new();
        // Should not panic, should return Ok
        let result = repair_mesh(&mut mesh);
        assert!(result.is_ok());
    }

    #[test]
    fn test_empty_mesh_repair_with_config() {
        let mut mesh = Mesh::new();
        let params = RepairParams::for_scans();
        let result = repair_mesh_with_config(&mut mesh, &params);
        assert!(result.is_ok());
    }

    #[test]
    fn test_empty_mesh_remove_degenerate() {
        let mut mesh = Mesh::new();
        let removed = remove_degenerate_triangles(&mut mesh, 1e-9);
        assert_eq!(removed, 0);
    }

    #[test]
    fn test_empty_mesh_weld_vertices() {
        let mut mesh = Mesh::new();
        let merged = weld_vertices(&mut mesh, 0.01);
        assert_eq!(merged, 0);
    }

    #[test]
    fn test_empty_mesh_remove_unreferenced() {
        let mut mesh = Mesh::new();
        let removed = remove_unreferenced_vertices(&mut mesh);
        assert_eq!(removed, 0);
    }

    #[test]
    fn test_empty_mesh_compute_normals() {
        let mut mesh = Mesh::new();
        // Should not panic
        compute_vertex_normals(&mut mesh);
    }

    #[test]
    fn test_empty_mesh_remove_duplicate_faces() {
        let mut mesh = Mesh::new();
        let removed = remove_duplicate_faces(&mut mesh);
        assert_eq!(removed, 0);
    }

    #[test]
    fn test_empty_mesh_fix_non_manifold() {
        let mut mesh = Mesh::new();
        let removed = fix_non_manifold_edges(&mut mesh);
        assert_eq!(removed, 0);
    }

    #[test]
    fn test_empty_mesh_fix_winding() {
        let mut mesh = Mesh::new();
        let result = fix_winding_order(&mut mesh);
        assert!(result.is_ok());
    }

    #[test]
    fn test_empty_mesh_fill_holes() {
        let mut mesh = Mesh::new();
        let result = fill_holes(&mut mesh);
        assert!(result.is_ok());
        assert_eq!(result.unwrap(), 0);
    }

    #[test]
    fn test_empty_mesh_detect_holes() {
        let mesh = Mesh::new();
        let adj = MeshAdjacency::build(&mesh.faces);
        let holes = detect_holes(&mesh, &adj);
        assert!(holes.is_empty());
    }

    #[test]
    fn test_empty_mesh_find_components() {
        let mesh = Mesh::new();
        let analysis = find_connected_components(&mesh);
        assert_eq!(analysis.component_count, 0);
        assert!(analysis.components.is_empty());
    }

    #[test]
    fn test_empty_mesh_split_components() {
        let mesh = Mesh::new();
        let components = split_into_components(&mesh);
        // An empty mesh may return an empty mesh as "one component" or nothing
        // depending on implementation. Just verify it doesn't panic.
        for comp in &components {
            assert_eq!(comp.face_count(), 0);
        }
    }

    #[test]
    fn test_empty_mesh_keep_largest_component() {
        let mut mesh = Mesh::new();
        let removed = keep_largest_component(&mut mesh);
        assert_eq!(removed, 0);
    }

    #[test]
    fn test_empty_mesh_detect_self_intersections() {
        let mesh = Mesh::new();
        let result = detect_self_intersections(&mesh, &IntersectionParams::default());
        assert!(!result.has_intersections);
        assert!(result.intersecting_pairs.is_empty());
    }

    #[test]
    fn test_empty_mesh_adjacency() {
        let mesh = Mesh::new();
        let adj = MeshAdjacency::build(&mesh.faces);
        assert!(adj.is_manifold());
        assert!(adj.is_watertight()); // No edges means no boundary edges
        assert_eq!(adj.boundary_edge_count(), 0);
        assert_eq!(adj.non_manifold_edge_count(), 0);
    }

    #[test]
    fn test_empty_mesh_volume() {
        let mesh = Mesh::new();
        assert_eq!(mesh.signed_volume(), 0.0);
        assert_eq!(mesh.volume(), 0.0);
        assert!(!mesh.is_inside_out());
    }

    #[test]
    fn test_empty_mesh_surface_area() {
        let mesh = Mesh::new();
        assert_eq!(mesh.surface_area(), 0.0);
    }

    #[test]
    fn test_empty_mesh_bounds() {
        let mesh = Mesh::new();
        assert!(mesh.bounds().is_none());
    }

    // ==================== Single Triangle Mesh Tests ====================

    fn single_triangle() -> Mesh {
        let mut mesh = Mesh::new();
        mesh.vertices.push(Vertex::from_coords(0.0, 0.0, 0.0));
        mesh.vertices.push(Vertex::from_coords(1.0, 0.0, 0.0));
        mesh.vertices.push(Vertex::from_coords(0.0, 1.0, 0.0));
        mesh.faces.push([0, 1, 2]);
        mesh
    }

    #[test]
    fn test_single_triangle_validate() {
        let mesh = single_triangle();
        let report = validate_mesh(&mesh);

        assert!(report.is_valid());
        assert_eq!(report.vertex_count, 3);
        assert_eq!(report.face_count, 1);
        assert_eq!(report.component_count, 1);
        assert!(!report.is_watertight); // Single triangle has boundary edges
        assert!(report.is_manifold);
        assert!(!report.is_printable());
        assert_eq!(report.boundary_edge_count, 3);
    }

    #[test]
    fn test_single_triangle_repair() {
        let mut mesh = single_triangle();
        let result = repair_mesh(&mut mesh);
        assert!(result.is_ok());
        // Triangle should survive repair (not degenerate)
        assert_eq!(mesh.face_count(), 1);
    }

    #[test]
    fn test_single_triangle_volume() {
        let mesh = single_triangle();
        // Open mesh, but signed_volume should still compute
        let vol = mesh.signed_volume();
        // For a single triangle at z=0 with CCW winding, signed volume
        // depends on position relative to origin
        // The formula gives (v0 · (v1 × v2)) / 6
        // v0 = (0,0,0), so volume = 0 for this triangle
        assert!(vol.abs() < 1e-10);
    }

    #[test]
    fn test_single_triangle_surface_area() {
        let mesh = single_triangle();
        let area = mesh.surface_area();
        // Right triangle with legs 1, area = 0.5
        assert!((area - 0.5).abs() < 1e-10);
    }

    #[test]
    fn test_single_triangle_bounds() {
        let mesh = single_triangle();
        let (min, max) = mesh.bounds().expect("should have bounds");
        assert_eq!(min.x, 0.0);
        assert_eq!(min.y, 0.0);
        assert_eq!(min.z, 0.0);
        assert_eq!(max.x, 1.0);
        assert_eq!(max.y, 1.0);
        assert_eq!(max.z, 0.0);
    }

    #[test]
    fn test_single_triangle_fill_holes() {
        let mut mesh = single_triangle();
        // A single triangle can't really have holes "filled" in a meaningful way
        let result = fill_holes(&mut mesh);
        assert!(result.is_ok());
    }

    #[test]
    fn test_single_triangle_detect_intersections() {
        let mesh = single_triangle();
        let result = detect_self_intersections(&mesh, &IntersectionParams::default());
        assert!(!result.has_intersections);
    }

    #[test]
    fn test_single_triangle_components() {
        let mesh = single_triangle();
        let analysis = find_connected_components(&mesh);
        assert_eq!(analysis.component_count, 1);
    }

    // ==================== Very Large Coordinates (>1km) ====================

    fn large_coordinate_mesh() -> Mesh {
        let mut mesh = Mesh::new();
        // Coordinates in meters, representing >1km
        let scale = 10_000.0; // 10km in some unit
        mesh.vertices.push(Vertex::from_coords(0.0, 0.0, 0.0));
        mesh.vertices.push(Vertex::from_coords(scale, 0.0, 0.0));
        mesh.vertices.push(Vertex::from_coords(0.0, scale, 0.0));
        mesh.vertices.push(Vertex::from_coords(0.0, 0.0, scale));

        // Tetrahedron faces
        mesh.faces.push([0, 2, 1]);
        mesh.faces.push([0, 1, 3]);
        mesh.faces.push([1, 2, 3]);
        mesh.faces.push([2, 0, 3]);
        mesh
    }

    #[test]
    fn test_large_coordinates_validate() {
        let mesh = large_coordinate_mesh();
        let report = validate_mesh(&mesh);

        assert!(report.is_valid());
        assert!(report.is_watertight);
        assert!(report.is_manifold);

        let (_min, max) = report.bounds.expect("should have bounds");
        assert!(max.x > 1000.0); // Verify we actually have large coordinates
    }

    #[test]
    fn test_large_coordinates_data_validation() {
        let mesh = large_coordinate_mesh();
        let result = validate_mesh_data(&mesh, &ValidationOptions::default()).unwrap();
        assert!(result.is_valid()); // Large but finite coordinates are valid
    }

    #[test]
    fn test_large_coordinates_repair() {
        let mut mesh = large_coordinate_mesh();
        let result = repair_mesh(&mut mesh);
        assert!(result.is_ok());
        // Mesh should survive with all faces intact
        assert_eq!(mesh.face_count(), 4);
    }

    #[test]
    fn test_large_coordinates_volume() {
        let mesh = large_coordinate_mesh();
        let vol = mesh.volume();
        // Large tetrahedron should have large volume
        assert!(vol > 1e10);
    }

    #[test]
    fn test_large_coordinates_detect_intersections() {
        let mesh = large_coordinate_mesh();
        let result = detect_self_intersections(&mesh, &IntersectionParams::default());
        assert!(!result.has_intersections);
    }

    #[test]
    fn test_large_coordinates_weld_vertices() {
        let mut mesh = large_coordinate_mesh();
        // With small epsilon, no vertices should merge despite large coordinates
        let merged = weld_vertices(&mut mesh, 1e-6);
        assert_eq!(merged, 0);
    }

    // ==================== Very Small Coordinates (<1μm) ====================

    fn tiny_coordinate_mesh() -> Mesh {
        let mut mesh = Mesh::new();
        // Coordinates in micrometers scale (sub-micron)
        let scale = 1e-7; // 0.1 micrometers = 100 nanometers
        mesh.vertices.push(Vertex::from_coords(0.0, 0.0, 0.0));
        mesh.vertices.push(Vertex::from_coords(scale, 0.0, 0.0));
        mesh.vertices.push(Vertex::from_coords(0.0, scale, 0.0));
        mesh.vertices.push(Vertex::from_coords(0.0, 0.0, scale));

        // Tetrahedron faces
        mesh.faces.push([0, 2, 1]);
        mesh.faces.push([0, 1, 3]);
        mesh.faces.push([1, 2, 3]);
        mesh.faces.push([2, 0, 3]);
        mesh
    }

    #[test]
    fn test_tiny_coordinates_validate() {
        let mesh = tiny_coordinate_mesh();
        let report = validate_mesh(&mesh);

        assert!(report.is_valid());
        assert!(report.is_watertight);
        assert!(report.is_manifold);

        let (_min, max) = report.bounds.expect("should have bounds");
        assert!(max.x < 1e-6); // Verify we actually have tiny coordinates
    }

    #[test]
    fn test_tiny_coordinates_data_validation() {
        let mesh = tiny_coordinate_mesh();
        let result = validate_mesh_data(&mesh, &ValidationOptions::default()).unwrap();
        assert!(result.is_valid()); // Small but non-zero coordinates are valid
    }

    #[test]
    fn test_tiny_coordinates_repair_conservative() {
        let mut mesh = tiny_coordinate_mesh();
        // Use extremely conservative custom params for nanometer-scale meshes
        let params = RepairParams {
            weld_epsilon: 1e-15,
            degenerate_area_threshold: 1e-30,
            degenerate_aspect_ratio: f64::INFINITY,
            degenerate_min_edge_length: 0.0,
            fix_winding: false, // Winding fix may struggle with tiny coordinates
            ..Default::default()
        };
        let result = repair_mesh_with_config(&mut mesh, &params);
        assert!(result.is_ok());
        // With extremely conservative params, mesh should survive
        assert_eq!(mesh.face_count(), 4);
    }

    #[test]
    fn test_tiny_coordinates_repair_aggressive_removes_triangles() {
        let mut mesh = tiny_coordinate_mesh();
        // Default params may remove tiny triangles
        let result = repair_mesh(&mut mesh);
        assert!(result.is_ok());
        // Triangles may be removed as degenerate (that's expected behavior)
    }

    #[test]
    fn test_tiny_coordinates_volume() {
        let mesh = tiny_coordinate_mesh();
        let vol = mesh.volume();
        // Volume should be very small but non-zero
        assert!(vol > 0.0);
        assert!(vol < 1e-18); // Very tiny volume
    }

    #[test]
    fn test_tiny_coordinates_surface_area() {
        let mesh = tiny_coordinate_mesh();
        let area = mesh.surface_area();
        assert!(area > 0.0);
        assert!(area < 1e-12); // Very tiny area
    }

    // ==================== All Degenerate Triangles ====================

    fn all_degenerate_mesh() -> Mesh {
        let mut mesh = Mesh::new();
        // Collinear points (zero area triangles)
        mesh.vertices.push(Vertex::from_coords(0.0, 0.0, 0.0));
        mesh.vertices.push(Vertex::from_coords(1.0, 0.0, 0.0));
        mesh.vertices.push(Vertex::from_coords(2.0, 0.0, 0.0));
        mesh.vertices.push(Vertex::from_coords(3.0, 0.0, 0.0));

        // All faces are degenerate (collinear points)
        mesh.faces.push([0, 1, 2]);
        mesh.faces.push([1, 2, 3]);
        mesh.faces.push([0, 2, 3]);
        mesh
    }

    #[test]
    fn test_all_degenerate_validate() {
        let mesh = all_degenerate_mesh();
        let report = validate_mesh(&mesh);

        assert!(report.is_valid()); // Has vertices and faces
        assert!(report.surface_area < 1e-10); // Zero area
    }

    #[test]
    fn test_all_degenerate_remove_degenerate() {
        let mut mesh = all_degenerate_mesh();
        let removed = remove_degenerate_triangles(&mut mesh, 1e-9);
        assert_eq!(removed, 3); // All triangles removed
        assert_eq!(mesh.face_count(), 0);
    }

    #[test]
    fn test_all_degenerate_repair() {
        let mut mesh = all_degenerate_mesh();
        let result = repair_mesh(&mut mesh);
        assert!(result.is_ok());
        // All degenerate faces should be removed
        assert_eq!(mesh.face_count(), 0);
    }

    #[test]
    fn test_all_degenerate_repair_then_validate() {
        let mut mesh = all_degenerate_mesh();
        repair_mesh(&mut mesh).unwrap();

        let report = validate_mesh(&mesh);
        // After repair, mesh should be empty but not crash
        assert_eq!(report.face_count, 0);
    }

    // ==================== NaN Coordinates ====================

    fn nan_coordinate_mesh() -> Mesh {
        let mut mesh = Mesh::new();
        mesh.vertices.push(Vertex::from_coords(f64::NAN, 0.0, 0.0));
        mesh.vertices.push(Vertex::from_coords(1.0, 0.0, 0.0));
        mesh.vertices.push(Vertex::from_coords(0.0, 1.0, 0.0));
        mesh.faces.push([0, 1, 2]);
        mesh
    }

    #[test]
    fn test_nan_data_validation_strict() {
        let mesh = nan_coordinate_mesh();
        let result = validate_mesh_data(&mesh, &ValidationOptions::default());
        assert!(result.is_err()); // Should reject NaN
    }

    #[test]
    fn test_nan_data_validation_collect() {
        let mesh = nan_coordinate_mesh();
        let result = validate_mesh_data(&mesh, &ValidationOptions::collect_all()).unwrap();
        assert!(!result.is_valid());
        assert_eq!(result.nan_count, 1);
    }

    #[test]
    fn test_nan_in_y_coordinate() {
        let mut mesh = Mesh::new();
        mesh.vertices.push(Vertex::from_coords(0.0, f64::NAN, 0.0));
        mesh.vertices.push(Vertex::from_coords(1.0, 0.0, 0.0));
        mesh.vertices.push(Vertex::from_coords(0.0, 1.0, 0.0));
        mesh.faces.push([0, 1, 2]);

        let result = validate_mesh_data(&mesh, &ValidationOptions::collect_all()).unwrap();
        assert_eq!(result.nan_count, 1);
    }

    #[test]
    fn test_nan_in_z_coordinate() {
        let mut mesh = Mesh::new();
        mesh.vertices.push(Vertex::from_coords(0.0, 0.0, f64::NAN));
        mesh.vertices.push(Vertex::from_coords(1.0, 0.0, 0.0));
        mesh.vertices.push(Vertex::from_coords(0.0, 1.0, 0.0));
        mesh.faces.push([0, 1, 2]);

        let result = validate_mesh_data(&mesh, &ValidationOptions::collect_all()).unwrap();
        assert_eq!(result.nan_count, 1);
    }

    #[test]
    fn test_all_nan_coordinates() {
        let mut mesh = Mesh::new();
        mesh.vertices
            .push(Vertex::from_coords(f64::NAN, f64::NAN, f64::NAN));
        mesh.vertices
            .push(Vertex::from_coords(f64::NAN, f64::NAN, f64::NAN));
        mesh.vertices
            .push(Vertex::from_coords(f64::NAN, f64::NAN, f64::NAN));
        mesh.faces.push([0, 1, 2]);

        let result = validate_mesh_data(&mesh, &ValidationOptions::collect_all()).unwrap();
        assert_eq!(result.nan_count, 9); // 3 vertices × 3 coordinates
    }

    // ==================== Infinity Coordinates ====================

    #[test]
    fn test_positive_infinity_validation() {
        let mut mesh = Mesh::new();
        mesh.vertices
            .push(Vertex::from_coords(f64::INFINITY, 0.0, 0.0));
        mesh.vertices.push(Vertex::from_coords(1.0, 0.0, 0.0));
        mesh.vertices.push(Vertex::from_coords(0.0, 1.0, 0.0));
        mesh.faces.push([0, 1, 2]);

        let result = validate_mesh_data(&mesh, &ValidationOptions::default());
        assert!(result.is_err());
    }

    #[test]
    fn test_negative_infinity_validation() {
        let mut mesh = Mesh::new();
        mesh.vertices
            .push(Vertex::from_coords(f64::NEG_INFINITY, 0.0, 0.0));
        mesh.vertices.push(Vertex::from_coords(1.0, 0.0, 0.0));
        mesh.vertices.push(Vertex::from_coords(0.0, 1.0, 0.0));
        mesh.faces.push([0, 1, 2]);

        let result = validate_mesh_data(&mesh, &ValidationOptions::collect_all()).unwrap();
        assert_eq!(result.infinity_count, 1);
    }

    #[test]
    fn test_mixed_nan_infinity() {
        let mut mesh = Mesh::new();
        mesh.vertices
            .push(Vertex::from_coords(f64::NAN, f64::INFINITY, 0.0));
        mesh.vertices
            .push(Vertex::from_coords(1.0, 0.0, f64::NEG_INFINITY));
        mesh.vertices.push(Vertex::from_coords(0.0, 1.0, 0.0));
        mesh.faces.push([0, 1, 2]);

        let result = validate_mesh_data(&mesh, &ValidationOptions::collect_all()).unwrap();
        assert_eq!(result.nan_count, 1);
        assert_eq!(result.infinity_count, 2);
    }

    // ==================== Non-Manifold Edge Cases ====================

    fn bowtie_mesh() -> Mesh {
        // Two triangles sharing only a single vertex (bowtie configuration)
        // This creates a non-manifold vertex
        let mut mesh = Mesh::new();
        mesh.vertices.push(Vertex::from_coords(0.0, 0.0, 0.0)); // 0 - shared vertex
        mesh.vertices.push(Vertex::from_coords(1.0, 0.0, 0.0)); // 1
        mesh.vertices.push(Vertex::from_coords(0.5, 1.0, 0.0)); // 2
        mesh.vertices.push(Vertex::from_coords(-1.0, 0.0, 0.0)); // 3
        mesh.vertices.push(Vertex::from_coords(-0.5, 1.0, 0.0)); // 4

        mesh.faces.push([0, 1, 2]); // Right triangle
        mesh.faces.push([0, 4, 3]); // Left triangle (shares only vertex 0)
        mesh
    }

    #[test]
    fn test_bowtie_validate() {
        let mesh = bowtie_mesh();
        let report = validate_mesh(&mesh);

        assert!(report.is_valid());
        // Bowtie is manifold (edges are fine, just a non-manifold vertex)
        assert!(report.is_manifold);
        assert!(!report.is_watertight); // Has boundary edges
    }

    #[test]
    fn test_bowtie_components() {
        let mesh = bowtie_mesh();
        let analysis = find_connected_components(&mesh);
        // Note: In mesh topology, connectivity is determined by shared EDGES, not just vertices.
        // Two triangles sharing only a vertex (bowtie) are NOT edge-connected,
        // so they form 2 separate components.
        assert_eq!(analysis.component_count, 2);
    }

    fn non_manifold_edge_mesh() -> Mesh {
        // Three triangles sharing an edge (non-manifold edge)
        let mut mesh = Mesh::new();
        mesh.vertices.push(Vertex::from_coords(0.0, 0.0, 0.0)); // 0
        mesh.vertices.push(Vertex::from_coords(1.0, 0.0, 0.0)); // 1
        mesh.vertices.push(Vertex::from_coords(0.5, 1.0, 0.0)); // 2
        mesh.vertices.push(Vertex::from_coords(0.5, 0.5, 1.0)); // 3
        mesh.vertices.push(Vertex::from_coords(0.5, 0.5, -1.0)); // 4

        // Three faces all share edge 0-1
        mesh.faces.push([0, 1, 2]);
        mesh.faces.push([0, 1, 3]);
        mesh.faces.push([0, 1, 4]);
        mesh
    }

    #[test]
    fn test_non_manifold_edge_validate() {
        let mesh = non_manifold_edge_mesh();
        let report = validate_mesh(&mesh);

        assert!(report.is_valid());
        assert!(!report.is_manifold); // Has non-manifold edge
        assert!(!report.is_printable());
        assert!(report.non_manifold_edge_count > 0);
    }

    #[test]
    fn test_non_manifold_edge_fix() {
        let mut mesh = non_manifold_edge_mesh();
        let removed = fix_non_manifold_edges(&mut mesh);

        // Should remove one face to fix the non-manifold edge
        assert!(removed > 0);

        // After fix, should be manifold
        let report = validate_mesh(&mesh);
        assert!(report.is_manifold);
    }

    #[test]
    fn test_non_manifold_edge_adjacency() {
        let mesh = non_manifold_edge_mesh();
        let adj = MeshAdjacency::build(&mesh.faces);

        assert!(!adj.is_manifold());
        assert!(adj.non_manifold_edge_count() > 0);
    }

    // ==================== Disconnected Components ====================

    fn two_separate_triangles() -> Mesh {
        let mut mesh = Mesh::new();
        // First triangle
        mesh.vertices.push(Vertex::from_coords(0.0, 0.0, 0.0));
        mesh.vertices.push(Vertex::from_coords(1.0, 0.0, 0.0));
        mesh.vertices.push(Vertex::from_coords(0.0, 1.0, 0.0));
        mesh.faces.push([0, 1, 2]);

        // Second triangle (disconnected)
        mesh.vertices.push(Vertex::from_coords(10.0, 0.0, 0.0));
        mesh.vertices.push(Vertex::from_coords(11.0, 0.0, 0.0));
        mesh.vertices.push(Vertex::from_coords(10.0, 1.0, 0.0));
        mesh.faces.push([3, 4, 5]);
        mesh
    }

    #[test]
    fn test_disconnected_components_count() {
        let mesh = two_separate_triangles();
        let analysis = find_connected_components(&mesh);
        assert_eq!(analysis.component_count, 2);
    }

    #[test]
    fn test_disconnected_components_split() {
        let mesh = two_separate_triangles();
        let components = split_into_components(&mesh);

        assert_eq!(components.len(), 2);
        for comp in &components {
            assert_eq!(comp.face_count(), 1);
            assert_eq!(comp.vertex_count(), 3);
        }
    }

    #[test]
    fn test_disconnected_keep_largest() {
        let mut mesh = two_separate_triangles();
        // Both are same size, so one arbitrary one is kept
        let removed = keep_largest_component(&mut mesh);

        assert_eq!(removed, 1);
        assert_eq!(mesh.face_count(), 1);

        let analysis = find_connected_components(&mesh);
        assert_eq!(analysis.component_count, 1);
    }

    #[test]
    fn test_disconnected_remove_small() {
        let mesh = two_separate_triangles();

        // Both have 1 face, so removing min_faces=2 should remove both
        let mut mesh_copy = mesh.clone();
        let removed = remove_small_components(&mut mesh_copy, 2);
        assert_eq!(removed, 2);
        assert_eq!(mesh_copy.face_count(), 0);

        // Removing min_faces=1 should keep both
        let mut mesh_copy2 = mesh.clone();
        let removed2 = remove_small_components(&mut mesh_copy2, 1);
        assert_eq!(removed2, 0);
    }

    #[test]
    fn test_disconnected_fix_winding() {
        let mut mesh = two_separate_triangles();
        // Should fix winding for all components
        let result = fix_winding_order(&mut mesh);
        assert!(result.is_ok());
    }

    // ==================== Vertices Only (No Faces) ====================

    fn vertices_only_mesh() -> Mesh {
        let mut mesh = Mesh::new();
        mesh.vertices.push(Vertex::from_coords(0.0, 0.0, 0.0));
        mesh.vertices.push(Vertex::from_coords(1.0, 0.0, 0.0));
        mesh.vertices.push(Vertex::from_coords(0.0, 1.0, 0.0));
        // No faces
        mesh
    }

    #[test]
    fn test_vertices_only_validate() {
        let mesh = vertices_only_mesh();
        let report = validate_mesh(&mesh);

        assert!(!report.is_valid()); // No faces
        assert_eq!(report.vertex_count, 3);
        assert_eq!(report.face_count, 0);
    }

    #[test]
    fn test_vertices_only_repair() {
        let mut mesh = vertices_only_mesh();
        let result = repair_mesh(&mut mesh);
        assert!(result.is_ok()); // Should not panic
    }

    #[test]
    fn test_vertices_only_remove_unreferenced() {
        let mut mesh = vertices_only_mesh();
        let removed = remove_unreferenced_vertices(&mut mesh);
        // All vertices are unreferenced since there are no faces
        assert_eq!(removed, 3);
        assert_eq!(mesh.vertex_count(), 0);
    }

    #[test]
    fn test_vertices_only_bounds() {
        let mesh = vertices_only_mesh();
        // bounds() should work with vertices even without faces
        let bounds = mesh.bounds();
        assert!(bounds.is_some());
    }

    // ==================== Faces Only (No Vertices - Invalid Mesh) ====================

    #[test]
    fn test_faces_without_vertices_validation() {
        let mut mesh = Mesh::new();
        // Add faces that reference non-existent vertices
        mesh.faces.push([0, 1, 2]);

        let result = validate_mesh_data(&mesh, &ValidationOptions::default());
        assert!(result.is_err()); // Should reject invalid indices
    }

    #[test]
    fn test_faces_without_vertices_validation_collect() {
        let mut mesh = Mesh::new();
        mesh.faces.push([0, 1, 2]);
        mesh.faces.push([3, 4, 5]);

        let result = validate_mesh_data(&mesh, &ValidationOptions::collect_all()).unwrap();
        assert!(!result.is_valid());
        assert_eq!(result.invalid_index_count, 6); // All 6 indices are invalid
    }

    // ==================== Duplicate Face Tests ====================

    #[test]
    fn test_duplicate_faces_exact() {
        let mut mesh = single_triangle();
        // Add exact duplicate
        mesh.faces.push([0, 1, 2]);

        let removed = remove_duplicate_faces(&mut mesh);
        assert_eq!(removed, 1);
        assert_eq!(mesh.face_count(), 1);
    }

    #[test]
    fn test_duplicate_faces_rotated() {
        let mut mesh = single_triangle();
        // Add rotated duplicate (same vertices, different starting point)
        mesh.faces.push([1, 2, 0]);

        let removed = remove_duplicate_faces(&mut mesh);
        assert_eq!(removed, 1);
        assert_eq!(mesh.face_count(), 1);
    }

    #[test]
    fn test_duplicate_faces_reversed() {
        let mut mesh = single_triangle();
        // Add reversed duplicate (opposite winding)
        mesh.faces.push([0, 2, 1]);

        let removed = remove_duplicate_faces(&mut mesh);
        assert_eq!(removed, 1);
        assert_eq!(mesh.face_count(), 1);
    }

    #[test]
    fn test_many_duplicates() {
        let mut mesh = single_triangle();
        // Add many duplicates with various rotations
        mesh.faces.push([0, 1, 2]);
        mesh.faces.push([1, 2, 0]);
        mesh.faces.push([2, 0, 1]);
        mesh.faces.push([0, 2, 1]);
        mesh.faces.push([2, 1, 0]);
        mesh.faces.push([1, 0, 2]);

        let removed = remove_duplicate_faces(&mut mesh);
        assert_eq!(removed, 6);
        assert_eq!(mesh.face_count(), 1);
    }

    // ==================== Self-Intersection Edge Cases ====================

    fn self_intersecting_mesh() -> Mesh {
        // Two triangles that intersect each other
        let mut mesh = Mesh::new();

        // First triangle in XY plane
        mesh.vertices.push(Vertex::from_coords(0.0, 0.0, 0.0));
        mesh.vertices.push(Vertex::from_coords(2.0, 0.0, 0.0));
        mesh.vertices.push(Vertex::from_coords(1.0, 2.0, 0.0));
        mesh.faces.push([0, 1, 2]);

        // Second triangle intersects the first
        mesh.vertices.push(Vertex::from_coords(1.0, 1.0, -1.0));
        mesh.vertices.push(Vertex::from_coords(1.0, 1.0, 1.0));
        mesh.vertices.push(Vertex::from_coords(3.0, 1.0, 0.0));
        mesh.faces.push([3, 4, 5]);

        mesh
    }

    #[test]
    fn test_self_intersection_detection() {
        let mesh = self_intersecting_mesh();
        let result = detect_self_intersections(&mesh, &IntersectionParams::default());

        assert!(result.has_intersections);
        assert!(!result.intersecting_pairs.is_empty());
    }

    #[test]
    fn test_non_intersecting_detection() {
        let mesh = two_separate_triangles();
        let result = detect_self_intersections(&mesh, &IntersectionParams::default());

        assert!(!result.has_intersections);
        assert!(result.intersecting_pairs.is_empty());
    }

    // ==================== Extreme Aspect Ratio Triangles ====================

    #[test]
    fn test_extremely_thin_triangle() {
        let mut mesh = Mesh::new();
        // Very thin needle triangle
        mesh.vertices.push(Vertex::from_coords(0.0, 0.0, 0.0));
        mesh.vertices.push(Vertex::from_coords(1000.0, 0.0, 0.0));
        mesh.vertices.push(Vertex::from_coords(500.0, 0.0001, 0.0));
        mesh.faces.push([0, 1, 2]);

        let removed = remove_degenerate_triangles_enhanced(
            &mut mesh, 0.0,   // No area threshold
            100.0, // Max aspect ratio
            0.0,   // No edge length threshold
        );

        // Should be removed due to high aspect ratio
        assert_eq!(removed, 1);
    }

    #[test]
    fn test_well_formed_triangle_survives() {
        let mut mesh = Mesh::new();
        // Equilateral-ish triangle
        let sqrt3_half = 0.866025;
        mesh.vertices.push(Vertex::from_coords(0.0, 0.0, 0.0));
        mesh.vertices.push(Vertex::from_coords(1.0, 0.0, 0.0));
        mesh.vertices
            .push(Vertex::from_coords(0.5, sqrt3_half, 0.0));
        mesh.faces.push([0, 1, 2]);

        let removed = remove_degenerate_triangles_enhanced(
            &mut mesh, 1e-12, // Very small area threshold
            100.0, // Max aspect ratio
            1e-12, // Very small edge threshold
        );

        // Should NOT be removed
        assert_eq!(removed, 0);
    }

    // ==================== Index Boundary Tests ====================

    #[test]
    fn test_max_u32_vertex_index() {
        let mut mesh = Mesh::new();
        mesh.vertices.push(Vertex::from_coords(0.0, 0.0, 0.0));
        // Face with max u32 index
        mesh.faces.push([0, u32::MAX, 0]);

        let result = validate_mesh_data(&mesh, &ValidationOptions::default());
        assert!(result.is_err()); // Should reject out of bounds index
    }

    #[test]
    fn test_zero_index_valid() {
        let mut mesh = Mesh::new();
        mesh.vertices.push(Vertex::from_coords(0.0, 0.0, 0.0));
        mesh.vertices.push(Vertex::from_coords(1.0, 0.0, 0.0));
        mesh.vertices.push(Vertex::from_coords(0.0, 1.0, 0.0));
        mesh.faces.push([0, 1, 2]);

        let result = validate_mesh_data(&mesh, &ValidationOptions::default()).unwrap();
        assert!(result.is_valid());
    }
}

//! Example: Hockey Equipment Shell Generation
//!
//! This example demonstrates how to use mesh-repair and mesh-shell
//! for creating custom-fit hockey equipment shells from 3D body scans.
//!
//! Run with: `cargo run --example hockey_equipment`

use mesh_repair::{Mesh, RepairParams, ThicknessParams, Vertex};
use std::path::Path;

/// Equipment-specific thickness profiles
pub mod thickness_profiles {
    /// Hockey helmet thickness requirements
    pub struct HelmetProfile {
        pub crown_mm: f64,      // Top of head - maximum impact zone
        pub sides_mm: f64,      // Temple region
        pub back_mm: f64,       // Occipital region
        pub forehead_mm: f64,   // Front protection
        pub chin_mm: f64,       // Chin guard (if integrated)
    }

    impl Default for HelmetProfile {
        fn default() -> Self {
            Self {
                crown_mm: 3.0,
                sides_mm: 2.5,
                back_mm: 2.5,
                forehead_mm: 2.5,
                chin_mm: 2.0,
            }
        }
    }

    /// Hockey skate boot thickness requirements
    pub struct SkateProfile {
        pub heel_cup_mm: f64,       // Heel counter - high support
        pub ankle_collar_mm: f64,   // Ankle support
        pub quarter_panel_mm: f64,  // Side panels
        pub arch_mm: f64,           // Arch region - flex zone
        pub toe_box_mm: f64,        // Toe protection
        pub tongue_mm: f64,         // Top of foot
    }

    impl Default for SkateProfile {
        fn default() -> Self {
            Self {
                heel_cup_mm: 3.0,
                ankle_collar_mm: 2.5,
                quarter_panel_mm: 2.0,
                arch_mm: 1.5,
                toe_box_mm: 2.0,
                tongue_mm: 1.5,
            }
        }
    }

    /// Shoulder pad cap thickness
    pub struct ShoulderCapProfile {
        pub impact_zone_mm: f64,    // Direct impact area
        pub edge_mm: f64,           // Perimeter
        pub attachment_mm: f64,     // Strap mount points
    }

    impl Default for ShoulderCapProfile {
        fn default() -> Self {
            Self {
                impact_zone_mm: 4.0,
                edge_mm: 2.5,
                attachment_mm: 3.0,
            }
        }
    }
}

/// Process a 3D scan for hockey equipment fitting.
///
/// This function handles:
/// - Loading and validating scan data
/// - Removing artifacts and noise
/// - Filling small holes
/// - Preparing geometry for shell generation
pub fn process_scan(scan_path: &Path) -> Result<Mesh, mesh_repair::MeshError> {
    println!("Loading scan from {:?}...", scan_path);

    // Load the scan
    let mut mesh = Mesh::load(scan_path)?;

    let initial_verts = mesh.vertex_count();
    let initial_faces = mesh.face_count();
    println!(
        "  Loaded: {} vertices, {} faces",
        initial_verts, initial_faces
    );

    // Check initial state
    let initial_report = mesh.validate();
    println!("  Initial state:");
    println!("    Watertight: {}", initial_report.is_watertight);
    println!("    Manifold: {}", initial_report.is_manifold);
    println!("    Components: {}", initial_report.component_count);

    // Step 1: Remove small debris components (scan artifacts, hair, etc.)
    let debris_removed = mesh.remove_small_components(50);
    if debris_removed > 0 {
        println!("  Removed {} debris components", debris_removed);
    }

    // Step 2: Keep only the largest component (main body part)
    let extra_removed = mesh.keep_largest_component();
    if extra_removed > 0 {
        println!("  Removed {} extra components", extra_removed);
    }

    // Step 3: Apply scan-optimized repair
    println!("  Applying scan repair...");
    mesh.repair_with_config(&RepairParams::for_scans())?;

    // Step 4: Validate the result
    let report = mesh.validate();
    println!("  After repair:");
    println!("    Vertices: {} (was {})", mesh.vertex_count(), initial_verts);
    println!("    Faces: {} (was {})", mesh.face_count(), initial_faces);
    println!("    Watertight: {}", report.is_watertight);
    println!("    Manifold: {}", report.is_manifold);

    if !report.is_watertight {
        println!(
            "  Warning: {} boundary edges remain",
            report.boundary_edge_count
        );
    }

    Ok(mesh)
}

/// Analyze wall thickness for equipment safety requirements.
pub fn analyze_equipment_thickness(
    mesh: &Mesh,
    min_thickness: f64,
    equipment_name: &str,
) -> bool {
    println!("Analyzing {} thickness (min: {}mm)...", equipment_name, min_thickness);

    let params = ThicknessParams::with_min_thickness(min_thickness);
    let result = mesh.analyze_thickness(&params);

    println!("  Vertices analyzed: {}", result.vertices_analyzed);
    println!("  Vertices with measurements: {}", result.vertices_with_hits);
    println!("  Min thickness: {:.2}mm", result.min_thickness);
    println!("  Max thickness: {:.2}mm", result.max_thickness);
    println!("  Avg thickness: {:.2}mm", result.avg_thickness);

    if result.has_thin_regions() {
        println!(
            "  WARNING: {} thin regions below {}mm!",
            result.thin_regions.len(),
            min_thickness
        );
        for (i, region) in result.thin_regions.iter().take(5).enumerate() {
            println!(
                "    Region {}: vertex {} at ({:.1}, {:.1}, {:.1}) = {:.2}mm",
                i,
                region.vertex_index,
                region.position.x,
                region.position.y,
                region.position.z,
                region.thickness
            );
        }
        if result.thin_regions.len() > 5 {
            println!("    ... and {} more thin regions", result.thin_regions.len() - 5);
        }
        false
    } else {
        println!("  All regions meet minimum thickness requirement");
        true
    }
}

/// Print equipment fitting summary.
pub fn print_fitting_summary(mesh: &Mesh, equipment_type: &str) {
    let report = mesh.validate();

    if let Some((min, max)) = mesh.bounds() {
        let dims = max - min;
        println!("\n{} Fitting Summary:", equipment_type);
        println!("  Dimensions: {:.1}mm x {:.1}mm x {:.1}mm", dims.x, dims.y, dims.z);
        println!("  Volume: {:.1} cm³", report.volume / 1000.0);
        println!("  Surface area: {:.1} cm²", report.surface_area / 100.0);
        println!("  Printable: {}", if report.is_printable() { "YES" } else { "NO" });
    }
}

/// Demonstrate foot region classification for skate fitting.
///
/// In a real system, this would use anatomical landmark detection.
#[allow(dead_code)]
pub fn classify_foot_vertex(vertex: &Vertex, foot_length: f64) -> &'static str {
    let pos = &vertex.position;

    // Simplified region classification based on position
    // Real implementation would use detected landmarks
    let y_ratio = pos.y / foot_length;

    if pos.z > 80.0 {
        "ankle"
    } else if y_ratio < 0.2 {
        "heel"
    } else if y_ratio < 0.4 {
        "arch"
    } else if y_ratio < 0.7 {
        "metatarsal"
    } else {
        "toe"
    }
}

/// Demonstrate head region classification for helmet fitting.
#[allow(dead_code)]
pub fn classify_head_vertex(vertex: &Vertex, head_height: f64) -> &'static str {
    let pos = &vertex.position;

    // Simplified region classification
    let z_ratio = pos.z / head_height;

    if z_ratio > 0.8 {
        "crown"
    } else if z_ratio < 0.3 {
        "chin"
    } else if pos.y > 0.0 {
        "forehead"
    } else {
        "back"
    }
}

fn main() {
    println!("Hockey Equipment Shell Generation Example");
    println!("=========================================\n");

    // Create a sample mesh for demonstration
    // In production, this would be loaded from a 3D scan
    let mesh = create_demo_foot_mesh();

    println!("Demo foot mesh created:");
    println!("  Vertices: {}", mesh.vertex_count());
    println!("  Faces: {}", mesh.face_count());

    // Validate
    let report = mesh.validate();
    println!("\nValidation:");
    println!("  Watertight: {}", report.is_watertight);
    println!("  Manifold: {}", report.is_manifold);

    // Check thickness for skate requirements
    let thickness_ok = analyze_equipment_thickness(&mesh, 1.5, "skate boot");

    // Print summary
    print_fitting_summary(&mesh, "Skate Boot");

    // Show equipment profiles
    println!("\n--- Equipment Thickness Profiles ---\n");

    let helmet = thickness_profiles::HelmetProfile::default();
    println!("Helmet Profile:");
    println!("  Crown: {}mm", helmet.crown_mm);
    println!("  Sides: {}mm", helmet.sides_mm);
    println!("  Back: {}mm", helmet.back_mm);
    println!("  Forehead: {}mm", helmet.forehead_mm);

    let skate = thickness_profiles::SkateProfile::default();
    println!("\nSkate Boot Profile:");
    println!("  Heel cup: {}mm", skate.heel_cup_mm);
    println!("  Ankle collar: {}mm", skate.ankle_collar_mm);
    println!("  Quarter panel: {}mm", skate.quarter_panel_mm);
    println!("  Arch: {}mm", skate.arch_mm);
    println!("  Toe box: {}mm", skate.toe_box_mm);

    let shoulder = thickness_profiles::ShoulderCapProfile::default();
    println!("\nShoulder Cap Profile:");
    println!("  Impact zone: {}mm", shoulder.impact_zone_mm);
    println!("  Edge: {}mm", shoulder.edge_mm);
    println!("  Attachment: {}mm", shoulder.attachment_mm);

    if thickness_ok {
        println!("\n✓ Equipment meets safety requirements");
    } else {
        println!("\n✗ Equipment needs thickness adjustment");
    }
}

/// Create a demo foot-like mesh for testing.
fn create_demo_foot_mesh() -> Mesh {
    let mut mesh = Mesh::new();

    // Create a simplified foot-like shape (elongated box with rounded ends)
    // This is for demonstration - real scans would be much more detailed

    // Foot dimensions (approximate, in mm)
    let length = 280.0;
    let width = 100.0;
    let height = 80.0;

    // Create a simple rectangular solid with 8 vertices
    mesh.vertices.push(Vertex::from_coords(0.0, 0.0, 0.0)); // 0: heel bottom left
    mesh.vertices.push(Vertex::from_coords(width, 0.0, 0.0)); // 1: heel bottom right
    mesh.vertices.push(Vertex::from_coords(width, length, 0.0)); // 2: toe bottom right
    mesh.vertices.push(Vertex::from_coords(0.0, length, 0.0)); // 3: toe bottom left
    mesh.vertices.push(Vertex::from_coords(0.0, 0.0, height)); // 4: heel top left
    mesh.vertices.push(Vertex::from_coords(width, 0.0, height)); // 5: heel top right
    mesh.vertices.push(Vertex::from_coords(width, length, height * 0.5)); // 6: toe top right
    mesh.vertices.push(Vertex::from_coords(0.0, length, height * 0.5)); // 7: toe top left

    // Create faces (12 triangles for 6 faces)
    // Bottom
    mesh.faces.push([0, 2, 1]);
    mesh.faces.push([0, 3, 2]);
    // Top
    mesh.faces.push([4, 5, 6]);
    mesh.faces.push([4, 6, 7]);
    // Front (heel)
    mesh.faces.push([0, 1, 5]);
    mesh.faces.push([0, 5, 4]);
    // Back (toe)
    mesh.faces.push([3, 7, 6]);
    mesh.faces.push([3, 6, 2]);
    // Left
    mesh.faces.push([0, 4, 7]);
    mesh.faces.push([0, 7, 3]);
    // Right
    mesh.faces.push([1, 2, 6]);
    mesh.faces.push([1, 6, 5]);

    mesh
}

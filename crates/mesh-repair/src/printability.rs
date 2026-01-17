//! Print validation and manufacturing analysis.
//!
//! This module provides tools for validating meshes for 3D printing,
//! detecting potential printing issues, and analyzing manufacturability.
//!
//! # Use Cases
//!
//! - Validate a mesh before sending to printer
//! - Detect areas that need support structures
//! - Find optimal orientation for printing
//! - Ensure minimum wall thickness requirements
//!
//! # Example
//!
//! ```
//! use mesh_repair::{Mesh, Vertex};
//! use mesh_repair::printability::{PrintValidation, validate_for_printing, PrinterConfig};
//!
//! // Create a simple mesh
//! let mut mesh = Mesh::new();
//! mesh.vertices.push(Vertex::from_coords(0.0, 0.0, 0.0));
//! mesh.vertices.push(Vertex::from_coords(10.0, 0.0, 0.0));
//! mesh.vertices.push(Vertex::from_coords(5.0, 10.0, 0.0));
//! mesh.vertices.push(Vertex::from_coords(5.0, 5.0, 5.0));
//! mesh.faces.push([0, 1, 3]);
//! mesh.faces.push([1, 2, 3]);
//! mesh.faces.push([2, 0, 3]);
//! mesh.faces.push([0, 2, 1]);
//!
//! // Validate for FDM printing
//! let config = PrinterConfig::fdm_default();
//! let result = validate_for_printing(&mesh, &config);
//!
//! if result.is_printable() {
//!     println!("Mesh is ready to print!");
//! } else {
//!     println!("Issues found: {:?}", result.issues);
//! }
//! ```

use crate::Mesh;
use crate::thickness::analyze_thickness;
use nalgebra::{Point3, UnitQuaternion, Vector3};
use std::f64::consts::PI;

/// Configuration for a specific 3D printer.
#[derive(Debug, Clone)]
pub struct PrinterConfig {
    /// Printer technology (FDM, SLA, SLS, etc.).
    pub technology: PrintTechnology,

    /// Minimum wall thickness in mm.
    pub min_wall_thickness: f64,

    /// Nozzle diameter in mm (for FDM).
    pub nozzle_diameter: f64,

    /// Layer height in mm.
    pub layer_height: f64,

    /// Maximum overhang angle in degrees (0 = vertical, 90 = horizontal).
    pub max_overhang_angle: f64,

    /// Minimum feature size that can be printed.
    pub min_feature_size: f64,

    /// Maximum bridge span in mm.
    pub max_bridge_span: f64,

    /// Build volume (X, Y, Z) in mm.
    pub build_volume: (f64, f64, f64),
}

impl Default for PrinterConfig {
    fn default() -> Self {
        Self::fdm_default()
    }
}

impl PrinterConfig {
    /// Default configuration for FDM printers.
    pub fn fdm_default() -> Self {
        Self {
            technology: PrintTechnology::Fdm,
            min_wall_thickness: 1.0,
            nozzle_diameter: 0.4,
            layer_height: 0.2,
            max_overhang_angle: 45.0,
            min_feature_size: 0.8,
            max_bridge_span: 10.0,
            build_volume: (200.0, 200.0, 200.0),
        }
    }

    /// Default configuration for SLA/resin printers.
    pub fn sla_default() -> Self {
        Self {
            technology: PrintTechnology::Sla,
            min_wall_thickness: 0.4,
            nozzle_diameter: 0.0, // N/A for SLA
            layer_height: 0.05,
            max_overhang_angle: 30.0, // More conservative for resin
            min_feature_size: 0.1,
            max_bridge_span: 5.0,
            build_volume: (120.0, 68.0, 155.0),
        }
    }

    /// Default configuration for SLS printers.
    pub fn sls_default() -> Self {
        Self {
            technology: PrintTechnology::Sls,
            min_wall_thickness: 0.7,
            nozzle_diameter: 0.0, // N/A for SLS
            layer_height: 0.1,
            max_overhang_angle: 90.0, // SLS doesn't need supports
            min_feature_size: 0.3,
            max_bridge_span: f64::INFINITY, // No bridges needed
            build_volume: (300.0, 300.0, 300.0),
        }
    }
}

/// 3D printing technology.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PrintTechnology {
    /// Fused Deposition Modeling (filament).
    Fdm,
    /// Stereolithography (resin).
    Sla,
    /// Selective Laser Sintering (powder).
    Sls,
    /// Other/custom technology.
    Other,
}

/// Result of print validation.
#[derive(Debug)]
pub struct PrintValidation {
    /// Whether the mesh can be printed.
    pub printable: bool,

    /// List of issues found.
    pub issues: Vec<PrintIssue>,

    /// Overall printability score (0.0 = unprintable, 1.0 = perfect).
    pub score: f64,

    /// Thin wall regions found.
    pub thin_walls: Vec<ThinWallRegion>,

    /// Overhang regions found.
    pub overhangs: Vec<OverhangRegion>,

    /// Support regions needed.
    pub support_regions: Vec<SupportRegion>,

    /// Estimated print time in minutes (rough estimate).
    pub estimated_print_time: Option<f64>,

    /// Estimated support volume in mm³.
    pub estimated_support_volume: f64,
}

impl PrintValidation {
    /// Check if the mesh is printable.
    pub fn is_printable(&self) -> bool {
        self.printable
    }

    /// Check if only warnings exist (printable but may have issues).
    pub fn has_warnings(&self) -> bool {
        self.issues
            .iter()
            .any(|i| matches!(i.severity, IssueSeverity::Warning))
    }

    /// Get critical issues that prevent printing.
    pub fn critical_issues(&self) -> Vec<&PrintIssue> {
        self.issues
            .iter()
            .filter(|i| matches!(i.severity, IssueSeverity::Critical))
            .collect()
    }
}

/// A print-related issue found during validation.
#[derive(Debug, Clone)]
pub struct PrintIssue {
    /// Type of issue.
    pub issue_type: PrintIssueType,

    /// Severity of the issue.
    pub severity: IssueSeverity,

    /// Human-readable description.
    pub description: String,

    /// Location in mesh (if applicable).
    pub location: Option<Point3<f64>>,

    /// Affected vertex or face indices.
    pub affected_elements: Vec<u32>,
}

/// Types of print issues.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PrintIssueType {
    /// Wall is too thin for the printer.
    ThinWall,
    /// Overhang angle exceeds maximum.
    ExcessiveOverhang,
    /// Bridge span is too long.
    LongBridge,
    /// Feature is smaller than minimum size.
    SmallFeature,
    /// Trapped volume (internal cavity).
    TrappedVolume,
    /// Mesh doesn't fit in build volume.
    ExceedsBuildVolume,
    /// Mesh is not watertight.
    NotWatertight,
    /// Mesh has non-manifold edges.
    NonManifold,
    /// Mesh has self-intersections.
    SelfIntersecting,
    /// Other issue.
    Other,
}

/// Severity of an issue.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum IssueSeverity {
    /// Informational only.
    Info,
    /// May cause problems but can still print.
    Warning,
    /// Will likely fail or have significant defects.
    Critical,
}

/// Information about a thin wall region.
#[derive(Debug, Clone)]
pub struct ThinWallRegion {
    /// Center of the thin region.
    pub center: Point3<f64>,
    /// Minimum thickness found.
    pub thickness: f64,
    /// Approximate area of the thin region.
    pub area: f64,
    /// Face indices in this region.
    pub faces: Vec<u32>,
}

/// Information about an overhang region.
#[derive(Debug, Clone)]
pub struct OverhangRegion {
    /// Center of the overhang region.
    pub center: Point3<f64>,
    /// Maximum overhang angle in degrees.
    pub angle: f64,
    /// Approximate area needing support.
    pub area: f64,
    /// Face indices in this region.
    pub faces: Vec<u32>,
}

/// Information about a region that needs support.
#[derive(Debug, Clone)]
pub struct SupportRegion {
    /// Center of the support region.
    pub center: Point3<f64>,
    /// Estimated support volume in mm³.
    pub volume: f64,
    /// Maximum height that needs support.
    pub max_height: f64,
    /// Face indices needing support.
    pub faces: Vec<u32>,
}

/// Parameters for auto-orientation.
#[derive(Debug, Clone)]
pub struct OrientParams {
    /// Weight for minimizing support volume (0.0-1.0).
    pub support_weight: f64,
    /// Weight for minimizing overhang area (0.0-1.0).
    pub overhang_weight: f64,
    /// Weight for minimizing print time (0.0-1.0).
    pub time_weight: f64,
    /// Weight for print quality (0.0-1.0).
    pub quality_weight: f64,
    /// Number of orientations to test.
    pub samples: usize,
}

impl Default for OrientParams {
    fn default() -> Self {
        Self {
            support_weight: 0.4,
            overhang_weight: 0.3,
            time_weight: 0.1,
            quality_weight: 0.2,
            samples: 100,
        }
    }
}

/// Result of auto-orientation.
#[derive(Debug)]
pub struct OrientResult {
    /// Optimal rotation to apply.
    pub rotation: UnitQuaternion<f64>,
    /// The oriented mesh.
    pub mesh: Mesh,
    /// Estimated support volume at this orientation.
    pub support_volume: f64,
    /// Overhang area at this orientation.
    pub overhang_area: f64,
    /// Orientation score (higher is better).
    pub score: f64,
    /// All tested orientations with scores.
    pub alternatives: Vec<(UnitQuaternion<f64>, f64)>,
}

/// Result of support region detection.
#[derive(Debug)]
pub struct SupportAnalysis {
    /// Regions that need support.
    pub regions: Vec<SupportRegion>,
    /// Total support volume estimate in mm³.
    pub total_volume: f64,
    /// Total supported area in mm².
    pub total_area: f64,
    /// Percentage of surface area needing support.
    pub support_percentage: f64,
}

// ============================================================================
// Main validation function
// ============================================================================

/// Validate a mesh for 3D printing.
pub fn validate_for_printing(mesh: &Mesh, config: &PrinterConfig) -> PrintValidation {
    let mut issues = Vec::new();
    let mut thin_walls = Vec::new();
    let mut overhangs = Vec::new();
    let mut support_regions = Vec::new();
    let mut estimated_support_volume = 0.0;

    // Check build volume
    let bbox = compute_bounding_box(mesh);
    let (dx, dy, dz) = (
        bbox.1.x - bbox.0.x,
        bbox.1.y - bbox.0.y,
        bbox.1.z - bbox.0.z,
    );

    if dx > config.build_volume.0 || dy > config.build_volume.1 || dz > config.build_volume.2 {
        issues.push(PrintIssue {
            issue_type: PrintIssueType::ExceedsBuildVolume,
            severity: IssueSeverity::Critical,
            description: format!(
                "Mesh dimensions ({:.1} x {:.1} x {:.1} mm) exceed build volume ({:.0} x {:.0} x {:.0} mm)",
                dx, dy, dz, config.build_volume.0, config.build_volume.1, config.build_volume.2
            ),
            location: None,
            affected_elements: Vec::new(),
        });
    }

    // Check watertightness (simplified - check for boundary edges)
    let boundary_edges = count_boundary_edges(mesh);
    if boundary_edges > 0 {
        issues.push(PrintIssue {
            issue_type: PrintIssueType::NotWatertight,
            severity: IssueSeverity::Critical,
            description: format!("Mesh has {} open edges (not watertight)", boundary_edges),
            location: None,
            affected_elements: Vec::new(),
        });
    }

    // Check wall thickness
    let thickness_result = analyze_thickness(
        mesh,
        &crate::thickness::ThicknessParams {
            min_thickness: config.min_wall_thickness,
            max_ray_distance: 50.0,
            epsilon: 1e-8,
            max_regions: 100, // Limit for faster validation
            require_normals: false,
        },
    );

    for region in &thickness_result.thin_regions {
        // Find faces containing this vertex
        let vi = region.vertex_index as u32;
        let affected_faces: Vec<u32> = mesh
            .faces
            .iter()
            .enumerate()
            .filter(|(_, f)| f.contains(&vi))
            .map(|(i, _)| i as u32)
            .collect();

        thin_walls.push(ThinWallRegion {
            center: region.position,
            thickness: region.thickness,
            area: 1.0, // Approximate - one vertex worth of area
            faces: affected_faces.clone(),
        });

        let severity = if region.thickness < config.min_wall_thickness * 0.5 {
            IssueSeverity::Critical
        } else {
            IssueSeverity::Warning
        };

        issues.push(PrintIssue {
            issue_type: PrintIssueType::ThinWall,
            severity,
            description: format!(
                "Thin wall detected: {:.2} mm (minimum: {:.2} mm)",
                region.thickness, config.min_wall_thickness
            ),
            location: Some(region.position),
            affected_elements: affected_faces,
        });
    }

    // Check overhangs (for FDM and SLA)
    if config.technology != PrintTechnology::Sls {
        let (_overhang_faces, overhang_regions_result) =
            detect_overhangs(mesh, config.max_overhang_angle);

        overhangs = overhang_regions_result;

        for overhang in &overhangs {
            let severity = if overhang.angle > config.max_overhang_angle + 15.0 {
                IssueSeverity::Critical
            } else {
                IssueSeverity::Warning
            };

            issues.push(PrintIssue {
                issue_type: PrintIssueType::ExcessiveOverhang,
                severity,
                description: format!(
                    "Overhang at {:.1}° (maximum: {:.1}°)",
                    overhang.angle, config.max_overhang_angle
                ),
                location: Some(overhang.center),
                affected_elements: overhang.faces.clone(),
            });

            // Estimate support volume for this overhang
            let support_height = (overhang.center.z - bbox.0.z).max(0.0);
            estimated_support_volume += overhang.area * support_height * 0.1; // Rough estimate
        }

        // Convert to support regions
        support_regions = overhangs
            .iter()
            .map(|o| SupportRegion {
                center: o.center,
                volume: o.area * (o.center.z - bbox.0.z).max(1.0) * 0.1,
                max_height: o.center.z - bbox.0.z,
                faces: o.faces.clone(),
            })
            .collect();
    }

    // Calculate overall score
    let score = calculate_printability_score(&issues);
    let printable = !issues
        .iter()
        .any(|i| matches!(i.severity, IssueSeverity::Critical));

    // Estimate print time (very rough)
    let volume = estimate_mesh_volume(mesh).abs();
    let estimated_print_time = if volume > 0.0 {
        // Rough estimate: 1 minute per cm³ for FDM, 0.5 for SLA
        let rate = match config.technology {
            PrintTechnology::Fdm => 60.0,  // mm³/min
            PrintTechnology::Sla => 120.0, // mm³/min
            _ => 100.0,
        };
        Some(volume / rate)
    } else {
        None
    };

    PrintValidation {
        printable,
        issues,
        score,
        thin_walls,
        overhangs,
        support_regions,
        estimated_print_time,
        estimated_support_volume,
    }
}

/// Detect regions that need support structures.
pub fn detect_support_regions(mesh: &Mesh, config: &PrinterConfig) -> SupportAnalysis {
    let (_, overhang_regions) = detect_overhangs(mesh, config.max_overhang_angle);
    let bbox = compute_bounding_box(mesh);

    let regions: Vec<SupportRegion> = overhang_regions
        .iter()
        .map(|o| {
            let height = (o.center.z - bbox.0.z).max(0.0);
            SupportRegion {
                center: o.center,
                volume: o.area * height * 0.1,
                max_height: height,
                faces: o.faces.clone(),
            }
        })
        .collect();

    let total_volume: f64 = regions.iter().map(|r| r.volume).sum();
    let total_area: f64 = overhang_regions.iter().map(|r| r.area).sum();
    let mesh_area = estimate_surface_area(mesh);
    let support_percentage = if mesh_area > 0.0 {
        (total_area / mesh_area) * 100.0
    } else {
        0.0
    };

    SupportAnalysis {
        regions,
        total_volume,
        total_area,
        support_percentage,
    }
}

/// Find the optimal orientation for printing.
pub fn auto_orient_for_printing(
    mesh: &Mesh,
    config: &PrinterConfig,
    params: &OrientParams,
) -> OrientResult {
    let mut best_rotation = UnitQuaternion::identity();
    let mut best_score = f64::NEG_INFINITY;
    let mut alternatives = Vec::new();

    // Generate sample orientations using Fibonacci sphere distribution
    let rotations = generate_sample_rotations(params.samples);

    for rotation in rotations {
        let rotated = rotate_mesh(mesh, &rotation);
        let score = evaluate_orientation(&rotated, config, params);

        alternatives.push((rotation, score));

        if score > best_score {
            best_score = score;
            best_rotation = rotation;
        }
    }

    // Sort alternatives by score
    alternatives.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

    let oriented_mesh = rotate_mesh(mesh, &best_rotation);
    let (support_volume, overhang_area) = estimate_support_metrics(&oriented_mesh, config);

    OrientResult {
        rotation: best_rotation,
        mesh: oriented_mesh,
        support_volume,
        overhang_area,
        score: best_score,
        alternatives,
    }
}

// ============================================================================
// Internal helper functions
// ============================================================================

fn compute_bounding_box(mesh: &Mesh) -> (Point3<f64>, Point3<f64>) {
    if mesh.vertices.is_empty() {
        return (Point3::origin(), Point3::origin());
    }

    let mut min = mesh.vertices[0].position;
    let mut max = mesh.vertices[0].position;

    for v in &mesh.vertices {
        min.x = min.x.min(v.position.x);
        min.y = min.y.min(v.position.y);
        min.z = min.z.min(v.position.z);
        max.x = max.x.max(v.position.x);
        max.y = max.y.max(v.position.y);
        max.z = max.z.max(v.position.z);
    }

    (min, max)
}

fn count_boundary_edges(mesh: &Mesh) -> usize {
    use std::collections::HashMap;

    let mut edge_count: HashMap<(u32, u32), usize> = HashMap::new();

    for face in &mesh.faces {
        for i in 0..3 {
            let v0 = face[i];
            let v1 = face[(i + 1) % 3];
            let edge = if v0 < v1 { (v0, v1) } else { (v1, v0) };
            *edge_count.entry(edge).or_insert(0) += 1;
        }
    }

    edge_count.values().filter(|&&c| c == 1).count()
}

fn detect_overhangs(mesh: &Mesh, max_angle: f64) -> (Vec<u32>, Vec<OverhangRegion>) {
    let up = Vector3::new(0.0, 0.0, 1.0);
    let threshold = (max_angle * PI / 180.0).cos();
    let mut overhang_faces = Vec::new();
    let mut regions = Vec::new();

    for (fi, face) in mesh.faces.iter().enumerate() {
        let v0 = mesh.vertices[face[0] as usize].position;
        let v1 = mesh.vertices[face[1] as usize].position;
        let v2 = mesh.vertices[face[2] as usize].position;

        let e1 = v1 - v0;
        let e2 = v2 - v0;
        let normal = e1.cross(&e2).normalize();

        // Check if face is facing downward
        let dot = normal.dot(&up);

        if dot < -threshold {
            overhang_faces.push(fi as u32);

            // Calculate overhang angle
            let angle = ((-dot).acos() * 180.0 / PI) - 90.0;

            // Calculate face area
            let area = e1.cross(&e2).norm() * 0.5;

            // Calculate face center
            let center = Point3::from((v0.coords + v1.coords + v2.coords) / 3.0);

            regions.push(OverhangRegion {
                center,
                angle,
                area,
                faces: vec![fi as u32],
            });
        }
    }

    // Merge nearby overhang regions (simplified - just return individual faces for now)
    (overhang_faces, regions)
}

fn calculate_printability_score(issues: &[PrintIssue]) -> f64 {
    let mut score: f64 = 1.0;

    for issue in issues {
        let penalty = match issue.severity {
            IssueSeverity::Info => 0.0,
            IssueSeverity::Warning => 0.1,
            IssueSeverity::Critical => 0.3,
        };
        score -= penalty;
    }

    score.max(0.0)
}

fn estimate_mesh_volume(mesh: &Mesh) -> f64 {
    let mut volume = 0.0;

    for face in &mesh.faces {
        let v0 = mesh.vertices[face[0] as usize].position;
        let v1 = mesh.vertices[face[1] as usize].position;
        let v2 = mesh.vertices[face[2] as usize].position;

        // Signed volume of tetrahedron with origin
        volume += v0.coords.dot(&v1.coords.cross(&v2.coords)) / 6.0;
    }

    volume
}

fn estimate_surface_area(mesh: &Mesh) -> f64 {
    let mut area = 0.0;

    for face in &mesh.faces {
        let v0 = mesh.vertices[face[0] as usize].position;
        let v1 = mesh.vertices[face[1] as usize].position;
        let v2 = mesh.vertices[face[2] as usize].position;

        let e1 = v1 - v0;
        let e2 = v2 - v0;
        area += e1.cross(&e2).norm() * 0.5;
    }

    area
}

fn generate_sample_rotations(n: usize) -> Vec<UnitQuaternion<f64>> {
    let mut rotations = vec![UnitQuaternion::identity()];

    // Use Fibonacci sphere distribution for uniform sampling
    let phi = PI * (3.0 - 5.0_f64.sqrt()); // Golden angle

    for i in 1..n {
        let y = 1.0 - (i as f64 / (n - 1) as f64) * 2.0; // y goes from 1 to -1
        let radius = (1.0 - y * y).sqrt();
        let theta = phi * i as f64;

        let x = theta.cos() * radius;
        let z = theta.sin() * radius;

        // Create rotation that aligns Z with this direction
        let dir = Vector3::new(x, y, z).normalize();
        let rotation = rotation_to_align_z(dir);
        rotations.push(rotation);
    }

    rotations
}

fn rotation_to_align_z(dir: Vector3<f64>) -> UnitQuaternion<f64> {
    let z = Vector3::new(0.0, 0.0, 1.0);

    if (dir - z).norm() < 1e-6 {
        return UnitQuaternion::identity();
    }

    if (dir + z).norm() < 1e-6 {
        return UnitQuaternion::from_axis_angle(&Vector3::x_axis(), PI);
    }

    let axis = z.cross(&dir).normalize();
    let angle = z.dot(&dir).acos();

    UnitQuaternion::from_axis_angle(&nalgebra::Unit::new_normalize(axis), angle)
}

fn rotate_mesh(mesh: &Mesh, rotation: &UnitQuaternion<f64>) -> Mesh {
    let mut result = mesh.clone();

    for vertex in &mut result.vertices {
        vertex.position = Point3::from(rotation * vertex.position.coords);
        if let Some(ref mut normal) = vertex.normal {
            *normal = rotation * *normal;
        }
    }

    result
}

fn evaluate_orientation(mesh: &Mesh, config: &PrinterConfig, params: &OrientParams) -> f64 {
    let (support_volume, overhang_area) = estimate_support_metrics(mesh, config);
    let bbox = compute_bounding_box(mesh);
    let height = bbox.1.z - bbox.0.z;

    // Normalize metrics
    let mesh_volume = estimate_mesh_volume(mesh).abs().max(1.0);
    let mesh_area = estimate_surface_area(mesh).max(1.0);

    let support_score = 1.0 - (support_volume / mesh_volume).min(1.0);
    let overhang_score = 1.0 - (overhang_area / mesh_area).min(1.0);
    let height_score = 1.0 - (height / config.build_volume.2).min(1.0);
    let quality_score = 1.0; // Could consider layer orientation for quality

    // Weighted sum
    let total_weight =
        params.support_weight + params.overhang_weight + params.time_weight + params.quality_weight;

    (params.support_weight * support_score
        + params.overhang_weight * overhang_score
        + params.time_weight * height_score
        + params.quality_weight * quality_score)
        / total_weight
}

fn estimate_support_metrics(mesh: &Mesh, config: &PrinterConfig) -> (f64, f64) {
    let (_, overhangs) = detect_overhangs(mesh, config.max_overhang_angle);
    let bbox = compute_bounding_box(mesh);

    let mut support_volume = 0.0;
    let mut overhang_area = 0.0;

    for overhang in &overhangs {
        let height = (overhang.center.z - bbox.0.z).max(0.0);
        support_volume += overhang.area * height * 0.1; // Rough cone approximation
        overhang_area += overhang.area;
    }

    (support_volume, overhang_area)
}

// ============================================================================
// Mesh extension methods
// ============================================================================

impl Mesh {
    /// Validate this mesh for 3D printing.
    pub fn validate_for_printing(&self) -> PrintValidation {
        validate_for_printing(self, &PrinterConfig::default())
    }

    /// Validate this mesh for printing with specific printer configuration.
    pub fn validate_for_printing_with_config(&self, config: &PrinterConfig) -> PrintValidation {
        validate_for_printing(self, config)
    }

    /// Detect regions that need support structures.
    pub fn detect_support_regions(&self) -> SupportAnalysis {
        detect_support_regions(self, &PrinterConfig::default())
    }

    /// Detect support regions with specific printer configuration.
    pub fn detect_support_regions_with_config(&self, config: &PrinterConfig) -> SupportAnalysis {
        detect_support_regions(self, config)
    }

    /// Find the optimal orientation for printing.
    pub fn auto_orient_for_printing(&self) -> OrientResult {
        auto_orient_for_printing(self, &PrinterConfig::default(), &OrientParams::default())
    }

    /// Auto-orient with specific configuration.
    pub fn auto_orient_for_printing_with_config(
        &self,
        config: &PrinterConfig,
        params: &OrientParams,
    ) -> OrientResult {
        auto_orient_for_printing(self, config, params)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::Vertex;

    fn create_test_tetrahedron() -> Mesh {
        let mut mesh = Mesh::new();
        mesh.vertices.push(Vertex::from_coords(0.0, 0.0, 0.0));
        mesh.vertices.push(Vertex::from_coords(10.0, 0.0, 0.0));
        mesh.vertices.push(Vertex::from_coords(5.0, 10.0, 0.0));
        mesh.vertices.push(Vertex::from_coords(5.0, 5.0, 10.0));

        mesh.faces.push([0, 1, 3]);
        mesh.faces.push([1, 2, 3]);
        mesh.faces.push([2, 0, 3]);
        mesh.faces.push([0, 2, 1]);
        mesh
    }

    #[test]
    fn test_printer_config_defaults() {
        let fdm = PrinterConfig::fdm_default();
        assert_eq!(fdm.technology, PrintTechnology::Fdm);
        assert!(fdm.min_wall_thickness > 0.0);

        let sla = PrinterConfig::sla_default();
        assert_eq!(sla.technology, PrintTechnology::Sla);
        assert!(sla.min_wall_thickness < fdm.min_wall_thickness);

        let sls = PrinterConfig::sls_default();
        assert_eq!(sls.technology, PrintTechnology::Sls);
    }

    #[test]
    fn test_validate_for_printing() {
        let mesh = create_test_tetrahedron();
        let result = validate_for_printing(&mesh, &PrinterConfig::fdm_default());

        // Should have some validation result
        assert!(result.score >= 0.0 && result.score <= 1.0);
    }

    #[test]
    fn test_detect_support_regions() {
        let mesh = create_test_tetrahedron();
        let analysis = detect_support_regions(&mesh, &PrinterConfig::fdm_default());

        // May or may not have support regions depending on orientation
        assert!(analysis.total_volume >= 0.0);
        assert!(analysis.support_percentage >= 0.0);
    }

    #[test]
    fn test_auto_orient() {
        let mesh = create_test_tetrahedron();
        let result = auto_orient_for_printing(
            &mesh,
            &PrinterConfig::fdm_default(),
            &OrientParams {
                samples: 10, // Fewer samples for faster test
                ..Default::default()
            },
        );

        assert!(!result.mesh.vertices.is_empty());
        assert!(result.score >= 0.0 && result.score <= 1.0);
        assert!(!result.alternatives.is_empty());
    }

    #[test]
    fn test_mesh_validate_method() {
        let mesh = create_test_tetrahedron();
        let result = mesh.validate_for_printing();

        assert!(result.score >= 0.0);
    }

    #[test]
    fn test_exceeds_build_volume() {
        let mut mesh = Mesh::new();
        // Create a mesh larger than default build volume (200mm)
        mesh.vertices.push(Vertex::from_coords(0.0, 0.0, 0.0));
        mesh.vertices.push(Vertex::from_coords(300.0, 0.0, 0.0));
        mesh.vertices.push(Vertex::from_coords(150.0, 300.0, 0.0));
        mesh.faces.push([0, 1, 2]);

        let result = validate_for_printing(&mesh, &PrinterConfig::fdm_default());

        // Should have a build volume issue
        assert!(
            result
                .issues
                .iter()
                .any(|i| i.issue_type == PrintIssueType::ExceedsBuildVolume)
        );
    }

    #[test]
    fn test_bounding_box() {
        let mesh = create_test_tetrahedron();
        let (min, max) = compute_bounding_box(&mesh);

        assert!(min.x < max.x);
        assert!(min.y < max.y);
        assert!(min.z < max.z);
    }
}

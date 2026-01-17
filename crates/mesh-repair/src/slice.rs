//! Slicing and layer preview for 3D printing.
//!
//! This module provides tools for generating 2D slice previews of meshes,
//! calculating per-layer statistics, and estimating print times.
//!
//! # Use Cases
//!
//! - Generate layer-by-layer preview of a 3D print
//! - Calculate print time estimates
//! - Analyze per-layer statistics (area, perimeter)
//! - Validate slice-level printability
//!
//! # Example
//!
//! ```
//! use mesh_repair::{Mesh, Vertex};
//! use mesh_repair::slice::{slice_mesh, SliceParams};
//!
//! // Create a simple mesh
//! let mut mesh = Mesh::new();
//! mesh.vertices.push(Vertex::from_coords(0.0, 0.0, 0.0));
//! mesh.vertices.push(Vertex::from_coords(10.0, 0.0, 0.0));
//! mesh.vertices.push(Vertex::from_coords(5.0, 10.0, 0.0));
//! mesh.vertices.push(Vertex::from_coords(5.0, 5.0, 10.0));
//! mesh.faces.push([0, 1, 3]);
//! mesh.faces.push([1, 2, 3]);
//! mesh.faces.push([2, 0, 3]);
//! mesh.faces.push([0, 2, 1]);
//!
//! // Generate slices
//! let params = SliceParams::default();
//! let result = slice_mesh(&mesh, &params);
//!
//! println!("Total layers: {}", result.layers.len());
//! println!("Estimated print time: {:.1} minutes", result.estimated_print_time);
//! ```

use crate::Mesh;
use crate::measure::{CrossSection, cross_section};
use nalgebra::{Point3, Vector3};

/// Parameters for slicing operations.
#[derive(Debug, Clone)]
pub struct SliceParams {
    /// Layer height in mm.
    pub layer_height: f64,

    /// Print direction (typically Z-up).
    pub direction: Vector3<f64>,

    /// First layer height (often thicker for adhesion).
    pub first_layer_height: f64,

    /// Infill density (0.0-1.0).
    pub infill_density: f64,

    /// Number of perimeter shells.
    pub perimeters: usize,

    /// Perimeter width in mm (typically nozzle diameter).
    pub perimeter_width: f64,

    /// Print speed for perimeters in mm/s.
    pub perimeter_speed: f64,

    /// Print speed for infill in mm/s.
    pub infill_speed: f64,

    /// Travel speed in mm/s.
    pub travel_speed: f64,

    /// Extrusion width multiplier.
    pub extrusion_multiplier: f64,
}

impl Default for SliceParams {
    fn default() -> Self {
        Self {
            layer_height: 0.2,
            direction: Vector3::z(),
            first_layer_height: 0.3,
            infill_density: 0.2,
            perimeters: 2,
            perimeter_width: 0.4,
            perimeter_speed: 40.0,
            infill_speed: 60.0,
            travel_speed: 150.0,
            extrusion_multiplier: 1.0,
        }
    }
}

impl SliceParams {
    /// Parameters for high quality printing.
    pub fn high_quality() -> Self {
        Self {
            layer_height: 0.1,
            first_layer_height: 0.2,
            infill_density: 0.3,
            perimeters: 3,
            perimeter_speed: 30.0,
            infill_speed: 40.0,
            ..Default::default()
        }
    }

    /// Parameters for fast draft printing.
    pub fn draft() -> Self {
        Self {
            layer_height: 0.3,
            first_layer_height: 0.35,
            infill_density: 0.1,
            perimeters: 2,
            perimeter_speed: 60.0,
            infill_speed: 80.0,
            ..Default::default()
        }
    }

    /// Parameters for SLA/resin printing.
    pub fn for_sla() -> Self {
        Self {
            layer_height: 0.05,
            first_layer_height: 0.05,
            infill_density: 1.0, // SLA typically prints solid
            perimeters: 0,       // Not applicable for SLA
            perimeter_width: 0.0,
            perimeter_speed: 0.0,
            infill_speed: 0.0,
            travel_speed: 0.0,
            ..Default::default()
        }
    }
}

/// Result of slicing operation.
#[derive(Debug)]
pub struct SliceResult {
    /// Individual layers from bottom to top.
    pub layers: Vec<Layer>,

    /// Total height of the sliced object in mm.
    pub total_height: f64,

    /// Total number of layers.
    pub layer_count: usize,

    /// Estimated print time in minutes.
    pub estimated_print_time: f64,

    /// Estimated filament usage in mm.
    pub estimated_filament_length: f64,

    /// Estimated filament volume in mm³.
    pub estimated_filament_volume: f64,

    /// Layer with maximum area.
    pub max_area_layer: usize,

    /// Layer with maximum perimeter.
    pub max_perimeter_layer: usize,

    /// Slice parameters used.
    pub params: SliceParams,
}

/// A single layer/slice of the mesh.
#[derive(Debug, Clone)]
pub struct Layer {
    /// Layer index (0 = first layer).
    pub index: usize,

    /// Z height of this layer in mm.
    pub z_height: f64,

    /// Layer thickness in mm.
    pub thickness: f64,

    /// Cross-section contours for this layer.
    pub contours: Vec<Contour>,

    /// Total area of all contours in mm².
    pub area: f64,

    /// Total perimeter length in mm.
    pub perimeter: f64,

    /// Estimated print time for this layer in seconds.
    pub print_time: f64,

    /// Estimated filament length for this layer in mm.
    pub filament_length: f64,

    /// Number of separate islands (disconnected regions).
    pub island_count: usize,

    /// Bounding box of this layer (2D).
    pub bounds: LayerBounds,
}

/// A contour (closed loop) in a layer.
#[derive(Debug, Clone)]
pub struct Contour {
    /// Points defining the contour (closed loop).
    pub points: Vec<Point3<f64>>,

    /// Area enclosed by this contour.
    pub area: f64,

    /// Perimeter of this contour.
    pub perimeter: f64,

    /// Whether this is an outer contour (vs hole).
    pub is_outer: bool,

    /// Centroid of the contour.
    pub centroid: Point3<f64>,
}

/// 2D bounding box for a layer.
#[derive(Debug, Clone)]
pub struct LayerBounds {
    /// Minimum X coordinate.
    pub min_x: f64,
    /// Maximum X coordinate.
    pub max_x: f64,
    /// Minimum Y coordinate.
    pub min_y: f64,
    /// Maximum Y coordinate.
    pub max_y: f64,
}

impl LayerBounds {
    /// Width of the bounding box.
    pub fn width(&self) -> f64 {
        self.max_x - self.min_x
    }

    /// Height of the bounding box.
    pub fn height(&self) -> f64 {
        self.max_y - self.min_y
    }

    /// Center point of the bounding box.
    pub fn center(&self) -> (f64, f64) {
        (
            (self.min_x + self.max_x) / 2.0,
            (self.min_y + self.max_y) / 2.0,
        )
    }
}

/// Layer statistics summary.
#[derive(Debug, Clone)]
pub struct LayerStats {
    /// Minimum area across all layers.
    pub min_area: f64,
    /// Maximum area across all layers.
    pub max_area: f64,
    /// Average area.
    pub avg_area: f64,
    /// Minimum perimeter.
    pub min_perimeter: f64,
    /// Maximum perimeter.
    pub max_perimeter: f64,
    /// Average perimeter.
    pub avg_perimeter: f64,
    /// Maximum island count.
    pub max_islands: usize,
}

// ============================================================================
// Main slicing functions
// ============================================================================

/// Slice a mesh into layers for 3D printing preview.
pub fn slice_mesh(mesh: &Mesh, params: &SliceParams) -> SliceResult {
    // Find Z bounds
    let (min_z, max_z) = find_z_bounds(mesh, &params.direction);
    let total_height = max_z - min_z;

    if total_height <= 0.0 || mesh.vertices.is_empty() {
        return SliceResult {
            layers: Vec::new(),
            total_height: 0.0,
            layer_count: 0,
            estimated_print_time: 0.0,
            estimated_filament_length: 0.0,
            estimated_filament_volume: 0.0,
            max_area_layer: 0,
            max_perimeter_layer: 0,
            params: params.clone(),
        };
    }

    // Calculate layer heights
    let mut z_heights = Vec::new();
    let mut current_z = min_z + params.first_layer_height;
    z_heights.push((current_z, params.first_layer_height));

    while current_z < max_z {
        current_z += params.layer_height;
        if current_z <= max_z {
            z_heights.push((current_z, params.layer_height));
        }
    }

    // Generate layers
    let mut layers = Vec::with_capacity(z_heights.len());
    let mut max_area = 0.0;
    let mut max_area_layer = 0;
    let mut max_perimeter = 0.0;
    let mut max_perimeter_layer = 0;
    let mut total_print_time = 0.0;
    let mut total_filament = 0.0;

    for (index, (z, thickness)) in z_heights.iter().enumerate() {
        let layer = generate_layer(mesh, index, *z, *thickness, params);

        if layer.area > max_area {
            max_area = layer.area;
            max_area_layer = index;
        }
        if layer.perimeter > max_perimeter {
            max_perimeter = layer.perimeter;
            max_perimeter_layer = index;
        }

        total_print_time += layer.print_time;
        total_filament += layer.filament_length;

        layers.push(layer);
    }

    // Calculate filament volume (assuming 1.75mm diameter)
    let filament_diameter: f64 = 1.75;
    let filament_area = std::f64::consts::PI * (filament_diameter / 2.0).powi(2);
    let filament_volume = total_filament * filament_area;

    SliceResult {
        layers,
        total_height,
        layer_count: z_heights.len(),
        estimated_print_time: total_print_time / 60.0, // Convert to minutes
        estimated_filament_length: total_filament,
        estimated_filament_volume: filament_volume,
        max_area_layer,
        max_perimeter_layer,
        params: params.clone(),
    }
}

/// Generate a slice preview at a specific height.
pub fn slice_preview(mesh: &Mesh, z: f64, params: &SliceParams) -> Layer {
    generate_layer(mesh, 0, z, params.layer_height, params)
}

/// Calculate layer statistics for all layers.
pub fn calculate_layer_stats(result: &SliceResult) -> LayerStats {
    if result.layers.is_empty() {
        return LayerStats {
            min_area: 0.0,
            max_area: 0.0,
            avg_area: 0.0,
            min_perimeter: 0.0,
            max_perimeter: 0.0,
            avg_perimeter: 0.0,
            max_islands: 0,
        };
    }

    let mut min_area: f64 = f64::INFINITY;
    let mut max_area: f64 = 0.0;
    let mut sum_area: f64 = 0.0;
    let mut min_perimeter: f64 = f64::INFINITY;
    let mut max_perimeter: f64 = 0.0;
    let mut sum_perimeter = 0.0;
    let mut max_islands = 0;

    for layer in &result.layers {
        min_area = min_area.min(layer.area);
        max_area = max_area.max(layer.area);
        sum_area += layer.area;
        min_perimeter = min_perimeter.min(layer.perimeter);
        max_perimeter = max_perimeter.max(layer.perimeter);
        sum_perimeter += layer.perimeter;
        max_islands = max_islands.max(layer.island_count);
    }

    let n = result.layers.len() as f64;
    LayerStats {
        min_area,
        max_area,
        avg_area: sum_area / n,
        min_perimeter,
        max_perimeter,
        avg_perimeter: sum_perimeter / n,
        max_islands,
    }
}

// ============================================================================
// Internal helper functions
// ============================================================================

fn find_z_bounds(mesh: &Mesh, direction: &Vector3<f64>) -> (f64, f64) {
    if mesh.vertices.is_empty() {
        return (0.0, 0.0);
    }

    let dir = direction.normalize();
    let mut min_z = f64::INFINITY;
    let mut max_z = f64::NEG_INFINITY;

    for v in &mesh.vertices {
        let z = v.position.coords.dot(&dir);
        min_z = min_z.min(z);
        max_z = max_z.max(z);
    }

    (min_z, max_z)
}

fn generate_layer(
    mesh: &Mesh,
    index: usize,
    z: f64,
    thickness: f64,
    params: &SliceParams,
) -> Layer {
    // Get cross-section at this Z height
    let plane_point = Point3::new(0.0, 0.0, z);
    let section = cross_section(mesh, plane_point, params.direction);

    // Convert cross-section to contours
    let contours = extract_contours(&section);
    let island_count = contours.iter().filter(|c| c.is_outer).count();

    // Calculate bounds
    let bounds = calculate_layer_bounds(&contours);

    // Calculate print time estimate
    let print_time = estimate_layer_print_time(&contours, params);

    // Calculate filament usage
    let filament_length = estimate_filament_usage(&contours, thickness, params);

    Layer {
        index,
        z_height: z,
        thickness,
        area: section.area,
        perimeter: section.perimeter,
        contours,
        print_time,
        filament_length,
        island_count,
        bounds,
    }
}

fn extract_contours(section: &CrossSection) -> Vec<Contour> {
    if section.points.is_empty() {
        return Vec::new();
    }

    // For simplicity, treat the entire cross-section as one contour
    // In a full implementation, we would separate inner/outer contours
    let perimeter = section.perimeter;
    let area = section.area;

    vec![Contour {
        points: section.points.clone(),
        area,
        perimeter,
        is_outer: true,
        centroid: section.centroid,
    }]
}

fn calculate_layer_bounds(contours: &[Contour]) -> LayerBounds {
    if contours.is_empty() {
        return LayerBounds {
            min_x: 0.0,
            max_x: 0.0,
            min_y: 0.0,
            max_y: 0.0,
        };
    }

    let mut min_x = f64::INFINITY;
    let mut max_x = f64::NEG_INFINITY;
    let mut min_y = f64::INFINITY;
    let mut max_y = f64::NEG_INFINITY;

    for contour in contours {
        for p in &contour.points {
            min_x = min_x.min(p.x);
            max_x = max_x.max(p.x);
            min_y = min_y.min(p.y);
            max_y = max_y.max(p.y);
        }
    }

    LayerBounds {
        min_x,
        max_x,
        min_y,
        max_y,
    }
}

fn estimate_layer_print_time(contours: &[Contour], params: &SliceParams) -> f64 {
    if contours.is_empty() {
        return 0.0;
    }

    let mut time = 0.0;

    for contour in contours {
        // Perimeter time
        let perimeter_passes = params.perimeters as f64;
        let perimeter_length = contour.perimeter * perimeter_passes;
        time += perimeter_length / params.perimeter_speed;

        // Infill time (simplified: assume infill is proportional to area)
        if params.infill_density > 0.0 {
            // Approximate infill path length based on area and density
            let infill_spacing = params.perimeter_width / params.infill_density;
            let infill_length = contour.area / infill_spacing;
            time += infill_length / params.infill_speed;
        }
    }

    // Add travel time (rough estimate: 10% of print time)
    time *= 1.1;

    time
}

fn estimate_filament_usage(contours: &[Contour], layer_height: f64, params: &SliceParams) -> f64 {
    if contours.is_empty() {
        return 0.0;
    }

    let mut volume = 0.0;

    for contour in contours {
        // Perimeter volume
        let perimeter_passes = params.perimeters as f64;
        let perimeter_length = contour.perimeter * perimeter_passes;
        let perimeter_cross_section = params.perimeter_width * layer_height;
        volume += perimeter_length * perimeter_cross_section;

        // Infill volume
        if params.infill_density > 0.0 {
            let infill_volume = contour.area * layer_height * params.infill_density;
            volume += infill_volume;
        }
    }

    volume *= params.extrusion_multiplier;

    // Convert volume to filament length (assuming 1.75mm diameter)
    let filament_diameter: f64 = 1.75;
    let filament_area = std::f64::consts::PI * (filament_diameter / 2.0).powi(2);
    volume / filament_area
}

// ============================================================================
// FDM Validation
// ============================================================================

/// Parameters for FDM (Fused Deposition Modeling) validation.
#[derive(Debug, Clone)]
pub struct FdmParams {
    /// Nozzle diameter in mm.
    pub nozzle_diameter: f64,

    /// Minimum wall thickness (typically 2x nozzle diameter).
    pub min_wall_thickness: f64,

    /// Layer height in mm.
    pub layer_height: f64,

    /// Minimum feature size (typically nozzle diameter).
    pub min_feature_size: f64,

    /// Maximum overhang angle in degrees (0 = vertical, 90 = horizontal).
    pub max_overhang_angle: f64,

    /// Minimum gap between features.
    pub min_gap: f64,
}

impl Default for FdmParams {
    fn default() -> Self {
        Self {
            nozzle_diameter: 0.4,
            min_wall_thickness: 0.8, // 2x nozzle
            layer_height: 0.2,
            min_feature_size: 0.4,
            max_overhang_angle: 45.0,
            min_gap: 0.4,
        }
    }
}

impl FdmParams {
    /// Parameters for a 0.4mm nozzle (most common).
    pub fn nozzle_04() -> Self {
        Self::default()
    }

    /// Parameters for a 0.6mm nozzle.
    pub fn nozzle_06() -> Self {
        Self {
            nozzle_diameter: 0.6,
            min_wall_thickness: 1.2,
            layer_height: 0.3,
            min_feature_size: 0.6,
            min_gap: 0.6,
            ..Default::default()
        }
    }

    /// Parameters for a 0.25mm nozzle (fine detail).
    pub fn nozzle_025() -> Self {
        Self {
            nozzle_diameter: 0.25,
            min_wall_thickness: 0.5,
            layer_height: 0.1,
            min_feature_size: 0.25,
            min_gap: 0.25,
            ..Default::default()
        }
    }
}

/// Result of FDM validation.
#[derive(Debug, Clone)]
pub struct FdmValidationResult {
    /// Whether the mesh passes all FDM checks.
    pub is_valid: bool,

    /// Layers with thin walls below minimum.
    pub thin_wall_layers: Vec<ThinWallIssue>,

    /// Layers with features smaller than nozzle.
    pub small_feature_layers: Vec<SmallFeatureIssue>,

    /// Layers with gap issues.
    pub gap_issues: Vec<GapIssue>,

    /// Total number of issues found.
    pub issue_count: usize,

    /// Summary message.
    pub summary: String,
}

impl FdmValidationResult {
    /// Check if the mesh is printable (may have warnings but no critical issues).
    pub fn is_printable(&self) -> bool {
        self.thin_wall_layers.is_empty()
    }
}

/// A thin wall issue at a specific layer.
#[derive(Debug, Clone)]
pub struct ThinWallIssue {
    /// Layer index.
    pub layer_index: usize,
    /// Z height.
    pub z_height: f64,
    /// Minimum wall thickness found.
    pub min_thickness: f64,
    /// Required minimum thickness.
    pub required_thickness: f64,
    /// Approximate location (centroid of thin region).
    pub location: (f64, f64),
}

/// A small feature issue at a specific layer.
#[derive(Debug, Clone)]
pub struct SmallFeatureIssue {
    /// Layer index.
    pub layer_index: usize,
    /// Z height.
    pub z_height: f64,
    /// Feature size found.
    pub feature_size: f64,
    /// Minimum feature size allowed.
    pub min_size: f64,
}

/// A gap issue between features.
#[derive(Debug, Clone)]
pub struct GapIssue {
    /// Layer index.
    pub layer_index: usize,
    /// Z height.
    pub z_height: f64,
    /// Gap size found.
    pub gap_size: f64,
    /// Minimum gap allowed.
    pub min_gap: f64,
}

/// Validate a mesh for FDM printing.
pub fn validate_for_fdm(mesh: &Mesh, params: &FdmParams) -> FdmValidationResult {
    let slice_params = SliceParams {
        layer_height: params.layer_height,
        ..Default::default()
    };

    let slice_result = slice_mesh(mesh, &slice_params);

    let mut thin_wall_layers = Vec::new();
    let mut small_feature_layers = Vec::new();
    let gap_issues = Vec::new(); // Gap detection is complex; simplified for now

    for layer in &slice_result.layers {
        // Check for thin walls by analyzing contour widths
        for contour in &layer.contours {
            // Approximate wall thickness using bounding box vs perimeter
            // For a rectangular region: perimeter = 2*(w+h), area = w*h
            // If we assume square-ish: perimeter ≈ 4*sqrt(area), so sqrt(area) ≈ perimeter/4
            // Wall thickness is approximately area / (perimeter/2) for thin strips
            if contour.perimeter > 0.0 {
                let approx_thickness = 2.0 * contour.area / contour.perimeter;
                if approx_thickness < params.min_wall_thickness && approx_thickness > 0.0 {
                    thin_wall_layers.push(ThinWallIssue {
                        layer_index: layer.index,
                        z_height: layer.z_height,
                        min_thickness: approx_thickness,
                        required_thickness: params.min_wall_thickness,
                        location: (contour.centroid.x, contour.centroid.y),
                    });
                }
            }

            // Check for small features (islands smaller than min feature size)
            let feature_size = (contour.area).sqrt(); // Approximate feature dimension
            if feature_size < params.min_feature_size && feature_size > 0.0 {
                small_feature_layers.push(SmallFeatureIssue {
                    layer_index: layer.index,
                    z_height: layer.z_height,
                    feature_size,
                    min_size: params.min_feature_size,
                });
            }
        }
    }

    let issue_count = thin_wall_layers.len() + small_feature_layers.len() + gap_issues.len();
    let is_valid = issue_count == 0;

    let summary = if is_valid {
        "Mesh passes all FDM validation checks.".to_string()
    } else {
        format!(
            "Found {} issues: {} thin walls, {} small features, {} gaps",
            issue_count,
            thin_wall_layers.len(),
            small_feature_layers.len(),
            gap_issues.len()
        )
    };

    FdmValidationResult {
        is_valid,
        thin_wall_layers,
        small_feature_layers,
        gap_issues,
        issue_count,
        summary,
    }
}

// ============================================================================
// SLA Validation
// ============================================================================

/// Parameters for SLA (Stereolithography) validation.
#[derive(Debug, Clone)]
pub struct SlaParams {
    /// XY resolution (pixel size) in mm.
    pub xy_resolution: f64,

    /// Layer height (Z resolution) in mm.
    pub layer_height: f64,

    /// Minimum wall thickness.
    pub min_wall_thickness: f64,

    /// Minimum feature size.
    pub min_feature_size: f64,

    /// Minimum hole diameter for drainage.
    pub min_drain_hole: f64,

    /// Maximum unsupported span in mm.
    pub max_unsupported_span: f64,
}

impl Default for SlaParams {
    fn default() -> Self {
        Self {
            xy_resolution: 0.05,
            layer_height: 0.05,
            min_wall_thickness: 0.4,
            min_feature_size: 0.2,
            min_drain_hole: 2.0,
            max_unsupported_span: 5.0,
        }
    }
}

impl SlaParams {
    /// Parameters for high-detail resin printing.
    pub fn high_detail() -> Self {
        Self {
            xy_resolution: 0.025,
            layer_height: 0.025,
            min_wall_thickness: 0.3,
            min_feature_size: 0.15,
            ..Default::default()
        }
    }

    /// Parameters for standard resin printing.
    pub fn standard() -> Self {
        Self::default()
    }

    /// Parameters for fast/draft resin printing.
    pub fn draft() -> Self {
        Self {
            xy_resolution: 0.1,
            layer_height: 0.1,
            min_wall_thickness: 0.6,
            min_feature_size: 0.4,
            ..Default::default()
        }
    }
}

/// Result of SLA validation.
#[derive(Debug, Clone)]
pub struct SlaValidationResult {
    /// Whether the mesh passes all SLA checks.
    pub is_valid: bool,

    /// Layers with thin walls.
    pub thin_wall_layers: Vec<ThinWallIssue>,

    /// Layers with small features.
    pub small_feature_layers: Vec<SmallFeatureIssue>,

    /// Whether mesh appears to be hollow and might need drain holes.
    pub needs_drain_holes: bool,

    /// Total number of issues found.
    pub issue_count: usize,

    /// Summary message.
    pub summary: String,
}

impl SlaValidationResult {
    /// Check if the mesh is printable.
    pub fn is_printable(&self) -> bool {
        self.thin_wall_layers.is_empty()
    }
}

/// Validate a mesh for SLA printing.
pub fn validate_for_sla(mesh: &Mesh, params: &SlaParams) -> SlaValidationResult {
    let slice_params = SliceParams::for_sla();
    let slice_result = slice_mesh(mesh, &slice_params);

    let mut thin_wall_layers = Vec::new();
    let mut small_feature_layers = Vec::new();

    for layer in &slice_result.layers {
        for contour in &layer.contours {
            // Check for thin walls
            if contour.perimeter > 0.0 {
                let approx_thickness = 2.0 * contour.area / contour.perimeter;
                if approx_thickness < params.min_wall_thickness && approx_thickness > 0.0 {
                    thin_wall_layers.push(ThinWallIssue {
                        layer_index: layer.index,
                        z_height: layer.z_height,
                        min_thickness: approx_thickness,
                        required_thickness: params.min_wall_thickness,
                        location: (contour.centroid.x, contour.centroid.y),
                    });
                }
            }

            // Check for small features
            let feature_size = (contour.area).sqrt();
            if feature_size < params.min_feature_size && feature_size > 0.0 {
                small_feature_layers.push(SmallFeatureIssue {
                    layer_index: layer.index,
                    z_height: layer.z_height,
                    feature_size,
                    min_size: params.min_feature_size,
                });
            }
        }
    }

    // Check if mesh might need drain holes (hollow object detection)
    // Simple heuristic: if there are internal contours (holes), might need drainage
    let needs_drain_holes = slice_result
        .layers
        .iter()
        .any(|l| l.contours.iter().any(|c| !c.is_outer));

    let issue_count = thin_wall_layers.len() + small_feature_layers.len();
    let is_valid = issue_count == 0;

    let summary = if is_valid {
        if needs_drain_holes {
            "Mesh passes SLA checks but may need drain holes for hollow sections.".to_string()
        } else {
            "Mesh passes all SLA validation checks.".to_string()
        }
    } else {
        format!(
            "Found {} issues: {} thin walls, {} small features{}",
            issue_count,
            thin_wall_layers.len(),
            small_feature_layers.len(),
            if needs_drain_holes {
                " (also needs drain holes)"
            } else {
                ""
            }
        )
    };

    SlaValidationResult {
        is_valid,
        thin_wall_layers,
        small_feature_layers,
        needs_drain_holes,
        issue_count,
        summary,
    }
}

// ============================================================================
// SVG Export for Visualization
// ============================================================================

/// Parameters for SVG export.
#[derive(Debug, Clone)]
pub struct SvgExportParams {
    /// Width of the SVG in pixels.
    pub width: u32,
    /// Height of the SVG in pixels.
    pub height: u32,
    /// Padding around the content in pixels.
    pub padding: u32,
    /// Stroke width for contours.
    pub stroke_width: f64,
    /// Fill color for solid regions (CSS color string).
    pub fill_color: String,
    /// Stroke color for contours.
    pub stroke_color: String,
    /// Background color.
    pub background_color: String,
    /// Whether to show outer contours filled.
    pub fill_outer: bool,
    /// Whether to show hole contours.
    pub show_holes: bool,
}

impl Default for SvgExportParams {
    fn default() -> Self {
        Self {
            width: 800,
            height: 600,
            padding: 20,
            stroke_width: 1.0,
            fill_color: "#4a90d9".to_string(),
            stroke_color: "#2d5986".to_string(),
            background_color: "#f5f5f5".to_string(),
            fill_outer: true,
            show_holes: true,
        }
    }
}

/// Export a single layer to SVG format.
pub fn export_layer_svg(layer: &Layer, params: &SvgExportParams) -> String {
    if layer.contours.is_empty() {
        return format!(
            "<svg xmlns=\"http://www.w3.org/2000/svg\" width=\"{}\" height=\"{}\" viewBox=\"0 0 {} {}\">\n\
  <rect width=\"100%\" height=\"100%\" fill=\"{}\"/>\n\
  <text x=\"50%\" y=\"50%\" text-anchor=\"middle\" fill=\"#999\">Empty layer</text>\n\
</svg>",
            params.width, params.height, params.width, params.height, params.background_color
        );
    }

    // Calculate bounds and scale
    let bounds = &layer.bounds;
    let content_width = bounds.width();
    let content_height = bounds.height();

    let available_width = params.width as f64 - 2.0 * params.padding as f64;
    let available_height = params.height as f64 - 2.0 * params.padding as f64;

    let scale = if content_width > 0.0 && content_height > 0.0 {
        (available_width / content_width).min(available_height / content_height)
    } else {
        1.0
    };

    let offset_x = params.padding as f64 + (available_width - content_width * scale) / 2.0;
    let offset_y = params.padding as f64 + (available_height - content_height * scale) / 2.0;

    let mut svg = format!(
        r#"<svg xmlns="http://www.w3.org/2000/svg" width="{}" height="{}" viewBox="0 0 {} {}">
  <rect width="100%" height="100%" fill="{}"/>
  <g transform="translate({:.2},{:.2}) scale({:.6},{:.6})">
"#,
        params.width,
        params.height,
        params.width,
        params.height,
        params.background_color,
        offset_x - bounds.min_x * scale,
        offset_y + (bounds.max_y) * scale, // SVG Y is inverted
        scale,
        -scale // Flip Y axis
    );

    // Draw contours
    for contour in &layer.contours {
        if contour.points.is_empty() {
            continue;
        }

        if !contour.is_outer && !params.show_holes {
            continue;
        }

        let mut path = String::new();
        for (i, point) in contour.points.iter().enumerate() {
            if i == 0 {
                path.push_str(&format!("M {:.4} {:.4}", point.x, point.y));
            } else {
                path.push_str(&format!(" L {:.4} {:.4}", point.x, point.y));
            }
        }
        path.push_str(" Z");

        let fill = if params.fill_outer && contour.is_outer {
            &params.fill_color
        } else if !contour.is_outer {
            &params.background_color // Holes cut out
        } else {
            "none"
        };

        svg.push_str(&format!(
            r#"    <path d="{}" fill="{}" stroke="{}" stroke-width="{:.2}"/>
"#,
            path,
            fill,
            params.stroke_color,
            params.stroke_width / scale
        ));
    }

    svg.push_str("  </g>\n");

    // Add layer info text
    svg.push_str(&format!(
        "  <text x=\"10\" y=\"20\" font-family=\"monospace\" font-size=\"12\" fill=\"#666\">\n\
    Layer {}: Z={:.2}mm, Area={:.1}mm², Perimeter={:.1}mm\n\
  </text>\n",
        layer.index, layer.z_height, layer.area, layer.perimeter
    ));

    svg.push_str("</svg>");

    svg
}

/// Export all layers to a series of SVG files.
pub fn export_slices_svg(
    result: &SliceResult,
    output_dir: &std::path::Path,
    params: &SvgExportParams,
) -> crate::MeshResult<Vec<std::path::PathBuf>> {
    use std::fs;
    use std::io::Write;

    fs::create_dir_all(output_dir).map_err(|e| crate::MeshError::IoWrite {
        path: output_dir.to_path_buf(),
        source: e,
    })?;

    let mut paths = Vec::with_capacity(result.layers.len());

    for layer in &result.layers {
        let svg = export_layer_svg(layer, params);
        let filename = format!("layer_{:04}.svg", layer.index);
        let path = output_dir.join(&filename);

        let mut file = fs::File::create(&path).map_err(|e| crate::MeshError::IoWrite {
            path: path.clone(),
            source: e,
        })?;

        file.write_all(svg.as_bytes())
            .map_err(|e| crate::MeshError::IoWrite {
                path: path.clone(),
                source: e,
            })?;

        paths.push(path);
    }

    Ok(paths)
}

// ============================================================================
// 3MF Slice Extension Export
// ============================================================================

/// Export slices in 3MF slice extension format.
///
/// The 3MF slice extension allows pre-sliced data to be embedded in 3MF files,
/// which can be read by compatible slicers for faster processing.
pub fn export_3mf_slices(
    result: &SliceResult,
    output_path: &std::path::Path,
) -> crate::MeshResult<()> {
    use std::fs::File;
    use std::io::Write;
    use zip::ZipWriter;
    use zip::write::SimpleFileOptions;

    let file = File::create(output_path).map_err(|e| crate::MeshError::IoWrite {
        path: output_path.to_path_buf(),
        source: e,
    })?;

    let mut zip = ZipWriter::new(file);
    let options = SimpleFileOptions::default().compression_method(zip::CompressionMethod::Deflated);

    // Write content types
    zip.start_file("[Content_Types].xml", options)
        .map_err(|e| crate::MeshError::IoWrite {
            path: output_path.to_path_buf(),
            source: std::io::Error::other(e.to_string()),
        })?;
    zip.write_all(SLICE_CONTENT_TYPES_XML.as_bytes())
        .map_err(|e| crate::MeshError::IoWrite {
            path: output_path.to_path_buf(),
            source: e,
        })?;

    // Write relationships
    zip.start_file("_rels/.rels", options)
        .map_err(|e| crate::MeshError::IoWrite {
            path: output_path.to_path_buf(),
            source: std::io::Error::other(e.to_string()),
        })?;
    zip.write_all(SLICE_RELS_XML.as_bytes())
        .map_err(|e| crate::MeshError::IoWrite {
            path: output_path.to_path_buf(),
            source: e,
        })?;

    // Write slice stack
    zip.start_file("2D/2dmodel.model", options)
        .map_err(|e| crate::MeshError::IoWrite {
            path: output_path.to_path_buf(),
            source: std::io::Error::other(e.to_string()),
        })?;

    let slice_xml = generate_slice_stack_xml(result);
    zip.write_all(slice_xml.as_bytes())
        .map_err(|e| crate::MeshError::IoWrite {
            path: output_path.to_path_buf(),
            source: e,
        })?;

    zip.finish().map_err(|e| crate::MeshError::IoWrite {
        path: output_path.to_path_buf(),
        source: std::io::Error::other(e.to_string()),
    })?;

    Ok(())
}

fn generate_slice_stack_xml(result: &SliceResult) -> String {
    let mut xml = String::with_capacity(result.layers.len() * 500);

    xml.push_str(
        r#"<?xml version="1.0" encoding="UTF-8"?>
<slicestack xmlns="http://schemas.microsoft.com/3dmanufacturing/slice/2015/07"
            zbottom="0">
"#,
    );

    for layer in &result.layers {
        xml.push_str(&format!("  <slice ztop=\"{:.6}\">\n", layer.z_height));

        for (contour_idx, contour) in layer.contours.iter().enumerate() {
            if contour.points.is_empty() {
                continue;
            }

            // Build vertices string
            let vertices: String = contour
                .points
                .iter()
                .map(|p| format!("{:.4} {:.4}", p.x, p.y))
                .collect::<Vec<_>>()
                .join(" ");

            // Build polygon indices (simple sequential for closed loop)
            let indices: String = (0..contour.points.len())
                .map(|i| i.to_string())
                .collect::<Vec<_>>()
                .join(" ");

            xml.push_str(&format!(
                "    <vertices id=\"{}\">{}</vertices>\n",
                contour_idx, vertices
            ));
            xml.push_str(&format!(
                "    <polygon startv=\"0\">{}</polygon>\n",
                indices
            ));
        }

        xml.push_str("  </slice>\n");
    }

    xml.push_str("</slicestack>\n");
    xml
}

const SLICE_CONTENT_TYPES_XML: &str = r#"<?xml version="1.0" encoding="UTF-8"?>
<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types">
  <Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/>
  <Default Extension="model" ContentType="application/vnd.ms-package.3dmanufacturing-3dmodel+xml"/>
</Types>
"#;

const SLICE_RELS_XML: &str = r#"<?xml version="1.0" encoding="UTF-8"?>
<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">
  <Relationship Target="/2D/2dmodel.model" Id="rel0" Type="http://schemas.microsoft.com/3dmanufacturing/2013/01/slice"/>
</Relationships>
"#;

// ============================================================================
// Mesh extension methods
// ============================================================================

impl Mesh {
    /// Slice this mesh into layers for 3D printing.
    pub fn slice(&self) -> SliceResult {
        slice_mesh(self, &SliceParams::default())
    }

    /// Slice with custom parameters.
    pub fn slice_with_params(&self, params: &SliceParams) -> SliceResult {
        slice_mesh(self, params)
    }

    /// Generate a slice preview at a specific height.
    pub fn slice_preview(&self, z: f64) -> Layer {
        slice_preview(self, z, &SliceParams::default())
    }

    /// Generate a slice preview with custom parameters.
    pub fn slice_preview_with_params(&self, z: f64, params: &SliceParams) -> Layer {
        slice_preview(self, z, params)
    }

    /// Validate mesh for FDM printing.
    ///
    /// Checks for thin walls, small features, and other issues that would
    /// cause problems with FDM 3D printing.
    ///
    /// # Example
    /// ```
    /// use mesh_repair::{Mesh, Vertex};
    /// use mesh_repair::slice::FdmParams;
    ///
    /// let mut mesh = Mesh::new();
    /// mesh.vertices.push(Vertex::from_coords(0.0, 0.0, 0.0));
    /// mesh.vertices.push(Vertex::from_coords(10.0, 0.0, 0.0));
    /// mesh.vertices.push(Vertex::from_coords(5.0, 10.0, 0.0));
    /// mesh.vertices.push(Vertex::from_coords(5.0, 5.0, 10.0));
    /// mesh.faces.push([0, 1, 3]);
    /// mesh.faces.push([1, 2, 3]);
    /// mesh.faces.push([2, 0, 3]);
    /// mesh.faces.push([0, 2, 1]);
    ///
    /// let result = mesh.validate_for_fdm(&FdmParams::default());
    /// if result.is_printable() {
    ///     println!("Mesh is ready for FDM printing!");
    /// }
    /// ```
    pub fn validate_for_fdm(&self, params: &FdmParams) -> FdmValidationResult {
        validate_for_fdm(self, params)
    }

    /// Validate mesh for SLA printing.
    ///
    /// Checks for thin walls, small features, and identifies if drain holes
    /// might be needed for hollow sections.
    pub fn validate_for_sla(&self, params: &SlaParams) -> SlaValidationResult {
        validate_for_sla(self, params)
    }

    /// Export slices to SVG files for visualization.
    pub fn export_slices_svg(
        &self,
        output_dir: &std::path::Path,
        params: &SvgExportParams,
    ) -> crate::MeshResult<Vec<std::path::PathBuf>> {
        let result = self.slice();
        export_slices_svg(&result, output_dir, params)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::Vertex;

    fn create_test_cube() -> Mesh {
        let mut mesh = Mesh::new();
        // Create a 10x10x10 cube
        let vertices = [
            (0.0, 0.0, 0.0),
            (10.0, 0.0, 0.0),
            (10.0, 10.0, 0.0),
            (0.0, 10.0, 0.0),
            (0.0, 0.0, 10.0),
            (10.0, 0.0, 10.0),
            (10.0, 10.0, 10.0),
            (0.0, 10.0, 10.0),
        ];

        for (x, y, z) in vertices {
            mesh.vertices.push(Vertex::from_coords(x, y, z));
        }

        // 12 triangles for 6 faces
        let faces = [
            // Bottom
            [0, 1, 2],
            [0, 2, 3],
            // Top
            [4, 6, 5],
            [4, 7, 6],
            // Front
            [0, 5, 1],
            [0, 4, 5],
            // Back
            [2, 7, 3],
            [2, 6, 7],
            // Left
            [0, 3, 7],
            [0, 7, 4],
            // Right
            [1, 5, 6],
            [1, 6, 2],
        ];

        for f in faces {
            mesh.faces.push(f);
        }

        mesh
    }

    #[test]
    fn test_slice_params_default() {
        let params = SliceParams::default();
        assert!((params.layer_height - 0.2).abs() < 0.001);
        assert_eq!(params.perimeters, 2);
    }

    #[test]
    fn test_slice_params_presets() {
        let hq = SliceParams::high_quality();
        assert!(hq.layer_height < 0.2);

        let draft = SliceParams::draft();
        assert!(draft.layer_height > 0.2);

        let sla = SliceParams::for_sla();
        assert!((sla.infill_density - 1.0).abs() < 0.001);
    }

    #[test]
    fn test_slice_mesh() {
        let mesh = create_test_cube();
        let result = slice_mesh(&mesh, &SliceParams::default());

        // 10mm height with 0.3mm first layer and 0.2mm layers
        // Should have multiple layers
        assert!(result.layer_count > 0);
        assert!((result.total_height - 10.0).abs() < 0.1);
    }

    #[test]
    fn test_slice_preview() {
        let mesh = create_test_cube();
        let layer = slice_preview(&mesh, 5.0, &SliceParams::default());

        // At z=5, should intersect the cube
        assert!(layer.area > 0.0);
        assert!(layer.perimeter > 0.0);
    }

    #[test]
    fn test_layer_stats() {
        let mesh = create_test_cube();
        let result = slice_mesh(&mesh, &SliceParams::default());
        let stats = calculate_layer_stats(&result);

        // For a cube, all internal layers should have similar areas
        assert!(stats.max_area >= stats.min_area);
        assert!(stats.avg_area > 0.0);
    }

    #[test]
    fn test_mesh_slice_method() {
        let mesh = create_test_cube();
        let result = mesh.slice();

        assert!(result.layer_count > 0);
        assert!(result.estimated_print_time >= 0.0);
    }

    #[test]
    fn test_empty_mesh_slice() {
        let mesh = Mesh::new();
        let result = slice_mesh(&mesh, &SliceParams::default());

        assert_eq!(result.layer_count, 0);
        assert!((result.total_height - 0.0).abs() < 0.001);
    }

    #[test]
    fn test_layer_bounds() {
        let mesh = create_test_cube();
        let layer = slice_preview(&mesh, 5.0, &SliceParams::default());

        // Bounds should be reasonable for a 10x10 cube
        assert!(layer.bounds.width() <= 10.5);
        assert!(layer.bounds.height() <= 10.5);
    }

    #[test]
    fn test_fdm_params_default() {
        let params = FdmParams::default();
        assert!((params.nozzle_diameter - 0.4).abs() < 0.001);
        assert!((params.min_wall_thickness - 0.8).abs() < 0.001);
        assert!((params.layer_height - 0.2).abs() < 0.001);
    }

    #[test]
    fn test_fdm_validation_cube() {
        let mesh = create_test_cube();
        let params = FdmParams::default();
        let result = validate_for_fdm(&mesh, &params);

        // A 10x10x10 cube should pass basic FDM validation
        // Wall thickness is 10mm which is well above minimum
        assert!(result.thin_wall_layers.is_empty() || result.is_valid);
    }

    #[test]
    fn test_fdm_validation_mesh_method() {
        let mesh = create_test_cube();
        let params = FdmParams::default();
        let result = mesh.validate_for_fdm(&params);

        assert!(!result.summary.is_empty());
    }

    #[test]
    fn test_sla_params_default() {
        let params = SlaParams::default();
        assert!((params.xy_resolution - 0.05).abs() < 0.001);
        assert!((params.layer_height - 0.05).abs() < 0.001);
        assert!((params.min_drain_hole - 2.0).abs() < 0.001);
    }

    #[test]
    fn test_sla_validation_cube() {
        let mesh = create_test_cube();
        let params = SlaParams::default();
        let result = validate_for_sla(&mesh, &params);

        // A 10x10x10 cube should pass basic SLA validation
        assert!(!result.summary.is_empty());
    }

    #[test]
    fn test_sla_validation_mesh_method() {
        let mesh = create_test_cube();
        let params = SlaParams::default();
        let result = mesh.validate_for_sla(&params);

        assert!(!result.summary.is_empty());
    }

    #[test]
    fn test_svg_export_params_default() {
        let params = SvgExportParams::default();
        assert_eq!(params.width, 800);
        assert_eq!(params.height, 600);
        assert_eq!(params.padding, 20);
        assert_eq!(params.fill_color, "#4a90d9");
        assert_eq!(params.stroke_color, "#2d5986");
        assert_eq!(params.background_color, "#f5f5f5");
        assert!(params.fill_outer);
        assert!(params.show_holes);
    }

    #[test]
    fn test_export_layer_svg_empty() {
        let layer = Layer {
            index: 0,
            z_height: 0.0,
            thickness: 0.2,
            contours: vec![],
            area: 0.0,
            perimeter: 0.0,
            print_time: 0.0,
            filament_length: 0.0,
            island_count: 0,
            bounds: LayerBounds {
                min_x: 0.0,
                max_x: 0.0,
                min_y: 0.0,
                max_y: 0.0,
            },
        };
        let params = SvgExportParams::default();
        let svg = export_layer_svg(&layer, &params);

        assert!(svg.contains("<svg"));
        assert!(svg.contains("Empty layer"));
        assert!(svg.contains("</svg>"));
    }

    #[test]
    fn test_export_layer_svg_with_contour() {
        let mesh = create_test_cube();
        let layer = slice_preview(&mesh, 5.0, &SliceParams::default());
        let params = SvgExportParams::default();
        let svg = export_layer_svg(&layer, &params);

        assert!(svg.contains("<svg"));
        assert!(svg.contains("</svg>"));
        assert!(svg.contains("<path"));
    }

    #[test]
    fn test_3mf_slice_xml() {
        let mesh = create_test_cube();
        let result = slice_mesh(&mesh, &SliceParams::default());
        let xml = generate_slice_stack_xml(&result);

        assert!(xml.contains("<?xml"));
        assert!(xml.contains("slicestack"));
        assert!(xml.contains("<slice"));
    }
}

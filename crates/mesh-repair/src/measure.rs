//! Measurement and dimensioning tools.
//!
//! This module provides tools for measuring distances, extracting cross-sections,
//! and computing dimensions of meshes.
//!
//! # Use Cases
//!
//! - Measure distances between points on a mesh
//! - Extract cross-sections at specific heights
//! - Compute bounding box dimensions
//! - Find circumference at a given height
//!
//! # Example
//!
//! ```
//! use mesh_repair::{Mesh, Vertex};
//! use mesh_repair::measure::{cross_section, dimensions};
//! use nalgebra::{Point3, Vector3};
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
//! // Get mesh dimensions
//! let dims = dimensions(&mesh);
//! println!("Size: {:.1} x {:.1} x {:.1} mm", dims.width, dims.depth, dims.height);
//!
//! // Extract a cross-section at z=5
//! let section = cross_section(&mesh, Point3::new(0.0, 0.0, 5.0), Vector3::z());
//! println!("Cross-section area: {:.2} mm²", section.area);
//! ```

use crate::Mesh;
use nalgebra::{Matrix3, Point3, Rotation3, Vector3};

/// Result of dimension extraction.
#[derive(Debug, Clone)]
pub struct Dimensions {
    /// Bounding box minimum point.
    pub min: Point3<f64>,
    /// Bounding box maximum point.
    pub max: Point3<f64>,
    /// Width (X dimension).
    pub width: f64,
    /// Depth (Y dimension).
    pub depth: f64,
    /// Height (Z dimension).
    pub height: f64,
    /// Diagonal length.
    pub diagonal: f64,
    /// Volume of bounding box.
    pub bounding_volume: f64,
    /// Center of bounding box.
    pub center: Point3<f64>,
}

/// Result of cross-section extraction.
#[derive(Debug, Clone)]
pub struct CrossSection {
    /// Points on the cross-section boundary (ordered).
    pub points: Vec<Point3<f64>>,
    /// Perimeter length.
    pub perimeter: f64,
    /// Area enclosed by the cross-section.
    pub area: f64,
    /// Centroid of the cross-section.
    pub centroid: Point3<f64>,
    /// Bounding box of the cross-section (2D, on the plane).
    pub bounds: (Point3<f64>, Point3<f64>),
    /// Plane origin point.
    pub plane_origin: Point3<f64>,
    /// Plane normal.
    pub plane_normal: Vector3<f64>,
    /// Number of separate contours.
    pub contour_count: usize,
}

/// Result of oriented bounding box computation.
#[derive(Debug, Clone)]
pub struct OrientedBoundingBox {
    /// Center of the OBB.
    pub center: Point3<f64>,
    /// Half-extents along each axis.
    pub half_extents: Vector3<f64>,
    /// Rotation of the OBB (local axes).
    pub rotation: Rotation3<f64>,
    /// Volume of the OBB.
    pub volume: f64,
    /// Vertices of the OBB (8 corners).
    pub vertices: [Point3<f64>; 8],
}

/// Distance measurement result.
#[derive(Debug, Clone)]
pub struct DistanceMeasurement {
    /// Start point.
    pub from: Point3<f64>,
    /// End point.
    pub to: Point3<f64>,
    /// Euclidean distance.
    pub distance: f64,
    /// Distance along X.
    pub dx: f64,
    /// Distance along Y.
    pub dy: f64,
    /// Distance along Z.
    pub dz: f64,
}

// ============================================================================
// Dimension functions
// ============================================================================

/// Extract dimensions of a mesh.
pub fn dimensions(mesh: &Mesh) -> Dimensions {
    if mesh.vertices.is_empty() {
        return Dimensions {
            min: Point3::origin(),
            max: Point3::origin(),
            width: 0.0,
            depth: 0.0,
            height: 0.0,
            diagonal: 0.0,
            bounding_volume: 0.0,
            center: Point3::origin(),
        };
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

    let width = max.x - min.x;
    let depth = max.y - min.y;
    let height = max.z - min.z;

    Dimensions {
        min,
        max,
        width,
        depth,
        height,
        diagonal: (width * width + depth * depth + height * height).sqrt(),
        bounding_volume: width * depth * height,
        center: Point3::new(
            (min.x + max.x) / 2.0,
            (min.y + max.y) / 2.0,
            (min.z + max.z) / 2.0,
        ),
    }
}

/// Compute an oriented bounding box (minimal volume).
pub fn oriented_bounding_box(mesh: &Mesh) -> OrientedBoundingBox {
    if mesh.vertices.is_empty() {
        return OrientedBoundingBox {
            center: Point3::origin(),
            half_extents: Vector3::zeros(),
            rotation: Rotation3::identity(),
            volume: 0.0,
            vertices: [Point3::origin(); 8],
        };
    }

    // Compute covariance matrix for PCA
    let centroid = compute_centroid(mesh);
    let cov = compute_covariance_matrix(mesh, &centroid);

    // Eigen decomposition for principal axes
    let (eigenvectors, _eigenvalues) = symmetric_eigen_decomposition(&cov);

    // Create rotation from eigenvectors (principal axes)
    let rotation = Rotation3::from_matrix_unchecked(eigenvectors);

    // Transform points to local coordinates and find AABB
    let mut local_min = Vector3::new(f64::INFINITY, f64::INFINITY, f64::INFINITY);
    let mut local_max = Vector3::new(f64::NEG_INFINITY, f64::NEG_INFINITY, f64::NEG_INFINITY);

    for v in &mesh.vertices {
        let local = rotation.inverse() * (v.position - centroid);
        local_min.x = local_min.x.min(local.x);
        local_min.y = local_min.y.min(local.y);
        local_min.z = local_min.z.min(local.z);
        local_max.x = local_max.x.max(local.x);
        local_max.y = local_max.y.max(local.y);
        local_max.z = local_max.z.max(local.z);
    }

    let half_extents = (local_max - local_min) / 2.0;
    let local_center = (local_min + local_max) / 2.0;
    let center = Point3::from(centroid.coords + rotation * local_center);

    let volume = 8.0 * half_extents.x * half_extents.y * half_extents.z;

    // Compute 8 corners
    let corners = [
        Vector3::new(-half_extents.x, -half_extents.y, -half_extents.z),
        Vector3::new(half_extents.x, -half_extents.y, -half_extents.z),
        Vector3::new(half_extents.x, half_extents.y, -half_extents.z),
        Vector3::new(-half_extents.x, half_extents.y, -half_extents.z),
        Vector3::new(-half_extents.x, -half_extents.y, half_extents.z),
        Vector3::new(half_extents.x, -half_extents.y, half_extents.z),
        Vector3::new(half_extents.x, half_extents.y, half_extents.z),
        Vector3::new(-half_extents.x, half_extents.y, half_extents.z),
    ];

    let vertices = corners.map(|c| Point3::from(center.coords + rotation * c));

    OrientedBoundingBox {
        center,
        half_extents,
        rotation,
        volume,
        vertices,
    }
}

// ============================================================================
// Cross-section functions
// ============================================================================

/// Extract a cross-section of the mesh at a given plane.
pub fn cross_section(
    mesh: &Mesh,
    plane_point: Point3<f64>,
    plane_normal: Vector3<f64>,
) -> CrossSection {
    let normal = plane_normal.normalize();
    let mut segments: Vec<(Point3<f64>, Point3<f64>)> = Vec::new();

    // Find all edge intersections with the plane
    for face in &mesh.faces {
        let v0 = mesh.vertices[face[0] as usize].position;
        let v1 = mesh.vertices[face[1] as usize].position;
        let v2 = mesh.vertices[face[2] as usize].position;

        let mut intersections = Vec::new();

        // Check each edge
        for (a, b) in [(v0, v1), (v1, v2), (v2, v0)] {
            if let Some(p) = plane_edge_intersection(plane_point, normal, a, b) {
                intersections.push(p);
            }
        }

        // If we have exactly 2 intersections, we have a segment
        if intersections.len() == 2 {
            segments.push((intersections[0], intersections[1]));
        }
    }

    if segments.is_empty() {
        return CrossSection {
            points: Vec::new(),
            perimeter: 0.0,
            area: 0.0,
            centroid: plane_point,
            bounds: (plane_point, plane_point),
            plane_origin: plane_point,
            plane_normal: normal,
            contour_count: 0,
        };
    }

    // Chain segments into contours
    let contours = chain_segments(&segments);
    let contour_count = contours.len();

    // Flatten all points for calculations
    let all_points: Vec<Point3<f64>> = contours.into_iter().flatten().collect();

    // Calculate perimeter
    let mut perimeter = 0.0;
    for segment in &segments {
        perimeter += (segment.1 - segment.0).norm();
    }

    // Calculate area using the shoelace formula (projected to 2D on plane)
    let area = calculate_cross_section_area(&all_points, normal);

    // Calculate centroid
    let centroid = if all_points.is_empty() {
        plane_point
    } else {
        let sum: Vector3<f64> = all_points.iter().map(|p| p.coords).sum();
        Point3::from(sum / all_points.len() as f64)
    };

    // Calculate bounds
    let (min, max) = compute_points_bounds(&all_points);

    CrossSection {
        points: all_points,
        perimeter,
        area,
        centroid,
        bounds: (min, max),
        plane_origin: plane_point,
        plane_normal: normal,
        contour_count,
    }
}

/// Extract multiple cross-sections at regular intervals.
pub fn cross_sections(
    mesh: &Mesh,
    start: Point3<f64>,
    normal: Vector3<f64>,
    count: usize,
    spacing: f64,
) -> Vec<CrossSection> {
    let normal = normal.normalize();
    let mut sections = Vec::with_capacity(count);

    for i in 0..count {
        let offset = i as f64 * spacing;
        let plane_point = Point3::from(start.coords + normal * offset);
        sections.push(cross_section(mesh, plane_point, normal));
    }

    sections
}

/// Measure circumference at a given height (Z coordinate).
pub fn circumference_at_height(mesh: &Mesh, z: f64) -> f64 {
    let section = cross_section(mesh, Point3::new(0.0, 0.0, z), Vector3::z());
    section.perimeter
}

// ============================================================================
// Distance measurement functions
// ============================================================================

/// Measure distance between two points.
pub fn measure_distance(from: Point3<f64>, to: Point3<f64>) -> DistanceMeasurement {
    let diff = to - from;
    DistanceMeasurement {
        from,
        to,
        distance: diff.norm(),
        dx: diff.x.abs(),
        dy: diff.y.abs(),
        dz: diff.z.abs(),
    }
}

/// Find the closest point on the mesh to a given point.
pub fn closest_point_on_mesh(mesh: &Mesh, point: Point3<f64>) -> Option<Point3<f64>> {
    if mesh.faces.is_empty() {
        return None;
    }

    let mut closest = None;
    let mut min_dist_sq = f64::INFINITY;

    for face in &mesh.faces {
        let v0 = mesh.vertices[face[0] as usize].position;
        let v1 = mesh.vertices[face[1] as usize].position;
        let v2 = mesh.vertices[face[2] as usize].position;

        let p = closest_point_on_triangle(point, v0, v1, v2);
        let dist_sq = (p - point).norm_squared();

        if dist_sq < min_dist_sq {
            min_dist_sq = dist_sq;
            closest = Some(p);
        }
    }

    closest
}

// Note: surface_area() and volume() functions are defined in types.rs

// ============================================================================
// Internal helper functions
// ============================================================================

fn compute_centroid(mesh: &Mesh) -> Point3<f64> {
    if mesh.vertices.is_empty() {
        return Point3::origin();
    }

    let sum: Vector3<f64> = mesh.vertices.iter().map(|v| v.position.coords).sum();
    Point3::from(sum / mesh.vertices.len() as f64)
}

fn compute_covariance_matrix(mesh: &Mesh, centroid: &Point3<f64>) -> Matrix3<f64> {
    let mut cov = Matrix3::zeros();

    for v in &mesh.vertices {
        let d = v.position - centroid;
        cov[(0, 0)] += d.x * d.x;
        cov[(0, 1)] += d.x * d.y;
        cov[(0, 2)] += d.x * d.z;
        cov[(1, 0)] += d.y * d.x;
        cov[(1, 1)] += d.y * d.y;
        cov[(1, 2)] += d.y * d.z;
        cov[(2, 0)] += d.z * d.x;
        cov[(2, 1)] += d.z * d.y;
        cov[(2, 2)] += d.z * d.z;
    }

    cov / mesh.vertices.len() as f64
}

fn symmetric_eigen_decomposition(m: &Matrix3<f64>) -> (Matrix3<f64>, Vector3<f64>) {
    // Simple power iteration for symmetric 3x3 matrix
    // For production, use a proper eigen solver
    let eigen = m.symmetric_eigen();
    (eigen.eigenvectors, eigen.eigenvalues)
}

fn plane_edge_intersection(
    plane_point: Point3<f64>,
    plane_normal: Vector3<f64>,
    a: Point3<f64>,
    b: Point3<f64>,
) -> Option<Point3<f64>> {
    let d_a = (a - plane_point).dot(&plane_normal);
    let d_b = (b - plane_point).dot(&plane_normal);

    // Check if edge crosses the plane
    if d_a * d_b > 0.0 {
        return None; // Same side of plane
    }

    if (d_a - d_b).abs() < 1e-10 {
        return None; // Edge parallel to plane
    }

    let t = d_a / (d_a - d_b);
    let direction = b - a;
    Some(Point3::from(a.coords + direction * t))
}

fn chain_segments(segments: &[(Point3<f64>, Point3<f64>)]) -> Vec<Vec<Point3<f64>>> {
    if segments.is_empty() {
        return Vec::new();
    }

    let mut remaining: Vec<_> = segments.to_vec();
    let mut contours = Vec::new();

    while !remaining.is_empty() {
        let mut contour = Vec::new();
        let first = remaining.remove(0);
        contour.push(first.0);
        contour.push(first.1);

        let mut changed = true;
        while changed {
            changed = false;

            // Copy endpoints to avoid borrow issues
            let start = *contour.first().unwrap();
            let end = *contour.last().unwrap();
            let eps = 1e-6;

            for i in (0..remaining.len()).rev() {
                let seg = &remaining[i];

                if (seg.0 - end).norm() < eps {
                    contour.push(seg.1);
                    remaining.remove(i);
                    changed = true;
                } else if (seg.1 - end).norm() < eps {
                    contour.push(seg.0);
                    remaining.remove(i);
                    changed = true;
                } else if (seg.0 - start).norm() < eps {
                    contour.insert(0, seg.1);
                    remaining.remove(i);
                    changed = true;
                } else if (seg.1 - start).norm() < eps {
                    contour.insert(0, seg.0);
                    remaining.remove(i);
                    changed = true;
                }
            }
        }

        contours.push(contour);
    }

    contours
}

fn calculate_cross_section_area(points: &[Point3<f64>], normal: Vector3<f64>) -> f64 {
    if points.len() < 3 {
        return 0.0;
    }

    // Project points onto 2D plane
    // Create orthonormal basis on the plane
    let u = if normal.x.abs() < 0.9 {
        Vector3::x().cross(&normal).normalize()
    } else {
        Vector3::y().cross(&normal).normalize()
    };
    let v = normal.cross(&u);

    // Project to 2D
    let points_2d: Vec<(f64, f64)> = points
        .iter()
        .map(|p| (p.coords.dot(&u), p.coords.dot(&v)))
        .collect();

    // Shoelace formula
    let mut area = 0.0;
    let n = points_2d.len();
    for i in 0..n {
        let j = (i + 1) % n;
        area += points_2d[i].0 * points_2d[j].1;
        area -= points_2d[j].0 * points_2d[i].1;
    }

    (area / 2.0).abs()
}

fn compute_points_bounds(points: &[Point3<f64>]) -> (Point3<f64>, Point3<f64>) {
    if points.is_empty() {
        return (Point3::origin(), Point3::origin());
    }

    let mut min = points[0];
    let mut max = points[0];

    for p in points {
        min.x = min.x.min(p.x);
        min.y = min.y.min(p.y);
        min.z = min.z.min(p.z);
        max.x = max.x.max(p.x);
        max.y = max.y.max(p.y);
        max.z = max.z.max(p.z);
    }

    (min, max)
}

fn closest_point_on_triangle(
    p: Point3<f64>,
    a: Point3<f64>,
    b: Point3<f64>,
    c: Point3<f64>,
) -> Point3<f64> {
    let ab = b - a;
    let ac = c - a;
    let ap = p - a;

    let d1 = ab.dot(&ap);
    let d2 = ac.dot(&ap);
    if d1 <= 0.0 && d2 <= 0.0 {
        return a;
    }

    let bp = p - b;
    let d3 = ab.dot(&bp);
    let d4 = ac.dot(&bp);
    if d3 >= 0.0 && d4 <= d3 {
        return b;
    }

    let vc = d1 * d4 - d3 * d2;
    if vc <= 0.0 && d1 >= 0.0 && d3 <= 0.0 {
        let v = d1 / (d1 - d3);
        return Point3::from(a.coords + ab * v);
    }

    let cp = p - c;
    let d5 = ab.dot(&cp);
    let d6 = ac.dot(&cp);
    if d6 >= 0.0 && d5 <= d6 {
        return c;
    }

    let vb = d5 * d2 - d1 * d6;
    if vb <= 0.0 && d2 >= 0.0 && d6 <= 0.0 {
        let w = d2 / (d2 - d6);
        return Point3::from(a.coords + ac * w);
    }

    let va = d3 * d6 - d5 * d4;
    if va <= 0.0 && (d4 - d3) >= 0.0 && (d5 - d6) >= 0.0 {
        let w = (d4 - d3) / ((d4 - d3) + (d5 - d6));
        return Point3::from(b.coords + (c - b) * w);
    }

    let denom = 1.0 / (va + vb + vc);
    let v = vb * denom;
    let w = vc * denom;
    Point3::from(a.coords + ab * v + ac * w)
}

// ============================================================================
// Mesh extension methods
// ============================================================================

impl Mesh {
    /// Get dimensions of this mesh.
    pub fn dimensions(&self) -> Dimensions {
        dimensions(self)
    }

    /// Compute oriented bounding box.
    pub fn oriented_bounding_box(&self) -> OrientedBoundingBox {
        oriented_bounding_box(self)
    }

    /// Extract a cross-section at a given plane.
    pub fn cross_section(
        &self,
        plane_point: Point3<f64>,
        plane_normal: Vector3<f64>,
    ) -> CrossSection {
        cross_section(self, plane_point, plane_normal)
    }

    /// Extract multiple cross-sections.
    pub fn cross_sections(
        &self,
        start: Point3<f64>,
        normal: Vector3<f64>,
        count: usize,
        spacing: f64,
    ) -> Vec<CrossSection> {
        cross_sections(self, start, normal, count, spacing)
    }

    /// Measure circumference at a given height (Z).
    pub fn circumference_at_height(&self, z: f64) -> f64 {
        circumference_at_height(self, z)
    }

    /// Measure distance between two points.
    pub fn measure_distance(&self, from: Point3<f64>, to: Point3<f64>) -> DistanceMeasurement {
        measure_distance(from, to)
    }

    /// Find closest point on mesh surface.
    pub fn closest_point(&self, point: Point3<f64>) -> Option<Point3<f64>> {
        closest_point_on_mesh(self, point)
    }
    // Note: surface_area() and volume() methods are defined in types.rs
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::Vertex;

    fn create_test_cube() -> Mesh {
        let mut mesh = Mesh::new();
        // Create a unit cube centered at origin
        let vertices = [
            (-5.0, -5.0, -5.0),
            (5.0, -5.0, -5.0),
            (5.0, 5.0, -5.0),
            (-5.0, 5.0, -5.0),
            (-5.0, -5.0, 5.0),
            (5.0, -5.0, 5.0),
            (5.0, 5.0, 5.0),
            (-5.0, 5.0, 5.0),
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
    fn test_dimensions() {
        let mesh = create_test_cube();
        let dims = dimensions(&mesh);

        assert!((dims.width - 10.0).abs() < 0.001);
        assert!((dims.depth - 10.0).abs() < 0.001);
        assert!((dims.height - 10.0).abs() < 0.001);
        assert!((dims.center.x).abs() < 0.001);
        assert!((dims.center.y).abs() < 0.001);
        assert!((dims.center.z).abs() < 0.001);
    }

    #[test]
    fn test_oriented_bounding_box() {
        let mesh = create_test_cube();
        let obb = oriented_bounding_box(&mesh);

        // Volume should be close to 1000 (10x10x10)
        assert!((obb.volume - 1000.0).abs() < 10.0);
    }

    #[test]
    fn test_cross_section() {
        let mesh = create_test_cube();
        // Cross-section at z=0 should give a 10x10 square
        let section = cross_section(&mesh, Point3::new(0.0, 0.0, 0.0), Vector3::z());

        // Area should be close to 100
        assert!((section.area - 100.0).abs() < 1.0);
    }

    #[test]
    fn test_cross_sections_multiple() {
        let mesh = create_test_cube();
        let sections = cross_sections(&mesh, Point3::new(0.0, 0.0, -5.0), Vector3::z(), 3, 5.0);

        assert_eq!(sections.len(), 3);
    }

    #[test]
    fn test_measure_distance() {
        let result = measure_distance(Point3::origin(), Point3::new(3.0, 4.0, 0.0));

        assert!((result.distance - 5.0).abs() < 0.001);
        assert!((result.dx - 3.0).abs() < 0.001);
        assert!((result.dy - 4.0).abs() < 0.001);
    }

    #[test]
    fn test_surface_area() {
        let mesh = create_test_cube();
        let area = mesh.surface_area();

        // 6 faces * 100 each = 600
        assert!((area - 600.0).abs() < 1.0);
    }

    #[test]
    fn test_volume() {
        let mesh = create_test_cube();
        let vol = mesh.volume();

        // 10^3 = 1000
        assert!((vol - 1000.0).abs() < 1.0);
    }

    #[test]
    fn test_closest_point() {
        let mesh = create_test_cube();
        let closest = closest_point_on_mesh(&mesh, Point3::new(10.0, 0.0, 0.0));

        assert!(closest.is_some());
        let p = closest.unwrap();
        // Should be on the +X face at (5, 0, 0)
        assert!((p.x - 5.0).abs() < 0.1);
    }

    #[test]
    fn test_mesh_methods() {
        let mesh = create_test_tetrahedron();

        let dims = mesh.dimensions();
        assert!(dims.width > 0.0);

        let area = mesh.surface_area();
        assert!(area > 0.0);

        let vol = mesh.volume();
        assert!(vol > 0.0);
    }

    #[test]
    fn test_circumference() {
        let mesh = create_test_cube();
        let circ = circumference_at_height(&mesh, 0.0);

        // Perimeter of 10x10 square = 40
        assert!((circ - 40.0).abs() < 1.0);
    }
}

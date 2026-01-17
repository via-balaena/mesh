//! Core mesh data types.

use nalgebra::{Point3, Vector3};

/// RGB color with 8-bit components.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct VertexColor {
    pub r: u8,
    pub g: u8,
    pub b: u8,
}

impl VertexColor {
    /// Create a new color from RGB components.
    #[inline]
    pub fn new(r: u8, g: u8, b: u8) -> Self {
        Self { r, g, b }
    }

    /// Create a color from floating point values in [0, 1] range.
    #[inline]
    pub fn from_float(r: f32, g: f32, b: f32) -> Self {
        Self {
            r: (r.clamp(0.0, 1.0) * 255.0) as u8,
            g: (g.clamp(0.0, 1.0) * 255.0) as u8,
            b: (b.clamp(0.0, 1.0) * 255.0) as u8,
        }
    }

    /// Convert to floating point values in [0, 1] range.
    #[inline]
    pub fn to_float(&self) -> (f32, f32, f32) {
        (
            self.r as f32 / 255.0,
            self.g as f32 / 255.0,
            self.b as f32 / 255.0,
        )
    }
}

/// A vertex in the mesh with optional computed attributes.
///
/// Coordinates are typically in millimeters but the library is unit-agnostic.
#[derive(Debug, Clone)]
pub struct Vertex {
    /// 3D position.
    pub position: Point3<f64>,

    /// Unit normal vector, computed from adjacent faces.
    pub normal: Option<Vector3<f64>>,

    /// Vertex color (RGB).
    pub color: Option<VertexColor>,

    /// Application-specific tag (e.g., zone ID, material ID).
    pub tag: Option<u32>,

    /// Offset distance for this vertex (used by mesh-shell for variable offset).
    /// Positive = outward expansion, negative = compression.
    pub offset: Option<f32>,
}

impl Vertex {
    /// Create a new vertex with only position set.
    #[inline]
    pub fn new(position: Point3<f64>) -> Self {
        Self {
            position,
            normal: None,
            color: None,
            tag: None,
            offset: None,
        }
    }

    /// Create a vertex from raw coordinates.
    #[inline]
    pub fn from_coords(x: f64, y: f64, z: f64) -> Self {
        Self::new(Point3::new(x, y, z))
    }

    /// Create a vertex with position and color.
    #[inline]
    pub fn with_color(position: Point3<f64>, color: VertexColor) -> Self {
        Self {
            position,
            normal: None,
            color: Some(color),
            tag: None,
            offset: None,
        }
    }
}

/// A triangle mesh with indexed vertices and faces.
#[derive(Debug, Clone)]
pub struct Mesh {
    /// Vertex data.
    pub vertices: Vec<Vertex>,

    /// Triangle faces as indices into the vertex array.
    /// Each face is [v0, v1, v2] with counter-clockwise winding.
    pub faces: Vec<[u32; 3]>,
}

impl Mesh {
    /// Create a new empty mesh.
    pub fn new() -> Self {
        Self {
            vertices: Vec::new(),
            faces: Vec::new(),
        }
    }

    /// Create a mesh with pre-allocated capacity.
    pub fn with_capacity(vertex_count: usize, face_count: usize) -> Self {
        Self {
            vertices: Vec::with_capacity(vertex_count),
            faces: Vec::with_capacity(face_count),
        }
    }

    /// Number of vertices in the mesh.
    #[inline]
    pub fn vertex_count(&self) -> usize {
        self.vertices.len()
    }

    /// Number of faces (triangles) in the mesh.
    #[inline]
    pub fn face_count(&self) -> usize {
        self.faces.len()
    }

    /// Check if mesh is empty (no vertices or faces).
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.vertices.is_empty() || self.faces.is_empty()
    }

    /// Compute the axis-aligned bounding box.
    /// Returns (min_corner, max_corner) or None if mesh is empty.
    pub fn bounds(&self) -> Option<(Point3<f64>, Point3<f64>)> {
        if self.vertices.is_empty() {
            return None;
        }

        let mut min = self.vertices[0].position;
        let mut max = self.vertices[0].position;

        for vertex in &self.vertices[1..] {
            let p = &vertex.position;
            min.x = min.x.min(p.x);
            min.y = min.y.min(p.y);
            min.z = min.z.min(p.z);
            max.x = max.x.max(p.x);
            max.y = max.y.max(p.y);
            max.z = max.z.max(p.z);
        }

        Some((min, max))
    }

    /// Iterate over triangles, yielding Triangle structs with actual vertex data.
    pub fn triangles(&self) -> impl Iterator<Item = Triangle> + '_ {
        self.faces.iter().map(|&[i0, i1, i2]| Triangle {
            v0: self.vertices[i0 as usize].position,
            v1: self.vertices[i1 as usize].position,
            v2: self.vertices[i2 as usize].position,
        })
    }

    /// Get a specific triangle by face index.
    pub fn triangle(&self, face_idx: usize) -> Option<Triangle> {
        self.faces.get(face_idx).map(|&[i0, i1, i2]| Triangle {
            v0: self.vertices[i0 as usize].position,
            v1: self.vertices[i1 as usize].position,
            v2: self.vertices[i2 as usize].position,
        })
    }

    /// Rotate mesh 90 degrees around X axis (Y becomes Z, Z becomes -Y).
    pub fn rotate_x_90(&mut self) {
        for vertex in &mut self.vertices {
            let old_y = vertex.position.y;
            let old_z = vertex.position.z;
            vertex.position.y = -old_z;
            vertex.position.z = old_y;

            if let Some(ref mut normal) = vertex.normal {
                let old_ny = normal.y;
                let old_nz = normal.z;
                normal.y = -old_nz;
                normal.z = old_ny;
            }
        }
    }

    /// Translate mesh so minimum Z is at zero.
    pub fn place_on_z_zero(&mut self) {
        if let Some((min, _)) = self.bounds() {
            let offset = -min.z;
            for vertex in &mut self.vertices {
                vertex.position.z += offset;
            }
        }
    }

    /// Translate mesh by the given vector.
    pub fn translate(&mut self, offset: Vector3<f64>) {
        for vertex in &mut self.vertices {
            vertex.position += offset;
        }
    }

    /// Scale mesh uniformly around the origin.
    pub fn scale(&mut self, factor: f64) {
        for vertex in &mut self.vertices {
            vertex.position.coords *= factor;
        }
    }

    /// Compute the signed volume of the mesh.
    ///
    /// Uses the divergence theorem: the signed volume is the sum of signed tetrahedra
    /// volumes formed by each face and the origin. For a closed mesh with outward-facing
    /// normals (CCW winding when viewed from outside), this returns a positive value.
    ///
    /// # Returns
    /// - Positive value: normals point outward (correct orientation)
    /// - Negative value: normals point inward (inside-out mesh)
    /// - Near-zero: mesh is not closed or has inconsistent winding
    ///
    /// # Note
    /// This calculation assumes the mesh is closed (watertight). For open meshes,
    /// the result is not meaningful as a volume measurement.
    pub fn signed_volume(&self) -> f64 {
        let mut volume = 0.0;

        for &[i0, i1, i2] in &self.faces {
            let v0 = &self.vertices[i0 as usize].position;
            let v1 = &self.vertices[i1 as usize].position;
            let v2 = &self.vertices[i2 as usize].position;

            // Signed volume of tetrahedron with origin = (v0 · (v1 × v2)) / 6
            // This is equivalent to the scalar triple product / 6
            let cross = Vector3::new(
                v1.y * v2.z - v1.z * v2.y,
                v1.z * v2.x - v1.x * v2.z,
                v1.x * v2.y - v1.y * v2.x,
            );
            volume += v0.x * cross.x + v0.y * cross.y + v0.z * cross.z;
        }

        volume / 6.0
    }

    /// Compute the absolute volume of the mesh.
    ///
    /// Returns the absolute value of `signed_volume()`. This gives the enclosed
    /// volume regardless of normal orientation.
    ///
    /// # Note
    /// This calculation assumes the mesh is closed (watertight). For open meshes,
    /// the result is not meaningful as a volume measurement.
    #[inline]
    pub fn volume(&self) -> f64 {
        self.signed_volume().abs()
    }

    /// Check if the mesh appears to be inside-out (inverted normals).
    ///
    /// A mesh is considered inside-out if its signed volume is negative,
    /// meaning the face normals point inward rather than outward.
    ///
    /// # Returns
    /// - `true` if signed volume is negative (inside-out)
    /// - `false` if signed volume is positive or zero
    ///
    /// # Note
    /// This is only meaningful for closed meshes. Open meshes or meshes
    /// with inconsistent winding may give unreliable results.
    #[inline]
    pub fn is_inside_out(&self) -> bool {
        self.signed_volume() < 0.0
    }

    /// Compute the total surface area of the mesh.
    ///
    /// Sums the area of all triangles in the mesh.
    pub fn surface_area(&self) -> f64 {
        self.triangles().map(|tri| tri.area()).sum()
    }
}

impl Default for Mesh {
    fn default() -> Self {
        Self::new()
    }
}

/// A triangle with concrete vertex positions.
///
/// Utility type for geometric calculations. Winding is counter-clockwise
/// when viewed from the front (normal points toward viewer).
#[derive(Debug, Clone, Copy)]
pub struct Triangle {
    pub v0: Point3<f64>,
    pub v1: Point3<f64>,
    pub v2: Point3<f64>,
}

impl Triangle {
    /// Create a new triangle from three points.
    #[inline]
    pub fn new(v0: Point3<f64>, v1: Point3<f64>, v2: Point3<f64>) -> Self {
        Self { v0, v1, v2 }
    }

    /// Compute the (unnormalized) face normal via cross product.
    /// The direction follows the right-hand rule with CCW winding.
    #[inline]
    pub fn normal_unnormalized(&self) -> Vector3<f64> {
        let e1 = self.v1 - self.v0;
        let e2 = self.v2 - self.v0;
        e1.cross(&e2)
    }

    /// Compute the unit face normal.
    /// Returns None for degenerate triangles (zero area).
    pub fn normal(&self) -> Option<Vector3<f64>> {
        let n = self.normal_unnormalized();
        let len_sq = n.norm_squared();
        if len_sq > f64::EPSILON {
            Some(n / len_sq.sqrt())
        } else {
            None
        }
    }

    /// Compute the area of the triangle.
    #[inline]
    pub fn area(&self) -> f64 {
        self.normal_unnormalized().norm() * 0.5
    }

    /// Compute the centroid (center of mass).
    #[inline]
    pub fn centroid(&self) -> Point3<f64> {
        Point3::new(
            (self.v0.x + self.v1.x + self.v2.x) / 3.0,
            (self.v0.y + self.v1.y + self.v2.y) / 3.0,
            (self.v0.z + self.v1.z + self.v2.z) / 3.0,
        )
    }

    /// Get the three edges as (start, end) pairs.
    pub fn edges(&self) -> [(Point3<f64>, Point3<f64>); 3] {
        [(self.v0, self.v1), (self.v1, self.v2), (self.v2, self.v0)]
    }

    /// Compute the lengths of the three edges.
    /// Returns [len01, len12, len20] where lenXY is the distance from vX to vY.
    #[inline]
    pub fn edge_lengths(&self) -> [f64; 3] {
        [
            (self.v1 - self.v0).norm(),
            (self.v2 - self.v1).norm(),
            (self.v0 - self.v2).norm(),
        ]
    }

    /// Get the length of the shortest edge.
    #[inline]
    pub fn min_edge_length(&self) -> f64 {
        let lengths = self.edge_lengths();
        lengths[0].min(lengths[1]).min(lengths[2])
    }

    /// Get the length of the longest edge.
    #[inline]
    pub fn max_edge_length(&self) -> f64 {
        let lengths = self.edge_lengths();
        lengths[0].max(lengths[1]).max(lengths[2])
    }

    /// Compute the aspect ratio of the triangle.
    ///
    /// Aspect ratio is defined as longest_edge / shortest_altitude.
    /// A well-shaped equilateral triangle has aspect ratio ≈ 1.15.
    /// Very thin/needle triangles have high aspect ratios (>10 is problematic).
    ///
    /// Returns `f64::INFINITY` for degenerate triangles (zero area).
    pub fn aspect_ratio(&self) -> f64 {
        let area = self.area();
        if area < f64::EPSILON {
            return f64::INFINITY;
        }

        let max_edge = self.max_edge_length();

        // Altitude = 2 * area / base
        // Shortest altitude corresponds to longest edge as base
        let shortest_altitude = 2.0 * area / max_edge;

        if shortest_altitude < f64::EPSILON {
            return f64::INFINITY;
        }

        max_edge / shortest_altitude
    }

    /// Check if triangle vertices are nearly collinear.
    ///
    /// Uses the cross product magnitude relative to edge lengths to detect
    /// triangles where all three vertices are nearly on a line.
    pub fn is_nearly_collinear(&self, epsilon: f64) -> bool {
        let e1 = self.v1 - self.v0;
        let e2 = self.v2 - self.v0;

        let cross_magnitude = e1.cross(&e2).norm();
        let edge_product = e1.norm() * e2.norm();

        if edge_product < f64::EPSILON {
            return true; // Degenerate edges
        }

        // sin(angle) = |cross| / (|e1| * |e2|)
        // For nearly collinear, sin(angle) ≈ 0
        cross_magnitude / edge_product < epsilon
    }

    /// Check if the triangle is degenerate (zero or near-zero area).
    pub fn is_degenerate(&self, epsilon: f64) -> bool {
        self.area() < epsilon
    }

    /// Check if the triangle is degenerate using multiple criteria.
    ///
    /// A triangle is considered degenerate if any of these conditions are met:
    /// - Area is below `area_threshold`
    /// - Aspect ratio exceeds `max_aspect_ratio`
    /// - Shortest edge is below `min_edge_length`
    ///
    /// # Arguments
    /// * `area_threshold` - Minimum acceptable area (e.g., 1e-9)
    /// * `max_aspect_ratio` - Maximum acceptable aspect ratio (e.g., 1000.0)
    /// * `min_edge_length` - Minimum acceptable edge length (e.g., 1e-9)
    pub fn is_degenerate_enhanced(
        &self,
        area_threshold: f64,
        max_aspect_ratio: f64,
        min_edge_length: f64,
    ) -> bool {
        // Check area
        if self.area() < area_threshold {
            return true;
        }

        // Check aspect ratio
        if self.aspect_ratio() > max_aspect_ratio {
            return true;
        }

        // Check minimum edge length
        if self.min_edge_length() < min_edge_length {
            return true;
        }

        false
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn approx_eq(a: f64, b: f64) -> bool {
        (a - b).abs() < 1e-10
    }

    #[test]
    fn test_vertex_creation() {
        let v = Vertex::from_coords(1.0, 2.0, 3.0);
        assert!(approx_eq(v.position.x, 1.0));
        assert!(approx_eq(v.position.y, 2.0));
        assert!(approx_eq(v.position.z, 3.0));
        assert!(v.normal.is_none());
        assert!(v.tag.is_none());
        assert!(v.offset.is_none());
    }

    #[test]
    fn test_triangle_normal() {
        let tri = Triangle::new(
            Point3::new(0.0, 0.0, 0.0),
            Point3::new(1.0, 0.0, 0.0),
            Point3::new(0.0, 1.0, 0.0),
        );

        let normal = tri.normal().expect("non-degenerate triangle");
        assert!(approx_eq(normal.x, 0.0));
        assert!(approx_eq(normal.y, 0.0));
        assert!(approx_eq(normal.z, 1.0));
    }

    #[test]
    fn test_triangle_area() {
        let tri = Triangle::new(
            Point3::new(0.0, 0.0, 0.0),
            Point3::new(1.0, 0.0, 0.0),
            Point3::new(0.0, 1.0, 0.0),
        );
        assert!(approx_eq(tri.area(), 0.5));
    }

    #[test]
    fn test_triangle_centroid() {
        let tri = Triangle::new(
            Point3::new(0.0, 0.0, 0.0),
            Point3::new(3.0, 0.0, 0.0),
            Point3::new(0.0, 3.0, 0.0),
        );
        let c = tri.centroid();
        assert!(approx_eq(c.x, 1.0));
        assert!(approx_eq(c.y, 1.0));
        assert!(approx_eq(c.z, 0.0));
    }

    #[test]
    fn test_degenerate_triangle_normal() {
        let tri = Triangle::new(
            Point3::new(0.0, 0.0, 0.0),
            Point3::new(1.0, 0.0, 0.0),
            Point3::new(2.0, 0.0, 0.0),
        );
        assert!(tri.normal().is_none());
    }

    #[test]
    fn test_mesh_bounds() {
        let mut mesh = Mesh::new();
        mesh.vertices.push(Vertex::from_coords(0.0, 0.0, 0.0));
        mesh.vertices.push(Vertex::from_coords(10.0, 5.0, 3.0));
        mesh.vertices.push(Vertex::from_coords(-2.0, 8.0, 1.0));

        let (min, max) = mesh.bounds().expect("non-empty mesh");
        assert!(approx_eq(min.x, -2.0));
        assert!(approx_eq(min.y, 0.0));
        assert!(approx_eq(min.z, 0.0));
        assert!(approx_eq(max.x, 10.0));
        assert!(approx_eq(max.y, 8.0));
        assert!(approx_eq(max.z, 3.0));
    }

    #[test]
    fn test_empty_mesh_bounds() {
        let mesh = Mesh::new();
        assert!(mesh.bounds().is_none());
    }

    #[test]
    fn test_mesh_is_empty() {
        let mesh = Mesh::new();
        assert!(mesh.is_empty());

        let mut mesh2 = Mesh::new();
        mesh2.vertices.push(Vertex::from_coords(0.0, 0.0, 0.0));
        assert!(mesh2.is_empty()); // no faces

        mesh2.faces.push([0, 0, 0]);
        assert!(!mesh2.is_empty());
    }

    #[test]
    fn test_triangle_edge_lengths() {
        // Right triangle with legs of length 3 and 4
        let tri = Triangle::new(
            Point3::new(0.0, 0.0, 0.0),
            Point3::new(3.0, 0.0, 0.0),
            Point3::new(0.0, 4.0, 0.0),
        );
        let lengths = tri.edge_lengths();
        assert!(approx_eq(lengths[0], 3.0)); // v0 -> v1
        assert!(approx_eq(lengths[1], 5.0)); // v1 -> v2 (hypotenuse)
        assert!(approx_eq(lengths[2], 4.0)); // v2 -> v0
    }

    #[test]
    fn test_triangle_min_max_edge_length() {
        // Right triangle with legs of length 3 and 4, hypotenuse 5
        let tri = Triangle::new(
            Point3::new(0.0, 0.0, 0.0),
            Point3::new(3.0, 0.0, 0.0),
            Point3::new(0.0, 4.0, 0.0),
        );
        assert!(approx_eq(tri.min_edge_length(), 3.0));
        assert!(approx_eq(tri.max_edge_length(), 5.0));
    }

    #[test]
    fn test_triangle_aspect_ratio_equilateral() {
        // Equilateral triangle with side length 2
        let sqrt3 = 3.0_f64.sqrt();
        let tri = Triangle::new(
            Point3::new(0.0, 0.0, 0.0),
            Point3::new(2.0, 0.0, 0.0),
            Point3::new(1.0, sqrt3, 0.0),
        );
        // For equilateral: aspect ratio = edge / altitude = 2 / sqrt(3) ≈ 1.1547
        let ar = tri.aspect_ratio();
        assert!(
            ar > 1.1 && ar < 1.2,
            "Equilateral aspect ratio should be ~1.15, got {}",
            ar
        );
    }

    #[test]
    fn test_triangle_aspect_ratio_thin() {
        // Very thin triangle (needle-like)
        let tri = Triangle::new(
            Point3::new(0.0, 0.0, 0.0),
            Point3::new(100.0, 0.0, 0.0),
            Point3::new(50.0, 0.1, 0.0),
        );
        let ar = tri.aspect_ratio();
        // Should be very high for thin triangles
        assert!(
            ar > 100.0,
            "Thin triangle should have high aspect ratio, got {}",
            ar
        );
    }

    #[test]
    fn test_triangle_aspect_ratio_degenerate() {
        // Collinear points (degenerate)
        let tri = Triangle::new(
            Point3::new(0.0, 0.0, 0.0),
            Point3::new(1.0, 0.0, 0.0),
            Point3::new(2.0, 0.0, 0.0),
        );
        assert!(tri.aspect_ratio().is_infinite());
    }

    #[test]
    fn test_triangle_is_nearly_collinear() {
        // Collinear points
        let tri_collinear = Triangle::new(
            Point3::new(0.0, 0.0, 0.0),
            Point3::new(1.0, 0.0, 0.0),
            Point3::new(2.0, 0.0, 0.0),
        );
        assert!(tri_collinear.is_nearly_collinear(0.01));

        // Nearly collinear (small deviation)
        let tri_nearly = Triangle::new(
            Point3::new(0.0, 0.0, 0.0),
            Point3::new(100.0, 0.0, 0.0),
            Point3::new(50.0, 0.001, 0.0),
        );
        assert!(tri_nearly.is_nearly_collinear(0.001));

        // Not collinear - well-formed triangle
        let tri_good = Triangle::new(
            Point3::new(0.0, 0.0, 0.0),
            Point3::new(1.0, 0.0, 0.0),
            Point3::new(0.5, 1.0, 0.0),
        );
        assert!(!tri_good.is_nearly_collinear(0.01));
    }

    #[test]
    fn test_triangle_is_degenerate_basic() {
        // Degenerate by area
        let tri_zero_area = Triangle::new(
            Point3::new(0.0, 0.0, 0.0),
            Point3::new(1.0, 0.0, 0.0),
            Point3::new(2.0, 0.0, 0.0),
        );
        assert!(tri_zero_area.is_degenerate(1e-9));

        // Not degenerate
        let tri_good = Triangle::new(
            Point3::new(0.0, 0.0, 0.0),
            Point3::new(1.0, 0.0, 0.0),
            Point3::new(0.0, 1.0, 0.0),
        );
        assert!(!tri_good.is_degenerate(1e-9));
    }

    #[test]
    fn test_triangle_is_degenerate_enhanced() {
        // Good triangle - should pass all checks
        let tri_good = Triangle::new(
            Point3::new(0.0, 0.0, 0.0),
            Point3::new(1.0, 0.0, 0.0),
            Point3::new(0.5, 1.0, 0.0),
        );
        assert!(!tri_good.is_degenerate_enhanced(1e-9, 1000.0, 1e-9));

        // Degenerate by area threshold
        let tri_tiny = Triangle::new(
            Point3::new(0.0, 0.0, 0.0),
            Point3::new(1e-6, 0.0, 0.0),
            Point3::new(0.0, 1e-6, 0.0),
        );
        assert!(tri_tiny.is_degenerate_enhanced(1e-9, 1000.0, 1e-9));

        // Degenerate by aspect ratio
        let tri_thin = Triangle::new(
            Point3::new(0.0, 0.0, 0.0),
            Point3::new(100.0, 0.0, 0.0),
            Point3::new(50.0, 0.01, 0.0),
        );
        assert!(tri_thin.is_degenerate_enhanced(1e-9, 100.0, 1e-9));

        // Degenerate by minimum edge length
        let tri_short_edge = Triangle::new(
            Point3::new(0.0, 0.0, 0.0),
            Point3::new(1e-6, 0.0, 0.0),
            Point3::new(0.5, 1.0, 0.0),
        );
        assert!(tri_short_edge.is_degenerate_enhanced(1e-12, 1000.0, 1e-5));
    }

    #[test]
    fn test_triangle_edges() {
        let tri = Triangle::new(
            Point3::new(0.0, 0.0, 0.0),
            Point3::new(1.0, 0.0, 0.0),
            Point3::new(0.0, 1.0, 0.0),
        );
        let edges = tri.edges();

        // Check edge 0: v0 -> v1
        assert!(approx_eq(edges[0].0.x, 0.0) && approx_eq(edges[0].1.x, 1.0));
        // Check edge 1: v1 -> v2
        assert!(approx_eq(edges[1].0.x, 1.0) && approx_eq(edges[1].1.y, 1.0));
        // Check edge 2: v2 -> v0
        assert!(approx_eq(edges[2].0.y, 1.0) && approx_eq(edges[2].1.x, 0.0));
    }

    /// Create a unit cube mesh with outward-facing normals (CCW winding from outside).
    /// Vertices at (0,0,0) to (1,1,1).
    fn make_unit_cube() -> Mesh {
        let mut mesh = Mesh::new();

        // 8 vertices of the cube
        mesh.vertices.push(Vertex::from_coords(0.0, 0.0, 0.0)); // 0
        mesh.vertices.push(Vertex::from_coords(1.0, 0.0, 0.0)); // 1
        mesh.vertices.push(Vertex::from_coords(1.0, 1.0, 0.0)); // 2
        mesh.vertices.push(Vertex::from_coords(0.0, 1.0, 0.0)); // 3
        mesh.vertices.push(Vertex::from_coords(0.0, 0.0, 1.0)); // 4
        mesh.vertices.push(Vertex::from_coords(1.0, 0.0, 1.0)); // 5
        mesh.vertices.push(Vertex::from_coords(1.0, 1.0, 1.0)); // 6
        mesh.vertices.push(Vertex::from_coords(0.0, 1.0, 1.0)); // 7

        // 12 triangles (2 per face), CCW winding when viewed from outside

        // Bottom face (z=0) - normal points -Z, CCW from below
        mesh.faces.push([0, 2, 1]);
        mesh.faces.push([0, 3, 2]);

        // Top face (z=1) - normal points +Z, CCW from above
        mesh.faces.push([4, 5, 6]);
        mesh.faces.push([4, 6, 7]);

        // Front face (y=0) - normal points -Y
        mesh.faces.push([0, 1, 5]);
        mesh.faces.push([0, 5, 4]);

        // Back face (y=1) - normal points +Y
        mesh.faces.push([3, 7, 6]);
        mesh.faces.push([3, 6, 2]);

        // Left face (x=0) - normal points -X
        mesh.faces.push([0, 4, 7]);
        mesh.faces.push([0, 7, 3]);

        // Right face (x=1) - normal points +X
        mesh.faces.push([1, 2, 6]);
        mesh.faces.push([1, 6, 5]);

        mesh
    }

    #[test]
    fn test_signed_volume_unit_cube() {
        let mesh = make_unit_cube();
        let vol = mesh.signed_volume();
        // Unit cube volume = 1.0
        assert!(
            (vol - 1.0).abs() < 1e-10,
            "Unit cube signed volume should be 1.0, got {}",
            vol
        );
    }

    #[test]
    fn test_volume_unit_cube() {
        let mesh = make_unit_cube();
        let vol = mesh.volume();
        assert!(
            (vol - 1.0).abs() < 1e-10,
            "Unit cube volume should be 1.0, got {}",
            vol
        );
    }

    #[test]
    fn test_signed_volume_scaled_cube() {
        let mut mesh = make_unit_cube();
        mesh.scale(2.0); // 2x2x2 cube
        let vol = mesh.signed_volume();
        // Volume = 8.0
        assert!(
            (vol - 8.0).abs() < 1e-10,
            "2x2x2 cube signed volume should be 8.0, got {}",
            vol
        );
    }

    #[test]
    fn test_signed_volume_inverted_cube() {
        let mut mesh = make_unit_cube();
        // Invert all faces by swapping indices
        for face in &mut mesh.faces {
            face.swap(1, 2);
        }
        let vol = mesh.signed_volume();
        // Should be negative for inside-out mesh
        assert!(
            (vol + 1.0).abs() < 1e-10,
            "Inverted cube signed volume should be -1.0, got {}",
            vol
        );
    }

    #[test]
    fn test_is_inside_out_normal_cube() {
        let mesh = make_unit_cube();
        assert!(
            !mesh.is_inside_out(),
            "Normal cube should not be inside-out"
        );
    }

    #[test]
    fn test_is_inside_out_inverted_cube() {
        let mut mesh = make_unit_cube();
        // Invert all faces
        for face in &mut mesh.faces {
            face.swap(1, 2);
        }
        assert!(mesh.is_inside_out(), "Inverted cube should be inside-out");
    }

    #[test]
    fn test_signed_volume_tetrahedron() {
        // Regular tetrahedron with one vertex at origin
        let mut mesh = Mesh::new();
        mesh.vertices.push(Vertex::from_coords(0.0, 0.0, 0.0)); // 0
        mesh.vertices.push(Vertex::from_coords(1.0, 0.0, 0.0)); // 1
        mesh.vertices.push(Vertex::from_coords(0.5, 0.866025, 0.0)); // 2 (approx sqrt(3)/2)
        mesh.vertices
            .push(Vertex::from_coords(0.5, 0.288675, 0.816497)); // 3 (apex)

        // Faces with outward normals (CCW from outside)
        mesh.faces.push([0, 2, 1]); // Bottom face
        mesh.faces.push([0, 1, 3]); // Front face
        mesh.faces.push([1, 2, 3]); // Right face
        mesh.faces.push([2, 0, 3]); // Left face

        let vol = mesh.signed_volume();
        // Tetrahedron volume = (edge^3) / (6 * sqrt(2)) ≈ 0.1178 for edge=1
        // With our vertices, the expected volume is approximately 0.1178
        assert!(
            vol > 0.1 && vol < 0.15,
            "Tetrahedron volume should be ~0.1178, got {}",
            vol
        );
    }

    #[test]
    fn test_signed_volume_translated_cube() {
        let mut mesh = make_unit_cube();
        // Translate away from origin
        mesh.translate(Vector3::new(10.0, 20.0, 30.0));
        let vol = mesh.signed_volume();
        // Volume should still be 1.0 (translation invariant)
        assert!(
            (vol - 1.0).abs() < 1e-10,
            "Translated cube volume should still be 1.0, got {}",
            vol
        );
    }

    #[test]
    fn test_signed_volume_empty_mesh() {
        let mesh = Mesh::new();
        let vol = mesh.signed_volume();
        assert!(
            vol.abs() < 1e-10,
            "Empty mesh volume should be 0, got {}",
            vol
        );
    }

    #[test]
    fn test_surface_area_unit_cube() {
        let mesh = make_unit_cube();
        let area = mesh.surface_area();
        // Unit cube surface area = 6 faces * 1.0 = 6.0
        assert!(
            (area - 6.0).abs() < 1e-10,
            "Unit cube surface area should be 6.0, got {}",
            area
        );
    }

    #[test]
    fn test_surface_area_single_triangle() {
        let mut mesh = Mesh::new();
        mesh.vertices.push(Vertex::from_coords(0.0, 0.0, 0.0));
        mesh.vertices.push(Vertex::from_coords(1.0, 0.0, 0.0));
        mesh.vertices.push(Vertex::from_coords(0.0, 1.0, 0.0));
        mesh.faces.push([0, 1, 2]);

        let area = mesh.surface_area();
        // Right triangle with legs 1 and 1, area = 0.5
        assert!(
            (area - 0.5).abs() < 1e-10,
            "Triangle area should be 0.5, got {}",
            area
        );
    }

    #[test]
    fn test_surface_area_empty_mesh() {
        let mesh = Mesh::new();
        let area = mesh.surface_area();
        assert!(
            area.abs() < 1e-10,
            "Empty mesh area should be 0, got {}",
            area
        );
    }
}

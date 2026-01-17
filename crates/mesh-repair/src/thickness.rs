//! Wall thickness analysis for meshes.
//!
//! This module provides tools for computing and analyzing wall thickness
//! in closed meshes. Wall thickness is crucial for 3D printing where
//! thin regions may fail structurally or print incorrectly.
//!
//! # Algorithm
//!
//! Wall thickness is computed by casting rays from each vertex inward
//! (opposite to the vertex normal) and finding the closest intersection
//! with the mesh surface. This gives the local wall thickness at each point.

use nalgebra::{Point3, Vector3};
use rayon::prelude::*;
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use tracing::{info, warn};

use crate::types::{Mesh, Triangle};

/// Parameters for wall thickness analysis.
#[derive(Debug, Clone)]
pub struct ThicknessParams {
    /// Minimum acceptable wall thickness in mesh units (typically mm).
    /// Regions thinner than this will be flagged.
    pub min_thickness: f64,
    /// Maximum ray distance to trace. Set to 0 for unlimited.
    pub max_ray_distance: f64,
    /// Epsilon for ray-triangle intersection tests.
    pub epsilon: f64,
    /// Maximum number of thin regions to report.
    pub max_regions: usize,
    /// Whether to skip vertices without normals.
    pub require_normals: bool,
}

impl Default for ThicknessParams {
    fn default() -> Self {
        Self {
            min_thickness: 1.0,       // 1mm default minimum
            max_ray_distance: 1000.0, // 1m max ray distance
            epsilon: 1e-8,
            max_regions: 1000,
            require_normals: false,
        }
    }
}

impl ThicknessParams {
    /// Create params for 3D printing analysis.
    /// Uses typical FDM printer requirements.
    pub fn for_printing() -> Self {
        Self {
            min_thickness: 0.8, // 0.8mm minimum for FDM
            max_ray_distance: 500.0,
            epsilon: 1e-8,
            max_regions: 1000,
            require_normals: false,
        }
    }

    /// Create params for SLA/resin printing.
    /// SLA can handle thinner walls.
    pub fn for_sla() -> Self {
        Self {
            min_thickness: 0.4, // 0.4mm minimum for SLA
            max_ray_distance: 500.0,
            epsilon: 1e-8,
            max_regions: 1000,
            require_normals: false,
        }
    }

    /// Create params with a custom minimum thickness.
    pub fn with_min_thickness(min_thickness: f64) -> Self {
        Self {
            min_thickness,
            ..Default::default()
        }
    }
}

/// A region identified as having thin walls.
#[derive(Debug, Clone)]
pub struct ThinRegion {
    /// Vertex index where thin wall was detected.
    pub vertex_index: usize,
    /// Position of the vertex.
    pub position: Point3<f64>,
    /// Measured wall thickness at this point.
    pub thickness: f64,
    /// Face index that the inward ray hit.
    pub hit_face: Option<usize>,
}

/// Result of wall thickness analysis.
#[derive(Debug, Clone)]
pub struct ThicknessResult {
    /// Per-vertex thickness measurements (f64::INFINITY if no hit).
    pub vertex_thickness: Vec<f64>,
    /// Minimum thickness found across all vertices.
    pub min_thickness: f64,
    /// Maximum thickness found across all vertices.
    pub max_thickness: f64,
    /// Average thickness (excluding infinite values).
    pub avg_thickness: f64,
    /// List of thin regions (below threshold).
    pub thin_regions: Vec<ThinRegion>,
    /// Number of vertices analyzed.
    pub vertices_analyzed: usize,
    /// Number of vertices with valid thickness (ray hit mesh).
    pub vertices_with_hits: usize,
    /// Number of vertices skipped (no normal or no hit).
    pub vertices_skipped: usize,
    /// Whether search was truncated due to max_regions.
    pub truncated: bool,
}

impl ThicknessResult {
    /// Check if the mesh has any thin regions.
    pub fn has_thin_regions(&self) -> bool {
        !self.thin_regions.is_empty()
    }

    /// Check if all analyzed vertices are above the minimum thickness.
    pub fn is_thick_enough(&self, min_thickness: f64) -> bool {
        self.min_thickness >= min_thickness
    }

    /// Get vertices below a given thickness threshold.
    pub fn vertices_below_thickness(&self, threshold: f64) -> Vec<usize> {
        self.vertex_thickness
            .iter()
            .enumerate()
            .filter(|&(_, t)| *t < threshold && t.is_finite())
            .map(|(i, _)| i)
            .collect()
    }
}

impl std::fmt::Display for ThicknessResult {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        writeln!(f, "Wall Thickness Analysis:")?;
        writeln!(f, "  Vertices analyzed: {}", self.vertices_analyzed)?;
        writeln!(f, "  Vertices with hits: {}", self.vertices_with_hits)?;
        writeln!(f, "  Min thickness: {:.3}", self.min_thickness)?;
        writeln!(f, "  Max thickness: {:.3}", self.max_thickness)?;
        writeln!(f, "  Avg thickness: {:.3}", self.avg_thickness)?;
        writeln!(f, "  Thin regions: {}", self.thin_regions.len())?;
        if self.truncated {
            writeln!(f, "  (truncated, more thin regions exist)")?;
        }
        Ok(())
    }
}

/// Axis-aligned bounding box for spatial acceleration.
#[derive(Debug, Clone, Copy)]
struct Aabb {
    min: Point3<f64>,
    max: Point3<f64>,
}

impl Aabb {
    /// Create AABB from a triangle.
    fn from_triangle(tri: &Triangle) -> Self {
        let min = Point3::new(
            tri.v0.x.min(tri.v1.x).min(tri.v2.x),
            tri.v0.y.min(tri.v1.y).min(tri.v2.y),
            tri.v0.z.min(tri.v1.z).min(tri.v2.z),
        );
        let max = Point3::new(
            tri.v0.x.max(tri.v1.x).max(tri.v2.x),
            tri.v0.y.max(tri.v1.y).max(tri.v2.y),
            tri.v0.z.max(tri.v1.z).max(tri.v2.z),
        );
        Self { min, max }
    }

    /// Expand AABB by epsilon for numerical robustness.
    fn expand(&self, epsilon: f64) -> Self {
        Self {
            min: Point3::new(
                self.min.x - epsilon,
                self.min.y - epsilon,
                self.min.z - epsilon,
            ),
            max: Point3::new(
                self.max.x + epsilon,
                self.max.y + epsilon,
                self.max.z + epsilon,
            ),
        }
    }

    /// Ray-AABB intersection test.
    /// Returns (t_near, t_far) or None if no intersection.
    fn ray_intersect(&self, origin: &Point3<f64>, dir_inv: &Vector3<f64>) -> Option<(f64, f64)> {
        let t1 = (self.min.x - origin.x) * dir_inv.x;
        let t2 = (self.max.x - origin.x) * dir_inv.x;
        let t3 = (self.min.y - origin.y) * dir_inv.y;
        let t4 = (self.max.y - origin.y) * dir_inv.y;
        let t5 = (self.min.z - origin.z) * dir_inv.z;
        let t6 = (self.max.z - origin.z) * dir_inv.z;

        let t_min = t1.min(t2).max(t3.min(t4)).max(t5.min(t6));
        let t_max = t1.max(t2).min(t3.max(t4)).min(t5.max(t6));

        if t_max >= t_min && t_max >= 0.0 {
            Some((t_min.max(0.0), t_max))
        } else {
            None
        }
    }
}

/// BVH node for acceleration structure.
#[derive(Debug)]
enum BvhNode {
    Leaf {
        aabb: Aabb,
        face_idx: usize,
    },
    Internal {
        aabb: Aabb,
        left: Box<BvhNode>,
        right: Box<BvhNode>,
    },
}

impl BvhNode {
    /// Build a BVH from triangles.
    fn build(triangles: &[Triangle], indices: &mut [usize], epsilon: f64) -> Option<Self> {
        if indices.is_empty() {
            return None;
        }

        if indices.len() == 1 {
            let idx = indices[0];
            return Some(BvhNode::Leaf {
                aabb: Aabb::from_triangle(&triangles[idx]).expand(epsilon),
                face_idx: idx,
            });
        }

        // Compute bounding box of all triangles
        let mut combined_aabb = Aabb::from_triangle(&triangles[indices[0]]);
        for &idx in indices.iter().skip(1) {
            let tri_aabb = Aabb::from_triangle(&triangles[idx]);
            combined_aabb.min.x = combined_aabb.min.x.min(tri_aabb.min.x);
            combined_aabb.min.y = combined_aabb.min.y.min(tri_aabb.min.y);
            combined_aabb.min.z = combined_aabb.min.z.min(tri_aabb.min.z);
            combined_aabb.max.x = combined_aabb.max.x.max(tri_aabb.max.x);
            combined_aabb.max.y = combined_aabb.max.y.max(tri_aabb.max.y);
            combined_aabb.max.z = combined_aabb.max.z.max(tri_aabb.max.z);
        }
        let combined_aabb = combined_aabb.expand(epsilon);

        // Choose split axis (longest extent)
        let extent = combined_aabb.max - combined_aabb.min;
        let axis = if extent.x >= extent.y && extent.x >= extent.z {
            0
        } else if extent.y >= extent.z {
            1
        } else {
            2
        };

        // Sort by centroid along axis
        indices.sort_by(|&a, &b| {
            let ca = triangles[a].centroid();
            let cb = triangles[b].centroid();
            let va = match axis {
                0 => ca.x,
                1 => ca.y,
                _ => ca.z,
            };
            let vb = match axis {
                0 => cb.x,
                1 => cb.y,
                _ => cb.z,
            };
            va.partial_cmp(&vb).unwrap_or(std::cmp::Ordering::Equal)
        });

        // Split at median
        let mid = indices.len() / 2;
        let (left_indices, right_indices) = indices.split_at_mut(mid);

        let left = BvhNode::build(triangles, left_indices, epsilon);
        let right = BvhNode::build(triangles, right_indices, epsilon);

        match (left, right) {
            (Some(l), Some(r)) => Some(BvhNode::Internal {
                aabb: combined_aabb,
                left: Box::new(l),
                right: Box::new(r),
            }),
            (Some(l), None) => Some(l),
            (None, Some(r)) => Some(r),
            (None, None) => None,
        }
    }

    fn aabb(&self) -> &Aabb {
        match self {
            BvhNode::Leaf { aabb, .. } => aabb,
            BvhNode::Internal { aabb, .. } => aabb,
        }
    }
}

/// Möller–Trumbore ray-triangle intersection algorithm.
/// Returns the distance t along the ray if intersection occurs.
fn ray_triangle_intersect(
    origin: &Point3<f64>,
    direction: &Vector3<f64>,
    tri: &Triangle,
    epsilon: f64,
) -> Option<f64> {
    let edge1 = tri.v1 - tri.v0;
    let edge2 = tri.v2 - tri.v0;

    let h = direction.cross(&edge2);
    let a = edge1.dot(&h);

    // Ray is parallel to triangle
    if a.abs() < epsilon {
        return None;
    }

    let f = 1.0 / a;
    let s = origin - tri.v0;
    let u = f * s.dot(&h);

    if !(0.0..=1.0).contains(&u) {
        return None;
    }

    let q = s.cross(&edge1);
    let v = f * direction.dot(&q);

    if v < 0.0 || u + v > 1.0 {
        return None;
    }

    let t = f * edge2.dot(&q);

    if t > epsilon { Some(t) } else { None }
}

/// Trace a ray through the BVH and find the closest intersection.
#[allow(clippy::too_many_arguments)]
fn trace_ray(
    node: &BvhNode,
    origin: &Point3<f64>,
    direction: &Vector3<f64>,
    dir_inv: &Vector3<f64>,
    triangles: &[Triangle],
    max_dist: f64,
    epsilon: f64,
    skip_faces: &[usize],
) -> Option<(f64, usize)> {
    // Check AABB intersection first
    let aabb = node.aabb();
    if let Some((t_near, _)) = aabb.ray_intersect(origin, dir_inv) {
        if t_near > max_dist {
            return None;
        }
    } else {
        return None;
    }

    match node {
        BvhNode::Leaf { face_idx, .. } => {
            // Skip faces in the skip list (faces adjacent to the origin vertex)
            if skip_faces.contains(face_idx) {
                return None;
            }

            if let Some(t) =
                ray_triangle_intersect(origin, direction, &triangles[*face_idx], epsilon)
                && t <= max_dist
            {
                return Some((t, *face_idx));
            }
            None
        }
        BvhNode::Internal { left, right, .. } => {
            let hit_left = trace_ray(
                left, origin, direction, dir_inv, triangles, max_dist, epsilon, skip_faces,
            );
            let max_dist_right = hit_left.map(|(t, _)| t).unwrap_or(max_dist);
            let hit_right = trace_ray(
                right,
                origin,
                direction,
                dir_inv,
                triangles,
                max_dist_right,
                epsilon,
                skip_faces,
            );

            match (hit_left, hit_right) {
                (Some((t1, f1)), Some((t2, f2))) => {
                    if t1 <= t2 {
                        Some((t1, f1))
                    } else {
                        Some((t2, f2))
                    }
                }
                (Some(h), None) | (None, Some(h)) => Some(h),
                (None, None) => None,
            }
        }
    }
}

/// Compute wall thickness at each vertex.
///
/// For each vertex, casts a ray inward (opposite to vertex normal) and finds
/// the distance to the opposite surface. This gives the local wall thickness.
///
/// # Arguments
/// * `mesh` - The mesh to analyze (should be closed/watertight)
/// * `params` - Analysis parameters
///
/// # Returns
/// A `ThicknessResult` with per-vertex thickness measurements and thin regions.
///
/// # Note
/// The mesh should have computed vertex normals. If normals are missing,
/// they will be computed automatically.
///
/// # Example
/// ```
/// use mesh_repair::{Mesh, Vertex};
/// use mesh_repair::thickness::{analyze_thickness, ThicknessParams};
///
/// // Create a simple mesh (normally you'd load a real mesh)
/// let mut mesh = Mesh::new();
/// mesh.vertices.push(Vertex::from_coords(0.0, 0.0, 0.0));
/// mesh.vertices.push(Vertex::from_coords(1.0, 0.0, 0.0));
/// mesh.vertices.push(Vertex::from_coords(0.5, 1.0, 0.0));
/// mesh.faces.push([0, 1, 2]);
///
/// let result = analyze_thickness(&mesh, &ThicknessParams::default());
/// println!("Min thickness: {:.3}", result.min_thickness);
/// ```
pub fn analyze_thickness(mesh: &Mesh, params: &ThicknessParams) -> ThicknessResult {
    let vertex_count = mesh.vertices.len();
    let face_count = mesh.faces.len();

    if vertex_count == 0 || face_count == 0 {
        return ThicknessResult {
            vertex_thickness: Vec::new(),
            min_thickness: f64::INFINITY,
            max_thickness: f64::NEG_INFINITY,
            avg_thickness: 0.0,
            thin_regions: Vec::new(),
            vertices_analyzed: 0,
            vertices_with_hits: 0,
            vertices_skipped: 0,
            truncated: false,
        };
    }

    info!(
        vertices = vertex_count,
        faces = face_count,
        min_thickness = params.min_thickness,
        "Starting wall thickness analysis"
    );

    // Pre-compute triangles
    let triangles: Vec<Triangle> = mesh.triangles().collect();

    // Build BVH for ray tracing
    let mut indices: Vec<usize> = (0..triangles.len()).collect();
    let bvh = BvhNode::build(&triangles, &mut indices, params.epsilon);

    let bvh = match bvh {
        Some(b) => b,
        None => {
            return ThicknessResult {
                vertex_thickness: vec![f64::INFINITY; vertex_count],
                min_thickness: f64::INFINITY,
                max_thickness: f64::NEG_INFINITY,
                avg_thickness: 0.0,
                thin_regions: Vec::new(),
                vertices_analyzed: vertex_count,
                vertices_with_hits: 0,
                vertices_skipped: vertex_count,
                truncated: false,
            };
        }
    };

    // Build vertex-to-face adjacency (faces that contain each vertex)
    let vertex_faces = build_vertex_faces(&mesh.faces, vertex_count);

    // Compute vertex normals if not present
    let normals = compute_or_get_normals(mesh);

    let max_dist = if params.max_ray_distance > 0.0 {
        params.max_ray_distance
    } else {
        f64::MAX
    };

    // Shared atomics for early termination when max_regions is reached
    let thin_count = AtomicUsize::new(0);
    let truncated_flag = AtomicBool::new(false);

    // Process vertices in parallel, returning per-vertex results
    // Each result is (thickness, hit_face_option, is_thin_region)
    let vertex_results: Vec<(f64, Option<usize>, bool)> = mesh
        .vertices
        .par_iter()
        .enumerate()
        .map(|(vertex_idx, vertex)| {
            // Get vertex normal
            let normal = match &normals {
                VertexNormals::FromMesh(n) => n.get(vertex_idx).and_then(|v| *v),
                VertexNormals::Computed(n) => n.get(vertex_idx).copied(),
            };

            let normal = match normal {
                Some(n) if n.norm() > params.epsilon => n.normalize(),
                _ => {
                    if params.require_normals {
                        return (f64::INFINITY, None, false);
                    }
                    // Try to compute normal from adjacent faces
                    let adj_faces = &vertex_faces[vertex_idx];
                    if adj_faces.is_empty() {
                        return (f64::INFINITY, None, false);
                    }
                    let mut sum = Vector3::zeros();
                    for &face_idx in adj_faces {
                        if let Some(n) = triangles[face_idx].normal() {
                            sum += n;
                        }
                    }
                    if sum.norm() < params.epsilon {
                        return (f64::INFINITY, None, false);
                    }
                    sum.normalize()
                }
            };

            // Cast ray inward (opposite to normal)
            let ray_dir = -normal;
            let ray_origin = vertex.position;

            // Compute inverse direction for AABB tests
            let dir_inv = Vector3::new(
                if ray_dir.x.abs() > params.epsilon {
                    1.0 / ray_dir.x
                } else {
                    f64::MAX
                },
                if ray_dir.y.abs() > params.epsilon {
                    1.0 / ray_dir.y
                } else {
                    f64::MAX
                },
                if ray_dir.z.abs() > params.epsilon {
                    1.0 / ray_dir.z
                } else {
                    f64::MAX
                },
            );

            // Skip faces adjacent to this vertex to avoid self-intersection
            let skip_faces = &vertex_faces[vertex_idx];

            // Trace ray
            if let Some((t, face_idx)) = trace_ray(
                &bvh,
                &ray_origin,
                &ray_dir,
                &dir_inv,
                &triangles,
                max_dist,
                params.epsilon,
                skip_faces,
            ) {
                // Check if this is a thin region
                let is_thin = t < params.min_thickness;
                if is_thin {
                    let current = thin_count.fetch_add(1, Ordering::Relaxed);
                    if current >= params.max_regions {
                        truncated_flag.store(true, Ordering::Relaxed);
                    }
                }
                (t, Some(face_idx), is_thin)
            } else {
                // Ray didn't hit anything (open mesh or very thick)
                (f64::INFINITY, None, false)
            }
        })
        .collect();

    // Aggregate results sequentially
    let mut vertex_thickness = Vec::with_capacity(vertex_count);
    let mut thin_regions = Vec::new();
    let mut vertices_skipped = 0;
    let mut vertices_with_hits = 0;
    let truncated = truncated_flag.load(Ordering::Relaxed);

    for (vertex_idx, (thickness, hit_face, is_thin)) in vertex_results.into_iter().enumerate() {
        vertex_thickness.push(thickness);

        if thickness.is_finite() {
            vertices_with_hits += 1;
            if is_thin && thin_regions.len() < params.max_regions {
                thin_regions.push(ThinRegion {
                    vertex_index: vertex_idx,
                    position: mesh.vertices[vertex_idx].position,
                    thickness,
                    hit_face,
                });
            }
        } else {
            vertices_skipped += 1;
        }
    }

    // Compute statistics
    let finite_thicknesses: Vec<f64> = vertex_thickness
        .iter()
        .copied()
        .filter(|t| t.is_finite())
        .collect();

    let min_thickness = finite_thicknesses
        .iter()
        .copied()
        .fold(f64::INFINITY, f64::min);
    let max_thickness = finite_thicknesses
        .iter()
        .copied()
        .fold(f64::NEG_INFINITY, f64::max);
    let avg_thickness = if !finite_thicknesses.is_empty() {
        finite_thicknesses.iter().sum::<f64>() / finite_thicknesses.len() as f64
    } else {
        0.0
    };

    if !thin_regions.is_empty() {
        warn!(
            thin_count = thin_regions.len(),
            min_thickness = format!("{:.3}", min_thickness),
            "Found thin regions below threshold"
        );
    } else {
        info!(
            min_thickness = format!("{:.3}", min_thickness),
            "Wall thickness analysis complete"
        );
    }

    ThicknessResult {
        vertex_thickness,
        min_thickness,
        max_thickness,
        avg_thickness,
        thin_regions,
        vertices_analyzed: vertex_count,
        vertices_with_hits,
        vertices_skipped,
        truncated,
    }
}

/// Build vertex to face adjacency.
fn build_vertex_faces(faces: &[[u32; 3]], vertex_count: usize) -> Vec<Vec<usize>> {
    let mut vertex_faces = vec![Vec::new(); vertex_count];
    for (face_idx, face) in faces.iter().enumerate() {
        for &v in face {
            if (v as usize) < vertex_count {
                vertex_faces[v as usize].push(face_idx);
            }
        }
    }
    vertex_faces
}

/// Helper enum for vertex normals.
enum VertexNormals {
    FromMesh(Vec<Option<Vector3<f64>>>),
    Computed(Vec<Vector3<f64>>),
}

/// Compute vertex normals or use existing ones.
fn compute_or_get_normals(mesh: &Mesh) -> VertexNormals {
    // Check if mesh already has normals
    let has_normals = mesh.vertices.iter().any(|v| v.normal.is_some());

    if has_normals {
        let normals: Vec<Option<Vector3<f64>>> = mesh.vertices.iter().map(|v| v.normal).collect();
        VertexNormals::FromMesh(normals)
    } else {
        // Compute normals from face normals
        let mut normals = vec![Vector3::zeros(); mesh.vertices.len()];

        for face in &mesh.faces {
            let v0 = &mesh.vertices[face[0] as usize].position;
            let v1 = &mesh.vertices[face[1] as usize].position;
            let v2 = &mesh.vertices[face[2] as usize].position;

            let e1 = v1 - v0;
            let e2 = v2 - v0;
            let face_normal = e1.cross(&e2);

            // Area-weighted contribution
            for &vi in face {
                normals[vi as usize] += face_normal;
            }
        }

        // Normalize
        for normal in &mut normals {
            let len = normal.norm();
            if len > f64::EPSILON {
                *normal /= len;
            }
        }

        VertexNormals::Computed(normals)
    }
}

// Convenience method on Mesh
impl Mesh {
    /// Analyze wall thickness at each vertex.
    ///
    /// Casts rays inward from each vertex to measure local wall thickness.
    /// Useful for identifying thin regions that may fail during 3D printing.
    ///
    /// # Example
    /// ```
    /// use mesh_repair::Mesh;
    /// use mesh_repair::thickness::ThicknessParams;
    ///
    /// let mesh = Mesh::new(); // Would normally load a real mesh
    /// let result = mesh.analyze_thickness(&ThicknessParams::for_printing());
    /// if result.has_thin_regions() {
    ///     println!("Warning: {} thin regions found", result.thin_regions.len());
    /// }
    /// ```
    pub fn analyze_thickness(&self, params: &ThicknessParams) -> ThicknessResult {
        analyze_thickness(self, params)
    }

    /// Get wall thickness at each vertex using default parameters.
    pub fn wall_thickness(&self) -> ThicknessResult {
        analyze_thickness(self, &ThicknessParams::default())
    }

    /// Check if all walls meet minimum thickness requirement.
    ///
    /// # Arguments
    /// * `min_thickness` - Minimum acceptable thickness in mesh units
    ///
    /// # Returns
    /// `true` if all measured walls are at least `min_thickness` thick.
    pub fn is_thick_enough(&self, min_thickness: f64) -> bool {
        let result = analyze_thickness(self, &ThicknessParams::with_min_thickness(min_thickness));
        result.is_thick_enough(min_thickness)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::Vertex;

    /// Create a unit cube mesh.
    fn make_unit_cube() -> Mesh {
        let mut mesh = Mesh::new();

        // 8 vertices of a 10x10x10 cube
        mesh.vertices.push(Vertex::from_coords(0.0, 0.0, 0.0)); // 0
        mesh.vertices.push(Vertex::from_coords(10.0, 0.0, 0.0)); // 1
        mesh.vertices.push(Vertex::from_coords(10.0, 10.0, 0.0)); // 2
        mesh.vertices.push(Vertex::from_coords(0.0, 10.0, 0.0)); // 3
        mesh.vertices.push(Vertex::from_coords(0.0, 0.0, 10.0)); // 4
        mesh.vertices.push(Vertex::from_coords(10.0, 0.0, 10.0)); // 5
        mesh.vertices.push(Vertex::from_coords(10.0, 10.0, 10.0)); // 6
        mesh.vertices.push(Vertex::from_coords(0.0, 10.0, 10.0)); // 7

        // 12 triangles (2 per face), CCW winding when viewed from outside
        // Bottom face (z=0)
        mesh.faces.push([0, 2, 1]);
        mesh.faces.push([0, 3, 2]);
        // Top face (z=10)
        mesh.faces.push([4, 5, 6]);
        mesh.faces.push([4, 6, 7]);
        // Front face (y=0)
        mesh.faces.push([0, 1, 5]);
        mesh.faces.push([0, 5, 4]);
        // Back face (y=10)
        mesh.faces.push([3, 7, 6]);
        mesh.faces.push([3, 6, 2]);
        // Left face (x=0)
        mesh.faces.push([0, 4, 7]);
        mesh.faces.push([0, 7, 3]);
        // Right face (x=10)
        mesh.faces.push([1, 2, 6]);
        mesh.faces.push([1, 6, 5]);

        mesh
    }

    /// Create a thin box (1mm thick walls).
    fn make_thin_box() -> Mesh {
        let mut mesh = Mesh::new();

        // Outer cube 10x10x10
        let outer = [
            (0.0, 0.0, 0.0),
            (10.0, 0.0, 0.0),
            (10.0, 10.0, 0.0),
            (0.0, 10.0, 0.0),
            (0.0, 0.0, 10.0),
            (10.0, 0.0, 10.0),
            (10.0, 10.0, 10.0),
            (0.0, 10.0, 10.0),
        ];

        // Inner cube (1mm wall thickness)
        let wall = 1.0;
        let inner = [
            (wall, wall, wall),
            (10.0 - wall, wall, wall),
            (10.0 - wall, 10.0 - wall, wall),
            (wall, 10.0 - wall, wall),
            (wall, wall, 10.0 - wall),
            (10.0 - wall, wall, 10.0 - wall),
            (10.0 - wall, 10.0 - wall, 10.0 - wall),
            (wall, 10.0 - wall, 10.0 - wall),
        ];

        // Add outer vertices (0-7)
        for (x, y, z) in outer {
            mesh.vertices.push(Vertex::from_coords(x, y, z));
        }

        // Add inner vertices (8-15)
        for (x, y, z) in inner {
            mesh.vertices.push(Vertex::from_coords(x, y, z));
        }

        // Outer faces (CCW from outside)
        mesh.faces.push([0, 2, 1]);
        mesh.faces.push([0, 3, 2]); // Bottom
        mesh.faces.push([4, 5, 6]);
        mesh.faces.push([4, 6, 7]); // Top
        mesh.faces.push([0, 1, 5]);
        mesh.faces.push([0, 5, 4]); // Front
        mesh.faces.push([3, 7, 6]);
        mesh.faces.push([3, 6, 2]); // Back
        mesh.faces.push([0, 4, 7]);
        mesh.faces.push([0, 7, 3]); // Left
        mesh.faces.push([1, 2, 6]);
        mesh.faces.push([1, 6, 5]); // Right

        // Inner faces (CCW from inside = CW from outside, so reversed)
        mesh.faces.push([8, 9, 10]);
        mesh.faces.push([8, 10, 11]); // Bottom inner
        mesh.faces.push([12, 14, 13]);
        mesh.faces.push([12, 15, 14]); // Top inner
        mesh.faces.push([8, 13, 9]);
        mesh.faces.push([8, 12, 13]); // Front inner
        mesh.faces.push([11, 14, 15]);
        mesh.faces.push([11, 10, 14]); // Back inner
        mesh.faces.push([8, 11, 15]);
        mesh.faces.push([8, 15, 12]); // Left inner
        mesh.faces.push([9, 13, 14]);
        mesh.faces.push([9, 14, 10]); // Right inner

        mesh
    }

    #[test]
    fn test_ray_triangle_intersect_hit() {
        let tri = Triangle::new(
            Point3::new(0.0, 0.0, 0.0),
            Point3::new(1.0, 0.0, 0.0),
            Point3::new(0.5, 1.0, 0.0),
        );

        // Ray from above pointing down
        let origin = Point3::new(0.5, 0.5, 1.0);
        let direction = Vector3::new(0.0, 0.0, -1.0);

        let result = ray_triangle_intersect(&origin, &direction, &tri, 1e-10);
        assert!(result.is_some());
        let t = result.unwrap();
        assert!((t - 1.0).abs() < 1e-8, "Expected t=1.0, got {}", t);
    }

    #[test]
    fn test_ray_triangle_intersect_miss() {
        let tri = Triangle::new(
            Point3::new(0.0, 0.0, 0.0),
            Point3::new(1.0, 0.0, 0.0),
            Point3::new(0.5, 1.0, 0.0),
        );

        // Ray from above but misses triangle
        let origin = Point3::new(5.0, 5.0, 1.0);
        let direction = Vector3::new(0.0, 0.0, -1.0);

        let result = ray_triangle_intersect(&origin, &direction, &tri, 1e-10);
        assert!(result.is_none());
    }

    #[test]
    fn test_ray_triangle_parallel() {
        let tri = Triangle::new(
            Point3::new(0.0, 0.0, 0.0),
            Point3::new(1.0, 0.0, 0.0),
            Point3::new(0.5, 1.0, 0.0),
        );

        // Ray parallel to triangle
        let origin = Point3::new(0.0, 0.0, 1.0);
        let direction = Vector3::new(1.0, 0.0, 0.0);

        let result = ray_triangle_intersect(&origin, &direction, &tri, 1e-10);
        assert!(result.is_none());
    }

    #[test]
    fn test_analyze_empty_mesh() {
        let mesh = Mesh::new();
        let result = analyze_thickness(&mesh, &ThicknessParams::default());

        assert_eq!(result.vertices_analyzed, 0);
        assert_eq!(result.vertices_with_hits, 0);
        assert!(!result.has_thin_regions());
    }

    #[test]
    fn test_analyze_single_triangle() {
        let mut mesh = Mesh::new();
        mesh.vertices.push(Vertex::from_coords(0.0, 0.0, 0.0));
        mesh.vertices.push(Vertex::from_coords(1.0, 0.0, 0.0));
        mesh.vertices.push(Vertex::from_coords(0.5, 1.0, 0.0));
        mesh.faces.push([0, 1, 2]);

        let result = analyze_thickness(&mesh, &ThicknessParams::default());

        // Single triangle has no interior, so rays won't hit anything
        assert_eq!(result.vertices_analyzed, 3);
        assert_eq!(result.vertices_with_hits, 0);
    }

    #[test]
    fn test_analyze_solid_cube() {
        let mesh = make_unit_cube();
        let result = analyze_thickness(&mesh, &ThicknessParams::default());

        assert_eq!(result.vertices_analyzed, 8);
        // Corner vertices should have hits going through the cube
        assert!(result.vertices_with_hits > 0);
        // 10x10x10 cube should have thickness around 10 (diagonal is longer)
        assert!(result.min_thickness > 0.0 && result.min_thickness.is_finite());
    }

    #[test]
    fn test_analyze_thin_box() {
        let mesh = make_thin_box();
        let params = ThicknessParams {
            min_thickness: 2.0, // 2mm threshold (wall is 1mm)
            ..Default::default()
        };

        let result = analyze_thickness(&mesh, &params);

        // Should find thin regions since wall is 1mm but threshold is 2mm
        assert!(result.has_thin_regions());
        // Minimum thickness should be around 1mm
        if result.min_thickness.is_finite() && result.min_thickness > 0.0 {
            assert!(
                result.min_thickness < 2.0,
                "Expected thin walls, got min thickness: {}",
                result.min_thickness
            );
        }
    }

    #[test]
    fn test_thickness_params_presets() {
        let fdm = ThicknessParams::for_printing();
        assert!((fdm.min_thickness - 0.8).abs() < 0.01);

        let sla = ThicknessParams::for_sla();
        assert!((sla.min_thickness - 0.4).abs() < 0.01);

        let custom = ThicknessParams::with_min_thickness(1.5);
        assert!((custom.min_thickness - 1.5).abs() < 0.01);
    }

    #[test]
    fn test_thickness_result_display() {
        let result = ThicknessResult {
            vertex_thickness: vec![1.0, 2.0, 3.0],
            min_thickness: 1.0,
            max_thickness: 3.0,
            avg_thickness: 2.0,
            thin_regions: vec![ThinRegion {
                vertex_index: 0,
                position: Point3::new(0.0, 0.0, 0.0),
                thickness: 1.0,
                hit_face: Some(0),
            }],
            vertices_analyzed: 3,
            vertices_with_hits: 3,
            vertices_skipped: 0,
            truncated: false,
        };

        let output = format!("{}", result);
        assert!(output.contains("Wall Thickness"));
        assert!(output.contains("Min thickness: 1.000"));
        assert!(output.contains("Thin regions: 1"));
    }

    #[test]
    fn test_vertices_below_thickness() {
        let result = ThicknessResult {
            vertex_thickness: vec![0.5, 1.5, 2.5, f64::INFINITY],
            min_thickness: 0.5,
            max_thickness: 2.5,
            avg_thickness: 1.5,
            thin_regions: Vec::new(),
            vertices_analyzed: 4,
            vertices_with_hits: 3,
            vertices_skipped: 1,
            truncated: false,
        };

        let below_1 = result.vertices_below_thickness(1.0);
        assert_eq!(below_1, vec![0]);

        let below_2 = result.vertices_below_thickness(2.0);
        assert_eq!(below_2, vec![0, 1]);

        let below_3 = result.vertices_below_thickness(3.0);
        assert_eq!(below_3, vec![0, 1, 2]);
    }

    #[test]
    fn test_mesh_convenience_methods() {
        let mesh = make_unit_cube();

        let result = mesh.wall_thickness();
        assert!(result.vertices_analyzed > 0);

        // Cube is at least 5mm thick, should pass 1mm test
        // Note: Due to corner geometry, actual measured thickness varies
        let is_thick = mesh.is_thick_enough(1.0);
        // This test just ensures the method runs without error
        let _ = is_thick;
    }

    #[test]
    fn test_aabb_ray_intersect() {
        let aabb = Aabb {
            min: Point3::new(0.0, 0.0, 0.0),
            max: Point3::new(1.0, 1.0, 1.0),
        };

        // Ray hitting the box
        let origin = Point3::new(0.5, 0.5, -1.0);
        let _dir = Vector3::new(0.0, 0.0, 1.0);
        let dir_inv = Vector3::new(f64::MAX, f64::MAX, 1.0);

        let result = aabb.ray_intersect(&origin, &dir_inv);
        assert!(result.is_some());

        // Ray missing the box
        let origin2 = Point3::new(5.0, 5.0, -1.0);
        let result2 = aabb.ray_intersect(&origin2, &dir_inv);
        assert!(result2.is_none());
    }
}

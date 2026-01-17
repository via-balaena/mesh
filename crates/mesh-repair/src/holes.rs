//! Hole detection and filling for mesh repair.

use hashbrown::{HashMap, HashSet};
use nalgebra::{Point3, Vector3};
use rayon::prelude::*;
use tracing::{debug, info, warn};

use crate::adjacency::MeshAdjacency;
use crate::error::MeshResult;
use crate::{Mesh, Triangle};

/// A boundary loop representing a hole in the mesh.
#[derive(Debug, Clone)]
pub struct BoundaryLoop {
    /// Ordered list of vertex indices forming the loop.
    pub vertices: Vec<u32>,
}

impl BoundaryLoop {
    /// Number of edges (and vertices) in the loop.
    pub fn edge_count(&self) -> usize {
        self.vertices.len()
    }
}

/// Detect all boundary loops (holes) in the mesh.
pub fn detect_holes(_mesh: &Mesh, adjacency: &MeshAdjacency) -> Vec<BoundaryLoop> {
    // Get all boundary edges
    let boundary_edges: Vec<(u32, u32)> = adjacency.boundary_edges().collect();

    if boundary_edges.is_empty() {
        return Vec::new();
    }

    debug!("Found {} boundary edges", boundary_edges.len());

    // Build adjacency for boundary edges
    let mut edge_neighbors: HashMap<u32, Vec<u32>> = HashMap::new();
    for &(a, b) in &boundary_edges {
        edge_neighbors.entry(a).or_default().push(b);
        edge_neighbors.entry(b).or_default().push(a);
    }

    // Find connected loops via DFS
    let mut visited_vertices: HashSet<u32> = HashSet::new();
    let mut loops = Vec::new();

    for &(start, _) in &boundary_edges {
        if visited_vertices.contains(&start) {
            continue;
        }

        // Trace the loop
        let mut loop_vertices = Vec::new();
        let mut current = start;
        let mut prev: Option<u32> = None;

        loop {
            visited_vertices.insert(current);
            loop_vertices.push(current);

            // Find next vertex in loop (not the one we came from)
            let neighbors = edge_neighbors
                .get(&current)
                .map(|v| v.as_slice())
                .unwrap_or(&[]);

            let next = neighbors
                .iter()
                .find(|&&n| Some(n) != prev && !visited_vertices.contains(&n))
                .or_else(|| {
                    // If we're closing the loop, allow going back to start
                    neighbors
                        .iter()
                        .find(|&&n| n == start && loop_vertices.len() > 2)
                });

            match next {
                Some(&n) if n == start => {
                    // Loop closed
                    break;
                }
                Some(&n) => {
                    prev = Some(current);
                    current = n;
                }
                None => {
                    // Dead end or malformed boundary
                    warn!("Boundary loop starting at vertex {} is not closed", start);
                    break;
                }
            }
        }

        if loop_vertices.len() >= 3 {
            loops.push(BoundaryLoop {
                vertices: loop_vertices,
            });
        }
    }

    info!(
        "Detected {} holes (boundary loops), sizes: {:?}",
        loops.len(),
        loops.iter().map(|l| l.edge_count()).collect::<Vec<_>>()
    );

    loops
}

/// Fill a hole using ear clipping triangulation.
///
/// Returns the new triangles to add to the mesh.
pub fn fill_hole_ear_clipping(mesh: &Mesh, boundary: &BoundaryLoop) -> Vec<[u32; 3]> {
    let n = boundary.vertices.len();
    if n < 3 {
        return Vec::new();
    }

    // Get positions
    let positions: Vec<Point3<f64>> = boundary
        .vertices
        .iter()
        .map(|&idx| mesh.vertices[idx as usize].position)
        .collect();

    // Compute average normal of the hole (for consistent winding)
    let centroid = positions.iter().fold(Point3::origin(), |acc, p| {
        Point3::new(acc.x + p.x, acc.y + p.y, acc.z + p.z)
    }) / (n as f64);

    let hole_normal = compute_hole_normal(&positions, &centroid);

    // Ear clipping
    let mut remaining: Vec<usize> = (0..n).collect();
    let mut triangles = Vec::new();

    while remaining.len() > 3 {
        let mut found_ear = false;

        for i in 0..remaining.len() {
            let prev = remaining[(i + remaining.len() - 1) % remaining.len()];
            let curr = remaining[i];
            let next = remaining[(i + 1) % remaining.len()];

            if is_ear(&positions, &remaining, prev, curr, next, &hole_normal) {
                // Add triangle with correct winding
                let tri = [
                    boundary.vertices[prev],
                    boundary.vertices[curr],
                    boundary.vertices[next],
                ];
                triangles.push(tri);

                remaining.remove(i);
                found_ear = true;
                break;
            }
        }

        if !found_ear {
            warn!(
                "Ear clipping stuck with {} vertices remaining, using fan triangulation",
                remaining.len()
            );
            // Fall back to simple fan
            break;
        }
    }

    // Handle remaining triangle
    if remaining.len() == 3 {
        triangles.push([
            boundary.vertices[remaining[0]],
            boundary.vertices[remaining[1]],
            boundary.vertices[remaining[2]],
        ]);
    } else if remaining.len() > 3 {
        // Fan triangulation fallback
        let center = remaining[0];
        for i in 1..remaining.len() - 1 {
            triangles.push([
                boundary.vertices[center],
                boundary.vertices[remaining[i]],
                boundary.vertices[remaining[i + 1]],
            ]);
        }
    }

    debug!(
        "Filled hole with {} edges using {} triangles",
        n,
        triangles.len()
    );

    triangles
}

/// Compute the average normal for a hole boundary.
fn compute_hole_normal(positions: &[Point3<f64>], centroid: &Point3<f64>) -> Vector3<f64> {
    let mut normal = Vector3::zeros();
    let n = positions.len();

    for i in 0..n {
        let p0 = positions[i];
        let p1 = positions[(i + 1) % n];

        let v0 = p0 - centroid;
        let v1 = p1 - centroid;

        normal += v0.cross(&v1);
    }

    let len = normal.norm();
    if len > f64::EPSILON {
        normal / len
    } else {
        Vector3::new(0.0, 0.0, 1.0) // Default up
    }
}

/// Check if vertex at index `curr` forms a valid ear.
fn is_ear(
    positions: &[Point3<f64>],
    remaining: &[usize],
    prev: usize,
    curr: usize,
    next: usize,
    hole_normal: &Vector3<f64>,
) -> bool {
    let p_prev = positions[prev];
    let p_curr = positions[curr];
    let p_next = positions[next];

    // Check that the triangle is convex (normal matches hole normal)
    let tri = Triangle::new(p_prev, p_curr, p_next);
    let tri_normal = match tri.normal() {
        Some(n) => n,
        None => return false, // Degenerate
    };

    // Normal should roughly match hole normal
    if tri_normal.dot(hole_normal) < 0.0 {
        return false; // Concave
    }

    // Check that no other vertices are inside the triangle
    for &idx in remaining {
        if idx == prev || idx == curr || idx == next {
            continue;
        }

        if point_in_triangle_2d(&positions[idx], &p_prev, &p_curr, &p_next, hole_normal) {
            return false;
        }
    }

    true
}

/// Check if point is inside triangle (projected onto plane defined by normal).
fn point_in_triangle_2d(
    p: &Point3<f64>,
    v0: &Point3<f64>,
    v1: &Point3<f64>,
    v2: &Point3<f64>,
    normal: &Vector3<f64>,
) -> bool {
    // Project to 2D by dropping the axis most aligned with normal
    let abs_normal = Vector3::new(normal.x.abs(), normal.y.abs(), normal.z.abs());

    let (p2, a2, b2, c2) = if abs_normal.z >= abs_normal.x && abs_normal.z >= abs_normal.y {
        // Drop Z
        ((p.x, p.y), (v0.x, v0.y), (v1.x, v1.y), (v2.x, v2.y))
    } else if abs_normal.y >= abs_normal.x {
        // Drop Y
        ((p.x, p.z), (v0.x, v0.z), (v1.x, v1.z), (v2.x, v2.z))
    } else {
        // Drop X
        ((p.y, p.z), (v0.y, v0.z), (v1.y, v1.z), (v2.y, v2.z))
    };

    point_in_triangle_2d_impl(p2, a2, b2, c2)
}

fn point_in_triangle_2d_impl(p: (f64, f64), a: (f64, f64), b: (f64, f64), c: (f64, f64)) -> bool {
    let sign = |p1: (f64, f64), p2: (f64, f64), p3: (f64, f64)| -> f64 {
        (p1.0 - p3.0) * (p2.1 - p3.1) - (p2.0 - p3.0) * (p1.1 - p3.1)
    };

    let d1 = sign(p, a, b);
    let d2 = sign(p, b, c);
    let d3 = sign(p, c, a);

    let has_neg = d1 < 0.0 || d2 < 0.0 || d3 < 0.0;
    let has_pos = d1 > 0.0 || d2 > 0.0 || d3 > 0.0;

    !(has_neg && has_pos)
}

/// Fill all holes in the mesh that are below the maximum edge count.
///
/// Returns the number of holes filled.
pub fn fill_holes(mesh: &mut Mesh) -> MeshResult<usize> {
    fill_holes_with_max_edges(mesh, 100)
}

/// Fill all holes in the mesh that are below the maximum edge count.
///
/// Returns the number of holes filled.
///
/// Uses parallel processing via rayon - each hole is filled independently,
/// then all triangles are merged into the mesh.
pub fn fill_holes_with_max_edges(mesh: &mut Mesh, max_hole_edges: usize) -> MeshResult<usize> {
    let adjacency = MeshAdjacency::build(&mesh.faces);
    let holes = detect_holes(mesh, &adjacency);

    // Partition holes into fillable and too-large
    let (fillable, skipped): (Vec<_>, Vec<_>) = holes
        .into_iter()
        .partition(|hole| hole.edge_count() <= max_hole_edges);

    // Log skipped holes
    for hole in &skipped {
        warn!(
            "Skipping large hole with {} edges (max: {})",
            hole.edge_count(),
            max_hole_edges
        );
    }

    // Fill holes in parallel - each hole is independent
    // The fill_hole_ear_clipping function only reads from mesh.vertices (immutable)
    // and produces new triangles without modifying the mesh
    let all_new_triangles: Vec<Vec<[u32; 3]>> = fillable
        .par_iter()
        .map(|hole| fill_hole_ear_clipping(mesh, hole))
        .collect();

    let filled_count = all_new_triangles.len();

    // Merge all new triangles into the mesh (sequential, but fast)
    for triangles in all_new_triangles {
        mesh.faces.extend(triangles);
    }

    if filled_count > 0 {
        info!("Filled {} holes", filled_count);
    }

    Ok(filled_count)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::Vertex;

    fn open_box_mesh() -> Mesh {
        // A box missing its top face (has a square hole)
        let mut mesh = Mesh::new();

        // 8 vertices of a unit cube
        mesh.vertices.push(Vertex::from_coords(0.0, 0.0, 0.0)); // 0
        mesh.vertices.push(Vertex::from_coords(1.0, 0.0, 0.0)); // 1
        mesh.vertices.push(Vertex::from_coords(1.0, 1.0, 0.0)); // 2
        mesh.vertices.push(Vertex::from_coords(0.0, 1.0, 0.0)); // 3
        mesh.vertices.push(Vertex::from_coords(0.0, 0.0, 1.0)); // 4
        mesh.vertices.push(Vertex::from_coords(1.0, 0.0, 1.0)); // 5
        mesh.vertices.push(Vertex::from_coords(1.0, 1.0, 1.0)); // 6
        mesh.vertices.push(Vertex::from_coords(0.0, 1.0, 1.0)); // 7

        // Bottom face
        mesh.faces.push([0, 2, 1]);
        mesh.faces.push([0, 3, 2]);

        // Front face
        mesh.faces.push([0, 1, 5]);
        mesh.faces.push([0, 5, 4]);

        // Right face
        mesh.faces.push([1, 2, 6]);
        mesh.faces.push([1, 6, 5]);

        // Back face
        mesh.faces.push([2, 3, 7]);
        mesh.faces.push([2, 7, 6]);

        // Left face
        mesh.faces.push([3, 0, 4]);
        mesh.faces.push([3, 4, 7]);

        // Top face is missing (hole at z=1)

        mesh
    }

    #[test]
    fn test_detect_holes() {
        let mesh = open_box_mesh();
        let adjacency = MeshAdjacency::build(&mesh.faces);
        let holes = detect_holes(&mesh, &adjacency);

        assert_eq!(holes.len(), 1);
        assert_eq!(holes[0].edge_count(), 4); // Square hole
    }

    #[test]
    fn test_fill_holes() {
        let mut mesh = open_box_mesh();
        let initial_faces = mesh.face_count();

        let filled = fill_holes(&mut mesh).unwrap();

        assert_eq!(filled, 1);
        assert!(mesh.face_count() > initial_faces);

        // Should now be watertight
        let adjacency = MeshAdjacency::build(&mesh.faces);
        assert!(adjacency.is_watertight());
    }
}

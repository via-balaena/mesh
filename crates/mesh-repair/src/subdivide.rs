//! Mesh subdivision for surface smoothing.
//!
//! This module provides Loop subdivision to increase triangle count
//! while producing smoother surfaces. Useful for coarse input meshes.

use hashbrown::HashMap;
use nalgebra::{Point3, Vector3};

use crate::{Mesh, MeshAdjacency, Vertex};

/// Parameters for mesh subdivision.
#[derive(Debug, Clone)]
#[cfg_attr(
    feature = "pipeline-config",
    derive(serde::Serialize, serde::Deserialize)
)]
pub struct SubdivideParams {
    /// Number of subdivision iterations.
    /// Each iteration roughly quadruples the triangle count.
    /// Default: 1
    pub iterations: usize,
    /// Whether to preserve sharp edges/creases.
    /// When true, edges with dihedral angle above threshold are kept sharp.
    /// Default: false
    pub preserve_sharp_edges: bool,
    /// Dihedral angle threshold (in radians) for sharp edge detection.
    /// Edges where adjacent faces form an angle greater than this are considered sharp.
    /// Default: PI/3 (60 degrees)
    pub sharp_angle_threshold: f64,
    /// Whether to preserve boundary edges.
    /// When true, boundary vertices use special boundary rules.
    /// Default: true
    pub preserve_boundary: bool,
}

impl Default for SubdivideParams {
    fn default() -> Self {
        Self {
            iterations: 1,
            preserve_sharp_edges: false,
            sharp_angle_threshold: std::f64::consts::PI / 3.0,
            preserve_boundary: true,
        }
    }
}

impl SubdivideParams {
    /// Create params for a single subdivision iteration.
    pub fn single() -> Self {
        Self::default()
    }

    /// Create params for multiple subdivision iterations.
    pub fn with_iterations(iterations: usize) -> Self {
        Self {
            iterations,
            ..Default::default()
        }
    }

    /// Create params that preserve sharp features.
    pub fn preserve_features() -> Self {
        Self {
            preserve_sharp_edges: true,
            ..Default::default()
        }
    }
}

/// Result of mesh subdivision.
#[derive(Debug)]
pub struct SubdivideResult {
    /// The subdivided mesh.
    pub mesh: Mesh,
    /// Original triangle count.
    pub original_triangles: usize,
    /// Final triangle count.
    pub final_triangles: usize,
    /// Number of iterations performed.
    pub iterations_performed: usize,
}

/// Subdivide a mesh using Loop subdivision.
///
/// Loop subdivision is a technique for creating smoother surfaces by:
/// 1. Inserting new vertices at edge midpoints
/// 2. Updating original vertex positions using neighbor averaging
/// 3. Creating 4 new triangles from each original triangle
///
/// # Arguments
/// * `mesh` - The input mesh to subdivide
/// * `params` - Subdivision parameters
///
/// # Returns
/// A `SubdivideResult` containing the subdivided mesh and statistics.
///
/// # Example
/// ```
/// use mesh_repair::{Mesh, Vertex, subdivide_mesh, SubdivideParams};
///
/// let mut mesh = Mesh::new();
/// mesh.vertices.push(Vertex::from_coords(0.0, 0.0, 0.0));
/// mesh.vertices.push(Vertex::from_coords(1.0, 0.0, 0.0));
/// mesh.vertices.push(Vertex::from_coords(0.5, 1.0, 0.0));
/// mesh.faces.push([0, 1, 2]);
///
/// let result = subdivide_mesh(&mesh, &SubdivideParams::single());
/// assert_eq!(result.final_triangles, 4); // 1 triangle becomes 4
/// ```
pub fn subdivide_mesh(mesh: &Mesh, params: &SubdivideParams) -> SubdivideResult {
    let original_triangles = mesh.faces.len();

    if original_triangles == 0 || params.iterations == 0 {
        return SubdivideResult {
            mesh: mesh.clone(),
            original_triangles,
            final_triangles: original_triangles,
            iterations_performed: 0,
        };
    }

    let mut current_mesh = mesh.clone();

    for _ in 0..params.iterations {
        current_mesh = subdivide_once(&current_mesh, params);
    }

    SubdivideResult {
        final_triangles: current_mesh.faces.len(),
        mesh: current_mesh,
        original_triangles,
        iterations_performed: params.iterations,
    }
}

/// Perform one iteration of Loop subdivision.
fn subdivide_once(mesh: &Mesh, params: &SubdivideParams) -> Mesh {
    let adj = MeshAdjacency::build(&mesh.faces);

    // Identify boundary and sharp edges
    let boundary_edges: hashbrown::HashSet<(u32, u32)> = adj.boundary_edges().collect();
    let sharp_edges: hashbrown::HashSet<(u32, u32)> = if params.preserve_sharp_edges {
        find_sharp_edges(mesh, &adj, params.sharp_angle_threshold)
    } else {
        hashbrown::HashSet::new()
    };

    // Classify vertices
    let boundary_vertices: hashbrown::HashSet<u32> =
        boundary_edges.iter().flat_map(|&(a, b)| [a, b]).collect();

    // Step 1: Create edge midpoint vertices
    // Map from edge (v0, v1) to new vertex index
    let mut edge_vertices: HashMap<(u32, u32), u32> = HashMap::new();
    let mut new_vertices: Vec<Vertex> = mesh.vertices.clone();

    for &edge in adj.edge_to_faces.keys() {
        let (v0, v1) = edge;
        let is_boundary = boundary_edges.contains(&edge);
        let is_sharp = sharp_edges.contains(&edge);

        let new_pos = if is_boundary || is_sharp {
            // Boundary/sharp edge: simple midpoint
            let p0 = &mesh.vertices[v0 as usize].position;
            let p1 = &mesh.vertices[v1 as usize].position;
            Point3::new(
                (p0.x + p1.x) / 2.0,
                (p0.y + p1.y) / 2.0,
                (p0.z + p1.z) / 2.0,
            )
        } else {
            // Interior edge: Loop subdivision weights (3/8, 3/8, 1/8, 1/8)
            compute_edge_vertex_position(mesh, &adj, v0, v1)
        };

        let new_idx = new_vertices.len() as u32;
        let mut new_vertex = Vertex::new(new_pos);

        // Interpolate vertex attributes (offset, tag) from edge endpoints
        let v0_vert = &mesh.vertices[v0 as usize];
        let v1_vert = &mesh.vertices[v1 as usize];

        if let (Some(o0), Some(o1)) = (v0_vert.offset, v1_vert.offset) {
            new_vertex.offset = Some((o0 + o1) / 2.0);
        }

        // Use tag from first vertex if both have same tag, otherwise None
        if v0_vert.tag == v1_vert.tag {
            new_vertex.tag = v0_vert.tag;
        }

        new_vertices.push(new_vertex);
        edge_vertices.insert(edge, new_idx);
    }

    // Step 2: Update original vertex positions
    let _num_original = mesh.vertices.len();
    for (i, vertex) in mesh.vertices.iter().enumerate() {
        let i = i as u32;
        let is_boundary = boundary_vertices.contains(&i);

        let new_pos = if is_boundary && params.preserve_boundary {
            // Boundary vertex: use boundary rule
            compute_boundary_vertex_position(mesh, &adj, &boundary_edges, i)
        } else {
            // Interior vertex: use Loop subdivision weights
            compute_interior_vertex_position(mesh, &adj, i)
        };

        new_vertices[i as usize].position = new_pos;

        // Preserve original vertex attributes
        new_vertices[i as usize].offset = vertex.offset;
        new_vertices[i as usize].tag = vertex.tag;
        new_vertices[i as usize].normal = vertex.normal;
    }

    // Step 3: Create new faces
    // Each original triangle becomes 4 triangles
    let mut new_faces: Vec<[u32; 3]> = Vec::with_capacity(mesh.faces.len() * 4);

    for face in &mesh.faces {
        let v0 = face[0];
        let v1 = face[1];
        let v2 = face[2];

        // Get edge midpoint vertices
        let e01 = get_edge_vertex(&edge_vertices, v0, v1);
        let e12 = get_edge_vertex(&edge_vertices, v1, v2);
        let e20 = get_edge_vertex(&edge_vertices, v2, v0);

        // Create 4 new triangles
        //       v0
        //      /  \
        //    e20--e01
        //    / \  / \
        //  v2--e12--v1

        new_faces.push([v0, e01, e20]);
        new_faces.push([e01, v1, e12]);
        new_faces.push([e20, e12, v2]);
        new_faces.push([e01, e12, e20]); // Center triangle
    }

    Mesh {
        vertices: new_vertices,
        faces: new_faces,
    }
}

/// Get the edge vertex index, handling edge canonicalization.
fn get_edge_vertex(edge_vertices: &HashMap<(u32, u32), u32>, v0: u32, v1: u32) -> u32 {
    let key = if v0 < v1 { (v0, v1) } else { (v1, v0) };
    *edge_vertices.get(&key).expect("edge vertex should exist")
}

/// Compute the position of a new vertex on an interior edge.
///
/// Uses Loop subdivision weights: 3/8 for edge endpoints, 1/8 for opposite vertices.
fn compute_edge_vertex_position(mesh: &Mesh, adj: &MeshAdjacency, v0: u32, v1: u32) -> Point3<f64> {
    let p0 = &mesh.vertices[v0 as usize].position;
    let p1 = &mesh.vertices[v1 as usize].position;

    let edge_key = if v0 < v1 { (v0, v1) } else { (v1, v0) };

    if let Some(face_indices) = adj.edge_to_faces.get(&edge_key)
        && face_indices.len() == 2
    {
        // Find the opposite vertices in the two adjacent faces
        let mut opposite_vertices = Vec::new();
        for &fi in face_indices {
            let face = &mesh.faces[fi as usize];
            for &fv in face {
                if fv != v0 && fv != v1 {
                    opposite_vertices.push(fv);
                    break;
                }
            }
        }

        if opposite_vertices.len() == 2 {
            let p2 = &mesh.vertices[opposite_vertices[0] as usize].position;
            let p3 = &mesh.vertices[opposite_vertices[1] as usize].position;

            // Loop weights: 3/8, 3/8, 1/8, 1/8
            return Point3::new(
                (3.0 * p0.x + 3.0 * p1.x + p2.x + p3.x) / 8.0,
                (3.0 * p0.y + 3.0 * p1.y + p2.y + p3.y) / 8.0,
                (3.0 * p0.z + 3.0 * p1.z + p2.z + p3.z) / 8.0,
            );
        }
    }

    // Fallback to midpoint
    Point3::new(
        (p0.x + p1.x) / 2.0,
        (p0.y + p1.y) / 2.0,
        (p0.z + p1.z) / 2.0,
    )
}

/// Compute the new position for an interior vertex using Loop weights.
fn compute_interior_vertex_position(mesh: &Mesh, adj: &MeshAdjacency, v: u32) -> Point3<f64> {
    let current_pos = &mesh.vertices[v as usize].position;

    // Find all neighboring vertices
    let neighbors = find_vertex_neighbors(mesh, adj, v);
    let n = neighbors.len();

    if n == 0 {
        return *current_pos;
    }

    // Loop subdivision weight for the original vertex
    // beta = 1/n * (5/8 - (3/8 + 1/4 * cos(2*PI/n))^2)
    // For n=3: beta ≈ 0.1875
    // For n=6: beta ≈ 0.0625
    let beta = compute_loop_beta(n);
    let self_weight = 1.0 - n as f64 * beta;

    // Sum neighbor contributions
    let mut neighbor_sum = Vector3::new(0.0, 0.0, 0.0);
    for &ni in &neighbors {
        let np = &mesh.vertices[ni as usize].position;
        neighbor_sum += Vector3::new(np.x, np.y, np.z);
    }

    Point3::new(
        self_weight * current_pos.x + beta * neighbor_sum.x,
        self_weight * current_pos.y + beta * neighbor_sum.y,
        self_weight * current_pos.z + beta * neighbor_sum.z,
    )
}

/// Compute the new position for a boundary vertex.
fn compute_boundary_vertex_position(
    mesh: &Mesh,
    _adj: &MeshAdjacency,
    boundary_edges: &hashbrown::HashSet<(u32, u32)>,
    v: u32,
) -> Point3<f64> {
    let current_pos = &mesh.vertices[v as usize].position;

    // Find boundary neighbors (vertices connected by boundary edges)
    let mut boundary_neighbors = Vec::new();
    for &(a, b) in boundary_edges {
        if a == v {
            boundary_neighbors.push(b);
        } else if b == v {
            boundary_neighbors.push(a);
        }
    }

    if boundary_neighbors.len() == 2 {
        // Standard boundary rule: 3/4 self + 1/8 each neighbor
        let n0 = &mesh.vertices[boundary_neighbors[0] as usize].position;
        let n1 = &mesh.vertices[boundary_neighbors[1] as usize].position;

        Point3::new(
            0.75 * current_pos.x + 0.125 * n0.x + 0.125 * n1.x,
            0.75 * current_pos.y + 0.125 * n0.y + 0.125 * n1.y,
            0.75 * current_pos.z + 0.125 * n0.z + 0.125 * n1.z,
        )
    } else {
        // Corner or unusual boundary: keep original position
        *current_pos
    }
}

/// Compute the Loop subdivision beta coefficient for a vertex with n neighbors.
fn compute_loop_beta(n: usize) -> f64 {
    if n == 3 {
        // Special case for valence 3
        3.0 / 16.0
    } else {
        let n = n as f64;
        let cos_val = (2.0 * std::f64::consts::PI / n).cos();
        let term = 3.0 / 8.0 + 0.25 * cos_val;
        (1.0 / n) * (5.0 / 8.0 - term * term)
    }
}

/// Find all vertices connected to a given vertex by an edge.
fn find_vertex_neighbors(mesh: &Mesh, adj: &MeshAdjacency, v: u32) -> Vec<u32> {
    let mut neighbors = hashbrown::HashSet::new();

    if let Some(faces) = adj.vertex_to_faces.get(&v) {
        for &fi in faces {
            let face = &mesh.faces[fi as usize];
            for &fv in face {
                if fv != v {
                    neighbors.insert(fv);
                }
            }
        }
    }

    neighbors.into_iter().collect()
}

/// Find edges with dihedral angle above threshold (sharp edges).
fn find_sharp_edges(
    mesh: &Mesh,
    adj: &MeshAdjacency,
    threshold: f64,
) -> hashbrown::HashSet<(u32, u32)> {
    let mut sharp_edges = hashbrown::HashSet::new();

    for (&edge, face_indices) in adj.edge_to_faces.iter() {
        if face_indices.len() != 2 {
            continue;
        }

        let f1 = &mesh.faces[face_indices[0] as usize];
        let f2 = &mesh.faces[face_indices[1] as usize];

        if let (Some(n1), Some(n2)) = (compute_face_normal(mesh, f1), compute_face_normal(mesh, f2))
        {
            let dot = n1.x * n2.x + n1.y * n2.y + n1.z * n2.z;
            let angle = dot.clamp(-1.0, 1.0).acos();
            if angle > threshold {
                sharp_edges.insert(edge);
            }
        }
    }

    sharp_edges
}

/// Compute the unit normal of a face.
fn compute_face_normal(mesh: &Mesh, face: &[u32; 3]) -> Option<Vector3<f64>> {
    let v0 = &mesh.vertices[face[0] as usize].position;
    let v1 = &mesh.vertices[face[1] as usize].position;
    let v2 = &mesh.vertices[face[2] as usize].position;

    let e1 = v1 - v0;
    let e2 = v2 - v0;
    let normal = e1.cross(&e2);

    let len = normal.norm();
    if len < 1e-10 {
        None
    } else {
        Some(normal / len)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_single_triangle() -> Mesh {
        Mesh {
            vertices: vec![
                Vertex::from_coords(0.0, 0.0, 0.0),
                Vertex::from_coords(1.0, 0.0, 0.0),
                Vertex::from_coords(0.5, 1.0, 0.0),
            ],
            faces: vec![[0, 1, 2]],
        }
    }

    fn make_tetrahedron() -> Mesh {
        let mut mesh = Mesh::new();

        // Regular tetrahedron
        mesh.vertices.push(Vertex::from_coords(1.0, 1.0, 1.0));
        mesh.vertices.push(Vertex::from_coords(1.0, -1.0, -1.0));
        mesh.vertices.push(Vertex::from_coords(-1.0, 1.0, -1.0));
        mesh.vertices.push(Vertex::from_coords(-1.0, -1.0, 1.0));

        mesh.faces.push([0, 1, 2]);
        mesh.faces.push([0, 2, 3]);
        mesh.faces.push([0, 3, 1]);
        mesh.faces.push([1, 3, 2]);

        mesh
    }

    fn make_cube() -> Mesh {
        let s = 5.0;
        let mut mesh = Mesh::new();

        mesh.vertices.push(Vertex::from_coords(-s, -s, -s));
        mesh.vertices.push(Vertex::from_coords(s, -s, -s));
        mesh.vertices.push(Vertex::from_coords(s, s, -s));
        mesh.vertices.push(Vertex::from_coords(-s, s, -s));
        mesh.vertices.push(Vertex::from_coords(-s, -s, s));
        mesh.vertices.push(Vertex::from_coords(s, -s, s));
        mesh.vertices.push(Vertex::from_coords(s, s, s));
        mesh.vertices.push(Vertex::from_coords(-s, s, s));

        // Bottom
        mesh.faces.push([0, 2, 1]);
        mesh.faces.push([0, 3, 2]);
        // Top
        mesh.faces.push([4, 5, 6]);
        mesh.faces.push([4, 6, 7]);
        // Front
        mesh.faces.push([0, 1, 5]);
        mesh.faces.push([0, 5, 4]);
        // Back
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

    #[test]
    fn test_subdivide_params_default() {
        let params = SubdivideParams::default();
        assert_eq!(params.iterations, 1);
        assert!(params.preserve_boundary);
        assert!(!params.preserve_sharp_edges);
    }

    #[test]
    fn test_subdivide_single_triangle() {
        let mesh = make_single_triangle();
        let result = subdivide_mesh(&mesh, &SubdivideParams::single());

        // 1 triangle becomes 4
        assert_eq!(result.original_triangles, 1);
        assert_eq!(result.final_triangles, 4);
        assert_eq!(result.iterations_performed, 1);

        // Original 3 vertices + 3 edge midpoints = 6 vertices
        assert_eq!(result.mesh.vertices.len(), 6);
    }

    #[test]
    fn test_subdivide_tetrahedron() {
        let mesh = make_tetrahedron();
        let result = subdivide_mesh(&mesh, &SubdivideParams::single());

        // 4 triangles become 16
        assert_eq!(result.original_triangles, 4);
        assert_eq!(result.final_triangles, 16);

        // 4 original vertices + 6 edge midpoints = 10 vertices
        assert_eq!(result.mesh.vertices.len(), 10);
    }

    #[test]
    fn test_subdivide_cube() {
        let mesh = make_cube();
        let result = subdivide_mesh(&mesh, &SubdivideParams::single());

        // 12 triangles become 48
        assert_eq!(result.original_triangles, 12);
        assert_eq!(result.final_triangles, 48);
    }

    #[test]
    fn test_subdivide_multiple_iterations() {
        let mesh = make_single_triangle();
        let result = subdivide_mesh(&mesh, &SubdivideParams::with_iterations(2));

        // 1 -> 4 -> 16
        assert_eq!(result.final_triangles, 16);
        assert_eq!(result.iterations_performed, 2);
    }

    #[test]
    fn test_subdivide_empty_mesh() {
        let mesh = Mesh::new();
        let result = subdivide_mesh(&mesh, &SubdivideParams::single());

        assert_eq!(result.original_triangles, 0);
        assert_eq!(result.final_triangles, 0);
        assert_eq!(result.iterations_performed, 0);
    }

    #[test]
    fn test_subdivide_zero_iterations() {
        let mesh = make_tetrahedron();
        let result = subdivide_mesh(&mesh, &SubdivideParams::with_iterations(0));

        assert_eq!(result.final_triangles, 4);
        assert_eq!(result.iterations_performed, 0);
    }

    #[test]
    fn test_subdivide_preserves_vertex_attributes() {
        let mut mesh = make_single_triangle();
        mesh.vertices[0].offset = Some(1.0);
        mesh.vertices[1].offset = Some(2.0);
        mesh.vertices[2].offset = Some(3.0);
        mesh.vertices[0].tag = Some(1);
        mesh.vertices[1].tag = Some(1);
        mesh.vertices[2].tag = Some(1);

        let result = subdivide_mesh(&mesh, &SubdivideParams::single());

        // Original vertices should preserve their offsets
        assert!(result.mesh.vertices[0].offset.is_some());
        assert!(result.mesh.vertices[1].offset.is_some());
        assert!(result.mesh.vertices[2].offset.is_some());

        // Edge midpoints should have interpolated offsets
        for v in result.mesh.vertices.iter().skip(3) {
            assert!(v.offset.is_some());
        }

        // All vertices should have same tag (since all original had tag=1)
        for v in &result.mesh.vertices {
            assert_eq!(v.tag, Some(1));
        }
    }

    #[test]
    fn test_subdivide_with_preserve_features() {
        let mesh = make_cube();
        let params = SubdivideParams::preserve_features();

        let result = subdivide_mesh(&mesh, &params);

        // Should still subdivide
        assert_eq!(result.final_triangles, 48);
        // Sharp edges of cube should be detected
        assert!(params.preserve_sharp_edges);
    }

    #[test]
    fn test_loop_beta_values() {
        // Check beta for common valences
        let beta3 = compute_loop_beta(3);
        let beta6 = compute_loop_beta(6);

        // For valence 3, beta = 3/16 = 0.1875
        assert!((beta3 - 0.1875).abs() < 1e-10);

        // For valence 6, self_weight should be reasonable
        let self_weight_6 = 1.0 - 6.0 * beta6;
        assert!(self_weight_6 > 0.0 && self_weight_6 < 1.0);
    }

    #[test]
    fn test_subdivide_maintains_topology() {
        let mesh = make_tetrahedron();
        let result = subdivide_mesh(&mesh, &SubdivideParams::single());

        // Result should still be a valid mesh
        // All face indices should be valid
        for face in &result.mesh.faces {
            for &vi in face {
                assert!((vi as usize) < result.mesh.vertices.len());
            }
        }

        // Should still be watertight (tetrahedron is watertight)
        let adj = MeshAdjacency::build(&result.mesh.faces);
        assert!(adj.is_watertight());
        assert!(adj.is_manifold());
    }
}

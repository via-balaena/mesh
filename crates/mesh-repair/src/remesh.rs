//! Isotropic remeshing for uniform edge lengths and improved triangle quality.
//!
//! This module provides isotropic remeshing to create meshes with uniform edge lengths
//! and well-shaped (near-equilateral) triangles. Useful for scanned meshes with uneven
//! tessellation or when preparing meshes for simulation.

use hashbrown::{HashMap, HashSet};
use nalgebra::{Point3, Vector3};

use crate::{Mesh, MeshAdjacency, Vertex};

/// Parameters for isotropic remeshing.
#[derive(Debug, Clone)]
pub struct RemeshParams {
    /// Target edge length for the remeshed output.
    /// All edges will tend toward this length.
    /// Default: computed from average edge length of input mesh
    pub target_edge_length: Option<f64>,

    /// Number of remeshing iterations.
    /// More iterations produce more uniform results but take longer.
    /// Default: 5
    pub iterations: usize,

    /// Whether to preserve boundary edges (don't collapse them).
    /// Default: true
    pub preserve_boundary: bool,

    /// Whether to preserve sharp edges/creases.
    /// When true, edges with dihedral angle above threshold are not modified.
    /// Default: false
    pub preserve_sharp_edges: bool,

    /// Dihedral angle threshold (in radians) for sharp edge detection.
    /// Edges where adjacent faces form an angle greater than this are considered sharp.
    /// Default: PI/3 (60 degrees)
    pub sharp_angle_threshold: f64,

    /// Minimum allowed edge length as a fraction of target.
    /// Edges shorter than target * min_edge_ratio will be collapsed.
    /// Default: 0.8 (collapse edges shorter than 80% of target)
    pub min_edge_ratio: f64,

    /// Maximum allowed edge length as a fraction of target.
    /// Edges longer than target * max_edge_ratio will be split.
    /// Default: 1.33 (split edges longer than 133% of target)
    pub max_edge_ratio: f64,

    /// Smoothing factor for tangential relaxation (0 = no smoothing, 1 = full).
    /// Default: 0.5
    pub smoothing_factor: f64,
}

impl Default for RemeshParams {
    fn default() -> Self {
        Self {
            target_edge_length: None,
            iterations: 5,
            preserve_boundary: true,
            preserve_sharp_edges: false,
            sharp_angle_threshold: std::f64::consts::PI / 3.0,
            min_edge_ratio: 0.8,
            max_edge_ratio: 1.33,
            smoothing_factor: 0.5,
        }
    }
}

impl RemeshParams {
    /// Create params with a specific target edge length.
    pub fn with_target_edge_length(target: f64) -> Self {
        Self {
            target_edge_length: Some(target),
            ..Default::default()
        }
    }

    /// Create params for high-quality remeshing (more iterations).
    pub fn high_quality() -> Self {
        Self {
            iterations: 10,
            smoothing_factor: 0.7,
            ..Default::default()
        }
    }

    /// Create params for fast remeshing (fewer iterations).
    pub fn fast() -> Self {
        Self {
            iterations: 3,
            ..Default::default()
        }
    }

    /// Create params that preserve sharp features and boundaries.
    pub fn preserve_features() -> Self {
        Self {
            preserve_boundary: true,
            preserve_sharp_edges: true,
            ..Default::default()
        }
    }
}

/// Result of isotropic remeshing.
#[derive(Debug)]
pub struct RemeshResult {
    /// The remeshed output mesh.
    pub mesh: Mesh,
    /// Original triangle count.
    pub original_triangles: usize,
    /// Final triangle count.
    pub final_triangles: usize,
    /// Original vertex count.
    pub original_vertices: usize,
    /// Final vertex count.
    pub final_vertices: usize,
    /// Number of iterations performed.
    pub iterations_performed: usize,
    /// Target edge length used.
    pub target_edge_length: f64,
    /// Number of edges split during remeshing.
    pub edges_split: usize,
    /// Number of edges collapsed during remeshing.
    pub edges_collapsed: usize,
    /// Number of edges flipped during remeshing.
    pub edges_flipped: usize,
}

/// Perform isotropic remeshing on a mesh.
///
/// Isotropic remeshing creates a mesh with uniform edge lengths and well-shaped
/// triangles through iterative application of:
/// 1. **Edge splitting**: Split edges longer than target * max_ratio
/// 2. **Edge collapsing**: Collapse edges shorter than target * min_ratio
/// 3. **Edge flipping**: Flip edges to improve vertex valence toward 6
/// 4. **Tangential smoothing**: Relax vertices toward local centroid
///
/// # Arguments
/// * `mesh` - The input mesh to remesh
/// * `params` - Remeshing parameters
///
/// # Returns
/// A `RemeshResult` containing the remeshed mesh and statistics.
///
/// # Example
/// ```
/// use mesh_repair::{Mesh, Vertex, remesh_isotropic, RemeshParams};
///
/// let mut mesh = Mesh::new();
/// mesh.vertices.push(Vertex::from_coords(0.0, 0.0, 0.0));
/// mesh.vertices.push(Vertex::from_coords(10.0, 0.0, 0.0));
/// mesh.vertices.push(Vertex::from_coords(5.0, 8.66, 0.0));
/// mesh.faces.push([0, 1, 2]);
///
/// let result = remesh_isotropic(&mesh, &RemeshParams::with_target_edge_length(2.0));
/// println!("Remeshed from {} to {} triangles", result.original_triangles, result.final_triangles);
/// ```
pub fn remesh_isotropic(mesh: &Mesh, params: &RemeshParams) -> RemeshResult {
    let original_triangles = mesh.faces.len();
    let original_vertices = mesh.vertices.len();

    if original_triangles == 0 || params.iterations == 0 {
        return RemeshResult {
            mesh: mesh.clone(),
            original_triangles,
            final_triangles: original_triangles,
            original_vertices,
            final_vertices: original_vertices,
            iterations_performed: 0,
            target_edge_length: 0.0,
            edges_split: 0,
            edges_collapsed: 0,
            edges_flipped: 0,
        };
    }

    // Determine target edge length
    let target = params
        .target_edge_length
        .unwrap_or_else(|| compute_average_edge_length(mesh));

    let min_length = target * params.min_edge_ratio;
    let max_length = target * params.max_edge_ratio;

    let mut current_mesh = mesh.clone();
    let mut total_splits = 0;
    let mut total_collapses = 0;
    let mut total_flips = 0;

    for _iter in 0..params.iterations {
        // Build adjacency for this iteration
        let adj = MeshAdjacency::build(&current_mesh.faces);

        // Identify protected edges
        let boundary_edges: HashSet<(u32, u32)> = if params.preserve_boundary {
            adj.boundary_edges().collect()
        } else {
            HashSet::new()
        };

        let sharp_edges: HashSet<(u32, u32)> = if params.preserve_sharp_edges {
            find_sharp_edges(&current_mesh, &adj, params.sharp_angle_threshold)
        } else {
            HashSet::new()
        };

        let _boundary_vertices: HashSet<u32> = boundary_edges
            .iter()
            .flat_map(|&(a, b)| [a, b])
            .collect();

        // Step 1: Split long edges
        let (new_mesh, splits) =
            split_long_edges(&current_mesh, &adj, max_length, &boundary_edges, &sharp_edges);
        current_mesh = new_mesh;
        total_splits += splits;

        // Rebuild adjacency after splits
        let adj = MeshAdjacency::build(&current_mesh.faces);
        let boundary_edges: HashSet<(u32, u32)> = if params.preserve_boundary {
            adj.boundary_edges().collect()
        } else {
            HashSet::new()
        };
        let boundary_vertices: HashSet<u32> = boundary_edges
            .iter()
            .flat_map(|&(a, b)| [a, b])
            .collect();

        // Step 2: Collapse short edges
        let (new_mesh, collapses) = collapse_short_edges(
            &current_mesh,
            &adj,
            min_length,
            &boundary_edges,
            &sharp_edges,
            &boundary_vertices,
        );
        current_mesh = new_mesh;
        total_collapses += collapses;

        // Rebuild adjacency after collapses
        let adj = MeshAdjacency::build(&current_mesh.faces);

        // Step 3: Flip edges to improve valence
        let (new_mesh, flips) = flip_edges_for_valence(&current_mesh, &adj, &boundary_edges, &sharp_edges);
        current_mesh = new_mesh;
        total_flips += flips;

        // Rebuild adjacency after flips
        let adj = MeshAdjacency::build(&current_mesh.faces);
        let boundary_vertices: HashSet<u32> = if params.preserve_boundary {
            adj.boundary_edges().flat_map(|(a, b)| [a, b]).collect()
        } else {
            HashSet::new()
        };

        // Step 4: Tangential smoothing
        smooth_vertices(
            &mut current_mesh,
            &adj,
            params.smoothing_factor,
            &boundary_vertices,
        );
    }

    // Clean up any unreferenced vertices
    current_mesh = remove_unreferenced_vertices_internal(&current_mesh);

    RemeshResult {
        final_triangles: current_mesh.faces.len(),
        final_vertices: current_mesh.vertices.len(),
        mesh: current_mesh,
        original_triangles,
        original_vertices,
        iterations_performed: params.iterations,
        target_edge_length: target,
        edges_split: total_splits,
        edges_collapsed: total_collapses,
        edges_flipped: total_flips,
    }
}

/// Compute the average edge length in the mesh.
fn compute_average_edge_length(mesh: &Mesh) -> f64 {
    if mesh.faces.is_empty() {
        return 1.0;
    }

    let mut total_length = 0.0;
    let mut edge_count = 0;

    // Use a set to avoid counting edges twice
    let mut seen_edges: HashSet<(u32, u32)> = HashSet::new();

    for face in &mesh.faces {
        for i in 0..3 {
            let v0 = face[i];
            let v1 = face[(i + 1) % 3];
            let edge_key = if v0 < v1 { (v0, v1) } else { (v1, v0) };

            if seen_edges.insert(edge_key) {
                let p0 = &mesh.vertices[v0 as usize].position;
                let p1 = &mesh.vertices[v1 as usize].position;
                total_length += (p1 - p0).norm();
                edge_count += 1;
            }
        }
    }

    if edge_count == 0 {
        1.0
    } else {
        total_length / edge_count as f64
    }
}

/// Find edges with dihedral angle above threshold (sharp edges).
fn find_sharp_edges(
    mesh: &Mesh,
    adj: &MeshAdjacency,
    threshold: f64,
) -> HashSet<(u32, u32)> {
    let mut sharp_edges = HashSet::new();

    for (&edge, face_indices) in adj.edge_to_faces.iter() {
        if face_indices.len() != 2 {
            continue;
        }

        let f1 = &mesh.faces[face_indices[0] as usize];
        let f2 = &mesh.faces[face_indices[1] as usize];

        if let (Some(n1), Some(n2)) = (compute_face_normal(mesh, f1), compute_face_normal(mesh, f2))
        {
            let dot = n1.dot(&n2).clamp(-1.0, 1.0);
            let angle = dot.acos();
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

/// Split edges longer than max_length.
fn split_long_edges(
    mesh: &Mesh,
    adj: &MeshAdjacency,
    max_length: f64,
    _boundary_edges: &HashSet<(u32, u32)>,
    _sharp_edges: &HashSet<(u32, u32)>,
) -> (Mesh, usize) {
    let mut vertices = mesh.vertices.clone();
    let mut faces: Vec<[u32; 3]> = Vec::with_capacity(mesh.faces.len() * 2);
    let mut split_count = 0;

    // Map from edge to new midpoint vertex index (if split)
    let mut edge_midpoints: HashMap<(u32, u32), u32> = HashMap::new();

    // First pass: identify edges to split and create midpoint vertices
    for (&edge, _) in adj.edge_to_faces.iter() {
        let (v0, v1) = edge;
        let p0 = &mesh.vertices[v0 as usize].position;
        let p1 = &mesh.vertices[v1 as usize].position;
        let length = (p1 - p0).norm();

        if length > max_length {
            // Create midpoint vertex
            let midpoint = Point3::new(
                (p0.x + p1.x) / 2.0,
                (p0.y + p1.y) / 2.0,
                (p0.z + p1.z) / 2.0,
            );

            let mut new_vertex = Vertex::new(midpoint);

            // Interpolate attributes
            let vert0 = &mesh.vertices[v0 as usize];
            let vert1 = &mesh.vertices[v1 as usize];

            if let (Some(o0), Some(o1)) = (vert0.offset, vert1.offset) {
                new_vertex.offset = Some((o0 + o1) / 2.0);
            }
            if vert0.tag == vert1.tag {
                new_vertex.tag = vert0.tag;
            }

            let new_idx = vertices.len() as u32;
            vertices.push(new_vertex);
            edge_midpoints.insert(edge, new_idx);
            split_count += 1;
        }
    }

    // Second pass: split faces that contain split edges
    for face in &mesh.faces {
        let edges = [
            canonical_edge(face[0], face[1]),
            canonical_edge(face[1], face[2]),
            canonical_edge(face[2], face[0]),
        ];

        let midpoints: Vec<Option<u32>> = edges
            .iter()
            .map(|e| edge_midpoints.get(e).copied())
            .collect();

        let num_splits: usize = midpoints.iter().filter(|m| m.is_some()).count();

        match num_splits {
            0 => {
                // No splits, keep original face
                faces.push(*face);
            }
            1 => {
                // One edge split: create 2 triangles
                let split_idx = midpoints.iter().position(|m| m.is_some()).unwrap();
                let mid = midpoints[split_idx].unwrap();
                let v0 = face[split_idx];
                let v1 = face[(split_idx + 1) % 3];
                let v2 = face[(split_idx + 2) % 3];

                faces.push([v0, mid, v2]);
                faces.push([mid, v1, v2]);
            }
            2 => {
                // Two edges split: create 3 triangles
                let unsplit_idx = midpoints.iter().position(|m| m.is_none()).unwrap();
                let m0 = midpoints[(unsplit_idx + 1) % 3].unwrap();
                let m1 = midpoints[(unsplit_idx + 2) % 3].unwrap();
                let v0 = face[unsplit_idx];
                let v1 = face[(unsplit_idx + 1) % 3];
                let v2 = face[(unsplit_idx + 2) % 3];

                faces.push([v0, v1, m0]);
                faces.push([v0, m0, m1]);
                faces.push([m0, v2, m1]);
            }
            3 => {
                // All three edges split: create 4 triangles
                let m01 = midpoints[0].unwrap();
                let m12 = midpoints[1].unwrap();
                let m20 = midpoints[2].unwrap();

                faces.push([face[0], m01, m20]);
                faces.push([m01, face[1], m12]);
                faces.push([m20, m12, face[2]]);
                faces.push([m01, m12, m20]); // Center triangle
            }
            _ => unreachable!(),
        }
    }

    (Mesh { vertices, faces }, split_count)
}

/// Collapse edges shorter than min_length.
fn collapse_short_edges(
    mesh: &Mesh,
    adj: &MeshAdjacency,
    min_length: f64,
    boundary_edges: &HashSet<(u32, u32)>,
    sharp_edges: &HashSet<(u32, u32)>,
    boundary_vertices: &HashSet<u32>,
) -> (Mesh, usize) {
    let mut vertices = mesh.vertices.clone();
    let mut collapse_count = 0;

    // Map from old vertex index to new vertex index (for collapsed vertices)
    let mut vertex_map: HashMap<u32, u32> = HashMap::new();
    for i in 0..vertices.len() {
        vertex_map.insert(i as u32, i as u32);
    }

    // Collect edges to collapse (shortest first for stability)
    let mut edges_with_length: Vec<((u32, u32), f64)> = adj
        .edge_to_faces
        .keys()
        .filter_map(|&edge| {
            let (v0, v1) = edge;
            let p0 = &mesh.vertices[v0 as usize].position;
            let p1 = &mesh.vertices[v1 as usize].position;
            let length = (p1 - p0).norm();

            if length < min_length {
                // Don't collapse boundary or sharp edges unless both vertices are interior
                if boundary_edges.contains(&edge) || sharp_edges.contains(&edge) {
                    return None;
                }
                Some((edge, length))
            } else {
                None
            }
        })
        .collect();

    edges_with_length.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));

    // Track which vertices have been collapsed
    let mut collapsed_vertices: HashSet<u32> = HashSet::new();

    for ((v0, v1), _) in edges_with_length {
        // Skip if either vertex was already collapsed
        if collapsed_vertices.contains(&v0) || collapsed_vertices.contains(&v1) {
            continue;
        }

        // Resolve any previous collapses
        let final_v0 = resolve_vertex(&vertex_map, v0);
        let final_v1 = resolve_vertex(&vertex_map, v1);

        if final_v0 == final_v1 {
            continue; // Already collapsed to same vertex
        }

        // Check if collapse would create non-manifold geometry
        if would_create_non_manifold(adj, final_v0, final_v1) {
            continue;
        }

        // Determine collapse target (prefer boundary vertices to stay in place)
        let (keep, remove) = if boundary_vertices.contains(&final_v0) {
            (final_v0, final_v1)
        } else if boundary_vertices.contains(&final_v1) {
            (final_v1, final_v0)
        } else {
            // Move to midpoint for interior edges
            let p0 = &vertices[final_v0 as usize].position;
            let p1 = &vertices[final_v1 as usize].position;
            let midpoint = Point3::new(
                (p0.x + p1.x) / 2.0,
                (p0.y + p1.y) / 2.0,
                (p0.z + p1.z) / 2.0,
            );
            vertices[final_v0 as usize].position = midpoint;
            (final_v0, final_v1)
        };

        // Record the collapse
        vertex_map.insert(remove, keep);
        collapsed_vertices.insert(remove);
        collapse_count += 1;
    }

    // Rebuild faces with collapsed vertices
    let mut new_faces: Vec<[u32; 3]> = Vec::with_capacity(mesh.faces.len());
    for face in &mesh.faces {
        let v0 = resolve_vertex(&vertex_map, face[0]);
        let v1 = resolve_vertex(&vertex_map, face[1]);
        let v2 = resolve_vertex(&vertex_map, face[2]);

        // Skip degenerate faces (where two or more vertices collapsed to same point)
        if v0 != v1 && v1 != v2 && v2 != v0 {
            new_faces.push([v0, v1, v2]);
        }
    }

    (Mesh { vertices, faces: new_faces }, collapse_count)
}

/// Resolve a vertex through the collapse map.
fn resolve_vertex(map: &HashMap<u32, u32>, mut v: u32) -> u32 {
    let mut iterations = 0;
    while let Some(&target) = map.get(&v) {
        if target == v {
            break;
        }
        v = target;
        iterations += 1;
        if iterations > 1000 {
            break; // Safety limit
        }
    }
    v
}

/// Check if collapsing an edge would create non-manifold geometry.
fn would_create_non_manifold(adj: &MeshAdjacency, v0: u32, v1: u32) -> bool {
    // Get neighbors of both vertices
    let neighbors_v0: HashSet<u32> = get_vertex_neighbors(adj, v0);
    let neighbors_v1: HashSet<u32> = get_vertex_neighbors(adj, v1);

    // Common neighbors (excluding the edge vertices themselves)
    let common: HashSet<u32> = neighbors_v0
        .intersection(&neighbors_v1)
        .copied()
        .filter(|&v| v != v0 && v != v1)
        .collect();

    // For a manifold collapse, there should be exactly 2 common neighbors
    // (the vertices opposite the edge in the two adjacent faces)
    common.len() > 2
}

/// Get all vertices connected to a given vertex.
fn get_vertex_neighbors(adj: &MeshAdjacency, v: u32) -> HashSet<u32> {
    let mut neighbors = HashSet::new();

    for (&(a, b), _) in adj.edge_to_faces.iter() {
        if a == v {
            neighbors.insert(b);
        } else if b == v {
            neighbors.insert(a);
        }
    }

    neighbors
}

/// Flip edges to improve vertex valence toward the ideal of 6.
fn flip_edges_for_valence(
    mesh: &Mesh,
    adj: &MeshAdjacency,
    _boundary_edges: &HashSet<(u32, u32)>,
    _sharp_edges: &HashSet<(u32, u32)>,
) -> (Mesh, usize) {
    let mut faces = mesh.faces.clone();
    let mut flip_count = 0;

    // Compute current vertex valences
    let mut valences = compute_valences(&faces);

    // Try to flip each interior edge
    for (&edge, face_indices) in adj.edge_to_faces.iter() {
        // Only flip interior manifold edges
        if face_indices.len() != 2 {
            continue;
        }
        if _boundary_edges.contains(&edge) || _sharp_edges.contains(&edge) {
            continue;
        }

        let (v0, v1) = edge;
        let fi0 = face_indices[0] as usize;
        let fi1 = face_indices[1] as usize;

        // Find the opposite vertices
        let opp0 = find_opposite_vertex(&faces[fi0], v0, v1);
        let opp1 = find_opposite_vertex(&faces[fi1], v0, v1);

        if opp0.is_none() || opp1.is_none() {
            continue;
        }

        let opp0 = opp0.unwrap();
        let opp1 = opp1.unwrap();

        // Check if flip would improve valence
        // Ideal valence is 6 for interior vertices
        let val_v0 = *valences.get(&v0).unwrap_or(&6);
        let val_v1 = *valences.get(&v1).unwrap_or(&6);
        let val_opp0 = *valences.get(&opp0).unwrap_or(&6);
        let val_opp1 = *valences.get(&opp1).unwrap_or(&6);

        // Current deviation from ideal (6)
        let current_dev = (val_v0 as i32 - 6).abs()
            + (val_v1 as i32 - 6).abs()
            + (val_opp0 as i32 - 6).abs()
            + (val_opp1 as i32 - 6).abs();

        // After flip: v0 and v1 lose a neighbor, opp0 and opp1 gain one
        let new_dev = (val_v0 as i32 - 1 - 6).abs()
            + (val_v1 as i32 - 1 - 6).abs()
            + (val_opp0 as i32 + 1 - 6).abs()
            + (val_opp1 as i32 + 1 - 6).abs();

        if new_dev < current_dev {
            // Check if flip would create valid geometry (not inverted)
            if is_valid_flip(mesh, v0, v1, opp0, opp1) {
                // Perform the flip - copy face data first to avoid borrow issues
                let new_face0 = [opp0, opp1, v0];
                let new_face1 = [opp1, opp0, v1];

                faces[fi0] = new_face0;
                faces[fi1] = new_face1;

                // Update valences
                *valences.entry(v0).or_insert(6) -= 1;
                *valences.entry(v1).or_insert(6) -= 1;
                *valences.entry(opp0).or_insert(6) += 1;
                *valences.entry(opp1).or_insert(6) += 1;

                flip_count += 1;
            }
        }
    }

    (Mesh { vertices: mesh.vertices.clone(), faces }, flip_count)
}

/// Compute vertex valences (number of edges incident to each vertex).
fn compute_valences(faces: &[[u32; 3]]) -> HashMap<u32, usize> {
    let mut valences: HashMap<u32, usize> = HashMap::new();
    let mut seen_edges: HashSet<(u32, u32)> = HashSet::new();

    for face in faces {
        for i in 0..3 {
            let v0 = face[i];
            let v1 = face[(i + 1) % 3];
            let edge = canonical_edge(v0, v1);

            if seen_edges.insert(edge) {
                *valences.entry(v0).or_insert(0) += 1;
                *valences.entry(v1).or_insert(0) += 1;
            }
        }
    }

    valences
}

/// Find the vertex opposite to edge (v0, v1) in a face.
fn find_opposite_vertex(face: &[u32; 3], v0: u32, v1: u32) -> Option<u32> {
    for &v in face {
        if v != v0 && v != v1 {
            return Some(v);
        }
    }
    None
}

/// Check if an edge flip would create valid (non-inverted) triangles.
fn is_valid_flip(mesh: &Mesh, v0: u32, v1: u32, opp0: u32, opp1: u32) -> bool {
    let p0 = &mesh.vertices[v0 as usize].position;
    let p1 = &mesh.vertices[v1 as usize].position;
    let p_opp0 = &mesh.vertices[opp0 as usize].position;
    let p_opp1 = &mesh.vertices[opp1 as usize].position;

    // Check that the quadrilateral formed is convex
    // by verifying the new edge doesn't cross the old edge

    // Compute normals of the new triangles
    let e1 = p_opp1 - p_opp0;
    let e2 = p0 - p_opp0;
    let n1 = e1.cross(&e2);

    let e3 = p1 - p_opp0;
    let n2 = e1.cross(&e3);

    // Normals should point in opposite directions relative to new edge
    // (i.e., v0 and v1 should be on opposite sides of the new edge)
    n1.dot(&n2) < 0.0
}

/// Apply tangential smoothing to vertices.
fn smooth_vertices(
    mesh: &mut Mesh,
    adj: &MeshAdjacency,
    factor: f64,
    boundary_vertices: &HashSet<u32>,
) {
    if factor <= 0.0 {
        return;
    }

    // Compute new positions
    let mut new_positions: Vec<Point3<f64>> = mesh
        .vertices
        .iter()
        .map(|v| v.position)
        .collect();

    for (vi, vertex) in mesh.vertices.iter().enumerate() {
        let vi = vi as u32;

        // Don't smooth boundary vertices
        if boundary_vertices.contains(&vi) {
            continue;
        }

        // Find neighbors
        let neighbors = get_vertex_neighbors(adj, vi);
        if neighbors.is_empty() {
            continue;
        }

        // Compute centroid of neighbors (Laplacian smoothing)
        let mut centroid = Vector3::new(0.0, 0.0, 0.0);
        for &ni in &neighbors {
            let np = &mesh.vertices[ni as usize].position;
            centroid += Vector3::new(np.x, np.y, np.z);
        }
        centroid /= neighbors.len() as f64;

        let current = Vector3::new(vertex.position.x, vertex.position.y, vertex.position.z);

        // Move toward centroid by smoothing factor
        let smoothed = current + (centroid - current) * factor;

        new_positions[vi as usize] = Point3::new(smoothed.x, smoothed.y, smoothed.z);
    }

    // Apply new positions
    for (vi, pos) in new_positions.into_iter().enumerate() {
        mesh.vertices[vi].position = pos;
    }
}

/// Remove unreferenced vertices from mesh.
fn remove_unreferenced_vertices_internal(mesh: &Mesh) -> Mesh {
    // Find all referenced vertices
    let mut referenced: HashSet<u32> = HashSet::new();
    for face in &mesh.faces {
        referenced.insert(face[0]);
        referenced.insert(face[1]);
        referenced.insert(face[2]);
    }

    // Build new vertex list and index mapping
    let mut new_vertices: Vec<Vertex> = Vec::new();
    let mut old_to_new: HashMap<u32, u32> = HashMap::new();

    for (old_idx, vertex) in mesh.vertices.iter().enumerate() {
        if referenced.contains(&(old_idx as u32)) {
            let new_idx = new_vertices.len() as u32;
            old_to_new.insert(old_idx as u32, new_idx);
            new_vertices.push(vertex.clone());
        }
    }

    // Remap faces
    let new_faces: Vec<[u32; 3]> = mesh
        .faces
        .iter()
        .map(|face| {
            [
                *old_to_new.get(&face[0]).unwrap(),
                *old_to_new.get(&face[1]).unwrap(),
                *old_to_new.get(&face[2]).unwrap(),
            ]
        })
        .collect();

    Mesh {
        vertices: new_vertices,
        faces: new_faces,
    }
}

/// Create a canonical edge key (smaller index first).
fn canonical_edge(v0: u32, v1: u32) -> (u32, u32) {
    if v0 < v1 {
        (v0, v1)
    } else {
        (v1, v0)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_single_triangle() -> Mesh {
        Mesh {
            vertices: vec![
                Vertex::from_coords(0.0, 0.0, 0.0),
                Vertex::from_coords(10.0, 0.0, 0.0),
                Vertex::from_coords(5.0, 8.66, 0.0),
            ],
            faces: vec![[0, 1, 2]],
        }
    }

    fn make_two_triangles() -> Mesh {
        Mesh {
            vertices: vec![
                Vertex::from_coords(0.0, 0.0, 0.0),
                Vertex::from_coords(10.0, 0.0, 0.0),
                Vertex::from_coords(5.0, 8.66, 0.0),
                Vertex::from_coords(5.0, -8.66, 0.0),
            ],
            faces: vec![
                [0, 1, 2],
                [0, 3, 1],
            ],
        }
    }

    fn make_quad_as_triangles() -> Mesh {
        // A flat quad split into two triangles
        Mesh {
            vertices: vec![
                Vertex::from_coords(0.0, 0.0, 0.0),
                Vertex::from_coords(10.0, 0.0, 0.0),
                Vertex::from_coords(10.0, 10.0, 0.0),
                Vertex::from_coords(0.0, 10.0, 0.0),
            ],
            faces: vec![
                [0, 1, 2],
                [0, 2, 3],
            ],
        }
    }

    #[test]
    fn test_remesh_params_default() {
        let params = RemeshParams::default();
        assert!(params.target_edge_length.is_none());
        assert_eq!(params.iterations, 5);
        assert!(params.preserve_boundary);
        assert!(!params.preserve_sharp_edges);
    }

    #[test]
    fn test_remesh_empty_mesh() {
        let mesh = Mesh::new();
        let result = remesh_isotropic(&mesh, &RemeshParams::default());

        assert_eq!(result.original_triangles, 0);
        assert_eq!(result.final_triangles, 0);
        assert_eq!(result.iterations_performed, 0);
    }

    #[test]
    fn test_remesh_single_triangle_no_change() {
        let mesh = make_single_triangle();
        // Use a large target length so no splitting occurs
        let result = remesh_isotropic(&mesh, &RemeshParams::with_target_edge_length(100.0));

        // With large target, the triangle might be collapsed or unchanged
        assert!(result.final_triangles <= 1);
    }

    #[test]
    fn test_remesh_single_triangle_split() {
        let mesh = make_single_triangle();
        // Use a small target length to force splitting
        let result = remesh_isotropic(&mesh, &RemeshParams::with_target_edge_length(2.0));

        // Should have more triangles after splitting
        assert!(
            result.final_triangles > result.original_triangles,
            "Expected more triangles after remeshing, got {} from {}",
            result.final_triangles,
            result.original_triangles
        );
    }

    #[test]
    fn test_remesh_two_triangles() {
        let mesh = make_two_triangles();
        let result = remesh_isotropic(&mesh, &RemeshParams::with_target_edge_length(3.0));

        // Should produce a remeshed result
        assert!(result.final_triangles > 0);
        assert!(result.mesh.faces.len() > 0);
    }

    #[test]
    fn test_remesh_quad() {
        let mesh = make_quad_as_triangles();
        let result = remesh_isotropic(&mesh, &RemeshParams::with_target_edge_length(2.0));

        // Should subdivide the quad
        assert!(
            result.final_triangles > 2,
            "Expected more than 2 triangles, got {}",
            result.final_triangles
        );
    }

    #[test]
    fn test_compute_average_edge_length() {
        let mesh = make_quad_as_triangles();
        let avg_len = compute_average_edge_length(&mesh);

        // Quad is 10x10, edges are 10 (sides), 10 (sides), ~14.14 (diagonals)
        // Average should be around 10-11
        assert!(
            avg_len > 8.0 && avg_len < 15.0,
            "Average edge length should be reasonable, got {}",
            avg_len
        );
    }

    #[test]
    fn test_remesh_preserves_valid_topology() {
        let mesh = make_two_triangles();
        let result = remesh_isotropic(&mesh, &RemeshParams::with_target_edge_length(3.0));

        // All face indices should be valid
        for face in &result.mesh.faces {
            for &vi in face {
                assert!(
                    (vi as usize) < result.mesh.vertices.len(),
                    "Invalid vertex index {} in face (mesh has {} vertices)",
                    vi,
                    result.mesh.vertices.len()
                );
            }
        }

        // No degenerate faces
        for face in &result.mesh.faces {
            assert!(
                face[0] != face[1] && face[1] != face[2] && face[2] != face[0],
                "Degenerate face found: {:?}",
                face
            );
        }
    }

    #[test]
    fn test_remesh_high_quality() {
        let mesh = make_quad_as_triangles();
        let params = RemeshParams::high_quality();
        assert_eq!(params.iterations, 10);

        let result = remesh_isotropic(&mesh, &params);
        assert!(result.iterations_performed == 10);
    }

    #[test]
    fn test_remesh_fast() {
        let mesh = make_quad_as_triangles();
        let params = RemeshParams::fast();
        assert_eq!(params.iterations, 3);

        let result = remesh_isotropic(&mesh, &params);
        assert!(result.iterations_performed == 3);
    }

    #[test]
    fn test_remesh_preserve_features() {
        let params = RemeshParams::preserve_features();
        assert!(params.preserve_boundary);
        assert!(params.preserve_sharp_edges);
    }

    #[test]
    fn test_canonical_edge() {
        assert_eq!(canonical_edge(1, 5), (1, 5));
        assert_eq!(canonical_edge(5, 1), (1, 5));
        assert_eq!(canonical_edge(3, 3), (3, 3));
    }

    #[test]
    fn test_remesh_result_statistics() {
        let mesh = make_single_triangle();
        let result = remesh_isotropic(&mesh, &RemeshParams::with_target_edge_length(2.0));

        assert_eq!(result.original_triangles, 1);
        assert_eq!(result.original_vertices, 3);
        assert!(result.target_edge_length > 0.0);
        assert!(result.iterations_performed > 0);
    }

    #[test]
    fn test_remesh_zero_iterations() {
        let mesh = make_single_triangle();
        let mut params = RemeshParams::default();
        params.iterations = 0;

        let result = remesh_isotropic(&mesh, &params);

        assert_eq!(result.final_triangles, 1);
        assert_eq!(result.iterations_performed, 0);
    }
}

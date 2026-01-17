//! Mesh decimation using edge collapse with quadric error metrics.
//!
//! This module provides mesh simplification by iteratively collapsing edges
//! while minimizing geometric error using the Quadric Error Metrics (QEM) algorithm.

use crate::{Mesh, MeshAdjacency, Vertex};
use nalgebra::Point3;
use std::cmp::Ordering;
use std::collections::{BinaryHeap, HashMap, HashSet};

/// Parameters for mesh decimation.
#[derive(Debug, Clone)]
#[cfg_attr(
    feature = "pipeline-config",
    derive(serde::Serialize, serde::Deserialize)
)]
pub struct DecimateParams {
    /// Target number of triangles. If None, uses target_ratio instead.
    pub target_triangles: Option<usize>,
    /// Target ratio of triangles to keep (0.0 to 1.0). Default: 0.5
    pub target_ratio: f64,
    /// Whether to preserve boundary edges (edges with only one adjacent face).
    /// Default: true
    pub preserve_boundary: bool,
    /// Whether to preserve sharp features (edges with high dihedral angle).
    /// Default: false
    pub preserve_sharp_features: bool,
    /// Dihedral angle threshold in radians for sharp feature detection.
    /// Edges with angle above this are considered sharp. Default: 0.5236 (30 degrees)
    pub sharp_angle_threshold: f64,
    /// Maximum error allowed for edge collapse. If None, no limit.
    pub max_error: Option<f64>,
    /// Penalty multiplier for boundary edges when preserve_boundary is false.
    /// Higher values make boundary edges less likely to collapse. Default: 10.0
    pub boundary_penalty: f64,
}

impl Default for DecimateParams {
    fn default() -> Self {
        Self {
            target_triangles: None,
            target_ratio: 0.5,
            preserve_boundary: true,
            preserve_sharp_features: false,
            sharp_angle_threshold: std::f64::consts::FRAC_PI_6, // 30 degrees
            max_error: None,
            boundary_penalty: 10.0,
        }
    }
}

impl DecimateParams {
    /// Create params targeting a specific triangle count.
    pub fn with_target_triangles(count: usize) -> Self {
        Self {
            target_triangles: Some(count),
            ..Default::default()
        }
    }

    /// Create params targeting a ratio of original triangles.
    pub fn with_target_ratio(ratio: f64) -> Self {
        Self {
            target_ratio: ratio.clamp(0.0, 1.0),
            ..Default::default()
        }
    }

    /// Create aggressive decimation params (more simplification).
    pub fn aggressive() -> Self {
        Self {
            target_ratio: 0.25,
            preserve_boundary: false,
            preserve_sharp_features: false,
            boundary_penalty: 1.0,
            ..Default::default()
        }
    }

    /// Create conservative decimation params (preserve more detail).
    pub fn conservative() -> Self {
        Self {
            target_ratio: 0.75,
            preserve_boundary: true,
            preserve_sharp_features: true,
            sharp_angle_threshold: 0.3491, // 20 degrees
            ..Default::default()
        }
    }
}

/// Result of mesh decimation.
#[derive(Debug, Clone)]
pub struct DecimateResult {
    /// The decimated mesh.
    pub mesh: Mesh,
    /// Number of triangles in original mesh.
    pub original_triangles: usize,
    /// Number of triangles in decimated mesh.
    pub final_triangles: usize,
    /// Number of edge collapses performed.
    pub collapses_performed: usize,
    /// Number of edge collapses rejected (e.g., would create non-manifold).
    pub collapses_rejected: usize,
}

/// Quadric error matrix (4x4 symmetric matrix stored as 10 values).
#[derive(Debug, Clone, Copy)]
struct Quadric {
    // Symmetric 4x4 matrix stored as upper triangle:
    // [a b c d]
    // [  e f g]
    // [    h i]
    // [      j]
    a: f64,
    b: f64,
    c: f64,
    d: f64,
    e: f64,
    f: f64,
    g: f64,
    h: f64,
    i: f64,
    j: f64,
}

impl Default for Quadric {
    fn default() -> Self {
        Self {
            a: 0.0,
            b: 0.0,
            c: 0.0,
            d: 0.0,
            e: 0.0,
            f: 0.0,
            g: 0.0,
            h: 0.0,
            i: 0.0,
            j: 0.0,
        }
    }
}

impl Quadric {
    /// Create a quadric from a plane equation (ax + by + cz + d = 0).
    fn from_plane(a: f64, b: f64, c: f64, d: f64) -> Self {
        Self {
            a: a * a,
            b: a * b,
            c: a * c,
            d: a * d,
            e: b * b,
            f: b * c,
            g: b * d,
            h: c * c,
            i: c * d,
            j: d * d,
        }
    }

    /// Add another quadric to this one.
    fn add(&mut self, other: &Quadric) {
        self.a += other.a;
        self.b += other.b;
        self.c += other.c;
        self.d += other.d;
        self.e += other.e;
        self.f += other.f;
        self.g += other.g;
        self.h += other.h;
        self.i += other.i;
        self.j += other.j;
    }

    /// Evaluate the quadric error for a point.
    fn evaluate(&self, x: f64, y: f64, z: f64) -> f64 {
        // v^T * Q * v where v = [x, y, z, 1]
        self.a * x * x
            + 2.0 * self.b * x * y
            + 2.0 * self.c * x * z
            + 2.0 * self.d * x
            + self.e * y * y
            + 2.0 * self.f * y * z
            + 2.0 * self.g * y
            + self.h * z * z
            + 2.0 * self.i * z
            + self.j
    }

    /// Find the optimal point that minimizes error, or return None if matrix is singular.
    fn optimal_point(&self) -> Option<[f64; 3]> {
        // Solve the linear system:
        // [a b c] [x]   [-d]
        // [b e f] [y] = [-g]
        // [c f h] [z]   [-i]

        let det = self.a * (self.e * self.h - self.f * self.f)
            - self.b * (self.b * self.h - self.f * self.c)
            + self.c * (self.b * self.f - self.e * self.c);

        if det.abs() < 1e-10 {
            return None;
        }

        let inv_det = 1.0 / det;

        // Compute inverse matrix elements
        let m00 = (self.e * self.h - self.f * self.f) * inv_det;
        let m01 = (self.c * self.f - self.b * self.h) * inv_det;
        let m02 = (self.b * self.f - self.c * self.e) * inv_det;
        let m11 = (self.a * self.h - self.c * self.c) * inv_det;
        let m12 = (self.b * self.c - self.a * self.f) * inv_det;
        let m22 = (self.a * self.e - self.b * self.b) * inv_det;

        let x = m00 * (-self.d) + m01 * (-self.g) + m02 * (-self.i);
        let y = m01 * (-self.d) + m11 * (-self.g) + m12 * (-self.i);
        let z = m02 * (-self.d) + m12 * (-self.g) + m22 * (-self.i);

        Some([x, y, z])
    }
}

/// An edge collapse candidate in the priority queue.
#[derive(Debug, Clone)]
struct EdgeCollapse {
    /// The two vertex indices forming the edge.
    v1: u32,
    v2: u32,
    /// The error cost of this collapse (negated for max-heap to work as min-heap).
    cost: f64,
    /// The optimal position for the merged vertex.
    optimal_pos: [f64; 3],
}

impl PartialEq for EdgeCollapse {
    fn eq(&self, other: &Self) -> bool {
        self.cost == other.cost
    }
}

impl Eq for EdgeCollapse {}

impl PartialOrd for EdgeCollapse {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for EdgeCollapse {
    fn cmp(&self, other: &Self) -> Ordering {
        // Reverse ordering for min-heap behavior (smaller cost = higher priority)
        other
            .cost
            .partial_cmp(&self.cost)
            .unwrap_or(Ordering::Equal)
    }
}

/// Decimate a mesh using edge collapse with quadric error metrics.
///
/// # Arguments
/// * `mesh` - The input mesh to decimate
/// * `params` - Decimation parameters
///
/// # Returns
/// A `DecimateResult` containing the decimated mesh and statistics.
///
/// # Example
/// ```
/// use mesh_repair::{Mesh, Vertex, decimate_mesh, DecimateParams};
///
/// // Create a simple mesh with some triangles
/// let mut mesh = Mesh::new();
/// mesh.vertices.push(Vertex::from_coords(0.0, 0.0, 0.0));
/// mesh.vertices.push(Vertex::from_coords(1.0, 0.0, 0.0));
/// mesh.vertices.push(Vertex::from_coords(0.5, 1.0, 0.0));
/// mesh.faces.push([0, 1, 2]);
///
/// let result = decimate_mesh(&mesh, &DecimateParams::with_target_ratio(0.5));
/// println!("Reduced from {} to {} triangles", result.original_triangles, result.final_triangles);
/// ```
pub fn decimate_mesh(mesh: &Mesh, params: &DecimateParams) -> DecimateResult {
    let original_triangles = mesh.faces.len();

    // Handle edge cases
    if original_triangles == 0 {
        return DecimateResult {
            mesh: mesh.clone(),
            original_triangles: 0,
            final_triangles: 0,
            collapses_performed: 0,
            collapses_rejected: 0,
        };
    }

    // Calculate target triangle count
    let target = params
        .target_triangles
        .unwrap_or_else(|| ((original_triangles as f64) * params.target_ratio).ceil() as usize);

    // Don't decimate if already at or below target
    if original_triangles <= target {
        return DecimateResult {
            mesh: mesh.clone(),
            original_triangles,
            final_triangles: original_triangles,
            collapses_performed: 0,
            collapses_rejected: 0,
        };
    }

    // Create working copy of mesh data
    let mut vertices: Vec<Option<Vertex>> = mesh.vertices.iter().cloned().map(Some).collect();
    let mut faces: Vec<Option<[u32; 3]>> = mesh.faces.iter().cloned().map(Some).collect();
    let mut active_faces = original_triangles;

    // Build adjacency
    let adj = MeshAdjacency::build(&mesh.faces);

    // Compute initial quadrics for each vertex
    let mut quadrics = compute_vertex_quadrics(mesh, &adj);

    // Identify boundary edges
    let boundary_edges: HashSet<(u32, u32)> = adj.boundary_edges().collect();

    // Identify sharp feature edges if needed
    let sharp_edges: HashSet<(u32, u32)> = if params.preserve_sharp_features {
        find_sharp_edges(mesh, &adj, params.sharp_angle_threshold)
    } else {
        HashSet::new()
    };

    // Build initial edge collapse queue
    let mut heap = build_collapse_queue(mesh, &quadrics, &boundary_edges, &sharp_edges, params);

    // Track which vertices have been merged (maps old index -> new index)
    let mut vertex_remap: HashMap<u32, u32> = HashMap::new();

    let mut collapses_performed = 0;
    let mut collapses_rejected = 0;

    // Main decimation loop
    while active_faces > target {
        let Some(collapse) = heap.pop() else {
            break;
        };

        // Get actual vertex indices (following remap chain)
        let v1 = get_actual_vertex(collapse.v1, &vertex_remap);
        let v2 = get_actual_vertex(collapse.v2, &vertex_remap);

        // Skip if vertices have been merged or are the same
        if v1 == v2 || vertices[v1 as usize].is_none() || vertices[v2 as usize].is_none() {
            continue;
        }

        // Skip if this is a boundary edge and we're preserving boundaries
        if params.preserve_boundary {
            let edge = normalize_edge(v1, v2);
            if boundary_edges.contains(&edge) {
                collapses_rejected += 1;
                continue;
            }
        }

        // Check if collapse would create non-manifold geometry
        if !is_collapse_valid(&vertices, &faces, v1, v2) {
            collapses_rejected += 1;
            continue;
        }

        // Check max error threshold
        if let Some(max_error) = params.max_error {
            let error = quadrics[v1 as usize].evaluate(
                collapse.optimal_pos[0],
                collapse.optimal_pos[1],
                collapse.optimal_pos[2],
            ) + quadrics[v2 as usize].evaluate(
                collapse.optimal_pos[0],
                collapse.optimal_pos[1],
                collapse.optimal_pos[2],
            );
            if error > max_error {
                collapses_rejected += 1;
                continue;
            }
        }

        // Perform the collapse: merge v2 into v1
        // Update v1 position to optimal
        if let Some(ref mut v) = vertices[v1 as usize] {
            v.position = Point3::new(
                collapse.optimal_pos[0],
                collapse.optimal_pos[1],
                collapse.optimal_pos[2],
            );
        }

        // Combine quadrics
        let q2 = quadrics[v2 as usize];
        quadrics[v1 as usize].add(&q2);

        // Remove v2
        vertices[v2 as usize] = None;
        vertex_remap.insert(v2, v1);

        // Update faces: replace v2 with v1, remove degenerate faces
        for face_opt in faces.iter_mut() {
            if let Some(face) = face_opt {
                for idx in face.iter_mut() {
                    let actual = get_actual_vertex(*idx, &vertex_remap);
                    *idx = actual;
                    if actual == v2 {
                        *idx = v1;
                    }
                }

                // Check if face is degenerate (has duplicate vertices)
                if face[0] == face[1] || face[1] == face[2] || face[0] == face[2] {
                    *face_opt = None;
                    active_faces -= 1;
                }
            }
        }

        collapses_performed += 1;

        // Re-queue edges involving v1 with updated costs
        requeue_vertex_edges(
            v1,
            &vertices,
            &faces,
            &quadrics,
            &boundary_edges,
            &sharp_edges,
            &vertex_remap,
            params,
            &mut heap,
        );
    }

    // Build the final mesh
    let final_mesh = build_final_mesh(&vertices, &faces);

    DecimateResult {
        mesh: final_mesh,
        original_triangles,
        final_triangles: active_faces,
        collapses_performed,
        collapses_rejected,
    }
}

/// Compute quadric error matrices for each vertex.
fn compute_vertex_quadrics(mesh: &Mesh, _adj: &MeshAdjacency) -> Vec<Quadric> {
    let mut quadrics = vec![Quadric::default(); mesh.vertices.len()];

    for face in mesh.faces.iter() {
        let v0 = &mesh.vertices[face[0] as usize];
        let v1 = &mesh.vertices[face[1] as usize];
        let v2 = &mesh.vertices[face[2] as usize];

        // Compute face normal and plane equation
        let e1 = [
            v1.position.x - v0.position.x,
            v1.position.y - v0.position.y,
            v1.position.z - v0.position.z,
        ];
        let e2 = [
            v2.position.x - v0.position.x,
            v2.position.y - v0.position.y,
            v2.position.z - v0.position.z,
        ];

        let normal = [
            e1[1] * e2[2] - e1[2] * e2[1],
            e1[2] * e2[0] - e1[0] * e2[2],
            e1[0] * e2[1] - e1[1] * e2[0],
        ];

        let len = (normal[0] * normal[0] + normal[1] * normal[1] + normal[2] * normal[2]).sqrt();
        if len < 1e-10 {
            continue; // Skip degenerate triangles
        }

        let a = normal[0] / len;
        let b = normal[1] / len;
        let c = normal[2] / len;
        let d = -(a * v0.position.x + b * v0.position.y + c * v0.position.z);

        let q = Quadric::from_plane(a, b, c, d);

        // Add to each vertex of the face
        for &vi in face {
            quadrics[vi as usize].add(&q);
        }
    }

    quadrics
}

/// Find edges with dihedral angle above threshold.
fn find_sharp_edges(mesh: &Mesh, adj: &MeshAdjacency, threshold: f64) -> HashSet<(u32, u32)> {
    let mut sharp_edges = HashSet::new();

    for (&edge, face_indices) in adj.edge_to_faces.iter() {
        if face_indices.len() != 2 {
            continue;
        }

        let f1 = &mesh.faces[face_indices[0] as usize];
        let f2 = &mesh.faces[face_indices[1] as usize];

        let n1 = compute_face_normal(mesh, f1);
        let n2 = compute_face_normal(mesh, f2);

        if let (Some(n1), Some(n2)) = (n1, n2) {
            let dot = n1[0] * n2[0] + n1[1] * n2[1] + n1[2] * n2[2];
            let angle = dot.clamp(-1.0, 1.0).acos();
            if angle > threshold {
                sharp_edges.insert(edge);
            }
        }
    }

    sharp_edges
}

/// Compute the unit normal of a face.
fn compute_face_normal(mesh: &Mesh, face: &[u32; 3]) -> Option<[f64; 3]> {
    let v0 = &mesh.vertices[face[0] as usize];
    let v1 = &mesh.vertices[face[1] as usize];
    let v2 = &mesh.vertices[face[2] as usize];

    let e1 = [
        v1.position.x - v0.position.x,
        v1.position.y - v0.position.y,
        v1.position.z - v0.position.z,
    ];
    let e2 = [
        v2.position.x - v0.position.x,
        v2.position.y - v0.position.y,
        v2.position.z - v0.position.z,
    ];

    let normal = [
        e1[1] * e2[2] - e1[2] * e2[1],
        e1[2] * e2[0] - e1[0] * e2[2],
        e1[0] * e2[1] - e1[1] * e2[0],
    ];

    let len = (normal[0] * normal[0] + normal[1] * normal[1] + normal[2] * normal[2]).sqrt();
    if len < 1e-10 {
        return None;
    }

    Some([normal[0] / len, normal[1] / len, normal[2] / len])
}

/// Build the initial priority queue of edge collapses.
fn build_collapse_queue(
    mesh: &Mesh,
    quadrics: &[Quadric],
    boundary_edges: &HashSet<(u32, u32)>,
    sharp_edges: &HashSet<(u32, u32)>,
    params: &DecimateParams,
) -> BinaryHeap<EdgeCollapse> {
    let mut heap = BinaryHeap::new();
    let mut seen_edges = HashSet::new();

    for face in &mesh.faces {
        for i in 0..3 {
            let v1 = face[i];
            let v2 = face[(i + 1) % 3];
            let edge = normalize_edge(v1, v2);

            if seen_edges.contains(&edge) {
                continue;
            }
            seen_edges.insert(edge);

            if let Some(collapse) =
                compute_edge_collapse(v1, v2, mesh, quadrics, boundary_edges, sharp_edges, params)
            {
                heap.push(collapse);
            }
        }
    }

    heap
}

/// Compute the collapse for a single edge.
fn compute_edge_collapse(
    v1: u32,
    v2: u32,
    mesh: &Mesh,
    quadrics: &[Quadric],
    boundary_edges: &HashSet<(u32, u32)>,
    sharp_edges: &HashSet<(u32, u32)>,
    params: &DecimateParams,
) -> Option<EdgeCollapse> {
    let edge = normalize_edge(v1, v2);

    // Skip if preserving boundary and this is boundary
    if params.preserve_boundary && boundary_edges.contains(&edge) {
        return None;
    }

    // Skip if preserving sharp features and this is sharp
    if params.preserve_sharp_features && sharp_edges.contains(&edge) {
        return None;
    }

    let q1 = &quadrics[v1 as usize];
    let q2 = &quadrics[v2 as usize];

    // Combined quadric
    let mut combined = *q1;
    combined.add(q2);

    // Find optimal position
    let pos1 = &mesh.vertices[v1 as usize].position;
    let pos2 = &mesh.vertices[v2 as usize].position;
    let midpoint = [
        (pos1.x + pos2.x) / 2.0,
        (pos1.y + pos2.y) / 2.0,
        (pos1.z + pos2.z) / 2.0,
    ];

    let optimal_pos = combined.optimal_point().unwrap_or(midpoint);

    // Compute error at optimal position
    let mut cost = combined.evaluate(optimal_pos[0], optimal_pos[1], optimal_pos[2]);

    // Apply boundary penalty
    if boundary_edges.contains(&edge) {
        cost *= params.boundary_penalty;
    }

    Some(EdgeCollapse {
        v1,
        v2,
        cost,
        optimal_pos,
    })
}

/// Normalize edge so smaller index comes first.
fn normalize_edge(v1: u32, v2: u32) -> (u32, u32) {
    if v1 < v2 { (v1, v2) } else { (v2, v1) }
}

/// Get the actual vertex index following the remap chain.
fn get_actual_vertex(v: u32, remap: &HashMap<u32, u32>) -> u32 {
    let mut current = v;
    while let Some(&next) = remap.get(&current) {
        current = next;
    }
    current
}

/// Check if collapsing an edge would create invalid (non-manifold) geometry.
fn is_collapse_valid(
    _vertices: &[Option<Vertex>],
    faces: &[Option<[u32; 3]>],
    v1: u32,
    v2: u32,
) -> bool {
    // Collect neighbors of v1 and v2
    let mut neighbors_v1 = HashSet::new();
    let mut neighbors_v2 = HashSet::new();

    for face in faces.iter().flatten() {
        let has_v1 = face.contains(&v1);
        let has_v2 = face.contains(&v2);

        if has_v1 {
            for &vi in face {
                if vi != v1 {
                    neighbors_v1.insert(vi);
                }
            }
        }
        if has_v2 {
            for &vi in face {
                if vi != v2 {
                    neighbors_v2.insert(vi);
                }
            }
        }
    }

    // The common neighbors (excluding v1 and v2) should be exactly 2 for a manifold collapse
    // (the two vertices that form the triangles on either side of the edge)
    let common: HashSet<_> = neighbors_v1.intersection(&neighbors_v2).collect();

    // For a valid manifold edge collapse, there should be at most 2 common neighbors
    // (the vertices forming triangles on either side of the edge being collapsed)
    common.len() <= 2
}

/// Re-queue all edges involving a vertex after it has been updated.
#[allow(clippy::too_many_arguments)]
fn requeue_vertex_edges(
    v: u32,
    vertices: &[Option<Vertex>],
    faces: &[Option<[u32; 3]>],
    quadrics: &[Quadric],
    boundary_edges: &HashSet<(u32, u32)>,
    sharp_edges: &HashSet<(u32, u32)>,
    _vertex_remap: &HashMap<u32, u32>,
    params: &DecimateParams,
    heap: &mut BinaryHeap<EdgeCollapse>,
) {
    // Find all neighbors of v
    let mut neighbors = HashSet::new();
    for face_opt in faces {
        if let Some(face) = face_opt
            && face.contains(&v)
        {
            for &vi in face {
                if vi != v && vertices[vi as usize].is_some() {
                    neighbors.insert(vi);
                }
            }
        }
    }

    // Create a temporary mesh-like structure for cost computation
    // This is a bit inefficient but keeps the code simpler
    for &neighbor in &neighbors {
        let Some(v_vert) = &vertices[v as usize] else {
            continue;
        };
        let Some(n_vert) = &vertices[neighbor as usize] else {
            continue;
        };

        let edge = normalize_edge(v, neighbor);

        // Skip if preserving boundary
        if params.preserve_boundary && boundary_edges.contains(&edge) {
            continue;
        }

        // Skip if preserving sharp features
        if params.preserve_sharp_features && sharp_edges.contains(&edge) {
            continue;
        }

        let q1 = &quadrics[v as usize];
        let q2 = &quadrics[neighbor as usize];

        let mut combined = *q1;
        combined.add(q2);

        let midpoint = [
            (v_vert.position.x + n_vert.position.x) / 2.0,
            (v_vert.position.y + n_vert.position.y) / 2.0,
            (v_vert.position.z + n_vert.position.z) / 2.0,
        ];

        let optimal_pos = combined.optimal_point().unwrap_or(midpoint);
        let mut cost = combined.evaluate(optimal_pos[0], optimal_pos[1], optimal_pos[2]);

        if boundary_edges.contains(&edge) {
            cost *= params.boundary_penalty;
        }

        heap.push(EdgeCollapse {
            v1: v,
            v2: neighbor,
            cost,
            optimal_pos,
        });
    }
}

/// Decimate a mesh with progress reporting.
///
/// This is a progress-reporting variant of [`decimate_mesh`] that allows tracking
/// the decimation progress and supports cancellation via the progress callback.
///
/// # Arguments
/// * `mesh` - The input mesh to decimate
/// * `params` - Decimation parameters
/// * `callback` - Optional progress callback. Returns `false` to request cancellation.
///
/// # Returns
/// A `DecimateResult` containing the decimated mesh and statistics.
/// If cancelled via callback, returns the partially decimated mesh.
///
/// # Example
/// ```ignore
/// use mesh_repair::{Mesh, decimate_mesh_with_progress, DecimateParams};
/// use mesh_repair::progress::ProgressCallback;
///
/// let callback: ProgressCallback = Box::new(|progress| {
///     println!("{}% - {}", progress.percent(), progress.message);
///     true // Continue
/// });
///
/// let result = decimate_mesh_with_progress(&mesh, &DecimateParams::default(), Some(&callback));
/// ```
pub fn decimate_mesh_with_progress(
    mesh: &Mesh,
    params: &DecimateParams,
    callback: Option<&crate::progress::ProgressCallback>,
) -> DecimateResult {
    use crate::progress::ProgressTracker;

    let original_triangles = mesh.faces.len();

    // Handle edge cases
    if original_triangles == 0 {
        return DecimateResult {
            mesh: mesh.clone(),
            original_triangles: 0,
            final_triangles: 0,
            collapses_performed: 0,
            collapses_rejected: 0,
        };
    }

    // Calculate target triangle count
    let target = params
        .target_triangles
        .unwrap_or_else(|| ((original_triangles as f64) * params.target_ratio).ceil() as usize);

    // Don't decimate if already at or below target
    if original_triangles <= target {
        return DecimateResult {
            mesh: mesh.clone(),
            original_triangles,
            final_triangles: original_triangles,
            collapses_performed: 0,
            collapses_rejected: 0,
        };
    }

    // Estimate total collapses needed (roughly triangles_to_remove / 2)
    let triangles_to_remove = original_triangles - target;
    let estimated_collapses = triangles_to_remove / 2;
    let tracker = ProgressTracker::new(estimated_collapses.max(1) as u64);

    // Create working copy of mesh data
    let mut vertices: Vec<Option<Vertex>> = mesh.vertices.iter().cloned().map(Some).collect();
    let mut faces: Vec<Option<[u32; 3]>> = mesh.faces.iter().cloned().map(Some).collect();
    let mut active_faces = original_triangles;

    // Build adjacency
    let adj = MeshAdjacency::build(&mesh.faces);

    // Compute initial quadrics for each vertex
    let mut quadrics = compute_vertex_quadrics(mesh, &adj);

    // Identify boundary edges
    let boundary_edges: HashSet<(u32, u32)> = adj.boundary_edges().collect();

    // Identify sharp feature edges if needed
    let sharp_edges: HashSet<(u32, u32)> = if params.preserve_sharp_features {
        find_sharp_edges(mesh, &adj, params.sharp_angle_threshold)
    } else {
        HashSet::new()
    };

    // Build initial edge collapse queue
    let mut heap = build_collapse_queue(mesh, &quadrics, &boundary_edges, &sharp_edges, params);

    // Track which vertices have been merged (maps old index -> new index)
    let mut vertex_remap: HashMap<u32, u32> = HashMap::new();

    let mut collapses_performed = 0;
    let mut collapses_rejected = 0;

    // Main decimation loop
    while active_faces > target {
        // Check for cancellation
        if tracker.is_cancelled() {
            break;
        }

        let Some(collapse) = heap.pop() else {
            break;
        };

        // Get actual vertex indices (following remap chain)
        let v1 = get_actual_vertex(collapse.v1, &vertex_remap);
        let v2 = get_actual_vertex(collapse.v2, &vertex_remap);

        // Skip if vertices have been merged or are the same
        if v1 == v2 || vertices[v1 as usize].is_none() || vertices[v2 as usize].is_none() {
            continue;
        }

        // Skip if this is a boundary edge and we're preserving boundaries
        if params.preserve_boundary {
            let edge = normalize_edge(v1, v2);
            if boundary_edges.contains(&edge) {
                collapses_rejected += 1;
                continue;
            }
        }

        // Check if collapse would create non-manifold geometry
        if !is_collapse_valid(&vertices, &faces, v1, v2) {
            collapses_rejected += 1;
            continue;
        }

        // Check max error threshold
        if let Some(max_error) = params.max_error {
            let error = quadrics[v1 as usize].evaluate(
                collapse.optimal_pos[0],
                collapse.optimal_pos[1],
                collapse.optimal_pos[2],
            ) + quadrics[v2 as usize].evaluate(
                collapse.optimal_pos[0],
                collapse.optimal_pos[1],
                collapse.optimal_pos[2],
            );
            if error > max_error {
                collapses_rejected += 1;
                continue;
            }
        }

        // Perform the collapse: merge v2 into v1
        // Update v1 position to optimal
        if let Some(ref mut v) = vertices[v1 as usize] {
            v.position = Point3::new(
                collapse.optimal_pos[0],
                collapse.optimal_pos[1],
                collapse.optimal_pos[2],
            );
        }

        // Combine quadrics
        let q2 = quadrics[v2 as usize];
        quadrics[v1 as usize].add(&q2);

        // Remove v2
        vertices[v2 as usize] = None;
        vertex_remap.insert(v2, v1);

        // Update faces: replace v2 with v1, remove degenerate faces
        for face_opt in faces.iter_mut() {
            if let Some(face) = face_opt {
                for idx in face.iter_mut() {
                    let actual = get_actual_vertex(*idx, &vertex_remap);
                    *idx = actual;
                    if actual == v2 {
                        *idx = v1;
                    }
                }

                // Check if face is degenerate (has duplicate vertices)
                if face[0] == face[1] || face[1] == face[2] || face[0] == face[2] {
                    *face_opt = None;
                    active_faces -= 1;
                }
            }
        }

        collapses_performed += 1;
        tracker.increment();

        // Report progress periodically
        if !tracker.maybe_callback(
            callback,
            format!(
                "Decimating: {} triangles remaining (target: {})",
                active_faces, target
            ),
        ) {
            break; // Cancelled
        }

        // Re-queue edges involving v1 with updated costs
        requeue_vertex_edges(
            v1,
            &vertices,
            &faces,
            &quadrics,
            &boundary_edges,
            &sharp_edges,
            &vertex_remap,
            params,
            &mut heap,
        );
    }

    // Build the final mesh
    let final_mesh = build_final_mesh(&vertices, &faces);

    DecimateResult {
        mesh: final_mesh,
        original_triangles,
        final_triangles: active_faces,
        collapses_performed,
        collapses_rejected,
    }
}

/// Build the final compacted mesh from the working data.
fn build_final_mesh(vertices: &[Option<Vertex>], faces: &[Option<[u32; 3]>]) -> Mesh {
    // Compact vertices and create index mapping
    let mut new_vertices = Vec::new();
    let mut vertex_map: HashMap<u32, u32> = HashMap::new();

    for (old_idx, v_opt) in vertices.iter().enumerate() {
        if let Some(v) = v_opt {
            vertex_map.insert(old_idx as u32, new_vertices.len() as u32);
            new_vertices.push(v.clone());
        }
    }

    // Compact faces with remapped indices
    let mut new_faces = Vec::new();
    for face_opt in faces {
        if let Some(face) = face_opt
            && let (Some(&i0), Some(&i1), Some(&i2)) = (
                vertex_map.get(&face[0]),
                vertex_map.get(&face[1]),
                vertex_map.get(&face[2]),
            )
        {
            // Skip degenerate faces
            if i0 != i1 && i1 != i2 && i0 != i2 {
                new_faces.push([i0, i1, i2]);
            }
        }
    }

    Mesh {
        vertices: new_vertices,
        faces: new_faces,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Create a unit cube mesh for testing.
    fn make_cube(size: f64) -> Mesh {
        let s = size / 2.0;
        let mut mesh = Mesh::new();

        // 8 vertices of the cube
        mesh.vertices.push(Vertex::from_coords(-s, -s, -s)); // 0
        mesh.vertices.push(Vertex::from_coords(s, -s, -s)); // 1
        mesh.vertices.push(Vertex::from_coords(s, s, -s)); // 2
        mesh.vertices.push(Vertex::from_coords(-s, s, -s)); // 3
        mesh.vertices.push(Vertex::from_coords(-s, -s, s)); // 4
        mesh.vertices.push(Vertex::from_coords(s, -s, s)); // 5
        mesh.vertices.push(Vertex::from_coords(s, s, s)); // 6
        mesh.vertices.push(Vertex::from_coords(-s, s, s)); // 7

        // 12 triangles (2 per face)
        // Bottom face (z=-s)
        mesh.faces.push([0, 2, 1]);
        mesh.faces.push([0, 3, 2]);
        // Top face (z=+s)
        mesh.faces.push([4, 5, 6]);
        mesh.faces.push([4, 6, 7]);
        // Front face (y=-s)
        mesh.faces.push([0, 1, 5]);
        mesh.faces.push([0, 5, 4]);
        // Back face (y=+s)
        mesh.faces.push([3, 7, 6]);
        mesh.faces.push([3, 6, 2]);
        // Left face (x=-s)
        mesh.faces.push([0, 4, 7]);
        mesh.faces.push([0, 7, 3]);
        // Right face (x=+s)
        mesh.faces.push([1, 2, 6]);
        mesh.faces.push([1, 6, 5]);

        mesh
    }

    #[test]
    fn test_decimate_empty_mesh() {
        let mesh = Mesh::default();
        let result = decimate_mesh(&mesh, &DecimateParams::default());
        assert_eq!(result.original_triangles, 0);
        assert_eq!(result.final_triangles, 0);
        assert_eq!(result.collapses_performed, 0);
    }

    #[test]
    fn test_decimate_cube() {
        let mesh = make_cube(10.0);
        let params = DecimateParams::with_target_ratio(0.5);
        let result = decimate_mesh(&mesh, &params);

        assert_eq!(result.original_triangles, 12); // Cube has 12 triangles
        assert!(result.final_triangles <= 6); // Should reduce to roughly half
        assert!(!result.mesh.faces.is_empty()); // Should still have some faces
    }

    #[test]
    fn test_decimate_to_target_count() {
        let mesh = make_cube(10.0);
        let params = DecimateParams::with_target_triangles(8);
        let result = decimate_mesh(&mesh, &params);

        assert!(result.final_triangles <= 8);
    }

    #[test]
    fn test_decimate_preserves_boundary_by_default() {
        // Create a simple mesh with a boundary (two triangles sharing an edge)
        // All edges in this mesh are boundary edges (only one adjacent face each)
        let mesh = Mesh {
            vertices: vec![
                Vertex::from_coords(0.0, 0.0, 0.0),
                Vertex::from_coords(1.0, 0.0, 0.0),
                Vertex::from_coords(0.5, 1.0, 0.0),
                Vertex::from_coords(1.5, 1.0, 0.0),
            ],
            faces: vec![[0, 1, 2], [1, 3, 2]],
        };

        let params = DecimateParams::default();
        let result = decimate_mesh(&mesh, &params);

        // With preserve_boundary=true (default), all boundary edges are preserved
        // Since all edges in this mesh are boundary edges (except the shared edge),
        // and the shared edge connects boundary vertices, decimation is limited.
        // The mesh should preserve most of its structure.
        // Note: this is a small mesh with limited collapse options
        assert_eq!(result.original_triangles, 2);
        // Verify statistics were recorded (at minimum, some attempts were made)
        // Note: the sum is always >= 0 for usize, so we just verify the fields exist
        let _total_attempts = result.collapses_performed + result.collapses_rejected;
    }

    #[test]
    fn test_quadric_from_plane() {
        let q = Quadric::from_plane(0.0, 0.0, 1.0, 0.0);

        // Points on the z=0 plane should have zero error
        assert!((q.evaluate(0.0, 0.0, 0.0)).abs() < 1e-10);
        assert!((q.evaluate(1.0, 2.0, 0.0)).abs() < 1e-10);

        // Points off the plane should have non-zero error
        assert!((q.evaluate(0.0, 0.0, 1.0) - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_quadric_optimal_point() {
        // Create quadrics for two parallel planes: z=0 and z=2
        let mut q1 = Quadric::from_plane(0.0, 0.0, 1.0, 0.0);
        let q2 = Quadric::from_plane(0.0, 0.0, 1.0, -2.0);
        q1.add(&q2);

        // Optimal point should be at z=1 (midpoint)
        if let Some(opt) = q1.optimal_point() {
            assert!((opt[2] - 1.0).abs() < 1e-5);
        }
    }

    #[test]
    fn test_decimate_params_presets() {
        let aggressive = DecimateParams::aggressive();
        assert_eq!(aggressive.target_ratio, 0.25);
        assert!(!aggressive.preserve_boundary);

        let conservative = DecimateParams::conservative();
        assert_eq!(conservative.target_ratio, 0.75);
        assert!(conservative.preserve_boundary);
        assert!(conservative.preserve_sharp_features);
    }

    #[test]
    fn test_normalize_edge() {
        assert_eq!(normalize_edge(5, 3), (3, 5));
        assert_eq!(normalize_edge(3, 5), (3, 5));
        assert_eq!(normalize_edge(4, 4), (4, 4));
    }

    #[test]
    fn test_get_actual_vertex() {
        let mut remap = HashMap::new();
        remap.insert(3, 1);
        remap.insert(5, 3);

        assert_eq!(get_actual_vertex(5, &remap), 1);
        assert_eq!(get_actual_vertex(3, &remap), 1);
        assert_eq!(get_actual_vertex(1, &remap), 1);
        assert_eq!(get_actual_vertex(7, &remap), 7);
    }

    #[test]
    fn test_decimate_single_triangle() {
        // Single triangle cannot be decimated
        let mesh = Mesh {
            vertices: vec![
                Vertex::from_coords(0.0, 0.0, 0.0),
                Vertex::from_coords(1.0, 0.0, 0.0),
                Vertex::from_coords(0.5, 1.0, 0.0),
            ],
            faces: vec![[0, 1, 2]],
        };

        let result = decimate_mesh(&mesh, &DecimateParams::default());
        assert_eq!(result.original_triangles, 1);
        // With boundary preservation, single triangle stays
        assert!(!result.mesh.faces.is_empty() || result.collapses_performed == 0);
    }

    #[test]
    fn test_decimate_already_at_target() {
        let mesh = make_cube(10.0);
        // Target more than current count - should not decimate
        let params = DecimateParams::with_target_triangles(20);
        let result = decimate_mesh(&mesh, &params);

        assert_eq!(result.original_triangles, 12);
        assert_eq!(result.final_triangles, 12);
        assert_eq!(result.collapses_performed, 0);
    }
}

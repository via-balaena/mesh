//! Isotropic and adaptive remeshing for uniform edge lengths and improved triangle quality.
//!
//! This module provides several remeshing algorithms:
//! - **Isotropic remeshing**: Create meshes with uniform edge lengths and well-shaped triangles
//! - **Curvature-adaptive remeshing**: Smaller triangles in high-curvature regions, larger in flat areas
//! - **Feature-preserving remeshing**: Detect and preserve sharp edges during remeshing
//! - **Anisotropic remeshing**: Align triangles with principal curvature directions
//!
//! Useful for scanned meshes with uneven tessellation or when preparing meshes for simulation.

use hashbrown::{HashMap, HashSet};
use nalgebra::{Matrix3, Point3, Vector3};
use rayon::prelude::*;

use crate::{Mesh, MeshAdjacency, Vertex};

/// Parameters for isotropic remeshing.
#[derive(Debug, Clone)]
#[cfg_attr(
    feature = "pipeline-config",
    derive(serde::Serialize, serde::Deserialize)
)]
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

    /// Enable curvature-adaptive remeshing.
    /// When true, edge lengths vary based on local curvature.
    /// Default: false
    pub adaptive_to_curvature: bool,

    /// Minimum curvature threshold for adaptive remeshing.
    /// Below this curvature, use max_edge_length_adaptive.
    /// Default: 0.01 (nearly flat)
    pub curvature_min_threshold: f64,

    /// Maximum curvature threshold for adaptive remeshing.
    /// Above this curvature, use min_edge_length_adaptive.
    /// Default: 1.0 (highly curved)
    pub curvature_max_threshold: f64,

    /// Minimum edge length for adaptive remeshing (in high curvature areas).
    /// Default: None (uses target * 0.25)
    pub min_edge_length_adaptive: Option<f64>,

    /// Maximum edge length for adaptive remeshing (in low curvature areas).
    /// Default: None (uses target * 2.0)
    pub max_edge_length_adaptive: Option<f64>,

    /// Enable anisotropic remeshing.
    /// When true, triangles are aligned with principal curvature directions.
    /// Default: false
    pub anisotropic: bool,

    /// Anisotropy ratio (max edge length / min edge length in different directions).
    /// Higher values create more elongated triangles.
    /// Default: 2.0
    pub anisotropy_ratio: f64,

    /// Custom direction field for anisotropic remeshing.
    /// Maps vertex index to preferred direction vector.
    /// If None, uses principal curvature directions.
    #[cfg_attr(feature = "pipeline-config", serde(skip))]
    pub direction_field: Option<HashMap<u32, Vector3<f64>>>,

    /// Preserve feature edges during remeshing (overrides preserve_sharp_edges).
    /// When set, these specific edges will be preserved regardless of dihedral angle.
    #[cfg_attr(feature = "pipeline-config", serde(skip))]
    pub preserve_feature_edges: Option<HashSet<(u32, u32)>>,
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
            adaptive_to_curvature: false,
            curvature_min_threshold: 0.01,
            curvature_max_threshold: 1.0,
            min_edge_length_adaptive: None,
            max_edge_length_adaptive: None,
            anisotropic: false,
            anisotropy_ratio: 2.0,
            direction_field: None,
            preserve_feature_edges: None,
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

    /// Create params for curvature-adaptive remeshing.
    ///
    /// This creates smaller triangles in high-curvature regions (detailed areas)
    /// and larger triangles in flat regions (to reduce triangle count).
    ///
    /// # Example
    /// ```
    /// use mesh_repair::RemeshParams;
    ///
    /// let params = RemeshParams::adaptive(2.0); // 2.0mm base target edge length
    /// ```
    pub fn adaptive(target_edge_length: f64) -> Self {
        Self {
            target_edge_length: Some(target_edge_length),
            adaptive_to_curvature: true,
            preserve_sharp_edges: true,
            ..Default::default()
        }
    }

    /// Create params for anisotropic remeshing.
    ///
    /// This aligns triangles with the principal curvature directions,
    /// creating elongated triangles that better follow surface features.
    ///
    /// # Example
    /// ```
    /// use mesh_repair::RemeshParams;
    ///
    /// let params = RemeshParams::anisotropic_with_ratio(2.0, 3.0);
    /// ```
    pub fn anisotropic_with_ratio(target_edge_length: f64, anisotropy_ratio: f64) -> Self {
        Self {
            target_edge_length: Some(target_edge_length),
            anisotropic: true,
            anisotropy_ratio,
            preserve_sharp_edges: true,
            ..Default::default()
        }
    }

    /// Enable curvature-adaptive remeshing on existing params.
    pub fn with_curvature_adaptation(mut self) -> Self {
        self.adaptive_to_curvature = true;
        self
    }

    /// Set curvature thresholds for adaptive remeshing.
    pub fn with_curvature_thresholds(mut self, min: f64, max: f64) -> Self {
        self.curvature_min_threshold = min;
        self.curvature_max_threshold = max;
        self
    }

    /// Set adaptive edge length range.
    pub fn with_adaptive_edge_range(mut self, min_length: f64, max_length: f64) -> Self {
        self.min_edge_length_adaptive = Some(min_length);
        self.max_edge_length_adaptive = Some(max_length);
        self
    }

    /// Enable anisotropic remeshing on existing params.
    pub fn with_anisotropy(mut self, ratio: f64) -> Self {
        self.anisotropic = true;
        self.anisotropy_ratio = ratio;
        self
    }

    /// Set a custom direction field for anisotropic remeshing.
    pub fn with_direction_field(mut self, field: HashMap<u32, Vector3<f64>>) -> Self {
        self.direction_field = Some(field);
        self
    }

    /// Set specific feature edges to preserve.
    pub fn with_feature_edges(mut self, edges: HashSet<(u32, u32)>) -> Self {
        self.preserve_feature_edges = Some(edges);
        self
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
    /// Number of feature edges detected (if feature detection was enabled).
    pub feature_edges_detected: usize,
    /// Whether curvature-adaptive remeshing was used.
    pub adaptive_enabled: bool,
    /// Whether anisotropic remeshing was used.
    pub anisotropic_enabled: bool,
}

/// A detected feature edge with its properties.
#[derive(Debug, Clone)]
pub struct FeatureEdge {
    /// The edge vertices (canonical order: smaller index first).
    pub edge: (u32, u32),
    /// The dihedral angle at this edge (in radians).
    pub dihedral_angle: f64,
    /// Whether this is a boundary edge.
    pub is_boundary: bool,
    /// Whether this is a sharp edge (dihedral angle above threshold).
    pub is_sharp: bool,
}

/// Result of feature edge detection.
#[derive(Debug)]
pub struct FeatureEdgeResult {
    /// All detected feature edges.
    pub edges: Vec<FeatureEdge>,
    /// Sharp edges only.
    pub sharp_edges: HashSet<(u32, u32)>,
    /// Boundary edges.
    pub boundary_edges: HashSet<(u32, u32)>,
    /// Mean dihedral angle across all non-boundary edges.
    pub mean_dihedral_angle: f64,
    /// Maximum dihedral angle found.
    pub max_dihedral_angle: f64,
}

/// Per-vertex curvature information.
#[derive(Debug, Clone, Default)]
pub struct VertexCurvature {
    /// Mean curvature (average of principal curvatures).
    pub mean: f64,
    /// Gaussian curvature (product of principal curvatures).
    pub gaussian: f64,
    /// Maximum principal curvature.
    pub k1: f64,
    /// Minimum principal curvature.
    pub k2: f64,
    /// Direction of maximum principal curvature.
    pub dir1: Vector3<f64>,
    /// Direction of minimum principal curvature.
    pub dir2: Vector3<f64>,
}

/// Result of curvature computation.
#[derive(Debug)]
pub struct CurvatureResult {
    /// Per-vertex curvature values.
    pub vertex_curvatures: Vec<VertexCurvature>,
    /// Minimum mean curvature across the mesh.
    pub min_mean_curvature: f64,
    /// Maximum mean curvature across the mesh.
    pub max_mean_curvature: f64,
    /// Average mean curvature across the mesh.
    pub avg_mean_curvature: f64,
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

    // Dispatch to appropriate remeshing algorithm based on params
    if params.adaptive_to_curvature {
        return remesh_adaptive(mesh, params);
    }
    if params.anisotropic {
        return remesh_anisotropic(mesh, params);
    }

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
            feature_edges_detected: 0,
            adaptive_enabled: false,
            anisotropic_enabled: false,
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

        let _boundary_vertices: HashSet<u32> =
            boundary_edges.iter().flat_map(|&(a, b)| [a, b]).collect();

        // Step 1: Split long edges
        let (new_mesh, splits) = split_long_edges(
            &current_mesh,
            &adj,
            max_length,
            &boundary_edges,
            &sharp_edges,
        );
        current_mesh = new_mesh;
        total_splits += splits;

        // Rebuild adjacency after splits
        let adj = MeshAdjacency::build(&current_mesh.faces);
        let boundary_edges: HashSet<(u32, u32)> = if params.preserve_boundary {
            adj.boundary_edges().collect()
        } else {
            HashSet::new()
        };
        let boundary_vertices: HashSet<u32> =
            boundary_edges.iter().flat_map(|&(a, b)| [a, b]).collect();

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
        let (new_mesh, flips) =
            flip_edges_for_valence(&current_mesh, &adj, &boundary_edges, &sharp_edges);
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

    // Count feature edges if preserving them
    let feature_edge_count = if params.preserve_sharp_edges {
        let adj = MeshAdjacency::build(&current_mesh.faces);
        find_sharp_edges(&current_mesh, &adj, params.sharp_angle_threshold).len()
    } else {
        0
    };

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
        feature_edges_detected: feature_edge_count,
        adaptive_enabled: false,
        anisotropic_enabled: false,
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
fn find_sharp_edges(mesh: &Mesh, adj: &MeshAdjacency, threshold: f64) -> HashSet<(u32, u32)> {
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

    (
        Mesh {
            vertices,
            faces: new_faces,
        },
        collapse_count,
    )
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

    (
        Mesh {
            vertices: mesh.vertices.clone(),
            faces,
        },
        flip_count,
    )
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
    face.iter().find(|&&v| v != v0 && v != v1).copied()
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
///
/// Uses parallel iteration via rayon for improved performance on large meshes.
fn smooth_vertices(
    mesh: &mut Mesh,
    adj: &MeshAdjacency,
    factor: f64,
    boundary_vertices: &HashSet<u32>,
) {
    if factor <= 0.0 {
        return;
    }

    // Compute new positions in parallel
    // Each vertex's new position depends only on reading neighbor positions (no writes)
    let vertices_ref = &mesh.vertices;
    let new_positions: Vec<Point3<f64>> = mesh
        .vertices
        .par_iter()
        .enumerate()
        .map(|(vi, vertex)| {
            let vi = vi as u32;

            // Don't smooth boundary vertices - keep original position
            if boundary_vertices.contains(&vi) {
                return vertex.position;
            }

            // Find neighbors
            let neighbors = get_vertex_neighbors(adj, vi);
            if neighbors.is_empty() {
                return vertex.position;
            }

            // Compute centroid of neighbors (Laplacian smoothing)
            let mut centroid = Vector3::new(0.0, 0.0, 0.0);
            for &ni in &neighbors {
                let np = &vertices_ref[ni as usize].position;
                centroid += Vector3::new(np.x, np.y, np.z);
            }
            centroid /= neighbors.len() as f64;

            let current = Vector3::new(vertex.position.x, vertex.position.y, vertex.position.z);

            // Move toward centroid by smoothing factor
            let smoothed = current + (centroid - current) * factor;

            Point3::new(smoothed.x, smoothed.y, smoothed.z)
        })
        .collect();

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
    if v0 < v1 { (v0, v1) } else { (v1, v0) }
}

// ============================================================================
// Feature Edge Detection
// ============================================================================

/// Detect feature edges in a mesh.
///
/// Feature edges are edges that lie on sharp corners, creases, or boundaries.
/// They are important for preserving the geometric character of a mesh during
/// remeshing operations.
///
/// # Arguments
/// * `mesh` - The input mesh
/// * `sharp_angle_threshold` - Dihedral angle (in radians) above which an edge is considered sharp.
///   Default is PI/3 (60 degrees).
///
/// # Returns
/// A `FeatureEdgeResult` containing all detected feature edges and statistics.
///
/// # Example
/// ```
/// use mesh_repair::{Mesh, Vertex, detect_feature_edges};
/// use std::f64::consts::PI;
///
/// let mut mesh = Mesh::new();
/// mesh.vertices.push(Vertex::from_coords(0.0, 0.0, 0.0));
/// mesh.vertices.push(Vertex::from_coords(1.0, 0.0, 0.0));
/// mesh.vertices.push(Vertex::from_coords(0.5, 1.0, 0.0));
/// mesh.vertices.push(Vertex::from_coords(0.5, 0.0, 1.0));
/// mesh.faces.push([0, 1, 2]);
/// mesh.faces.push([0, 3, 1]);
///
/// let result = detect_feature_edges(&mesh, PI / 4.0); // 45 degrees
/// println!("Found {} sharp edges", result.sharp_edges.len());
/// ```
pub fn detect_feature_edges(mesh: &Mesh, sharp_angle_threshold: f64) -> FeatureEdgeResult {
    let adj = MeshAdjacency::build(&mesh.faces);

    let mut edges = Vec::new();
    let mut sharp_edges = HashSet::new();
    let mut boundary_edges = HashSet::new();
    let mut total_angle = 0.0;
    let mut interior_edge_count = 0;
    let mut max_angle = 0.0f64;

    for (&edge, face_indices) in adj.edge_to_faces.iter() {
        let is_boundary = face_indices.len() == 1;

        if is_boundary {
            boundary_edges.insert(edge);
            edges.push(FeatureEdge {
                edge,
                dihedral_angle: std::f64::consts::PI, // Boundary edges treated as max angle
                is_boundary: true,
                is_sharp: true, // Boundaries are always features
            });
            continue;
        }

        if face_indices.len() != 2 {
            // Non-manifold edge - treat as feature
            sharp_edges.insert(edge);
            edges.push(FeatureEdge {
                edge,
                dihedral_angle: std::f64::consts::PI,
                is_boundary: false,
                is_sharp: true,
            });
            continue;
        }

        // Compute dihedral angle
        let f1 = &mesh.faces[face_indices[0] as usize];
        let f2 = &mesh.faces[face_indices[1] as usize];

        if let (Some(n1), Some(n2)) = (compute_face_normal(mesh, f1), compute_face_normal(mesh, f2))
        {
            let dot = n1.dot(&n2).clamp(-1.0, 1.0);
            let angle = dot.acos();

            total_angle += angle;
            interior_edge_count += 1;
            max_angle = max_angle.max(angle);

            let is_sharp = angle > sharp_angle_threshold;
            if is_sharp {
                sharp_edges.insert(edge);
            }

            edges.push(FeatureEdge {
                edge,
                dihedral_angle: angle,
                is_boundary: false,
                is_sharp,
            });
        }
    }

    let mean_dihedral_angle = if interior_edge_count > 0 {
        total_angle / interior_edge_count as f64
    } else {
        0.0
    };

    FeatureEdgeResult {
        edges,
        sharp_edges,
        boundary_edges,
        mean_dihedral_angle,
        max_dihedral_angle: max_angle,
    }
}

// ============================================================================
// Curvature Computation
// ============================================================================

/// Compute per-vertex curvature for a mesh.
///
/// Uses the discrete curvature approximation based on the shape operator.
/// For each vertex, computes:
/// - Mean curvature (H)
/// - Gaussian curvature (K)
/// - Principal curvatures (k1, k2)
/// - Principal curvature directions (dir1, dir2)
///
/// # Arguments
/// * `mesh` - The input mesh
///
/// # Returns
/// A `CurvatureResult` containing per-vertex curvature values and statistics.
///
/// # Example
/// ```
/// use mesh_repair::{Mesh, Vertex, compute_curvature};
///
/// let mut mesh = Mesh::new();
/// // Create a simple mesh...
/// mesh.vertices.push(Vertex::from_coords(0.0, 0.0, 0.0));
/// mesh.vertices.push(Vertex::from_coords(1.0, 0.0, 0.0));
/// mesh.vertices.push(Vertex::from_coords(0.5, 1.0, 0.0));
/// mesh.faces.push([0, 1, 2]);
///
/// let result = compute_curvature(&mesh);
/// println!("Mean curvature range: {} to {}", result.min_mean_curvature, result.max_mean_curvature);
/// ```
pub fn compute_curvature(mesh: &Mesh) -> CurvatureResult {
    let adj = MeshAdjacency::build(&mesh.faces);
    let vertex_count = mesh.vertices.len();

    let mut vertex_curvatures = Vec::with_capacity(vertex_count);
    let mut min_mean = f64::MAX;
    let mut max_mean = f64::MIN;
    let mut total_mean = 0.0;

    // Compute vertex normals for later use
    let vertex_normals = compute_vertex_normals_internal(mesh, &adj);

    for vi in 0..vertex_count {
        let curv = compute_vertex_curvature(mesh, &adj, vi as u32, &vertex_normals);

        if curv.mean.is_finite() {
            min_mean = min_mean.min(curv.mean.abs());
            max_mean = max_mean.max(curv.mean.abs());
            total_mean += curv.mean.abs();
        }

        vertex_curvatures.push(curv);
    }

    let avg_mean = if vertex_count > 0 {
        total_mean / vertex_count as f64
    } else {
        0.0
    };

    CurvatureResult {
        vertex_curvatures,
        min_mean_curvature: if min_mean == f64::MAX { 0.0 } else { min_mean },
        max_mean_curvature: if max_mean == f64::MIN { 0.0 } else { max_mean },
        avg_mean_curvature: avg_mean,
    }
}

/// Compute curvature for a single vertex.
fn compute_vertex_curvature(
    mesh: &Mesh,
    adj: &MeshAdjacency,
    vertex_idx: u32,
    vertex_normals: &[Vector3<f64>],
) -> VertexCurvature {
    let neighbors = get_vertex_neighbors(adj, vertex_idx);

    if neighbors.is_empty() {
        return VertexCurvature::default();
    }

    let p = &mesh.vertices[vertex_idx as usize].position;
    let normal = &vertex_normals[vertex_idx as usize];

    // Build local coordinate frame
    let (tangent1, tangent2) = build_tangent_frame(normal);

    // Compute shape operator via edge-based curvature
    // Using the method from Meyer et al. "Discrete Differential-Geometry Operators"
    let mut shape_matrix = Matrix3::zeros();
    let mut total_weight = 0.0;

    for &ni in &neighbors {
        let np = &mesh.vertices[ni as usize].position;
        let edge = np - p;
        let edge_len = edge.norm();

        if edge_len < 1e-10 {
            continue;
        }

        // Project edge onto tangent plane
        let edge_normalized = edge / edge_len;
        let edge_tangent = edge_normalized - normal * normal.dot(&edge_normalized);
        let edge_tangent_len = edge_tangent.norm();

        if edge_tangent_len < 1e-10 {
            continue;
        }

        // Compute curvature along this edge
        let nn = &vertex_normals[ni as usize];
        let normal_diff = nn - normal;
        let kappa = -normal_diff.dot(&edge_normalized) / edge_len;

        // Weight by edge length (cotangent weights would be better but more complex)
        let weight = edge_len;
        total_weight += weight;

        // Add contribution to shape matrix
        let edge_2d = Vector3::new(
            edge_tangent.dot(&tangent1),
            edge_tangent.dot(&tangent2),
            0.0,
        );
        let edge_2d_normalized = edge_2d / edge_2d.norm().max(1e-10);

        for i in 0..2 {
            for j in 0..2 {
                shape_matrix[(i, j)] +=
                    weight * kappa * edge_2d_normalized[i] * edge_2d_normalized[j];
            }
        }
    }

    if total_weight < 1e-10 {
        return VertexCurvature::default();
    }

    // Normalize by total weight
    shape_matrix /= total_weight;

    // Extract 2x2 submatrix for eigenvalue decomposition
    let shape_2x2 = Matrix3::new(
        shape_matrix[(0, 0)],
        shape_matrix[(0, 1)],
        0.0,
        shape_matrix[(1, 0)],
        shape_matrix[(1, 1)],
        0.0,
        0.0,
        0.0,
        0.0,
    );

    // Compute eigenvalues (principal curvatures) using 2x2 formula
    let a = shape_2x2[(0, 0)];
    let b = shape_2x2[(0, 1)];
    let c = shape_2x2[(1, 0)];
    let d = shape_2x2[(1, 1)];

    let trace = a + d;
    let det = a * d - b * c;

    let discriminant = (trace * trace - 4.0 * det).max(0.0);
    let sqrt_disc = discriminant.sqrt();

    let k1 = (trace + sqrt_disc) / 2.0; // Max principal curvature
    let k2 = (trace - sqrt_disc) / 2.0; // Min principal curvature

    // Compute eigenvectors (principal directions)
    let (dir1, dir2) = if (a - k1).abs() > 1e-10 || b.abs() > 1e-10 {
        let ev1 = if b.abs() > 1e-10 {
            Vector3::new(b, k1 - a, 0.0).normalize()
        } else if (a - k1).abs() > 1e-10 {
            Vector3::new(k1 - d, c, 0.0).normalize()
        } else {
            Vector3::new(1.0, 0.0, 0.0)
        };

        let ev2 = if b.abs() > 1e-10 {
            Vector3::new(b, k2 - a, 0.0).normalize()
        } else if (a - k2).abs() > 1e-10 {
            Vector3::new(k2 - d, c, 0.0).normalize()
        } else {
            Vector3::new(0.0, 1.0, 0.0)
        };

        // Convert back to 3D
        let d1 = tangent1 * ev1.x + tangent2 * ev1.y;
        let d2 = tangent1 * ev2.x + tangent2 * ev2.y;
        (d1, d2)
    } else {
        (tangent1, tangent2)
    };

    VertexCurvature {
        mean: (k1 + k2) / 2.0,
        gaussian: k1 * k2,
        k1,
        k2,
        dir1,
        dir2,
    }
}

/// Build an orthonormal tangent frame from a normal vector.
fn build_tangent_frame(normal: &Vector3<f64>) -> (Vector3<f64>, Vector3<f64>) {
    // Find a vector not parallel to normal
    let up = if normal.x.abs() < 0.9 {
        Vector3::new(1.0, 0.0, 0.0)
    } else {
        Vector3::new(0.0, 1.0, 0.0)
    };

    let tangent1 = normal.cross(&up).normalize();
    let tangent2 = normal.cross(&tangent1);

    (tangent1, tangent2)
}

/// Compute vertex normals from face normals (area-weighted average).
fn compute_vertex_normals_internal(mesh: &Mesh, _adj: &MeshAdjacency) -> Vec<Vector3<f64>> {
    let mut normals = vec![Vector3::zeros(); mesh.vertices.len()];

    for face in &mesh.faces {
        if compute_face_normal(mesh, face).is_some() {
            // Weight by face area (implicit in unnormalized cross product)
            let v0 = &mesh.vertices[face[0] as usize].position;
            let v1 = &mesh.vertices[face[1] as usize].position;
            let v2 = &mesh.vertices[face[2] as usize].position;

            let e1 = v1 - v0;
            let e2 = v2 - v0;
            let area_normal = e1.cross(&e2);

            for &vi in face {
                normals[vi as usize] += area_normal;
            }
        }
    }

    // Normalize
    for normal in &mut normals {
        let len = normal.norm();
        if len > 1e-10 {
            *normal /= len;
        } else {
            *normal = Vector3::new(0.0, 0.0, 1.0); // Default up
        }
    }

    normals
}

// ============================================================================
// Curvature-Adaptive Remeshing
// ============================================================================

/// Perform curvature-adaptive remeshing on a mesh.
///
/// This creates smaller triangles in high-curvature regions and larger triangles
/// in flat regions, resulting in better detail preservation with fewer triangles.
///
/// # Arguments
/// * `mesh` - The input mesh
/// * `params` - Remeshing parameters with `adaptive_to_curvature` enabled
///
/// # Returns
/// A `RemeshResult` containing the remeshed mesh and statistics.
///
/// # Example
/// ```
/// use mesh_repair::{Mesh, Vertex, remesh_adaptive, RemeshParams};
///
/// let mut mesh = Mesh::new();
/// mesh.vertices.push(Vertex::from_coords(0.0, 0.0, 0.0));
/// mesh.vertices.push(Vertex::from_coords(10.0, 0.0, 0.0));
/// mesh.vertices.push(Vertex::from_coords(5.0, 8.66, 0.0));
/// mesh.faces.push([0, 1, 2]);
///
/// let result = remesh_adaptive(&mesh, &RemeshParams::adaptive(2.0));
/// println!("Adaptive remeshing produced {} triangles", result.final_triangles);
/// ```
pub fn remesh_adaptive(mesh: &Mesh, params: &RemeshParams) -> RemeshResult {
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
            feature_edges_detected: 0,
            adaptive_enabled: true,
            anisotropic_enabled: params.anisotropic,
        };
    }

    // Compute curvature for the mesh
    let curvature_result = compute_curvature(mesh);

    // Determine base target edge length
    let base_target = params
        .target_edge_length
        .unwrap_or_else(|| compute_average_edge_length(mesh));

    let min_adaptive = params
        .min_edge_length_adaptive
        .unwrap_or(base_target * 0.25);
    let max_adaptive = params.max_edge_length_adaptive.unwrap_or(base_target * 2.0);

    // Compute per-vertex target edge lengths based on curvature
    let vertex_targets = compute_adaptive_edge_lengths(
        &curvature_result,
        base_target,
        min_adaptive,
        max_adaptive,
        params.curvature_min_threshold,
        params.curvature_max_threshold,
    );

    let mut current_mesh = mesh.clone();
    let mut total_splits = 0;
    let mut total_collapses = 0;
    let mut total_flips = 0;
    let mut feature_edge_count = 0;

    for _iter in 0..params.iterations {
        // Build adjacency
        let adj = MeshAdjacency::build(&current_mesh.faces);

        // Identify protected edges
        let boundary_edges: HashSet<(u32, u32)> = if params.preserve_boundary {
            adj.boundary_edges().collect()
        } else {
            HashSet::new()
        };

        let sharp_edges: HashSet<(u32, u32)> = if params.preserve_sharp_edges {
            let feature_result = detect_feature_edges(&current_mesh, params.sharp_angle_threshold);
            feature_edge_count = feature_result.sharp_edges.len();
            feature_result.sharp_edges
        } else if let Some(ref custom_edges) = params.preserve_feature_edges {
            feature_edge_count = custom_edges.len();
            custom_edges.clone()
        } else {
            HashSet::new()
        };

        // Step 1: Split long edges (adaptive)
        let (new_mesh, splits) = split_long_edges_adaptive(
            &current_mesh,
            &adj,
            &vertex_targets,
            params.max_edge_ratio,
            &boundary_edges,
            &sharp_edges,
        );
        current_mesh = new_mesh;
        total_splits += splits;

        // Rebuild adjacency and recompute targets for new vertices
        let adj = MeshAdjacency::build(&current_mesh.faces);
        let curvature_result = compute_curvature(&current_mesh);
        let vertex_targets = compute_adaptive_edge_lengths(
            &curvature_result,
            base_target,
            min_adaptive,
            max_adaptive,
            params.curvature_min_threshold,
            params.curvature_max_threshold,
        );

        let boundary_edges: HashSet<(u32, u32)> = if params.preserve_boundary {
            adj.boundary_edges().collect()
        } else {
            HashSet::new()
        };
        let boundary_vertices: HashSet<u32> =
            boundary_edges.iter().flat_map(|&(a, b)| [a, b]).collect();

        // Step 2: Collapse short edges (adaptive)
        let (new_mesh, collapses) = collapse_short_edges_adaptive(
            &current_mesh,
            &adj,
            &vertex_targets,
            params.min_edge_ratio,
            &boundary_edges,
            &sharp_edges,
            &boundary_vertices,
        );
        current_mesh = new_mesh;
        total_collapses += collapses;

        // Rebuild adjacency after collapses
        let adj = MeshAdjacency::build(&current_mesh.faces);

        // Step 3: Flip edges to improve valence
        let (new_mesh, flips) =
            flip_edges_for_valence(&current_mesh, &adj, &boundary_edges, &sharp_edges);
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

    // Clean up
    current_mesh = remove_unreferenced_vertices_internal(&current_mesh);

    RemeshResult {
        final_triangles: current_mesh.faces.len(),
        final_vertices: current_mesh.vertices.len(),
        mesh: current_mesh,
        original_triangles,
        original_vertices,
        iterations_performed: params.iterations,
        target_edge_length: base_target,
        edges_split: total_splits,
        edges_collapsed: total_collapses,
        edges_flipped: total_flips,
        feature_edges_detected: feature_edge_count,
        adaptive_enabled: true,
        anisotropic_enabled: params.anisotropic,
    }
}

/// Compute per-vertex target edge lengths based on curvature.
fn compute_adaptive_edge_lengths(
    curvature: &CurvatureResult,
    _base_target: f64,
    min_length: f64,
    max_length: f64,
    curv_min: f64,
    curv_max: f64,
) -> Vec<f64> {
    curvature
        .vertex_curvatures
        .iter()
        .map(|vc| {
            let curv = vc.mean.abs();

            // Clamp curvature to threshold range
            let curv_clamped = curv.clamp(curv_min, curv_max);

            // Linear interpolation: high curvature -> small edges, low curvature -> large edges
            let t = if curv_max > curv_min {
                (curv_clamped - curv_min) / (curv_max - curv_min)
            } else {
                0.5
            };

            // Interpolate between max_length (flat) and min_length (curved)
            max_length + (min_length - max_length) * t
        })
        .collect()
}

/// Split edges that are longer than their adaptive target.
fn split_long_edges_adaptive(
    mesh: &Mesh,
    adj: &MeshAdjacency,
    vertex_targets: &[f64],
    max_ratio: f64,
    _boundary_edges: &HashSet<(u32, u32)>,
    _sharp_edges: &HashSet<(u32, u32)>,
) -> (Mesh, usize) {
    let mut vertices = mesh.vertices.clone();
    let mut faces: Vec<[u32; 3]> = Vec::with_capacity(mesh.faces.len() * 2);
    let mut split_count = 0;

    let mut edge_midpoints: HashMap<(u32, u32), u32> = HashMap::new();

    // First pass: identify edges to split
    for (&edge, _) in adj.edge_to_faces.iter() {
        let (v0, v1) = edge;
        let p0 = &mesh.vertices[v0 as usize].position;
        let p1 = &mesh.vertices[v1 as usize].position;
        let length = (p1 - p0).norm();

        // Use average of vertex targets for this edge
        let target_v0 = vertex_targets.get(v0 as usize).copied().unwrap_or(1.0);
        let target_v1 = vertex_targets.get(v1 as usize).copied().unwrap_or(1.0);
        let edge_target = (target_v0 + target_v1) / 2.0;
        let max_length = edge_target * max_ratio;

        if length > max_length {
            let midpoint = Point3::new(
                (p0.x + p1.x) / 2.0,
                (p0.y + p1.y) / 2.0,
                (p0.z + p1.z) / 2.0,
            );

            let mut new_vertex = Vertex::new(midpoint);
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

    // Second pass: rebuild faces
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
            0 => faces.push(*face),
            1 => {
                let split_idx = midpoints.iter().position(|m| m.is_some()).unwrap();
                let mid = midpoints[split_idx].unwrap();
                let v0 = face[split_idx];
                let v1 = face[(split_idx + 1) % 3];
                let v2 = face[(split_idx + 2) % 3];
                faces.push([v0, mid, v2]);
                faces.push([mid, v1, v2]);
            }
            2 => {
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
                let m01 = midpoints[0].unwrap();
                let m12 = midpoints[1].unwrap();
                let m20 = midpoints[2].unwrap();
                faces.push([face[0], m01, m20]);
                faces.push([m01, face[1], m12]);
                faces.push([m20, m12, face[2]]);
                faces.push([m01, m12, m20]);
            }
            _ => unreachable!(),
        }
    }

    (Mesh { vertices, faces }, split_count)
}

/// Collapse edges that are shorter than their adaptive target.
fn collapse_short_edges_adaptive(
    mesh: &Mesh,
    adj: &MeshAdjacency,
    vertex_targets: &[f64],
    min_ratio: f64,
    boundary_edges: &HashSet<(u32, u32)>,
    sharp_edges: &HashSet<(u32, u32)>,
    boundary_vertices: &HashSet<u32>,
) -> (Mesh, usize) {
    let mut vertices = mesh.vertices.clone();
    let mut collapse_count = 0;

    let mut vertex_map: HashMap<u32, u32> = HashMap::new();
    for i in 0..vertices.len() {
        vertex_map.insert(i as u32, i as u32);
    }

    let mut edges_with_length: Vec<((u32, u32), f64, f64)> = adj
        .edge_to_faces
        .keys()
        .filter_map(|&edge| {
            let (v0, v1) = edge;
            let p0 = &mesh.vertices[v0 as usize].position;
            let p1 = &mesh.vertices[v1 as usize].position;
            let length = (p1 - p0).norm();

            let target_v0 = vertex_targets.get(v0 as usize).copied().unwrap_or(1.0);
            let target_v1 = vertex_targets.get(v1 as usize).copied().unwrap_or(1.0);
            let edge_target = (target_v0 + target_v1) / 2.0;
            let min_length = edge_target * min_ratio;

            if length < min_length {
                if boundary_edges.contains(&edge) || sharp_edges.contains(&edge) {
                    return None;
                }
                Some((edge, length, min_length))
            } else {
                None
            }
        })
        .collect();

    edges_with_length.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));

    let mut collapsed_vertices: HashSet<u32> = HashSet::new();

    for ((v0, v1), _, _) in edges_with_length {
        if collapsed_vertices.contains(&v0) || collapsed_vertices.contains(&v1) {
            continue;
        }

        let final_v0 = resolve_vertex(&vertex_map, v0);
        let final_v1 = resolve_vertex(&vertex_map, v1);

        if final_v0 == final_v1 {
            continue;
        }

        if would_create_non_manifold(adj, final_v0, final_v1) {
            continue;
        }

        let (keep, remove) = if boundary_vertices.contains(&final_v0) {
            (final_v0, final_v1)
        } else if boundary_vertices.contains(&final_v1) {
            (final_v1, final_v0)
        } else {
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

        vertex_map.insert(remove, keep);
        collapsed_vertices.insert(remove);
        collapse_count += 1;
    }

    let mut new_faces: Vec<[u32; 3]> = Vec::with_capacity(mesh.faces.len());
    for face in &mesh.faces {
        let v0 = resolve_vertex(&vertex_map, face[0]);
        let v1 = resolve_vertex(&vertex_map, face[1]);
        let v2 = resolve_vertex(&vertex_map, face[2]);

        if v0 != v1 && v1 != v2 && v2 != v0 {
            new_faces.push([v0, v1, v2]);
        }
    }

    (
        Mesh {
            vertices,
            faces: new_faces,
        },
        collapse_count,
    )
}

// ============================================================================
// Anisotropic Remeshing
// ============================================================================

/// Perform anisotropic remeshing on a mesh.
///
/// This aligns triangles with the principal curvature directions, creating
/// elongated triangles that better follow surface features (like cylinders,
/// ridges, or valleys).
///
/// # Arguments
/// * `mesh` - The input mesh
/// * `params` - Remeshing parameters with `anisotropic` enabled
///
/// # Returns
/// A `RemeshResult` containing the remeshed mesh and statistics.
///
/// # Example
/// ```
/// use mesh_repair::{Mesh, Vertex, remesh_anisotropic, RemeshParams};
///
/// let mut mesh = Mesh::new();
/// mesh.vertices.push(Vertex::from_coords(0.0, 0.0, 0.0));
/// mesh.vertices.push(Vertex::from_coords(10.0, 0.0, 0.0));
/// mesh.vertices.push(Vertex::from_coords(5.0, 8.66, 0.0));
/// mesh.faces.push([0, 1, 2]);
///
/// let result = remesh_anisotropic(&mesh, &RemeshParams::anisotropic_with_ratio(2.0, 3.0));
/// println!("Anisotropic remeshing produced {} triangles", result.final_triangles);
/// ```
pub fn remesh_anisotropic(mesh: &Mesh, params: &RemeshParams) -> RemeshResult {
    // For anisotropic remeshing, we use a modified version that considers direction
    // For now, we'll use the adaptive remeshing as a base and add directional awareness

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
            feature_edges_detected: 0,
            adaptive_enabled: params.adaptive_to_curvature,
            anisotropic_enabled: true,
        };
    }

    // Get base target
    let base_target = params
        .target_edge_length
        .unwrap_or_else(|| compute_average_edge_length(mesh));

    // Compute curvature and direction field
    let curvature_result = compute_curvature(mesh);

    // Build direction field (either from params or computed from curvature)
    let direction_field: HashMap<u32, Vector3<f64>> =
        if let Some(ref custom_field) = params.direction_field {
            custom_field.clone()
        } else {
            // Use principal curvature directions
            curvature_result
                .vertex_curvatures
                .iter()
                .enumerate()
                .map(|(i, vc)| (i as u32, vc.dir1))
                .collect()
        };

    let mut current_mesh = mesh.clone();
    let mut total_splits = 0;
    let mut total_collapses = 0;
    let mut total_flips = 0;
    let mut feature_edge_count = 0;

    for _iter in 0..params.iterations {
        let adj = MeshAdjacency::build(&current_mesh.faces);

        let boundary_edges: HashSet<(u32, u32)> = if params.preserve_boundary {
            adj.boundary_edges().collect()
        } else {
            HashSet::new()
        };

        let sharp_edges: HashSet<(u32, u32)> = if params.preserve_sharp_edges {
            let feature_result = detect_feature_edges(&current_mesh, params.sharp_angle_threshold);
            feature_edge_count = feature_result.sharp_edges.len();
            feature_result.sharp_edges
        } else {
            HashSet::new()
        };

        // Anisotropic edge splitting
        let (new_mesh, splits) = split_long_edges_anisotropic(
            &current_mesh,
            &adj,
            base_target,
            params.anisotropy_ratio,
            params.max_edge_ratio,
            &direction_field,
            &boundary_edges,
            &sharp_edges,
        );
        current_mesh = new_mesh;
        total_splits += splits;

        // Rebuild adjacency and direction field
        let adj = MeshAdjacency::build(&current_mesh.faces);
        let curvature_result = compute_curvature(&current_mesh);
        let direction_field: HashMap<u32, Vector3<f64>> = curvature_result
            .vertex_curvatures
            .iter()
            .enumerate()
            .map(|(i, vc)| (i as u32, vc.dir1))
            .collect();

        let boundary_edges: HashSet<(u32, u32)> = if params.preserve_boundary {
            adj.boundary_edges().collect()
        } else {
            HashSet::new()
        };
        let boundary_vertices: HashSet<u32> =
            boundary_edges.iter().flat_map(|&(a, b)| [a, b]).collect();

        // Anisotropic edge collapsing
        let (new_mesh, collapses) = collapse_short_edges_anisotropic(
            &current_mesh,
            &adj,
            base_target,
            params.anisotropy_ratio,
            params.min_edge_ratio,
            &direction_field,
            &boundary_edges,
            &sharp_edges,
            &boundary_vertices,
        );
        current_mesh = new_mesh;
        total_collapses += collapses;

        // Edge flipping to improve anisotropic alignment
        let adj = MeshAdjacency::build(&current_mesh.faces);
        let (new_mesh, flips) = flip_edges_for_anisotropy(
            &current_mesh,
            &adj,
            &direction_field,
            &boundary_edges,
            &sharp_edges,
        );
        current_mesh = new_mesh;
        total_flips += flips;

        // Smoothing
        let adj = MeshAdjacency::build(&current_mesh.faces);
        let boundary_vertices: HashSet<u32> = if params.preserve_boundary {
            adj.boundary_edges().flat_map(|(a, b)| [a, b]).collect()
        } else {
            HashSet::new()
        };

        smooth_vertices(
            &mut current_mesh,
            &adj,
            params.smoothing_factor,
            &boundary_vertices,
        );
    }

    current_mesh = remove_unreferenced_vertices_internal(&current_mesh);

    RemeshResult {
        final_triangles: current_mesh.faces.len(),
        final_vertices: current_mesh.vertices.len(),
        mesh: current_mesh,
        original_triangles,
        original_vertices,
        iterations_performed: params.iterations,
        target_edge_length: base_target,
        edges_split: total_splits,
        edges_collapsed: total_collapses,
        edges_flipped: total_flips,
        feature_edges_detected: feature_edge_count,
        adaptive_enabled: params.adaptive_to_curvature,
        anisotropic_enabled: true,
    }
}

/// Split edges based on anisotropic length criteria.
#[allow(clippy::too_many_arguments)]
fn split_long_edges_anisotropic(
    mesh: &Mesh,
    adj: &MeshAdjacency,
    base_target: f64,
    anisotropy_ratio: f64,
    max_ratio: f64,
    direction_field: &HashMap<u32, Vector3<f64>>,
    _boundary_edges: &HashSet<(u32, u32)>,
    _sharp_edges: &HashSet<(u32, u32)>,
) -> (Mesh, usize) {
    let mut vertices = mesh.vertices.clone();
    let mut faces: Vec<[u32; 3]> = Vec::with_capacity(mesh.faces.len() * 2);
    let mut split_count = 0;

    let mut edge_midpoints: HashMap<(u32, u32), u32> = HashMap::new();

    for (&edge, _) in adj.edge_to_faces.iter() {
        let (v0, v1) = edge;
        let p0 = &mesh.vertices[v0 as usize].position;
        let p1 = &mesh.vertices[v1 as usize].position;
        let edge_vec = p1 - p0;
        let length = edge_vec.norm();

        if length < 1e-10 {
            continue;
        }

        // Get direction at edge midpoint (average of endpoint directions)
        let dir0 = direction_field
            .get(&v0)
            .copied()
            .unwrap_or(Vector3::new(1.0, 0.0, 0.0));
        let dir1 = direction_field
            .get(&v1)
            .copied()
            .unwrap_or(Vector3::new(1.0, 0.0, 0.0));
        let avg_dir = (dir0 + dir1).normalize();

        // Compute edge alignment with principal direction
        let edge_dir = edge_vec / length;
        let alignment = edge_dir.dot(&avg_dir).abs();

        // Target length varies with alignment:
        // - Aligned with principal direction: longer edges (base_target * anisotropy_ratio)
        // - Perpendicular: shorter edges (base_target)
        let target_length = base_target * (1.0 + (anisotropy_ratio - 1.0) * alignment);
        let max_length = target_length * max_ratio;

        if length > max_length {
            let midpoint = Point3::new(
                (p0.x + p1.x) / 2.0,
                (p0.y + p1.y) / 2.0,
                (p0.z + p1.z) / 2.0,
            );

            let mut new_vertex = Vertex::new(midpoint);
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

    // Rebuild faces (same as before)
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
            0 => faces.push(*face),
            1 => {
                let split_idx = midpoints.iter().position(|m| m.is_some()).unwrap();
                let mid = midpoints[split_idx].unwrap();
                let v0 = face[split_idx];
                let v1 = face[(split_idx + 1) % 3];
                let v2 = face[(split_idx + 2) % 3];
                faces.push([v0, mid, v2]);
                faces.push([mid, v1, v2]);
            }
            2 => {
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
                let m01 = midpoints[0].unwrap();
                let m12 = midpoints[1].unwrap();
                let m20 = midpoints[2].unwrap();
                faces.push([face[0], m01, m20]);
                faces.push([m01, face[1], m12]);
                faces.push([m20, m12, face[2]]);
                faces.push([m01, m12, m20]);
            }
            _ => unreachable!(),
        }
    }

    (Mesh { vertices, faces }, split_count)
}

/// Collapse edges based on anisotropic length criteria.
#[allow(clippy::too_many_arguments)]
fn collapse_short_edges_anisotropic(
    mesh: &Mesh,
    adj: &MeshAdjacency,
    base_target: f64,
    anisotropy_ratio: f64,
    min_ratio: f64,
    direction_field: &HashMap<u32, Vector3<f64>>,
    boundary_edges: &HashSet<(u32, u32)>,
    sharp_edges: &HashSet<(u32, u32)>,
    boundary_vertices: &HashSet<u32>,
) -> (Mesh, usize) {
    let mut vertices = mesh.vertices.clone();
    let mut collapse_count = 0;

    let mut vertex_map: HashMap<u32, u32> = HashMap::new();
    for i in 0..vertices.len() {
        vertex_map.insert(i as u32, i as u32);
    }

    let mut edges_with_info: Vec<((u32, u32), f64, f64)> = adj
        .edge_to_faces
        .keys()
        .filter_map(|&edge| {
            let (v0, v1) = edge;
            let p0 = &mesh.vertices[v0 as usize].position;
            let p1 = &mesh.vertices[v1 as usize].position;
            let edge_vec = p1 - p0;
            let length = edge_vec.norm();

            if length < 1e-10 {
                return None;
            }

            let dir0 = direction_field
                .get(&v0)
                .copied()
                .unwrap_or(Vector3::new(1.0, 0.0, 0.0));
            let dir1 = direction_field
                .get(&v1)
                .copied()
                .unwrap_or(Vector3::new(1.0, 0.0, 0.0));
            let avg_dir = (dir0 + dir1).normalize();

            let edge_dir = edge_vec / length;
            let alignment = edge_dir.dot(&avg_dir).abs();

            let target_length = base_target * (1.0 + (anisotropy_ratio - 1.0) * alignment);
            let min_length = target_length * min_ratio;

            if length < min_length {
                if boundary_edges.contains(&edge) || sharp_edges.contains(&edge) {
                    return None;
                }
                Some((edge, length, min_length))
            } else {
                None
            }
        })
        .collect();

    edges_with_info.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));

    let mut collapsed_vertices: HashSet<u32> = HashSet::new();

    for ((v0, v1), _, _) in edges_with_info {
        if collapsed_vertices.contains(&v0) || collapsed_vertices.contains(&v1) {
            continue;
        }

        let final_v0 = resolve_vertex(&vertex_map, v0);
        let final_v1 = resolve_vertex(&vertex_map, v1);

        if final_v0 == final_v1 {
            continue;
        }

        if would_create_non_manifold(adj, final_v0, final_v1) {
            continue;
        }

        let (keep, remove) = if boundary_vertices.contains(&final_v0) {
            (final_v0, final_v1)
        } else if boundary_vertices.contains(&final_v1) {
            (final_v1, final_v0)
        } else {
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

        vertex_map.insert(remove, keep);
        collapsed_vertices.insert(remove);
        collapse_count += 1;
    }

    let mut new_faces: Vec<[u32; 3]> = Vec::with_capacity(mesh.faces.len());
    for face in &mesh.faces {
        let v0 = resolve_vertex(&vertex_map, face[0]);
        let v1 = resolve_vertex(&vertex_map, face[1]);
        let v2 = resolve_vertex(&vertex_map, face[2]);

        if v0 != v1 && v1 != v2 && v2 != v0 {
            new_faces.push([v0, v1, v2]);
        }
    }

    (
        Mesh {
            vertices,
            faces: new_faces,
        },
        collapse_count,
    )
}

/// Flip edges to improve alignment with direction field.
fn flip_edges_for_anisotropy(
    mesh: &Mesh,
    adj: &MeshAdjacency,
    direction_field: &HashMap<u32, Vector3<f64>>,
    boundary_edges: &HashSet<(u32, u32)>,
    sharp_edges: &HashSet<(u32, u32)>,
) -> (Mesh, usize) {
    let mut faces = mesh.faces.clone();
    let mut flip_count = 0;

    for (&edge, face_indices) in adj.edge_to_faces.iter() {
        if face_indices.len() != 2 {
            continue;
        }
        if boundary_edges.contains(&edge) || sharp_edges.contains(&edge) {
            continue;
        }

        let (v0, v1) = edge;
        let fi0 = face_indices[0] as usize;
        let fi1 = face_indices[1] as usize;

        let opp0 = find_opposite_vertex(&faces[fi0], v0, v1);
        let opp1 = find_opposite_vertex(&faces[fi1], v0, v1);

        if opp0.is_none() || opp1.is_none() {
            continue;
        }

        let opp0 = opp0.unwrap();
        let opp1 = opp1.unwrap();

        // Get average direction for this region
        let dirs: Vec<Vector3<f64>> = [v0, v1, opp0, opp1]
            .iter()
            .filter_map(|&v| direction_field.get(&v).copied())
            .collect();

        if dirs.is_empty() {
            continue;
        }

        let avg_dir = dirs.iter().fold(Vector3::zeros(), |acc, d| acc + d) / dirs.len() as f64;
        let avg_dir = avg_dir.normalize();

        // Current edge direction
        let p0 = &mesh.vertices[v0 as usize].position;
        let p1 = &mesh.vertices[v1 as usize].position;
        let current_edge = (p1 - p0).normalize();
        let current_alignment = current_edge.dot(&avg_dir).abs();

        // Potential new edge direction
        let p_opp0 = &mesh.vertices[opp0 as usize].position;
        let p_opp1 = &mesh.vertices[opp1 as usize].position;
        let new_edge = (p_opp1 - p_opp0).normalize();
        let new_alignment = new_edge.dot(&avg_dir).abs();

        // Flip if new edge is better aligned with direction field
        if new_alignment > current_alignment + 0.1 && is_valid_flip(mesh, v0, v1, opp0, opp1) {
            let new_face0 = [opp0, opp1, v0];
            let new_face1 = [opp1, opp0, v1];

            faces[fi0] = new_face0;
            faces[fi1] = new_face1;

            flip_count += 1;
        }
    }

    (
        Mesh {
            vertices: mesh.vertices.clone(),
            faces,
        },
        flip_count,
    )
}

/// Perform isotropic remeshing with progress reporting.
///
/// This is a progress-reporting variant of [`remesh_isotropic`] that allows tracking
/// the remeshing progress and supports cancellation via the progress callback.
///
/// # Arguments
/// * `mesh` - The input mesh to remesh
/// * `params` - Remeshing parameters
/// * `callback` - Optional progress callback. Returns `false` to request cancellation.
///
/// # Returns
/// A `RemeshResult` containing the remeshed mesh and statistics.
/// If cancelled via callback, returns the partially remeshed mesh.
///
/// # Example
/// ```ignore
/// use mesh_repair::{Mesh, remesh_isotropic_with_progress, RemeshParams};
/// use mesh_repair::progress::ProgressCallback;
///
/// let callback: ProgressCallback = Box::new(|progress| {
///     println!("Iteration {}/{}: {}", progress.current, progress.total, progress.message);
///     true // Continue
/// });
///
/// let result = remesh_isotropic_with_progress(&mesh, &RemeshParams::default(), Some(&callback));
/// ```
pub fn remesh_isotropic_with_progress(
    mesh: &Mesh,
    params: &RemeshParams,
    callback: Option<&crate::progress::ProgressCallback>,
) -> RemeshResult {
    use crate::progress::ProgressTracker;

    let original_triangles = mesh.faces.len();
    let original_vertices = mesh.vertices.len();

    // Dispatch to appropriate remeshing algorithm based on params
    if params.adaptive_to_curvature {
        return remesh_adaptive(mesh, params);
    }
    if params.anisotropic {
        return remesh_anisotropic(mesh, params);
    }

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
            feature_edges_detected: 0,
            adaptive_enabled: false,
            anisotropic_enabled: false,
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

    // Create progress tracker for iterations
    let tracker = ProgressTracker::new(params.iterations as u64);

    for iter in 0..params.iterations {
        // Check for cancellation
        if tracker.is_cancelled() {
            break;
        }

        // Report progress at start of iteration
        tracker.set(iter as u64);
        if !tracker.maybe_callback(
            callback,
            format!(
                "Remeshing iteration {}/{}: {} vertices, {} faces",
                iter + 1,
                params.iterations,
                current_mesh.vertices.len(),
                current_mesh.faces.len()
            ),
        ) {
            break; // Cancelled
        }

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

        let _boundary_vertices: HashSet<u32> =
            boundary_edges.iter().flat_map(|&(a, b)| [a, b]).collect();

        // Step 1: Split long edges
        let (new_mesh, splits) = split_long_edges(
            &current_mesh,
            &adj,
            max_length,
            &boundary_edges,
            &sharp_edges,
        );
        current_mesh = new_mesh;
        total_splits += splits;

        // Rebuild adjacency after splits
        let adj = MeshAdjacency::build(&current_mesh.faces);
        let boundary_edges: HashSet<(u32, u32)> = if params.preserve_boundary {
            adj.boundary_edges().collect()
        } else {
            HashSet::new()
        };
        let boundary_vertices: HashSet<u32> =
            boundary_edges.iter().flat_map(|&(a, b)| [a, b]).collect();

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
        let (new_mesh, flips) =
            flip_edges_for_valence(&current_mesh, &adj, &boundary_edges, &sharp_edges);
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

    // Final progress update
    tracker.set(params.iterations as u64);
    let _ = tracker.maybe_callback(callback, "Remeshing complete".to_string());

    // Clean up any unreferenced vertices
    current_mesh = remove_unreferenced_vertices_internal(&current_mesh);

    // Count feature edges if preserving them
    let feature_edge_count = if params.preserve_sharp_edges {
        let adj = MeshAdjacency::build(&current_mesh.faces);
        find_sharp_edges(&current_mesh, &adj, params.sharp_angle_threshold).len()
    } else {
        0
    };

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
        feature_edges_detected: feature_edge_count,
        adaptive_enabled: false,
        anisotropic_enabled: false,
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
            faces: vec![[0, 1, 2], [0, 3, 1]],
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
            faces: vec![[0, 1, 2], [0, 2, 3]],
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
        assert!(!result.mesh.faces.is_empty());
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
        let params = RemeshParams {
            iterations: 0,
            ..Default::default()
        };

        let result = remesh_isotropic(&mesh, &params);

        assert_eq!(result.final_triangles, 1);
        assert_eq!(result.iterations_performed, 0);
    }

    // =========================================================================
    // Feature Edge Detection Tests
    // =========================================================================

    fn make_cube() -> Mesh {
        // A simple cube with 12 triangles (6 faces, 2 triangles each)
        Mesh {
            vertices: vec![
                Vertex::from_coords(0.0, 0.0, 0.0), // 0
                Vertex::from_coords(1.0, 0.0, 0.0), // 1
                Vertex::from_coords(1.0, 1.0, 0.0), // 2
                Vertex::from_coords(0.0, 1.0, 0.0), // 3
                Vertex::from_coords(0.0, 0.0, 1.0), // 4
                Vertex::from_coords(1.0, 0.0, 1.0), // 5
                Vertex::from_coords(1.0, 1.0, 1.0), // 6
                Vertex::from_coords(0.0, 1.0, 1.0), // 7
            ],
            faces: vec![
                // Front face (z = 0)
                [0, 1, 2],
                [0, 2, 3],
                // Back face (z = 1)
                [5, 4, 7],
                [5, 7, 6],
                // Bottom face (y = 0)
                [0, 4, 5],
                [0, 5, 1],
                // Top face (y = 1)
                [3, 2, 6],
                [3, 6, 7],
                // Left face (x = 0)
                [0, 3, 7],
                [0, 7, 4],
                // Right face (x = 1)
                [1, 5, 6],
                [1, 6, 2],
            ],
        }
    }

    fn make_folded_surface() -> Mesh {
        // Two triangles forming a 90-degree fold
        Mesh {
            vertices: vec![
                Vertex::from_coords(0.0, 0.0, 0.0),
                Vertex::from_coords(1.0, 0.0, 0.0),
                Vertex::from_coords(0.5, 1.0, 0.0),
                Vertex::from_coords(0.5, 0.0, 1.0),
            ],
            faces: vec![
                [0, 1, 2], // Flat triangle in XY plane
                [0, 3, 1], // Triangle in XZ plane - forms 90 degree angle
            ],
        }
    }

    #[test]
    fn test_detect_feature_edges_single_triangle() {
        let mesh = make_single_triangle();
        let result = detect_feature_edges(&mesh, std::f64::consts::PI / 3.0);

        // Single triangle should have 3 boundary edges
        assert_eq!(result.boundary_edges.len(), 3);
        assert_eq!(result.sharp_edges.len(), 0); // No interior sharp edges
    }

    #[test]
    fn test_detect_feature_edges_two_triangles() {
        let mesh = make_two_triangles();
        let result = detect_feature_edges(&mesh, std::f64::consts::PI / 3.0);

        // Two triangles sharing an edge should have 4 boundary edges
        assert_eq!(result.boundary_edges.len(), 4);
    }

    #[test]
    fn test_detect_feature_edges_folded_surface() {
        let mesh = make_folded_surface();

        // With a 45-degree threshold, the 90-degree edge should be detected
        let result = detect_feature_edges(&mesh, std::f64::consts::PI / 4.0);

        // The shared edge should be detected as sharp (90 degree angle)
        assert!(
            !result.sharp_edges.is_empty(),
            "Should detect the sharp edge"
        );
    }

    #[test]
    fn test_detect_feature_edges_cube() {
        let mesh = make_cube();

        // Cube edges are 90 degrees, should be detected with 60-degree threshold
        let result = detect_feature_edges(&mesh, std::f64::consts::PI / 3.0);

        // Cube has 12 edges, all should be sharp (90 degrees)
        assert!(
            !result.sharp_edges.is_empty(),
            "Cube should have sharp edges"
        );
    }

    #[test]
    fn test_feature_edge_result_statistics() {
        let mesh = make_folded_surface();
        let result = detect_feature_edges(&mesh, std::f64::consts::PI / 4.0);

        // Should have valid statistics
        assert!(result.max_dihedral_angle >= result.mean_dihedral_angle);
    }

    // =========================================================================
    // Curvature Computation Tests
    // =========================================================================

    #[test]
    fn test_compute_curvature_single_triangle() {
        let mesh = make_single_triangle();
        let result = compute_curvature(&mesh);

        // Single triangle has 3 vertices
        assert_eq!(result.vertex_curvatures.len(), 3);

        // Flat triangle should have zero/near-zero curvature
        for vc in &result.vertex_curvatures {
            assert!(
                vc.mean.abs() < 1.0,
                "Flat triangle should have low curvature, got {}",
                vc.mean
            );
        }
    }

    #[test]
    fn test_compute_curvature_quad() {
        let mesh = make_quad_as_triangles();
        let result = compute_curvature(&mesh);

        // 4 vertices
        assert_eq!(result.vertex_curvatures.len(), 4);
    }

    #[test]
    fn test_compute_curvature_folded() {
        let mesh = make_folded_surface();
        let result = compute_curvature(&mesh);

        // 4 vertices
        assert_eq!(result.vertex_curvatures.len(), 4);

        // Should have some valid statistics
        assert!(result.avg_mean_curvature.is_finite());
    }

    #[test]
    fn test_curvature_directions() {
        let mesh = make_cube();
        let result = compute_curvature(&mesh);

        // All curvature directions should be unit vectors
        for vc in &result.vertex_curvatures {
            let dir1_len = vc.dir1.norm();
            let dir2_len = vc.dir2.norm();

            if dir1_len > 0.0 {
                assert!((dir1_len - 1.0).abs() < 1e-6, "dir1 should be normalized");
            }
            if dir2_len > 0.0 {
                assert!((dir2_len - 1.0).abs() < 1e-6, "dir2 should be normalized");
            }
        }
    }

    // =========================================================================
    // Curvature-Adaptive Remeshing Tests
    // =========================================================================

    #[test]
    fn test_remesh_adaptive_params() {
        let params = RemeshParams::adaptive(2.0);

        assert!(params.adaptive_to_curvature);
        assert!(params.preserve_sharp_edges);
        assert_eq!(params.target_edge_length, Some(2.0));
    }

    #[test]
    fn test_remesh_adaptive_single_triangle() {
        let mesh = make_single_triangle();
        let result = remesh_adaptive(&mesh, &RemeshParams::adaptive(2.0));

        assert!(result.adaptive_enabled);
        assert!(result.final_triangles >= 1);
    }

    #[test]
    fn test_remesh_adaptive_produces_mesh() {
        let mesh = make_quad_as_triangles();
        let result = remesh_adaptive(&mesh, &RemeshParams::adaptive(2.0));

        assert!(!result.mesh.faces.is_empty());
        assert!(!result.mesh.vertices.is_empty());

        // Topology should be valid
        for face in &result.mesh.faces {
            for &vi in face {
                assert!((vi as usize) < result.mesh.vertices.len());
            }
        }
    }

    #[test]
    fn test_remesh_adaptive_curvature_thresholds() {
        let params = RemeshParams::adaptive(2.0).with_curvature_thresholds(0.001, 2.0);

        assert_eq!(params.curvature_min_threshold, 0.001);
        assert_eq!(params.curvature_max_threshold, 2.0);
    }

    #[test]
    fn test_remesh_adaptive_edge_range() {
        let params = RemeshParams::adaptive(2.0).with_adaptive_edge_range(0.5, 4.0);

        assert_eq!(params.min_edge_length_adaptive, Some(0.5));
        assert_eq!(params.max_edge_length_adaptive, Some(4.0));
    }

    // =========================================================================
    // Anisotropic Remeshing Tests
    // =========================================================================

    #[test]
    fn test_remesh_anisotropic_params() {
        let params = RemeshParams::anisotropic_with_ratio(2.0, 3.0);

        assert!(params.anisotropic);
        assert_eq!(params.anisotropy_ratio, 3.0);
        assert_eq!(params.target_edge_length, Some(2.0));
    }

    #[test]
    fn test_remesh_anisotropic_single_triangle() {
        let mesh = make_single_triangle();
        let result = remesh_anisotropic(&mesh, &RemeshParams::anisotropic_with_ratio(2.0, 2.0));

        assert!(result.anisotropic_enabled);
        assert!(result.final_triangles >= 1);
    }

    #[test]
    fn test_remesh_anisotropic_produces_mesh() {
        let mesh = make_quad_as_triangles();
        let result = remesh_anisotropic(&mesh, &RemeshParams::anisotropic_with_ratio(2.0, 2.0));

        assert!(!result.mesh.faces.is_empty());
        assert!(!result.mesh.vertices.is_empty());

        // Topology should be valid
        for face in &result.mesh.faces {
            for &vi in face {
                assert!((vi as usize) < result.mesh.vertices.len());
            }
        }
    }

    #[test]
    fn test_remesh_with_direction_field() {
        use hashbrown::HashMap;

        let mesh = make_quad_as_triangles();

        // Create a custom direction field (all vertices pointing X)
        let mut field = HashMap::new();
        for i in 0..mesh.vertices.len() {
            field.insert(i as u32, Vector3::new(1.0, 0.0, 0.0));
        }

        let params = RemeshParams::anisotropic_with_ratio(2.0, 2.0).with_direction_field(field);

        let result = remesh_isotropic(&mesh, &params);

        assert!(result.anisotropic_enabled);
    }

    // =========================================================================
    // RemeshParams Builder Tests
    // =========================================================================

    #[test]
    fn test_remesh_params_with_curvature_adaptation() {
        let params = RemeshParams::with_target_edge_length(2.0).with_curvature_adaptation();

        assert!(params.adaptive_to_curvature);
    }

    #[test]
    fn test_remesh_params_with_anisotropy() {
        let params = RemeshParams::with_target_edge_length(2.0).with_anisotropy(4.0);

        assert!(params.anisotropic);
        assert_eq!(params.anisotropy_ratio, 4.0);
    }

    #[test]
    fn test_remesh_params_with_feature_edges() {
        use hashbrown::HashSet;

        let mut edges = HashSet::new();
        edges.insert((0, 1));
        edges.insert((1, 2));

        let params = RemeshParams::default().with_feature_edges(edges.clone());

        assert!(params.preserve_feature_edges.is_some());
        assert_eq!(params.preserve_feature_edges.unwrap().len(), 2);
    }

    // =========================================================================
    // Result Field Tests
    // =========================================================================

    #[test]
    fn test_remesh_result_has_new_fields() {
        let mesh = make_quad_as_triangles();
        let result = remesh_isotropic(&mesh, &RemeshParams::with_target_edge_length(2.0));

        // Check new fields exist and have sensible values
        assert!(!result.adaptive_enabled);
        assert!(!result.anisotropic_enabled);
        // feature_edges_detected is 0 when preserve_sharp_edges is false (default)
        assert_eq!(result.feature_edges_detected, 0);
    }

    #[test]
    fn test_remesh_result_with_feature_preservation() {
        let mesh = make_cube();
        let params = RemeshParams::preserve_features();
        let result = remesh_isotropic(&mesh, &params);

        // With preserve_features, the field should be populated
        // (the exact count depends on the mesh structure after remeshing)
        let _edges = result.feature_edges_detected; // Verify field exists
    }

    // =========================================================================
    // Integration Tests
    // =========================================================================

    #[test]
    fn test_dispatch_to_adaptive() {
        let mesh = make_quad_as_triangles();
        let params = RemeshParams::adaptive(3.0);

        // Should automatically dispatch to adaptive remeshing
        let result = remesh_isotropic(&mesh, &params);

        assert!(result.adaptive_enabled);
    }

    #[test]
    fn test_dispatch_to_anisotropic() {
        let mesh = make_quad_as_triangles();
        let params = RemeshParams::anisotropic_with_ratio(3.0, 2.0);

        // Should automatically dispatch to anisotropic remeshing
        let result = remesh_isotropic(&mesh, &params);

        assert!(result.anisotropic_enabled);
    }

    #[test]
    fn test_empty_mesh_adaptive() {
        let mesh = Mesh::new();
        let result = remesh_adaptive(&mesh, &RemeshParams::adaptive(2.0));

        assert_eq!(result.final_triangles, 0);
        assert!(result.adaptive_enabled);
    }

    #[test]
    fn test_empty_mesh_anisotropic() {
        let mesh = Mesh::new();
        let result = remesh_anisotropic(&mesh, &RemeshParams::anisotropic_with_ratio(2.0, 2.0));

        assert_eq!(result.final_triangles, 0);
        assert!(result.anisotropic_enabled);
    }
}

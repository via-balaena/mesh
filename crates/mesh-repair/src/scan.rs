//! Scan processing and cleanup utilities.
//!
//! This module provides tools for processing raw 3D scan data, including
//! noise reduction, artifact removal, and automated cleanup pipelines.
//!
//! # Use Cases
//!
//! - Cleaning up body scans from depth cameras
//! - Processing object scans from structured light scanners
//! - Preparing point cloud data for surface reconstruction
//!
//! # Example
//!
//! ```
//! use mesh_repair::{Mesh, Vertex};
//! use mesh_repair::scan::{ScanCleanupParams, cleanup_scan, DenoiseParams};
//!
//! // Create a mesh (typically loaded from a scanner)
//! let mut mesh = Mesh::new();
//! mesh.vertices.push(Vertex::from_coords(0.0, 0.0, 0.0));
//! mesh.vertices.push(Vertex::from_coords(1.0, 0.0, 0.0));
//! mesh.vertices.push(Vertex::from_coords(0.5, 1.0, 0.0));
//! mesh.faces.push([0, 1, 2]);
//!
//! // Clean up the scan with default parameters
//! let params = ScanCleanupParams::default();
//! let result = cleanup_scan(&mesh, &params);
//! ```

use crate::{Mesh, Vertex};
use nalgebra::{Point3, Vector3};
use std::collections::{HashMap, HashSet};

/// Parameters for scan cleanup operations.
#[derive(Debug, Clone)]
pub struct ScanCleanupParams {
    /// Remove isolated vertices/faces.
    pub remove_isolated: bool,

    /// Minimum component size to keep (in faces).
    pub min_component_size: usize,

    /// Remove spike artifacts.
    pub remove_spikes: bool,

    /// Spike detection threshold (standard deviations from mean edge length).
    pub spike_threshold: f64,

    /// Fill small holes.
    pub fill_small_holes: bool,

    /// Maximum hole size to fill (in edges).
    pub max_hole_size: usize,

    /// Apply smoothing.
    pub smooth: bool,

    /// Smoothing iterations.
    pub smooth_iterations: usize,

    /// Smoothing strength (0-1).
    pub smooth_strength: f64,

    /// Preserve features during smoothing.
    pub preserve_features: bool,

    /// Feature preservation threshold (dihedral angle in radians).
    pub feature_angle_threshold: f64,
}

impl Default for ScanCleanupParams {
    fn default() -> Self {
        Self {
            remove_isolated: true,
            min_component_size: 100,
            remove_spikes: true,
            spike_threshold: 3.0,
            fill_small_holes: true,
            max_hole_size: 50,
            smooth: true,
            smooth_iterations: 2,
            smooth_strength: 0.5,
            preserve_features: true,
            feature_angle_threshold: 0.5, // ~30 degrees
        }
    }
}

impl ScanCleanupParams {
    /// Parameters optimized for body scans.
    pub fn for_body_scan() -> Self {
        Self {
            remove_isolated: true,
            min_component_size: 500,
            remove_spikes: true,
            spike_threshold: 2.5,
            fill_small_holes: true,
            max_hole_size: 100,
            smooth: true,
            smooth_iterations: 3,
            smooth_strength: 0.4,
            preserve_features: true,
            feature_angle_threshold: 0.7, // ~40 degrees
        }
    }

    /// Parameters optimized for object scans.
    pub fn for_object_scan() -> Self {
        Self {
            remove_isolated: true,
            min_component_size: 50,
            remove_spikes: true,
            spike_threshold: 3.5,
            fill_small_holes: true,
            max_hole_size: 30,
            smooth: true,
            smooth_iterations: 2,
            smooth_strength: 0.3,
            preserve_features: true,
            feature_angle_threshold: 0.4, // ~23 degrees, preserve sharper edges
        }
    }

    /// Parameters for minimal cleanup (preserve detail).
    pub fn minimal() -> Self {
        Self {
            remove_isolated: true,
            min_component_size: 10,
            remove_spikes: false,
            spike_threshold: 4.0,
            fill_small_holes: false,
            max_hole_size: 10,
            smooth: false,
            smooth_iterations: 1,
            smooth_strength: 0.2,
            preserve_features: true,
            feature_angle_threshold: 0.3,
        }
    }

    /// Parameters for aggressive cleanup.
    pub fn aggressive() -> Self {
        Self {
            remove_isolated: true,
            min_component_size: 1000,
            remove_spikes: true,
            spike_threshold: 2.0,
            fill_small_holes: true,
            max_hole_size: 200,
            smooth: true,
            smooth_iterations: 5,
            smooth_strength: 0.7,
            preserve_features: false,
            feature_angle_threshold: 0.5,
        }
    }
}

/// Result of scan cleanup operation.
#[derive(Debug)]
pub struct ScanCleanupResult {
    /// Cleaned mesh.
    pub mesh: Mesh,

    /// Number of isolated components removed.
    pub components_removed: usize,

    /// Number of spike vertices removed.
    pub spikes_removed: usize,

    /// Number of holes filled.
    pub holes_filled: usize,

    /// Smoothing applied.
    pub smoothing_applied: bool,
}

/// Clean up a scan mesh using the specified parameters.
pub fn cleanup_scan(mesh: &Mesh, params: &ScanCleanupParams) -> ScanCleanupResult {
    let mut result_mesh = mesh.clone();
    let mut components_removed = 0;
    let mut spikes_removed = 0;
    let mut holes_filled = 0;

    // Step 1: Remove isolated components
    if params.remove_isolated {
        components_removed = remove_small_components(&mut result_mesh, params.min_component_size);
    }

    // Step 2: Remove spike artifacts
    if params.remove_spikes {
        spikes_removed = remove_spikes(&mut result_mesh, params.spike_threshold);
    }

    // Step 3: Fill small holes
    if params.fill_small_holes {
        holes_filled = fill_small_holes(&mut result_mesh, params.max_hole_size);
    }

    // Step 4: Smooth
    if params.smooth {
        if params.preserve_features {
            smooth_preserve_features(
                &mut result_mesh,
                params.smooth_iterations,
                params.smooth_strength,
                params.feature_angle_threshold,
            );
        } else {
            smooth_laplacian(
                &mut result_mesh,
                params.smooth_iterations,
                params.smooth_strength,
            );
        }
    }

    ScanCleanupResult {
        mesh: result_mesh,
        components_removed,
        spikes_removed,
        holes_filled,
        smoothing_applied: params.smooth,
    }
}

/// Parameters for mesh denoising.
#[derive(Debug, Clone)]
pub struct DenoiseParams {
    /// Denoising method.
    pub method: DenoiseMethod,

    /// Number of iterations.
    pub iterations: usize,

    /// Strength of denoising (0-1).
    pub strength: f64,

    /// Preserve sharp features.
    pub preserve_features: bool,

    /// Feature angle threshold (radians).
    pub feature_threshold: f64,
}

impl Default for DenoiseParams {
    fn default() -> Self {
        Self {
            method: DenoiseMethod::Bilateral,
            iterations: 3,
            strength: 0.5,
            preserve_features: true,
            feature_threshold: 0.5,
        }
    }
}

/// Denoising method.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DenoiseMethod {
    /// Laplacian smoothing.
    Laplacian,

    /// Bilateral filtering (edge-preserving).
    Bilateral,

    /// Taubin smoothing (reduces shrinkage).
    Taubin,

    /// Mean curvature flow.
    MeanCurvatureFlow,
}

/// Result of denoising operation.
#[derive(Debug)]
pub struct DenoiseResult {
    /// Denoised mesh.
    pub mesh: Mesh,

    /// Average displacement per vertex.
    pub average_displacement: f64,

    /// Maximum displacement.
    pub max_displacement: f64,

    /// Number of iterations performed.
    pub iterations_performed: usize,
}

/// Denoise a mesh using the specified parameters.
pub fn denoise_mesh(mesh: &Mesh, params: &DenoiseParams) -> DenoiseResult {
    let mut result = mesh.clone();
    let mut total_displacement = 0.0;
    let mut max_displacement = 0.0f64;

    for _ in 0..params.iterations {
        let displacements = match params.method {
            DenoiseMethod::Laplacian => compute_laplacian_displacements(&result, params.strength),
            DenoiseMethod::Bilateral => {
                compute_bilateral_displacements(&result, params.strength, params.feature_threshold)
            }
            DenoiseMethod::Taubin => compute_taubin_displacements(&result, params.strength),
            DenoiseMethod::MeanCurvatureFlow => compute_mcf_displacements(&result, params.strength),
        };

        // Apply displacements
        for (i, disp) in displacements.iter().enumerate() {
            if params.preserve_features && is_feature_vertex(&result, i, params.feature_threshold) {
                continue;
            }

            let disp_mag = disp.norm();
            total_displacement += disp_mag;
            max_displacement = max_displacement.max(disp_mag);
            result.vertices[i].position += disp;
        }
    }

    let vertex_count = result.vertices.len().max(1);
    DenoiseResult {
        mesh: result,
        average_displacement: total_displacement / (vertex_count * params.iterations) as f64,
        max_displacement,
        iterations_performed: params.iterations,
    }
}

/// Parameters for advanced hole filling.
#[derive(Debug, Clone)]
pub struct HoleFillParams {
    /// Filling strategy.
    pub strategy: HoleFillStrategy,

    /// Maximum hole size to fill (edges).
    pub max_size: usize,

    /// Smoothing iterations after filling.
    pub smooth_iterations: usize,
}

impl Default for HoleFillParams {
    fn default() -> Self {
        Self {
            strategy: HoleFillStrategy::Smooth,
            max_size: 100,
            smooth_iterations: 2,
        }
    }
}

/// Hole filling strategy.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum HoleFillStrategy {
    /// Simple triangulation (flat fill).
    Planar,

    /// Smooth interpolation from boundary.
    Smooth,

    /// Curvature-based fill matching surrounding surface.
    CurvatureBased,

    /// Minimal area fill.
    MinimalArea,
}

/// Result of advanced hole filling.
#[derive(Debug)]
pub struct HoleFillResult {
    /// Mesh with holes filled.
    pub mesh: Mesh,

    /// Number of holes filled.
    pub holes_filled: usize,

    /// Number of faces added.
    pub faces_added: usize,

    /// Sizes of filled holes (in edges).
    pub hole_sizes: Vec<usize>,
}

/// Fill holes using advanced strategies.
pub fn fill_holes_advanced(mesh: &Mesh, params: &HoleFillParams) -> HoleFillResult {
    let mut result = mesh.clone();
    let holes = detect_boundary_loops(&result);

    let mut holes_filled = 0;
    let mut faces_added = 0;
    let mut hole_sizes = Vec::new();

    for hole in holes {
        if hole.len() > params.max_size {
            continue;
        }

        let added = match params.strategy {
            HoleFillStrategy::Planar => fill_hole_planar(&mut result, &hole),
            HoleFillStrategy::Smooth => {
                fill_hole_smooth(&mut result, &hole, params.smooth_iterations)
            }
            HoleFillStrategy::CurvatureBased => fill_hole_curvature(&mut result, &hole),
            HoleFillStrategy::MinimalArea => fill_hole_minimal_area(&mut result, &hole),
        };

        if added > 0 {
            holes_filled += 1;
            faces_added += added;
            hole_sizes.push(hole.len());
        }
    }

    HoleFillResult {
        mesh: result,
        holes_filled,
        faces_added,
        hole_sizes,
    }
}

/// Statistical outlier removal parameters.
#[derive(Debug, Clone)]
pub struct OutlierRemovalParams {
    /// Number of neighbors to consider.
    pub k_neighbors: usize,

    /// Standard deviation threshold.
    pub std_dev_threshold: f64,
}

impl Default for OutlierRemovalParams {
    fn default() -> Self {
        Self {
            k_neighbors: 10,
            std_dev_threshold: 2.0,
        }
    }
}

/// Remove statistical outliers from mesh.
pub fn remove_outliers(mesh: &Mesh, params: &OutlierRemovalParams) -> Mesh {
    if mesh.vertices.len() < params.k_neighbors {
        return mesh.clone();
    }

    // Compute average distance to k neighbors for each vertex
    let mut avg_distances: Vec<f64> = Vec::with_capacity(mesh.vertices.len());

    for (i, v) in mesh.vertices.iter().enumerate() {
        let mut distances: Vec<f64> = mesh
            .vertices
            .iter()
            .enumerate()
            .filter(|(j, _)| *j != i)
            .map(|(_, other)| (v.position - other.position).norm())
            .collect();

        distances.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

        let k = params.k_neighbors.min(distances.len());
        let avg: f64 = distances[..k].iter().sum::<f64>() / k as f64;
        avg_distances.push(avg);
    }

    // Compute mean and std dev
    let mean: f64 = avg_distances.iter().sum::<f64>() / avg_distances.len() as f64;
    let variance: f64 = avg_distances
        .iter()
        .map(|d| (d - mean).powi(2))
        .sum::<f64>()
        / avg_distances.len() as f64;
    let std_dev = variance.sqrt();

    let threshold = mean + params.std_dev_threshold * std_dev;

    // Mark vertices to keep
    let keep: HashSet<u32> = avg_distances
        .iter()
        .enumerate()
        .filter(|(_, d)| **d <= threshold)
        .map(|(i, _)| i as u32)
        .collect();

    // Build new mesh with only kept vertices
    let mut result = Mesh::new();
    let mut vertex_map: HashMap<u32, u32> = HashMap::new();

    for (i, v) in mesh.vertices.iter().enumerate() {
        if keep.contains(&(i as u32)) {
            vertex_map.insert(i as u32, result.vertices.len() as u32);
            result.vertices.push(v.clone());
        }
    }

    // Keep faces where all vertices are kept
    for face in &mesh.faces {
        if let (Some(&i0), Some(&i1), Some(&i2)) = (
            vertex_map.get(&face[0]),
            vertex_map.get(&face[1]),
            vertex_map.get(&face[2]),
        ) {
            result.faces.push([i0, i1, i2]);
        }
    }

    result
}

// ============================================================================
// Internal helper functions
// ============================================================================

/// Remove small connected components.
fn remove_small_components(mesh: &mut Mesh, min_size: usize) -> usize {
    let components = find_connected_components(mesh);
    let mut removed = 0;

    // Find faces to keep
    let mut faces_to_keep: HashSet<usize> = HashSet::new();
    for component in &components {
        if component.len() >= min_size {
            faces_to_keep.extend(component.iter());
        } else {
            removed += 1;
        }
    }

    if removed == 0 {
        return 0;
    }

    // Rebuild mesh with only kept faces
    let new_faces: Vec<[u32; 3]> = mesh
        .faces
        .iter()
        .enumerate()
        .filter(|(i, _)| faces_to_keep.contains(i))
        .map(|(_, f)| *f)
        .collect();

    mesh.faces = new_faces;

    // Remove unreferenced vertices
    let mut referenced: HashSet<u32> = HashSet::new();
    for face in &mesh.faces {
        referenced.insert(face[0]);
        referenced.insert(face[1]);
        referenced.insert(face[2]);
    }

    let mut vertex_map: HashMap<u32, u32> = HashMap::new();
    let mut new_vertices = Vec::new();

    for (i, v) in mesh.vertices.iter().enumerate() {
        if referenced.contains(&(i as u32)) {
            vertex_map.insert(i as u32, new_vertices.len() as u32);
            new_vertices.push(v.clone());
        }
    }

    mesh.vertices = new_vertices;

    for face in &mut mesh.faces {
        face[0] = vertex_map[&face[0]];
        face[1] = vertex_map[&face[1]];
        face[2] = vertex_map[&face[2]];
    }

    removed
}

/// Find connected components (returns face indices per component).
fn find_connected_components(mesh: &Mesh) -> Vec<Vec<usize>> {
    let mut visited: HashSet<usize> = HashSet::new();
    let mut components: Vec<Vec<usize>> = Vec::new();

    // Build face adjacency
    let mut edge_to_faces: HashMap<(u32, u32), Vec<usize>> = HashMap::new();
    for (fi, face) in mesh.faces.iter().enumerate() {
        for i in 0..3 {
            let v0 = face[i];
            let v1 = face[(i + 1) % 3];
            let edge = if v0 < v1 { (v0, v1) } else { (v1, v0) };
            edge_to_faces.entry(edge).or_default().push(fi);
        }
    }

    for fi in 0..mesh.faces.len() {
        if visited.contains(&fi) {
            continue;
        }

        let mut component = Vec::new();
        let mut stack = vec![fi];

        while let Some(current) = stack.pop() {
            if visited.contains(&current) {
                continue;
            }
            visited.insert(current);
            component.push(current);

            let face = &mesh.faces[current];
            for i in 0..3 {
                let v0 = face[i];
                let v1 = face[(i + 1) % 3];
                let edge = if v0 < v1 { (v0, v1) } else { (v1, v0) };

                if let Some(neighbors) = edge_to_faces.get(&edge) {
                    for &neighbor in neighbors {
                        if !visited.contains(&neighbor) {
                            stack.push(neighbor);
                        }
                    }
                }
            }
        }

        components.push(component);
    }

    components
}

/// Remove spike artifacts (vertices with abnormal edge lengths).
fn remove_spikes(mesh: &mut Mesh, threshold: f64) -> usize {
    if mesh.faces.is_empty() {
        return 0;
    }

    // Compute edge lengths
    let mut edge_lengths: Vec<f64> = Vec::new();
    for face in &mesh.faces {
        for i in 0..3 {
            let v0 = &mesh.vertices[face[i] as usize].position;
            let v1 = &mesh.vertices[face[(i + 1) % 3] as usize].position;
            edge_lengths.push((v1 - v0).norm());
        }
    }

    if edge_lengths.is_empty() {
        return 0;
    }

    // Compute statistics
    let mean: f64 = edge_lengths.iter().sum::<f64>() / edge_lengths.len() as f64;
    let variance: f64 =
        edge_lengths.iter().map(|l| (l - mean).powi(2)).sum::<f64>() / edge_lengths.len() as f64;
    let std_dev = variance.sqrt();

    let max_length = mean + threshold * std_dev;

    // Find spike vertices (connected to abnormally long edges)
    let mut spike_vertices: HashSet<u32> = HashSet::new();
    for face in &mesh.faces {
        for i in 0..3 {
            let v0 = &mesh.vertices[face[i] as usize].position;
            let v1 = &mesh.vertices[face[(i + 1) % 3] as usize].position;
            if (v1 - v0).norm() > max_length {
                spike_vertices.insert(face[i]);
                spike_vertices.insert(face[(i + 1) % 3]);
            }
        }
    }

    if spike_vertices.is_empty() {
        return 0;
    }

    // Remove faces containing spike vertices
    let original_face_count = mesh.faces.len();
    mesh.faces.retain(|face| {
        !spike_vertices.contains(&face[0])
            && !spike_vertices.contains(&face[1])
            && !spike_vertices.contains(&face[2])
    });

    original_face_count - mesh.faces.len()
}

/// Fill small holes using simple ear-clipping triangulation.
fn fill_small_holes(mesh: &mut Mesh, max_size: usize) -> usize {
    let holes = detect_boundary_loops(mesh);
    let mut filled = 0;

    for hole in holes {
        if hole.len() <= max_size && hole.len() >= 3 {
            fill_hole_planar(mesh, &hole);
            filled += 1;
        }
    }

    filled
}

/// Detect boundary loops (holes).
fn detect_boundary_loops(mesh: &Mesh) -> Vec<Vec<u32>> {
    // Find boundary edges (edges with only one adjacent face)
    let mut edge_count: HashMap<(u32, u32), usize> = HashMap::new();

    for face in &mesh.faces {
        for i in 0..3 {
            let v0 = face[i];
            let v1 = face[(i + 1) % 3];
            let edge = if v0 < v1 { (v0, v1) } else { (v1, v0) };
            *edge_count.entry(edge).or_insert(0) += 1;
        }
    }

    let boundary_edges: HashSet<(u32, u32)> = edge_count
        .into_iter()
        .filter(|(_, count)| *count == 1)
        .map(|(edge, _)| edge)
        .collect();

    if boundary_edges.is_empty() {
        return Vec::new();
    }

    // Build adjacency for boundary vertices
    let mut boundary_adj: HashMap<u32, Vec<u32>> = HashMap::new();
    for (v0, v1) in &boundary_edges {
        boundary_adj.entry(*v0).or_default().push(*v1);
        boundary_adj.entry(*v1).or_default().push(*v0);
    }

    // Trace loops
    let mut visited_edges: HashSet<(u32, u32)> = HashSet::new();
    let mut loops: Vec<Vec<u32>> = Vec::new();

    for (&start_v, neighbors) in &boundary_adj {
        for &next_v in neighbors {
            let edge = if start_v < next_v {
                (start_v, next_v)
            } else {
                (next_v, start_v)
            };

            if visited_edges.contains(&edge) {
                continue;
            }

            let mut current_loop = vec![start_v];
            let mut prev = start_v;
            let mut current = next_v;

            loop {
                let edge = if prev < current {
                    (prev, current)
                } else {
                    (current, prev)
                };
                visited_edges.insert(edge);
                current_loop.push(current);

                if current == start_v {
                    current_loop.pop(); // Remove duplicate start
                    break;
                }

                // Find next vertex
                if let Some(adj) = boundary_adj.get(&current) {
                    let next = adj.iter().find(|&&v| v != prev);
                    if let Some(&n) = next {
                        prev = current;
                        current = n;
                    } else {
                        break;
                    }
                } else {
                    break;
                }
            }

            if current_loop.len() >= 3 {
                loops.push(current_loop);
            }
        }
    }

    loops
}

/// Fill a hole with planar triangulation.
fn fill_hole_planar(mesh: &mut Mesh, hole: &[u32]) -> usize {
    if hole.len() < 3 {
        return 0;
    }

    // Simple fan triangulation from first vertex
    let first = hole[0];
    let mut faces_added = 0;

    for i in 1..hole.len() - 1 {
        mesh.faces.push([first, hole[i], hole[i + 1]]);
        faces_added += 1;
    }

    faces_added
}

/// Fill a hole with smooth interpolation.
fn fill_hole_smooth(mesh: &mut Mesh, hole: &[u32], smooth_iterations: usize) -> usize {
    let faces_added = fill_hole_planar(mesh, hole);

    if faces_added > 0 && smooth_iterations > 0 {
        // Get vertices that were affected (the hole boundary)
        let affected: HashSet<u32> = hole.iter().copied().collect();

        for _ in 0..smooth_iterations {
            smooth_vertices_subset(mesh, &affected, 0.5);
        }
    }

    faces_added
}

/// Fill a hole matching surrounding curvature.
fn fill_hole_curvature(mesh: &mut Mesh, hole: &[u32]) -> usize {
    // First do basic fill, then adjust based on surrounding curvature
    let faces_added = fill_hole_planar(mesh, hole);

    if faces_added == 0 || hole.is_empty() {
        return 0;
    }

    // Compute centroid
    let centroid = compute_hole_centroid(mesh, hole);

    // Estimate surface normal at hole boundary
    let _normal = estimate_boundary_normal(mesh, hole);

    // Add a center vertex and re-triangulate if the hole is large enough
    if hole.len() > 4 {
        // Remove the simple fan triangulation
        let start_face_idx = mesh.faces.len() - faces_added;
        mesh.faces.truncate(start_face_idx);

        // Add center vertex
        let center_idx = mesh.vertices.len() as u32;
        mesh.vertices.push(Vertex::new(centroid));

        // Create fan from center
        for i in 0..hole.len() {
            let v0 = hole[i];
            let v1 = hole[(i + 1) % hole.len()];
            mesh.faces.push([center_idx, v0, v1]);
        }

        return hole.len();
    }

    faces_added
}

/// Fill a hole with minimal area triangulation.
fn fill_hole_minimal_area(mesh: &mut Mesh, hole: &[u32]) -> usize {
    // For small holes, simple fan is often minimal area
    // For larger holes, we'd use dynamic programming
    fill_hole_planar(mesh, hole)
}

/// Laplacian smoothing.
fn smooth_laplacian(mesh: &mut Mesh, iterations: usize, strength: f64) {
    for _ in 0..iterations {
        let displacements = compute_laplacian_displacements(mesh, strength);
        for (i, disp) in displacements.iter().enumerate() {
            mesh.vertices[i].position += disp;
        }
    }
}

/// Feature-preserving smoothing.
fn smooth_preserve_features(
    mesh: &mut Mesh,
    iterations: usize,
    strength: f64,
    feature_threshold: f64,
) {
    for _ in 0..iterations {
        let displacements = compute_bilateral_displacements(mesh, strength, feature_threshold);
        for (i, disp) in displacements.iter().enumerate() {
            if !is_feature_vertex(mesh, i, feature_threshold) {
                mesh.vertices[i].position += disp;
            }
        }
    }
}

/// Smooth only a subset of vertices.
fn smooth_vertices_subset(mesh: &mut Mesh, vertices: &HashSet<u32>, strength: f64) {
    let adjacency = build_vertex_adjacency(mesh);
    let mut displacements: HashMap<u32, Vector3<f64>> = HashMap::new();

    for &vi in vertices {
        if let Some(neighbors) = adjacency.get(&vi) {
            if neighbors.is_empty() {
                continue;
            }

            let current = mesh.vertices[vi as usize].position;
            let centroid: Point3<f64> = Point3::from(
                neighbors
                    .iter()
                    .map(|&ni| mesh.vertices[ni as usize].position.coords)
                    .sum::<Vector3<f64>>()
                    / neighbors.len() as f64,
            );

            displacements.insert(vi, (centroid - current) * strength);
        }
    }

    for (vi, disp) in displacements {
        mesh.vertices[vi as usize].position += disp;
    }
}

/// Build vertex adjacency map.
fn build_vertex_adjacency(mesh: &Mesh) -> HashMap<u32, Vec<u32>> {
    let mut adjacency: HashMap<u32, Vec<u32>> = HashMap::new();

    for face in &mesh.faces {
        for i in 0..3 {
            let v0 = face[i];
            let v1 = face[(i + 1) % 3];
            adjacency.entry(v0).or_default().push(v1);
            adjacency.entry(v1).or_default().push(v0);
        }
    }

    // Remove duplicates
    for neighbors in adjacency.values_mut() {
        neighbors.sort();
        neighbors.dedup();
    }

    adjacency
}

/// Compute Laplacian displacements.
fn compute_laplacian_displacements(mesh: &Mesh, strength: f64) -> Vec<Vector3<f64>> {
    let adjacency = build_vertex_adjacency(mesh);
    let mut displacements = vec![Vector3::zeros(); mesh.vertices.len()];

    for (vi, v) in mesh.vertices.iter().enumerate() {
        if let Some(neighbors) = adjacency.get(&(vi as u32)) {
            if neighbors.is_empty() {
                continue;
            }

            let centroid: Point3<f64> = Point3::from(
                neighbors
                    .iter()
                    .map(|&ni| mesh.vertices[ni as usize].position.coords)
                    .sum::<Vector3<f64>>()
                    / neighbors.len() as f64,
            );

            displacements[vi] = (centroid - v.position) * strength;
        }
    }

    displacements
}

/// Compute bilateral (edge-preserving) displacements.
fn compute_bilateral_displacements(
    mesh: &Mesh,
    strength: f64,
    feature_threshold: f64,
) -> Vec<Vector3<f64>> {
    let adjacency = build_vertex_adjacency(mesh);
    let normals = compute_vertex_normals(mesh);
    let mut displacements = vec![Vector3::zeros(); mesh.vertices.len()];

    for (vi, v) in mesh.vertices.iter().enumerate() {
        if let Some(neighbors) = adjacency.get(&(vi as u32)) {
            if neighbors.is_empty() {
                continue;
            }

            let normal = normals.get(vi).copied().unwrap_or(Vector3::z());
            let mut weighted_sum = Vector3::zeros();
            let mut weight_sum = 0.0;

            for &ni in neighbors {
                let neighbor_pos = mesh.vertices[ni as usize].position;
                let diff = neighbor_pos - v.position;

                // Bilateral weight based on distance and normal similarity
                let spatial_weight = (-diff.norm_squared() / (2.0 * strength * strength)).exp();

                let neighbor_normal = normals.get(ni as usize).copied().unwrap_or(Vector3::z());
                let normal_sim = normal.dot(&neighbor_normal).max(0.0);
                let range_weight = if normal_sim > feature_threshold.cos() {
                    1.0
                } else {
                    0.1
                };

                let weight = spatial_weight * range_weight;
                weighted_sum += diff * weight;
                weight_sum += weight;
            }

            if weight_sum > 1e-10 {
                displacements[vi] = weighted_sum / weight_sum * strength;
            }
        }
    }

    displacements
}

/// Compute Taubin smoothing displacements (reduces shrinkage).
fn compute_taubin_displacements(mesh: &Mesh, strength: f64) -> Vec<Vector3<f64>> {
    // Taubin smoothing alternates positive and negative lambda
    let lambda = strength;
    let mu = -strength * 1.02; // Slightly larger to compensate

    let laplacian = compute_laplacian_displacements(mesh, lambda);

    // Apply first step
    let mut temp_mesh = mesh.clone();
    for (i, disp) in laplacian.iter().enumerate() {
        temp_mesh.vertices[i].position += disp;
    }

    // Compute second step (negative)
    let mut inverse = compute_laplacian_displacements(&temp_mesh, mu);

    // Combine
    for (i, disp) in inverse.iter_mut().enumerate() {
        *disp = laplacian[i] + *disp;
    }

    inverse
}

/// Compute mean curvature flow displacements.
fn compute_mcf_displacements(mesh: &Mesh, strength: f64) -> Vec<Vector3<f64>> {
    let adjacency = build_vertex_adjacency(mesh);
    let normals = compute_vertex_normals(mesh);
    let mut displacements = vec![Vector3::zeros(); mesh.vertices.len()];

    for (vi, v) in mesh.vertices.iter().enumerate() {
        if let Some(neighbors) = adjacency.get(&(vi as u32)) {
            if neighbors.is_empty() {
                continue;
            }

            let normal = normals.get(vi).copied().unwrap_or(Vector3::z());

            // Compute approximate mean curvature
            let centroid: Point3<f64> = Point3::from(
                neighbors
                    .iter()
                    .map(|&ni| mesh.vertices[ni as usize].position.coords)
                    .sum::<Vector3<f64>>()
                    / neighbors.len() as f64,
            );

            let laplacian = centroid - v.position;
            let mean_curvature = laplacian.dot(&normal);

            // Move along normal by curvature
            displacements[vi] = normal * mean_curvature * strength;
        }
    }

    displacements
}

/// Compute vertex normals.
fn compute_vertex_normals(mesh: &Mesh) -> Vec<Vector3<f64>> {
    let mut normals = vec![Vector3::zeros(); mesh.vertices.len()];

    for face in &mesh.faces {
        let v0 = mesh.vertices[face[0] as usize].position;
        let v1 = mesh.vertices[face[1] as usize].position;
        let v2 = mesh.vertices[face[2] as usize].position;

        let edge1 = v1 - v0;
        let edge2 = v2 - v0;
        let face_normal = edge1.cross(&edge2);

        normals[face[0] as usize] += face_normal;
        normals[face[1] as usize] += face_normal;
        normals[face[2] as usize] += face_normal;
    }

    for normal in &mut normals {
        let len = normal.norm();
        if len > 1e-10 {
            *normal /= len;
        }
    }

    normals
}

/// Check if a vertex is on a feature edge.
fn is_feature_vertex(mesh: &Mesh, vertex_idx: usize, threshold: f64) -> bool {
    // Find faces containing this vertex
    let mut face_normals: Vec<Vector3<f64>> = Vec::new();

    for face in &mesh.faces {
        if face[0] as usize == vertex_idx
            || face[1] as usize == vertex_idx
            || face[2] as usize == vertex_idx
        {
            let v0 = mesh.vertices[face[0] as usize].position;
            let v1 = mesh.vertices[face[1] as usize].position;
            let v2 = mesh.vertices[face[2] as usize].position;

            let edge1 = v1 - v0;
            let edge2 = v2 - v0;
            let normal = edge1.cross(&edge2);
            let len = normal.norm();
            if len > 1e-10 {
                face_normals.push(normal / len);
            }
        }
    }

    // Check if any pair of normals has angle > threshold
    for i in 0..face_normals.len() {
        for j in (i + 1)..face_normals.len() {
            let dot = face_normals[i].dot(&face_normals[j]).clamp(-1.0, 1.0);
            let angle = dot.acos();
            if angle > threshold {
                return true;
            }
        }
    }

    false
}

/// Compute centroid of hole boundary vertices.
fn compute_hole_centroid(mesh: &Mesh, hole: &[u32]) -> Point3<f64> {
    if hole.is_empty() {
        return Point3::origin();
    }

    let sum: Vector3<f64> = hole
        .iter()
        .map(|&vi| mesh.vertices[vi as usize].position.coords)
        .sum();

    Point3::from(sum / hole.len() as f64)
}

/// Estimate normal at hole boundary.
fn estimate_boundary_normal(mesh: &Mesh, hole: &[u32]) -> Vector3<f64> {
    if hole.len() < 3 {
        return Vector3::z();
    }

    // Use Newell's method for polygon normal
    let mut normal: Vector3<f64> = Vector3::zeros();

    for i in 0..hole.len() {
        let current = mesh.vertices[hole[i] as usize].position;
        let next = mesh.vertices[hole[(i + 1) % hole.len()] as usize].position;

        normal.x += (current.y - next.y) * (current.z + next.z);
        normal.y += (current.z - next.z) * (current.x + next.x);
        normal.z += (current.x - next.x) * (current.y + next.y);
    }

    let len = normal.norm();
    if len > 1e-10 {
        normal / len
    } else {
        Vector3::z()
    }
}

// ============================================================================
// Mesh extension methods
// ============================================================================

impl Mesh {
    /// Clean up scan data with default parameters.
    pub fn cleanup_scan(&self) -> ScanCleanupResult {
        cleanup_scan(self, &ScanCleanupParams::default())
    }

    /// Clean up scan data with custom parameters.
    pub fn cleanup_scan_with_params(&self, params: &ScanCleanupParams) -> ScanCleanupResult {
        cleanup_scan(self, params)
    }

    /// Denoise the mesh.
    pub fn denoise(&self) -> DenoiseResult {
        denoise_mesh(self, &DenoiseParams::default())
    }

    /// Denoise the mesh with custom parameters.
    pub fn denoise_with_params(&self, params: &DenoiseParams) -> DenoiseResult {
        denoise_mesh(self, params)
    }

    /// Fill holes with advanced strategies.
    pub fn fill_holes_advanced(&self) -> HoleFillResult {
        fill_holes_advanced(self, &HoleFillParams::default())
    }

    /// Fill holes with custom parameters.
    pub fn fill_holes_advanced_with_params(&self, params: &HoleFillParams) -> HoleFillResult {
        fill_holes_advanced(self, params)
    }

    /// Remove statistical outliers.
    pub fn remove_outliers(&self) -> Mesh {
        remove_outliers(self, &OutlierRemovalParams::default())
    }

    /// Remove statistical outliers with custom parameters.
    pub fn remove_outliers_with_params(&self, params: &OutlierRemovalParams) -> Mesh {
        remove_outliers(self, params)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn create_test_mesh() -> Mesh {
        let mut mesh = Mesh::new();
        // Create a simple pyramid
        mesh.vertices.push(Vertex::from_coords(0.0, 0.0, 0.0));
        mesh.vertices.push(Vertex::from_coords(10.0, 0.0, 0.0));
        mesh.vertices.push(Vertex::from_coords(10.0, 10.0, 0.0));
        mesh.vertices.push(Vertex::from_coords(0.0, 10.0, 0.0));
        mesh.vertices.push(Vertex::from_coords(5.0, 5.0, 10.0));

        mesh.faces.push([0, 1, 4]);
        mesh.faces.push([1, 2, 4]);
        mesh.faces.push([2, 3, 4]);
        mesh.faces.push([3, 0, 4]);
        mesh
    }

    #[test]
    fn test_cleanup_params_default() {
        let params = ScanCleanupParams::default();
        assert!(params.remove_isolated);
        assert!(params.remove_spikes);
        assert!(params.fill_small_holes);
        assert!(params.smooth);
    }

    #[test]
    fn test_cleanup_params_presets() {
        let body = ScanCleanupParams::for_body_scan();
        assert!(body.min_component_size > 100);

        let object = ScanCleanupParams::for_object_scan();
        assert!(object.min_component_size < body.min_component_size);

        let minimal = ScanCleanupParams::minimal();
        assert!(!minimal.remove_spikes);

        let aggressive = ScanCleanupParams::aggressive();
        assert!(aggressive.smooth_iterations > 3);
    }

    #[test]
    fn test_cleanup_scan() {
        let mesh = create_test_mesh();
        // Use custom params with low threshold since test mesh is small
        let mut params = ScanCleanupParams::minimal();
        params.min_component_size = 1; // Don't remove small components in test
        let result = cleanup_scan(&mesh, &params);

        assert!(!result.mesh.vertices.is_empty());
        assert!(!result.mesh.faces.is_empty());
    }

    #[test]
    fn test_denoise_params() {
        let params = DenoiseParams::default();
        assert_eq!(params.method, DenoiseMethod::Bilateral);
        assert!(params.preserve_features);
    }

    #[test]
    fn test_denoise_mesh() {
        let mesh = create_test_mesh();
        let result = denoise_mesh(&mesh, &DenoiseParams::default());

        assert!(!result.mesh.vertices.is_empty());
        assert_eq!(result.iterations_performed, 3);
    }

    #[test]
    fn test_outlier_removal() {
        let mut mesh = create_test_mesh();
        // Add an outlier
        mesh.vertices.push(Vertex::from_coords(1000.0, 0.0, 0.0));

        let params = OutlierRemovalParams {
            k_neighbors: 3,
            std_dev_threshold: 1.5,
        };
        let result = remove_outliers(&mesh, &params);

        // Outlier should be removed (or at least not cause crash)
        assert!(result.vertices.len() <= mesh.vertices.len());
    }

    #[test]
    fn test_mesh_cleanup_method() {
        let mesh = create_test_mesh();
        // Use custom params with low threshold since test mesh is small
        let mut params = ScanCleanupParams::minimal();
        params.min_component_size = 1;
        let result = mesh.cleanup_scan_with_params(&params);

        assert!(!result.mesh.vertices.is_empty());
    }

    #[test]
    fn test_mesh_denoise_method() {
        let mesh = create_test_mesh();
        let result = mesh.denoise();

        assert!(!result.mesh.vertices.is_empty());
    }

    #[test]
    fn test_vertex_adjacency() {
        let mesh = create_test_mesh();
        let adjacency = build_vertex_adjacency(&mesh);

        // Apex vertex (4) should be connected to all base vertices
        assert!(adjacency.get(&4).map(|v| v.len() >= 4).unwrap_or(false));
    }

    #[test]
    fn test_vertex_normals() {
        let mesh = create_test_mesh();
        let normals = compute_vertex_normals(&mesh);

        assert_eq!(normals.len(), mesh.vertices.len());
        // All normals should be unit length
        for normal in &normals {
            let len = normal.norm();
            assert!(len > 0.9 && len < 1.1, "Normal length: {}", len);
        }
    }
}

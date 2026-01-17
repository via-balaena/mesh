//! Mesh repair operations: degenerate removal, welding, compaction.

use hashbrown::{HashMap, HashSet};
use nalgebra::Point3;
use tracing::{debug, info, warn};

use crate::adjacency::MeshAdjacency;
use crate::error::MeshResult;
use crate::holes::fill_holes_with_max_edges;
use crate::winding::fix_winding_order;
use crate::{Mesh, Triangle};

/// Configuration parameters for mesh repair operations.
///
/// All thresholds are in the same units as the mesh coordinates (typically millimeters).
///
/// # Example
///
/// ```
/// use mesh_repair::RepairParams;
///
/// // Use defaults (good for mm-scale meshes)
/// let params = RepairParams::default();
///
/// // Or customize for your use case
/// let params = RepairParams {
///     weld_epsilon: 0.01,  // More aggressive welding for noisy scans
///     degenerate_area_threshold: 0.001,
///     ..Default::default()
/// };
/// ```
#[derive(Debug, Clone)]
#[cfg_attr(
    feature = "pipeline-config",
    derive(serde::Serialize, serde::Deserialize)
)]
pub struct RepairParams {
    /// Distance threshold for vertex welding.
    ///
    /// Vertices closer than this distance will be merged into one.
    /// Larger values are more aggressive and may merge intentional detail.
    /// Smaller values preserve more detail but may leave gaps.
    ///
    /// Default: `1e-6` (0.000001 mm, extremely conservative)
    ///
    /// Recommended ranges:
    /// - High-precision CAD: `1e-9` to `1e-6`
    /// - 3D scans: `0.001` to `0.1` (depending on scan noise)
    /// - Low-poly models: `0.01` to `1.0`
    pub weld_epsilon: f64,

    /// Minimum triangle area threshold.
    ///
    /// Triangles with area below this threshold are considered degenerate
    /// and will be removed. Very small triangles often cause numerical
    /// issues in downstream processing.
    ///
    /// Default: `1e-9` (effectively zero for mm-scale meshes)
    pub degenerate_area_threshold: f64,

    /// Maximum triangle aspect ratio threshold.
    ///
    /// Triangles with aspect ratio (longest edge / shortest altitude) above
    /// this threshold are considered degenerate "sliver" triangles.
    /// Set to `f64::INFINITY` to disable this check.
    ///
    /// Default: `1000.0` (very thin triangles are removed)
    pub degenerate_aspect_ratio: f64,

    /// Minimum edge length threshold.
    ///
    /// Triangles with any edge shorter than this are considered degenerate.
    /// Set to `0.0` to disable this check.
    ///
    /// Default: `1e-9` (effectively zero)
    pub degenerate_min_edge_length: f64,

    /// Maximum number of edges in a hole to auto-fill.
    ///
    /// Holes with more edges than this will be skipped (with a warning).
    /// Larger holes often require manual intervention or more sophisticated
    /// filling algorithms.
    ///
    /// Default: `100`
    pub max_hole_edges: usize,

    /// Whether to attempt to fix winding order inconsistencies.
    ///
    /// When enabled, the repair pipeline will try to make all face normals
    /// point consistently outward. This requires the mesh to have a clear
    /// "outside" direction.
    ///
    /// Default: `true`
    pub fix_winding: bool,

    /// Whether to remove non-manifold edges.
    ///
    /// Non-manifold edges are edges shared by more than 2 faces.
    /// When enabled, excess faces are removed (keeping the 2 largest).
    ///
    /// Default: `true`
    pub fix_non_manifold: bool,

    /// Whether to fill holes after other repairs.
    ///
    /// When enabled, boundary loops (holes) up to `max_hole_edges` in size
    /// will be filled using ear-clipping triangulation.
    ///
    /// Default: `false` (holes are often intentional)
    pub fill_holes: bool,

    /// Whether to compute vertex normals after repair.
    ///
    /// Default: `true`
    pub compute_normals: bool,

    /// Whether to remove unreferenced vertices after repair.
    ///
    /// Default: `true`
    pub remove_unreferenced: bool,
}

impl Default for RepairParams {
    fn default() -> Self {
        Self {
            weld_epsilon: 1e-6,
            degenerate_area_threshold: 1e-9,
            degenerate_aspect_ratio: 1000.0,
            degenerate_min_edge_length: 1e-9,
            max_hole_edges: 100,
            fix_winding: true,
            fix_non_manifold: true,
            fill_holes: false,
            compute_normals: true,
            remove_unreferenced: true,
        }
    }
}

impl RepairParams {
    /// Create params optimized for 3D scan data.
    ///
    /// Uses more aggressive welding and degenerate removal suitable
    /// for noisy scan data (e.g., from structured light or photogrammetry).
    pub fn for_scans() -> Self {
        Self {
            weld_epsilon: 0.01,                // 0.01mm - typical scan noise level
            degenerate_area_threshold: 0.0001, // 0.0001 mm²
            degenerate_aspect_ratio: 100.0,    // More aggressive sliver removal
            degenerate_min_edge_length: 0.001, // 0.001mm minimum edge
            max_hole_edges: 200,               // Allow larger hole filling
            fill_holes: true,                  // Auto-fill small holes from scan gaps
            ..Default::default()
        }
    }

    /// Create params optimized for CAD models.
    ///
    /// Uses conservative settings to preserve intentional geometry.
    pub fn for_cad() -> Self {
        Self {
            weld_epsilon: 1e-9,
            degenerate_area_threshold: 1e-12,
            degenerate_aspect_ratio: f64::INFINITY, // Don't remove thin triangles
            degenerate_min_edge_length: 0.0,
            max_hole_edges: 50,
            fill_holes: false, // CAD holes are usually intentional
            ..Default::default()
        }
    }

    /// Create params optimized for 3D printing preparation.
    ///
    /// Ensures watertight, manifold output suitable for slicing.
    pub fn for_printing() -> Self {
        Self {
            weld_epsilon: 0.001,                // 0.001mm
            degenerate_area_threshold: 0.00001, // 0.00001 mm²
            degenerate_aspect_ratio: 500.0,
            degenerate_min_edge_length: 0.0001,
            max_hole_edges: 500, // Fill even large holes
            fix_winding: true,
            fix_non_manifold: true,
            fill_holes: true, // Important for watertight output
            compute_normals: true,
            remove_unreferenced: true,
        }
    }
}

/// Remove triangles with area below threshold.
///
/// Returns the number of triangles removed.
pub fn remove_degenerate_triangles(mesh: &mut Mesh, area_threshold: f64) -> usize {
    let original_count = mesh.faces.len();

    mesh.faces.retain(|&[i0, i1, i2]| {
        let tri = Triangle::new(
            mesh.vertices[i0 as usize].position,
            mesh.vertices[i1 as usize].position,
            mesh.vertices[i2 as usize].position,
        );
        tri.area() >= area_threshold
    });

    let removed = original_count - mesh.faces.len();
    if removed > 0 {
        info!(
            "Removed {} degenerate triangles (area < {:.6})",
            removed, area_threshold
        );
    }
    removed
}

/// Remove degenerate triangles using multiple criteria.
///
/// A triangle is considered degenerate if:
/// - Area is below `area_threshold`
/// - Aspect ratio exceeds `max_aspect_ratio` (unless set to infinity)
/// - Any edge is shorter than `min_edge_length` (unless set to 0)
///
/// Returns the number of triangles removed.
pub fn remove_degenerate_triangles_enhanced(
    mesh: &mut Mesh,
    area_threshold: f64,
    max_aspect_ratio: f64,
    min_edge_length: f64,
) -> usize {
    let original_count = mesh.faces.len();

    mesh.faces.retain(|&[i0, i1, i2]| {
        let p0 = mesh.vertices[i0 as usize].position;
        let p1 = mesh.vertices[i1 as usize].position;
        let p2 = mesh.vertices[i2 as usize].position;

        let tri = Triangle::new(p0, p1, p2);

        // Check area
        let area = tri.area();
        if area < area_threshold {
            return false;
        }

        // Check edge lengths if threshold is set
        if min_edge_length > 0.0 {
            let e0 = (p1 - p0).norm();
            let e1 = (p2 - p1).norm();
            let e2 = (p0 - p2).norm();
            if e0 < min_edge_length || e1 < min_edge_length || e2 < min_edge_length {
                return false;
            }
        }

        // Check aspect ratio if threshold is finite
        if max_aspect_ratio.is_finite() {
            let e0 = (p1 - p0).norm();
            let e1 = (p2 - p1).norm();
            let e2 = (p0 - p2).norm();
            let longest_edge = e0.max(e1).max(e2);

            // Aspect ratio = longest edge / (2 * area / longest edge)
            // = longest_edge^2 / (2 * area)
            if area > 0.0 {
                let aspect = (longest_edge * longest_edge) / (2.0 * area);
                if aspect > max_aspect_ratio {
                    return false;
                }
            }
        }

        true
    });

    let removed = original_count - mesh.faces.len();
    if removed > 0 {
        info!(
            "Removed {} degenerate triangles (area<{:.2e}, aspect>{:.0}, edge<{:.2e})",
            removed, area_threshold, max_aspect_ratio, min_edge_length
        );
    }
    removed
}

/// Weld vertices that are within epsilon distance of each other.
///
/// Uses spatial hashing for efficiency. Returns the number of vertices merged.
pub fn weld_vertices(mesh: &mut Mesh, epsilon: f64) -> usize {
    let original_count = mesh.vertices.len();
    if original_count == 0 {
        return 0;
    }

    // Cell size for spatial hashing (2x epsilon as recommended)
    let cell_size = epsilon * 2.0;

    // Build spatial hash: cell -> list of vertex indices
    let mut spatial_hash: HashMap<(i64, i64, i64), Vec<u32>> = HashMap::new();

    for (idx, vertex) in mesh.vertices.iter().enumerate() {
        let cell = pos_to_cell(&vertex.position, cell_size);
        spatial_hash.entry(cell).or_default().push(idx as u32);
    }

    // For each vertex, find its canonical representative (smallest index in cluster)
    let mut vertex_remap: Vec<u32> = (0..mesh.vertices.len() as u32).collect();
    let mut merged_count = 0;

    for (idx, vertex) in mesh.vertices.iter().enumerate() {
        let idx = idx as u32;
        if vertex_remap[idx as usize] != idx {
            // Already merged into another vertex
            continue;
        }

        let cell = pos_to_cell(&vertex.position, cell_size);

        // Check 3x3x3 neighborhood
        for dx in -1..=1 {
            for dy in -1..=1 {
                for dz in -1..=1 {
                    let neighbor_cell = (cell.0 + dx, cell.1 + dy, cell.2 + dz);

                    if let Some(candidates) = spatial_hash.get(&neighbor_cell) {
                        for &other_idx in candidates {
                            if other_idx <= idx {
                                continue; // Only merge into smaller indices
                            }
                            if vertex_remap[other_idx as usize] != other_idx {
                                continue; // Already merged
                            }

                            let other_pos = &mesh.vertices[other_idx as usize].position;
                            let dist = (vertex.position - other_pos).norm();

                            if dist < epsilon {
                                vertex_remap[other_idx as usize] = idx;
                                merged_count += 1;
                            }
                        }
                    }
                }
            }
        }
    }

    if merged_count == 0 {
        return 0;
    }

    // Resolve transitive merges (A->B, B->C => A->C)
    for i in 0..vertex_remap.len() {
        let mut target = vertex_remap[i];
        while vertex_remap[target as usize] != target {
            target = vertex_remap[target as usize];
        }
        vertex_remap[i] = target;
    }

    // Remap face indices
    for face in &mut mesh.faces {
        face[0] = vertex_remap[face[0] as usize];
        face[1] = vertex_remap[face[1] as usize];
        face[2] = vertex_remap[face[2] as usize];
    }

    // Remove faces that became degenerate after welding
    mesh.faces
        .retain(|&[i0, i1, i2]| i0 != i1 && i1 != i2 && i0 != i2);

    info!(
        "Welded {} vertices (epsilon = {:.3}): {} → {}",
        merged_count,
        epsilon,
        original_count,
        original_count - merged_count
    );

    merged_count
}

/// Remove unreferenced vertices and compact the vertex array.
///
/// Returns the number of vertices removed.
pub fn remove_unreferenced_vertices(mesh: &mut Mesh) -> usize {
    let original_count = mesh.vertices.len();

    // Find all referenced vertices
    let mut referenced: HashSet<u32> = HashSet::new();
    for face in &mesh.faces {
        referenced.insert(face[0]);
        referenced.insert(face[1]);
        referenced.insert(face[2]);
    }

    if referenced.len() == original_count {
        return 0; // All vertices are referenced
    }

    // Build compacted vertex list and remap
    let mut new_vertices = Vec::with_capacity(referenced.len());
    let mut remap: HashMap<u32, u32> = HashMap::new();

    for (old_idx, vertex) in mesh.vertices.iter().enumerate() {
        if referenced.contains(&(old_idx as u32)) {
            let new_idx = new_vertices.len() as u32;
            remap.insert(old_idx as u32, new_idx);
            new_vertices.push(vertex.clone());
        }
    }

    // Remap face indices
    for face in &mut mesh.faces {
        face[0] = remap[&face[0]];
        face[1] = remap[&face[1]];
        face[2] = remap[&face[2]];
    }

    let removed = original_count - new_vertices.len();
    mesh.vertices = new_vertices;

    if removed > 0 {
        info!("Removed {} unreferenced vertices", removed);
    }

    removed
}

/// Compute vertex normals as area-weighted average of adjacent face normals.
pub fn compute_vertex_normals(mesh: &mut Mesh) {
    // Reset all normals
    for vertex in &mut mesh.vertices {
        vertex.normal = None;
    }

    // Accumulate face normals weighted by area
    let mut normal_accum: Vec<nalgebra::Vector3<f64>> =
        vec![nalgebra::Vector3::zeros(); mesh.vertices.len()];

    for face in &mesh.faces {
        let tri = Triangle::new(
            mesh.vertices[face[0] as usize].position,
            mesh.vertices[face[1] as usize].position,
            mesh.vertices[face[2] as usize].position,
        );

        // Use unnormalized normal (length = 2*area) for area weighting
        let weighted_normal = tri.normal_unnormalized();

        normal_accum[face[0] as usize] += weighted_normal;
        normal_accum[face[1] as usize] += weighted_normal;
        normal_accum[face[2] as usize] += weighted_normal;
    }

    // Normalize and assign
    for (idx, accum) in normal_accum.into_iter().enumerate() {
        let len_sq = accum.norm_squared();
        if len_sq > f64::EPSILON {
            mesh.vertices[idx].normal = Some(accum / len_sq.sqrt());
        }
    }

    debug!(
        "Computed vertex normals for {} vertices",
        mesh.vertices.len()
    );
}

/// Convert position to spatial hash cell.
fn pos_to_cell(pos: &Point3<f64>, cell_size: f64) -> (i64, i64, i64) {
    (
        (pos.x / cell_size).floor() as i64,
        (pos.y / cell_size).floor() as i64,
        (pos.z / cell_size).floor() as i64,
    )
}

/// Remove duplicate faces from the mesh.
///
/// Faces are considered duplicate if they have the same set of vertices
/// (regardless of winding order or starting vertex). This function removes
/// all copies except the first occurrence.
///
/// Returns the number of duplicate faces removed.
pub fn remove_duplicate_faces(mesh: &mut Mesh) -> usize {
    let original_count = mesh.faces.len();

    // Normalize face to smallest vertex first, maintaining cyclic order
    fn normalize_face(face: [u32; 3]) -> [u32; 3] {
        let mut min_idx = 0;
        for i in 1..3 {
            if face[i] < face[min_idx] {
                min_idx = i;
            }
        }
        [
            face[min_idx],
            face[(min_idx + 1) % 3],
            face[(min_idx + 2) % 3],
        ]
    }

    let mut seen: HashSet<[u32; 3]> = HashSet::new();
    let mut duplicate_indices: HashSet<usize> = HashSet::new();

    for (i, face) in mesh.faces.iter().enumerate() {
        let fwd = normalize_face(*face);
        let rev = normalize_face([face[0], face[2], face[1]]);

        // Check if we've seen this face (either winding direction)
        if seen.contains(&fwd) || seen.contains(&rev) {
            duplicate_indices.insert(i);
        } else {
            seen.insert(fwd);
        }
    }

    if duplicate_indices.is_empty() {
        return 0;
    }

    // Remove duplicates by retaining only non-duplicate faces
    let mut idx = 0;
    mesh.faces.retain(|_| {
        let keep = !duplicate_indices.contains(&idx);
        idx += 1;
        keep
    });

    let removed = original_count - mesh.faces.len();
    if removed > 0 {
        info!("Removed {} duplicate faces", removed);
    }

    removed
}

/// Fix non-manifold edges by removing excess faces.
///
/// Non-manifold edges are edges shared by more than 2 faces. For each such edge,
/// this function keeps the 2 largest-area faces and removes the rest.
///
/// Returns the number of faces removed.
pub fn fix_non_manifold_edges(mesh: &mut Mesh) -> usize {
    let adjacency = MeshAdjacency::build(&mesh.faces);
    let nm_edges: Vec<(u32, u32)> = adjacency.non_manifold_edges().collect();

    if nm_edges.is_empty() {
        return 0;
    }

    debug!("Found {} non-manifold edges to fix", nm_edges.len());

    let mut faces_to_remove: HashSet<usize> = HashSet::new();

    for (v0, v1) in &nm_edges {
        // Find all faces sharing this edge
        let mut faces_with_edge: Vec<(usize, f64)> = Vec::new();

        for (fi, face) in mesh.faces.iter().enumerate() {
            let has_v0 = face.contains(v0);
            let has_v1 = face.contains(v1);
            if has_v0 && has_v1 {
                // Compute area
                let p0 = mesh.vertices[face[0] as usize].position;
                let p1 = mesh.vertices[face[1] as usize].position;
                let p2 = mesh.vertices[face[2] as usize].position;
                let tri = Triangle::new(p0, p1, p2);
                let area = tri.area();
                faces_with_edge.push((fi, area));
            }
        }

        if faces_with_edge.len() <= 2 {
            continue; // Not actually non-manifold
        }

        // Sort by area descending (keep largest)
        faces_with_edge.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

        // Mark all but the 2 largest for removal
        for (fi, _area) in faces_with_edge.iter().skip(2) {
            faces_to_remove.insert(*fi);
        }
    }

    if faces_to_remove.is_empty() {
        return 0;
    }

    let removed_count = faces_to_remove.len();

    // Remove marked faces
    let mut idx = 0;
    mesh.faces.retain(|_| {
        let keep = !faces_to_remove.contains(&idx);
        idx += 1;
        keep
    });

    info!(
        "Fixed {} non-manifold edges by removing {} faces",
        nm_edges.len(),
        removed_count
    );

    removed_count
}

/// Fix inverted triangles by flipping their winding order.
///
/// Compares each face's normal against its original normal direction.
/// If the normal has flipped (due to offset or other operations), the face
/// winding is corrected by swapping indices 1 and 2.
///
/// # Arguments
/// * `mesh` - The mesh to fix (modified in place)
/// * `original` - The original mesh to compare against (same topology expected)
///
/// # Returns
/// The number of faces that were flipped.
pub fn fix_inverted_faces(mesh: &mut Mesh, original: &Mesh) -> usize {
    if mesh.faces.len() != original.faces.len() {
        // Topology mismatch - can't fix
        return 0;
    }

    let mut flipped_count = 0;

    for (i, face) in mesh.faces.iter_mut().enumerate() {
        let v0 = &mesh.vertices[face[0] as usize].position;
        let v1 = &mesh.vertices[face[1] as usize].position;
        let v2 = &mesh.vertices[face[2] as usize].position;

        let e1 = v1 - v0;
        let e2 = v2 - v0;
        let new_normal = e1.cross(&e2);

        // Skip degenerate faces
        if new_normal.norm_squared() < 1e-20 {
            continue;
        }

        let orig_face = &original.faces[i];
        let ov0 = &original.vertices[orig_face[0] as usize].position;
        let ov1 = &original.vertices[orig_face[1] as usize].position;
        let ov2 = &original.vertices[orig_face[2] as usize].position;

        let orig_e1 = ov1 - ov0;
        let orig_e2 = ov2 - ov0;
        let orig_normal = orig_e1.cross(&orig_e2);

        // Skip degenerate original faces
        if orig_normal.norm_squared() < 1e-20 {
            continue;
        }

        // If normals point opposite directions, face is inverted
        if new_normal.dot(&orig_normal) < 0.0 {
            // Flip winding by swapping indices 1 and 2
            face.swap(1, 2);
            flipped_count += 1;
        }
    }

    if flipped_count > 0 {
        info!(
            "Fixed {} inverted faces by flipping winding order",
            flipped_count
        );
    }

    flipped_count
}

/// Run the full repair pipeline on a mesh using default parameters.
///
/// This is equivalent to `repair_mesh_with_config(mesh, &RepairParams::default())`.
///
/// # Repair Steps
///
/// 1. Remove degenerate triangles (area, aspect ratio, edge length checks)
/// 2. Weld nearby vertices
/// 3. Remove duplicate faces
/// 4. Fix non-manifold edges (optional)
/// 5. Fix winding order (optional)
/// 6. Fill holes (optional)
/// 7. Remove unreferenced vertices
/// 8. Compute vertex normals
///
/// # Example
///
/// ```
/// use mesh_repair::{Mesh, repair_mesh};
///
/// let mut mesh = Mesh::new();
/// // ... populate mesh ...
/// repair_mesh(&mut mesh).unwrap();
/// ```
pub fn repair_mesh(mesh: &mut Mesh) -> MeshResult<()> {
    repair_mesh_with_config(mesh, &RepairParams::default())
}

/// Run the full repair pipeline on a mesh with custom parameters.
///
/// # Deprecated
///
/// Use `repair_mesh_with_config` instead for more control.
///
/// # Arguments
/// * `mesh` - The mesh to repair
/// * `weld_epsilon` - Distance threshold for welding vertices
/// * `degenerate_threshold` - Area threshold for removing degenerate triangles
#[deprecated(
    since = "0.2.0",
    note = "Use repair_mesh_with_config with RepairParams instead"
)]
pub fn repair_mesh_with_params(
    mesh: &mut Mesh,
    weld_epsilon: f64,
    degenerate_threshold: f64,
) -> MeshResult<()> {
    let params = RepairParams {
        weld_epsilon,
        degenerate_area_threshold: degenerate_threshold,
        ..Default::default()
    };
    repair_mesh_with_config(mesh, &params)
}

/// Run the full repair pipeline on a mesh with configurable parameters.
///
/// This is the main entry point for mesh repair with full control over
/// all repair parameters.
///
/// # Arguments
/// * `mesh` - The mesh to repair (modified in place)
/// * `params` - Configuration parameters for repair operations
///
/// # Example
///
/// ```
/// use mesh_repair::{Mesh, RepairParams, repair_mesh_with_config};
///
/// let mut mesh = Mesh::new();
/// // ... populate mesh ...
///
/// // Use scan-optimized parameters
/// let params = RepairParams::for_scans();
/// repair_mesh_with_config(&mut mesh, &params).unwrap();
/// ```
pub fn repair_mesh_with_config(mesh: &mut Mesh, params: &RepairParams) -> MeshResult<()> {
    info!(
        "Starting mesh repair pipeline (weld={:.2e}, area={:.2e}, aspect={:.0})",
        params.weld_epsilon, params.degenerate_area_threshold, params.degenerate_aspect_ratio
    );

    let initial_verts = mesh.vertex_count();
    let initial_faces = mesh.face_count();

    if initial_faces == 0 {
        warn!("Mesh has no faces, skipping repair");
        return Ok(());
    }

    // 1. Remove degenerate triangles (enhanced version with multiple criteria)
    remove_degenerate_triangles_enhanced(
        mesh,
        params.degenerate_area_threshold,
        params.degenerate_aspect_ratio,
        params.degenerate_min_edge_length,
    );

    // 2. Weld vertices
    weld_vertices(mesh, params.weld_epsilon);

    // 3. Remove duplicate faces (welding can create duplicates)
    remove_duplicate_faces(mesh);

    // 4. Fix non-manifold edges (optional)
    if params.fix_non_manifold {
        fix_non_manifold_edges(mesh);
    }

    // 5. Fix winding order (optional)
    if params.fix_winding
        && let Err(e) = fix_winding_order(mesh)
    {
        warn!("Could not fix winding order: {:?}", e);
    }

    // 6. Fill holes (optional)
    if params.fill_holes {
        match fill_holes_with_max_edges(mesh, params.max_hole_edges) {
            Ok(filled) => {
                if filled > 0 {
                    debug!("Filled {} holes", filled);
                }
            }
            Err(e) => {
                warn!("Could not fill holes: {:?}", e);
            }
        }
    }

    // 7. Remove unreferenced vertices (optional but usually wanted)
    if params.remove_unreferenced {
        remove_unreferenced_vertices(mesh);
    }

    // 8. Compute vertex normals (optional)
    if params.compute_normals {
        compute_vertex_normals(mesh);
    }

    info!(
        "Repair complete: {} verts → {}, {} faces → {}",
        initial_verts,
        mesh.vertex_count(),
        initial_faces,
        mesh.face_count()
    );

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::Vertex;
    use approx::assert_relative_eq;

    fn simple_mesh() -> Mesh {
        let mut mesh = Mesh::new();
        mesh.vertices.push(Vertex::from_coords(0.0, 0.0, 0.0));
        mesh.vertices.push(Vertex::from_coords(10.0, 0.0, 0.0));
        mesh.vertices.push(Vertex::from_coords(0.0, 10.0, 0.0));
        mesh.faces.push([0, 1, 2]);
        mesh
    }

    #[test]
    fn test_remove_degenerate_triangles() {
        let mut mesh = Mesh::new();
        mesh.vertices.push(Vertex::from_coords(0.0, 0.0, 0.0));
        mesh.vertices.push(Vertex::from_coords(10.0, 0.0, 0.0));
        mesh.vertices.push(Vertex::from_coords(0.0, 10.0, 0.0));
        // Normal triangle
        mesh.faces.push([0, 1, 2]);
        // Degenerate triangle (collinear points)
        mesh.vertices.push(Vertex::from_coords(5.0, 0.0, 0.0));
        mesh.faces.push([0, 1, 3]); // This has zero area

        let removed = remove_degenerate_triangles(&mut mesh, 0.0001);
        assert_eq!(removed, 1);
        assert_eq!(mesh.face_count(), 1);
    }

    #[test]
    fn test_weld_vertices() {
        let mut mesh = Mesh::new();
        // Two triangles with nearly-coincident vertices
        // 5 vertices total, vertex 3 is a near-duplicate of vertex 1
        mesh.vertices.push(Vertex::from_coords(0.0, 0.0, 0.0)); // 0
        mesh.vertices.push(Vertex::from_coords(10.0, 0.0, 0.0)); // 1
        mesh.vertices.push(Vertex::from_coords(0.0, 10.0, 0.0)); // 2
        mesh.vertices.push(Vertex::from_coords(10.001, 0.0, 0.0)); // 3 (near-duplicate of 1)
        mesh.vertices.push(Vertex::from_coords(10.0, 10.0, 0.0)); // 4

        mesh.faces.push([0, 1, 2]);
        mesh.faces.push([3, 2, 4]); // Uses near-duplicate

        let merged = weld_vertices(&mut mesh, 0.01);
        assert_eq!(merged, 1);

        // After merging, vertex 3 should be remapped to vertex 1
        // So all face indices should be valid (< 5, the original vertex count)
        // and the second face should now reference vertex 1 instead of 3
        assert!(
            mesh.faces
                .iter()
                .all(|f| f[0] <= 4 && f[1] <= 4 && f[2] <= 4)
        );

        // Second face should have been remapped: [3, 2, 4] -> [1, 2, 4]
        assert_eq!(mesh.faces[1][0], 1); // Vertex 3 was merged into vertex 1
    }

    #[test]
    fn test_remove_unreferenced() {
        let mut mesh = simple_mesh();
        // Add unreferenced vertex
        mesh.vertices.push(Vertex::from_coords(100.0, 100.0, 100.0));

        let removed = remove_unreferenced_vertices(&mut mesh);
        assert_eq!(removed, 1);
        assert_eq!(mesh.vertex_count(), 3);
    }

    #[test]
    fn test_compute_vertex_normals() {
        // Triangle in XY plane
        let mut mesh = simple_mesh();
        compute_vertex_normals(&mut mesh);

        // All vertices should have normal pointing in +Z
        for v in &mesh.vertices {
            let n = v.normal.expect("should have normal");
            assert_relative_eq!(n.x, 0.0, epsilon = 1e-10);
            assert_relative_eq!(n.y, 0.0, epsilon = 1e-10);
            assert_relative_eq!(n.z, 1.0, epsilon = 1e-10);
        }
    }

    #[test]
    fn test_fix_inverted_faces() {
        // Create a simple triangle in XY plane
        let mut original = Mesh::new();
        original.vertices.push(Vertex::from_coords(0.0, 0.0, 0.0));
        original.vertices.push(Vertex::from_coords(10.0, 0.0, 0.0));
        original.vertices.push(Vertex::from_coords(0.0, 10.0, 0.0));
        original.faces.push([0, 1, 2]); // CCW winding, normal points +Z

        // Create an "offset" mesh with inverted winding (simulates bad offset)
        let mut mesh = original.clone();
        // Swap vertices 1 and 2 to invert the winding
        mesh.faces[0] = [0, 2, 1]; // Now CW winding, normal points -Z

        // The face should be inverted (dot product of normals < 0)
        let v0 = &mesh.vertices[mesh.faces[0][0] as usize].position;
        let v1 = &mesh.vertices[mesh.faces[0][1] as usize].position;
        let v2 = &mesh.vertices[mesh.faces[0][2] as usize].position;
        let e1 = v1 - v0;
        let e2 = v2 - v0;
        let new_normal = e1.cross(&e2);

        let ov0 = &original.vertices[original.faces[0][0] as usize].position;
        let ov1 = &original.vertices[original.faces[0][1] as usize].position;
        let ov2 = &original.vertices[original.faces[0][2] as usize].position;
        let orig_e1 = ov1 - ov0;
        let orig_e2 = ov2 - ov0;
        let orig_normal = orig_e1.cross(&orig_e2);

        // Verify the face is inverted before fix
        assert!(
            new_normal.dot(&orig_normal) < 0.0,
            "Face should be inverted before fix"
        );

        // Fix the inverted face
        let fixed_count = fix_inverted_faces(&mut mesh, &original);
        assert_eq!(fixed_count, 1, "Should fix 1 face");

        // After fix, the face should have correct winding
        assert_eq!(
            mesh.faces[0],
            [0, 1, 2],
            "Face should be restored to original winding"
        );
    }

    #[test]
    fn test_fix_inverted_faces_no_change_needed() {
        // Create a simple triangle in XY plane
        let mut original = Mesh::new();
        original.vertices.push(Vertex::from_coords(0.0, 0.0, 0.0));
        original.vertices.push(Vertex::from_coords(10.0, 0.0, 0.0));
        original.vertices.push(Vertex::from_coords(0.0, 10.0, 0.0));
        original.faces.push([0, 1, 2]);

        // Create a mesh with same winding (no inversion)
        let mut mesh = original.clone();

        // No fix should be needed
        let fixed_count = fix_inverted_faces(&mut mesh, &original);
        assert_eq!(fixed_count, 0, "Should not fix any faces");
        assert_eq!(mesh.faces[0], [0, 1, 2], "Face should remain unchanged");
    }

    #[test]
    fn test_repair_params_default() {
        let params = RepairParams::default();
        assert_eq!(params.weld_epsilon, 1e-6);
        assert_eq!(params.degenerate_area_threshold, 1e-9);
        assert_eq!(params.degenerate_aspect_ratio, 1000.0);
        assert_eq!(params.max_hole_edges, 100);
        assert!(params.fix_winding);
        assert!(params.fix_non_manifold);
        assert!(!params.fill_holes);
        assert!(params.compute_normals);
    }

    #[test]
    fn test_repair_params_for_scans() {
        let params = RepairParams::for_scans();
        // Scan params should be more aggressive
        assert!(params.weld_epsilon > RepairParams::default().weld_epsilon);
        assert!(params.fill_holes); // Scans often have gaps
    }

    #[test]
    fn test_repair_params_for_cad() {
        let params = RepairParams::for_cad();
        // CAD params should be more conservative
        assert!(params.weld_epsilon < RepairParams::default().weld_epsilon);
        assert!(!params.fill_holes); // CAD holes are intentional
        assert!(params.degenerate_aspect_ratio.is_infinite());
    }

    #[test]
    fn test_repair_params_for_printing() {
        let params = RepairParams::for_printing();
        assert!(params.fill_holes); // Printing needs watertight
        assert!(params.fix_winding);
        assert!(params.fix_non_manifold);
    }

    #[test]
    fn test_remove_degenerate_triangles_enhanced_area() {
        let mut mesh = Mesh::new();
        mesh.vertices.push(Vertex::from_coords(0.0, 0.0, 0.0));
        mesh.vertices.push(Vertex::from_coords(10.0, 0.0, 0.0));
        mesh.vertices.push(Vertex::from_coords(0.0, 10.0, 0.0));
        // Normal triangle (area = 50)
        mesh.faces.push([0, 1, 2]);

        // Tiny triangle (area ~= 0.00005)
        mesh.vertices.push(Vertex::from_coords(0.0, 0.0, 0.0));
        mesh.vertices.push(Vertex::from_coords(0.01, 0.0, 0.0));
        mesh.vertices.push(Vertex::from_coords(0.0, 0.01, 0.0));
        mesh.faces.push([3, 4, 5]);

        let removed = remove_degenerate_triangles_enhanced(
            &mut mesh,
            0.001,         // area threshold
            f64::INFINITY, // no aspect ratio check
            0.0,           // no edge length check
        );

        assert_eq!(removed, 1);
        assert_eq!(mesh.face_count(), 1);
    }

    #[test]
    fn test_remove_degenerate_triangles_enhanced_aspect_ratio() {
        let mut mesh = Mesh::new();
        // Normal triangle
        mesh.vertices.push(Vertex::from_coords(0.0, 0.0, 0.0));
        mesh.vertices.push(Vertex::from_coords(10.0, 0.0, 0.0));
        mesh.vertices.push(Vertex::from_coords(5.0, 8.66, 0.0)); // ~equilateral
        mesh.faces.push([0, 1, 2]);

        // Very thin sliver triangle (high aspect ratio)
        mesh.vertices.push(Vertex::from_coords(0.0, 0.0, 0.0));
        mesh.vertices.push(Vertex::from_coords(100.0, 0.0, 0.0));
        mesh.vertices.push(Vertex::from_coords(50.0, 0.01, 0.0)); // Very thin
        mesh.faces.push([3, 4, 5]);

        let removed = remove_degenerate_triangles_enhanced(
            &mut mesh, 0.0,   // no area check
            100.0, // aspect ratio threshold
            0.0,   // no edge length check
        );

        assert_eq!(removed, 1);
        assert_eq!(mesh.face_count(), 1);
    }

    #[test]
    fn test_remove_degenerate_triangles_enhanced_edge_length() {
        let mut mesh = Mesh::new();
        // Normal triangle with reasonable edges
        mesh.vertices.push(Vertex::from_coords(0.0, 0.0, 0.0));
        mesh.vertices.push(Vertex::from_coords(10.0, 0.0, 0.0));
        mesh.vertices.push(Vertex::from_coords(0.0, 10.0, 0.0));
        mesh.faces.push([0, 1, 2]);

        // Triangle with a tiny edge
        mesh.vertices.push(Vertex::from_coords(0.0, 0.0, 0.0));
        mesh.vertices.push(Vertex::from_coords(0.0001, 0.0, 0.0)); // Very short edge
        mesh.vertices.push(Vertex::from_coords(0.0, 10.0, 0.0));
        mesh.faces.push([3, 4, 5]);

        let removed = remove_degenerate_triangles_enhanced(
            &mut mesh,
            0.0,           // no area check
            f64::INFINITY, // no aspect ratio check
            0.001,         // min edge length
        );

        assert_eq!(removed, 1);
        assert_eq!(mesh.face_count(), 1);
    }

    #[test]
    fn test_repair_mesh_with_config() {
        let mut mesh = Mesh::new();
        // Create a simple mesh that needs repair
        mesh.vertices.push(Vertex::from_coords(0.0, 0.0, 0.0));
        mesh.vertices.push(Vertex::from_coords(10.0, 0.0, 0.0));
        mesh.vertices.push(Vertex::from_coords(0.0, 10.0, 0.0));
        mesh.vertices.push(Vertex::from_coords(10.0001, 0.0, 0.0)); // Near-duplicate
        mesh.faces.push([0, 1, 2]);
        mesh.faces.push([0, 3, 2]); // Uses near-duplicate

        let params = RepairParams {
            weld_epsilon: 0.001, // Will merge vertex 3 into vertex 1
            fix_winding: false,  // Keep it simple
            fill_holes: false,
            ..Default::default()
        };

        let result = super::repair_mesh_with_config(&mut mesh, &params);
        assert!(result.is_ok());

        // Vertices should have been welded
        // Face indices should be valid
        for face in &mesh.faces {
            assert!((face[0] as usize) < mesh.vertices.len());
            assert!((face[1] as usize) < mesh.vertices.len());
            assert!((face[2] as usize) < mesh.vertices.len());
        }
    }
}

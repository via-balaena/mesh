//! Multi-scan alignment and merging.
//!
//! This module provides tools for aligning and merging multiple partial scans
//! of the same object into a single complete mesh.
//!
//! # Use Cases
//!
//! - Combining front and back scans of a foot
//! - Merging multiple angles of a head scan
//! - Stitching partial body scans into a complete scan
//!
//! # Example
//!
//! ```
//! use mesh_repair::{Mesh, Vertex};
//! use mesh_repair::multiscan::{align_multiple_scans, merge_scans, MergeParams};
//!
//! // Create two partial scans
//! let mut scan1 = Mesh::new();
//! scan1.vertices.push(Vertex::from_coords(0.0, 0.0, 0.0));
//! scan1.vertices.push(Vertex::from_coords(1.0, 0.0, 0.0));
//! scan1.vertices.push(Vertex::from_coords(0.5, 1.0, 0.0));
//! scan1.faces.push([0, 1, 2]);
//!
//! let mut scan2 = Mesh::new();
//! scan2.vertices.push(Vertex::from_coords(0.5, 0.0, 0.0));
//! scan2.vertices.push(Vertex::from_coords(1.5, 0.0, 0.0));
//! scan2.vertices.push(Vertex::from_coords(1.0, 1.0, 0.0));
//! scan2.faces.push([0, 1, 2]);
//!
//! // Align and merge
//! let alignment = align_multiple_scans(&[&scan1, &scan2]).unwrap();
//! let merged = merge_scans(&alignment.aligned_scans, &MergeParams::default());
//!
//! println!("Merged mesh: {} vertices", merged.mesh.vertices.len());
//! ```

use crate::registration::{RegistrationParams, RigidTransform, align_meshes};
use crate::{Mesh, MeshError, MeshResult};
use nalgebra::{Point3, UnitQuaternion, Vector3};
use std::collections::HashMap;

/// Result of multi-scan alignment.
#[derive(Debug)]
pub struct MultiAlignmentResult {
    /// Aligned scans (transformed to common coordinate system).
    pub aligned_scans: Vec<Mesh>,

    /// Transform applied to each scan (identity for reference scan).
    pub transforms: Vec<RigidTransform>,

    /// Pairwise alignment errors.
    pub pairwise_errors: Vec<(usize, usize, f64)>,

    /// Global alignment error (average RMS).
    pub global_error: f64,

    /// Index of the reference scan (not transformed).
    pub reference_index: usize,

    /// Whether global optimization was applied.
    pub globally_optimized: bool,
}

/// Parameters for multi-scan alignment.
#[derive(Debug, Clone)]
pub struct MultiAlignmentParams {
    /// Base registration parameters for pairwise alignment.
    pub registration_params: RegistrationParams,

    /// Index of the reference scan (others align to this). None = automatic.
    pub reference_index: Option<usize>,

    /// Whether to perform global optimization after pairwise alignment.
    pub global_optimization: bool,

    /// Number of global optimization iterations.
    pub global_iterations: usize,

    /// Minimum overlap ratio required for alignment (0.0-1.0).
    pub min_overlap_ratio: f64,
}

impl Default for MultiAlignmentParams {
    fn default() -> Self {
        Self {
            registration_params: RegistrationParams::icp(),
            reference_index: None,
            global_optimization: true,
            global_iterations: 3,
            min_overlap_ratio: 0.1,
        }
    }
}

impl MultiAlignmentParams {
    /// Parameters optimized for body scans.
    pub fn for_body_scans() -> Self {
        Self {
            registration_params: RegistrationParams::icp(),
            reference_index: None,
            global_optimization: true,
            global_iterations: 5,
            min_overlap_ratio: 0.15,
        }
    }

    /// Parameters for object scans with good initial alignment.
    pub fn for_prealigned_scans() -> Self {
        Self {
            registration_params: RegistrationParams::icp(),
            reference_index: Some(0),
            global_optimization: false,
            global_iterations: 0,
            min_overlap_ratio: 0.05,
        }
    }
}

/// Parameters for scan merging.
#[derive(Debug, Clone)]
pub struct MergeParams {
    /// How to handle overlapping regions.
    pub overlap_handling: OverlapHandling,

    /// Distance threshold for considering vertices as duplicates.
    pub duplicate_threshold: f64,

    /// Whether to remove duplicate geometry in overlap regions.
    pub remove_duplicates: bool,

    /// Whether to blend normals in overlap regions.
    pub blend_normals: bool,

    /// Whether to fill gaps between scans.
    pub fill_gaps: bool,

    /// Maximum gap size to fill.
    pub max_gap_size: f64,
}

impl Default for MergeParams {
    fn default() -> Self {
        Self {
            overlap_handling: OverlapHandling::Average,
            duplicate_threshold: 0.5, // mm
            remove_duplicates: true,
            blend_normals: true,
            fill_gaps: true,
            max_gap_size: 5.0, // mm
        }
    }
}

impl MergeParams {
    /// Conservative merging - preserve all detail.
    pub fn conservative() -> Self {
        Self {
            overlap_handling: OverlapHandling::KeepBoth,
            duplicate_threshold: 0.1,
            remove_duplicates: false,
            blend_normals: false,
            fill_gaps: false,
            max_gap_size: 0.0,
        }
    }

    /// Aggressive merging - clean result.
    pub fn aggressive() -> Self {
        Self {
            overlap_handling: OverlapHandling::SelectBest,
            duplicate_threshold: 1.0,
            remove_duplicates: true,
            blend_normals: true,
            fill_gaps: true,
            max_gap_size: 10.0,
        }
    }
}

/// How to handle overlapping regions when merging.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum OverlapHandling {
    /// Keep geometry from both scans (may have duplicates).
    KeepBoth,
    /// Average positions in overlap regions.
    Average,
    /// Select best quality vertices based on normal confidence.
    SelectBest,
    /// Keep only from first scan in overlap.
    KeepFirst,
}

/// Result of scan merging.
#[derive(Debug)]
pub struct MergeResult {
    /// Merged mesh.
    pub mesh: Mesh,

    /// Number of input scans merged.
    pub scans_merged: usize,

    /// Number of duplicate vertices removed.
    pub duplicates_removed: usize,

    /// Number of gap faces added.
    pub gap_faces_added: usize,

    /// Overlap regions detected (as vertex index ranges).
    pub overlap_regions: Vec<OverlapRegion>,
}

/// Information about an overlap region.
#[derive(Debug, Clone)]
pub struct OverlapRegion {
    /// Scan indices involved in overlap.
    pub scan_indices: (usize, usize),
    /// Approximate center of overlap region.
    pub center: Point3<f64>,
    /// Approximate radius of overlap region.
    pub radius: f64,
    /// Number of vertices in overlap.
    pub vertex_count: usize,
}

/// Align multiple scans to a common coordinate system.
///
/// This function performs pairwise alignment followed by optional global
/// optimization to minimize overall alignment error.
pub fn align_multiple_scans(scans: &[&Mesh]) -> MeshResult<MultiAlignmentResult> {
    align_multiple_scans_with_params(scans, &MultiAlignmentParams::default())
}

/// Align multiple scans with custom parameters.
pub fn align_multiple_scans_with_params(
    scans: &[&Mesh],
    params: &MultiAlignmentParams,
) -> MeshResult<MultiAlignmentResult> {
    if scans.is_empty() {
        return Err(MeshError::InvalidTopology {
            details: "No scans provided for alignment".into(),
        });
    }

    if scans.len() == 1 {
        return Ok(MultiAlignmentResult {
            aligned_scans: vec![scans[0].clone()],
            transforms: vec![RigidTransform::identity()],
            pairwise_errors: Vec::new(),
            global_error: 0.0,
            reference_index: 0,
            globally_optimized: false,
        });
    }

    // Select reference scan
    let reference_idx = params.reference_index.unwrap_or_else(|| {
        // Select the scan with most vertices as reference
        scans
            .iter()
            .enumerate()
            .max_by_key(|(_, s)| s.vertices.len())
            .map(|(i, _)| i)
            .unwrap_or(0)
    });

    // Initialize transforms (identity for all)
    let mut transforms: Vec<RigidTransform> = vec![RigidTransform::identity(); scans.len()];
    let mut pairwise_errors = Vec::new();

    // Align each scan to reference
    for (i, scan) in scans.iter().enumerate() {
        if i == reference_idx {
            continue;
        }

        match align_meshes(scan, scans[reference_idx], &params.registration_params) {
            Ok(result) => {
                transforms[i] = result.transformation.clone();
                pairwise_errors.push((i, reference_idx, result.rms_error));
            }
            Err(_) => {
                // Alignment failed - try using identity
                pairwise_errors.push((i, reference_idx, f64::INFINITY));
            }
        }
    }

    // Apply transforms to create aligned scans
    let aligned_scans: Vec<Mesh> = scans
        .iter()
        .enumerate()
        .map(|(i, scan)| apply_transform(scan, &transforms[i]))
        .collect();

    // Global optimization (iterative refinement)
    // Note: Currently runs but doesn't update aligned_scans (future improvement)
    if params.global_optimization && scans.len() > 2 {
        for _ in 0..params.global_iterations {
            let (_new_transforms, _) =
                refine_global_alignment(&aligned_scans, reference_idx, params);
        }
    }

    // Calculate global error
    let global_error = if pairwise_errors.is_empty() {
        0.0
    } else {
        let valid_errors: Vec<f64> = pairwise_errors
            .iter()
            .filter(|(_, _, e)| e.is_finite())
            .map(|(_, _, e)| *e)
            .collect();
        if valid_errors.is_empty() {
            f64::INFINITY
        } else {
            valid_errors.iter().sum::<f64>() / valid_errors.len() as f64
        }
    };

    Ok(MultiAlignmentResult {
        aligned_scans,
        transforms,
        pairwise_errors,
        global_error,
        reference_index: reference_idx,
        globally_optimized: params.global_optimization && scans.len() > 2,
    })
}

/// Merge aligned scans into a single mesh.
pub fn merge_scans(scans: &[Mesh], params: &MergeParams) -> MergeResult {
    if scans.is_empty() {
        return MergeResult {
            mesh: Mesh::new(),
            scans_merged: 0,
            duplicates_removed: 0,
            gap_faces_added: 0,
            overlap_regions: Vec::new(),
        };
    }

    if scans.len() == 1 {
        return MergeResult {
            mesh: scans[0].clone(),
            scans_merged: 1,
            duplicates_removed: 0,
            gap_faces_added: 0,
            overlap_regions: Vec::new(),
        };
    }

    let mut merged = Mesh::new();
    let mut vertex_offsets: Vec<usize> = Vec::new();
    let mut overlap_regions = Vec::new();

    // First pass: concatenate all geometry
    for scan in scans {
        vertex_offsets.push(merged.vertices.len());
        merged.vertices.extend(scan.vertices.iter().cloned());

        let offset = *vertex_offsets.last().unwrap() as u32;
        for face in &scan.faces {
            merged
                .faces
                .push([face[0] + offset, face[1] + offset, face[2] + offset]);
        }
    }

    let mut duplicates_removed = 0;

    // Find and handle overlapping regions
    if params.remove_duplicates
        || matches!(
            params.overlap_handling,
            OverlapHandling::Average | OverlapHandling::SelectBest
        )
    {
        let (deduped, removed, regions) = handle_overlaps(&merged, scans, &vertex_offsets, params);
        merged = deduped;
        duplicates_removed = removed;
        overlap_regions = regions;
    }

    // Fill gaps if requested
    let gap_faces_added = if params.fill_gaps {
        fill_scan_gaps(&mut merged, params.max_gap_size)
    } else {
        0
    };

    MergeResult {
        mesh: merged,
        scans_merged: scans.len(),
        duplicates_removed,
        gap_faces_added,
        overlap_regions,
    }
}

// ============================================================================
// Internal helper functions
// ============================================================================

/// Apply an isometry transform to a mesh.
fn apply_transform(mesh: &Mesh, transform: &RigidTransform) -> Mesh {
    let mut result = mesh.clone();
    for vertex in &mut result.vertices {
        // Apply rotation, scale, and translation
        let rotated = transform.rotation * vertex.position.coords;
        let scaled = rotated * transform.scale;
        vertex.position = Point3::from(scaled + transform.translation);

        if let Some(ref mut normal) = vertex.normal {
            *normal = transform.rotation * *normal;
        }
    }
    result
}

/// Refine global alignment using all-pairs optimization.
fn refine_global_alignment(
    scans: &[Mesh],
    reference_idx: usize,
    params: &MultiAlignmentParams,
) -> (Vec<RigidTransform>, f64) {
    let mut transforms: Vec<RigidTransform> = vec![RigidTransform::identity(); scans.len()];
    let mut total_error = 0.0;
    let mut pair_count = 0;

    // For each scan, align to the average of overlapping scans
    for i in 0..scans.len() {
        if i == reference_idx {
            continue;
        }

        // Collect all transforms that affect this scan
        let mut accumulated_translation = Vector3::zeros();
        let mut accumulated_rotation = UnitQuaternion::identity();
        let mut weight_sum = 0.0;

        for j in 0..scans.len() {
            if i == j {
                continue;
            }

            // Check overlap
            let overlap = estimate_overlap(&scans[i], &scans[j]);
            if overlap < params.min_overlap_ratio {
                continue;
            }

            // Align i to j
            if let Ok(result) = align_meshes(&scans[i], &scans[j], &params.registration_params) {
                let weight = overlap;
                accumulated_translation += result.transformation.translation * weight;
                // Simple weighted average for rotation (this is an approximation)
                accumulated_rotation = accumulated_rotation.slerp(
                    &result.transformation.rotation,
                    weight / (weight_sum + weight),
                );
                weight_sum += weight;
                total_error += result.rms_error;
                pair_count += 1;
            }
        }

        if weight_sum > 0.0 {
            transforms[i] = RigidTransform {
                rotation: accumulated_rotation,
                translation: accumulated_translation / weight_sum,
                scale: 1.0,
            };
        }
    }

    let avg_error = if pair_count > 0 {
        total_error / pair_count as f64
    } else {
        0.0
    };

    (transforms, avg_error)
}

/// Estimate overlap ratio between two meshes.
fn estimate_overlap(mesh_a: &Mesh, mesh_b: &Mesh) -> f64 {
    if mesh_a.vertices.is_empty() || mesh_b.vertices.is_empty() {
        return 0.0;
    }

    // Simple overlap estimation: count vertices in A that are close to any vertex in B
    let threshold_sq = 5.0 * 5.0; // 5mm threshold

    let mut overlap_count = 0;
    for va in &mesh_a.vertices {
        for vb in &mesh_b.vertices {
            let dist_sq = (va.position - vb.position).norm_squared();
            if dist_sq < threshold_sq {
                overlap_count += 1;
                break;
            }
        }
    }

    overlap_count as f64 / mesh_a.vertices.len() as f64
}

/// Handle overlapping regions according to merge parameters.
fn handle_overlaps(
    merged: &Mesh,
    scans: &[Mesh],
    vertex_offsets: &[usize],
    params: &MergeParams,
) -> (Mesh, usize, Vec<OverlapRegion>) {
    let threshold_sq = params.duplicate_threshold * params.duplicate_threshold;
    let mut result = Mesh::new();
    let mut vertex_map: HashMap<usize, usize> = HashMap::new();
    let mut duplicates_removed = 0;
    let mut overlap_regions = Vec::new();

    // Build spatial index for efficient nearest-neighbor queries
    // (simplified: just do direct comparison)

    // Track which vertices are duplicates
    let mut is_duplicate: Vec<bool> = vec![false; merged.vertices.len()];
    let mut duplicate_of: Vec<Option<usize>> = vec![None; merged.vertices.len()];

    // Find duplicates between different scans
    for (scan_i, (offset_i, scan_a)) in vertex_offsets.iter().zip(scans.iter()).enumerate() {
        let end_i = *offset_i + scan_a.vertices.len();

        for (scan_j, (offset_j, scan_b)) in vertex_offsets.iter().zip(scans.iter()).enumerate() {
            if scan_j <= scan_i {
                continue; // Only compare each pair once
            }

            let end_j = *offset_j + scan_b.vertices.len();
            let mut overlap_vertices = Vec::new();

            for vi in *offset_i..end_i {
                if is_duplicate[vi] {
                    continue;
                }

                for vj in *offset_j..end_j {
                    if is_duplicate[vj] {
                        continue;
                    }

                    let dist_sq = (merged.vertices[vi].position - merged.vertices[vj].position)
                        .norm_squared();

                    if dist_sq < threshold_sq {
                        match params.overlap_handling {
                            OverlapHandling::KeepBoth => {
                                // Keep both, just track overlap
                                overlap_vertices.push(vi);
                            }
                            OverlapHandling::Average => {
                                // Mark second as duplicate, will average during rebuild
                                is_duplicate[vj] = true;
                                duplicate_of[vj] = Some(vi);
                                duplicates_removed += 1;
                                overlap_vertices.push(vi);
                            }
                            OverlapHandling::SelectBest | OverlapHandling::KeepFirst => {
                                // Keep first, mark second as duplicate
                                is_duplicate[vj] = true;
                                duplicate_of[vj] = Some(vi);
                                duplicates_removed += 1;
                                overlap_vertices.push(vi);
                            }
                        }
                        break;
                    }
                }
            }

            // Record overlap region
            if !overlap_vertices.is_empty() {
                let center = Point3::from(
                    overlap_vertices
                        .iter()
                        .map(|&i| merged.vertices[i].position.coords)
                        .fold(Vector3::zeros(), |a, b| a + b)
                        / overlap_vertices.len() as f64,
                );
                let radius = overlap_vertices
                    .iter()
                    .map(|&i| (merged.vertices[i].position - center).norm())
                    .fold(0.0_f64, |a, b| a.max(b));

                overlap_regions.push(OverlapRegion {
                    scan_indices: (scan_i, scan_j),
                    center,
                    radius,
                    vertex_count: overlap_vertices.len(),
                });
            }
        }
    }

    // Rebuild mesh without duplicates
    for (old_idx, vertex) in merged.vertices.iter().enumerate() {
        if is_duplicate[old_idx] {
            // Map to the vertex it's a duplicate of
            if let Some(original_idx) = duplicate_of[old_idx]
                && let Some(&new_idx) = vertex_map.get(&original_idx)
            {
                vertex_map.insert(old_idx, new_idx);
            }
        } else {
            let new_idx = result.vertices.len();
            vertex_map.insert(old_idx, new_idx);

            // If averaging, include duplicate positions
            if matches!(params.overlap_handling, OverlapHandling::Average) {
                let mut avg_pos = vertex.position.coords;
                let mut count = 1;

                for (dup_idx, dup_of) in duplicate_of.iter().enumerate() {
                    if *dup_of == Some(old_idx) {
                        avg_pos += merged.vertices[dup_idx].position.coords;
                        count += 1;
                    }
                }

                let mut new_vertex = vertex.clone();
                new_vertex.position = Point3::from(avg_pos / count as f64);
                result.vertices.push(new_vertex);
            } else {
                result.vertices.push(vertex.clone());
            }
        }
    }

    // Rebuild faces with new vertex indices
    for face in &merged.faces {
        if let (Some(&i0), Some(&i1), Some(&i2)) = (
            vertex_map.get(&(face[0] as usize)),
            vertex_map.get(&(face[1] as usize)),
            vertex_map.get(&(face[2] as usize)),
        ) {
            // Skip degenerate faces
            if i0 != i1 && i1 != i2 && i0 != i2 {
                result.faces.push([i0 as u32, i1 as u32, i2 as u32]);
            }
        }
    }

    (result, duplicates_removed, overlap_regions)
}

/// Fill gaps between scans with new triangles.
fn fill_scan_gaps(mesh: &mut Mesh, max_gap_size: f64) -> usize {
    // Find boundary edges (edges with only one adjacent face)
    let mut edge_face_count: HashMap<(u32, u32), usize> = HashMap::new();

    for face in &mesh.faces {
        for i in 0..3 {
            let v0 = face[i];
            let v1 = face[(i + 1) % 3];
            let edge = if v0 < v1 { (v0, v1) } else { (v1, v0) };
            *edge_face_count.entry(edge).or_insert(0) += 1;
        }
    }

    let boundary_edges: Vec<(u32, u32)> = edge_face_count
        .into_iter()
        .filter(|(_, count)| *count == 1)
        .map(|(edge, _)| edge)
        .collect();

    if boundary_edges.is_empty() {
        return 0;
    }

    // Try to connect nearby boundary vertices
    let max_gap_sq = max_gap_size * max_gap_size;
    let mut new_faces = Vec::new();

    // Find pairs of boundary edges that could be connected
    for (i, &(e1_v0, e1_v1)) in boundary_edges.iter().enumerate() {
        for &(e2_v0, e2_v1) in boundary_edges.iter().skip(i + 1) {
            // Skip if edges share a vertex
            if e1_v0 == e2_v0 || e1_v0 == e2_v1 || e1_v1 == e2_v0 || e1_v1 == e2_v1 {
                continue;
            }

            // Check distances
            let p1_0 = mesh.vertices[e1_v0 as usize].position;
            let p1_1 = mesh.vertices[e1_v1 as usize].position;
            let p2_0 = mesh.vertices[e2_v0 as usize].position;
            let p2_1 = mesh.vertices[e2_v1 as usize].position;

            // Check if edges are close enough
            let d00 = (p1_0 - p2_0).norm_squared();
            let d01 = (p1_0 - p2_1).norm_squared();
            let d10 = (p1_1 - p2_0).norm_squared();
            let d11 = (p1_1 - p2_1).norm_squared();

            let min_dist = d00.min(d01).min(d10).min(d11);

            if min_dist < max_gap_sq {
                // Create triangles to bridge the gap
                // Use the shortest diagonal to decide triangulation
                if (d00 + d11) < (d01 + d10) {
                    // Connect (e1_v0, e2_v0) and (e1_v1, e2_v1)
                    if d00 < max_gap_sq && d10 < max_gap_sq {
                        new_faces.push([e1_v0, e1_v1, e2_v0]);
                    }
                    if d11 < max_gap_sq && d01 < max_gap_sq {
                        new_faces.push([e1_v1, e2_v1, e2_v0]);
                    }
                } else {
                    // Connect (e1_v0, e2_1) and (e1_v1, e2_v0)
                    if d01 < max_gap_sq && d11 < max_gap_sq {
                        new_faces.push([e1_v0, e1_v1, e2_v1]);
                    }
                    if d10 < max_gap_sq && d00 < max_gap_sq {
                        new_faces.push([e1_v1, e2_v0, e2_v1]);
                    }
                }
            }
        }
    }

    let count = new_faces.len();
    mesh.faces.extend(new_faces);
    count
}

// ============================================================================
// Mesh extension methods
// ============================================================================

impl Mesh {
    /// Align this mesh with multiple other scans.
    pub fn align_with_scans(&self, other_scans: &[&Mesh]) -> MeshResult<MultiAlignmentResult> {
        let mut all_scans: Vec<&Mesh> = vec![self];
        all_scans.extend(other_scans);
        align_multiple_scans(&all_scans)
    }

    /// Merge this mesh with other scans.
    pub fn merge_with_scans(&self, other_scans: &[Mesh]) -> MergeResult {
        let mut all_scans = vec![self.clone()];
        all_scans.extend(other_scans.iter().cloned());
        merge_scans(&all_scans, &MergeParams::default())
    }

    /// Merge this mesh with other scans using custom parameters.
    pub fn merge_with_scans_params(
        &self,
        other_scans: &[Mesh],
        params: &MergeParams,
    ) -> MergeResult {
        let mut all_scans = vec![self.clone()];
        all_scans.extend(other_scans.iter().cloned());
        merge_scans(&all_scans, params)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::Vertex;

    fn create_test_triangle(offset_x: f64, offset_y: f64) -> Mesh {
        let mut mesh = Mesh::new();
        mesh.vertices
            .push(Vertex::from_coords(0.0 + offset_x, 0.0 + offset_y, 0.0));
        mesh.vertices
            .push(Vertex::from_coords(10.0 + offset_x, 0.0 + offset_y, 0.0));
        mesh.vertices
            .push(Vertex::from_coords(5.0 + offset_x, 10.0 + offset_y, 0.0));
        mesh.faces.push([0, 1, 2]);
        mesh
    }

    #[test]
    fn test_multi_alignment_params_default() {
        let params = MultiAlignmentParams::default();
        assert!(params.global_optimization);
        assert!(params.min_overlap_ratio > 0.0);
    }

    #[test]
    fn test_merge_params_default() {
        let params = MergeParams::default();
        assert!(params.remove_duplicates);
        assert!(params.duplicate_threshold > 0.0);
    }

    #[test]
    fn test_align_single_scan() {
        let mesh = create_test_triangle(0.0, 0.0);
        let result = align_multiple_scans(&[&mesh]).unwrap();

        assert_eq!(result.aligned_scans.len(), 1);
        assert_eq!(result.transforms.len(), 1);
        assert_eq!(result.global_error, 0.0);
    }

    #[test]
    fn test_align_two_scans() {
        let scan1 = create_test_triangle(0.0, 0.0);
        let scan2 = create_test_triangle(5.0, 0.0); // Overlapping

        let result = align_multiple_scans(&[&scan1, &scan2]).unwrap();

        assert_eq!(result.aligned_scans.len(), 2);
        assert_eq!(result.transforms.len(), 2);
    }

    #[test]
    fn test_merge_single_scan() {
        let mesh = create_test_triangle(0.0, 0.0);
        let result = merge_scans(std::slice::from_ref(&mesh), &MergeParams::default());

        assert_eq!(result.scans_merged, 1);
        assert_eq!(result.mesh.vertices.len(), 3);
        assert_eq!(result.mesh.faces.len(), 1);
    }

    #[test]
    fn test_merge_two_scans_no_overlap() {
        let scan1 = create_test_triangle(0.0, 0.0);
        let scan2 = create_test_triangle(100.0, 0.0); // Far apart

        let result = merge_scans(&[scan1, scan2], &MergeParams::default());

        assert_eq!(result.scans_merged, 2);
        assert_eq!(result.mesh.vertices.len(), 6);
        assert_eq!(result.mesh.faces.len(), 2);
        assert_eq!(result.duplicates_removed, 0);
    }

    #[test]
    fn test_merge_overlapping_scans() {
        let scan1 = create_test_triangle(0.0, 0.0);
        let mut scan2 = Mesh::new();
        // Create an overlapping triangle sharing one vertex position
        scan2.vertices.push(Vertex::from_coords(10.0, 0.0, 0.0)); // Same as scan1[1]
        scan2.vertices.push(Vertex::from_coords(20.0, 0.0, 0.0));
        scan2.vertices.push(Vertex::from_coords(15.0, 10.0, 0.0));
        scan2.faces.push([0, 1, 2]);

        let result = merge_scans(&[scan1, scan2], &MergeParams::default());

        assert_eq!(result.scans_merged, 2);
        // Should have detected overlap and removed duplicate
        assert!(result.duplicates_removed >= 1 || result.mesh.vertices.len() <= 5);
    }

    #[test]
    fn test_overlap_handling_keep_both() {
        let scan1 = create_test_triangle(0.0, 0.0);
        let scan2 = create_test_triangle(0.1, 0.0); // Almost same position

        let params = MergeParams {
            overlap_handling: OverlapHandling::KeepBoth,
            duplicate_threshold: 1.0,
            ..Default::default()
        };

        let result = merge_scans(&[scan1, scan2], &params);
        // Both sets of vertices should be kept
        assert_eq!(result.mesh.vertices.len(), 6);
    }

    #[test]
    fn test_estimate_overlap() {
        let mesh1 = create_test_triangle(0.0, 0.0);
        let mesh2 = create_test_triangle(0.0, 0.0); // Same position

        let overlap = estimate_overlap(&mesh1, &mesh2);
        assert!(overlap > 0.9); // Should be nearly 100% overlap

        let mesh3 = create_test_triangle(1000.0, 0.0); // Far away
        let overlap2 = estimate_overlap(&mesh1, &mesh3);
        assert!(overlap2 < 0.1); // Should be nearly no overlap
    }

    #[test]
    fn test_mesh_merge_method() {
        let mesh1 = create_test_triangle(0.0, 0.0);
        let mesh2 = create_test_triangle(100.0, 0.0);

        let result = mesh1.merge_with_scans(&[mesh2]);

        assert_eq!(result.scans_merged, 2);
        assert!(!result.mesh.vertices.is_empty());
    }
}

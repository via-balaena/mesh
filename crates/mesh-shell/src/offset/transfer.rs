//! Transfer vertex attributes from original mesh to extracted surface.

use parry3d::query::PointQuery;
use parry3d::shape::TriMesh;
use rayon::prelude::*;
use tracing::{debug, info};

use mesh_repair::Mesh;

use crate::error::{ShellError, ShellResult};

/// Transfer tag and offset data from original mesh to extracted surface.
///
/// For each vertex in the output mesh, finds the closest point on the
/// original mesh and interpolates the attributes.
pub fn transfer_vertex_data(original: &Mesh, output: &mut Mesh) -> ShellResult<()> {
    if original.vertices.is_empty() || original.faces.is_empty() {
        return Err(ShellError::TagTransferFailed {
            details: "Original mesh is empty".to_string(),
        });
    }

    info!(
        original_verts = original.vertices.len(),
        output_verts = output.vertices.len(),
        "Transferring vertex data"
    );

    // Build parry3d TriMesh for closest point queries
    let vertices: Vec<parry3d::math::Point<f32>> = original
        .vertices
        .iter()
        .map(|v| {
            parry3d::math::Point::new(
                v.position.x as f32,
                v.position.y as f32,
                v.position.z as f32,
            )
        })
        .collect();

    let indices: Vec<[u32; 3]> = original.faces.clone();

    let trimesh = TriMesh::new(vertices, indices);

    // Pre-extract attributes from original mesh
    let original_tags: Vec<Option<u32>> = original.vertices.iter().map(|v| v.tag).collect();
    let original_offsets: Vec<Option<f32>> = original.vertices.iter().map(|v| v.offset).collect();

    // Transfer data in parallel
    let results: Vec<(Option<u32>, Option<f32>)> = output
        .vertices
        .par_iter()
        .map(|v| {
            let query_point = parry3d::math::Point::new(
                v.position.x as f32,
                v.position.y as f32,
                v.position.z as f32,
            );

            // Find closest point on original mesh
            let (projection, feature) = trimesh.project_local_point_and_get_feature(&query_point);

            // Get the face index from the feature
            let face_idx = match feature {
                parry3d::shape::FeatureId::Face(idx) => idx as usize,
                _ => {
                    // Fallback: find nearest vertex directly
                    let mut min_dist = f32::MAX;
                    let mut nearest_idx = 0;
                    for (i, orig_v) in original.vertices.iter().enumerate() {
                        let d = (v.position.x - orig_v.position.x).powi(2)
                            + (v.position.y - orig_v.position.y).powi(2)
                            + (v.position.z - orig_v.position.z).powi(2);
                        if (d as f32) < min_dist {
                            min_dist = d as f32;
                            nearest_idx = i;
                        }
                    }
                    return (original_tags[nearest_idx], original_offsets[nearest_idx]);
                }
            };

            if face_idx >= original.faces.len() {
                return (None, None);
            }

            let face = original.faces[face_idx];
            let [v0, v1, v2] = face;

            // Compute barycentric coordinates
            let p0 = &original.vertices[v0 as usize].position;
            let p1 = &original.vertices[v1 as usize].position;
            let p2 = &original.vertices[v2 as usize].position;

            let proj_point = nalgebra::Point3::new(
                projection.point.x as f64,
                projection.point.y as f64,
                projection.point.z as f64,
            );

            let bary = compute_barycentric(&proj_point, p0, p1, p2);

            // Interpolate tag using majority vote weighted by barycentric coords
            let tag = majority_vote_tag(
                original_tags[v0 as usize],
                original_tags[v1 as usize],
                original_tags[v2 as usize],
                bary,
            );

            // Interpolate offset using weighted average
            let offset = interpolate_offset(
                original_offsets[v0 as usize],
                original_offsets[v1 as usize],
                original_offsets[v2 as usize],
                bary,
            );

            (tag, offset)
        })
        .collect();

    // Apply results to output mesh
    for (i, (tag, offset)) in results.into_iter().enumerate() {
        output.vertices[i].tag = tag;
        output.vertices[i].offset = offset;
    }

    let with_tags = output.vertices.iter().filter(|v| v.tag.is_some()).count();
    let with_offsets = output
        .vertices
        .iter()
        .filter(|v| v.offset.is_some())
        .count();

    debug!(
        with_tags,
        with_offsets,
        total = output.vertices.len(),
        "Vertex data transferred"
    );

    Ok(())
}

/// Compute barycentric coordinates for point p in triangle (p0, p1, p2).
fn compute_barycentric(
    p: &nalgebra::Point3<f64>,
    p0: &nalgebra::Point3<f64>,
    p1: &nalgebra::Point3<f64>,
    p2: &nalgebra::Point3<f64>,
) -> [f64; 3] {
    let v0 = p1 - p0;
    let v1 = p2 - p0;
    let v2 = p - p0;

    let d00 = v0.dot(&v0);
    let d01 = v0.dot(&v1);
    let d11 = v1.dot(&v1);
    let d20 = v2.dot(&v0);
    let d21 = v2.dot(&v1);

    let denom = d00 * d11 - d01 * d01;
    if denom.abs() < 1e-10 {
        // Degenerate triangle, return equal weights
        return [1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0];
    }

    let v = (d11 * d20 - d01 * d21) / denom;
    let w = (d00 * d21 - d01 * d20) / denom;
    let u = 1.0 - v - w;

    // Clamp to valid range
    [u.clamp(0.0, 1.0), v.clamp(0.0, 1.0), w.clamp(0.0, 1.0)]
}

/// Select tag using majority vote weighted by barycentric coordinates.
fn majority_vote_tag(
    t0: Option<u32>,
    t1: Option<u32>,
    t2: Option<u32>,
    bary: [f64; 3],
) -> Option<u32> {
    // Find the vertex with the largest barycentric weight
    let max_idx = if bary[0] >= bary[1] && bary[0] >= bary[2] {
        0
    } else if bary[1] >= bary[2] {
        1
    } else {
        2
    };

    match max_idx {
        0 => t0,
        1 => t1,
        _ => t2,
    }
}

/// Interpolate offset using barycentric weighted average.
fn interpolate_offset(
    o0: Option<f32>,
    o1: Option<f32>,
    o2: Option<f32>,
    bary: [f64; 3],
) -> Option<f32> {
    let v0 = o0.unwrap_or(0.0) as f64;
    let v1 = o1.unwrap_or(0.0) as f64;
    let v2 = o2.unwrap_or(0.0) as f64;

    Some((bary[0] * v0 + bary[1] * v1 + bary[2] * v2) as f32)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_compute_barycentric() {
        let p0 = nalgebra::Point3::new(0.0, 0.0, 0.0);
        let p1 = nalgebra::Point3::new(1.0, 0.0, 0.0);
        let p2 = nalgebra::Point3::new(0.0, 1.0, 0.0);

        // Point at v0
        let bary = compute_barycentric(&p0, &p0, &p1, &p2);
        assert!((bary[0] - 1.0).abs() < 0.01);

        // Centroid
        let centroid = nalgebra::Point3::new(1.0 / 3.0, 1.0 / 3.0, 0.0);
        let bary = compute_barycentric(&centroid, &p0, &p1, &p2);
        assert!((bary[0] - 1.0 / 3.0).abs() < 0.01);
        assert!((bary[1] - 1.0 / 3.0).abs() < 0.01);
        assert!((bary[2] - 1.0 / 3.0).abs() < 0.01);
    }

    #[test]
    fn test_majority_vote() {
        let tag = majority_vote_tag(Some(1), Some(2), Some(3), [0.6, 0.3, 0.1]);
        assert_eq!(tag, Some(1));

        let tag = majority_vote_tag(Some(1), Some(2), Some(3), [0.1, 0.6, 0.3]);
        assert_eq!(tag, Some(2));
    }
}

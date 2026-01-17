//! SDF-based mesh offset for handling complex concavities.

use std::time::Instant;
use tracing::{debug, info, warn};

use mesh_repair::Mesh;

use crate::error::{ShellError, ShellResult};

use super::adaptive::{AdaptiveSdfParams, create_adaptive_grid, interpolate_offsets_adaptive};
use super::extract::extract_isosurface;
use super::grid::{SdfGrid, SdfOffsetParams};
use super::transfer::transfer_vertex_data;

/// Statistics from SDF offset operation.
#[derive(Debug, Clone)]
pub struct SdfOffsetStats {
    /// Grid dimensions [x, y, z].
    pub grid_dims: [usize; 3],
    /// Total number of voxels in grid.
    pub total_voxels: usize,
    /// Time spent computing SDF (ms).
    pub sdf_time_ms: u64,
    /// Time spent extracting isosurface (ms).
    pub extraction_time_ms: u64,
    /// Time spent transferring vertex data (ms).
    pub transfer_time_ms: u64,
    /// Number of vertices in input mesh.
    pub input_vertices: usize,
    /// Number of vertices in output mesh.
    pub output_vertices: usize,
    /// Number of faces in output mesh.
    pub output_faces: usize,
    /// Whether adaptive resolution was used.
    pub adaptive_resolution: bool,
    /// Estimated memory savings from adaptive resolution (percentage, 0 if not used).
    pub memory_savings_percent: f64,
}

/// Result of SDF offset operation.
#[derive(Debug)]
pub struct SdfOffsetResult {
    /// The offset mesh with vertex data transferred.
    pub mesh: Mesh,
    /// Statistics about the operation.
    pub stats: SdfOffsetStats,
}

/// Apply SDF-based offset to create a new mesh with variable offsets.
///
/// This is the main entry point for SDF offset. It takes a mesh with
/// offset values assigned to vertices and produces a new mesh offset
/// by those values.
///
/// # Arguments
/// * `mesh` - Input mesh with offset values assigned via `vertex.offset`
/// * `params` - SDF offset parameters (voxel size, padding, etc.)
///
/// # Returns
/// A new mesh representing the offset surface, with vertex data transferred.
///
/// # Example
/// ```ignore
/// let params = SdfOffsetParams::default();
/// let result = apply_sdf_offset(&mesh, &params)?;
/// println!("Generated {} vertices", result.stats.output_vertices);
/// ```
pub fn apply_sdf_offset(mesh: &Mesh, params: &SdfOffsetParams) -> ShellResult<SdfOffsetResult> {
    let total_start = Instant::now();

    if mesh.vertices.is_empty() {
        return Err(ShellError::EmptyMesh);
    }

    // Check that vertices have offset values
    let missing_offset = mesh.vertices.iter().filter(|v| v.offset.is_none()).count();

    if missing_offset > 0 {
        warn!(
            missing = missing_offset,
            total = mesh.vertices.len(),
            "Some vertices missing offset values, using 0.0"
        );
    }

    let input_vertices = mesh.vertices.len();

    info!(
        vertices = input_vertices,
        faces = mesh.faces.len(),
        voxel_size_mm = params.voxel_size_mm,
        padding_mm = params.padding_mm,
        adaptive = params.adaptive_resolution,
        "Starting SDF offset"
    );

    // Choose between adaptive and standard grid based on params
    if params.adaptive_resolution {
        apply_sdf_offset_adaptive(mesh, params, input_vertices, total_start)
    } else {
        apply_sdf_offset_standard(mesh, params, input_vertices, total_start)
    }
}

/// Standard (non-adaptive) SDF offset implementation.
fn apply_sdf_offset_standard(
    mesh: &Mesh,
    params: &SdfOffsetParams,
    input_vertices: usize,
    total_start: Instant,
) -> ShellResult<SdfOffsetResult> {
    // Step 1: Create voxel grid
    let mut grid = SdfGrid::from_mesh_bounds(
        mesh,
        params.voxel_size_mm,
        params.padding_mm,
        params.max_voxels,
    )?;

    info!(
        dims = ?grid.dims,
        total_voxels = grid.total_voxels(),
        "Grid created (standard)"
    );

    // Step 2: Compute base SDF
    let sdf_start = Instant::now();
    grid.compute_sdf(mesh);
    let sdf_time_ms = sdf_start.elapsed().as_millis() as u64;

    debug!(sdf_time_ms, "SDF computation complete");

    // Step 3: Interpolate offsets into grid
    grid.interpolate_offsets(mesh, params.offset_neighbors);

    // Step 4: Apply variable offset
    grid.apply_variable_offset();

    // Step 5: Extract isosurface
    let extract_start = Instant::now();
    let mut output_mesh = extract_isosurface(&grid)?;
    let extraction_time_ms = extract_start.elapsed().as_millis() as u64;

    debug!(
        extraction_time_ms,
        vertices = output_mesh.vertices.len(),
        faces = output_mesh.faces.len(),
        "Surface extraction complete"
    );

    // Step 6: Transfer vertex data from original mesh
    let transfer_start = Instant::now();
    transfer_vertex_data(mesh, &mut output_mesh)?;
    let transfer_time_ms = transfer_start.elapsed().as_millis() as u64;

    debug!(transfer_time_ms, "Vertex data transfer complete");

    let total_time_ms = total_start.elapsed().as_millis();

    let stats = SdfOffsetStats {
        grid_dims: grid.dims,
        total_voxels: grid.total_voxels(),
        sdf_time_ms,
        extraction_time_ms,
        transfer_time_ms,
        input_vertices,
        output_vertices: output_mesh.vertices.len(),
        output_faces: output_mesh.faces.len(),
        adaptive_resolution: false,
        memory_savings_percent: 0.0,
    };

    info!(
        total_time_ms,
        input_vertices = stats.input_vertices,
        output_vertices = stats.output_vertices,
        output_faces = stats.output_faces,
        "SDF offset complete (standard)"
    );

    Ok(SdfOffsetResult {
        mesh: output_mesh,
        stats,
    })
}

/// Adaptive multi-resolution SDF offset implementation.
fn apply_sdf_offset_adaptive(
    mesh: &Mesh,
    params: &SdfOffsetParams,
    input_vertices: usize,
    total_start: Instant,
) -> ShellResult<SdfOffsetResult> {
    // Convert SdfOffsetParams to AdaptiveSdfParams
    let adaptive_params = AdaptiveSdfParams {
        fine_voxel_size_mm: params.voxel_size_mm,
        coarse_voxel_size_mm: params.coarse_voxel_size_mm(),
        refinement_distance_mm: params.refinement_distance_mm,
        padding_mm: params.padding_mm,
        max_voxels: params.max_voxels,
        offset_neighbors: params.offset_neighbors,
    };

    // Step 1-2: Create adaptive grid and compute SDF
    let sdf_start = Instant::now();
    let adaptive_result = create_adaptive_grid(mesh, &adaptive_params)?;
    let mut grid = adaptive_result.grid;
    let adaptive_stats = adaptive_result.stats;
    let sdf_time_ms = sdf_start.elapsed().as_millis() as u64;

    info!(
        dims = ?grid.dims,
        total_voxels = grid.total_voxels(),
        coarse_voxels = adaptive_stats.coarse_voxels,
        refined_voxels = adaptive_stats.refined_coarse_voxels,
        memory_savings = format!("{:.1}%", adaptive_stats.memory_savings_percent),
        "Adaptive grid created"
    );

    debug!(sdf_time_ms, "Adaptive SDF computation complete");

    // Step 3: Interpolate offsets using adaptive method
    interpolate_offsets_adaptive(&mut grid, mesh, &adaptive_params);

    // Step 4: Apply variable offset
    grid.apply_variable_offset();

    // Step 5: Extract isosurface
    let extract_start = Instant::now();
    let mut output_mesh = extract_isosurface(&grid)?;
    let extraction_time_ms = extract_start.elapsed().as_millis() as u64;

    debug!(
        extraction_time_ms,
        vertices = output_mesh.vertices.len(),
        faces = output_mesh.faces.len(),
        "Surface extraction complete"
    );

    // Step 6: Transfer vertex data from original mesh
    let transfer_start = Instant::now();
    transfer_vertex_data(mesh, &mut output_mesh)?;
    let transfer_time_ms = transfer_start.elapsed().as_millis() as u64;

    debug!(transfer_time_ms, "Vertex data transfer complete");

    let total_time_ms = total_start.elapsed().as_millis();

    let stats = SdfOffsetStats {
        grid_dims: grid.dims,
        total_voxels: grid.total_voxels(),
        sdf_time_ms,
        extraction_time_ms,
        transfer_time_ms,
        input_vertices,
        output_vertices: output_mesh.vertices.len(),
        output_faces: output_mesh.faces.len(),
        adaptive_resolution: true,
        memory_savings_percent: adaptive_stats.memory_savings_percent,
    };

    info!(
        total_time_ms,
        input_vertices = stats.input_vertices,
        output_vertices = stats.output_vertices,
        output_faces = stats.output_faces,
        memory_savings = format!("{:.1}%", stats.memory_savings_percent),
        "SDF offset complete (adaptive)"
    );

    Ok(SdfOffsetResult {
        mesh: output_mesh,
        stats,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use mesh_repair::Vertex;

    fn create_unit_cube() -> Mesh {
        let mut mesh = Mesh::new();

        // Cube vertices (0-10mm)
        for z in [0.0, 10.0] {
            for y in [0.0, 10.0] {
                for x in [0.0, 10.0] {
                    let mut v = Vertex::from_coords(x, y, z);
                    v.offset = Some(1.0); // 1mm uniform offset
                    v.tag = Some(1);
                    mesh.vertices.push(v);
                }
            }
        }

        // Cube faces (2 triangles per face)
        mesh.faces.push([0, 1, 3]);
        mesh.faces.push([0, 3, 2]);
        mesh.faces.push([4, 7, 5]);
        mesh.faces.push([4, 6, 7]);
        mesh.faces.push([0, 5, 1]);
        mesh.faces.push([0, 4, 5]);
        mesh.faces.push([2, 3, 7]);
        mesh.faces.push([2, 7, 6]);
        mesh.faces.push([0, 2, 6]);
        mesh.faces.push([0, 6, 4]);
        mesh.faces.push([1, 5, 7]);
        mesh.faces.push([1, 7, 3]);

        mesh
    }

    #[test]
    fn test_sdf_offset_cube() {
        let mesh = create_unit_cube();

        let params = SdfOffsetParams {
            voxel_size_mm: 1.0,
            padding_mm: 5.0,
            max_voxels: 1_000_000,
            offset_neighbors: 4,
            adaptive_resolution: false,
            coarse_voxel_multiplier: 4.0,
            refinement_distance_mm: 5.0,
            use_gpu: false,
        };

        let result = apply_sdf_offset(&mesh, &params).unwrap();

        // Should produce a valid mesh
        assert!(!result.mesh.vertices.is_empty());
        assert!(!result.mesh.faces.is_empty());
        assert!(!result.stats.adaptive_resolution);

        // Output should be larger than input (we're expanding)
        let input_bounds = mesh.bounds().unwrap();
        let output_bounds = result.mesh.bounds().unwrap();

        let input_extent = input_bounds.1 - input_bounds.0;
        let output_extent = output_bounds.1 - output_bounds.0;

        // With 1mm offset, output should be ~2mm larger in each dimension
        assert!(
            output_extent.x > input_extent.x,
            "Output should be wider: {} vs {}",
            output_extent.x,
            input_extent.x
        );
    }

    #[test]
    fn test_sdf_offset_cube_adaptive() {
        let mesh = create_unit_cube();

        let params = SdfOffsetParams::adaptive();

        let result = apply_sdf_offset(&mesh, &params).unwrap();

        // Should produce a valid mesh
        assert!(!result.mesh.vertices.is_empty());
        assert!(!result.mesh.faces.is_empty());
        assert!(result.stats.adaptive_resolution);

        // Output should be larger than input (we're expanding)
        let input_bounds = mesh.bounds().unwrap();
        let output_bounds = result.mesh.bounds().unwrap();

        let input_extent = input_bounds.1 - input_bounds.0;
        let output_extent = output_bounds.1 - output_bounds.0;

        // With offset, output should be larger
        assert!(
            output_extent.x > input_extent.x,
            "Adaptive output should be wider: {} vs {}",
            output_extent.x,
            input_extent.x
        );
    }

    #[test]
    fn test_sdf_offset_adaptive_presets() {
        let mesh = create_unit_cube();

        // Test all adaptive presets compile and run
        for params in [
            SdfOffsetParams::adaptive(),
            SdfOffsetParams::adaptive_high_quality(),
            SdfOffsetParams::adaptive_large_mesh(),
        ] {
            let result = apply_sdf_offset(&mesh, &params).unwrap();
            assert!(!result.mesh.vertices.is_empty());
            assert!(result.stats.adaptive_resolution);
        }
    }

    #[test]
    fn test_sdf_offset_empty_mesh() {
        let mesh = Mesh::new();
        let params = SdfOffsetParams::default();

        let result = apply_sdf_offset(&mesh, &params);
        assert!(result.is_err());
    }
}

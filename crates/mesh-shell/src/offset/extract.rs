//! Surface extraction from SDF grid using Surface Nets.

use tracing::{debug, info};

use mesh_repair::{Mesh, Vertex};

use crate::error::{ShellError, ShellResult};

use super::grid::SdfGrid;

/// Extract isosurface from SDF grid using Surface Nets algorithm.
///
/// Returns a new mesh representing the zero isosurface of the SDF.
pub fn extract_isosurface(grid: &SdfGrid) -> ShellResult<Mesh> {
    use fast_surface_nets::{SurfaceNetsBuffer, ndshape::RuntimeShape, surface_nets};

    info!(dims = ?grid.dims, "Extracting isosurface");

    // fast-surface-nets expects the SDF values in a specific format
    // We need to pad the array for the algorithm
    let padded_dims = [grid.dims[0] + 2, grid.dims[1] + 2, grid.dims[2] + 2];
    let padded_size = padded_dims[0] * padded_dims[1] * padded_dims[2];

    // Create padded SDF array (exterior = positive = outside)
    let mut padded_sdf = vec![1000.0f32; padded_size];

    // Copy grid values into center of padded array
    for z in 0..grid.dims[2] {
        for y in 0..grid.dims[1] {
            for x in 0..grid.dims[0] {
                let src_idx = x + y * grid.dims[0] + z * grid.dims[0] * grid.dims[1];
                let dst_idx =
                    (x + 1) + (y + 1) * padded_dims[0] + (z + 1) * padded_dims[0] * padded_dims[1];
                padded_sdf[dst_idx] = grid.values[src_idx];
            }
        }
    }

    // Create runtime shape for fast-surface-nets
    let shape = RuntimeShape::<u32, 3>::new([
        padded_dims[0] as u32,
        padded_dims[1] as u32,
        padded_dims[2] as u32,
    ]);

    // Run surface nets
    let mut buffer = SurfaceNetsBuffer::default();
    surface_nets(
        &padded_sdf,
        &shape,
        [0, 0, 0],
        [
            padded_dims[0] as u32 - 1,
            padded_dims[1] as u32 - 1,
            padded_dims[2] as u32 - 1,
        ],
        &mut buffer,
    );

    if buffer.positions.is_empty() {
        return Err(ShellError::EmptyIsosurface);
    }

    debug!(
        positions = buffer.positions.len(),
        indices = buffer.indices.len(),
        "Surface nets complete"
    );

    // Convert to Mesh
    let mut mesh = Mesh::new();

    // Convert positions to vertices (account for padding offset)
    for pos in &buffer.positions {
        // Positions are in grid coordinates, convert to world
        // Subtract 1 from each coordinate to account for padding
        let world_x = grid.origin.x + (pos[0] - 1.0) as f64 * grid.voxel_size;
        let world_y = grid.origin.y + (pos[1] - 1.0) as f64 * grid.voxel_size;
        let world_z = grid.origin.z + (pos[2] - 1.0) as f64 * grid.voxel_size;

        mesh.vertices
            .push(Vertex::from_coords(world_x, world_y, world_z));
    }

    // Convert indices to faces (already triangulated)
    for chunk in buffer.indices.chunks(3) {
        if chunk.len() == 3 {
            mesh.faces.push([chunk[0], chunk[1], chunk[2]]);
        }
    }

    info!(
        vertices = mesh.vertices.len(),
        faces = mesh.faces.len(),
        "Isosurface mesh created"
    );

    Ok(mesh)
}

#[cfg(test)]
mod tests {
    use super::*;
    use mesh_repair::Vertex;

    fn create_unit_cube() -> Mesh {
        let mut mesh = Mesh::new();

        for z in [0.0, 10.0] {
            for y in [0.0, 10.0] {
                for x in [0.0, 10.0] {
                    mesh.vertices.push(Vertex::from_coords(x, y, z));
                }
            }
        }

        mesh.faces.push([0, 1, 2]);
        mesh.faces.push([0, 2, 3]);
        mesh.faces.push([4, 6, 5]);
        mesh.faces.push([4, 7, 6]);
        mesh.faces.push([0, 5, 1]);
        mesh.faces.push([0, 4, 5]);
        mesh.faces.push([2, 7, 3]);
        mesh.faces.push([2, 6, 7]);
        mesh.faces.push([0, 3, 7]);
        mesh.faces.push([0, 7, 4]);
        mesh.faces.push([1, 5, 6]);
        mesh.faces.push([1, 6, 2]);

        mesh
    }

    #[test]
    fn test_extract_isosurface() {
        let mesh = create_unit_cube();
        let mut grid = SdfGrid::from_mesh_bounds(&mesh, 2.0, 5.0, 1_000_000).unwrap();
        grid.compute_sdf(&mesh);

        let result = extract_isosurface(&grid);
        assert!(result.is_ok());

        let extracted = result.unwrap();
        assert!(!extracted.vertices.is_empty());
        assert!(!extracted.faces.is_empty());
    }
}

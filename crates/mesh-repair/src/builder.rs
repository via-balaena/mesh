//! Fluent builder APIs for mesh repair operations.
//!
//! This module provides ergonomic builder patterns for configuring and executing
//! mesh repair operations. Builders allow chaining configuration methods
//! and repair operations before executing them.
//!
//! # Example
//!
//! ```no_run
//! use mesh_repair::{Mesh, RepairBuilder};
//!
//! let mesh = Mesh::load("scan.stl").unwrap();
//!
//! // Fluent API for mesh repair
//! let repaired = RepairBuilder::new(mesh)
//!     .for_scans()                    // Use scan-optimized settings
//!     .weld_vertices(0.01)            // Weld vertices within 0.01mm
//!     .remove_degenerates()           // Remove degenerate triangles
//!     .fix_winding()                  // Fix winding order
//!     .fill_holes(100)                // Fill holes up to 100 edges
//!     .build()
//!     .unwrap();
//!
//! repaired.mesh.save("repaired.stl").unwrap();
//! ```

use crate::Mesh;
use crate::components::remove_small_components;
use crate::error::MeshResult;
use crate::holes::fill_holes_with_max_edges;
use crate::progress::ProgressCallback;
use crate::repair::{
    RepairParams, compute_vertex_normals, fix_non_manifold_edges,
    remove_degenerate_triangles_enhanced, remove_duplicate_faces, remove_unreferenced_vertices,
    weld_vertices,
};
use crate::winding::fix_winding_order;

/// Result from RepairBuilder containing the repaired mesh and statistics.
#[derive(Debug, Clone, Default)]
pub struct RepairResult {
    /// The repaired mesh.
    pub mesh: Mesh,
    /// Number of vertices welded.
    pub vertices_welded: usize,
    /// Number of degenerate triangles removed.
    pub degenerates_removed: usize,
    /// Number of duplicate faces removed.
    pub duplicates_removed: usize,
    /// Number of non-manifold edges fixed.
    pub non_manifold_fixed: usize,
    /// Number of holes filled.
    pub holes_filled: usize,
    /// Number of small components removed.
    pub components_removed: usize,
    /// Number of unreferenced vertices removed.
    pub unreferenced_removed: usize,
    /// Whether winding order was fixed.
    pub winding_fixed: bool,
}

/// Queued repair operation.
#[derive(Debug, Clone)]
enum RepairOp {
    WeldVertices(f64),
    RemoveDegenerates {
        area_threshold: f64,
        aspect_ratio: f64,
        min_edge_length: f64,
    },
    RemoveDuplicates,
    FixNonManifold,
    FixWinding,
    FillHoles(usize),
    RemoveSmallComponents(usize),
    RemoveUnreferenced,
    ComputeNormals,
}

/// Fluent builder for mesh repair operations.
///
/// RepairBuilder provides a chainable API for configuring and executing
/// mesh repair operations. Operations are queued and executed in order
/// when `build()` is called.
///
/// # Example
///
/// ```no_run
/// use mesh_repair::{Mesh, RepairBuilder};
///
/// let mesh = Mesh::load("scan.stl").unwrap();
///
/// // Chain multiple repair operations
/// let repaired = RepairBuilder::new(mesh)
///     .weld_vertices(0.01)
///     .remove_degenerates()
///     .fix_winding()
///     .build()
///     .unwrap();
/// ```
///
/// # Operation Order
///
/// Operations are executed in the order they are added. For best results,
/// the typical repair sequence is:
///
/// 1. `weld_vertices()` - Merge nearby vertices
/// 2. `remove_degenerates()` - Remove bad triangles
/// 3. `remove_duplicates()` - Remove duplicate faces
/// 4. `fix_non_manifold()` - Fix non-manifold edges
/// 5. `fix_winding()` - Make normals consistent
/// 6. `fill_holes()` - Close holes
/// 7. `remove_small_components()` - Remove debris
/// 8. `remove_unreferenced()` - Clean up orphan vertices
/// 9. `compute_normals()` - Recompute normals
///
/// Use the preset methods (`for_scans()`, `for_cad()`, `for_printing()`)
/// to automatically queue the appropriate operations.
pub struct RepairBuilder {
    mesh: Mesh,
    operations: Vec<RepairOp>,
    progress_callback: Option<ProgressCallback>,
}

impl RepairBuilder {
    /// Create a new RepairBuilder for the given mesh.
    ///
    /// # Arguments
    ///
    /// * `mesh` - The input mesh to repair (takes ownership)
    ///
    /// # Example
    ///
    /// ```no_run
    /// use mesh_repair::{Mesh, RepairBuilder};
    ///
    /// let mesh = Mesh::load("scan.stl").unwrap();
    /// let builder = RepairBuilder::new(mesh);
    /// ```
    pub fn new(mesh: Mesh) -> Self {
        Self {
            mesh,
            operations: Vec::new(),
            progress_callback: None,
        }
    }

    // =========================================================================
    // Presets
    // =========================================================================

    /// Configure for 3D scan data.
    ///
    /// Uses aggressive settings suitable for noisy scan data. Queues:
    /// - Weld vertices (0.01mm)
    /// - Remove degenerates (aggressive)
    /// - Remove duplicates
    /// - Fix non-manifold edges
    /// - Fix winding order
    /// - Fill holes (up to 200 edges)
    /// - Remove unreferenced vertices
    /// - Compute normals
    ///
    /// # Example
    ///
    /// ```no_run
    /// use mesh_repair::{Mesh, RepairBuilder};
    ///
    /// let mesh = Mesh::load("scan.stl").unwrap();
    /// let repaired = RepairBuilder::new(mesh)
    ///     .for_scans()
    ///     .build()
    ///     .unwrap();
    /// ```
    pub fn for_scans(mut self) -> Self {
        let params = RepairParams::for_scans();
        self.operations.clear();
        self.operations
            .push(RepairOp::WeldVertices(params.weld_epsilon));
        self.operations.push(RepairOp::RemoveDegenerates {
            area_threshold: params.degenerate_area_threshold,
            aspect_ratio: params.degenerate_aspect_ratio,
            min_edge_length: params.degenerate_min_edge_length,
        });
        self.operations.push(RepairOp::RemoveDuplicates);
        self.operations.push(RepairOp::FixNonManifold);
        self.operations.push(RepairOp::FixWinding);
        self.operations
            .push(RepairOp::FillHoles(params.max_hole_edges));
        self.operations.push(RepairOp::RemoveUnreferenced);
        self.operations.push(RepairOp::ComputeNormals);
        self
    }

    /// Configure for CAD models.
    ///
    /// Uses conservative settings to preserve intentional geometry. Queues:
    /// - Weld vertices (1e-9mm)
    /// - Remove degenerates (conservative)
    /// - Remove duplicates
    /// - Fix non-manifold edges
    /// - Fix winding order
    /// - Remove unreferenced vertices
    /// - Compute normals
    ///
    /// Note: Does NOT fill holes (often intentional in CAD).
    pub fn for_cad(mut self) -> Self {
        let params = RepairParams::for_cad();
        self.operations.clear();
        self.operations
            .push(RepairOp::WeldVertices(params.weld_epsilon));
        self.operations.push(RepairOp::RemoveDegenerates {
            area_threshold: params.degenerate_area_threshold,
            aspect_ratio: params.degenerate_aspect_ratio,
            min_edge_length: params.degenerate_min_edge_length,
        });
        self.operations.push(RepairOp::RemoveDuplicates);
        self.operations.push(RepairOp::FixNonManifold);
        self.operations.push(RepairOp::FixWinding);
        self.operations.push(RepairOp::RemoveUnreferenced);
        self.operations.push(RepairOp::ComputeNormals);
        self
    }

    /// Configure for 3D printing preparation.
    ///
    /// Ensures watertight, manifold output suitable for slicing. Queues:
    /// - Weld vertices (0.001mm)
    /// - Remove degenerates (moderate)
    /// - Remove duplicates
    /// - Fix non-manifold edges
    /// - Fix winding order
    /// - Fill holes (up to 500 edges)
    /// - Remove unreferenced vertices
    /// - Compute normals
    pub fn for_printing(mut self) -> Self {
        let params = RepairParams::for_printing();
        self.operations.clear();
        self.operations
            .push(RepairOp::WeldVertices(params.weld_epsilon));
        self.operations.push(RepairOp::RemoveDegenerates {
            area_threshold: params.degenerate_area_threshold,
            aspect_ratio: params.degenerate_aspect_ratio,
            min_edge_length: params.degenerate_min_edge_length,
        });
        self.operations.push(RepairOp::RemoveDuplicates);
        self.operations.push(RepairOp::FixNonManifold);
        self.operations.push(RepairOp::FixWinding);
        self.operations
            .push(RepairOp::FillHoles(params.max_hole_edges));
        self.operations.push(RepairOp::RemoveUnreferenced);
        self.operations.push(RepairOp::ComputeNormals);
        self
    }

    // =========================================================================
    // Individual Operations
    // =========================================================================

    /// Queue vertex welding operation.
    ///
    /// Merges vertices that are within `epsilon` distance of each other.
    /// This fixes gaps from non-watertight meshes and reduces vertex count.
    ///
    /// # Arguments
    ///
    /// * `epsilon` - Maximum distance between vertices to merge (in mesh units, typically mm)
    ///
    /// # Recommended Values
    ///
    /// - High-precision CAD: `1e-9` to `1e-6`
    /// - 3D scans: `0.001` to `0.1`
    /// - Low-poly models: `0.01` to `1.0`
    pub fn weld_vertices(mut self, epsilon: f64) -> Self {
        self.operations.push(RepairOp::WeldVertices(epsilon));
        self
    }

    /// Queue degenerate triangle removal with default thresholds.
    ///
    /// Removes triangles that have:
    /// - Area below 1e-9 mm²
    /// - Aspect ratio above 1000
    /// - Any edge shorter than 1e-9 mm
    pub fn remove_degenerates(self) -> Self {
        self.remove_degenerates_with_thresholds(1e-9, 1000.0, 1e-9)
    }

    /// Queue degenerate triangle removal with custom thresholds.
    ///
    /// # Arguments
    ///
    /// * `area_threshold` - Minimum triangle area
    /// * `aspect_ratio` - Maximum aspect ratio (longest edge / shortest altitude)
    /// * `min_edge_length` - Minimum edge length
    pub fn remove_degenerates_with_thresholds(
        mut self,
        area_threshold: f64,
        aspect_ratio: f64,
        min_edge_length: f64,
    ) -> Self {
        self.operations.push(RepairOp::RemoveDegenerates {
            area_threshold,
            aspect_ratio,
            min_edge_length,
        });
        self
    }

    /// Queue duplicate face removal.
    ///
    /// Removes faces that share the same vertices (in any winding order).
    pub fn remove_duplicates(mut self) -> Self {
        self.operations.push(RepairOp::RemoveDuplicates);
        self
    }

    /// Queue non-manifold edge fixing.
    ///
    /// Non-manifold edges are edges shared by more than 2 faces.
    /// This operation removes excess faces, keeping the 2 largest.
    pub fn fix_non_manifold(mut self) -> Self {
        self.operations.push(RepairOp::FixNonManifold);
        self
    }

    /// Queue winding order fixing.
    ///
    /// Makes all face normals point consistently outward.
    /// Requires the mesh to have a clear "outside" direction.
    pub fn fix_winding(mut self) -> Self {
        self.operations.push(RepairOp::FixWinding);
        self
    }

    /// Queue hole filling.
    ///
    /// Fills boundary loops (holes) up to `max_edges` in size
    /// using ear-clipping triangulation.
    ///
    /// # Arguments
    ///
    /// * `max_edges` - Maximum number of edges in a hole to fill
    pub fn fill_holes(mut self, max_edges: usize) -> Self {
        self.operations.push(RepairOp::FillHoles(max_edges));
        self
    }

    /// Queue small component removal.
    ///
    /// Removes disconnected mesh components with fewer than `min_faces` faces.
    /// Useful for removing floating debris from scans.
    ///
    /// # Arguments
    ///
    /// * `min_faces` - Minimum number of faces for a component to keep
    pub fn remove_small_components(mut self, min_faces: usize) -> Self {
        self.operations
            .push(RepairOp::RemoveSmallComponents(min_faces));
        self
    }

    /// Queue unreferenced vertex removal.
    ///
    /// Removes vertices that are not referenced by any face.
    pub fn remove_unreferenced(mut self) -> Self {
        self.operations.push(RepairOp::RemoveUnreferenced);
        self
    }

    /// Queue vertex normal computation.
    ///
    /// Computes smooth vertex normals by averaging face normals.
    pub fn compute_normals(mut self) -> Self {
        self.operations.push(RepairOp::ComputeNormals);
        self
    }

    // =========================================================================
    // Progress Callback
    // =========================================================================

    /// Set a progress callback for long-running operations.
    ///
    /// The callback receives progress information and should return `true`
    /// to continue or `false` to request cancellation.
    ///
    /// # Example
    ///
    /// ```no_run
    /// use mesh_repair::{Mesh, RepairBuilder};
    /// use mesh_repair::progress::ProgressCallback;
    ///
    /// let mesh = Mesh::load("scan.stl").unwrap();
    ///
    /// let callback: ProgressCallback = Box::new(|progress| {
    ///     println!("{}%: {}", progress.percent(), progress.message);
    ///     true
    /// });
    ///
    /// let repaired = RepairBuilder::new(mesh)
    ///     .for_scans()
    ///     .with_progress(callback)
    ///     .build()
    ///     .unwrap();
    /// ```
    pub fn with_progress(mut self, callback: ProgressCallback) -> Self {
        self.progress_callback = Some(callback);
        self
    }

    // =========================================================================
    // Build
    // =========================================================================

    /// Execute all queued repair operations and return the result.
    ///
    /// Operations are executed in the order they were added.
    ///
    /// # Returns
    ///
    /// A `RepairResult` containing the repaired mesh and statistics.
    ///
    /// # Errors
    ///
    /// Returns an error if any repair operation fails.
    pub fn build(mut self) -> MeshResult<RepairResult> {
        use crate::progress::{Progress, ProgressTracker};

        let mut result = RepairResult::default();
        let total_ops = self.operations.len() as u64;

        // Progress tracking
        let tracker = ProgressTracker::new(total_ops * 100);

        for (i, op) in self.operations.iter().enumerate() {
            let progress_base = (i as u64) * 100;

            // Report progress
            if let Some(ref callback) = self.progress_callback {
                let progress = Progress::new(
                    progress_base,
                    total_ops * 100,
                    format!("Executing repair step {}/{}", i + 1, total_ops),
                );
                if !callback(&progress) {
                    // Cancelled
                    result.mesh = self.mesh;
                    return Ok(result);
                }
            }

            match op {
                RepairOp::WeldVertices(epsilon) => {
                    result.vertices_welded += weld_vertices(&mut self.mesh, *epsilon);
                }
                RepairOp::RemoveDegenerates {
                    area_threshold,
                    aspect_ratio,
                    min_edge_length,
                } => {
                    result.degenerates_removed += remove_degenerate_triangles_enhanced(
                        &mut self.mesh,
                        *area_threshold,
                        *aspect_ratio,
                        *min_edge_length,
                    );
                }
                RepairOp::RemoveDuplicates => {
                    result.duplicates_removed += remove_duplicate_faces(&mut self.mesh);
                }
                RepairOp::FixNonManifold => {
                    result.non_manifold_fixed += fix_non_manifold_edges(&mut self.mesh);
                }
                RepairOp::FixWinding => {
                    if fix_winding_order(&mut self.mesh).is_ok() {
                        result.winding_fixed = true;
                    }
                }
                RepairOp::FillHoles(max_edges) => {
                    if let Ok(count) = fill_holes_with_max_edges(&mut self.mesh, *max_edges) {
                        result.holes_filled += count;
                    }
                }
                RepairOp::RemoveSmallComponents(min_faces) => {
                    result.components_removed +=
                        remove_small_components(&mut self.mesh, *min_faces);
                }
                RepairOp::RemoveUnreferenced => {
                    result.unreferenced_removed += remove_unreferenced_vertices(&mut self.mesh);
                }
                RepairOp::ComputeNormals => {
                    compute_vertex_normals(&mut self.mesh);
                }
            }

            tracker.set(progress_base + 100);
        }

        // Final progress report
        if let Some(ref callback) = self.progress_callback {
            let progress = Progress::new(total_ops * 100, total_ops * 100, "Repair complete");
            callback(&progress);
        }

        result.mesh = self.mesh;
        Ok(result)
    }

    /// Get the current mesh without executing any remaining operations.
    ///
    /// This consumes the builder and returns the mesh in its current state.
    pub fn into_mesh(self) -> Mesh {
        self.mesh
    }

    /// Get a reference to the current mesh.
    pub fn mesh(&self) -> &Mesh {
        &self.mesh
    }

    /// Get the number of queued operations.
    pub fn operation_count(&self) -> usize {
        self.operations.len()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::Vertex;

    fn create_test_cube() -> Mesh {
        let mut mesh = Mesh::new();

        // 8 vertices of a unit cube
        mesh.vertices = vec![
            Vertex::from_coords(0.0, 0.0, 0.0),
            Vertex::from_coords(10.0, 0.0, 0.0),
            Vertex::from_coords(10.0, 10.0, 0.0),
            Vertex::from_coords(0.0, 10.0, 0.0),
            Vertex::from_coords(0.0, 0.0, 10.0),
            Vertex::from_coords(10.0, 0.0, 10.0),
            Vertex::from_coords(10.0, 10.0, 10.0),
            Vertex::from_coords(0.0, 10.0, 10.0),
        ];

        // 12 triangles (2 per face)
        mesh.faces = vec![
            // Bottom
            [0, 2, 1],
            [0, 3, 2],
            // Top
            [4, 5, 6],
            [4, 6, 7],
            // Front
            [0, 1, 5],
            [0, 5, 4],
            // Back
            [2, 3, 7],
            [2, 7, 6],
            // Left
            [0, 4, 7],
            [0, 7, 3],
            // Right
            [1, 2, 6],
            [1, 6, 5],
        ];

        mesh
    }

    #[test]
    fn test_builder_defaults() {
        let mesh = create_test_cube();
        let builder = RepairBuilder::new(mesh);

        assert_eq!(builder.operation_count(), 0);
    }

    #[test]
    fn test_builder_chaining() {
        let mesh = create_test_cube();
        let builder = RepairBuilder::new(mesh)
            .weld_vertices(0.01)
            .remove_degenerates()
            .fix_winding();

        assert_eq!(builder.operation_count(), 3);
    }

    #[test]
    fn test_for_scans_preset() {
        let mesh = create_test_cube();
        let builder = RepairBuilder::new(mesh).for_scans();

        // Should have multiple operations queued
        assert!(builder.operation_count() >= 5);
    }

    #[test]
    fn test_for_printing_preset() {
        let mesh = create_test_cube();
        let builder = RepairBuilder::new(mesh).for_printing();

        // Should have multiple operations queued
        assert!(builder.operation_count() >= 5);
    }

    #[test]
    fn test_build_simple() {
        let mesh = create_test_cube();
        let result = RepairBuilder::new(mesh)
            .remove_degenerates()
            .compute_normals()
            .build()
            .unwrap();

        // Cube should be intact (no degenerates to remove)
        assert_eq!(result.mesh.faces.len(), 12);
        assert_eq!(result.degenerates_removed, 0);
    }

    #[test]
    fn test_into_mesh() {
        let mesh = create_test_cube();
        let vertex_count = mesh.vertices.len();

        let recovered = RepairBuilder::new(mesh).into_mesh();

        assert_eq!(recovered.vertices.len(), vertex_count);
    }
}

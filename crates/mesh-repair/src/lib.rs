//! Triangle mesh repair and processing utilities.
//!
//! This crate provides comprehensive tools for loading, validating, repairing, and
//! transforming triangle meshes. It's designed for 3D printing pipelines, mesh
//! processing, and geometry operations.
//!
//! # Features
//!
//! - **File I/O**: Load and save STL, OBJ, 3MF, and PLY formats
//! - **Validation**: Check for non-manifold edges, holes, self-intersections, winding issues
//! - **Repair**: Fill holes, fix winding, remove degenerates, weld vertices
//! - **Analysis**: Component detection, wall thickness measurement, volume/surface area
//! - **Transformation**: Decimation, subdivision, isotropic remeshing
//!
//! # Units and Scale
//!
//! **This library assumes millimeter (mm) units.**
//!
//! - Default hole filling skips holes larger than 500 edges (adjustable via params)
//! - Default vertex welding tolerance is 1e-6 (sub-micron precision)
//! - Wall thickness analysis defaults: 1mm minimum for FDM, 0.4mm for SLA
//! - Ray casting max distance defaults to 1000mm (1 meter)
//!
//! If your mesh uses different units, scale accordingly:
//! - **Meters → mm**: Multiply all coordinates by 1000
//! - **Inches → mm**: Multiply all coordinates by 25.4
//! - **Microns → mm**: Divide all coordinates by 1000
//!
//! # Coordinate System
//!
//! The library uses a **right-handed coordinate system**:
//! - X: typically width (left/right)
//! - Y: typically depth (front/back)
//! - Z: typically height (up/down)
//!
//! Face winding is **counter-clockwise (CCW) when viewed from outside** the mesh.
//! This means normals point outward by the right-hand rule.
//!
//! # Quick Start
//!
//! ```no_run
//! use mesh_repair::Mesh;
//!
//! // Load a mesh from any supported format
//! let mut mesh = Mesh::load("model.stl").unwrap();
//!
//! // Validate and check for issues
//! let report = mesh.validate();
//! println!("{}", report);
//!
//! // Repair common issues
//! mesh.repair().unwrap();
//!
//! // Save to any supported format
//! mesh.save("repaired.3mf").unwrap();
//! ```
//!
//! # Common Workflows
//!
//! ## 3D Printing Pipeline
//!
//! ```no_run
//! use mesh_repair::{Mesh, RepairParams, ThicknessParams};
//!
//! let mut mesh = Mesh::load("scan.stl").unwrap();
//!
//! // Use printing-optimized repair settings
//! mesh.repair_with_config(&RepairParams::for_printing()).unwrap();
//!
//! // Check printability requirements
//! let report = mesh.validate();
//! if report.is_printable() {
//!     println!("Mesh is ready for printing!");
//! } else {
//!     if !report.is_watertight {
//!         println!("Has {} boundary edges", report.boundary_edge_count);
//!     }
//!     if !report.is_manifold {
//!         println!("Has {} non-manifold edges", report.non_manifold_edge_count);
//!     }
//!     if report.is_inside_out {
//!         println!("Normals are inverted");
//!     }
//! }
//!
//! // Check wall thickness for FDM printing
//! let thickness = mesh.analyze_thickness(&ThicknessParams::for_printing());
//! if thickness.has_thin_regions() {
//!     println!("Warning: {} thin regions below 0.8mm", thickness.thin_regions.len());
//! }
//!
//! mesh.save("print_ready.3mf").unwrap();
//! ```
//!
//! ## Processing 3D Scans
//!
//! ```no_run
//! use mesh_repair::{Mesh, RepairParams};
//!
//! let mut mesh = Mesh::load("scan.ply").unwrap();
//!
//! // Remove small debris/noise components
//! let removed = mesh.remove_small_components(100); // Remove components < 100 faces
//! println!("Removed {} noise components", removed);
//!
//! // Use scan-optimized repair (smaller hole filling, more aggressive cleanup)
//! mesh.repair_with_config(&RepairParams::for_scans()).unwrap();
//!
//! // Remesh for uniform triangle quality
//! let remeshed = mesh.remesh_with_edge_length(2.0); // 2mm target edge length
//!
//! remeshed.mesh.save("processed_scan.obj").unwrap();
//! ```
//!
//! ## CAD Model Cleanup
//!
//! ```no_run
//! use mesh_repair::{Mesh, RepairParams};
//!
//! let mut mesh = Mesh::load("cad_export.stl").unwrap();
//!
//! // CAD models often have precise vertices that shouldn't be welded aggressively
//! mesh.repair_with_config(&RepairParams::for_cad()).unwrap();
//!
//! // Check for self-intersections (common in boolean operation results)
//! let intersections = mesh.detect_self_intersections();
//! if !intersections.is_clean() {
//!     println!("Warning: {} self-intersecting triangle pairs", intersections.intersection_count);
//! }
//!
//! mesh.save("cleaned.stl").unwrap();
//! ```
//!
//! ## Mesh Simplification
//!
//! ```no_run
//! use mesh_repair::{Mesh, DecimateParams};
//!
//! let mesh = Mesh::load("high_poly.obj").unwrap();
//!
//! // Decimate to 25% of original triangles
//! let result = mesh.decimate_with_params(&DecimateParams::with_target_ratio(0.25));
//! println!("Reduced from {} to {} triangles", result.original_triangles, result.final_triangles);
//!
//! // Or decimate to a specific count
//! let result = mesh.decimate_to_count(10000);
//!
//! result.mesh.save("low_poly.obj").unwrap();
//! ```
//!
//! # Error Handling
//!
//! Most operations return `MeshResult<T>`, which is `Result<T, MeshError>`.
//!
//! ```
//! use mesh_repair::{Mesh, MeshError};
//!
//! fn process_mesh(path: &str) -> Result<(), MeshError> {
//!     let mut mesh = Mesh::load(path)?;
//!     mesh.repair()?;
//!     mesh.save("output.stl")?;
//!     Ok(())
//! }
//!
//! // Handle specific errors
//! match Mesh::load("nonexistent.stl") {
//!     Ok(_) => println!("Loaded successfully"),
//!     Err(MeshError::IoRead { path, source }) => {
//!         println!("Failed to read {:?}: {}", path, source);
//!     }
//!     Err(MeshError::ParseError { path, details }) => {
//!         println!("Failed to parse {:?}: {}", path, details);
//!     }
//!     Err(MeshError::UnsupportedFormat { extension }) => {
//!         println!("Unsupported format: {:?}", extension);
//!     }
//!     Err(e) => println!("Other error: {}", e),
//! }
//! ```
//!
//! # Troubleshooting
//!
//! ## "Mesh appears inside-out"
//!
//! This means face normals point inward instead of outward. Fix with:
//! ```
//! use mesh_repair::Mesh;
//! let mut mesh = Mesh::new();
//! // ... load or create mesh
//! mesh.fix_winding().unwrap();
//! ```
//!
//! ## "Mesh has holes / not watertight"
//!
//! Boundary edges indicate gaps in the surface. Fill holes with:
//! ```
//! use mesh_repair::Mesh;
//! let mut mesh = Mesh::new();
//! // ... load or create mesh
//! let filled = mesh.fill_holes().unwrap();
//! println!("Filled {} holes", filled);
//! ```
//!
//! ## "Non-manifold edges detected"
//!
//! This means some edges have more than 2 adjacent faces. Use full repair:
//! ```
//! use mesh_repair::{Mesh, RepairParams};
//! let mut mesh = Mesh::new();
//! // ... load or create mesh
//! let mut params = RepairParams::default();
//! params.fix_non_manifold = true;
//! mesh.repair_with_config(&params).unwrap();
//! ```
//!
//! ## "Scale seems wrong"
//!
//! Check the mesh dimensions and scale if needed:
//! ```
//! use mesh_repair::Mesh;
//! let mesh = Mesh::new();
//! // ... load or create mesh
//! if let Some((min, max)) = mesh.bounds() {
//!     let dims = max - min;
//!     println!("Dimensions: {:.1} x {:.1} x {:.1} mm", dims.x, dims.y, dims.z);
//! }
//! // If dimensions are in meters, they'll be 1000x too small
//! // If dimensions are in inches, they'll be ~25x too small
//! ```
//!
//! ## "Multiple disconnected parts"
//!
//! Keep only the main component or split into separate meshes:
//! ```
//! use mesh_repair::Mesh;
//! let mut mesh = Mesh::new();
//! // ... load or create mesh
//! // Option 1: Keep only largest component
//! let removed = mesh.keep_largest_component();
//! println!("Removed {} small components", removed);
//!
//! // Option 2: Split into separate meshes
//! let parts = mesh.split_components();
//! for (i, part) in parts.iter().enumerate() {
//!     println!("Component {}: {} faces", i, part.face_count());
//! }
//! ```
//!
//! # Supported Formats
//!
//! | Format | Extension | Load | Save | Index Preservation | Notes |
//! |--------|-----------|------|------|-------------------|-------|
//! | STL    | `.stl`    | ✓    | ✓    | ✗                 | Binary & ASCII, common for printing |
//! | OBJ    | `.obj`    | ✓    | ✓    | ✓                 | ASCII, preserves vertex order |
//! | 3MF    | `.3mf`    | ✓    | ✓    | ✓                 | ZIP-compressed XML, modern standard |
//! | PLY    | `.ply`    | ✓    | ✓    | ✓                 | ASCII & binary, supports colors/normals |
//!
//! Note: STL format does not preserve vertex indices because it stores triangles
//! independently. OBJ, 3MF, and PLY use indexed storage and preserve vertex order.

mod error;
mod types;

#[cfg(test)]
mod edge_cases;

pub mod adjacency;
pub mod components;
pub mod decimate;
pub mod holes;
pub mod intersect;
pub mod io;
pub mod remesh;
pub mod repair;
pub mod subdivide;
pub mod thickness;
pub mod validate;
pub mod winding;

// Re-export core types at crate root
pub use error::{MeshError, MeshResult, ValidationIssue};
pub use types::{Mesh, Triangle, Vertex, VertexColor};

// Re-export adjacency at crate root for convenience
pub use adjacency::MeshAdjacency;

// Re-export commonly used functions
pub use io::{load_mesh, save_mesh, save_stl, save_obj, save_3mf, save_ply, save_ply_ascii, MeshFormat};
pub use repair::{
    compute_vertex_normals, fix_inverted_faces, remove_duplicate_faces, fix_non_manifold_edges,
    repair_mesh, repair_mesh_with_config, RepairParams,
    remove_degenerate_triangles, remove_degenerate_triangles_enhanced,
    weld_vertices, remove_unreferenced_vertices,
};
pub use validate::{
    validate_mesh, validate_mesh_data, validate_mesh_data_strict,
    MeshReport, DataValidationResult, ValidationOptions,
};
pub use holes::{fill_holes, fill_holes_with_max_edges, detect_holes, BoundaryLoop};
pub use winding::fix_winding_order;
pub use components::{
    find_connected_components, split_into_components, keep_largest_component,
    remove_small_components, ComponentAnalysis,
};
pub use intersect::{
    detect_self_intersections, IntersectionParams, SelfIntersectionResult,
};
pub use decimate::{decimate_mesh, DecimateParams, DecimateResult};
pub use subdivide::{subdivide_mesh, SubdivideParams, SubdivideResult};
pub use remesh::{remesh_isotropic, RemeshParams, RemeshResult};
pub use thickness::{analyze_thickness, ThicknessParams, ThicknessResult, ThinRegion};

// Convenience methods on Mesh
impl Mesh {
    /// Load a mesh from a file, auto-detecting format from extension.
    pub fn load(path: impl AsRef<std::path::Path>) -> MeshResult<Self> {
        io::load_mesh(path.as_ref())
    }

    /// Save the mesh to a file, auto-detecting format from extension.
    pub fn save(&self, path: impl AsRef<std::path::Path>) -> MeshResult<()> {
        io::save_mesh(self, path.as_ref())
    }

    /// Validate the mesh and return a report of any issues.
    pub fn validate(&self) -> MeshReport {
        validate::validate_mesh(self)
    }

    /// Repair common mesh issues using default parameters.
    ///
    /// For more control, use `repair_with_config`.
    pub fn repair(&mut self) -> MeshResult<()> {
        repair::repair_mesh(self)
    }

    /// Repair common mesh issues with custom parameters.
    ///
    /// # Example
    ///
    /// ```
    /// use mesh_repair::{Mesh, RepairParams};
    ///
    /// let mut mesh = Mesh::new();
    /// // Use scan-optimized parameters
    /// mesh.repair_with_config(&RepairParams::for_scans()).unwrap();
    /// ```
    pub fn repair_with_config(&mut self, params: &repair::RepairParams) -> MeshResult<()> {
        repair::repair_mesh_with_config(self, params)
    }

    /// Compute vertex normals from face normals (area-weighted average).
    pub fn compute_normals(&mut self) {
        repair::compute_vertex_normals(self)
    }

    /// Fix inconsistent face winding to ensure all faces have consistent orientation.
    pub fn fix_winding(&mut self) -> MeshResult<()> {
        winding::fix_winding_order(self)
    }

    /// Fill holes in the mesh.
    pub fn fill_holes(&mut self) -> MeshResult<usize> {
        holes::fill_holes(self)
    }

    /// Find connected components in the mesh.
    pub fn find_components(&self) -> components::ComponentAnalysis {
        components::find_connected_components(self)
    }

    /// Split the mesh into separate meshes, one per connected component.
    pub fn split_components(&self) -> Vec<Mesh> {
        components::split_into_components(self)
    }

    /// Keep only the largest connected component, removing all others.
    /// Returns the number of components removed.
    pub fn keep_largest_component(&mut self) -> usize {
        components::keep_largest_component(self)
    }

    /// Remove components with fewer than `min_faces` faces.
    /// Returns the number of components removed.
    pub fn remove_small_components(&mut self, min_faces: usize) -> usize {
        components::remove_small_components(self, min_faces)
    }

    /// Check for self-intersecting triangles.
    pub fn detect_self_intersections(&self) -> intersect::SelfIntersectionResult {
        intersect::detect_self_intersections(self, &intersect::IntersectionParams::default())
    }

    /// Check for self-intersecting triangles with custom parameters.
    pub fn detect_self_intersections_with_params(
        &self,
        params: &intersect::IntersectionParams,
    ) -> intersect::SelfIntersectionResult {
        intersect::detect_self_intersections(self, params)
    }

    /// Decimate the mesh to reduce triangle count using edge collapse.
    ///
    /// Uses default parameters (50% reduction, preserve boundary).
    /// For more control, use `decimate_with_params`.
    pub fn decimate(&self) -> decimate::DecimateResult {
        decimate::decimate_mesh(self, &decimate::DecimateParams::default())
    }

    /// Decimate the mesh with custom parameters.
    ///
    /// # Example
    ///
    /// ```
    /// use mesh_repair::{Mesh, Vertex, DecimateParams};
    ///
    /// // Create a simple test mesh
    /// let mut mesh = Mesh::new();
    /// mesh.vertices.push(Vertex::from_coords(0.0, 0.0, 0.0));
    /// mesh.vertices.push(Vertex::from_coords(1.0, 0.0, 0.0));
    /// mesh.vertices.push(Vertex::from_coords(0.5, 1.0, 0.0));
    /// mesh.faces.push([0, 1, 2]);
    ///
    /// let result = mesh.decimate_with_params(&DecimateParams::with_target_ratio(0.25));
    /// println!("Reduced from {} to {} triangles", result.original_triangles, result.final_triangles);
    /// ```
    pub fn decimate_with_params(&self, params: &decimate::DecimateParams) -> decimate::DecimateResult {
        decimate::decimate_mesh(self, params)
    }

    /// Decimate the mesh to a target triangle count.
    ///
    /// # Example
    ///
    /// ```
    /// use mesh_repair::{Mesh, Vertex};
    ///
    /// // Create a simple test mesh
    /// let mut mesh = Mesh::new();
    /// mesh.vertices.push(Vertex::from_coords(0.0, 0.0, 0.0));
    /// mesh.vertices.push(Vertex::from_coords(1.0, 0.0, 0.0));
    /// mesh.vertices.push(Vertex::from_coords(0.5, 1.0, 0.0));
    /// mesh.faces.push([0, 1, 2]);
    ///
    /// let result = mesh.decimate_to_count(1);
    /// // With a single triangle, decimation is limited
    /// assert!(result.original_triangles == 1);
    /// ```
    pub fn decimate_to_count(&self, target: usize) -> decimate::DecimateResult {
        decimate::decimate_mesh(self, &decimate::DecimateParams::with_target_triangles(target))
    }

    /// Subdivide the mesh to increase triangle count and smooth the surface.
    ///
    /// Uses Loop subdivision with default parameters (1 iteration).
    /// For more control, use `subdivide_with_params`.
    ///
    /// # Example
    ///
    /// ```
    /// use mesh_repair::{Mesh, Vertex};
    ///
    /// let mut mesh = Mesh::new();
    /// mesh.vertices.push(Vertex::from_coords(0.0, 0.0, 0.0));
    /// mesh.vertices.push(Vertex::from_coords(1.0, 0.0, 0.0));
    /// mesh.vertices.push(Vertex::from_coords(0.5, 1.0, 0.0));
    /// mesh.faces.push([0, 1, 2]);
    ///
    /// let result = mesh.subdivide();
    /// assert_eq!(result.final_triangles, 4); // 1 triangle becomes 4
    /// ```
    pub fn subdivide(&self) -> subdivide::SubdivideResult {
        subdivide::subdivide_mesh(self, &subdivide::SubdivideParams::default())
    }

    /// Subdivide the mesh with custom parameters.
    ///
    /// # Example
    ///
    /// ```
    /// use mesh_repair::{Mesh, Vertex, SubdivideParams};
    ///
    /// let mut mesh = Mesh::new();
    /// mesh.vertices.push(Vertex::from_coords(0.0, 0.0, 0.0));
    /// mesh.vertices.push(Vertex::from_coords(1.0, 0.0, 0.0));
    /// mesh.vertices.push(Vertex::from_coords(0.5, 1.0, 0.0));
    /// mesh.faces.push([0, 1, 2]);
    ///
    /// // Two iterations: 1 -> 4 -> 16 triangles
    /// let result = mesh.subdivide_with_params(&SubdivideParams::with_iterations(2));
    /// assert_eq!(result.final_triangles, 16);
    /// ```
    pub fn subdivide_with_params(&self, params: &subdivide::SubdivideParams) -> subdivide::SubdivideResult {
        subdivide::subdivide_mesh(self, params)
    }

    /// Subdivide the mesh a specific number of times.
    ///
    /// Each iteration roughly quadruples the triangle count.
    ///
    /// # Example
    ///
    /// ```
    /// use mesh_repair::{Mesh, Vertex};
    ///
    /// let mut mesh = Mesh::new();
    /// mesh.vertices.push(Vertex::from_coords(0.0, 0.0, 0.0));
    /// mesh.vertices.push(Vertex::from_coords(1.0, 0.0, 0.0));
    /// mesh.vertices.push(Vertex::from_coords(0.5, 1.0, 0.0));
    /// mesh.faces.push([0, 1, 2]);
    ///
    /// let result = mesh.subdivide_n(3);
    /// // 1 -> 4 -> 16 -> 64
    /// assert_eq!(result.final_triangles, 64);
    /// ```
    pub fn subdivide_n(&self, iterations: usize) -> subdivide::SubdivideResult {
        subdivide::subdivide_mesh(self, &subdivide::SubdivideParams::with_iterations(iterations))
    }

    /// Remesh the mesh to achieve uniform edge lengths and improve triangle quality.
    ///
    /// Uses isotropic remeshing with default parameters (auto-detect target edge length).
    /// For more control, use `remesh_with_params`.
    ///
    /// # Example
    ///
    /// ```
    /// use mesh_repair::{Mesh, Vertex};
    ///
    /// let mut mesh = Mesh::new();
    /// mesh.vertices.push(Vertex::from_coords(0.0, 0.0, 0.0));
    /// mesh.vertices.push(Vertex::from_coords(10.0, 0.0, 0.0));
    /// mesh.vertices.push(Vertex::from_coords(5.0, 8.66, 0.0));
    /// mesh.faces.push([0, 1, 2]);
    ///
    /// let result = mesh.remesh();
    /// println!("Remeshed from {} to {} triangles", result.original_triangles, result.final_triangles);
    /// ```
    pub fn remesh(&self) -> remesh::RemeshResult {
        remesh::remesh_isotropic(self, &remesh::RemeshParams::default())
    }

    /// Remesh the mesh with custom parameters.
    ///
    /// # Example
    ///
    /// ```
    /// use mesh_repair::{Mesh, Vertex, RemeshParams};
    ///
    /// let mut mesh = Mesh::new();
    /// mesh.vertices.push(Vertex::from_coords(0.0, 0.0, 0.0));
    /// mesh.vertices.push(Vertex::from_coords(10.0, 0.0, 0.0));
    /// mesh.vertices.push(Vertex::from_coords(5.0, 8.66, 0.0));
    /// mesh.faces.push([0, 1, 2]);
    ///
    /// let result = mesh.remesh_with_params(&RemeshParams::with_target_edge_length(2.0));
    /// println!("Remeshed to {} triangles", result.final_triangles);
    /// ```
    pub fn remesh_with_params(&self, params: &remesh::RemeshParams) -> remesh::RemeshResult {
        remesh::remesh_isotropic(self, params)
    }

    /// Remesh the mesh with a specific target edge length.
    ///
    /// # Example
    ///
    /// ```
    /// use mesh_repair::{Mesh, Vertex};
    ///
    /// let mut mesh = Mesh::new();
    /// mesh.vertices.push(Vertex::from_coords(0.0, 0.0, 0.0));
    /// mesh.vertices.push(Vertex::from_coords(10.0, 0.0, 0.0));
    /// mesh.vertices.push(Vertex::from_coords(5.0, 8.66, 0.0));
    /// mesh.faces.push([0, 1, 2]);
    ///
    /// let result = mesh.remesh_with_edge_length(2.0);
    /// assert!(result.final_triangles > 1);
    /// ```
    pub fn remesh_with_edge_length(&self, target_edge_length: f64) -> remesh::RemeshResult {
        remesh::remesh_isotropic(self, &remesh::RemeshParams::with_target_edge_length(target_edge_length))
    }
}

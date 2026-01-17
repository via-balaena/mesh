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
//! ## Processing 3D Scans (with RepairBuilder)
//!
//! ```no_run
//! use mesh_repair::{Mesh, RepairBuilder};
//!
//! let mesh = Mesh::load("scan.ply").unwrap();
//!
//! // Use fluent builder API for repair operations
//! let result = RepairBuilder::new(mesh)
//!     .for_scans()                        // Use scan-optimized settings
//!     .remove_small_components(100)       // Remove debris < 100 faces
//!     .build()
//!     .unwrap();
//!
//! println!("Welded {} vertices, removed {} degenerates",
//!     result.vertices_welded, result.degenerates_removed);
//!
//! result.mesh.save("processed_scan.obj").unwrap();
//! ```
//!
//! ## Processing 3D Scans (with params)
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

mod builder;
mod error;
mod fitting;
mod pipeline;
pub mod tracing_ext;
mod types;

#[cfg(test)]
mod edge_cases;

pub mod adjacency;
pub mod assembly;
pub mod boolean;
pub mod components;
pub mod decimate;
pub mod holes;
pub mod intersect;
pub mod io;
pub mod lattice;
pub mod measure;
pub mod morph;
pub mod multiscan;
pub mod pointcloud;
pub mod printability;
pub mod progress;
pub mod region;
pub mod registration;
pub mod remesh;
pub mod repair;
pub mod scan;
pub mod slice;
pub mod subdivide;
pub mod template;
pub mod thickness;
pub mod validate;
pub mod winding;

// STEP export (feature-gated)
#[cfg(feature = "step")]
pub mod step;

// Re-export STEP types when feature is enabled
#[cfg(feature = "step")]
pub use step::{StepExportParams, StepExportResult, export_step, export_step_to_string};

// Re-export core types at crate root
pub use error::{
    ErrorCode, IssueSeverity, MeshError, MeshResult, RecoverySuggestion, ValidationIssue,
};
pub use types::{Mesh, Triangle, Vertex, VertexColor};

// Re-export adjacency at crate root for convenience
pub use adjacency::MeshAdjacency;

// Re-export commonly used functions
pub use io::{
    // 3MF Beam Lattice Extension types
    Beam,
    BeamCap,
    BeamLatticeData,
    BeamSet,
    // 3MF Color Group types
    ColorGroup,
    MeshFormat,
    ThreeMfExportParams,
    ThreeMfLoadResult,
    TriangleColors,
    load_3mf_with_materials,
    load_mesh,
    save_3mf,
    // 3MF extended export with all extensions
    save_3mf_extended,
    // 3MF with materials support
    save_3mf_with_materials,
    save_mesh,
    save_obj,
    save_ply,
    save_ply_ascii,
    save_stl,
};
pub use repair::{
    RepairParams, compute_vertex_normals, fix_inverted_faces, fix_non_manifold_edges,
    remove_degenerate_triangles, remove_degenerate_triangles_enhanced, remove_duplicate_faces,
    remove_unreferenced_vertices, repair_mesh, repair_mesh_with_config, weld_vertices,
};

// Builder API
pub use builder::{RepairBuilder, RepairResult};
pub use fitting::{FittingBuilder, FittingResult};
pub use pipeline::{IntoPipeline, Pipeline, PipelineResult};

// Pipeline serialization (requires pipeline-config feature)
pub use components::{
    ComponentAnalysis, find_connected_components, keep_largest_component, remove_small_components,
    split_into_components,
};
pub use decimate::{DecimateParams, DecimateResult, decimate_mesh, decimate_mesh_with_progress};
pub use holes::{BoundaryLoop, detect_holes, fill_holes, fill_holes_with_max_edges};
pub use intersect::{IntersectionParams, SelfIntersectionResult, detect_self_intersections};
#[cfg(feature = "pipeline-config")]
pub use pipeline::{PipelineConfig, PipelineConfigError, PipelineStep};
pub use remesh::{
    CurvatureResult, FeatureEdge, FeatureEdgeResult, RemeshParams, RemeshResult, VertexCurvature,
    compute_curvature, detect_feature_edges, remesh_adaptive, remesh_anisotropic, remesh_isotropic,
    remesh_isotropic_with_progress,
};
pub use subdivide::{SubdivideParams, SubdivideResult, subdivide_mesh};
pub use thickness::{ThicknessParams, ThicknessResult, ThinRegion, analyze_thickness};
pub use validate::{
    DataValidationResult, MeshReport, ValidationOptions, validate_mesh, validate_mesh_data,
    validate_mesh_data_strict,
};
pub use winding::fix_winding_order;

// Re-export morphing and registration types
pub use morph::{Constraint, MorphAlgorithm, MorphParams, MorphResult, RbfKernel, morph_mesh};
pub use registration::{
    Landmark, NonRigidParams, NonRigidRegistrationResult, RegistrationAlgorithm,
    RegistrationParams, RegistrationResult, RigidTransform, align_meshes, non_rigid_align,
};
pub use template::{
    ControlRegion, FitParams, FitResult, FitStage, FitTemplate, Measurement, MeasurementType,
    RegionDefinition,
};

// Re-export region types for variable thickness and material zones
pub use region::{
    FloodFillCriteria, MaterialProperties, MaterialZone, MeshRegion, RegionMap, RegionSelector,
    ThicknessMap,
};

// Re-export assembly types for multi-part management
pub use assembly::{
    Assembly, AssemblyExportFormat, AssemblyValidation, BillOfMaterials, BomItem, ClearanceResult,
    Connection, ConnectionParams, ConnectionType, InterferenceResult, Part,
};

// Re-export lattice types for infill generation
pub use lattice::{
    DensityMap, InfillParams, InfillResult, LatticeParams, LatticeResult, LatticeType,
    generate_infill, generate_lattice,
};

// Re-export boolean types for CSG operations
pub use boolean::{
    BooleanOp, BooleanParams, BooleanResult, BooleanStats, CoplanarStrategy, boolean_operation,
    boolean_operation_with_progress,
};

// Re-export scan processing types
pub use scan::{
    DenoiseMethod, DenoiseParams, DenoiseResult, HoleFillParams, HoleFillResult, HoleFillStrategy,
    OutlierRemovalParams, ScanCleanupParams, ScanCleanupResult, cleanup_scan, denoise_mesh,
    fill_holes_advanced, remove_outliers,
};

// Re-export multi-scan alignment and merging types
pub use multiscan::{
    MergeParams, MergeResult, MultiAlignmentParams, MultiAlignmentResult, OverlapHandling,
    OverlapRegion, align_multiple_scans, align_multiple_scans_with_params, merge_scans,
};

// Re-export printability/manufacturing types
pub use printability::{
    IssueSeverity as PrintIssueSeverity, OrientParams, OrientResult, OverhangRegion, PrintIssue,
    PrintIssueType, PrintTechnology, PrintValidation, PrinterConfig, SupportAnalysis,
    SupportRegion, ThinWallRegion, auto_orient_for_printing, detect_support_regions,
    validate_for_printing,
};

// Re-export measurement types
pub use measure::{
    CrossSection, Dimensions, DistanceMeasurement, OrientedBoundingBox, circumference_at_height,
    closest_point_on_mesh, cross_section, cross_sections, dimensions, measure_distance,
    oriented_bounding_box,
};

// Re-export slicing types for 3D print preview
pub use slice::{
    Contour, FdmParams, FdmValidationResult, GapIssue, Layer, LayerBounds, LayerStats, SlaParams,
    SlaValidationResult, SliceParams, SliceResult, SmallFeatureIssue, SvgExportParams,
    ThinWallIssue, calculate_layer_stats, export_3mf_slices, export_layer_svg, export_slices_svg,
    slice_mesh, slice_preview, validate_for_fdm, validate_for_sla,
};

// Re-export point cloud types for scanner data processing
pub use pointcloud::{
    CloudPoint, PointCloud, PointCloudFormat, ReconstructionAlgorithm, ReconstructionParams,
    ReconstructionResult,
};

// Re-export progress tracking types for long-running operations
pub use progress::{
    OperationEstimate, OperationType, Progress, ProgressCallback, ProgressReporter,
    ProgressTracker, SharedProgressTracker, estimate_operation_time,
};

// Re-export tracing extensions for structured logging and performance monitoring
pub use tracing_ext::{
    OperationTimer, log_io_operation, log_mesh_stats, log_mesh_stats_detailed, log_perf_section,
    log_progress, log_repair_result, log_validation_result,
};

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
    pub fn decimate_with_params(
        &self,
        params: &decimate::DecimateParams,
    ) -> decimate::DecimateResult {
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
        decimate::decimate_mesh(
            self,
            &decimate::DecimateParams::with_target_triangles(target),
        )
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
    pub fn subdivide_with_params(
        &self,
        params: &subdivide::SubdivideParams,
    ) -> subdivide::SubdivideResult {
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
        subdivide::subdivide_mesh(
            self,
            &subdivide::SubdivideParams::with_iterations(iterations),
        )
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
        remesh::remesh_isotropic(
            self,
            &remesh::RemeshParams::with_target_edge_length(target_edge_length),
        )
    }

    /// Remesh with curvature-adaptive edge lengths.
    ///
    /// Creates smaller triangles in high-curvature regions and larger triangles
    /// in flat regions.
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
    /// let result = mesh.remesh_adaptive(2.0);
    /// println!("Adaptive remeshing: {} triangles", result.final_triangles);
    /// ```
    pub fn remesh_adaptive(&self, target_edge_length: f64) -> remesh::RemeshResult {
        remesh::remesh_adaptive(self, &remesh::RemeshParams::adaptive(target_edge_length))
    }

    /// Remesh with anisotropic triangles aligned to surface curvature.
    ///
    /// Creates elongated triangles that follow principal curvature directions,
    /// useful for cylindrical or ridge-like surfaces.
    ///
    /// # Arguments
    /// * `target_edge_length` - Base target edge length
    /// * `anisotropy_ratio` - Ratio of max to min edge length (e.g., 2.0 for 2:1)
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
    /// let result = mesh.remesh_anisotropic(2.0, 3.0);
    /// println!("Anisotropic remeshing: {} triangles", result.final_triangles);
    /// ```
    pub fn remesh_anisotropic(
        &self,
        target_edge_length: f64,
        anisotropy_ratio: f64,
    ) -> remesh::RemeshResult {
        remesh::remesh_anisotropic(
            self,
            &remesh::RemeshParams::anisotropic_with_ratio(target_edge_length, anisotropy_ratio),
        )
    }

    /// Detect feature edges (sharp edges and boundaries) in the mesh.
    ///
    /// # Arguments
    /// * `angle_threshold` - Dihedral angle threshold in radians (e.g., PI/3 for 60 degrees)
    ///
    /// # Example
    ///
    /// ```
    /// use mesh_repair::{Mesh, Vertex};
    /// use std::f64::consts::PI;
    ///
    /// let mut mesh = Mesh::new();
    /// mesh.vertices.push(Vertex::from_coords(0.0, 0.0, 0.0));
    /// mesh.vertices.push(Vertex::from_coords(1.0, 0.0, 0.0));
    /// mesh.vertices.push(Vertex::from_coords(0.5, 1.0, 0.0));
    /// mesh.faces.push([0, 1, 2]);
    ///
    /// let result = mesh.detect_feature_edges(PI / 3.0);
    /// println!("Found {} boundary edges", result.boundary_edges.len());
    /// ```
    pub fn detect_feature_edges(&self, angle_threshold: f64) -> remesh::FeatureEdgeResult {
        remesh::detect_feature_edges(self, angle_threshold)
    }

    /// Compute per-vertex curvature information.
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
    /// let curvature = mesh.compute_curvature();
    /// println!("Mean curvature range: {} to {}", curvature.min_mean_curvature, curvature.max_mean_curvature);
    /// ```
    pub fn compute_curvature(&self) -> remesh::CurvatureResult {
        remesh::compute_curvature(self)
    }

    /// Morph the mesh using RBF with the given constraints.
    ///
    /// This is a convenience method for simple morphing operations.
    /// For more control, use `morph_with_params`.
    ///
    /// # Example
    ///
    /// ```
    /// use mesh_repair::{Mesh, Vertex, Constraint};
    /// use nalgebra::{Point3, Vector3};
    ///
    /// let mut mesh = Mesh::new();
    /// mesh.vertices.push(Vertex::from_coords(0.0, 0.0, 0.0));
    /// mesh.vertices.push(Vertex::from_coords(10.0, 0.0, 0.0));
    /// mesh.vertices.push(Vertex::from_coords(5.0, 8.66, 0.0));
    /// mesh.vertices.push(Vertex::from_coords(5.0, 2.89, 8.16));
    /// mesh.faces.push([0, 2, 1]);
    /// mesh.faces.push([0, 1, 3]);
    /// mesh.faces.push([1, 2, 3]);
    /// mesh.faces.push([2, 0, 3]);
    ///
    /// let constraints = vec![
    ///     Constraint::displacement(Point3::new(5.0, 2.89, 8.16), Vector3::new(0.0, 0.0, 2.0)),
    /// ];
    /// let result = mesh.morph(&constraints).unwrap();
    /// ```
    pub fn morph(&self, constraints: &[morph::Constraint]) -> MeshResult<morph::MorphResult> {
        let params = morph::MorphParams::rbf().with_constraints(constraints.to_vec());
        morph::morph_mesh(self, &params)
    }

    /// Morph the mesh with custom parameters.
    ///
    /// # Example
    ///
    /// ```
    /// use mesh_repair::{Mesh, Vertex, MorphParams, Constraint};
    /// use nalgebra::Point3;
    ///
    /// let mut mesh = Mesh::new();
    /// mesh.vertices.push(Vertex::from_coords(0.0, 0.0, 0.0));
    /// mesh.vertices.push(Vertex::from_coords(10.0, 0.0, 0.0));
    /// mesh.vertices.push(Vertex::from_coords(5.0, 8.66, 0.0));
    /// mesh.vertices.push(Vertex::from_coords(5.0, 2.89, 8.16));
    /// mesh.faces.push([0, 2, 1]);
    /// mesh.faces.push([0, 1, 3]);
    /// mesh.faces.push([1, 2, 3]);
    /// mesh.faces.push([2, 0, 3]);
    ///
    /// let constraints = vec![
    ///     Constraint::point(Point3::new(5.0, 2.89, 8.16), Point3::new(5.0, 2.89, 10.0)),
    /// ];
    /// let params = MorphParams::ffd().with_constraints(constraints);
    /// let result = mesh.morph_with_params(&params).unwrap();
    /// ```
    pub fn morph_with_params(&self, params: &morph::MorphParams) -> MeshResult<morph::MorphResult> {
        morph::morph_mesh(self, params)
    }

    /// Align this mesh to a target mesh using ICP.
    ///
    /// # Example
    ///
    /// ```
    /// use mesh_repair::{Mesh, Vertex};
    ///
    /// let mut source = Mesh::new();
    /// source.vertices.push(Vertex::from_coords(0.0, 0.0, 0.0));
    /// source.vertices.push(Vertex::from_coords(1.0, 0.0, 0.0));
    /// source.vertices.push(Vertex::from_coords(0.5, 1.0, 0.0));
    /// source.faces.push([0, 1, 2]);
    ///
    /// let target = source.clone();
    /// let result = source.align_to(&target).unwrap();
    /// assert!(result.converged);
    /// ```
    pub fn align_to(&self, target: &Mesh) -> MeshResult<registration::RegistrationResult> {
        registration::align_meshes(self, target, &registration::RegistrationParams::icp())
    }

    /// Align this mesh to a target mesh with custom parameters.
    pub fn align_to_with_params(
        &self,
        target: &Mesh,
        params: &registration::RegistrationParams,
    ) -> MeshResult<registration::RegistrationResult> {
        registration::align_meshes(self, target, params)
    }

    /// Register this mesh to a target using landmarks.
    ///
    /// # Example
    ///
    /// ```
    /// use mesh_repair::{Mesh, Vertex, Landmark};
    /// use nalgebra::Point3;
    ///
    /// let mut source = Mesh::new();
    /// source.vertices.push(Vertex::from_coords(0.0, 0.0, 0.0));
    /// source.vertices.push(Vertex::from_coords(1.0, 0.0, 0.0));
    /// source.vertices.push(Vertex::from_coords(0.5, 1.0, 0.0));
    /// source.faces.push([0, 1, 2]);
    ///
    /// let target = source.clone();
    /// let landmarks = vec![
    ///     Landmark::new(Point3::new(0.0, 0.0, 0.0), Point3::new(0.0, 0.0, 0.0)),
    ///     Landmark::new(Point3::new(1.0, 0.0, 0.0), Point3::new(1.0, 0.0, 0.0)),
    ///     Landmark::new(Point3::new(0.5, 1.0, 0.0), Point3::new(0.5, 1.0, 0.0)),
    /// ];
    /// let result = source.register_to(&target, &landmarks).unwrap();
    /// ```
    pub fn register_to(
        &self,
        target: &Mesh,
        landmarks: &[registration::Landmark],
    ) -> MeshResult<registration::RegistrationResult> {
        let params = registration::RegistrationParams::landmark_based(landmarks.to_vec());
        registration::align_meshes(self, target, &params)
    }

    /// Define a region on this mesh using a selector.
    ///
    /// Returns a `MeshRegion` containing the selected vertices and faces.
    ///
    /// # Example
    ///
    /// ```
    /// use mesh_repair::{Mesh, Vertex, RegionSelector};
    /// use nalgebra::Point3;
    ///
    /// let mut mesh = Mesh::new();
    /// mesh.vertices.push(Vertex::from_coords(0.0, 0.0, 0.0));
    /// mesh.vertices.push(Vertex::from_coords(10.0, 0.0, 0.0));
    /// mesh.vertices.push(Vertex::from_coords(5.0, 10.0, 0.0));
    /// mesh.faces.push([0, 1, 2]);
    ///
    /// // Define a region for the lower half of the mesh
    /// let lower_region = mesh.define_region("lower", RegionSelector::bounds(
    ///     Point3::new(-1.0, -1.0, -1.0),
    ///     Point3::new(11.0, 5.0, 1.0),
    /// ));
    /// ```
    pub fn define_region(
        &self,
        name: impl Into<String>,
        selector: region::RegionSelector,
    ) -> region::MeshRegion {
        region::MeshRegion::from_selector(self, name, selector)
    }

    /// Create a `RegionMap` for this mesh.
    ///
    /// This is a convenience method to start defining multiple regions.
    pub fn create_region_map(&self) -> region::RegionMap {
        region::RegionMap::new()
    }

    /// Create a `ThicknessMap` for this mesh.
    ///
    /// The thickness map starts with a uniform default thickness.
    ///
    /// # Example
    ///
    /// ```
    /// use mesh_repair::{Mesh, Vertex, RegionSelector};
    /// use nalgebra::Point3;
    ///
    /// let mut mesh = Mesh::new();
    /// mesh.vertices.push(Vertex::from_coords(0.0, 0.0, 0.0));
    /// mesh.vertices.push(Vertex::from_coords(10.0, 0.0, 0.0));
    /// mesh.vertices.push(Vertex::from_coords(5.0, 10.0, 0.0));
    /// mesh.faces.push([0, 1, 2]);
    ///
    /// // Create a thickness map with 2mm default
    /// let mut thickness = mesh.create_thickness_map(2.0);
    ///
    /// // Set different thickness for a region
    /// let top_region = mesh.define_region("top", RegionSelector::bounds(
    ///     Point3::new(-1.0, 5.0, -1.0),
    ///     Point3::new(11.0, 11.0, 1.0),
    /// ));
    /// thickness.set_region_thickness(&top_region, 3.5);
    /// ```
    pub fn create_thickness_map(&self, default_thickness: f64) -> region::ThicknessMap {
        region::ThicknessMap::new(default_thickness)
    }

    /// Export the mesh to STEP format for CAD interchange.
    ///
    /// STEP (ISO 10303) is a widely-supported format for exchanging geometry
    /// with CAD systems. Triangle meshes are exported as faceted B-rep geometry.
    ///
    /// # Example
    ///
    /// ```no_run
    /// # #[cfg(feature = "step")]
    /// # fn main() -> mesh_repair::MeshResult<()> {
    /// use mesh_repair::Mesh;
    ///
    /// let mesh = Mesh::load("model.stl")?;
    /// mesh.save_step("model.step")?;
    /// # Ok(())
    /// # }
    /// # #[cfg(not(feature = "step"))]
    /// # fn main() {}
    /// ```
    #[cfg(feature = "step")]
    pub fn save_step(
        &self,
        path: impl AsRef<std::path::Path>,
    ) -> MeshResult<step::StepExportResult> {
        step::export_step(self, path, &step::StepExportParams::default())
    }

    /// Export the mesh to STEP format with custom parameters.
    ///
    /// # Example
    ///
    /// ```no_run
    /// # #[cfg(feature = "step")]
    /// # fn main() -> mesh_repair::MeshResult<()> {
    /// use mesh_repair::{Mesh, StepExportParams};
    ///
    /// let mesh = Mesh::load("model.stl")?;
    /// let params = StepExportParams::default()
    ///     .with_description("My CAD model")
    ///     .with_author("Jane Doe", "ACME Corp");
    /// mesh.save_step_with_params("model.step", &params)?;
    /// # Ok(())
    /// # }
    /// # #[cfg(not(feature = "step"))]
    /// # fn main() {}
    /// ```
    #[cfg(feature = "step")]
    pub fn save_step_with_params(
        &self,
        path: impl AsRef<std::path::Path>,
        params: &step::StepExportParams,
    ) -> MeshResult<step::StepExportResult> {
        step::export_step(self, path, params)
    }
}

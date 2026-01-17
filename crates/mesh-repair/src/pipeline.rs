//! Pipeline API for chaining mesh operations.
//!
//! This module provides a fluent API for building mesh processing pipelines
//! that can chain multiple operations together. Pipelines support:
//!
//! - Loading from various formats
//! - Repair operations
//! - Transformations (remesh, decimate, subdivide)
//! - Analysis (validation, thickness)
//! - Export to various formats
//!
//! # Example
//!
//! ```no_run
//! use mesh_repair::Pipeline;
//!
//! // Complete pipeline from load to save
//! let result = Pipeline::load("scan.stl")
//!     .unwrap()
//!     .repair_for_printing()
//!     .remesh(2.0)
//!     .validate()
//!     .save("processed.3mf")
//!     .unwrap();
//!
//! println!("Processed mesh with {} triangles", result.mesh.faces.len());
//! ```

use crate::Mesh;
use crate::decimate::{DecimateParams, decimate_mesh};
use crate::error::MeshResult;
use crate::progress::ProgressCallback;
#[cfg(feature = "pipeline-config")]
use crate::remesh::RemeshParams;
use crate::remesh::RemeshResult;
use crate::repair::{RepairParams, compute_vertex_normals, repair_mesh_with_config};
use crate::subdivide::{SubdivideParams, subdivide_mesh};
use crate::validate::MeshReport;

// =========================================================================
// Pipeline Configuration (Serialization)
// =========================================================================

/// A single step in a pipeline configuration.
///
/// Each step represents one operation to be performed on the mesh.
#[cfg(feature = "pipeline-config")]
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
#[serde(tag = "operation", rename_all = "snake_case")]
pub enum PipelineStep {
    /// Apply default repair.
    Repair,
    /// Apply repair optimized for 3D scans.
    RepairForScans,
    /// Apply repair optimized for 3D printing.
    RepairForPrinting,
    /// Apply repair optimized for CAD models.
    RepairForCad,
    /// Apply repair with custom parameters.
    RepairWithParams {
        #[serde(flatten)]
        params: RepairParams,
    },
    /// Fill holes up to a maximum edge count.
    FillHoles { max_edges: usize },
    /// Fix winding order for consistent normals.
    FixWinding,
    /// Remove small disconnected components.
    RemoveSmallComponents { min_faces: usize },
    /// Remesh with uniform edge length.
    Remesh { target_edge_length: f64 },
    /// Remesh with custom parameters.
    RemeshWithParams {
        #[serde(flatten)]
        params: RemeshParams,
    },
    /// Decimate to target triangle count.
    DecimateToCount { target_count: usize },
    /// Decimate to ratio of original triangles.
    DecimateToRatio { ratio: f64 },
    /// Decimate with custom parameters.
    DecimateWithParams {
        #[serde(flatten)]
        params: DecimateParams,
    },
    /// Subdivide the mesh.
    Subdivide { iterations: usize },
    /// Subdivide with custom parameters.
    SubdivideWithParams {
        #[serde(flatten)]
        params: SubdivideParams,
    },
    /// Compute vertex normals.
    ComputeNormals,
    /// Validate the mesh.
    Validate,
    /// Require mesh to be printable (watertight + manifold).
    RequirePrintable,
}

/// A serializable pipeline configuration.
///
/// `PipelineConfig` allows pipeline workflows to be saved and loaded as
/// TOML or JSON files. This enables:
///
/// - Reproducible processing workflows
/// - User-editable configuration files
/// - Batch processing with the same settings
///
/// # Example TOML
///
/// ```toml
/// name = "scan-to-print"
/// description = "Prepare 3D scan for printing"
///
/// [[steps]]
/// operation = "repair_for_scans"
///
/// [[steps]]
/// operation = "remesh"
/// target_edge_length = 2.0
///
/// [[steps]]
/// operation = "decimate_to_ratio"
/// ratio = 0.5
///
/// [[steps]]
/// operation = "validate"
/// ```
///
/// # Example
///
/// ```ignore
/// use mesh_repair::{Pipeline, PipelineConfig};
///
/// // Load config from file
/// let config = PipelineConfig::from_toml_file("workflow.toml")?;
///
/// // Run the pipeline
/// let result = Pipeline::load("model.stl")?
///     .run_config(&config)
///     .save("output.stl")?;
/// ```
#[cfg(feature = "pipeline-config")]
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct PipelineConfig {
    /// Optional name for this workflow.
    #[serde(default)]
    pub name: Option<String>,
    /// Optional description of what this workflow does.
    #[serde(default)]
    pub description: Option<String>,
    /// The sequence of operations to perform.
    pub steps: Vec<PipelineStep>,
}

#[cfg(feature = "pipeline-config")]
impl PipelineConfig {
    /// Create an empty pipeline configuration.
    pub fn new() -> Self {
        Self {
            name: None,
            description: None,
            steps: Vec::new(),
        }
    }

    /// Create a configuration with a name.
    pub fn with_name(name: impl Into<String>) -> Self {
        Self {
            name: Some(name.into()),
            description: None,
            steps: Vec::new(),
        }
    }

    /// Add a step to the configuration.
    pub fn add_step(mut self, step: PipelineStep) -> Self {
        self.steps.push(step);
        self
    }

    /// Set the description.
    pub fn description(mut self, desc: impl Into<String>) -> Self {
        self.description = Some(desc.into());
        self
    }

    /// Load configuration from a TOML string.
    ///
    /// # Errors
    ///
    /// Returns an error if the TOML is invalid or doesn't match the schema.
    pub fn from_toml(toml_str: &str) -> Result<Self, toml::de::Error> {
        toml::from_str(toml_str)
    }

    /// Load configuration from a TOML file.
    ///
    /// # Errors
    ///
    /// Returns an error if the file can't be read or the TOML is invalid.
    pub fn from_toml_file(path: impl AsRef<std::path::Path>) -> Result<Self, PipelineConfigError> {
        let contents = std::fs::read_to_string(path.as_ref())?;
        Ok(toml::from_str(&contents)?)
    }

    /// Serialize to TOML string.
    pub fn to_toml(&self) -> Result<String, toml::ser::Error> {
        toml::to_string_pretty(self)
    }

    /// Save configuration to a TOML file.
    pub fn save_toml(&self, path: impl AsRef<std::path::Path>) -> Result<(), PipelineConfigError> {
        let toml_str = self.to_toml()?;
        std::fs::write(path, toml_str)?;
        Ok(())
    }

    /// Load configuration from a JSON string.
    #[cfg(feature = "pipeline-config")]
    pub fn from_json(json_str: &str) -> Result<Self, serde_json::Error> {
        serde_json::from_str(json_str)
    }

    /// Serialize to JSON string.
    #[cfg(feature = "pipeline-config")]
    pub fn to_json(&self) -> Result<String, serde_json::Error> {
        serde_json::to_string_pretty(self)
    }

    /// Create a preset configuration for preparing scans for 3D printing.
    pub fn preset_scan_to_print() -> Self {
        Self::with_name("scan-to-print")
            .description("Prepare 3D scan data for 3D printing")
            .add_step(PipelineStep::RepairForScans)
            .add_step(PipelineStep::Remesh {
                target_edge_length: 2.0,
            })
            .add_step(PipelineStep::Validate)
    }

    /// Create a preset configuration for mesh simplification.
    pub fn preset_simplify(ratio: f64) -> Self {
        Self::with_name("simplify")
            .description("Simplify mesh while preserving shape")
            .add_step(PipelineStep::Repair)
            .add_step(PipelineStep::DecimateToRatio { ratio })
            .add_step(PipelineStep::ComputeNormals)
    }

    /// Create a preset configuration for mesh refinement.
    pub fn preset_refine(iterations: usize, target_edge_length: f64) -> Self {
        Self::with_name("refine")
            .description("Refine mesh with subdivision and remeshing")
            .add_step(PipelineStep::Repair)
            .add_step(PipelineStep::Subdivide { iterations })
            .add_step(PipelineStep::Remesh { target_edge_length })
            .add_step(PipelineStep::ComputeNormals)
    }
}

#[cfg(feature = "pipeline-config")]
impl Default for PipelineConfig {
    fn default() -> Self {
        Self::new()
    }
}

/// Errors that can occur when loading or saving pipeline configurations.
#[cfg(feature = "pipeline-config")]
#[derive(Debug)]
pub enum PipelineConfigError {
    /// I/O error reading or writing file.
    Io(std::io::Error),
    /// TOML parsing error.
    TomlParse(toml::de::Error),
    /// TOML serialization error.
    TomlSerialize(toml::ser::Error),
}

#[cfg(feature = "pipeline-config")]
impl std::fmt::Display for PipelineConfigError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Io(e) => write!(f, "I/O error: {}", e),
            Self::TomlParse(e) => write!(f, "TOML parse error: {}", e),
            Self::TomlSerialize(e) => write!(f, "TOML serialize error: {}", e),
        }
    }
}

#[cfg(feature = "pipeline-config")]
impl std::error::Error for PipelineConfigError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Io(e) => Some(e),
            Self::TomlParse(e) => Some(e),
            Self::TomlSerialize(e) => Some(e),
        }
    }
}

#[cfg(feature = "pipeline-config")]
impl From<std::io::Error> for PipelineConfigError {
    fn from(e: std::io::Error) -> Self {
        Self::Io(e)
    }
}

#[cfg(feature = "pipeline-config")]
impl From<toml::de::Error> for PipelineConfigError {
    fn from(e: toml::de::Error) -> Self {
        Self::TomlParse(e)
    }
}

#[cfg(feature = "pipeline-config")]
impl From<toml::ser::Error> for PipelineConfigError {
    fn from(e: toml::ser::Error) -> Self {
        Self::TomlSerialize(e)
    }
}

/// Result of a pipeline execution.
#[derive(Debug)]
pub struct PipelineResult {
    /// The processed mesh.
    pub mesh: Mesh,
    /// Validation report (if validation was performed).
    pub validation: Option<MeshReport>,
    /// Number of pipeline stages executed.
    pub stages_executed: usize,
    /// Log of operations performed.
    pub operation_log: Vec<String>,
}

/// A mesh processing pipeline.
///
/// Pipeline provides a fluent API for chaining mesh operations.
/// Each operation returns the pipeline, allowing method chaining.
///
/// # Example
///
/// ```no_run
/// use mesh_repair::Pipeline;
///
/// let result = Pipeline::load("model.stl")
///     .unwrap()
///     .repair_for_scans()
///     .decimate_to_ratio(0.5)
///     .save("reduced.obj")
///     .unwrap();
/// ```
pub struct Pipeline {
    mesh: Mesh,
    validation: Option<MeshReport>,
    stages_executed: usize,
    operation_log: Vec<String>,
    progress_callback: Option<ProgressCallback>,
}

impl Pipeline {
    /// Start a pipeline with an existing mesh.
    ///
    /// # Arguments
    ///
    /// * `mesh` - The mesh to process (takes ownership)
    pub fn new(mesh: Mesh) -> Self {
        Self {
            mesh,
            validation: None,
            stages_executed: 0,
            operation_log: Vec::new(),
            progress_callback: None,
        }
    }

    /// Start a pipeline by loading a mesh from a file.
    ///
    /// Supports STL, OBJ, 3MF, and PLY formats based on file extension.
    ///
    /// # Arguments
    ///
    /// * `path` - Path to the mesh file
    ///
    /// # Example
    ///
    /// ```no_run
    /// use mesh_repair::Pipeline;
    ///
    /// let pipeline = Pipeline::load("model.stl").unwrap();
    /// ```
    pub fn load(path: impl AsRef<std::path::Path>) -> MeshResult<Self> {
        let path = path.as_ref();
        let mesh = Mesh::load(path)?;
        let mut pipeline = Self::new(mesh);
        pipeline.log(format!("Loaded mesh from {}", path.display()));
        Ok(pipeline)
    }

    /// Set a progress callback for long-running operations.
    pub fn with_progress(mut self, callback: ProgressCallback) -> Self {
        self.progress_callback = Some(callback);
        self
    }

    // =========================================================================
    // Repair Operations
    // =========================================================================

    /// Apply default repair operations.
    ///
    /// This is a general-purpose repair that handles common issues.
    pub fn repair(mut self) -> Self {
        if let Err(e) = self.mesh.repair() {
            self.log(format!("Repair warning: {}", e));
        } else {
            self.log("Applied default repair".to_string());
        }
        self.stages_executed += 1;
        self
    }

    /// Apply repair optimized for 3D scan data.
    ///
    /// Uses aggressive settings suitable for noisy scan data.
    pub fn repair_for_scans(mut self) -> Self {
        let params = RepairParams::for_scans();
        if let Err(e) = repair_mesh_with_config(&mut self.mesh, &params) {
            self.log(format!("Scan repair warning: {}", e));
        } else {
            self.log("Applied scan-optimized repair".to_string());
        }
        self.stages_executed += 1;
        self
    }

    /// Apply repair optimized for 3D printing.
    ///
    /// Ensures watertight, manifold output.
    pub fn repair_for_printing(mut self) -> Self {
        let params = RepairParams::for_printing();
        if let Err(e) = repair_mesh_with_config(&mut self.mesh, &params) {
            self.log(format!("Print repair warning: {}", e));
        } else {
            self.log("Applied printing-optimized repair".to_string());
        }
        self.stages_executed += 1;
        self
    }

    /// Apply repair optimized for CAD models.
    ///
    /// Uses conservative settings to preserve intentional geometry.
    pub fn repair_for_cad(mut self) -> Self {
        let params = RepairParams::for_cad();
        if let Err(e) = repair_mesh_with_config(&mut self.mesh, &params) {
            self.log(format!("CAD repair warning: {}", e));
        } else {
            self.log("Applied CAD-optimized repair".to_string());
        }
        self.stages_executed += 1;
        self
    }

    /// Apply repair with custom parameters.
    pub fn repair_with_params(mut self, params: &RepairParams) -> Self {
        if let Err(e) = repair_mesh_with_config(&mut self.mesh, params) {
            self.log(format!("Custom repair warning: {}", e));
        } else {
            self.log("Applied custom repair".to_string());
        }
        self.stages_executed += 1;
        self
    }

    /// Fill holes in the mesh.
    ///
    /// # Arguments
    ///
    /// * `max_edges` - Maximum number of edges in a hole to fill
    pub fn fill_holes(mut self, max_edges: usize) -> Self {
        match crate::holes::fill_holes_with_max_edges(&mut self.mesh, max_edges) {
            Ok(count) => self.log(format!("Filled {} holes (max {} edges)", count, max_edges)),
            Err(e) => self.log(format!("Hole filling warning: {}", e)),
        }
        self.stages_executed += 1;
        self
    }

    /// Fix winding order to make normals consistent.
    pub fn fix_winding(mut self) -> Self {
        if let Err(e) = crate::winding::fix_winding_order(&mut self.mesh) {
            self.log(format!("Winding fix warning: {}", e));
        } else {
            self.log("Fixed winding order".to_string());
        }
        self.stages_executed += 1;
        self
    }

    /// Remove small disconnected components.
    ///
    /// # Arguments
    ///
    /// * `min_faces` - Minimum faces for a component to keep
    pub fn remove_small_components(mut self, min_faces: usize) -> Self {
        let removed = crate::components::remove_small_components(&mut self.mesh, min_faces);
        self.log(format!(
            "Removed {} small components (< {} faces)",
            removed, min_faces
        ));
        self.stages_executed += 1;
        self
    }

    // =========================================================================
    // Transformation Operations
    // =========================================================================

    /// Remesh with uniform edge length.
    ///
    /// # Arguments
    ///
    /// * `target_edge_length` - Target edge length in mm
    pub fn remesh(mut self, target_edge_length: f64) -> Self {
        let result = self.mesh.remesh_with_edge_length(target_edge_length);
        self.mesh = result.mesh;
        self.log(format!(
            "Remeshed to {:.2}mm edge length ({} triangles)",
            target_edge_length,
            self.mesh.faces.len()
        ));
        self.stages_executed += 1;
        self
    }

    /// Remesh with detailed parameters.
    pub fn remesh_with_params(mut self, params: &crate::remesh::RemeshParams) -> Self {
        let result = self.mesh.remesh_with_params(params);
        self.mesh = result.mesh;
        self.log(format!(
            "Remeshed with custom params ({} triangles)",
            self.mesh.faces.len()
        ));
        self.stages_executed += 1;
        self
    }

    /// Decimate to a target triangle count.
    ///
    /// # Arguments
    ///
    /// * `target_count` - Target number of triangles
    pub fn decimate_to_count(mut self, target_count: usize) -> Self {
        let params = DecimateParams::with_target_triangles(target_count);
        let result = decimate_mesh(&self.mesh, &params);
        let original = self.mesh.faces.len();
        self.mesh = result.mesh;
        self.log(format!(
            "Decimated from {} to {} triangles",
            original,
            self.mesh.faces.len()
        ));
        self.stages_executed += 1;
        self
    }

    /// Decimate to a ratio of original triangles.
    ///
    /// # Arguments
    ///
    /// * `ratio` - Target ratio (0.0 to 1.0), e.g., 0.5 for 50%
    pub fn decimate_to_ratio(mut self, ratio: f64) -> Self {
        let params = DecimateParams::with_target_ratio(ratio);
        let result = decimate_mesh(&self.mesh, &params);
        let original = self.mesh.faces.len();
        self.mesh = result.mesh;
        self.log(format!(
            "Decimated from {} to {} triangles ({:.0}%)",
            original,
            self.mesh.faces.len(),
            ratio * 100.0
        ));
        self.stages_executed += 1;
        self
    }

    /// Decimate with custom parameters.
    pub fn decimate_with_params(mut self, params: &DecimateParams) -> Self {
        let result = decimate_mesh(&self.mesh, params);
        let original = self.mesh.faces.len();
        self.mesh = result.mesh;
        self.log(format!(
            "Decimated from {} to {} triangles (custom params)",
            original,
            self.mesh.faces.len()
        ));
        self.stages_executed += 1;
        self
    }

    /// Subdivide the mesh.
    ///
    /// # Arguments
    ///
    /// * `iterations` - Number of subdivision iterations
    pub fn subdivide(mut self, iterations: usize) -> Self {
        let params = SubdivideParams::with_iterations(iterations);
        let result = subdivide_mesh(&self.mesh, &params);
        self.mesh = result.mesh;
        self.log(format!(
            "Subdivided {} times ({} triangles)",
            iterations,
            self.mesh.faces.len()
        ));
        self.stages_executed += 1;
        self
    }

    /// Subdivide with custom parameters.
    pub fn subdivide_with_params(mut self, params: &SubdivideParams) -> Self {
        let result = subdivide_mesh(&self.mesh, params);
        self.mesh = result.mesh;
        self.log(format!(
            "Subdivided with custom params ({} triangles)",
            self.mesh.faces.len()
        ));
        self.stages_executed += 1;
        self
    }

    /// Compute or recompute vertex normals.
    pub fn compute_normals(mut self) -> Self {
        compute_vertex_normals(&mut self.mesh);
        self.log("Computed vertex normals".to_string());
        self.stages_executed += 1;
        self
    }

    // =========================================================================
    // Analysis Operations
    // =========================================================================

    /// Validate the mesh and store the report.
    ///
    /// The validation report is stored and can be retrieved from the result.
    pub fn validate(mut self) -> Self {
        let report = self.mesh.validate();
        self.log(format!(
            "Validated: {} triangles, watertight={}, manifold={}",
            self.mesh.faces.len(),
            report.is_watertight,
            report.is_manifold
        ));
        self.validation = Some(report);
        self.stages_executed += 1;
        self
    }

    /// Check if mesh is printable and return early if not.
    ///
    /// This validates the mesh and returns an error if it's not suitable
    /// for 3D printing (not watertight or not manifold).
    pub fn require_printable(mut self) -> MeshResult<Self> {
        let report = self.mesh.validate();
        if !report.is_printable() {
            return Err(crate::error::MeshError::invalid_topology(format!(
                "Mesh not printable: watertight={}, manifold={}",
                report.is_watertight, report.is_manifold
            )));
        }
        self.log("Verified mesh is printable".to_string());
        self.validation = Some(report);
        self.stages_executed += 1;
        Ok(self)
    }

    // =========================================================================
    // Configuration-Based Execution
    // =========================================================================

    /// Run a pipeline from a configuration.
    ///
    /// Executes all steps defined in the configuration in order.
    /// This allows running pre-defined workflows loaded from TOML or JSON files.
    ///
    /// # Arguments
    ///
    /// * `config` - The pipeline configuration to execute
    ///
    /// # Example
    ///
    /// ```ignore
    /// use mesh_repair::{Pipeline, PipelineConfig};
    ///
    /// let config = PipelineConfig::from_toml_file("workflow.toml")?;
    /// let result = Pipeline::load("model.stl")?
    ///     .run_config(&config)
    ///     .save("output.stl")?;
    /// ```
    #[cfg(feature = "pipeline-config")]
    pub fn run_config(mut self, config: &PipelineConfig) -> MeshResult<Self> {
        if let Some(name) = &config.name {
            self.log(format!("Running pipeline: {}", name));
        }

        for step in &config.steps {
            self = self.run_step(step)?;
        }

        Ok(self)
    }

    /// Execute a single pipeline step.
    #[cfg(feature = "pipeline-config")]
    fn run_step(self, step: &PipelineStep) -> MeshResult<Self> {
        match step {
            PipelineStep::Repair => Ok(self.repair()),
            PipelineStep::RepairForScans => Ok(self.repair_for_scans()),
            PipelineStep::RepairForPrinting => Ok(self.repair_for_printing()),
            PipelineStep::RepairForCad => Ok(self.repair_for_cad()),
            PipelineStep::RepairWithParams { params } => Ok(self.repair_with_params(params)),
            PipelineStep::FillHoles { max_edges } => Ok(self.fill_holes(*max_edges)),
            PipelineStep::FixWinding => Ok(self.fix_winding()),
            PipelineStep::RemoveSmallComponents { min_faces } => {
                Ok(self.remove_small_components(*min_faces))
            }
            PipelineStep::Remesh { target_edge_length } => Ok(self.remesh(*target_edge_length)),
            PipelineStep::RemeshWithParams { params } => Ok(self.remesh_with_params(params)),
            PipelineStep::DecimateToCount { target_count } => {
                Ok(self.decimate_to_count(*target_count))
            }
            PipelineStep::DecimateToRatio { ratio } => Ok(self.decimate_to_ratio(*ratio)),
            PipelineStep::DecimateWithParams { params } => Ok(self.decimate_with_params(params)),
            PipelineStep::Subdivide { iterations } => Ok(self.subdivide(*iterations)),
            PipelineStep::SubdivideWithParams { params } => Ok(self.subdivide_with_params(params)),
            PipelineStep::ComputeNormals => Ok(self.compute_normals()),
            PipelineStep::Validate => Ok(self.validate()),
            PipelineStep::RequirePrintable => self.require_printable(),
        }
    }

    // =========================================================================
    // Output Operations
    // =========================================================================

    /// Save the mesh to a file.
    ///
    /// Format is determined by file extension.
    ///
    /// # Arguments
    ///
    /// * `path` - Output file path
    pub fn save(mut self, path: impl AsRef<std::path::Path>) -> MeshResult<PipelineResult> {
        let path = path.as_ref();
        self.mesh.save(path)?;
        self.log(format!("Saved mesh to {}", path.display()));
        self.stages_executed += 1;
        Ok(self.finish())
    }

    /// Finish the pipeline and return the result without saving.
    pub fn finish(self) -> PipelineResult {
        PipelineResult {
            mesh: self.mesh,
            validation: self.validation,
            stages_executed: self.stages_executed,
            operation_log: self.operation_log,
        }
    }

    /// Get a reference to the current mesh state.
    pub fn mesh(&self) -> &Mesh {
        &self.mesh
    }

    /// Get the current validation report (if any).
    pub fn validation_report(&self) -> Option<&MeshReport> {
        self.validation.as_ref()
    }

    /// Get the operation log.
    pub fn log_entries(&self) -> &[String] {
        &self.operation_log
    }

    /// Get the number of stages executed.
    pub fn stages_executed(&self) -> usize {
        self.stages_executed
    }

    // =========================================================================
    // Internal
    // =========================================================================

    fn log(&mut self, message: String) {
        self.operation_log.push(message);
    }
}

/// Trait for types that can be converted into a Pipeline.
pub trait IntoPipeline {
    /// Convert into a Pipeline.
    fn into_pipeline(self) -> Pipeline;
}

impl IntoPipeline for Mesh {
    fn into_pipeline(self) -> Pipeline {
        Pipeline::new(self)
    }
}

impl IntoPipeline for RemeshResult {
    fn into_pipeline(self) -> Pipeline {
        Pipeline::new(self.mesh)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::Vertex;

    fn create_test_cube() -> Mesh {
        let mut mesh = Mesh::new();

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

        mesh.faces = vec![
            [0, 2, 1],
            [0, 3, 2],
            [4, 5, 6],
            [4, 6, 7],
            [0, 1, 5],
            [0, 5, 4],
            [2, 3, 7],
            [2, 7, 6],
            [0, 4, 7],
            [0, 7, 3],
            [1, 2, 6],
            [1, 6, 5],
        ];

        mesh
    }

    #[test]
    fn test_pipeline_new() {
        let mesh = create_test_cube();
        let pipeline = Pipeline::new(mesh);

        assert_eq!(pipeline.stages_executed(), 0);
        assert_eq!(pipeline.mesh().faces.len(), 12);
    }

    #[test]
    fn test_pipeline_chaining() {
        let mesh = create_test_cube();
        let result = Pipeline::new(mesh)
            .repair()
            .compute_normals()
            .validate()
            .finish();

        assert_eq!(result.stages_executed, 3);
        assert!(result.validation.is_some());
    }

    #[test]
    fn test_pipeline_log() {
        let mesh = create_test_cube();
        let result = Pipeline::new(mesh).repair().compute_normals().finish();

        assert_eq!(result.operation_log.len(), 2);
        assert!(result.operation_log[0].contains("repair"));
        assert!(result.operation_log[1].contains("normals"));
    }

    #[test]
    fn test_into_pipeline() {
        let mesh = create_test_cube();
        let pipeline = mesh.into_pipeline();

        assert_eq!(pipeline.mesh().faces.len(), 12);
    }

    #[test]
    fn test_decimate_to_ratio() {
        let mesh = create_test_cube();
        let original_count = mesh.faces.len();
        let result = Pipeline::new(mesh).decimate_to_ratio(0.5).finish();

        assert!(result.mesh.faces.len() <= original_count);
    }
}

#[cfg(all(test, feature = "pipeline-config"))]
mod config_tests {
    use super::*;
    use crate::Vertex;

    fn create_test_cube() -> Mesh {
        let mut mesh = Mesh::new();

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

        mesh.faces = vec![
            [0, 2, 1],
            [0, 3, 2],
            [4, 5, 6],
            [4, 6, 7],
            [0, 1, 5],
            [0, 5, 4],
            [2, 3, 7],
            [2, 7, 6],
            [0, 4, 7],
            [0, 7, 3],
            [1, 2, 6],
            [1, 6, 5],
        ];

        mesh
    }

    #[test]
    fn test_pipeline_config_new() {
        let config = PipelineConfig::new();
        assert!(config.name.is_none());
        assert!(config.steps.is_empty());
    }

    #[test]
    fn test_pipeline_config_builder() {
        let config = PipelineConfig::with_name("test-pipeline")
            .description("A test pipeline")
            .add_step(PipelineStep::Repair)
            .add_step(PipelineStep::Remesh {
                target_edge_length: 2.0,
            })
            .add_step(PipelineStep::Validate);

        assert_eq!(config.name.as_deref(), Some("test-pipeline"));
        assert_eq!(config.description.as_deref(), Some("A test pipeline"));
        assert_eq!(config.steps.len(), 3);
    }

    #[test]
    fn test_pipeline_config_toml_roundtrip() {
        let config = PipelineConfig::with_name("roundtrip-test")
            .add_step(PipelineStep::RepairForScans)
            .add_step(PipelineStep::DecimateToRatio { ratio: 0.5 })
            .add_step(PipelineStep::ComputeNormals);

        let toml_str = config.to_toml().unwrap();
        let parsed = PipelineConfig::from_toml(&toml_str).unwrap();

        assert_eq!(parsed.name, config.name);
        assert_eq!(parsed.steps.len(), config.steps.len());
    }

    #[test]
    fn test_pipeline_config_json_roundtrip() {
        let config = PipelineConfig::with_name("json-test")
            .add_step(PipelineStep::Repair)
            .add_step(PipelineStep::FillHoles { max_edges: 50 });

        let json_str = config.to_json().unwrap();
        let parsed = PipelineConfig::from_json(&json_str).unwrap();

        assert_eq!(parsed.name, config.name);
        assert_eq!(parsed.steps.len(), config.steps.len());
    }

    #[test]
    fn test_pipeline_config_from_toml_string() {
        let toml = r#"
            name = "scan-workflow"
            description = "Process 3D scan data"

            [[steps]]
            operation = "repair_for_scans"

            [[steps]]
            operation = "remesh"
            target_edge_length = 1.5

            [[steps]]
            operation = "decimate_to_ratio"
            ratio = 0.75

            [[steps]]
            operation = "validate"
        "#;

        let config = PipelineConfig::from_toml(toml).unwrap();

        assert_eq!(config.name.as_deref(), Some("scan-workflow"));
        assert_eq!(config.steps.len(), 4);
    }

    #[test]
    fn test_pipeline_run_config() {
        let config = PipelineConfig::with_name("test")
            .add_step(PipelineStep::Repair)
            .add_step(PipelineStep::ComputeNormals)
            .add_step(PipelineStep::Validate);

        let mesh = create_test_cube();
        let result = Pipeline::new(mesh).run_config(&config).unwrap().finish();

        // 1 for config name log + 3 steps
        assert!(result.stages_executed >= 3);
        assert!(result.validation.is_some());
    }

    #[test]
    fn test_preset_scan_to_print() {
        let config = PipelineConfig::preset_scan_to_print();

        assert_eq!(config.name.as_deref(), Some("scan-to-print"));
        assert!(!config.steps.is_empty());

        let mesh = create_test_cube();
        let result = Pipeline::new(mesh).run_config(&config).unwrap().finish();

        assert!(result.validation.is_some());
    }

    #[test]
    fn test_preset_simplify() {
        let config = PipelineConfig::preset_simplify(0.5);

        assert_eq!(config.name.as_deref(), Some("simplify"));
        assert!(!config.steps.is_empty());

        let mesh = create_test_cube();
        let original_count = mesh.faces.len();
        let result = Pipeline::new(mesh).run_config(&config).unwrap().finish();

        assert!(result.mesh.faces.len() <= original_count);
    }

    #[test]
    fn test_repair_params_serialization() {
        let params = RepairParams::for_scans();
        let toml_str = toml::to_string_pretty(&params).unwrap();
        let parsed: RepairParams = toml::from_str(&toml_str).unwrap();

        assert!((parsed.weld_epsilon - params.weld_epsilon).abs() < 1e-9);
        assert_eq!(parsed.fill_holes, params.fill_holes);
    }

    #[test]
    fn test_decimate_params_serialization() {
        let params = DecimateParams::with_target_ratio(0.3);
        let json_str = serde_json::to_string_pretty(&params).unwrap();
        let parsed: DecimateParams = serde_json::from_str(&json_str).unwrap();

        assert!((parsed.target_ratio - params.target_ratio).abs() < 1e-9);
        assert_eq!(parsed.preserve_boundary, params.preserve_boundary);
    }

    #[test]
    fn test_remesh_params_serialization() {
        let params = RemeshParams {
            target_edge_length: Some(2.5),
            iterations: 3,
            ..Default::default()
        };
        let toml_str = toml::to_string_pretty(&params).unwrap();
        let parsed: RemeshParams = toml::from_str(&toml_str).unwrap();

        assert_eq!(parsed.target_edge_length, params.target_edge_length);
        assert_eq!(parsed.iterations, params.iterations);
        // Non-serializable fields should be None/default
        assert!(parsed.direction_field.is_none());
        assert!(parsed.preserve_feature_edges.is_none());
    }

    #[test]
    fn test_subdivide_params_serialization() {
        let params = SubdivideParams::with_iterations(2);
        let json_str = serde_json::to_string_pretty(&params).unwrap();
        let parsed: SubdivideParams = serde_json::from_str(&json_str).unwrap();

        assert_eq!(parsed.iterations, params.iterations);
    }

    #[test]
    fn test_pipeline_config_save_load_file() {
        let config = PipelineConfig::with_name("file-test")
            .add_step(PipelineStep::Repair)
            .add_step(PipelineStep::Validate);

        let temp_dir = tempfile::tempdir().unwrap();
        let path = temp_dir.path().join("pipeline.toml");

        config.save_toml(&path).unwrap();
        let loaded = PipelineConfig::from_toml_file(&path).unwrap();

        assert_eq!(loaded.name, config.name);
        assert_eq!(loaded.steps.len(), config.steps.len());
    }
}

// Allow unused_assignments lint for error struct fields that are used in thiserror Display macros
// but appear as "never read" to the compiler. This is a false positive in newer Rust versions.
#![allow(unused_assignments)]

//! Error types for shell operations with rich diagnostics.
//!
//! This module provides comprehensive error handling with:
//! - Machine-readable error codes for programmatic handling
//! - Rich context (grid dimensions, operation parameters)
//! - Recovery suggestions for common issues
//! - Beautiful terminal display via miette

use miette::Diagnostic;
use thiserror::Error;

/// Result type alias for shell operations.
pub type ShellResult<T> = Result<T, ShellError>;

/// Machine-readable error codes for shell operations.
///
/// Codes follow the pattern `SHELL-XXXX` where:
/// - 1xxx = Input validation errors
/// - 2xxx = Computation errors
/// - 3xxx = Output generation errors
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ShellErrorCode {
    /// SHELL-1001: Input mesh is empty
    EmptyMesh = 1001,
    /// SHELL-1002: Invalid parameters
    InvalidParams = 1002,

    /// SHELL-2001: SDF grid too large
    GridTooLarge = 2001,
    /// SHELL-2002: SDF computation failed
    SdfFailed = 2002,
    /// SHELL-2003: Isosurface extraction failed
    IsosurfaceFailed = 2003,

    /// SHELL-3001: Tag transfer failed
    TagTransferFailed = 3001,
    /// SHELL-3002: Shell generation failed
    ShellGenerationFailed = 3002,
    /// SHELL-3003: Rim generation failed
    RimGenerationFailed = 3003,
}

impl ShellErrorCode {
    /// Returns the error code as a string in the format `SHELL-XXXX`.
    pub fn as_str(&self) -> &'static str {
        match self {
            ShellErrorCode::EmptyMesh => "SHELL-1001",
            ShellErrorCode::InvalidParams => "SHELL-1002",
            ShellErrorCode::GridTooLarge => "SHELL-2001",
            ShellErrorCode::SdfFailed => "SHELL-2002",
            ShellErrorCode::IsosurfaceFailed => "SHELL-2003",
            ShellErrorCode::TagTransferFailed => "SHELL-3001",
            ShellErrorCode::ShellGenerationFailed => "SHELL-3002",
            ShellErrorCode::RimGenerationFailed => "SHELL-3003",
        }
    }
}

impl std::fmt::Display for ShellErrorCode {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.as_str())
    }
}

/// Recovery suggestions for shell errors.
#[derive(Debug, Clone, PartialEq)]
pub enum ShellRecoverySuggestion {
    /// Reduce grid resolution.
    ReduceGridResolution {
        current: [usize; 3],
        suggested: [usize; 3],
    },
    /// Use adaptive grid.
    UseAdaptiveGrid,
    /// Repair input mesh first.
    RepairInputMesh,
    /// Adjust shell thickness.
    AdjustThickness { current: f64, suggested: f64 },
    /// Simplify input mesh.
    SimplifyMesh { target_faces: usize },
    /// No specific suggestion.
    None,
}

impl std::fmt::Display for ShellRecoverySuggestion {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ShellRecoverySuggestion::ReduceGridResolution { current, suggested } => {
                write!(
                    f,
                    "Reduce grid resolution from {:?} to {:?}",
                    current, suggested
                )
            }
            ShellRecoverySuggestion::UseAdaptiveGrid => {
                write!(f, "Enable adaptive grid resolution for better performance")
            }
            ShellRecoverySuggestion::RepairInputMesh => {
                write!(f, "Run `mesh repair` on the input mesh first")
            }
            ShellRecoverySuggestion::AdjustThickness { current, suggested } => {
                write!(
                    f,
                    "Adjust thickness from {:.2}mm to {:.2}mm",
                    current, suggested
                )
            }
            ShellRecoverySuggestion::SimplifyMesh { target_faces } => {
                write!(
                    f,
                    "Simplify mesh to ~{} faces using decimation",
                    target_faces
                )
            }
            ShellRecoverySuggestion::None => {
                write!(f, "No specific suggestion available")
            }
        }
    }
}

/// Errors that can occur during shell operations.
#[derive(Debug, Error, Diagnostic)]
pub enum ShellError {
    /// Input mesh is empty.
    #[error("input mesh is empty")]
    #[diagnostic(
        code(shell::input::empty),
        help(
            "The input mesh must have at least one vertex and one face. Check that the mesh was loaded correctly."
        )
    )]
    EmptyMesh,

    /// SDF grid would be too large.
    #[error("SDF grid too large: {dims:?} = {total} voxels exceeds limit of {max}")]
    #[diagnostic(
        code(shell::grid::too_large),
        help("Reduce the grid resolution or use adaptive grid mode. Consider: --voxel-size {}",
             (*max as f64).powf(1.0/3.0).ceil() as usize)
    )]
    GridTooLarge {
        dims: [usize; 3],
        total: usize,
        max: usize,
    },

    /// Isosurface extraction failed.
    #[error("isosurface extraction produced empty mesh")]
    #[diagnostic(
        code(shell::isosurface::empty),
        help(
            "The shell thickness may be larger than the mesh dimensions, or the mesh may have degenerate geometry. Try reducing thickness or repairing the mesh."
        )
    )]
    EmptyIsosurface,

    /// Tag transfer failed.
    #[error("failed to transfer vertex tags: {details}")]
    #[diagnostic(
        code(shell::tags::transfer_failed),
        help(
            "Tag transfer requires a well-formed source mesh. Ensure the input mesh has valid vertex indices."
        )
    )]
    TagTransferFailed { details: String },

    /// Shell generation failed.
    #[error("shell generation failed: {details}")]
    #[diagnostic(
        code(shell::generation::failed),
        help("{}", suggestion.as_ref().map(|s| s.to_string()).unwrap_or_else(|| "Try repairing the input mesh or adjusting shell parameters.".to_string()))
    )]
    ShellGenerationFailed {
        details: String,
        suggestion: Option<ShellRecoverySuggestion>,
    },

    /// SDF computation failed.
    #[error("SDF computation failed: {details}")]
    #[diagnostic(
        code(shell::sdf::failed),
        help(
            "The mesh may have degenerate triangles or non-manifold geometry. Run `mesh repair` first."
        )
    )]
    SdfFailed {
        details: String,
        grid_dims: Option<[usize; 3]>,
    },

    /// Invalid parameters.
    #[error("invalid shell parameters: {details}")]
    #[diagnostic(
        code(shell::params::invalid),
        help("Check parameter values: thickness > 0, voxel_size > 0, etc.")
    )]
    InvalidParams {
        details: String,
        param_name: Option<String>,
        param_value: Option<String>,
    },

    /// Rim generation failed.
    #[error("rim generation failed: {details}")]
    #[diagnostic(
        code(shell::rim::failed),
        help("Rim generation requires open boundaries. Ensure the shell has valid open edges.")
    )]
    RimGenerationFailed { details: String },

    /// Underlying mesh error.
    #[error("mesh operation failed: {0}")]
    #[diagnostic(code(shell::mesh::error))]
    MeshError(#[from] mesh_repair::MeshError),
}

impl ShellError {
    /// Returns the machine-readable error code.
    pub fn code(&self) -> ShellErrorCode {
        match self {
            ShellError::EmptyMesh => ShellErrorCode::EmptyMesh,
            ShellError::GridTooLarge { .. } => ShellErrorCode::GridTooLarge,
            ShellError::EmptyIsosurface => ShellErrorCode::IsosurfaceFailed,
            ShellError::TagTransferFailed { .. } => ShellErrorCode::TagTransferFailed,
            ShellError::ShellGenerationFailed { .. } => ShellErrorCode::ShellGenerationFailed,
            ShellError::SdfFailed { .. } => ShellErrorCode::SdfFailed,
            ShellError::InvalidParams { .. } => ShellErrorCode::InvalidParams,
            ShellError::RimGenerationFailed { .. } => ShellErrorCode::RimGenerationFailed,
            ShellError::MeshError(_) => ShellErrorCode::ShellGenerationFailed,
        }
    }

    /// Returns a recovery suggestion for this error.
    pub fn recovery_suggestion(&self) -> ShellRecoverySuggestion {
        match self {
            ShellError::EmptyMesh => ShellRecoverySuggestion::RepairInputMesh,
            ShellError::GridTooLarge { dims, max, .. } => {
                // Calculate suggested dimensions that would fit within max
                let scale = (*max as f64 / (dims[0] * dims[1] * dims[2]) as f64).powf(1.0 / 3.0);
                let suggested = [
                    ((dims[0] as f64 * scale) as usize).max(1),
                    ((dims[1] as f64 * scale) as usize).max(1),
                    ((dims[2] as f64 * scale) as usize).max(1),
                ];
                ShellRecoverySuggestion::ReduceGridResolution {
                    current: *dims,
                    suggested,
                }
            }
            ShellError::EmptyIsosurface => ShellRecoverySuggestion::AdjustThickness {
                current: 0.0,
                suggested: 1.0,
            },
            ShellError::TagTransferFailed { .. } => ShellRecoverySuggestion::RepairInputMesh,
            ShellError::ShellGenerationFailed { suggestion, .. } => suggestion
                .clone()
                .unwrap_or(ShellRecoverySuggestion::RepairInputMesh),
            ShellError::SdfFailed { .. } => ShellRecoverySuggestion::RepairInputMesh,
            ShellError::InvalidParams { .. } => ShellRecoverySuggestion::None,
            ShellError::RimGenerationFailed { .. } => ShellRecoverySuggestion::RepairInputMesh,
            ShellError::MeshError(_) => ShellRecoverySuggestion::RepairInputMesh,
        }
    }

    // Constructor helpers

    /// Create an empty mesh error.
    pub fn empty_mesh() -> Self {
        ShellError::EmptyMesh
    }

    /// Create a grid too large error.
    pub fn grid_too_large(dims: [usize; 3], max: usize) -> Self {
        ShellError::GridTooLarge {
            dims,
            total: dims[0] * dims[1] * dims[2],
            max,
        }
    }

    /// Create an empty isosurface error.
    pub fn empty_isosurface() -> Self {
        ShellError::EmptyIsosurface
    }

    /// Create a tag transfer failed error.
    pub fn tag_transfer_failed(details: impl Into<String>) -> Self {
        ShellError::TagTransferFailed {
            details: details.into(),
        }
    }

    /// Create a shell generation failed error.
    pub fn shell_generation_failed(details: impl Into<String>) -> Self {
        ShellError::ShellGenerationFailed {
            details: details.into(),
            suggestion: None,
        }
    }

    /// Create a shell generation failed error with suggestion.
    pub fn shell_generation_failed_with_suggestion(
        details: impl Into<String>,
        suggestion: ShellRecoverySuggestion,
    ) -> Self {
        ShellError::ShellGenerationFailed {
            details: details.into(),
            suggestion: Some(suggestion),
        }
    }

    /// Create an SDF failed error.
    pub fn sdf_failed(details: impl Into<String>) -> Self {
        ShellError::SdfFailed {
            details: details.into(),
            grid_dims: None,
        }
    }

    /// Create an SDF failed error with grid dimensions.
    pub fn sdf_failed_with_grid(details: impl Into<String>, grid_dims: [usize; 3]) -> Self {
        ShellError::SdfFailed {
            details: details.into(),
            grid_dims: Some(grid_dims),
        }
    }

    /// Create an invalid params error.
    pub fn invalid_params(details: impl Into<String>) -> Self {
        ShellError::InvalidParams {
            details: details.into(),
            param_name: None,
            param_value: None,
        }
    }

    /// Create an invalid params error with param info.
    pub fn invalid_param(
        param_name: impl Into<String>,
        param_value: impl Into<String>,
        details: impl Into<String>,
    ) -> Self {
        ShellError::InvalidParams {
            details: details.into(),
            param_name: Some(param_name.into()),
            param_value: Some(param_value.into()),
        }
    }

    /// Create a rim generation failed error.
    pub fn rim_generation_failed(details: impl Into<String>) -> Self {
        ShellError::RimGenerationFailed {
            details: details.into(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_error_codes() {
        let err = ShellError::empty_mesh();
        assert_eq!(err.code(), ShellErrorCode::EmptyMesh);
        assert_eq!(err.code().as_str(), "SHELL-1001");
    }

    #[test]
    fn test_grid_too_large_suggestion() {
        let err = ShellError::grid_too_large([200, 200, 200], 1_000_000);
        let suggestion = err.recovery_suggestion();
        match suggestion {
            ShellRecoverySuggestion::ReduceGridResolution { current, suggested } => {
                assert_eq!(current, [200, 200, 200]);
                // Suggested should be smaller
                assert!(suggested[0] < 200 || suggested[1] < 200 || suggested[2] < 200);
            }
            _ => panic!("Expected ReduceGridResolution suggestion"),
        }
    }

    #[test]
    fn test_error_display() {
        let err = ShellError::grid_too_large([100, 100, 100], 500_000);
        let display = format!("{}", err);
        assert!(display.contains("1000000 voxels"));
        assert!(display.contains("500000"));
    }

    #[test]
    fn test_from_mesh_error() {
        let mesh_err = mesh_repair::MeshError::empty_mesh("test");
        let shell_err: ShellError = mesh_err.into();
        assert!(matches!(shell_err, ShellError::MeshError(_)));
    }
}

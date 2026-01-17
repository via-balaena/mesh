//! Error types for mesh operations with rich diagnostics.
//!
//! This module provides comprehensive error handling with:
//! - Machine-readable error codes for programmatic handling
//! - Rich context (which vertex, which face, what went wrong)
//! - Recovery suggestions for common issues
//! - Beautiful terminal display via miette
//!
//! # Error Codes
//!
//! Each error has a unique code in the format `MESH-XXXX`:
//! - `MESH-1xxx`: I/O errors (file reading, writing, parsing)
//! - `MESH-2xxx`: Validation errors (topology, coordinates)
//! - `MESH-3xxx`: Repair errors (operations that couldn't complete)
//! - `MESH-4xxx`: Format errors (unsupported or malformed data)
//!
//! # Example
//!
//! ```rust,ignore
//! use mesh_repair::{MeshError, ErrorCode};
//!
//! let err = MeshError::invalid_vertex_index(5, 100, 50);
//! println!("Error code: {}", err.code()); // MESH-2001
//! println!("Recovery: {:?}", err.recovery_suggestion());
//! ```

use miette::Diagnostic;
use std::path::PathBuf;
use thiserror::Error;

/// Result type alias for mesh operations.
pub type MeshResult<T> = Result<T, MeshError>;

/// Machine-readable error codes for mesh operations.
///
/// Codes follow the pattern `MESH-XXXX` where:
/// - 1xxx = I/O errors
/// - 2xxx = Validation errors
/// - 3xxx = Repair errors
/// - 4xxx = Format errors
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ErrorCode {
    // I/O errors (1xxx)
    /// MESH-1001: Failed to read file
    IoRead = 1001,
    /// MESH-1002: Failed to write file
    IoWrite = 1002,
    /// MESH-1003: Failed to parse file format
    ParseError = 1003,

    // Validation errors (2xxx)
    /// MESH-2001: Face references invalid vertex index
    InvalidVertexIndex = 2001,
    /// MESH-2002: Vertex has NaN or Infinity coordinate
    InvalidCoordinate = 2002,
    /// MESH-2003: Mesh has no vertices or faces
    EmptyMesh = 2003,
    /// MESH-2004: Invalid mesh topology (non-manifold, etc.)
    InvalidTopology = 2004,

    // Repair errors (3xxx)
    /// MESH-3001: Repair operation failed
    RepairFailed = 3001,
    /// MESH-3002: Hole filling failed
    HoleFillFailed = 3002,
    /// MESH-3003: Winding correction failed
    WindingFailed = 3003,
    /// MESH-3004: Decimation failed
    DecimationFailed = 3004,
    /// MESH-3005: Remeshing failed
    RemeshingFailed = 3005,
    /// MESH-3006: Boolean operation failed
    BooleanFailed = 3006,

    // Format errors (4xxx)
    /// MESH-4001: Unsupported file format
    UnsupportedFormat = 4001,
    /// MESH-4002: Malformed file structure
    MalformedFile = 4002,
}

impl ErrorCode {
    /// Returns the error code as a string in the format `MESH-XXXX`.
    pub fn as_str(&self) -> &'static str {
        match self {
            ErrorCode::IoRead => "MESH-1001",
            ErrorCode::IoWrite => "MESH-1002",
            ErrorCode::ParseError => "MESH-1003",
            ErrorCode::InvalidVertexIndex => "MESH-2001",
            ErrorCode::InvalidCoordinate => "MESH-2002",
            ErrorCode::EmptyMesh => "MESH-2003",
            ErrorCode::InvalidTopology => "MESH-2004",
            ErrorCode::RepairFailed => "MESH-3001",
            ErrorCode::HoleFillFailed => "MESH-3002",
            ErrorCode::WindingFailed => "MESH-3003",
            ErrorCode::DecimationFailed => "MESH-3004",
            ErrorCode::RemeshingFailed => "MESH-3005",
            ErrorCode::BooleanFailed => "MESH-3006",
            ErrorCode::UnsupportedFormat => "MESH-4001",
            ErrorCode::MalformedFile => "MESH-4002",
        }
    }
}

impl std::fmt::Display for ErrorCode {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.as_str())
    }
}

/// Recovery suggestions for mesh errors.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum RecoverySuggestion {
    /// Re-export the file from the original software with different settings.
    ReexportFile { format: Option<String> },
    /// Run repair operations to fix the issue.
    RunRepair { operations: Vec<String> },
    /// Use a different file format.
    UseDifferentFormat { suggested: Vec<String> },
    /// Check the original mesh for issues.
    CheckSourceMesh { checks: Vec<String> },
    /// Adjust parameters for the operation.
    AdjustParameters { parameters: Vec<(String, String)> },
    /// The mesh may be too complex for the operation.
    SimplifyMesh { target_faces: Option<usize> },
    /// Manual intervention may be required.
    ManualIntervention { description: String },
    /// No automatic recovery available.
    None,
}

impl std::fmt::Display for RecoverySuggestion {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            RecoverySuggestion::ReexportFile { format } => {
                if let Some(fmt) = format {
                    write!(
                        f,
                        "Try re-exporting the mesh as {} from the original software",
                        fmt
                    )
                } else {
                    write!(f, "Try re-exporting the mesh from the original software")
                }
            }
            RecoverySuggestion::RunRepair { operations } => {
                write!(f, "Run repair operations: {}", operations.join(", "))
            }
            RecoverySuggestion::UseDifferentFormat { suggested } => {
                write!(f, "Try using a different format: {}", suggested.join(", "))
            }
            RecoverySuggestion::CheckSourceMesh { checks } => {
                write!(f, "Check the source mesh for: {}", checks.join(", "))
            }
            RecoverySuggestion::AdjustParameters { parameters } => {
                let params: Vec<String> = parameters
                    .iter()
                    .map(|(k, v)| format!("{} = {}", k, v))
                    .collect();
                write!(f, "Try adjusting: {}", params.join(", "))
            }
            RecoverySuggestion::SimplifyMesh { target_faces } => {
                if let Some(target) = target_faces {
                    write!(f, "Try simplifying the mesh to ~{} faces first", target)
                } else {
                    write!(f, "Try simplifying the mesh first using decimation")
                }
            }
            RecoverySuggestion::ManualIntervention { description } => {
                write!(f, "{}", description)
            }
            RecoverySuggestion::None => {
                write!(f, "No automatic recovery available")
            }
        }
    }
}

/// Location information for mesh errors.
#[derive(Debug, Clone)]
pub enum MeshLocation {
    /// Error at a specific vertex.
    Vertex {
        index: usize,
        position: Option<[f64; 3]>,
    },
    /// Error at a specific face.
    Face {
        index: usize,
        vertices: Option<[u32; 3]>,
    },
    /// Error at a specific edge.
    Edge { vertex_a: usize, vertex_b: usize },
    /// Error in a file at a specific location.
    File {
        path: PathBuf,
        line: Option<usize>,
        column: Option<usize>,
    },
    /// Error in a region of the mesh.
    Region {
        description: String,
        face_count: usize,
    },
    /// No specific location.
    Unknown,
}

impl std::fmt::Display for MeshLocation {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            MeshLocation::Vertex { index, position } => {
                if let Some([x, y, z]) = position {
                    write!(f, "vertex {} at ({:.3}, {:.3}, {:.3})", index, x, y, z)
                } else {
                    write!(f, "vertex {}", index)
                }
            }
            MeshLocation::Face { index, vertices } => {
                if let Some([a, b, c]) = vertices {
                    write!(f, "face {} with vertices [{}, {}, {}]", index, a, b, c)
                } else {
                    write!(f, "face {}", index)
                }
            }
            MeshLocation::Edge { vertex_a, vertex_b } => {
                write!(f, "edge between vertices {} and {}", vertex_a, vertex_b)
            }
            MeshLocation::File { path, line, column } => {
                let mut result = path.display().to_string();
                if let Some(l) = line {
                    result.push_str(&format!(":{}", l));
                    if let Some(c) = column {
                        result.push_str(&format!(":{}", c));
                    }
                }
                write!(f, "{}", result)
            }
            MeshLocation::Region {
                description,
                face_count,
            } => {
                write!(f, "{} ({} faces)", description, face_count)
            }
            MeshLocation::Unknown => {
                write!(f, "unknown location")
            }
        }
    }
}

/// Errors that can occur during mesh operations.
///
/// Each error variant includes:
/// - A human-readable message
/// - A machine-readable error code
/// - Optional location information
/// - Recovery suggestions when available
#[derive(Debug, Error, Diagnostic)]
pub enum MeshError {
    /// Error reading from a file.
    #[error("failed to read mesh from {path}")]
    #[diagnostic(
        code(mesh::io::read),
        help("Check that the file exists and is readable. Try: ls -la {}", path.display())
    )]
    IoRead {
        path: PathBuf,
        #[source]
        source: std::io::Error,
    },

    /// Error writing to a file.
    #[error("failed to write mesh to {path}")]
    #[diagnostic(
        code(mesh::io::write),
        help("Check that the directory exists and is writable")
    )]
    IoWrite {
        path: PathBuf,
        #[source]
        source: std::io::Error,
    },

    /// Error parsing mesh file format.
    #[error("failed to parse mesh from {path}: {details}")]
    #[diagnostic(
        code(mesh::parse::error),
        help(
            "The file may be corrupted or in an unsupported format variant. Try re-exporting from the original software."
        )
    )]
    ParseError { path: PathBuf, details: String },

    /// Unsupported file format.
    #[error("unsupported mesh format: {extension:?}")]
    #[diagnostic(
        code(mesh::format::unsupported),
        help("Supported formats: STL, OBJ, PLY, 3MF, STEP (with 'step' feature)")
    )]
    UnsupportedFormat { extension: Option<String> },

    /// Empty mesh (no vertices or faces).
    #[error("mesh is empty: {details}")]
    #[diagnostic(
        code(mesh::validation::empty),
        help(
            "The mesh must have at least one vertex and one face. Check that the file was exported correctly."
        )
    )]
    EmptyMesh { details: String },

    /// Invalid mesh topology.
    #[error("invalid mesh topology: {details}")]
    #[diagnostic(
        code(mesh::validation::topology),
        help(
            "Try running `mesh repair` to fix topology issues, or use `mesh validate` for a detailed report."
        )
    )]
    InvalidTopology { details: String },

    /// Mesh repair failed.
    #[error("mesh repair failed: {details}")]
    #[diagnostic(
        code(mesh::repair::failed),
        help("Try running individual repair operations to identify the specific issue.")
    )]
    RepairFailed { details: String },

    /// Invalid vertex index in face data.
    #[error(
        "invalid vertex index: face {face_index} references vertex {vertex_index}, but mesh only has {vertex_count} vertices"
    )]
    #[diagnostic(
        code(mesh::validation::vertex_index),
        help(
            "Run `mesh repair` to remove faces with invalid vertex references, or check the mesh export settings."
        )
    )]
    InvalidVertexIndex {
        face_index: usize,
        vertex_index: u32,
        vertex_count: usize,
    },

    /// Invalid coordinate value (NaN or Infinity).
    #[error("invalid coordinate at vertex {vertex_index}: {coordinate} is {value}")]
    #[diagnostic(
        code(mesh::validation::coordinate),
        help(
            "Check for numerical issues in the source data. This often happens with very small or very large values."
        )
    )]
    InvalidCoordinate {
        vertex_index: usize,
        coordinate: &'static str,
        value: f64,
    },

    /// Hole filling failed.
    #[error("hole filling failed: {details}")]
    #[diagnostic(
        code(mesh::repair::hole_fill),
        help(
            "The hole may be too complex or have self-intersecting boundaries. Try splitting the mesh or filling manually."
        )
    )]
    HoleFillFailed { details: String },

    /// Boolean operation failed.
    #[error("boolean operation failed: {details}")]
    #[diagnostic(
        code(mesh::boolean::failed),
        help(
            "Ensure both meshes are watertight and non-self-intersecting. Try running `mesh repair` on both inputs first."
        )
    )]
    BooleanFailed { details: String, operation: String },

    /// Decimation failed.
    #[error("decimation failed: {details}")]
    #[diagnostic(
        code(mesh::decimate::failed),
        help(
            "Try a less aggressive target ratio or ensure the mesh has valid topology before decimation."
        )
    )]
    DecimationFailed { details: String },

    /// Remeshing failed.
    #[error("remeshing failed: {details}")]
    #[diagnostic(
        code(mesh::remesh::failed),
        help("Try adjusting the target edge length or repairing the mesh first.")
    )]
    RemeshingFailed { details: String },
}

impl MeshError {
    /// Returns the machine-readable error code.
    pub fn code(&self) -> ErrorCode {
        match self {
            MeshError::IoRead { .. } => ErrorCode::IoRead,
            MeshError::IoWrite { .. } => ErrorCode::IoWrite,
            MeshError::ParseError { .. } => ErrorCode::ParseError,
            MeshError::UnsupportedFormat { .. } => ErrorCode::UnsupportedFormat,
            MeshError::EmptyMesh { .. } => ErrorCode::EmptyMesh,
            MeshError::InvalidTopology { .. } => ErrorCode::InvalidTopology,
            MeshError::RepairFailed { .. } => ErrorCode::RepairFailed,
            MeshError::InvalidVertexIndex { .. } => ErrorCode::InvalidVertexIndex,
            MeshError::InvalidCoordinate { .. } => ErrorCode::InvalidCoordinate,
            MeshError::HoleFillFailed { .. } => ErrorCode::HoleFillFailed,
            MeshError::BooleanFailed { .. } => ErrorCode::BooleanFailed,
            MeshError::DecimationFailed { .. } => ErrorCode::DecimationFailed,
            MeshError::RemeshingFailed { .. } => ErrorCode::RemeshingFailed,
        }
    }

    /// Returns a recovery suggestion for this error.
    pub fn recovery_suggestion(&self) -> RecoverySuggestion {
        match self {
            MeshError::IoRead { .. } => RecoverySuggestion::CheckSourceMesh {
                checks: vec!["file exists".into(), "file permissions".into()],
            },
            MeshError::IoWrite { .. } => RecoverySuggestion::CheckSourceMesh {
                checks: vec!["directory exists".into(), "write permissions".into()],
            },
            MeshError::ParseError { .. } => RecoverySuggestion::ReexportFile {
                format: Some("binary STL or OBJ".into()),
            },
            MeshError::UnsupportedFormat { .. } => RecoverySuggestion::UseDifferentFormat {
                suggested: vec!["STL".into(), "OBJ".into(), "PLY".into(), "3MF".into()],
            },
            MeshError::EmptyMesh { .. } => RecoverySuggestion::CheckSourceMesh {
                checks: vec!["mesh has geometry".into(), "correct export settings".into()],
            },
            MeshError::InvalidTopology { .. } => RecoverySuggestion::RunRepair {
                operations: vec!["fix_winding".into(), "remove_degenerate".into()],
            },
            MeshError::RepairFailed { .. } => RecoverySuggestion::ManualIntervention {
                description: "Try running individual repair operations to identify the issue"
                    .into(),
            },
            MeshError::InvalidVertexIndex { .. } => RecoverySuggestion::RunRepair {
                operations: vec!["validate".into(), "remove_invalid_faces".into()],
            },
            MeshError::InvalidCoordinate { .. } => RecoverySuggestion::CheckSourceMesh {
                checks: vec!["coordinate values".into(), "export precision".into()],
            },
            MeshError::HoleFillFailed { .. } => RecoverySuggestion::RunRepair {
                operations: vec!["fill_holes with max_edges parameter".into()],
            },
            MeshError::BooleanFailed { .. } => RecoverySuggestion::RunRepair {
                operations: vec![
                    "repair both meshes".into(),
                    "check for self-intersections".into(),
                ],
            },
            MeshError::DecimationFailed { .. } => RecoverySuggestion::AdjustParameters {
                parameters: vec![("target_ratio".into(), "try a higher value".into())],
            },
            MeshError::RemeshingFailed { .. } => RecoverySuggestion::AdjustParameters {
                parameters: vec![("target_edge_length".into(), "try a larger value".into())],
            },
        }
    }

    /// Returns location information if available.
    pub fn location(&self) -> Option<MeshLocation> {
        match self {
            MeshError::InvalidVertexIndex { face_index, .. } => Some(MeshLocation::Face {
                index: *face_index,
                vertices: None,
            }),
            MeshError::InvalidCoordinate { vertex_index, .. } => Some(MeshLocation::Vertex {
                index: *vertex_index,
                position: None,
            }),
            MeshError::ParseError { path, .. } => Some(MeshLocation::File {
                path: path.clone(),
                line: None,
                column: None,
            }),
            MeshError::IoRead { path, .. } => Some(MeshLocation::File {
                path: path.clone(),
                line: None,
                column: None,
            }),
            MeshError::IoWrite { path, .. } => Some(MeshLocation::File {
                path: path.clone(),
                line: None,
                column: None,
            }),
            _ => None,
        }
    }

    // Constructor helpers for common error patterns

    /// Create an IoRead error.
    pub fn io_read(path: impl Into<PathBuf>, source: std::io::Error) -> Self {
        MeshError::IoRead {
            path: path.into(),
            source,
        }
    }

    /// Create an IoWrite error.
    pub fn io_write(path: impl Into<PathBuf>, source: std::io::Error) -> Self {
        MeshError::IoWrite {
            path: path.into(),
            source,
        }
    }

    /// Create a ParseError.
    pub fn parse_error(path: impl Into<PathBuf>, details: impl Into<String>) -> Self {
        MeshError::ParseError {
            path: path.into(),
            details: details.into(),
        }
    }

    /// Create an InvalidVertexIndex error.
    pub fn invalid_vertex_index(face_index: usize, vertex_index: u32, vertex_count: usize) -> Self {
        MeshError::InvalidVertexIndex {
            face_index,
            vertex_index,
            vertex_count,
        }
    }

    /// Create an InvalidCoordinate error.
    pub fn invalid_coordinate(vertex_index: usize, coordinate: &'static str, value: f64) -> Self {
        MeshError::InvalidCoordinate {
            vertex_index,
            coordinate,
            value,
        }
    }

    /// Create an EmptyMesh error.
    pub fn empty_mesh(details: impl Into<String>) -> Self {
        MeshError::EmptyMesh {
            details: details.into(),
        }
    }

    /// Create an InvalidTopology error.
    pub fn invalid_topology(details: impl Into<String>) -> Self {
        MeshError::InvalidTopology {
            details: details.into(),
        }
    }

    /// Create a RepairFailed error.
    pub fn repair_failed(details: impl Into<String>) -> Self {
        MeshError::RepairFailed {
            details: details.into(),
        }
    }

    /// Create a HoleFillFailed error.
    pub fn hole_fill_failed(details: impl Into<String>) -> Self {
        MeshError::HoleFillFailed {
            details: details.into(),
        }
    }

    /// Create a BooleanFailed error.
    pub fn boolean_failed(operation: impl Into<String>, details: impl Into<String>) -> Self {
        MeshError::BooleanFailed {
            details: details.into(),
            operation: operation.into(),
        }
    }

    /// Create a DecimationFailed error.
    pub fn decimation_failed(details: impl Into<String>) -> Self {
        MeshError::DecimationFailed {
            details: details.into(),
        }
    }

    /// Create a RemeshingFailed error.
    pub fn remeshing_failed(details: impl Into<String>) -> Self {
        MeshError::RemeshingFailed {
            details: details.into(),
        }
    }

    /// Create an UnsupportedFormat error.
    pub fn unsupported_format(extension: Option<String>) -> Self {
        MeshError::UnsupportedFormat { extension }
    }
}

/// Validation issues that can be collected during mesh validation.
///
/// Unlike `MeshError`, these represent issues that may be warnings rather than errors,
/// and multiple issues can be collected without stopping validation.
#[derive(Debug, Clone)]
pub enum ValidationIssue {
    /// Face references a vertex index that doesn't exist.
    InvalidVertexIndex {
        face_index: usize,
        vertex_index: u32,
        vertex_count: usize,
    },
    /// Vertex has NaN coordinate.
    NaNCoordinate {
        vertex_index: usize,
        coordinate: &'static str,
    },
    /// Vertex has infinite coordinate.
    InfiniteCoordinate {
        vertex_index: usize,
        coordinate: &'static str,
        value: f64,
    },
    /// Degenerate face (zero area).
    DegenerateFace { face_index: usize, area: f64 },
    /// Non-manifold edge (shared by more than 2 faces).
    NonManifoldEdge {
        vertex_a: usize,
        vertex_b: usize,
        face_count: usize,
    },
    /// Inconsistent winding order.
    InconsistentWinding {
        face_index: usize,
        neighbor_index: usize,
    },
    /// Self-intersection detected.
    SelfIntersection { face_a: usize, face_b: usize },
}

impl ValidationIssue {
    /// Returns a severity level for the issue.
    pub fn severity(&self) -> IssueSeverity {
        match self {
            ValidationIssue::InvalidVertexIndex { .. } => IssueSeverity::Error,
            ValidationIssue::NaNCoordinate { .. } => IssueSeverity::Error,
            ValidationIssue::InfiniteCoordinate { .. } => IssueSeverity::Error,
            ValidationIssue::DegenerateFace { .. } => IssueSeverity::Warning,
            ValidationIssue::NonManifoldEdge { .. } => IssueSeverity::Warning,
            ValidationIssue::InconsistentWinding { .. } => IssueSeverity::Warning,
            ValidationIssue::SelfIntersection { .. } => IssueSeverity::Warning,
        }
    }

    /// Returns an error code for programmatic handling.
    pub fn code(&self) -> &'static str {
        match self {
            ValidationIssue::InvalidVertexIndex { .. } => "MESH-2001",
            ValidationIssue::NaNCoordinate { .. } => "MESH-2002",
            ValidationIssue::InfiniteCoordinate { .. } => "MESH-2002",
            ValidationIssue::DegenerateFace { .. } => "MESH-2005",
            ValidationIssue::NonManifoldEdge { .. } => "MESH-2006",
            ValidationIssue::InconsistentWinding { .. } => "MESH-2007",
            ValidationIssue::SelfIntersection { .. } => "MESH-2008",
        }
    }

    /// Returns a recovery suggestion.
    pub fn suggestion(&self) -> &'static str {
        match self {
            ValidationIssue::InvalidVertexIndex { .. } => {
                "Remove faces with invalid vertex references using `mesh repair`"
            }
            ValidationIssue::NaNCoordinate { .. } | ValidationIssue::InfiniteCoordinate { .. } => {
                "Check source data for numerical issues; try re-exporting"
            }
            ValidationIssue::DegenerateFace { .. } => {
                "Run `mesh repair` to remove degenerate faces"
            }
            ValidationIssue::NonManifoldEdge { .. } => {
                "Run `mesh repair` to fix non-manifold edges"
            }
            ValidationIssue::InconsistentWinding { .. } => "Run `mesh repair` to fix winding order",
            ValidationIssue::SelfIntersection { .. } => {
                "Self-intersections may need manual repair in a 3D editor"
            }
        }
    }
}

/// Severity levels for validation issues.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum IssueSeverity {
    /// Informational, no action needed.
    Info,
    /// Warning, mesh may have issues.
    Warning,
    /// Error, mesh is invalid.
    Error,
}

impl std::fmt::Display for ValidationIssue {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ValidationIssue::InvalidVertexIndex {
                face_index,
                vertex_index,
                vertex_count,
            } => {
                write!(
                    f,
                    "face {} references vertex {}, but mesh only has {} vertices",
                    face_index, vertex_index, vertex_count
                )
            }
            ValidationIssue::NaNCoordinate {
                vertex_index,
                coordinate,
            } => {
                write!(
                    f,
                    "vertex {} has NaN {} coordinate",
                    vertex_index, coordinate
                )
            }
            ValidationIssue::InfiniteCoordinate {
                vertex_index,
                coordinate,
                value,
            } => {
                write!(
                    f,
                    "vertex {} has infinite {} coordinate ({})",
                    vertex_index, coordinate, value
                )
            }
            ValidationIssue::DegenerateFace { face_index, area } => {
                write!(f, "face {} is degenerate (area: {:.2e})", face_index, area)
            }
            ValidationIssue::NonManifoldEdge {
                vertex_a,
                vertex_b,
                face_count,
            } => {
                write!(
                    f,
                    "edge ({}, {}) is non-manifold (shared by {} faces)",
                    vertex_a, vertex_b, face_count
                )
            }
            ValidationIssue::InconsistentWinding {
                face_index,
                neighbor_index,
            } => {
                write!(
                    f,
                    "face {} has inconsistent winding with neighbor {}",
                    face_index, neighbor_index
                )
            }
            ValidationIssue::SelfIntersection { face_a, face_b } => {
                write!(f, "faces {} and {} self-intersect", face_a, face_b)
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_error_codes() {
        let err = MeshError::invalid_vertex_index(5, 100, 50);
        assert_eq!(err.code(), ErrorCode::InvalidVertexIndex);
        assert_eq!(err.code().as_str(), "MESH-2001");
    }

    #[test]
    fn test_recovery_suggestions() {
        let err = MeshError::invalid_topology("non-manifold edge");
        let suggestion = err.recovery_suggestion();
        match suggestion {
            RecoverySuggestion::RunRepair { operations } => {
                assert!(!operations.is_empty());
            }
            _ => panic!("Expected RunRepair suggestion"),
        }
    }

    #[test]
    fn test_location_info() {
        let err = MeshError::invalid_vertex_index(5, 100, 50);
        let location = err.location();
        assert!(location.is_some());
        match location.unwrap() {
            MeshLocation::Face { index, .. } => {
                assert_eq!(index, 5);
            }
            _ => panic!("Expected Face location"),
        }
    }

    #[test]
    fn test_validation_issue_severity() {
        let issue = ValidationIssue::DegenerateFace {
            face_index: 0,
            area: 0.0,
        };
        assert_eq!(issue.severity(), IssueSeverity::Warning);

        let issue = ValidationIssue::InvalidVertexIndex {
            face_index: 0,
            vertex_index: 100,
            vertex_count: 50,
        };
        assert_eq!(issue.severity(), IssueSeverity::Error);
    }

    #[test]
    fn test_error_display() {
        let err = MeshError::invalid_vertex_index(5, 100, 50);
        let display = format!("{}", err);
        assert!(display.contains("face 5"));
        assert!(display.contains("vertex 100"));
        assert!(display.contains("50 vertices"));
    }
}

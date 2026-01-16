//! Error types for mesh operations.

use std::path::PathBuf;
use thiserror::Error;

/// Result type alias for mesh operations.
pub type MeshResult<T> = Result<T, MeshError>;

/// Errors that can occur during mesh operations.
#[derive(Debug, Error)]
pub enum MeshError {
    /// Error reading from a file.
    #[error("failed to read mesh from {path}: {source}")]
    IoRead {
        path: PathBuf,
        #[source]
        source: std::io::Error,
    },

    /// Error writing to a file.
    #[error("failed to write mesh to {path}: {source}")]
    IoWrite {
        path: PathBuf,
        #[source]
        source: std::io::Error,
    },

    /// Error parsing mesh file format.
    #[error("failed to parse mesh from {path}: {details}")]
    ParseError { path: PathBuf, details: String },

    /// Unsupported file format.
    #[error("unsupported mesh format: {extension:?}")]
    UnsupportedFormat { extension: Option<String> },

    /// Empty mesh (no vertices or faces).
    #[error("mesh is empty: {details}")]
    EmptyMesh { details: String },

    /// Invalid mesh topology.
    #[error("invalid mesh topology: {details}")]
    InvalidTopology { details: String },

    /// Mesh repair failed.
    #[error("mesh repair failed: {details}")]
    RepairFailed { details: String },

    /// Invalid vertex index in face data.
    #[error("invalid vertex index: face {face_index} references vertex {vertex_index}, but mesh only has {vertex_count} vertices")]
    InvalidVertexIndex {
        face_index: usize,
        vertex_index: u32,
        vertex_count: usize,
    },

    /// Invalid coordinate value (NaN or Infinity).
    #[error("invalid coordinate at vertex {vertex_index}: {coordinate} is {value}")]
    InvalidCoordinate {
        vertex_index: usize,
        coordinate: &'static str,
        value: f64,
    },
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
}

impl std::fmt::Display for ValidationIssue {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ValidationIssue::InvalidVertexIndex { face_index, vertex_index, vertex_count } => {
                write!(f, "face {} references vertex {}, but mesh only has {} vertices",
                    face_index, vertex_index, vertex_count)
            }
            ValidationIssue::NaNCoordinate { vertex_index, coordinate } => {
                write!(f, "vertex {} has NaN {} coordinate", vertex_index, coordinate)
            }
            ValidationIssue::InfiniteCoordinate { vertex_index, coordinate, value } => {
                write!(f, "vertex {} has infinite {} coordinate ({})", vertex_index, coordinate, value)
            }
        }
    }
}

//! SDF-based mesh offset.
//!
//! This module provides the core SDF offset functionality for creating
//! offset surfaces without self-intersections.

pub(crate) mod grid;
pub(crate) mod extract;
mod transfer;
mod sdf;
mod adaptive;

pub use sdf::{apply_sdf_offset, SdfOffsetResult, SdfOffsetStats};
pub use grid::SdfOffsetParams;
pub use adaptive::{
    AdaptiveSdfParams, AdaptiveGridStats, AdaptiveGridResult,
    create_adaptive_grid, interpolate_offsets_adaptive,
};

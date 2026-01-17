//! SDF-based mesh offset.
//!
//! This module provides the core SDF offset functionality for creating
//! offset surfaces without self-intersections.

mod adaptive;
pub(crate) mod extract;
pub(crate) mod grid;
mod sdf;
mod transfer;

pub use grid::SdfOffsetParams;
pub use sdf::{SdfOffsetResult, SdfOffsetStats, apply_sdf_offset};

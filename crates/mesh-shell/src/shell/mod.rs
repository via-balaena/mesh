//! Shell generation for printable geometry.
//!
//! Transforms the inner surface into a printable shell with walls.

mod generate;
pub mod rim;
pub mod validation;

pub use generate::{
    ShellParams, ShellResult, WallGenerationMethod, generate_shell, generate_shell_no_validation,
    generate_shell_with_progress,
};
pub use rim::{
    BoundaryAnalysis, BoundaryLoop, RimResult, analyze_boundary, generate_rim,
    generate_rim_advanced, generate_rim_for_sdf_shell, validate_boundary_for_rim,
};
pub use validation::{
    ShellIssue, ShellRepairResult, ShellValidationResult, repair_shell, validate_and_repair_shell,
    validate_shell,
};

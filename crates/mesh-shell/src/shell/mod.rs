//! Shell generation for printable geometry.
//!
//! Transforms the inner surface into a printable shell with walls.

mod generate;
pub mod rim;
pub mod validation;

pub use generate::{
    generate_shell, generate_shell_no_validation, generate_shell_with_progress,
    ShellParams, ShellResult, WallGenerationMethod,
};
pub use validation::{
    validate_shell, validate_and_repair_shell, repair_shell,
    ShellValidationResult, ShellIssue, ShellRepairResult,
};
pub use rim::{
    analyze_boundary, generate_rim, generate_rim_advanced, generate_rim_for_sdf_shell,
    validate_boundary_for_rim, BoundaryAnalysis, BoundaryLoop, RimResult,
};

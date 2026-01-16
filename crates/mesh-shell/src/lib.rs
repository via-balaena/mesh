//! Shell generation around 3D meshes using SDF-based offset.
//!
//! This crate provides tools for generating printable shells around 3D meshes
//! using signed distance field (SDF) based offset techniques.
//!
//! # Features
//!
//! - **SDF-based offset**: Robust offset that avoids self-intersections
//! - **Variable offset**: Per-vertex offset values for complex shapes
//! - **Shell generation**: Create watertight shells with walls
//! - **Rim generation**: Clean boundary edges connecting inner and outer surfaces
//!
//! # Example
//!
//! ```no_run
//! use mesh_repair::Mesh;
//! use mesh_shell::{apply_sdf_offset, generate_shell, SdfOffsetParams, ShellParams};
//!
//! // Load and prepare mesh
//! let mut mesh = Mesh::load("scan.stl").unwrap();
//!
//! // Set offset values on vertices (uses mesh.vertices[i].offset field)
//! for v in &mut mesh.vertices {
//!     v.offset = Some(2.0); // 2mm outward offset
//! }
//!
//! // Apply SDF offset to create the inner shell
//! let params = SdfOffsetParams::default();
//! let result = apply_sdf_offset(&mesh, &params).unwrap();
//! let inner_shell = result.mesh;
//!
//! // Generate printable shell with walls
//! let shell_params = ShellParams::default();
//! let (shell, stats) = generate_shell(&inner_shell, &shell_params);
//!
//! shell.save("shell.3mf").unwrap();
//! ```

mod error;
mod offset;
mod shell;

pub use error::{ShellError, ShellResult};

// SDF offset
pub use offset::{
    apply_sdf_offset, SdfOffsetParams, SdfOffsetResult, SdfOffsetStats,
};

// Shell generation (rename to avoid conflict with error::ShellResult)
pub use shell::{
    generate_shell, generate_shell_no_validation,
    ShellParams, ShellResult as ShellGenerationResult, WallGenerationMethod,
};

// Shell validation and repair
pub use shell::{
    validate_shell, validate_and_repair_shell, repair_shell,
    ShellValidationResult, ShellIssue, ShellRepairResult,
};

// Rim generation and boundary analysis
pub use shell::{
    analyze_boundary, generate_rim, generate_rim_advanced, generate_rim_for_sdf_shell,
    validate_boundary_for_rim, BoundaryAnalysis, BoundaryLoop, RimResult,
};

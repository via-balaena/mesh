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
//! - **Builder API**: Fluent builder pattern for ergonomic configuration
//!
//! # Quick Start with ShellBuilder
//!
//! The recommended way to generate shells is using the [`ShellBuilder`]:
//!
//! ```no_run
//! use mesh_repair::Mesh;
//! use mesh_shell::ShellBuilder;
//!
//! let mesh = Mesh::load("scan.stl").unwrap();
//!
//! // Simple: generate shell with defaults
//! let result = ShellBuilder::new(&mesh)
//!     .offset(2.0)           // 2mm outward offset
//!     .wall_thickness(2.5)   // 2.5mm walls
//!     .build()
//!     .unwrap();
//!
//! result.mesh.save("shell.3mf").unwrap();
//! ```
//!
//! # Advanced Configuration
//!
//! ```no_run
//! use mesh_repair::Mesh;
//! use mesh_repair::progress::ProgressCallback;
//! use mesh_shell::ShellBuilder;
//!
//! let mesh = Mesh::load("scan.stl").unwrap();
//!
//! let callback: ProgressCallback = Box::new(|progress| {
//!     println!("{}%: {}", progress.percent(), progress.message);
//!     true // continue
//! });
//!
//! let result = ShellBuilder::new(&mesh)
//!     .offset(3.0)
//!     .wall_thickness(2.0)
//!     .voxel_size(0.5)       // Fine resolution
//!     .high_quality()         // SDF-based walls
//!     .use_gpu(true)          // GPU acceleration
//!     .with_progress(callback)
//!     .build()
//!     .unwrap();
//! ```
//!
//! # Low-Level API
//!
//! For more control, use the low-level functions directly:
//!
//! ```no_run
//! use mesh_repair::Mesh;
//! use mesh_shell::{apply_sdf_offset, generate_shell, SdfOffsetParams, ShellParams};
//!
//! let mut mesh = Mesh::load("scan.stl").unwrap();
//!
//! // Set offset values on vertices
//! for v in &mut mesh.vertices {
//!     v.offset = Some(2.0);
//! }
//!
//! // Apply SDF offset
//! let params = SdfOffsetParams::default();
//! let result = apply_sdf_offset(&mesh, &params).unwrap();
//!
//! // Generate shell with walls
//! let shell_params = ShellParams::default();
//! let (shell, stats) = generate_shell(&result.mesh, &shell_params);
//!
//! shell.save("shell.3mf").unwrap();
//! ```

mod builder;
mod error;
mod offset;
mod shell;

pub use error::{ShellError, ShellResult};

// Builder API (recommended)
pub use builder::{ShellBuildResult, ShellBuilder};

// SDF offset
pub use offset::{SdfOffsetParams, SdfOffsetResult, SdfOffsetStats, apply_sdf_offset};

// Shell generation (rename to avoid conflict with error::ShellResult)
pub use shell::{
    ShellParams, ShellResult as ShellGenerationResult, WallGenerationMethod, generate_shell,
    generate_shell_no_validation, generate_shell_with_progress,
};

// Shell validation and repair
pub use shell::{
    ShellIssue, ShellRepairResult, ShellValidationResult, repair_shell, validate_and_repair_shell,
    validate_shell,
};

// Rim generation and boundary analysis
pub use shell::{
    BoundaryAnalysis, BoundaryLoop, RimResult, analyze_boundary, generate_rim,
    generate_rim_advanced, generate_rim_for_sdf_shell, validate_boundary_for_rim,
};

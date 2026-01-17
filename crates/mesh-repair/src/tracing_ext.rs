//! Tracing extensions for mesh operations.
//!
//! This module provides structured logging and performance tracing for mesh operations.
//! It integrates with the `tracing` ecosystem to provide:
//!
//! - **Performance spans**: Track operation timing with `#[instrument]`
//! - **Structured fields**: Log mesh dimensions, face counts, timing
//! - **Progress events**: Emit progress updates for long-running operations
//! - **Debug logging**: Detailed state logging for troubleshooting
//!
//! # Usage
//!
//! Enable tracing by initializing a subscriber in your application:
//!
//! ```rust,ignore
//! use tracing_subscriber::{fmt, prelude::*, EnvFilter};
//!
//! // Initialize with environment filter
//! tracing_subscriber::registry()
//!     .with(fmt::layer())
//!     .with(EnvFilter::from_default_env())
//!     .init();
//!
//! // Now mesh operations will emit traces
//! // Set RUST_LOG=mesh_repair=debug for detailed output
//! ```
//!
//! # Log Levels
//!
//! - **ERROR**: Operation failures, invalid data
//! - **WARN**: Recoverable issues, deprecated usage
//! - **INFO**: High-level operation summaries, timing
//! - **DEBUG**: Detailed operation progress, intermediate states
//! - **TRACE**: Very detailed state dumps, per-vertex/face logging

use std::time::Instant;
use tracing::{Span, debug, info, trace, warn};

/// A performance timer that logs duration on drop.
///
/// Use this to measure and log operation timing automatically.
///
/// # Example
///
/// ```rust,ignore
/// use mesh_repair::tracing_ext::OperationTimer;
///
/// fn expensive_operation() {
///     let _timer = OperationTimer::new("expensive_operation");
///     // ... do work ...
/// } // Timer logs duration when dropped
/// ```
pub struct OperationTimer {
    name: &'static str,
    start: Instant,
    span: Span,
}

impl OperationTimer {
    /// Create a new operation timer.
    pub fn new(name: &'static str) -> Self {
        let span = tracing::info_span!("mesh_operation", operation = name);
        debug!(target: "mesh_repair::timing", operation = name, "Starting operation");
        Self {
            name,
            start: Instant::now(),
            span,
        }
    }

    /// Create a timer with additional context fields.
    pub fn with_context(name: &'static str, face_count: usize, vertex_count: usize) -> Self {
        let span = tracing::info_span!(
            "mesh_operation",
            operation = name,
            faces = face_count,
            vertices = vertex_count
        );
        debug!(
            target: "mesh_repair::timing",
            operation = name,
            faces = face_count,
            vertices = vertex_count,
            "Starting operation"
        );
        Self {
            name,
            start: Instant::now(),
            span,
        }
    }

    /// Get the elapsed time.
    pub fn elapsed_ms(&self) -> f64 {
        self.start.elapsed().as_secs_f64() * 1000.0
    }

    /// Get the span for this timer.
    pub fn span(&self) -> &Span {
        &self.span
    }
}

impl Drop for OperationTimer {
    fn drop(&mut self) {
        let elapsed_ms = self.elapsed_ms();
        info!(
            target: "mesh_repair::timing",
            operation = self.name,
            elapsed_ms = format!("{:.2}", elapsed_ms),
            "Operation completed"
        );
    }
}

/// Log mesh statistics at debug level.
///
/// Use this to log the current state of a mesh for debugging.
pub fn log_mesh_stats(mesh: &crate::Mesh, context: &str) {
    let (min_bounds, max_bounds) = mesh.bounds().unwrap_or_default();
    let dims = max_bounds - min_bounds;

    debug!(
        target: "mesh_repair::mesh_state",
        context = context,
        vertices = mesh.vertex_count(),
        faces = mesh.face_count(),
        dimensions = format!("{:.2} x {:.2} x {:.2}", dims.x, dims.y, dims.z),
        "Mesh state"
    );
}

/// Log mesh statistics at trace level (more detailed).
pub fn log_mesh_stats_detailed(mesh: &crate::Mesh, context: &str) {
    let (min_bounds, max_bounds) = mesh.bounds().unwrap_or_default();
    let dims = max_bounds - min_bounds;

    // Calculate some basic statistics
    let has_normals = mesh.vertices.iter().any(|v| v.normal.is_some());
    let has_colors = mesh.vertices.iter().any(|v| v.color.is_some());

    trace!(
        target: "mesh_repair::mesh_state",
        context = context,
        vertices = mesh.vertex_count(),
        faces = mesh.face_count(),
        min_x = format!("{:.4}", min_bounds.x),
        min_y = format!("{:.4}", min_bounds.y),
        min_z = format!("{:.4}", min_bounds.z),
        max_x = format!("{:.4}", max_bounds.x),
        max_y = format!("{:.4}", max_bounds.y),
        max_z = format!("{:.4}", max_bounds.z),
        width = format!("{:.4}", dims.x),
        height = format!("{:.4}", dims.y),
        depth = format!("{:.4}", dims.z),
        has_normals = has_normals,
        has_colors = has_colors,
        "Detailed mesh state"
    );
}

/// Log a validation result.
pub fn log_validation_result(report: &crate::MeshReport) {
    if report.is_valid() {
        info!(
            target: "mesh_repair::validation",
            is_watertight = report.is_watertight,
            is_manifold = report.is_manifold,
            vertex_count = report.vertex_count,
            face_count = report.face_count,
            "Mesh validation passed"
        );
    } else {
        warn!(
            target: "mesh_repair::validation",
            is_watertight = report.is_watertight,
            is_manifold = report.is_manifold,
            boundary_edges = report.boundary_edge_count,
            non_manifold_edges = report.non_manifold_edge_count,
            is_inside_out = report.is_inside_out,
            "Mesh validation found issues"
        );
    }
}

/// Log a repair operation result.
pub fn log_repair_result(operation: &str, items_fixed: usize, elapsed_ms: f64) {
    info!(
        target: "mesh_repair::repair",
        operation = operation,
        items_fixed = items_fixed,
        elapsed_ms = format!("{:.2}", elapsed_ms),
        "Repair operation completed"
    );
}

/// Log progress for a long-running operation.
pub fn log_progress(operation: &str, current: usize, total: usize, stage: Option<&str>) {
    let percent = if total > 0 {
        (current as f64 / total as f64 * 100.0) as u32
    } else {
        0
    };

    debug!(
        target: "mesh_repair::progress",
        operation = operation,
        current = current,
        total = total,
        percent = percent,
        stage = stage.unwrap_or("processing"),
        "Progress update"
    );
}

/// Log a file I/O operation.
pub fn log_io_operation(
    operation: &str,
    path: &std::path::Path,
    format: Option<&str>,
    success: bool,
) {
    if success {
        info!(
            target: "mesh_repair::io",
            operation = operation,
            path = path.display().to_string(),
            format = format.unwrap_or("auto"),
            "I/O operation completed"
        );
    } else {
        warn!(
            target: "mesh_repair::io",
            operation = operation,
            path = path.display().to_string(),
            format = format.unwrap_or("auto"),
            "I/O operation failed"
        );
    }
}

/// Log a performance-critical section.
///
/// Returns a guard that logs when dropped.
#[must_use]
pub fn log_perf_section(name: &'static str) -> impl Drop {
    struct PerfGuard {
        name: &'static str,
        start: Instant,
    }
    impl Drop for PerfGuard {
        fn drop(&mut self) {
            let elapsed = self.start.elapsed();
            trace!(
                target: "mesh_repair::perf",
                section = self.name,
                elapsed_us = elapsed.as_micros(),
                "Performance section completed"
            );
        }
    }
    PerfGuard {
        name,
        start: Instant::now(),
    }
}

/// Macro for creating instrumented mesh operation spans.
///
/// This macro creates a tracing span with common mesh operation fields.
#[macro_export]
macro_rules! mesh_span {
    ($name:expr, $mesh:expr) => {
        tracing::info_span!(
            $name,
            vertices = $mesh.vertex_count(),
            faces = $mesh.face_count()
        )
    };
    ($name:expr, $mesh:expr, $($field:tt)*) => {
        tracing::info_span!(
            $name,
            vertices = $mesh.vertex_count(),
            faces = $mesh.face_count(),
            $($field)*
        )
    };
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::Mesh;

    #[test]
    fn test_operation_timer() {
        let timer = OperationTimer::new("test_operation");
        std::thread::sleep(std::time::Duration::from_millis(10));
        assert!(timer.elapsed_ms() >= 10.0);
    }

    #[test]
    fn test_log_mesh_stats() {
        let mesh = Mesh::new();
        // Just verify it doesn't panic
        log_mesh_stats(&mesh, "test");
        log_mesh_stats_detailed(&mesh, "test");
    }
}

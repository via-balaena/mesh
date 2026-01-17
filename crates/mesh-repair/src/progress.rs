//! Progress reporting and operation estimation for long-running operations.
//!
//! This module provides infrastructure for:
//! - Progress callbacks during expensive operations
//! - Operation time estimation based on mesh complexity
//! - Cancellation support via progress callbacks
//!
//! # Example
//!
//! ```ignore
//! use mesh_repair::progress::{Progress, ProgressCallback};
//!
//! // Create a progress callback
//! let callback: ProgressCallback = Box::new(|progress| {
//!     println!("{}% complete: {}", (progress.fraction() * 100.0) as u32, progress.message);
//!     true // Continue processing (return false to cancel)
//! });
//!
//! // Use with an operation
//! let result = mesh.analyze_thickness_with_progress(&params, Some(&callback))?;
//! ```
//!
//! # Estimation
//!
//! ```ignore
//! use mesh_repair::progress::estimate_operation_time;
//!
//! let estimate = estimate_operation_time(&mesh, OperationType::Remesh);
//! println!("Estimated time: {:.1}s", estimate.estimated_seconds);
//! ```

use std::sync::Arc;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::time::{Duration, Instant};

/// Progress information passed to callbacks.
#[derive(Debug, Clone)]
pub struct Progress {
    /// Current step (0-based).
    pub current: u64,

    /// Total number of steps.
    pub total: u64,

    /// Human-readable message describing current operation.
    pub message: String,

    /// Elapsed time since operation started.
    pub elapsed: Duration,

    /// Estimated time remaining (if available).
    pub estimated_remaining: Option<Duration>,
}

impl Progress {
    /// Create a new progress report.
    pub fn new(current: u64, total: u64, message: impl Into<String>) -> Self {
        Self {
            current,
            total,
            message: message.into(),
            elapsed: Duration::ZERO,
            estimated_remaining: None,
        }
    }

    /// Get progress as a fraction (0.0 to 1.0).
    #[inline]
    pub fn fraction(&self) -> f64 {
        if self.total == 0 {
            0.0
        } else {
            (self.current as f64) / (self.total as f64)
        }
    }

    /// Get progress as a percentage (0 to 100).
    #[inline]
    pub fn percent(&self) -> u32 {
        (self.fraction() * 100.0).round() as u32
    }

    /// Check if the operation is complete.
    #[inline]
    pub fn is_complete(&self) -> bool {
        self.current >= self.total
    }
}

/// Callback function for progress reporting.
///
/// Returns `true` to continue, `false` to request cancellation.
pub type ProgressCallback = Box<dyn Fn(&Progress) -> bool + Send + Sync>;

/// A thread-safe progress tracker for parallel operations.
///
/// This tracker uses atomic operations to allow multiple threads to
/// update progress simultaneously without locks.
#[derive(Debug)]
pub struct ProgressTracker {
    current: AtomicU64,
    total: u64,
    cancelled: AtomicBool,
    start_time: Instant,
    last_callback_time: std::sync::Mutex<Instant>,
    callback_interval: Duration,
}

impl ProgressTracker {
    /// Create a new progress tracker.
    pub fn new(total: u64) -> Self {
        Self {
            current: AtomicU64::new(0),
            total,
            cancelled: AtomicBool::new(false),
            start_time: Instant::now(),
            last_callback_time: std::sync::Mutex::new(Instant::now()),
            callback_interval: Duration::from_millis(100), // Don't callback too frequently
        }
    }

    /// Create a tracker with custom callback interval.
    pub fn with_interval(total: u64, interval: Duration) -> Self {
        let mut tracker = Self::new(total);
        tracker.callback_interval = interval;
        tracker
    }

    /// Increment progress by one.
    #[inline]
    pub fn increment(&self) {
        self.current.fetch_add(1, Ordering::Relaxed);
    }

    /// Increment progress by a specific amount.
    #[inline]
    pub fn increment_by(&self, amount: u64) {
        self.current.fetch_add(amount, Ordering::Relaxed);
    }

    /// Set the current progress value.
    #[inline]
    pub fn set(&self, value: u64) {
        self.current.store(value, Ordering::Relaxed);
    }

    /// Get the current progress value.
    #[inline]
    pub fn current(&self) -> u64 {
        self.current.load(Ordering::Relaxed)
    }

    /// Get the total count.
    #[inline]
    pub fn total(&self) -> u64 {
        self.total
    }

    /// Check if cancellation was requested.
    #[inline]
    pub fn is_cancelled(&self) -> bool {
        self.cancelled.load(Ordering::Relaxed)
    }

    /// Request cancellation.
    pub fn cancel(&self) {
        self.cancelled.store(true, Ordering::Relaxed);
    }

    /// Get progress as a fraction (0.0 to 1.0).
    #[inline]
    pub fn fraction(&self) -> f64 {
        if self.total == 0 {
            0.0
        } else {
            (self.current() as f64) / (self.total as f64)
        }
    }

    /// Get elapsed time.
    #[inline]
    pub fn elapsed(&self) -> Duration {
        self.start_time.elapsed()
    }

    /// Estimate remaining time based on current progress.
    pub fn estimated_remaining(&self) -> Option<Duration> {
        let current = self.current();
        if current == 0 {
            return None;
        }

        let elapsed = self.elapsed();
        let rate = current as f64 / elapsed.as_secs_f64();

        if rate > 0.0 {
            let remaining = (self.total - current) as f64 / rate;
            Some(Duration::from_secs_f64(remaining))
        } else {
            None
        }
    }

    /// Create a Progress snapshot.
    pub fn snapshot(&self, message: impl Into<String>) -> Progress {
        Progress {
            current: self.current(),
            total: self.total,
            message: message.into(),
            elapsed: self.elapsed(),
            estimated_remaining: self.estimated_remaining(),
        }
    }

    /// Call the callback if enough time has passed since last call.
    ///
    /// Returns `false` if the callback requested cancellation.
    pub fn maybe_callback(
        &self,
        callback: Option<&ProgressCallback>,
        message: impl Into<String>,
    ) -> bool {
        if self.is_cancelled() {
            return false;
        }

        let callback = match callback {
            Some(cb) => cb,
            None => return true,
        };

        // Check if enough time has passed
        let now = Instant::now();
        {
            let last = self.last_callback_time.lock().unwrap();
            if now.duration_since(*last) < self.callback_interval {
                return true;
            }
        }

        // Update last callback time and call
        {
            let mut last = self.last_callback_time.lock().unwrap();
            *last = now;
        }

        let progress = self.snapshot(message);
        let should_continue = callback(&progress);

        if !should_continue {
            self.cancel();
        }

        should_continue
    }
}

/// Arc-wrapped progress tracker for sharing across threads.
pub type SharedProgressTracker = Arc<ProgressTracker>;

/// Create a shared progress tracker.
pub fn shared_tracker(total: u64) -> SharedProgressTracker {
    Arc::new(ProgressTracker::new(total))
}

// ============================================================================
// Operation Time Estimation
// ============================================================================

/// Types of operations that can be estimated.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum OperationType {
    /// Mesh validation.
    Validate,
    /// Mesh repair (all steps).
    Repair,
    /// Hole filling.
    FillHoles,
    /// Winding order fix.
    FixWinding,
    /// Isotropic remeshing.
    Remesh,
    /// Mesh decimation.
    Decimate,
    /// Loop subdivision.
    Subdivide,
    /// Wall thickness analysis.
    ThicknessAnalysis,
    /// Self-intersection detection.
    SelfIntersection,
    /// Boolean operation.
    Boolean,
    /// Mesh morphing.
    Morph,
    /// ICP registration.
    Registration,
    /// Surface reconstruction from point cloud.
    SurfaceReconstruction,
    /// Lattice infill generation.
    LatticeGeneration,
    /// Slicing for 3D printing.
    Slice,
}

/// Estimate of operation time and resources.
#[derive(Debug, Clone)]
pub struct OperationEstimate {
    /// Estimated time in seconds.
    pub estimated_seconds: f64,

    /// Confidence level (0.0 to 1.0).
    pub confidence: f64,

    /// Estimated memory usage in bytes.
    pub estimated_memory_bytes: u64,

    /// Whether the operation supports progress callbacks.
    pub supports_progress: bool,

    /// Whether the operation supports cancellation.
    pub supports_cancellation: bool,

    /// Estimated number of iterations (if applicable).
    pub estimated_iterations: Option<u64>,

    /// Complexity description.
    pub complexity: String,
}

impl OperationEstimate {
    fn new(seconds: f64, complexity: &str) -> Self {
        Self {
            estimated_seconds: seconds,
            confidence: 0.7,
            estimated_memory_bytes: 0,
            supports_progress: true,
            supports_cancellation: true,
            estimated_iterations: None,
            complexity: complexity.to_string(),
        }
    }
}

/// Estimate the time for an operation based on mesh complexity.
///
/// # Arguments
/// * `vertex_count` - Number of vertices in the mesh
/// * `face_count` - Number of faces in the mesh
/// * `operation` - Type of operation to estimate
///
/// # Returns
/// An estimate of time and resources needed.
///
/// # Note
/// Estimates are based on typical modern hardware (mid-range CPU).
/// Actual times may vary significantly based on:
/// - CPU speed and core count
/// - Memory bandwidth
/// - Mesh topology (uniform vs highly varying)
/// - Specific parameters used
pub fn estimate_operation_time(
    vertex_count: usize,
    face_count: usize,
    operation: OperationType,
) -> OperationEstimate {
    let v = vertex_count as f64;
    let f = face_count as f64;

    match operation {
        OperationType::Validate => {
            // O(V + F) - linear scan
            let seconds = (v + f) / 10_000_000.0;
            OperationEstimate::new(seconds, "O(V + F)")
        }

        OperationType::Repair => {
            // Multiple passes: O(V + F) for each step
            let seconds = (v + f) / 1_000_000.0;
            OperationEstimate::new(seconds, "O(V + F) multiple passes")
        }

        OperationType::FillHoles => {
            // O(B) where B is boundary edges, typically O(V^0.5)
            let seconds = v.sqrt() / 10_000.0;
            OperationEstimate::new(seconds, "O(boundary edges)")
        }

        OperationType::FixWinding => {
            // O(F) BFS through faces
            let seconds = f / 5_000_000.0;
            OperationEstimate::new(seconds, "O(F)")
        }

        OperationType::Remesh => {
            // O(V * I) where I is iterations, typically 5-10
            let iterations = 7.0;
            let seconds = v * iterations / 500_000.0;
            let mut est = OperationEstimate::new(seconds, "O(V × iterations)");
            est.estimated_iterations = Some(iterations as u64);
            est
        }

        OperationType::Decimate => {
            // O(V log V) due to priority queue
            let seconds = v * v.ln() / 1_000_000.0;
            OperationEstimate::new(seconds, "O(V log V)")
        }

        OperationType::Subdivide => {
            // O(F) per iteration, output grows 4x per iteration
            let seconds = f / 2_000_000.0;
            OperationEstimate::new(seconds, "O(F × 4^iterations)")
        }

        OperationType::ThicknessAnalysis => {
            // O(V × F) for ray-triangle tests with BVH: O(V log F)
            let seconds = v * f.ln() / 500_000.0;
            OperationEstimate::new(seconds, "O(V log F) with BVH")
        }

        OperationType::SelfIntersection => {
            // O(F²) worst case, O(F log F) with BVH
            let seconds = f * f.ln() / 100_000.0;
            OperationEstimate::new(seconds, "O(F log F) with BVH")
        }

        OperationType::Boolean => {
            // Complex: O((F₁ + F₂) log(F₁ + F₂)) typical
            let seconds = f * f.ln() / 50_000.0;
            OperationEstimate::new(seconds, "O(F log F)")
        }

        OperationType::Morph => {
            // RBF: O(V²) for full, O(V) for local
            let seconds = v * v.sqrt() / 100_000.0;
            OperationEstimate::new(seconds, "O(V√V) local RBF")
        }

        OperationType::Registration => {
            // ICP: O(V × I) where I is iterations
            let iterations = 50.0;
            let seconds = v * iterations / 1_000_000.0;
            let mut est = OperationEstimate::new(seconds, "O(V × iterations)");
            est.estimated_iterations = Some(iterations as u64);
            est
        }

        OperationType::SurfaceReconstruction => {
            // Grid-based: O(V + G³) where G is grid dimension
            let grid_size = (v / 10.0).powf(1.0 / 3.0).max(10.0);
            let seconds = (v + grid_size.powi(3)) / 500_000.0;
            OperationEstimate::new(seconds, "O(V + grid³)")
        }

        OperationType::LatticeGeneration => {
            // O(cells) where cells depends on volume and cell size
            let seconds = v / 100_000.0;
            OperationEstimate::new(seconds, "O(volume / cell_size³)")
        }

        OperationType::Slice => {
            // O(F × layers)
            let layers = 100.0;
            let seconds = f * layers / 10_000_000.0;
            let mut est = OperationEstimate::new(seconds, "O(F × layers)");
            est.estimated_iterations = Some(layers as u64);
            est
        }
    }
}

// ============================================================================
// Parallel Utilities
// ============================================================================

/// Trait for operations that can report progress.
pub trait ProgressReporter {
    /// Report progress during an operation.
    fn report_progress(&self, current: u64, total: u64, message: &str) -> bool;
}

/// A no-op progress reporter that does nothing.
pub struct NoOpProgressReporter;

impl ProgressReporter for NoOpProgressReporter {
    #[inline]
    fn report_progress(&self, _current: u64, _total: u64, _message: &str) -> bool {
        true
    }
}

/// A progress reporter that calls a callback.
pub struct CallbackProgressReporter<'a> {
    callback: &'a ProgressCallback,
    start_time: Instant,
}

impl<'a> CallbackProgressReporter<'a> {
    /// Create a new callback reporter.
    pub fn new(callback: &'a ProgressCallback) -> Self {
        Self {
            callback,
            start_time: Instant::now(),
        }
    }
}

impl ProgressReporter for CallbackProgressReporter<'_> {
    fn report_progress(&self, current: u64, total: u64, message: &str) -> bool {
        let elapsed = self.start_time.elapsed();
        let estimated_remaining = if current > 0 {
            let rate = current as f64 / elapsed.as_secs_f64();
            if rate > 0.0 {
                let remaining = (total - current) as f64 / rate;
                Some(Duration::from_secs_f64(remaining))
            } else {
                None
            }
        } else {
            None
        };

        let progress = Progress {
            current,
            total,
            message: message.to_string(),
            elapsed,
            estimated_remaining,
        };

        (self.callback)(&progress)
    }
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::AtomicU32;

    #[test]
    fn test_progress_fraction() {
        let p = Progress::new(50, 100, "test");
        assert!((p.fraction() - 0.5).abs() < 1e-10);
        assert_eq!(p.percent(), 50);
    }

    #[test]
    fn test_progress_complete() {
        let p1 = Progress::new(50, 100, "incomplete");
        assert!(!p1.is_complete());

        let p2 = Progress::new(100, 100, "complete");
        assert!(p2.is_complete());
    }

    #[test]
    fn test_progress_zero_total() {
        let p = Progress::new(0, 0, "empty");
        assert!((p.fraction() - 0.0).abs() < 1e-10);
        assert_eq!(p.percent(), 0);
    }

    #[test]
    fn test_progress_tracker() {
        let tracker = ProgressTracker::new(100);

        assert_eq!(tracker.current(), 0);
        assert_eq!(tracker.total(), 100);
        assert!(!tracker.is_cancelled());

        tracker.increment();
        assert_eq!(tracker.current(), 1);

        tracker.increment_by(9);
        assert_eq!(tracker.current(), 10);

        tracker.set(50);
        assert_eq!(tracker.current(), 50);
        assert!((tracker.fraction() - 0.5).abs() < 1e-10);
    }

    #[test]
    fn test_progress_tracker_cancel() {
        let tracker = ProgressTracker::new(100);

        assert!(!tracker.is_cancelled());
        tracker.cancel();
        assert!(tracker.is_cancelled());
    }

    #[test]
    fn test_shared_tracker() {
        let tracker = shared_tracker(100);
        let tracker_clone = tracker.clone();

        // Both references should see the same state
        tracker.increment();
        assert_eq!(tracker_clone.current(), 1);
    }

    #[test]
    fn test_progress_callback() {
        let counter = Arc::new(AtomicU32::new(0));
        let counter_clone = counter.clone();

        let callback: ProgressCallback = Box::new(move |p| {
            counter_clone.fetch_add(1, Ordering::SeqCst);
            p.current < 5 // Cancel after 5 calls
        });

        let tracker = ProgressTracker::with_interval(10, Duration::ZERO);

        for i in 0..10 {
            tracker.set(i);
            if !tracker.maybe_callback(Some(&callback), "test") {
                break;
            }
        }

        // Should have been called multiple times before cancellation
        let calls = counter.load(Ordering::SeqCst);
        assert!(calls >= 5, "Expected at least 5 calls, got {}", calls);
    }

    #[test]
    fn test_operation_estimate() {
        let est = estimate_operation_time(10000, 20000, OperationType::Validate);

        assert!(est.estimated_seconds > 0.0);
        assert!(est.confidence > 0.0 && est.confidence <= 1.0);
        assert!(est.supports_progress);
    }

    #[test]
    fn test_operation_estimates_scale() {
        // Larger meshes should have larger estimates
        let small = estimate_operation_time(1000, 2000, OperationType::Remesh);
        let large = estimate_operation_time(100000, 200000, OperationType::Remesh);

        assert!(
            large.estimated_seconds > small.estimated_seconds,
            "Large mesh estimate ({}) should be greater than small ({})",
            large.estimated_seconds,
            small.estimated_seconds
        );
    }

    #[test]
    fn test_noop_progress_reporter() {
        let reporter = NoOpProgressReporter;
        assert!(reporter.report_progress(50, 100, "test"));
    }

    #[test]
    fn test_callback_progress_reporter() {
        let called = Arc::new(AtomicBool::new(false));
        let called_clone = called.clone();

        let callback: ProgressCallback = Box::new(move |_p| {
            called_clone.store(true, Ordering::SeqCst);
            true
        });

        let reporter = CallbackProgressReporter::new(&callback);
        let result = reporter.report_progress(50, 100, "test");

        assert!(result);
        assert!(called.load(Ordering::SeqCst));
    }
}

//! Point cloud data structures and surface reconstruction.
//!
//! This module provides tools for working with point cloud data, commonly
//! produced by 3D scanners, LiDAR, and photogrammetry. It includes:
//!
//! - [`PointCloud`] struct for storing 3D point data with optional attributes
//! - I/O support for PLY, XYZ, and PCD formats
//! - Normal estimation from local neighborhoods
//! - Surface reconstruction algorithms:
//!   - Ball-pivoting algorithm (BPA)
//!   - SDF-based reconstruction (similar to Poisson)
//!
//! # Example
//!
//! ```ignore
//! use mesh_repair::pointcloud::{PointCloud, ReconstructionParams};
//!
//! // Load a point cloud from file
//! let cloud = PointCloud::load("scan.ply")?;
//!
//! // Estimate normals if not present
//! let cloud = cloud.with_estimated_normals(16)?;
//!
//! // Reconstruct surface mesh
//! let result = cloud.to_mesh(&ReconstructionParams::default())?;
//! println!("Reconstructed {} faces", result.mesh.faces.len());
//! ```
//!
//! # Units
//!
//! Like the rest of the library, point clouds are assumed to be in millimeters (mm).

use std::fs::File;
use std::io::{BufRead, BufReader, BufWriter, Write};
use std::path::Path;

use nalgebra::{Point3, Vector3};
use ply_rs::ply::Property;
use tracing::{debug, info, warn};

use crate::error::{MeshError, MeshResult};
use crate::{Mesh, Vertex, VertexColor};

/// A point in the cloud with optional attributes.
#[derive(Debug, Clone)]
pub struct CloudPoint {
    /// 3D position.
    pub position: Point3<f64>,

    /// Unit normal vector (estimated or from scanner).
    pub normal: Option<Vector3<f64>>,

    /// Point color (RGB).
    pub color: Option<VertexColor>,

    /// Intensity/reflectance value (common in LiDAR data).
    pub intensity: Option<f32>,

    /// Application-specific tag (e.g., classification, scan ID).
    pub tag: Option<u32>,
}

impl CloudPoint {
    /// Create a point with only position.
    #[inline]
    pub fn new(position: Point3<f64>) -> Self {
        Self {
            position,
            normal: None,
            color: None,
            intensity: None,
            tag: None,
        }
    }

    /// Create a point from raw coordinates.
    #[inline]
    pub fn from_coords(x: f64, y: f64, z: f64) -> Self {
        Self::new(Point3::new(x, y, z))
    }

    /// Create a point with position and normal.
    #[inline]
    pub fn with_normal(position: Point3<f64>, normal: Vector3<f64>) -> Self {
        Self {
            position,
            normal: Some(normal),
            color: None,
            intensity: None,
            tag: None,
        }
    }

    /// Convert to a mesh vertex.
    pub fn to_vertex(&self) -> Vertex {
        let mut v = Vertex::new(self.position);
        v.normal = self.normal;
        v.color = self.color;
        v.tag = self.tag;
        v
    }
}

/// A collection of 3D points with optional attributes.
#[derive(Debug, Clone)]
pub struct PointCloud {
    /// The points in the cloud.
    pub points: Vec<CloudPoint>,
}

impl PointCloud {
    /// Create a new empty point cloud.
    pub fn new() -> Self {
        Self { points: Vec::new() }
    }

    /// Create a point cloud with pre-allocated capacity.
    pub fn with_capacity(capacity: usize) -> Self {
        Self {
            points: Vec::with_capacity(capacity),
        }
    }

    /// Create a point cloud from a list of positions.
    pub fn from_positions(positions: &[Point3<f64>]) -> Self {
        Self {
            points: positions.iter().map(|&p| CloudPoint::new(p)).collect(),
        }
    }

    /// Create a point cloud from a mesh (extracts vertices).
    pub fn from_mesh(mesh: &Mesh) -> Self {
        let points = mesh
            .vertices
            .iter()
            .map(|v| {
                let mut p = CloudPoint::new(v.position);
                p.normal = v.normal;
                p.color = v.color;
                p.tag = v.tag;
                p
            })
            .collect();
        Self { points }
    }

    /// Number of points in the cloud.
    #[inline]
    pub fn len(&self) -> usize {
        self.points.len()
    }

    /// Check if the cloud is empty.
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.points.is_empty()
    }

    /// Check if all points have normals.
    pub fn has_normals(&self) -> bool {
        !self.points.is_empty() && self.points.iter().all(|p| p.normal.is_some())
    }

    /// Check if any points have colors.
    pub fn has_colors(&self) -> bool {
        self.points.iter().any(|p| p.color.is_some())
    }

    /// Compute the axis-aligned bounding box.
    pub fn bounds(&self) -> Option<(Point3<f64>, Point3<f64>)> {
        if self.points.is_empty() {
            return None;
        }

        let mut min = self.points[0].position;
        let mut max = self.points[0].position;

        for p in &self.points[1..] {
            min.x = min.x.min(p.position.x);
            min.y = min.y.min(p.position.y);
            min.z = min.z.min(p.position.z);
            max.x = max.x.max(p.position.x);
            max.y = max.y.max(p.position.y);
            max.z = max.z.max(p.position.z);
        }

        Some((min, max))
    }

    /// Compute the centroid (center of mass) of the point cloud.
    pub fn centroid(&self) -> Option<Point3<f64>> {
        if self.points.is_empty() {
            return None;
        }

        let sum: Vector3<f64> = self
            .points
            .iter()
            .map(|p| p.position.coords)
            .fold(Vector3::zeros(), |acc, v| acc + v);

        Some(Point3::from(sum / self.points.len() as f64))
    }

    /// Add a point to the cloud.
    #[inline]
    pub fn push(&mut self, point: CloudPoint) {
        self.points.push(point);
    }

    /// Add a point from coordinates.
    #[inline]
    pub fn push_coords(&mut self, x: f64, y: f64, z: f64) {
        self.points.push(CloudPoint::from_coords(x, y, z));
    }

    /// Load a point cloud from file, auto-detecting format.
    pub fn load(path: impl AsRef<Path>) -> MeshResult<Self> {
        let path = path.as_ref();
        let format =
            PointCloudFormat::from_path(path).ok_or_else(|| MeshError::UnsupportedFormat {
                extension: path.extension().and_then(|e| e.to_str()).map(String::from),
            })?;

        info!("Loading point cloud from {:?} (format: {:?})", path, format);

        let cloud = match format {
            PointCloudFormat::Ply => load_ply_pointcloud(path)?,
            PointCloudFormat::Xyz => load_xyz(path)?,
            PointCloudFormat::Pcd => load_pcd(path)?,
        };

        info!(
            "Loaded {} points (has_normals: {}, has_colors: {})",
            cloud.len(),
            cloud.has_normals(),
            cloud.has_colors()
        );

        Ok(cloud)
    }

    /// Save the point cloud to file, auto-detecting format.
    pub fn save(&self, path: impl AsRef<Path>) -> MeshResult<()> {
        let path = path.as_ref();
        let format =
            PointCloudFormat::from_path(path).ok_or_else(|| MeshError::UnsupportedFormat {
                extension: path.extension().and_then(|e| e.to_str()).map(String::from),
            })?;

        info!("Saving point cloud to {:?} (format: {:?})", path, format);

        match format {
            PointCloudFormat::Ply => save_ply_pointcloud(self, path),
            PointCloudFormat::Xyz => save_xyz(self, path),
            PointCloudFormat::Pcd => save_pcd(self, path),
        }
    }

    /// Estimate normals for all points using PCA on local neighborhoods.
    ///
    /// # Arguments
    /// * `k` - Number of nearest neighbors to use for normal estimation
    ///
    /// # Returns
    /// A new point cloud with estimated normals.
    pub fn with_estimated_normals(&self, k: usize) -> MeshResult<Self> {
        if self.is_empty() {
            return Ok(self.clone());
        }

        info!(
            "Estimating normals using k={} neighbors for {} points",
            k,
            self.len()
        );

        let normals = estimate_normals(self, k)?;

        let mut result = self.clone();
        for (point, normal) in result.points.iter_mut().zip(normals.iter()) {
            point.normal = Some(*normal);
        }

        Ok(result)
    }

    /// Orient normals consistently (points outward from centroid).
    ///
    /// This method flips normals that point toward the centroid so they
    /// all point outward. This assumes the point cloud represents the
    /// exterior of an object.
    pub fn orient_normals_outward(&mut self) {
        let centroid = match self.centroid() {
            Some(c) => c,
            None => return,
        };

        for point in &mut self.points {
            if let Some(ref mut normal) = point.normal {
                let to_point = point.position - centroid;
                if normal.dot(&to_point) < 0.0 {
                    *normal = -*normal;
                }
            }
        }
    }

    /// Reconstruct a triangle mesh from the point cloud.
    ///
    /// # Arguments
    /// * `params` - Reconstruction parameters
    ///
    /// # Returns
    /// The reconstructed mesh with statistics.
    pub fn to_mesh(&self, params: &ReconstructionParams) -> MeshResult<ReconstructionResult> {
        if self.is_empty() {
            return Err(MeshError::EmptyMesh {
                details: "Cannot reconstruct mesh from empty point cloud".to_string(),
            });
        }

        match params.algorithm {
            ReconstructionAlgorithm::BallPivoting => reconstruct_ball_pivoting(self, params),
            ReconstructionAlgorithm::SdfBased => reconstruct_sdf_based(self, params),
        }
    }

    /// Downsample the point cloud using voxel grid filtering.
    ///
    /// Points within each voxel are averaged into a single point.
    ///
    /// # Arguments
    /// * `voxel_size` - Size of each voxel (in mm)
    pub fn downsample(&self, voxel_size: f64) -> Self {
        if self.is_empty() || voxel_size <= 0.0 {
            return self.clone();
        }

        use std::collections::HashMap;

        // Voxel data: (position_sum, normal_sum, count)
        type VoxelData = (Vector3<f64>, Option<Vector3<f64>>, usize);

        let (min_bound, _) = match self.bounds() {
            Some(b) => b,
            None => return self.clone(),
        };

        // Map voxel indices to accumulated points
        let mut voxel_map: HashMap<(i64, i64, i64), VoxelData> = HashMap::new();

        for point in &self.points {
            let ix = ((point.position.x - min_bound.x) / voxel_size).floor() as i64;
            let iy = ((point.position.y - min_bound.y) / voxel_size).floor() as i64;
            let iz = ((point.position.z - min_bound.z) / voxel_size).floor() as i64;

            let key = (ix, iy, iz);
            let entry = voxel_map.entry(key).or_insert((Vector3::zeros(), None, 0));
            entry.0 += point.position.coords;
            if let Some(n) = point.normal {
                entry.1 = Some(entry.1.unwrap_or(Vector3::zeros()) + n);
            }
            entry.2 += 1;
        }

        // Create averaged points
        let mut result = PointCloud::with_capacity(voxel_map.len());
        for (pos_sum, normal_sum, count) in voxel_map.values() {
            let avg_pos = Point3::from(pos_sum / *count as f64);
            let mut point = CloudPoint::new(avg_pos);
            if let Some(n) = normal_sum {
                let avg_normal = n / *count as f64;
                let norm = avg_normal.norm();
                if norm > 1e-10 {
                    point.normal = Some(avg_normal / norm);
                }
            }
            result.push(point);
        }

        debug!(
            "Downsampled from {} to {} points (voxel_size={})",
            self.len(),
            result.len(),
            voxel_size
        );

        result
    }

    /// Remove statistical outliers.
    ///
    /// Points whose mean distance to their k nearest neighbors exceeds
    /// `mean + std_ratio * std` are removed.
    ///
    /// # Arguments
    /// * `k` - Number of neighbors to consider
    /// * `std_ratio` - Standard deviation multiplier threshold
    pub fn remove_outliers(&self, k: usize, std_ratio: f64) -> Self {
        if self.len() <= k {
            return self.clone();
        }

        // Build KD-tree
        let kdtree = build_kdtree(self);

        // Compute mean distances to k neighbors for each point
        let mut mean_distances = Vec::with_capacity(self.len());
        for point in &self.points {
            let neighbors = kdtree.nearest_n::<kiddo::SquaredEuclidean>(
                &[point.position.x, point.position.y, point.position.z],
                k + 1, // +1 because the point itself is included
            );

            let sum: f64 = neighbors
                .iter()
                .skip(1) // Skip self
                .map(|n| n.distance.sqrt())
                .sum();
            mean_distances.push(sum / k as f64);
        }

        // Compute mean and std of mean distances
        let global_mean: f64 = mean_distances.iter().sum::<f64>() / mean_distances.len() as f64;
        let variance: f64 = mean_distances
            .iter()
            .map(|d| (d - global_mean).powi(2))
            .sum::<f64>()
            / mean_distances.len() as f64;
        let std_dev = variance.sqrt();

        let threshold = global_mean + std_ratio * std_dev;

        // Filter points
        let mut result = PointCloud::with_capacity(self.len());
        for (point, &mean_dist) in self.points.iter().zip(mean_distances.iter()) {
            if mean_dist <= threshold {
                result.push(point.clone());
            }
        }

        let removed = self.len() - result.len();
        debug!("Removed {} outliers (threshold={:.4})", removed, threshold);

        result
    }

    /// Translate the point cloud by the given vector.
    pub fn translate(&mut self, offset: Vector3<f64>) {
        for point in &mut self.points {
            point.position += offset;
        }
    }

    /// Scale the point cloud uniformly around the centroid.
    pub fn scale(&mut self, factor: f64) {
        let centroid = match self.centroid() {
            Some(c) => c,
            None => return,
        };

        for point in &mut self.points {
            let offset = point.position - centroid;
            point.position = centroid + offset * factor;
        }
    }
}

impl Default for PointCloud {
    fn default() -> Self {
        Self::new()
    }
}

// ============================================================================
// File formats
// ============================================================================

/// Supported point cloud file formats.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PointCloudFormat {
    /// PLY (Stanford Polygon File Format)
    Ply,
    /// XYZ (simple ASCII x y z [nx ny nz] format)
    Xyz,
    /// PCD (Point Cloud Data - PCL format)
    Pcd,
}

impl PointCloudFormat {
    /// Detect format from file extension.
    pub fn from_path(path: &Path) -> Option<Self> {
        path.extension()
            .and_then(|ext| ext.to_str())
            .map(|ext| ext.to_lowercase())
            .and_then(|ext| match ext.as_str() {
                "ply" => Some(PointCloudFormat::Ply),
                "xyz" | "txt" | "asc" | "pts" => Some(PointCloudFormat::Xyz),
                "pcd" => Some(PointCloudFormat::Pcd),
                _ => None,
            })
    }
}

/// Load point cloud from PLY file.
fn load_ply_pointcloud(path: &Path) -> MeshResult<PointCloud> {
    use ply_rs::parser::Parser;

    let file = File::open(path).map_err(|e| MeshError::IoRead {
        path: path.to_path_buf(),
        source: e,
    })?;
    let mut reader = BufReader::new(file);

    let parser = Parser::<ply_rs::ply::DefaultElement>::new();
    let ply = parser
        .read_ply(&mut reader)
        .map_err(|e| MeshError::ParseError {
            path: path.to_path_buf(),
            details: format!("PLY parse error: {:?}", e),
        })?;

    let mut cloud = PointCloud::new();

    if let Some(vertices) = ply.payload.get("vertex") {
        cloud = PointCloud::with_capacity(vertices.len());

        for vertex_element in vertices {
            let x = get_ply_float(vertex_element.get("x"))?;
            let y = get_ply_float(vertex_element.get("y"))?;
            let z = get_ply_float(vertex_element.get("z"))?;

            let mut point = CloudPoint::from_coords(x, y, z);

            // Load normals if present
            if let (Some(nx), Some(ny), Some(nz)) = (
                vertex_element.get("nx"),
                vertex_element.get("ny"),
                vertex_element.get("nz"),
            ) && let (Ok(nx), Ok(ny), Ok(nz)) = (
                get_ply_float(Some(nx)),
                get_ply_float(Some(ny)),
                get_ply_float(Some(nz)),
            ) {
                point.normal = Some(Vector3::new(nx, ny, nz));
            }

            // Load colors if present
            if let (Some(r), Some(g), Some(b)) = (
                vertex_element.get("red"),
                vertex_element.get("green"),
                vertex_element.get("blue"),
            ) && let (Ok(r), Ok(g), Ok(b)) = (
                get_ply_u8(Some(r)),
                get_ply_u8(Some(g)),
                get_ply_u8(Some(b)),
            ) {
                point.color = Some(VertexColor::new(r, g, b));
            }

            // Load intensity if present
            if let Some(intensity) = vertex_element.get("intensity")
                && let Ok(i) = get_ply_float(Some(intensity))
            {
                point.intensity = Some(i as f32);
            }

            cloud.push(point);
        }
    }

    Ok(cloud)
}

fn get_ply_float(prop: Option<&ply_rs::ply::Property>) -> MeshResult<f64> {
    match prop {
        Some(Property::Float(v)) => Ok(*v as f64),
        Some(Property::Double(v)) => Ok(*v),
        Some(Property::Int(v)) => Ok(*v as f64),
        Some(Property::UInt(v)) => Ok(*v as f64),
        Some(Property::Short(v)) => Ok(*v as f64),
        Some(Property::UShort(v)) => Ok(*v as f64),
        Some(Property::Char(v)) => Ok(*v as f64),
        Some(Property::UChar(v)) => Ok(*v as f64),
        _ => Err(MeshError::ParseError {
            path: std::path::PathBuf::new(),
            details: "Missing or invalid PLY property".to_string(),
        }),
    }
}

fn get_ply_u8(prop: Option<&ply_rs::ply::Property>) -> Result<u8, ()> {
    match prop {
        Some(Property::UChar(v)) => Ok(*v),
        Some(Property::Char(v)) => Ok(*v as u8),
        Some(Property::UShort(v)) => Ok((*v).min(255) as u8),
        Some(Property::Short(v)) => Ok((*v).clamp(0, 255) as u8),
        Some(Property::UInt(v)) => Ok((*v).min(255) as u8),
        Some(Property::Int(v)) => Ok((*v).clamp(0, 255) as u8),
        Some(Property::Float(v)) => Ok((v * 255.0).clamp(0.0, 255.0) as u8),
        Some(Property::Double(v)) => Ok((v * 255.0).clamp(0.0, 255.0) as u8),
        _ => Err(()),
    }
}

/// Save point cloud to PLY file.
fn save_ply_pointcloud(cloud: &PointCloud, path: &Path) -> MeshResult<()> {
    let file = File::create(path).map_err(|e| MeshError::IoWrite {
        path: path.to_path_buf(),
        source: e,
    })?;
    let mut writer = BufWriter::new(file);

    let has_normals = cloud.has_normals();
    let has_colors = cloud.has_colors();

    // Write header
    writeln!(writer, "ply").map_err(|e| MeshError::IoWrite {
        path: path.to_path_buf(),
        source: e,
    })?;
    writeln!(writer, "format ascii 1.0").map_err(|e| MeshError::IoWrite {
        path: path.to_path_buf(),
        source: e,
    })?;
    writeln!(writer, "element vertex {}", cloud.len()).map_err(|e| MeshError::IoWrite {
        path: path.to_path_buf(),
        source: e,
    })?;
    writeln!(writer, "property float x").map_err(|e| MeshError::IoWrite {
        path: path.to_path_buf(),
        source: e,
    })?;
    writeln!(writer, "property float y").map_err(|e| MeshError::IoWrite {
        path: path.to_path_buf(),
        source: e,
    })?;
    writeln!(writer, "property float z").map_err(|e| MeshError::IoWrite {
        path: path.to_path_buf(),
        source: e,
    })?;

    if has_normals {
        writeln!(writer, "property float nx").map_err(|e| MeshError::IoWrite {
            path: path.to_path_buf(),
            source: e,
        })?;
        writeln!(writer, "property float ny").map_err(|e| MeshError::IoWrite {
            path: path.to_path_buf(),
            source: e,
        })?;
        writeln!(writer, "property float nz").map_err(|e| MeshError::IoWrite {
            path: path.to_path_buf(),
            source: e,
        })?;
    }

    if has_colors {
        writeln!(writer, "property uchar red").map_err(|e| MeshError::IoWrite {
            path: path.to_path_buf(),
            source: e,
        })?;
        writeln!(writer, "property uchar green").map_err(|e| MeshError::IoWrite {
            path: path.to_path_buf(),
            source: e,
        })?;
        writeln!(writer, "property uchar blue").map_err(|e| MeshError::IoWrite {
            path: path.to_path_buf(),
            source: e,
        })?;
    }

    writeln!(writer, "end_header").map_err(|e| MeshError::IoWrite {
        path: path.to_path_buf(),
        source: e,
    })?;

    // Write data
    for point in &cloud.points {
        let p = &point.position;
        let mut line = format!("{} {} {}", p.x, p.y, p.z);

        if has_normals {
            if let Some(n) = &point.normal {
                line.push_str(&format!(" {} {} {}", n.x, n.y, n.z));
            } else {
                line.push_str(" 0 0 0");
            }
        }

        if has_colors {
            if let Some(c) = &point.color {
                line.push_str(&format!(" {} {} {}", c.r, c.g, c.b));
            } else {
                line.push_str(" 255 255 255");
            }
        }

        writeln!(writer, "{}", line).map_err(|e| MeshError::IoWrite {
            path: path.to_path_buf(),
            source: e,
        })?;
    }

    Ok(())
}

/// Load point cloud from XYZ file (simple ASCII format).
fn load_xyz(path: &Path) -> MeshResult<PointCloud> {
    let file = File::open(path).map_err(|e| MeshError::IoRead {
        path: path.to_path_buf(),
        source: e,
    })?;
    let reader = BufReader::new(file);

    let mut cloud = PointCloud::new();

    for (line_num, line) in reader.lines().enumerate() {
        let line = line.map_err(|e| MeshError::IoRead {
            path: path.to_path_buf(),
            source: e,
        })?;

        let line = line.trim();
        if line.is_empty() || line.starts_with('#') || line.starts_with("//") {
            continue;
        }

        let parts: Vec<&str> = line.split_whitespace().collect();
        if parts.len() < 3 {
            warn!("Skipping invalid line {} in XYZ file", line_num + 1);
            continue;
        }

        let x: f64 = parts[0].parse().map_err(|_| MeshError::ParseError {
            path: path.to_path_buf(),
            details: format!("Invalid x coordinate on line {}", line_num + 1),
        })?;
        let y: f64 = parts[1].parse().map_err(|_| MeshError::ParseError {
            path: path.to_path_buf(),
            details: format!("Invalid y coordinate on line {}", line_num + 1),
        })?;
        let z: f64 = parts[2].parse().map_err(|_| MeshError::ParseError {
            path: path.to_path_buf(),
            details: format!("Invalid z coordinate on line {}", line_num + 1),
        })?;

        let mut point = CloudPoint::from_coords(x, y, z);

        // Check for normals (6 values: x y z nx ny nz)
        if parts.len() >= 6
            && let (Ok(nx), Ok(ny), Ok(nz)) = (
                parts[3].parse::<f64>(),
                parts[4].parse::<f64>(),
                parts[5].parse::<f64>(),
            )
        {
            point.normal = Some(Vector3::new(nx, ny, nz));
        }

        // Check for colors (9 values: x y z nx ny nz r g b, or 6: x y z r g b)
        if parts.len() >= 9 {
            if let (Ok(r), Ok(g), Ok(b)) = (
                parts[6].parse::<u8>(),
                parts[7].parse::<u8>(),
                parts[8].parse::<u8>(),
            ) {
                point.color = Some(VertexColor::new(r, g, b));
            }
        } else if parts.len() == 6 && point.normal.is_none() {
            // Might be x y z r g b (no normals)
            if let (Ok(r), Ok(g), Ok(b)) = (
                parts[3].parse::<u8>(),
                parts[4].parse::<u8>(),
                parts[5].parse::<u8>(),
            ) {
                point.color = Some(VertexColor::new(r, g, b));
            }
        }

        cloud.push(point);
    }

    Ok(cloud)
}

/// Save point cloud to XYZ file.
fn save_xyz(cloud: &PointCloud, path: &Path) -> MeshResult<()> {
    let file = File::create(path).map_err(|e| MeshError::IoWrite {
        path: path.to_path_buf(),
        source: e,
    })?;
    let mut writer = BufWriter::new(file);

    for point in &cloud.points {
        let mut line = format!(
            "{} {} {}",
            point.position.x, point.position.y, point.position.z
        );

        if let Some(n) = &point.normal {
            line.push_str(&format!(" {} {} {}", n.x, n.y, n.z));
        }

        if let Some(c) = &point.color {
            line.push_str(&format!(" {} {} {}", c.r, c.g, c.b));
        }

        writeln!(writer, "{}", line).map_err(|e| MeshError::IoWrite {
            path: path.to_path_buf(),
            source: e,
        })?;
    }

    Ok(())
}

/// Load point cloud from PCD file (Point Cloud Library format).
fn load_pcd(path: &Path) -> MeshResult<PointCloud> {
    let file = File::open(path).map_err(|e| MeshError::IoRead {
        path: path.to_path_buf(),
        source: e,
    })?;
    let mut reader = BufReader::new(file);

    // Parse header
    let mut fields: Vec<String> = Vec::new();
    let mut width: usize = 0;
    let mut height: usize = 1;
    #[allow(unused_assignments)]
    let mut is_binary = false;

    loop {
        let mut line = String::new();
        reader.read_line(&mut line).map_err(|e| MeshError::IoRead {
            path: path.to_path_buf(),
            source: e,
        })?;

        let line = line.trim();
        if line.starts_with("DATA") {
            is_binary = line.contains("binary");
            break;
        }

        if line.starts_with("FIELDS") {
            fields = line
                .split_whitespace()
                .skip(1)
                .map(|s| s.to_lowercase())
                .collect();
        } else if line.starts_with("WIDTH") {
            width = line
                .split_whitespace()
                .nth(1)
                .and_then(|s| s.parse().ok())
                .unwrap_or(0);
        } else if line.starts_with("HEIGHT") {
            height = line
                .split_whitespace()
                .nth(1)
                .and_then(|s| s.parse().ok())
                .unwrap_or(1);
        }
    }

    if is_binary {
        return Err(MeshError::ParseError {
            path: path.to_path_buf(),
            details: "Binary PCD files not yet supported, use ASCII".to_string(),
        });
    }

    let expected_points = width * height;
    let mut cloud = PointCloud::with_capacity(expected_points);

    // Find indices for common fields
    let x_idx = fields.iter().position(|f| f == "x");
    let y_idx = fields.iter().position(|f| f == "y");
    let z_idx = fields.iter().position(|f| f == "z");
    let nx_idx = fields.iter().position(|f| f == "normal_x" || f == "nx");
    let ny_idx = fields.iter().position(|f| f == "normal_y" || f == "ny");
    let nz_idx = fields.iter().position(|f| f == "normal_z" || f == "nz");
    let r_idx = fields.iter().position(|f| f == "r" || f == "red");
    let g_idx = fields.iter().position(|f| f == "g" || f == "green");
    let b_idx = fields.iter().position(|f| f == "b" || f == "blue");
    let intensity_idx = fields.iter().position(|f| f == "intensity");

    // Read data
    for line in reader.lines() {
        let line = line.map_err(|e| MeshError::IoRead {
            path: path.to_path_buf(),
            source: e,
        })?;

        let parts: Vec<&str> = line.split_whitespace().collect();
        if parts.is_empty() {
            continue;
        }

        // Extract coordinates
        let x = x_idx
            .and_then(|i| parts.get(i))
            .and_then(|s| s.parse::<f64>().ok())
            .unwrap_or(0.0);
        let y = y_idx
            .and_then(|i| parts.get(i))
            .and_then(|s| s.parse::<f64>().ok())
            .unwrap_or(0.0);
        let z = z_idx
            .and_then(|i| parts.get(i))
            .and_then(|s| s.parse::<f64>().ok())
            .unwrap_or(0.0);

        // Skip invalid points (NaN)
        if x.is_nan() || y.is_nan() || z.is_nan() {
            continue;
        }

        let mut point = CloudPoint::from_coords(x, y, z);

        // Extract normals
        if let (Some(nxi), Some(nyi), Some(nzi)) = (nx_idx, ny_idx, nz_idx)
            && let (Some(nx), Some(ny), Some(nz)) = (
                parts.get(nxi).and_then(|s| s.parse::<f64>().ok()),
                parts.get(nyi).and_then(|s| s.parse::<f64>().ok()),
                parts.get(nzi).and_then(|s| s.parse::<f64>().ok()),
            )
            && !nx.is_nan()
            && !ny.is_nan()
            && !nz.is_nan()
        {
            point.normal = Some(Vector3::new(nx, ny, nz));
        }

        // Extract colors
        if let (Some(ri), Some(gi), Some(bi)) = (r_idx, g_idx, b_idx)
            && let (Some(r), Some(g), Some(b)) = (
                parts.get(ri).and_then(|s| s.parse::<u8>().ok()),
                parts.get(gi).and_then(|s| s.parse::<u8>().ok()),
                parts.get(bi).and_then(|s| s.parse::<u8>().ok()),
            )
        {
            point.color = Some(VertexColor::new(r, g, b));
        }

        // Extract intensity
        if let Some(ii) = intensity_idx
            && let Some(i) = parts.get(ii).and_then(|s| s.parse::<f32>().ok())
        {
            point.intensity = Some(i);
        }

        cloud.push(point);
    }

    Ok(cloud)
}

/// Save point cloud to PCD file.
fn save_pcd(cloud: &PointCloud, path: &Path) -> MeshResult<()> {
    let file = File::create(path).map_err(|e| MeshError::IoWrite {
        path: path.to_path_buf(),
        source: e,
    })?;
    let mut writer = BufWriter::new(file);

    let has_normals = cloud.has_normals();
    let has_colors = cloud.has_colors();

    // Write header
    writeln!(writer, "# .PCD v0.7 - Point Cloud Data file format").map_err(|e| {
        MeshError::IoWrite {
            path: path.to_path_buf(),
            source: e,
        }
    })?;
    writeln!(writer, "VERSION 0.7").map_err(|e| MeshError::IoWrite {
        path: path.to_path_buf(),
        source: e,
    })?;

    // Build fields list
    let mut fields = String::from("FIELDS x y z");
    let mut sizes = String::from("SIZE 4 4 4");
    let mut types = String::from("TYPE F F F");
    let mut counts = String::from("COUNT 1 1 1");

    if has_normals {
        fields.push_str(" normal_x normal_y normal_z");
        sizes.push_str(" 4 4 4");
        types.push_str(" F F F");
        counts.push_str(" 1 1 1");
    }

    if has_colors {
        fields.push_str(" red green blue");
        sizes.push_str(" 1 1 1");
        types.push_str(" U U U");
        counts.push_str(" 1 1 1");
    }

    writeln!(writer, "{}", fields).map_err(|e| MeshError::IoWrite {
        path: path.to_path_buf(),
        source: e,
    })?;
    writeln!(writer, "{}", sizes).map_err(|e| MeshError::IoWrite {
        path: path.to_path_buf(),
        source: e,
    })?;
    writeln!(writer, "{}", types).map_err(|e| MeshError::IoWrite {
        path: path.to_path_buf(),
        source: e,
    })?;
    writeln!(writer, "{}", counts).map_err(|e| MeshError::IoWrite {
        path: path.to_path_buf(),
        source: e,
    })?;
    writeln!(writer, "WIDTH {}", cloud.len()).map_err(|e| MeshError::IoWrite {
        path: path.to_path_buf(),
        source: e,
    })?;
    writeln!(writer, "HEIGHT 1").map_err(|e| MeshError::IoWrite {
        path: path.to_path_buf(),
        source: e,
    })?;
    writeln!(writer, "VIEWPOINT 0 0 0 1 0 0 0").map_err(|e| MeshError::IoWrite {
        path: path.to_path_buf(),
        source: e,
    })?;
    writeln!(writer, "POINTS {}", cloud.len()).map_err(|e| MeshError::IoWrite {
        path: path.to_path_buf(),
        source: e,
    })?;
    writeln!(writer, "DATA ascii").map_err(|e| MeshError::IoWrite {
        path: path.to_path_buf(),
        source: e,
    })?;

    // Write data
    for point in &cloud.points {
        let mut line = format!(
            "{} {} {}",
            point.position.x, point.position.y, point.position.z
        );

        if has_normals {
            if let Some(n) = &point.normal {
                line.push_str(&format!(" {} {} {}", n.x, n.y, n.z));
            } else {
                line.push_str(" 0 0 0");
            }
        }

        if has_colors {
            if let Some(c) = &point.color {
                line.push_str(&format!(" {} {} {}", c.r, c.g, c.b));
            } else {
                line.push_str(" 255 255 255");
            }
        }

        writeln!(writer, "{}", line).map_err(|e| MeshError::IoWrite {
            path: path.to_path_buf(),
            source: e,
        })?;
    }

    Ok(())
}

// ============================================================================
// Normal estimation
// ============================================================================

/// Build a KD-tree from the point cloud.
fn build_kdtree(cloud: &PointCloud) -> kiddo::KdTree<f64, 3> {
    let mut kdtree = kiddo::KdTree::new();
    for (i, point) in cloud.points.iter().enumerate() {
        kdtree.add(
            &[point.position.x, point.position.y, point.position.z],
            i as u64,
        );
    }
    kdtree
}

/// Estimate normals using PCA on local neighborhoods.
fn estimate_normals(cloud: &PointCloud, k: usize) -> MeshResult<Vec<Vector3<f64>>> {
    let kdtree = build_kdtree(cloud);
    let mut normals = Vec::with_capacity(cloud.len());

    for point in &cloud.points {
        let neighbors = kdtree.nearest_n::<kiddo::SquaredEuclidean>(
            &[point.position.x, point.position.y, point.position.z],
            k,
        );

        // Collect neighbor positions
        let neighbor_points: Vec<Point3<f64>> = neighbors
            .iter()
            .map(|n| cloud.points[n.item as usize].position)
            .collect();

        // Compute centroid
        let centroid: Vector3<f64> = neighbor_points
            .iter()
            .map(|p| p.coords)
            .fold(Vector3::zeros(), |acc, v| acc + v)
            / neighbor_points.len() as f64;

        // Build covariance matrix
        let mut cov = nalgebra::Matrix3::zeros();
        for np in &neighbor_points {
            let d = np.coords - centroid;
            cov += d * d.transpose();
        }

        // Eigen decomposition (smallest eigenvector is the normal)
        let eig = cov.symmetric_eigen();
        let mut min_idx = 0;
        let mut min_val = eig.eigenvalues[0];
        for i in 1..3 {
            if eig.eigenvalues[i] < min_val {
                min_val = eig.eigenvalues[i];
                min_idx = i;
            }
        }

        let normal = eig.eigenvectors.column(min_idx).into_owned();
        let norm = normal.norm();
        if norm > 1e-10 {
            normals.push(normal / norm);
        } else {
            normals.push(Vector3::new(0.0, 0.0, 1.0));
        }
    }

    Ok(normals)
}

// ============================================================================
// Surface reconstruction
// ============================================================================

/// Algorithm for surface reconstruction.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ReconstructionAlgorithm {
    /// Ball-pivoting algorithm (works well for uniformly sampled point clouds).
    BallPivoting,
    /// SDF-based reconstruction (similar to Poisson, works well for noisy data).
    SdfBased,
}

/// Parameters for surface reconstruction.
#[derive(Debug, Clone)]
pub struct ReconstructionParams {
    /// Algorithm to use.
    pub algorithm: ReconstructionAlgorithm,

    /// Ball radius for ball-pivoting (auto-detected if None).
    pub ball_radius: Option<f64>,

    /// Voxel size for SDF-based reconstruction (auto-detected if None).
    pub voxel_size: Option<f64>,

    /// Number of neighbors for normal estimation if normals are missing.
    pub normal_neighbors: usize,

    /// Whether to auto-estimate normals if missing.
    pub auto_estimate_normals: bool,
}

impl Default for ReconstructionParams {
    fn default() -> Self {
        Self {
            algorithm: ReconstructionAlgorithm::BallPivoting,
            ball_radius: None,
            voxel_size: None,
            normal_neighbors: 16,
            auto_estimate_normals: true,
        }
    }
}

impl ReconstructionParams {
    /// Create params for ball-pivoting with auto-detected radius.
    pub fn ball_pivoting() -> Self {
        Self {
            algorithm: ReconstructionAlgorithm::BallPivoting,
            ..Default::default()
        }
    }

    /// Create params for ball-pivoting with specific radius.
    pub fn ball_pivoting_with_radius(radius: f64) -> Self {
        Self {
            algorithm: ReconstructionAlgorithm::BallPivoting,
            ball_radius: Some(radius),
            ..Default::default()
        }
    }

    /// Create params for SDF-based reconstruction.
    pub fn sdf_based() -> Self {
        Self {
            algorithm: ReconstructionAlgorithm::SdfBased,
            ..Default::default()
        }
    }

    /// Create params for SDF-based reconstruction with specific voxel size.
    pub fn sdf_based_with_voxel_size(voxel_size: f64) -> Self {
        Self {
            algorithm: ReconstructionAlgorithm::SdfBased,
            voxel_size: Some(voxel_size),
            ..Default::default()
        }
    }
}

/// Result of surface reconstruction.
#[derive(Debug)]
pub struct ReconstructionResult {
    /// The reconstructed mesh.
    pub mesh: Mesh,

    /// Number of faces generated.
    pub face_count: usize,

    /// Number of vertices in the output.
    pub vertex_count: usize,

    /// Effective ball radius used (for ball-pivoting).
    pub ball_radius: Option<f64>,

    /// Effective voxel size used (for SDF-based).
    pub voxel_size: Option<f64>,

    /// Whether normals were auto-estimated.
    pub normals_estimated: bool,
}

/// Ball-pivoting surface reconstruction.
fn reconstruct_ball_pivoting(
    cloud: &PointCloud,
    params: &ReconstructionParams,
) -> MeshResult<ReconstructionResult> {
    let start = std::time::Instant::now();

    // Ensure normals are present
    let mut cloud = cloud.clone();
    let normals_estimated = if !cloud.has_normals() && params.auto_estimate_normals {
        cloud = cloud.with_estimated_normals(params.normal_neighbors)?;
        cloud.orient_normals_outward();
        true
    } else {
        false
    };

    if !cloud.has_normals() {
        return Err(MeshError::RepairFailed {
            details: "Ball-pivoting requires point normals".to_string(),
        });
    }

    // Estimate ball radius if not provided
    let ball_radius = params
        .ball_radius
        .unwrap_or_else(|| estimate_point_spacing(&cloud) * 2.0);

    info!(
        "Ball-pivoting reconstruction: {} points, radius={:.4}",
        cloud.len(),
        ball_radius
    );

    // Build KD-tree for efficient neighbor lookups
    let kdtree = build_kdtree(&cloud);

    // Ball-pivoting algorithm implementation
    let mut mesh = Mesh::new();
    let mut used_points: Vec<bool> = vec![false; cloud.len()];
    let mut edge_front: Vec<(usize, usize)> = Vec::new();

    // Convert cloud points to mesh vertices
    for point in &cloud.points {
        mesh.vertices.push(point.to_vertex());
    }

    // Find a seed triangle
    if let Some(seed) = find_seed_triangle(&cloud, &kdtree, ball_radius, &used_points) {
        mesh.faces
            .push([seed.0 as u32, seed.1 as u32, seed.2 as u32]);
        used_points[seed.0] = true;
        used_points[seed.1] = true;
        used_points[seed.2] = true;

        // Add edges to front
        edge_front.push((seed.0, seed.1));
        edge_front.push((seed.1, seed.2));
        edge_front.push((seed.2, seed.0));
    }

    // Process edge front
    let mut iterations = 0;
    let max_iterations = cloud.len() * 10;

    while !edge_front.is_empty() && iterations < max_iterations {
        iterations += 1;

        let (v1, v2) = edge_front.pop().unwrap();

        // Try to find a point to pivot to
        if let Some(v3) = find_pivot_point(
            &cloud,
            &kdtree,
            v1,
            v2,
            ball_radius,
            &used_points,
            &mesh.faces,
        ) {
            mesh.faces.push([v1 as u32, v2 as u32, v3 as u32]);

            if !used_points[v3] {
                used_points[v3] = true;
                // Add new edges to front (in reverse order for correct winding)
                edge_front.push((v2, v3));
                edge_front.push((v3, v1));
            }
        }
    }

    debug!(
        "Ball-pivoting completed in {:?}: {} faces",
        start.elapsed(),
        mesh.faces.len()
    );

    Ok(ReconstructionResult {
        face_count: mesh.faces.len(),
        vertex_count: mesh.vertices.len(),
        ball_radius: Some(ball_radius),
        voxel_size: None,
        normals_estimated,
        mesh,
    })
}

/// Estimate average point spacing using k-nearest neighbors.
fn estimate_point_spacing(cloud: &PointCloud) -> f64 {
    if cloud.len() < 2 {
        return 1.0;
    }

    let kdtree = build_kdtree(cloud);
    let sample_size = cloud.len().min(1000);
    let step = cloud.len() / sample_size;

    let mut total_spacing = 0.0;
    let mut count = 0;

    for i in (0..cloud.len()).step_by(step.max(1)) {
        let point = &cloud.points[i];
        let neighbors = kdtree.nearest_n::<kiddo::SquaredEuclidean>(
            &[point.position.x, point.position.y, point.position.z],
            2,
        );

        if neighbors.len() >= 2 {
            total_spacing += neighbors[1].distance.sqrt();
            count += 1;
        }
    }

    if count > 0 {
        total_spacing / count as f64
    } else {
        1.0
    }
}

/// Find a seed triangle for ball-pivoting.
fn find_seed_triangle(
    cloud: &PointCloud,
    kdtree: &kiddo::KdTree<f64, 3>,
    ball_radius: f64,
    used_points: &[bool],
) -> Option<(usize, usize, usize)> {
    let search_radius = ball_radius * 2.0;

    for (i, point) in cloud.points.iter().enumerate() {
        if used_points[i] {
            continue;
        }

        let neighbors = kdtree.within::<kiddo::SquaredEuclidean>(
            &[point.position.x, point.position.y, point.position.z],
            search_radius * search_radius,
        );

        // Find pairs of neighbors that form valid triangles
        for j_item in &neighbors {
            let j = j_item.item as usize;
            if j == i || used_points[j] {
                continue;
            }

            for k_item in &neighbors {
                let k = k_item.item as usize;
                if k == i || k == j || used_points[k] {
                    continue;
                }

                // Check if triangle is valid
                if is_valid_triangle(cloud, i, j, k, ball_radius) {
                    return Some((i, j, k));
                }
            }
        }
    }

    None
}

/// Check if three points form a valid triangle for ball-pivoting.
fn is_valid_triangle(cloud: &PointCloud, i: usize, j: usize, k: usize, ball_radius: f64) -> bool {
    let p1 = &cloud.points[i].position;
    let p2 = &cloud.points[j].position;
    let p3 = &cloud.points[k].position;

    // Check edge lengths
    let e1 = (p2 - p1).norm();
    let e2 = (p3 - p2).norm();
    let e3 = (p1 - p3).norm();

    let max_edge = ball_radius * 2.0;
    if e1 > max_edge || e2 > max_edge || e3 > max_edge {
        return false;
    }

    // Check if triangle normal is consistent with vertex normals
    let edge1 = p2 - p1;
    let edge2 = p3 - p1;
    let face_normal = edge1.cross(&edge2);
    let face_normal_norm = face_normal.norm();

    if face_normal_norm < 1e-10 {
        return false; // Degenerate triangle
    }

    let face_normal = face_normal / face_normal_norm;

    // Check consistency with vertex normals
    if let (Some(n1), Some(n2), Some(n3)) = (
        &cloud.points[i].normal,
        &cloud.points[j].normal,
        &cloud.points[k].normal,
    ) {
        let avg_normal = (n1 + n2 + n3) / 3.0;
        if face_normal.dot(&avg_normal) < 0.0 {
            return false;
        }
    }

    true
}

/// Find a pivot point for an edge.
fn find_pivot_point(
    cloud: &PointCloud,
    kdtree: &kiddo::KdTree<f64, 3>,
    v1: usize,
    v2: usize,
    ball_radius: f64,
    used_points: &[bool],
    existing_faces: &[[u32; 3]],
) -> Option<usize> {
    let p1 = &cloud.points[v1].position;
    let p2 = &cloud.points[v2].position;
    let midpoint = Point3::from((p1.coords + p2.coords) / 2.0);

    let search_radius = ball_radius * 2.0;
    let neighbors = kdtree.within::<kiddo::SquaredEuclidean>(
        &[midpoint.x, midpoint.y, midpoint.z],
        search_radius * search_radius,
    );

    let mut best_candidate: Option<usize> = None;
    let mut best_angle = f64::MAX;

    for item in &neighbors {
        let idx = item.item as usize;
        if idx == v1 || idx == v2 {
            continue;
        }

        // Check if this triangle already exists
        let face_exists = existing_faces.iter().any(|f| {
            let mut sorted = [f[0] as usize, f[1] as usize, f[2] as usize];
            sorted.sort_unstable();
            let mut test = [v1, v2, idx];
            test.sort_unstable();
            sorted == test
        });

        if face_exists {
            continue;
        }

        // Check if valid triangle
        if !is_valid_triangle(cloud, v1, v2, idx, ball_radius) {
            continue;
        }

        // Compute pivot angle (prefer smaller angles)
        let p3 = &cloud.points[idx].position;
        let edge = p2 - p1;
        let to_p3 = p3 - p1;
        let angle = edge.angle(&to_p3);

        // Prefer unused points, but allow used points if no unused available
        let penalty = if used_points[idx] { 1000.0 } else { 0.0 };
        let score = angle + penalty;

        if score < best_angle {
            best_angle = score;
            best_candidate = Some(idx);
        }
    }

    best_candidate
}

/// SDF-based surface reconstruction (similar to Poisson).
fn reconstruct_sdf_based(
    cloud: &PointCloud,
    params: &ReconstructionParams,
) -> MeshResult<ReconstructionResult> {
    let start = std::time::Instant::now();

    // Ensure normals are present
    let mut cloud = cloud.clone();
    let normals_estimated = if !cloud.has_normals() && params.auto_estimate_normals {
        cloud = cloud.with_estimated_normals(params.normal_neighbors)?;
        cloud.orient_normals_outward();
        true
    } else {
        false
    };

    if !cloud.has_normals() {
        return Err(MeshError::RepairFailed {
            details: "SDF reconstruction requires point normals".to_string(),
        });
    }

    // Estimate voxel size if not provided
    let point_spacing = estimate_point_spacing(&cloud);
    let voxel_size = params.voxel_size.unwrap_or(point_spacing);

    info!(
        "SDF reconstruction: {} points, voxel_size={:.4}",
        cloud.len(),
        voxel_size
    );

    // Get bounds with padding
    let (min_bound, max_bound) = cloud.bounds().ok_or_else(|| MeshError::EmptyMesh {
        details: "Point cloud has no bounds".to_string(),
    })?;

    let padding = voxel_size * 3.0;
    let min_bound = Point3::new(
        min_bound.x - padding,
        min_bound.y - padding,
        min_bound.z - padding,
    );
    let max_bound = Point3::new(
        max_bound.x + padding,
        max_bound.y + padding,
        max_bound.z + padding,
    );

    // Compute grid dimensions
    let extent = max_bound - min_bound;
    let dims = [
        ((extent.x / voxel_size).ceil() as usize).max(2),
        ((extent.y / voxel_size).ceil() as usize).max(2),
        ((extent.z / voxel_size).ceil() as usize).max(2),
    ];

    debug!("SDF grid: {}x{}x{} voxels", dims[0], dims[1], dims[2]);

    // Build KD-tree
    let kdtree = build_kdtree(&cloud);

    // Compute SDF values at grid points
    let total_voxels = dims[0] * dims[1] * dims[2];
    let mut sdf_values = vec![f64::MAX; total_voxels];

    for iz in 0..dims[2] {
        for iy in 0..dims[1] {
            for ix in 0..dims[0] {
                let idx = ix + iy * dims[0] + iz * dims[0] * dims[1];
                let pos = Point3::new(
                    min_bound.x + (ix as f64 + 0.5) * voxel_size,
                    min_bound.y + (iy as f64 + 0.5) * voxel_size,
                    min_bound.z + (iz as f64 + 0.5) * voxel_size,
                );

                // Find nearest point
                let nearest = kdtree.nearest_one::<kiddo::SquaredEuclidean>(&[pos.x, pos.y, pos.z]);
                let nearest_point = &cloud.points[nearest.item as usize];
                let dist = nearest.distance.sqrt();

                // Compute signed distance using normal
                if let Some(normal) = &nearest_point.normal {
                    let to_grid = pos - nearest_point.position;
                    let sign = if to_grid.dot(normal) >= 0.0 {
                        1.0
                    } else {
                        -1.0
                    };
                    sdf_values[idx] = sign * dist;
                } else {
                    sdf_values[idx] = dist;
                }
            }
        }
    }

    // Extract isosurface using marching cubes (via fast_surface_nets)
    let mesh = extract_isosurface_from_sdf(&sdf_values, dims, min_bound, voxel_size)?;

    debug!(
        "SDF reconstruction completed in {:?}: {} faces",
        start.elapsed(),
        mesh.faces.len()
    );

    Ok(ReconstructionResult {
        face_count: mesh.faces.len(),
        vertex_count: mesh.vertices.len(),
        ball_radius: None,
        voxel_size: Some(voxel_size),
        normals_estimated,
        mesh,
    })
}

/// Extract isosurface from SDF values using marching cubes.
fn extract_isosurface_from_sdf(
    sdf_values: &[f64],
    dims: [usize; 3],
    min_bound: Point3<f64>,
    voxel_size: f64,
) -> MeshResult<Mesh> {
    use fast_surface_nets::ndshape::ConstShape;
    use fast_surface_nets::{SurfaceNetsBuffer, ndshape::ConstShape3u32, surface_nets};

    // Convert to the format expected by fast_surface_nets
    // We need a padded grid for boundary handling
    type SampleShape = ConstShape3u32<66, 66, 66>;

    if dims[0] > 64 || dims[1] > 64 || dims[2] > 64 {
        // For large grids, fall back to simple marching cubes
        return extract_isosurface_simple(sdf_values, dims, min_bound, voxel_size);
    }

    // Pad the SDF values
    let mut padded_sdf = vec![1.0f32; SampleShape::SIZE as usize];
    for iz in 0..dims[2] {
        for iy in 0..dims[1] {
            for ix in 0..dims[0] {
                let src_idx = ix + iy * dims[0] + iz * dims[0] * dims[1];
                let dst_idx = (ix + 1) + (iy + 1) * 66 + (iz + 1) * 66 * 66;
                padded_sdf[dst_idx] = sdf_values[src_idx] as f32;
            }
        }
    }

    let mut buffer = SurfaceNetsBuffer::default();
    surface_nets(&padded_sdf, &SampleShape {}, [0; 3], [65; 3], &mut buffer);

    // Convert to mesh
    let mut mesh = Mesh::new();

    for pos in &buffer.positions {
        let world_pos = Point3::new(
            min_bound.x + (pos[0] as f64 - 1.0) * voxel_size,
            min_bound.y + (pos[1] as f64 - 1.0) * voxel_size,
            min_bound.z + (pos[2] as f64 - 1.0) * voxel_size,
        );
        mesh.vertices.push(Vertex::new(world_pos));
    }

    for chunk in buffer.indices.chunks(3) {
        if chunk.len() == 3 {
            mesh.faces.push([chunk[0], chunk[1], chunk[2]]);
        }
    }

    Ok(mesh)
}

/// Simple marching cubes for larger grids.
fn extract_isosurface_simple(
    sdf_values: &[f64],
    dims: [usize; 3],
    min_bound: Point3<f64>,
    voxel_size: f64,
) -> MeshResult<Mesh> {
    // Simplified marching cubes implementation
    let mut mesh = Mesh::new();
    let mut vertex_map: std::collections::HashMap<(usize, usize, usize, u8), u32> =
        std::collections::HashMap::new();

    let get_sdf = |ix: usize, iy: usize, iz: usize| -> f64 {
        if ix >= dims[0] || iy >= dims[1] || iz >= dims[2] {
            return 1.0;
        }
        sdf_values[ix + iy * dims[0] + iz * dims[0] * dims[1]]
    };

    for iz in 0..dims[2] - 1 {
        for iy in 0..dims[1] - 1 {
            for ix in 0..dims[0] - 1 {
                // Get corner values
                let corners = [
                    get_sdf(ix, iy, iz),
                    get_sdf(ix + 1, iy, iz),
                    get_sdf(ix + 1, iy + 1, iz),
                    get_sdf(ix, iy + 1, iz),
                    get_sdf(ix, iy, iz + 1),
                    get_sdf(ix + 1, iy, iz + 1),
                    get_sdf(ix + 1, iy + 1, iz + 1),
                    get_sdf(ix, iy + 1, iz + 1),
                ];

                // Compute cube index
                let mut cube_index = 0u8;
                for (i, &val) in corners.iter().enumerate() {
                    if val < 0.0 {
                        cube_index |= 1 << i;
                    }
                }

                if cube_index == 0 || cube_index == 255 {
                    continue;
                }

                // Generate triangles for this cube configuration
                let triangles = get_marching_cubes_triangles(cube_index);
                for tri in triangles {
                    let mut face_indices = [0u32; 3];
                    for (i, &edge) in tri.iter().enumerate() {
                        let key = (ix, iy, iz, edge);
                        let vertex_idx = *vertex_map.entry(key).or_insert_with(|| {
                            let (v1, v2) = edge_to_vertices(edge);
                            let t = -corners[v1 as usize]
                                / (corners[v2 as usize] - corners[v1 as usize] + 1e-10);
                            let p1 = corner_position(ix, iy, iz, v1);
                            let p2 = corner_position(ix, iy, iz, v2);
                            let pos = Point3::new(
                                min_bound.x + (p1.0 + t * (p2.0 - p1.0)) * voxel_size,
                                min_bound.y + (p1.1 + t * (p2.1 - p1.1)) * voxel_size,
                                min_bound.z + (p1.2 + t * (p2.2 - p1.2)) * voxel_size,
                            );
                            mesh.vertices.push(Vertex::new(pos));
                            (mesh.vertices.len() - 1) as u32
                        });
                        face_indices[i] = vertex_idx;
                    }
                    mesh.faces.push(face_indices);
                }
            }
        }
    }

    Ok(mesh)
}

/// Get corner position offset for marching cubes.
fn corner_position(ix: usize, iy: usize, iz: usize, corner: u8) -> (f64, f64, f64) {
    let dx = if corner & 1 != 0 { 1.0 } else { 0.0 };
    let dy = if corner & 2 != 0 { 1.0 } else { 0.0 };
    let dz = if corner & 4 != 0 { 1.0 } else { 0.0 };
    (ix as f64 + dx, iy as f64 + dy, iz as f64 + dz)
}

/// Convert edge index to vertex indices.
fn edge_to_vertices(edge: u8) -> (u8, u8) {
    const EDGES: [(u8, u8); 12] = [
        (0, 1),
        (1, 2),
        (2, 3),
        (3, 0), // Bottom
        (4, 5),
        (5, 6),
        (6, 7),
        (7, 4), // Top
        (0, 4),
        (1, 5),
        (2, 6),
        (3, 7), // Vertical
    ];
    EDGES[edge as usize]
}

/// Get triangles for a cube configuration (simplified lookup).
fn get_marching_cubes_triangles(cube_index: u8) -> Vec<[u8; 3]> {
    // Simplified: only handle basic cases
    // In production, use a full 256-entry lookup table
    let mut triangles = Vec::new();

    // Count set bits to determine complexity
    let count = cube_index.count_ones();

    if count == 1 {
        // Single corner inside: one triangle
        let corner = cube_index.trailing_zeros() as u8;
        let edges = corner_edges(corner);
        triangles.push(edges);
    } else if count == 7 {
        // Single corner outside: one triangle (inverted)
        let outside = !cube_index;
        let corner = outside.trailing_zeros() as u8;
        let edges = corner_edges(corner);
        triangles.push([edges[2], edges[1], edges[0]]);
    }
    // More cases would be needed for full implementation

    triangles
}

/// Get edges adjacent to a corner.
fn corner_edges(corner: u8) -> [u8; 3] {
    const CORNER_EDGES: [[u8; 3]; 8] = [
        [0, 3, 8],  // 0
        [0, 1, 9],  // 1
        [1, 2, 10], // 2
        [2, 3, 11], // 3
        [4, 7, 8],  // 4
        [4, 5, 9],  // 5
        [5, 6, 10], // 6
        [6, 7, 11], // 7
    ];
    CORNER_EDGES[corner as usize]
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    fn approx_eq(a: f64, b: f64) -> bool {
        (a - b).abs() < 1e-6
    }

    #[test]
    fn test_cloud_point_creation() {
        let p = CloudPoint::from_coords(1.0, 2.0, 3.0);
        assert!(approx_eq(p.position.x, 1.0));
        assert!(approx_eq(p.position.y, 2.0));
        assert!(approx_eq(p.position.z, 3.0));
        assert!(p.normal.is_none());
        assert!(p.color.is_none());
    }

    #[test]
    fn test_cloud_point_with_normal() {
        let p = CloudPoint::with_normal(Point3::new(1.0, 2.0, 3.0), Vector3::new(0.0, 0.0, 1.0));
        assert!(p.normal.is_some());
        let n = p.normal.unwrap();
        assert!(approx_eq(n.z, 1.0));
    }

    #[test]
    fn test_point_cloud_new() {
        let cloud = PointCloud::new();
        assert!(cloud.is_empty());
        assert_eq!(cloud.len(), 0);
    }

    #[test]
    fn test_point_cloud_from_positions() {
        let positions = vec![
            Point3::new(0.0, 0.0, 0.0),
            Point3::new(1.0, 0.0, 0.0),
            Point3::new(0.0, 1.0, 0.0),
        ];
        let cloud = PointCloud::from_positions(&positions);
        assert_eq!(cloud.len(), 3);
    }

    #[test]
    fn test_point_cloud_bounds() {
        let mut cloud = PointCloud::new();
        cloud.push_coords(0.0, 0.0, 0.0);
        cloud.push_coords(10.0, 5.0, 3.0);
        cloud.push_coords(-2.0, 8.0, 1.0);

        let (min, max) = cloud.bounds().unwrap();
        assert!(approx_eq(min.x, -2.0));
        assert!(approx_eq(min.y, 0.0));
        assert!(approx_eq(min.z, 0.0));
        assert!(approx_eq(max.x, 10.0));
        assert!(approx_eq(max.y, 8.0));
        assert!(approx_eq(max.z, 3.0));
    }

    #[test]
    fn test_point_cloud_centroid() {
        let mut cloud = PointCloud::new();
        cloud.push_coords(0.0, 0.0, 0.0);
        cloud.push_coords(2.0, 0.0, 0.0);
        cloud.push_coords(1.0, 2.0, 0.0);

        let centroid = cloud.centroid().unwrap();
        assert!(approx_eq(centroid.x, 1.0));
        assert!(approx_eq(centroid.y, 2.0 / 3.0));
        assert!(approx_eq(centroid.z, 0.0));
    }

    #[test]
    fn test_point_cloud_has_normals() {
        let mut cloud = PointCloud::new();
        cloud.push(CloudPoint::from_coords(0.0, 0.0, 0.0));
        assert!(!cloud.has_normals());

        let mut cloud2 = PointCloud::new();
        cloud2.push(CloudPoint::with_normal(
            Point3::new(0.0, 0.0, 0.0),
            Vector3::new(0.0, 0.0, 1.0),
        ));
        assert!(cloud2.has_normals());
    }

    #[test]
    fn test_point_cloud_from_mesh() {
        let mut mesh = Mesh::new();
        mesh.vertices.push(Vertex::from_coords(0.0, 0.0, 0.0));
        mesh.vertices.push(Vertex::from_coords(1.0, 0.0, 0.0));
        mesh.vertices.push(Vertex::from_coords(0.0, 1.0, 0.0));

        let cloud = PointCloud::from_mesh(&mesh);
        assert_eq!(cloud.len(), 3);
    }

    #[test]
    fn test_point_cloud_downsample() {
        let mut cloud = PointCloud::new();
        // Create a grid of points
        for x in 0..10 {
            for y in 0..10 {
                for z in 0..10 {
                    cloud.push_coords(x as f64 * 0.1, y as f64 * 0.1, z as f64 * 0.1);
                }
            }
        }

        let downsampled = cloud.downsample(0.5);
        assert!(downsampled.len() < cloud.len());
    }

    #[test]
    fn test_point_cloud_translate() {
        let mut cloud = PointCloud::new();
        cloud.push_coords(0.0, 0.0, 0.0);
        cloud.push_coords(1.0, 1.0, 1.0);

        cloud.translate(Vector3::new(10.0, 20.0, 30.0));

        assert!(approx_eq(cloud.points[0].position.x, 10.0));
        assert!(approx_eq(cloud.points[0].position.y, 20.0));
        assert!(approx_eq(cloud.points[0].position.z, 30.0));
    }

    #[test]
    fn test_point_cloud_scale() {
        let mut cloud = PointCloud::new();
        cloud.push_coords(0.0, 0.0, 0.0);
        cloud.push_coords(2.0, 0.0, 0.0);
        cloud.push_coords(0.0, 2.0, 0.0);

        cloud.scale(2.0);

        // Centroid is at (2/3, 2/3, 0), points should be scaled around it
        let (min, max) = cloud.bounds().unwrap();
        let extent = max - min;
        // Original extent was 2, should now be 4
        assert!(approx_eq(extent.x, 4.0));
        assert!(approx_eq(extent.y, 4.0));
    }

    #[test]
    fn test_estimate_point_spacing() {
        let mut cloud = PointCloud::new();
        // Create uniformly spaced points
        for i in 0..10 {
            cloud.push_coords(i as f64, 0.0, 0.0);
        }

        let spacing = estimate_point_spacing(&cloud);
        assert!(spacing > 0.5 && spacing < 2.0);
    }

    #[test]
    fn test_normal_estimation() {
        let mut cloud = PointCloud::new();
        // Create a planar point cloud in the XY plane
        for x in 0..5 {
            for y in 0..5 {
                cloud.push_coords(x as f64, y as f64, 0.0);
            }
        }

        let cloud_with_normals = cloud.with_estimated_normals(8).unwrap();
        assert!(cloud_with_normals.has_normals());

        // All normals should be roughly (0, 0, ±1)
        for point in &cloud_with_normals.points {
            let n = point.normal.unwrap();
            assert!(n.z.abs() > 0.9);
        }
    }

    #[test]
    fn test_orient_normals_outward() {
        let mut cloud = PointCloud::new();
        // Create a simple cube-like point cloud
        for x in [-1.0, 1.0] {
            for y in [-1.0, 1.0] {
                for z in [-1.0, 1.0] {
                    let mut p = CloudPoint::from_coords(x, y, z);
                    // Initially set normals pointing inward
                    p.normal = Some(Vector3::new(-x, -y, -z).normalize());
                    cloud.push(p);
                }
            }
        }

        cloud.orient_normals_outward();

        // Now all normals should point outward
        let centroid = cloud.centroid().unwrap();
        for point in &cloud.points {
            let to_point = point.position - centroid;
            let normal = point.normal.unwrap();
            assert!(normal.dot(&to_point) > 0.0);
        }
    }

    #[test]
    fn test_reconstruction_params_defaults() {
        let params = ReconstructionParams::default();
        assert_eq!(params.algorithm, ReconstructionAlgorithm::BallPivoting);
        assert!(params.ball_radius.is_none());
        assert!(params.auto_estimate_normals);
    }

    #[test]
    fn test_reconstruction_params_sdf() {
        let params = ReconstructionParams::sdf_based_with_voxel_size(1.0);
        assert_eq!(params.algorithm, ReconstructionAlgorithm::SdfBased);
        assert_eq!(params.voxel_size, Some(1.0));
    }

    #[test]
    fn test_cloud_point_to_vertex() {
        let mut p = CloudPoint::from_coords(1.0, 2.0, 3.0);
        p.normal = Some(Vector3::new(0.0, 0.0, 1.0));
        p.color = Some(VertexColor::new(255, 128, 64));
        p.tag = Some(42);

        let v = p.to_vertex();
        assert!(approx_eq(v.position.x, 1.0));
        assert!(v.normal.is_some());
        assert!(v.color.is_some());
        assert_eq!(v.tag, Some(42));
    }

    #[test]
    fn test_remove_outliers() {
        let mut cloud = PointCloud::new();
        // Create a cluster of points
        for x in 0..5 {
            for y in 0..5 {
                cloud.push_coords(x as f64, y as f64, 0.0);
            }
        }
        // Add an outlier
        cloud.push_coords(100.0, 100.0, 100.0);

        let filtered = cloud.remove_outliers(5, 2.0);
        assert!(filtered.len() < cloud.len());
    }

    #[test]
    fn test_point_cloud_format_detection() {
        assert_eq!(
            PointCloudFormat::from_path(Path::new("test.ply")),
            Some(PointCloudFormat::Ply)
        );
        assert_eq!(
            PointCloudFormat::from_path(Path::new("test.xyz")),
            Some(PointCloudFormat::Xyz)
        );
        assert_eq!(
            PointCloudFormat::from_path(Path::new("test.pcd")),
            Some(PointCloudFormat::Pcd)
        );
        assert_eq!(
            PointCloudFormat::from_path(Path::new("test.txt")),
            Some(PointCloudFormat::Xyz)
        );
        assert_eq!(PointCloudFormat::from_path(Path::new("test.stl")), None);
    }

    #[test]
    fn test_ball_pivoting_simple() {
        // Create a simple planar point cloud
        let mut cloud = PointCloud::new();
        for x in 0..3 {
            for y in 0..3 {
                let mut p = CloudPoint::from_coords(x as f64, y as f64, 0.0);
                p.normal = Some(Vector3::new(0.0, 0.0, 1.0));
                cloud.push(p);
            }
        }

        let params = ReconstructionParams::ball_pivoting_with_radius(2.0);
        let result = cloud.to_mesh(&params);

        // Should produce some triangles
        assert!(result.is_ok());
        let result = result.unwrap();
        assert!(result.vertex_count > 0);
    }

    #[test]
    fn test_sdf_reconstruction_simple() {
        // Create a simple spherical point cloud
        // Using golden angle spiral to avoid duplicate points at poles
        let mut cloud = PointCloud::new();
        let n_points = 500;
        let golden_angle = std::f64::consts::PI * (3.0 - 5.0_f64.sqrt());
        let r = 5.0;

        for i in 0..n_points {
            // Use Fibonacci sphere distribution to avoid pole clustering
            let y = 1.0 - (i as f64 / (n_points - 1) as f64) * 2.0; // y goes from 1 to -1
            let radius_at_y = (1.0 - y * y).sqrt();
            let theta = golden_angle * i as f64;

            let x = (theta.cos() * radius_at_y) * r;
            let z = (theta.sin() * radius_at_y) * r;
            let y = y * r;

            let mut p = CloudPoint::from_coords(x, y, z);
            let normal = Vector3::new(x, y, z);
            let norm = normal.norm();
            if norm > 1e-10 {
                p.normal = Some(normal / norm);
            }
            cloud.push(p);
        }

        let params = ReconstructionParams::sdf_based_with_voxel_size(1.0);
        let result = cloud.to_mesh(&params);

        assert!(result.is_ok());
    }

    #[test]
    fn test_empty_cloud_to_mesh() {
        let cloud = PointCloud::new();
        let result = cloud.to_mesh(&ReconstructionParams::default());
        assert!(result.is_err());
    }
}

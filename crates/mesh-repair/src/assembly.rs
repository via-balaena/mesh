//! Multi-part assembly management.
//!
//! This module provides tools for managing assemblies of multiple mesh parts,
//! supporting hierarchical relationships, connections, and export.
//!
//! # Use Cases
//!
//! - Skate boot assembly (boot + blade holder + liner)
//! - Helmet assembly (shell + liner + visor mount)
//! - Multi-part custom devices with snap-fit connections
//!
//! # Example
//!
//! ```
//! use mesh_repair::{Mesh, Vertex};
//! use mesh_repair::assembly::{Assembly, Part, Connection, ConnectionType};
//! use nalgebra::{Isometry3, Vector3};
//!
//! // Create a simple assembly
//! let mut assembly = Assembly::new("skate_boot");
//!
//! // Create a boot mesh (simplified)
//! let mut boot = Mesh::new();
//! boot.vertices.push(Vertex::from_coords(0.0, 0.0, 0.0));
//! boot.vertices.push(Vertex::from_coords(10.0, 0.0, 0.0));
//! boot.vertices.push(Vertex::from_coords(5.0, 10.0, 0.0));
//! boot.faces.push([0, 1, 2]);
//!
//! // Add the boot as a part
//! let boot_part = Part::new("boot_shell", boot);
//! assembly.add_part(boot_part);
//!
//! // Create a liner mesh
//! let mut liner = Mesh::new();
//! liner.vertices.push(Vertex::from_coords(1.0, 1.0, 0.0));
//! liner.vertices.push(Vertex::from_coords(9.0, 1.0, 0.0));
//! liner.vertices.push(Vertex::from_coords(5.0, 9.0, 0.0));
//! liner.faces.push([0, 1, 2]);
//!
//! // Add liner as a child of the boot
//! let liner_part = Part::new("liner", liner)
//!     .with_parent("boot_shell");
//! assembly.add_part(liner_part);
//!
//! println!("Assembly: {} parts", assembly.part_count());
//! ```

use crate::{Mesh, MeshError, MeshResult};
use nalgebra::{Isometry3, Point3, UnitQuaternion, Vector3};
use std::collections::HashMap;
use std::io::Write;
use std::path::Path;

/// A multi-part assembly.
#[derive(Debug, Clone)]
pub struct Assembly {
    /// Assembly name/identifier.
    pub name: String,

    /// Parts in this assembly, keyed by ID.
    parts: HashMap<String, Part>,

    /// Connections between parts.
    connections: Vec<Connection>,

    /// Assembly-level metadata.
    pub metadata: HashMap<String, String>,

    /// Assembly version.
    pub version: Option<String>,
}

impl Assembly {
    /// Create a new empty assembly.
    pub fn new(name: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            parts: HashMap::new(),
            connections: Vec::new(),
            metadata: HashMap::new(),
            version: None,
        }
    }

    /// Add a part to the assembly.
    ///
    /// Returns an error if a part with the same ID already exists.
    pub fn add_part(&mut self, part: Part) -> MeshResult<()> {
        if self.parts.contains_key(&part.id) {
            return Err(MeshError::invalid_topology(format!(
                "Part with ID '{}' already exists",
                part.id
            )));
        }

        // Validate parent exists if specified
        if let Some(ref parent_id) = part.parent_id
            && !self.parts.contains_key(parent_id)
        {
            return Err(MeshError::invalid_topology(format!(
                "Parent part '{}' does not exist for part '{}'",
                parent_id, part.id
            )));
        }

        self.parts.insert(part.id.clone(), part);
        Ok(())
    }

    /// Remove a part from the assembly.
    ///
    /// Returns the removed part, or None if not found.
    /// Also removes any connections involving this part.
    pub fn remove_part(&mut self, part_id: &str) -> Option<Part> {
        let part = self.parts.remove(part_id)?;

        // Remove connections involving this part
        self.connections
            .retain(|conn| conn.from_part != part_id && conn.to_part != part_id);

        // Clear parent references from children
        for other_part in self.parts.values_mut() {
            if other_part.parent_id.as_deref() == Some(part_id) {
                other_part.parent_id = None;
            }
        }

        Some(part)
    }

    /// Get a part by ID.
    pub fn get_part(&self, part_id: &str) -> Option<&Part> {
        self.parts.get(part_id)
    }

    /// Get a mutable reference to a part by ID.
    pub fn get_part_mut(&mut self, part_id: &str) -> Option<&mut Part> {
        self.parts.get_mut(part_id)
    }

    /// List all part IDs.
    pub fn list_parts(&self) -> impl Iterator<Item = &str> {
        self.parts.keys().map(|s| s.as_str())
    }

    /// Get all parts.
    pub fn parts(&self) -> impl Iterator<Item = &Part> {
        self.parts.values()
    }

    /// Get all parts mutably.
    pub fn parts_mut(&mut self) -> impl Iterator<Item = &mut Part> {
        self.parts.values_mut()
    }

    /// Number of parts in the assembly.
    pub fn part_count(&self) -> usize {
        self.parts.len()
    }

    /// Check if the assembly is empty.
    pub fn is_empty(&self) -> bool {
        self.parts.is_empty()
    }

    /// Define a connection between two parts.
    pub fn define_connection(&mut self, connection: Connection) -> MeshResult<()> {
        // Validate both parts exist
        if !self.parts.contains_key(&connection.from_part) {
            return Err(MeshError::invalid_topology(format!(
                "Part '{}' does not exist",
                connection.from_part
            )));
        }
        if !self.parts.contains_key(&connection.to_part) {
            return Err(MeshError::invalid_topology(format!(
                "Part '{}' does not exist",
                connection.to_part
            )));
        }

        self.connections.push(connection);
        Ok(())
    }

    /// Get all connections.
    pub fn connections(&self) -> &[Connection] {
        &self.connections
    }

    /// Get connections for a specific part.
    pub fn connections_for_part(&self, part_id: &str) -> Vec<&Connection> {
        self.connections
            .iter()
            .filter(|c| c.from_part == part_id || c.to_part == part_id)
            .collect()
    }

    /// Get child parts of a parent.
    pub fn get_children(&self, parent_id: &str) -> Vec<&Part> {
        self.parts
            .values()
            .filter(|p| p.parent_id.as_deref() == Some(parent_id))
            .collect()
    }

    /// Get root parts (parts with no parent).
    pub fn get_root_parts(&self) -> Vec<&Part> {
        self.parts
            .values()
            .filter(|p| p.parent_id.is_none())
            .collect()
    }

    /// Compute the world transform for a part (including parent transforms).
    pub fn get_world_transform(&self, part_id: &str) -> Option<Isometry3<f64>> {
        let part = self.parts.get(part_id)?;
        let mut transform = part.transform;

        // Walk up the parent chain
        let mut current_parent_id = part.parent_id.as_deref();
        while let Some(parent_id) = current_parent_id {
            if let Some(parent) = self.parts.get(parent_id) {
                transform = parent.transform * transform;
                current_parent_id = parent.parent_id.as_deref();
            } else {
                break;
            }
        }

        Some(transform)
    }

    /// Get a transformed copy of a part's mesh (world coordinates).
    pub fn get_transformed_mesh(&self, part_id: &str) -> Option<Mesh> {
        let part = self.parts.get(part_id)?;
        let world_transform = self.get_world_transform(part_id)?;

        let mut mesh = part.mesh.clone();
        for vertex in &mut mesh.vertices {
            vertex.position = world_transform * vertex.position;
        }

        Some(mesh)
    }

    /// Merge all parts into a single mesh (world coordinates).
    pub fn to_merged_mesh(&self) -> Mesh {
        let mut result = Mesh::new();

        for part_id in self.parts.keys() {
            if let Some(mesh) = self.get_transformed_mesh(part_id) {
                let vertex_offset = result.vertices.len() as u32;

                // Add vertices
                result.vertices.extend(mesh.vertices);

                // Add faces with offset
                for face in &mesh.faces {
                    result.faces.push([
                        face[0] + vertex_offset,
                        face[1] + vertex_offset,
                        face[2] + vertex_offset,
                    ]);
                }
            }
        }

        result
    }

    /// Validate the assembly.
    pub fn validate(&self) -> AssemblyValidation {
        let mut result = AssemblyValidation::default();

        // Check for orphan parent references
        for part in self.parts.values() {
            if let Some(ref parent_id) = part.parent_id
                && !self.parts.contains_key(parent_id)
            {
                result
                    .orphan_references
                    .push((part.id.clone(), parent_id.clone()));
            }
        }

        // Check for circular parent references
        for part in self.parts.values() {
            if self.has_circular_reference(&part.id) {
                result.circular_references.push(part.id.clone());
            }
        }

        // Check connections
        for conn in &self.connections {
            if !self.parts.contains_key(&conn.from_part) {
                result
                    .invalid_connections
                    .push((conn.clone(), format!("Missing part: {}", conn.from_part)));
            }
            if !self.parts.contains_key(&conn.to_part) {
                result
                    .invalid_connections
                    .push((conn.clone(), format!("Missing part: {}", conn.to_part)));
            }
        }

        result
    }

    fn has_circular_reference(&self, part_id: &str) -> bool {
        let mut visited = std::collections::HashSet::new();
        let mut current = Some(part_id);

        while let Some(id) = current {
            if visited.contains(id) {
                return true;
            }
            visited.insert(id);

            current = self.parts.get(id).and_then(|p| p.parent_id.as_deref());
        }

        false
    }

    /// Check interference between two parts.
    pub fn check_interference(&self, part_a: &str, part_b: &str) -> MeshResult<InterferenceResult> {
        let mesh_a = self
            .get_transformed_mesh(part_a)
            .ok_or_else(|| MeshError::invalid_topology(format!("Part '{}' not found", part_a)))?;

        let mesh_b = self
            .get_transformed_mesh(part_b)
            .ok_or_else(|| MeshError::invalid_topology(format!("Part '{}' not found", part_b)))?;

        // Compute bounding boxes for quick rejection
        let bbox_a = compute_bbox(&mesh_a);
        let bbox_b = compute_bbox(&mesh_b);

        if !bboxes_overlap(&bbox_a, &bbox_b) {
            return Ok(InterferenceResult {
                has_interference: false,
                overlap_volume: 0.0,
                min_clearance: Some(bbox_distance(&bbox_a, &bbox_b)),
            });
        }

        // For more detailed interference, we'd need proper mesh boolean ops
        // For now, report that bounding boxes overlap
        Ok(InterferenceResult {
            has_interference: true,
            overlap_volume: 0.0, // Would need CSG for accurate volume
            min_clearance: None,
        })
    }

    /// Check clearance between two parts.
    pub fn check_clearance(
        &self,
        part_a: &str,
        part_b: &str,
        min_required: f64,
    ) -> MeshResult<ClearanceResult> {
        let mesh_a = self
            .get_transformed_mesh(part_a)
            .ok_or_else(|| MeshError::invalid_topology(format!("Part '{}' not found", part_a)))?;

        let mesh_b = self
            .get_transformed_mesh(part_b)
            .ok_or_else(|| MeshError::invalid_topology(format!("Part '{}' not found", part_b)))?;

        // Compute approximate clearance using bounding boxes
        let bbox_a = compute_bbox(&mesh_a);
        let bbox_b = compute_bbox(&mesh_b);

        let clearance = bbox_distance(&bbox_a, &bbox_b);

        Ok(ClearanceResult {
            meets_requirement: clearance >= min_required,
            actual_clearance: clearance,
            required_clearance: min_required,
        })
    }

    // =========================================================================
    // Export Methods
    // =========================================================================

    /// Save the assembly to a file.
    ///
    /// The format is determined by the file extension or explicit format option.
    ///
    /// # Supported Formats
    /// - `.3mf` - 3MF with multiple objects and build items
    /// - `.stl` - Merged mesh as single STL (use `save_stl_separate` for individual parts)
    ///
    /// # Example
    /// ```no_run
    /// use mesh_repair::assembly::Assembly;
    /// use std::path::Path;
    ///
    /// let assembly = Assembly::new("my_assembly");
    /// assembly.save(Path::new("output.3mf"), None).unwrap();
    /// ```
    pub fn save(&self, path: &Path, format: Option<AssemblyExportFormat>) -> MeshResult<()> {
        let format = format.unwrap_or_else(|| {
            AssemblyExportFormat::from_path(path).unwrap_or(AssemblyExportFormat::ThreeMf)
        });

        match format {
            AssemblyExportFormat::ThreeMf => self.save_3mf(path),
            AssemblyExportFormat::StlMerged => {
                let merged = self.to_merged_mesh();
                crate::io::save_stl(&merged, path)
            }
            AssemblyExportFormat::StlSeparate => self.save_stl_separate(path),
        }
    }

    /// Save the assembly to a 3MF file with multiple objects and build items.
    ///
    /// Each part becomes a separate object in the 3MF file, with its own
    /// build item that specifies its transform.
    ///
    /// # Example
    /// ```no_run
    /// use mesh_repair::assembly::Assembly;
    /// use std::path::Path;
    ///
    /// let assembly = Assembly::new("skate");
    /// // ... add parts ...
    /// assembly.save_3mf(Path::new("skate.3mf")).unwrap();
    /// ```
    pub fn save_3mf(&self, path: &Path) -> MeshResult<()> {
        use std::fs::File;
        use zip::ZipWriter;
        use zip::write::SimpleFileOptions;

        if self.is_empty() {
            return Err(MeshError::EmptyMesh {
                details: "Cannot save empty assembly".to_string(),
            });
        }

        let file = File::create(path).map_err(|e| MeshError::IoWrite {
            path: path.to_path_buf(),
            source: e,
        })?;

        let mut zip = ZipWriter::new(file);
        let options =
            SimpleFileOptions::default().compression_method(zip::CompressionMethod::Deflated);

        // Write content types file
        zip.start_file("[Content_Types].xml", options)
            .map_err(|e| MeshError::IoWrite {
                path: path.to_path_buf(),
                source: std::io::Error::other(e.to_string()),
            })?;
        zip.write_all(ASSEMBLY_CONTENT_TYPES_XML.as_bytes())
            .map_err(|e| MeshError::IoWrite {
                path: path.to_path_buf(),
                source: e,
            })?;

        // Write relationships file
        zip.start_file("_rels/.rels", options)
            .map_err(|e| MeshError::IoWrite {
                path: path.to_path_buf(),
                source: std::io::Error::other(e.to_string()),
            })?;
        zip.write_all(ASSEMBLY_RELS_XML.as_bytes())
            .map_err(|e| MeshError::IoWrite {
                path: path.to_path_buf(),
                source: e,
            })?;

        // Write the model file
        zip.start_file("3D/3dmodel.model", options)
            .map_err(|e| MeshError::IoWrite {
                path: path.to_path_buf(),
                source: std::io::Error::other(e.to_string()),
            })?;

        let model_xml = self.generate_3mf_model_xml();
        zip.write_all(model_xml.as_bytes())
            .map_err(|e| MeshError::IoWrite {
                path: path.to_path_buf(),
                source: e,
            })?;

        zip.finish().map_err(|e| MeshError::IoWrite {
            path: path.to_path_buf(),
            source: std::io::Error::other(e.to_string()),
        })?;

        Ok(())
    }

    /// Generate 3MF model XML for the assembly.
    fn generate_3mf_model_xml(&self) -> String {
        let mut xml = String::with_capacity(self.parts.len() * 1000);

        xml.push_str(r#"<?xml version="1.0" encoding="UTF-8"?>
<model unit="millimeter" xml:lang="en-US" xmlns="http://schemas.microsoft.com/3dmanufacturing/core/2015/02">
  <metadata name="Title">"#);
        xml.push_str(&escape_xml(&self.name));
        xml.push_str("</metadata>\n");

        if let Some(ref version) = self.version {
            xml.push_str("  <metadata name=\"Version\">");
            xml.push_str(&escape_xml(version));
            xml.push_str("</metadata>\n");
        }

        xml.push_str("  <resources>\n");

        // Create a stable ordering for parts
        let mut part_ids: Vec<&String> = self.parts.keys().collect();
        part_ids.sort();

        // Write each part as a separate object
        for (obj_id, part_id) in part_ids.iter().enumerate() {
            let part = &self.parts[*part_id];
            let object_id = obj_id + 1; // 3MF IDs start at 1

            xml.push_str(&format!(
                "    <object id=\"{}\" type=\"model\" name=\"{}\">\n",
                object_id,
                escape_xml(&part.id)
            ));
            xml.push_str("      <mesh>\n        <vertices>\n");

            // Write vertices
            for v in &part.mesh.vertices {
                xml.push_str(&format!(
                    "          <vertex x=\"{:.6}\" y=\"{:.6}\" z=\"{:.6}\"/>\n",
                    v.position.x, v.position.y, v.position.z
                ));
            }

            xml.push_str("        </vertices>\n        <triangles>\n");

            // Write triangles
            for face in &part.mesh.faces {
                xml.push_str(&format!(
                    "          <triangle v1=\"{}\" v2=\"{}\" v3=\"{}\"/>\n",
                    face[0], face[1], face[2]
                ));
            }

            xml.push_str("        </triangles>\n      </mesh>\n    </object>\n");
        }

        xml.push_str("  </resources>\n  <build>\n");

        // Write build items with transforms
        for (obj_id, part_id) in part_ids.iter().enumerate() {
            let object_id = obj_id + 1;

            // Get world transform for this part
            let world_transform = self
                .get_world_transform(part_id)
                .unwrap_or_else(Isometry3::identity);

            // Only include transform attribute if it's not identity
            if is_identity_transform(&world_transform) {
                xml.push_str(&format!("    <item objectid=\"{}\"/>\n", object_id));
            } else {
                // 3MF uses a 3x4 affine matrix (row-major)
                let matrix = transform_to_3mf_matrix(&world_transform);
                xml.push_str(&format!(
                    "    <item objectid=\"{}\" transform=\"{}\"/>\n",
                    object_id, matrix
                ));
            }
        }

        xml.push_str("  </build>\n</model>\n");

        xml
    }

    /// Save each part as a separate STL file.
    ///
    /// Files are named `{basename}_{part_id}.stl` in the same directory as `path`.
    ///
    /// # Example
    /// ```no_run
    /// use mesh_repair::assembly::Assembly;
    /// use std::path::Path;
    ///
    /// let assembly = Assembly::new("skate");
    /// // ... add parts ...
    /// // Creates: skate_boot.stl, skate_liner.stl, etc.
    /// assembly.save_stl_separate(Path::new("skate.stl")).unwrap();
    /// ```
    pub fn save_stl_separate(&self, path: &Path) -> MeshResult<()> {
        if self.is_empty() {
            return Err(MeshError::EmptyMesh {
                details: "Cannot save empty assembly".to_string(),
            });
        }

        let parent = path.parent().unwrap_or(Path::new("."));
        let stem = path
            .file_stem()
            .and_then(|s| s.to_str())
            .unwrap_or("assembly");

        for (part_id, part) in &self.parts {
            // Skip invisible parts
            if !part.visible {
                continue;
            }

            // Get transformed mesh
            let mesh = self
                .get_transformed_mesh(part_id)
                .unwrap_or_else(|| part.mesh.clone());

            // Create filename
            let filename = format!("{}_{}.stl", stem, sanitize_filename(part_id));
            let file_path = parent.join(filename);

            crate::io::save_stl(&mesh, &file_path)?;
        }

        Ok(())
    }

    /// Generate a bill of materials (BOM) for the assembly.
    ///
    /// Returns a structured list of all parts with their properties.
    ///
    /// # Example
    /// ```
    /// use mesh_repair::{Mesh, Vertex};
    /// use mesh_repair::assembly::{Assembly, Part};
    ///
    /// let mut assembly = Assembly::new("product");
    /// let mut mesh = Mesh::new();
    /// mesh.vertices.push(Vertex::from_coords(0.0, 0.0, 0.0));
    /// mesh.vertices.push(Vertex::from_coords(1.0, 0.0, 0.0));
    /// mesh.vertices.push(Vertex::from_coords(0.5, 1.0, 0.0));
    /// mesh.faces.push([0, 1, 2]);
    ///
    /// assembly.add_part(Part::new("shell", mesh.clone()).with_material("PA12")).unwrap();
    /// assembly.add_part(Part::new("liner", mesh).with_material("TPU")).unwrap();
    ///
    /// let bom = assembly.generate_bom();
    /// assert_eq!(bom.items.len(), 2);
    /// ```
    pub fn generate_bom(&self) -> BillOfMaterials {
        let mut items = Vec::with_capacity(self.parts.len());

        for (part_id, part) in &self.parts {
            let mesh = self
                .get_transformed_mesh(part_id)
                .unwrap_or_else(|| part.mesh.clone());
            let (min, max) = compute_bbox(&mesh);
            let dimensions = max - min;

            // Estimate volume (approximate using bounding box)
            let bbox_volume = dimensions.x * dimensions.y * dimensions.z;

            // Count triangles
            let triangle_count = mesh.faces.len();

            items.push(BomItem {
                part_id: part_id.clone(),
                name: part_id.clone(),
                material: part.material.clone(),
                quantity: 1,
                dimensions: (dimensions.x, dimensions.y, dimensions.z),
                bounding_volume: bbox_volume,
                triangle_count,
                parent: part.parent_id.clone(),
                metadata: part.metadata.clone(),
            });
        }

        // Sort by part ID for consistent output
        items.sort_by(|a, b| a.part_id.cmp(&b.part_id));

        BillOfMaterials {
            assembly_name: self.name.clone(),
            version: self.version.clone(),
            items,
            connections: self.connections.clone(),
        }
    }

    /// Export the BOM to a CSV file.
    ///
    /// # Example
    /// ```no_run
    /// use mesh_repair::assembly::Assembly;
    /// use std::path::Path;
    ///
    /// let assembly = Assembly::new("product");
    /// assembly.export_bom_csv(Path::new("bom.csv")).unwrap();
    /// ```
    pub fn export_bom_csv(&self, path: &Path) -> MeshResult<()> {
        use std::fs::File;

        let bom = self.generate_bom();

        let file = File::create(path).map_err(|e| MeshError::IoWrite {
            path: path.to_path_buf(),
            source: e,
        })?;

        let mut writer = std::io::BufWriter::new(file);

        // Write header
        writeln!(
            writer,
            "Part ID,Material,Quantity,Width (mm),Height (mm),Depth (mm),Volume (mm³),Triangles,Parent"
        )
        .map_err(|e| MeshError::IoWrite {
            path: path.to_path_buf(),
            source: e,
        })?;

        // Write items
        for item in &bom.items {
            writeln!(
                writer,
                "{},{},{},{:.2},{:.2},{:.2},{:.2},{},{}",
                escape_csv(&item.part_id),
                escape_csv(item.material.as_deref().unwrap_or("")),
                item.quantity,
                item.dimensions.0,
                item.dimensions.1,
                item.dimensions.2,
                item.bounding_volume,
                item.triangle_count,
                escape_csv(item.parent.as_deref().unwrap_or(""))
            )
            .map_err(|e| MeshError::IoWrite {
                path: path.to_path_buf(),
                source: e,
            })?;
        }

        Ok(())
    }
}

/// Assembly export format.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AssemblyExportFormat {
    /// 3MF with multiple objects and build items.
    ThreeMf,
    /// Single merged STL file.
    StlMerged,
    /// Separate STL files for each part.
    StlSeparate,
}

impl AssemblyExportFormat {
    /// Determine format from file extension.
    pub fn from_path(path: &Path) -> Option<Self> {
        let ext = path.extension()?.to_str()?.to_lowercase();
        match ext.as_str() {
            "3mf" => Some(Self::ThreeMf),
            "stl" => Some(Self::StlMerged),
            _ => None,
        }
    }
}

/// Bill of materials for an assembly.
#[derive(Debug, Clone)]
pub struct BillOfMaterials {
    /// Assembly name.
    pub assembly_name: String,

    /// Assembly version.
    pub version: Option<String>,

    /// List of items in the BOM.
    pub items: Vec<BomItem>,

    /// Connections between parts.
    pub connections: Vec<Connection>,
}

impl BillOfMaterials {
    /// Get total part count.
    pub fn total_parts(&self) -> usize {
        self.items.iter().map(|i| i.quantity).sum()
    }

    /// Get unique material count.
    pub fn unique_materials(&self) -> Vec<&str> {
        let mut materials: Vec<&str> = self
            .items
            .iter()
            .filter_map(|i| i.material.as_deref())
            .collect();
        materials.sort();
        materials.dedup();
        materials
    }

    /// Get parts by material.
    pub fn parts_by_material(&self, material: &str) -> Vec<&BomItem> {
        self.items
            .iter()
            .filter(|i| i.material.as_deref() == Some(material))
            .collect()
    }
}

/// A single item in the bill of materials.
#[derive(Debug, Clone)]
pub struct BomItem {
    /// Part ID.
    pub part_id: String,

    /// Display name.
    pub name: String,

    /// Material name.
    pub material: Option<String>,

    /// Quantity.
    pub quantity: usize,

    /// Bounding box dimensions (width, height, depth) in mm.
    pub dimensions: (f64, f64, f64),

    /// Bounding box volume in mm³.
    pub bounding_volume: f64,

    /// Number of triangles.
    pub triangle_count: usize,

    /// Parent part ID.
    pub parent: Option<String>,

    /// Additional metadata.
    pub metadata: HashMap<String, String>,
}

/// Content types XML for 3MF assembly.
const ASSEMBLY_CONTENT_TYPES_XML: &str = r#"<?xml version="1.0" encoding="UTF-8"?>
<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types">
  <Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/>
  <Default Extension="model" ContentType="application/vnd.ms-package.3dmanufacturing-3dmodel+xml"/>
</Types>
"#;

/// Relationships XML for 3MF assembly.
const ASSEMBLY_RELS_XML: &str = r#"<?xml version="1.0" encoding="UTF-8"?>
<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">
  <Relationship Target="/3D/3dmodel.model" Id="rel0" Type="http://schemas.microsoft.com/3dmanufacturing/2013/01/3dmodel"/>
</Relationships>
"#;

/// Check if a transform is approximately identity.
fn is_identity_transform(t: &Isometry3<f64>) -> bool {
    let eps = 1e-10;
    let translation_zero = t.translation.vector.norm() < eps;
    let rotation_identity =
        (t.rotation.angle() < eps) || (t.rotation.angle() - std::f64::consts::TAU).abs() < eps;
    translation_zero && rotation_identity
}

/// Convert an Isometry3 to a 3MF transform matrix string.
fn transform_to_3mf_matrix(t: &Isometry3<f64>) -> String {
    // 3MF uses a 3x4 affine matrix in row-major order:
    // m00 m01 m02 m03 m10 m11 m12 m13 m20 m21 m22 m23
    let rot = t.rotation.to_rotation_matrix();
    let trans = t.translation.vector;

    format!(
        "{:.6} {:.6} {:.6} {:.6} {:.6} {:.6} {:.6} {:.6} {:.6} {:.6} {:.6} {:.6}",
        rot[(0, 0)],
        rot[(0, 1)],
        rot[(0, 2)],
        trans.x,
        rot[(1, 0)],
        rot[(1, 1)],
        rot[(1, 2)],
        trans.y,
        rot[(2, 0)],
        rot[(2, 1)],
        rot[(2, 2)],
        trans.z
    )
}

/// Escape special XML characters.
fn escape_xml(s: &str) -> String {
    s.replace('&', "&amp;")
        .replace('<', "&lt;")
        .replace('>', "&gt;")
        .replace('"', "&quot;")
        .replace('\'', "&apos;")
}

/// Escape special characters for CSV.
fn escape_csv(s: &str) -> String {
    if s.contains(',') || s.contains('"') || s.contains('\n') {
        format!("\"{}\"", s.replace('"', "\"\""))
    } else {
        s.to_string()
    }
}

/// Sanitize a string for use as a filename.
fn sanitize_filename(s: &str) -> String {
    s.chars()
        .map(|c| match c {
            '/' | '\\' | ':' | '*' | '?' | '"' | '<' | '>' | '|' => '_',
            _ => c,
        })
        .collect()
}

/// A single part in an assembly.
#[derive(Debug, Clone)]
pub struct Part {
    /// Unique identifier for this part.
    pub id: String,

    /// The mesh geometry.
    pub mesh: Mesh,

    /// Transform relative to parent (or world if no parent).
    pub transform: Isometry3<f64>,

    /// Parent part ID (if any).
    pub parent_id: Option<String>,

    /// Part metadata.
    pub metadata: HashMap<String, String>,

    /// Material name.
    pub material: Option<String>,

    /// Is this part visible?
    pub visible: bool,
}

impl Part {
    /// Create a new part with identity transform.
    pub fn new(id: impl Into<String>, mesh: Mesh) -> Self {
        Self {
            id: id.into(),
            mesh,
            transform: Isometry3::identity(),
            parent_id: None,
            metadata: HashMap::new(),
            material: None,
            visible: true,
        }
    }

    /// Set the parent part ID.
    pub fn with_parent(mut self, parent_id: impl Into<String>) -> Self {
        self.parent_id = Some(parent_id.into());
        self
    }

    /// Set the transform.
    pub fn with_transform(mut self, transform: Isometry3<f64>) -> Self {
        self.transform = transform;
        self
    }

    /// Set translation.
    pub fn with_translation(mut self, x: f64, y: f64, z: f64) -> Self {
        self.transform.translation.vector = Vector3::new(x, y, z);
        self
    }

    /// Set rotation from axis-angle.
    pub fn with_rotation(mut self, axis: Vector3<f64>, angle: f64) -> Self {
        if let Some(axis_unit) = nalgebra::Unit::try_new(axis, 1e-10) {
            self.transform.rotation = UnitQuaternion::from_axis_angle(&axis_unit, angle);
        }
        self
    }

    /// Set material name.
    pub fn with_material(mut self, material: impl Into<String>) -> Self {
        self.material = Some(material.into());
        self
    }

    /// Set visibility.
    pub fn with_visible(mut self, visible: bool) -> Self {
        self.visible = visible;
        self
    }

    /// Add metadata.
    pub fn with_metadata(mut self, key: impl Into<String>, value: impl Into<String>) -> Self {
        self.metadata.insert(key.into(), value.into());
        self
    }

    /// Get the local bounding box of this part.
    pub fn bounding_box(&self) -> (Point3<f64>, Point3<f64>) {
        compute_bbox(&self.mesh)
    }
}

/// Connection between two parts.
#[derive(Debug, Clone)]
pub struct Connection {
    /// Source part ID.
    pub from_part: String,

    /// Target part ID.
    pub to_part: String,

    /// Connection type.
    pub connection_type: ConnectionType,

    /// Connection parameters.
    pub params: ConnectionParams,

    /// Name/description.
    pub name: Option<String>,
}

impl Connection {
    /// Create a new connection.
    pub fn new(
        from_part: impl Into<String>,
        to_part: impl Into<String>,
        connection_type: ConnectionType,
    ) -> Self {
        Self {
            from_part: from_part.into(),
            to_part: to_part.into(),
            connection_type,
            params: ConnectionParams::default(),
            name: None,
        }
    }

    /// Create a snap-fit connection.
    pub fn snap_fit(from_part: impl Into<String>, to_part: impl Into<String>) -> Self {
        Self::new(from_part, to_part, ConnectionType::SnapFit)
    }

    /// Create a press-fit connection.
    pub fn press_fit(
        from_part: impl Into<String>,
        to_part: impl Into<String>,
        interference: f64,
    ) -> Self {
        let mut conn = Self::new(from_part, to_part, ConnectionType::PressFit);
        conn.params.interference = Some(interference);
        conn
    }

    /// Create a clearance connection.
    pub fn clearance(
        from_part: impl Into<String>,
        to_part: impl Into<String>,
        min_clearance: f64,
    ) -> Self {
        let mut conn = Self::new(from_part, to_part, ConnectionType::Clearance);
        conn.params.clearance = Some(min_clearance);
        conn
    }

    /// Set connection name.
    pub fn with_name(mut self, name: impl Into<String>) -> Self {
        self.name = Some(name.into());
        self
    }
}

/// Type of connection between parts.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ConnectionType {
    /// Snap-fit (male/female interlocking).
    SnapFit,

    /// Press-fit (interference fit).
    PressFit,

    /// Clearance fit (loose with minimum gap).
    Clearance,

    /// Glue/adhesive bond.
    Adhesive,

    /// Threaded fastener.
    Threaded,

    /// Sliding fit.
    Sliding,

    /// Custom connection type.
    Custom,
}

/// Parameters for a connection.
#[derive(Debug, Clone, Default)]
pub struct ConnectionParams {
    /// Interference amount (for press-fit), positive = overlap.
    pub interference: Option<f64>,

    /// Clearance amount (for clearance fit), minimum gap.
    pub clearance: Option<f64>,

    /// Snap feature height (for snap-fit).
    pub snap_height: Option<f64>,

    /// Undercut angle for snap (degrees).
    pub undercut_angle: Option<f64>,

    /// Connection location (relative to from_part).
    pub location: Option<Point3<f64>>,

    /// Custom parameters.
    pub custom: HashMap<String, String>,
}

/// Assembly validation result.
#[derive(Debug, Clone, Default)]
pub struct AssemblyValidation {
    /// Parts with orphan parent references.
    pub orphan_references: Vec<(String, String)>,

    /// Parts with circular parent references.
    pub circular_references: Vec<String>,

    /// Invalid connections.
    pub invalid_connections: Vec<(Connection, String)>,
}

impl AssemblyValidation {
    /// Check if the assembly is valid.
    pub fn is_valid(&self) -> bool {
        self.orphan_references.is_empty()
            && self.circular_references.is_empty()
            && self.invalid_connections.is_empty()
    }
}

/// Result of interference check.
#[derive(Debug, Clone)]
pub struct InterferenceResult {
    /// Whether parts interfere.
    pub has_interference: bool,

    /// Volume of overlap (if calculable).
    pub overlap_volume: f64,

    /// Minimum clearance (if no interference).
    pub min_clearance: Option<f64>,
}

/// Result of clearance check.
#[derive(Debug, Clone)]
pub struct ClearanceResult {
    /// Whether the clearance requirement is met.
    pub meets_requirement: bool,

    /// Actual clearance measured.
    pub actual_clearance: f64,

    /// Required clearance.
    pub required_clearance: f64,
}

/// Compute the axis-aligned bounding box of a mesh.
fn compute_bbox(mesh: &Mesh) -> (Point3<f64>, Point3<f64>) {
    if mesh.vertices.is_empty() {
        return (Point3::origin(), Point3::origin());
    }

    let mut min = mesh.vertices[0].position;
    let mut max = mesh.vertices[0].position;

    for v in &mesh.vertices {
        min.x = min.x.min(v.position.x);
        min.y = min.y.min(v.position.y);
        min.z = min.z.min(v.position.z);
        max.x = max.x.max(v.position.x);
        max.y = max.y.max(v.position.y);
        max.z = max.z.max(v.position.z);
    }

    (min, max)
}

/// Check if two bounding boxes overlap.
fn bboxes_overlap(a: &(Point3<f64>, Point3<f64>), b: &(Point3<f64>, Point3<f64>)) -> bool {
    let (a_min, a_max) = a;
    let (b_min, b_max) = b;

    !(a_max.x < b_min.x
        || b_max.x < a_min.x
        || a_max.y < b_min.y
        || b_max.y < a_min.y
        || a_max.z < b_min.z
        || b_max.z < a_min.z)
}

/// Compute distance between two bounding boxes.
fn bbox_distance(a: &(Point3<f64>, Point3<f64>), b: &(Point3<f64>, Point3<f64>)) -> f64 {
    let (a_min, a_max) = a;
    let (b_min, b_max) = b;

    let dx = (b_min.x - a_max.x).max(a_min.x - b_max.x).max(0.0);
    let dy = (b_min.y - a_max.y).max(a_min.y - b_max.y).max(0.0);
    let dz = (b_min.z - a_max.z).max(a_min.z - b_max.z).max(0.0);

    (dx * dx + dy * dy + dz * dz).sqrt()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::Vertex;

    fn create_test_mesh() -> Mesh {
        let mut mesh = Mesh::new();
        mesh.vertices.push(Vertex::from_coords(0.0, 0.0, 0.0));
        mesh.vertices.push(Vertex::from_coords(1.0, 0.0, 0.0));
        mesh.vertices.push(Vertex::from_coords(0.5, 1.0, 0.0));
        mesh.faces.push([0, 1, 2]);
        mesh
    }

    #[test]
    fn test_assembly_new() {
        let assembly = Assembly::new("test_assembly");
        assert_eq!(assembly.name, "test_assembly");
        assert!(assembly.is_empty());
        assert_eq!(assembly.part_count(), 0);
    }

    #[test]
    fn test_add_part() {
        let mut assembly = Assembly::new("test");
        let part = Part::new("part1", create_test_mesh());

        assembly.add_part(part).unwrap();
        assert_eq!(assembly.part_count(), 1);
        assert!(assembly.get_part("part1").is_some());
    }

    #[test]
    fn test_add_duplicate_part_fails() {
        let mut assembly = Assembly::new("test");
        assembly
            .add_part(Part::new("part1", create_test_mesh()))
            .unwrap();

        let result = assembly.add_part(Part::new("part1", create_test_mesh()));
        assert!(result.is_err());
    }

    #[test]
    fn test_remove_part() {
        let mut assembly = Assembly::new("test");
        assembly
            .add_part(Part::new("part1", create_test_mesh()))
            .unwrap();

        let removed = assembly.remove_part("part1");
        assert!(removed.is_some());
        assert!(assembly.is_empty());
    }

    #[test]
    fn test_parent_child() {
        let mut assembly = Assembly::new("test");
        assembly
            .add_part(Part::new("parent", create_test_mesh()))
            .unwrap();
        assembly
            .add_part(Part::new("child", create_test_mesh()).with_parent("parent"))
            .unwrap();

        let children = assembly.get_children("parent");
        assert_eq!(children.len(), 1);
        assert_eq!(children[0].id, "child");
    }

    #[test]
    fn test_root_parts() {
        let mut assembly = Assembly::new("test");
        assembly
            .add_part(Part::new("root1", create_test_mesh()))
            .unwrap();
        assembly
            .add_part(Part::new("root2", create_test_mesh()))
            .unwrap();
        assembly
            .add_part(Part::new("child", create_test_mesh()).with_parent("root1"))
            .unwrap();

        let roots = assembly.get_root_parts();
        assert_eq!(roots.len(), 2);
    }

    #[test]
    fn test_world_transform() {
        let mut assembly = Assembly::new("test");

        let parent = Part::new("parent", create_test_mesh()).with_translation(10.0, 0.0, 0.0);
        assembly.add_part(parent).unwrap();

        let child = Part::new("child", create_test_mesh())
            .with_parent("parent")
            .with_translation(5.0, 0.0, 0.0);
        assembly.add_part(child).unwrap();

        let world_transform = assembly.get_world_transform("child").unwrap();
        assert!((world_transform.translation.vector.x - 15.0).abs() < 1e-10);
    }

    #[test]
    fn test_define_connection() {
        let mut assembly = Assembly::new("test");
        assembly
            .add_part(Part::new("part1", create_test_mesh()))
            .unwrap();
        assembly
            .add_part(Part::new("part2", create_test_mesh()))
            .unwrap();

        let conn = Connection::snap_fit("part1", "part2");
        assembly.define_connection(conn).unwrap();

        assert_eq!(assembly.connections().len(), 1);
    }

    #[test]
    fn test_connection_for_missing_part_fails() {
        let mut assembly = Assembly::new("test");
        assembly
            .add_part(Part::new("part1", create_test_mesh()))
            .unwrap();

        let conn = Connection::snap_fit("part1", "missing");
        let result = assembly.define_connection(conn);
        assert!(result.is_err());
    }

    #[test]
    fn test_validate() {
        let mut assembly = Assembly::new("test");
        assembly
            .add_part(Part::new("part1", create_test_mesh()))
            .unwrap();

        let validation = assembly.validate();
        assert!(validation.is_valid());
    }

    #[test]
    fn test_to_merged_mesh() {
        let mut assembly = Assembly::new("test");
        assembly
            .add_part(Part::new("part1", create_test_mesh()))
            .unwrap();
        assembly
            .add_part(Part::new("part2", create_test_mesh()))
            .unwrap();

        let merged = assembly.to_merged_mesh();
        assert_eq!(merged.vertices.len(), 6); // 3 + 3
        assert_eq!(merged.faces.len(), 2); // 1 + 1
    }

    #[test]
    fn test_check_clearance() {
        let mut assembly = Assembly::new("test");
        assembly
            .add_part(Part::new("part1", create_test_mesh()).with_translation(0.0, 0.0, 0.0))
            .unwrap();
        assembly
            .add_part(Part::new("part2", create_test_mesh()).with_translation(10.0, 0.0, 0.0))
            .unwrap();

        let result = assembly.check_clearance("part1", "part2", 5.0).unwrap();
        assert!(result.meets_requirement);
        assert!(result.actual_clearance > 5.0);
    }

    #[test]
    fn test_part_builder() {
        let part = Part::new("test", create_test_mesh())
            .with_parent("parent")
            .with_translation(1.0, 2.0, 3.0)
            .with_material("TPU")
            .with_visible(false)
            .with_metadata("key", "value");

        assert_eq!(part.parent_id, Some("parent".to_string()));
        assert!((part.transform.translation.vector.x - 1.0).abs() < 1e-10);
        assert_eq!(part.material, Some("TPU".to_string()));
        assert!(!part.visible);
        assert_eq!(part.metadata.get("key"), Some(&"value".to_string()));
    }

    #[test]
    fn test_connection_types() {
        let snap = Connection::snap_fit("a", "b");
        assert_eq!(snap.connection_type, ConnectionType::SnapFit);

        let press = Connection::press_fit("a", "b", 0.1);
        assert_eq!(press.connection_type, ConnectionType::PressFit);
        assert_eq!(press.params.interference, Some(0.1));

        let clearance = Connection::clearance("a", "b", 0.5);
        assert_eq!(clearance.connection_type, ConnectionType::Clearance);
        assert_eq!(clearance.params.clearance, Some(0.5));
    }

    #[test]
    fn test_generate_bom() {
        let mut assembly = Assembly::new("test_assembly");
        assembly.version = Some("1.0".to_string());

        assembly
            .add_part(Part::new("part1", create_test_mesh()).with_material("PLA"))
            .unwrap();
        assembly
            .add_part(Part::new("part2", create_test_mesh()).with_material("TPU"))
            .unwrap();
        assembly
            .add_part(
                Part::new("part3", create_test_mesh())
                    .with_material("PLA")
                    .with_parent("part1"),
            )
            .unwrap();

        let bom = assembly.generate_bom();
        assert_eq!(bom.assembly_name, "test_assembly");
        assert_eq!(bom.version, Some("1.0".to_string()));
        assert_eq!(bom.items.len(), 3);
        assert_eq!(bom.total_parts(), 3);

        let materials = bom.unique_materials();
        assert_eq!(materials.len(), 2);
        assert!(materials.contains(&"PLA"));
        assert!(materials.contains(&"TPU"));

        let pla_parts = bom.parts_by_material("PLA");
        assert_eq!(pla_parts.len(), 2);
    }

    #[test]
    fn test_bom_item_dimensions() {
        let mut assembly = Assembly::new("test");
        assembly
            .add_part(Part::new("part1", create_test_mesh()))
            .unwrap();

        let bom = assembly.generate_bom();
        let item = &bom.items[0];

        // Test mesh is a triangle from (0,0,0) to (1,0,0) to (0.5,1,0)
        assert!((item.dimensions.0 - 1.0).abs() < 1e-6); // width
        assert!((item.dimensions.1 - 1.0).abs() < 1e-6); // height
        assert!(item.dimensions.2 < 1e-6); // depth (flat)
        assert_eq!(item.triangle_count, 1);
    }

    #[test]
    fn test_save_3mf_roundtrip() {
        let mut assembly = Assembly::new("test_assembly");
        assembly
            .metadata
            .insert("author".to_string(), "Test Author".to_string());

        assembly
            .add_part(Part::new("part1", create_test_mesh()).with_translation(0.0, 0.0, 0.0))
            .unwrap();
        assembly
            .add_part(Part::new("part2", create_test_mesh()).with_translation(5.0, 0.0, 0.0))
            .unwrap();

        let temp_dir = std::env::temp_dir();
        let path = temp_dir.join("test_assembly_export.3mf");

        // Save
        assembly.save_3mf(&path).unwrap();
        assert!(path.exists());

        // Verify it's a valid zip with expected structure
        let file = std::fs::File::open(&path).unwrap();
        let mut archive = zip::ZipArchive::new(file).unwrap();

        // Check expected files exist
        assert!(archive.by_name("[Content_Types].xml").is_ok());
        assert!(archive.by_name("_rels/.rels").is_ok());
        assert!(archive.by_name("3D/3dmodel.model").is_ok());

        // Read the model and verify it has objects
        let mut model_file = archive.by_name("3D/3dmodel.model").unwrap();
        let mut model_content = String::new();
        std::io::Read::read_to_string(&mut model_file, &mut model_content).unwrap();

        // Should have 2 objects and 2 build items
        assert!(model_content.contains("<object id=\"1\""));
        assert!(model_content.contains("<object id=\"2\""));
        assert!(model_content.contains("<item objectid=\"1\""));
        assert!(model_content.contains("<item objectid=\"2\""));

        // Clean up
        std::fs::remove_file(&path).ok();
    }

    #[test]
    fn test_save_stl_separate() {
        let mut assembly = Assembly::new("test_assembly");
        assembly
            .add_part(Part::new("part1", create_test_mesh()))
            .unwrap();
        assembly
            .add_part(Part::new("part2", create_test_mesh()))
            .unwrap();

        let temp_dir = std::env::temp_dir().join("test_stl_separate");
        std::fs::create_dir_all(&temp_dir).ok();

        // save_stl_separate expects a file path, not a directory
        // It uses the file stem as a prefix for individual files
        let base_path = temp_dir.join("assembly.stl");
        assembly.save_stl_separate(&base_path).unwrap();

        // Check that individual STL files were created (named as stem_partid.stl)
        assert!(temp_dir.join("assembly_part1.stl").exists());
        assert!(temp_dir.join("assembly_part2.stl").exists());

        // Clean up
        std::fs::remove_dir_all(&temp_dir).ok();
    }

    #[test]
    fn test_export_bom_csv() {
        let mut assembly = Assembly::new("test_assembly");
        assembly
            .add_part(Part::new("part1", create_test_mesh()).with_material("PLA"))
            .unwrap();
        assembly
            .add_part(Part::new("part2", create_test_mesh()).with_material("TPU"))
            .unwrap();

        let temp_dir = std::env::temp_dir();
        let path = temp_dir.join("test_bom.csv");

        assembly.export_bom_csv(&path).unwrap();
        assert!(path.exists());

        let content = std::fs::read_to_string(&path).unwrap();

        // Check header (note: actual format is Part ID,Material,Quantity,...)
        assert!(content.contains("Part ID,Material,Quantity"));
        // Check parts
        assert!(content.contains("part1"));
        assert!(content.contains("part2"));
        assert!(content.contains("PLA"));
        assert!(content.contains("TPU"));

        // Clean up
        std::fs::remove_file(&path).ok();
    }

    #[test]
    fn test_save_with_format_detection() {
        let mut assembly = Assembly::new("test");
        assembly
            .add_part(Part::new("part1", create_test_mesh()))
            .unwrap();

        let temp_dir = std::env::temp_dir();

        // Test 3MF detection
        let path_3mf = temp_dir.join("test_format.3mf");
        assembly.save(&path_3mf, None).unwrap();
        assert!(path_3mf.exists());
        std::fs::remove_file(&path_3mf).ok();

        // Test STL detection
        let path_stl = temp_dir.join("test_format.stl");
        assembly.save(&path_stl, None).unwrap();
        assert!(path_stl.exists());
        std::fs::remove_file(&path_stl).ok();
    }

    #[test]
    fn test_assembly_export_format_from_path() {
        assert_eq!(
            AssemblyExportFormat::from_path(Path::new("test.3mf")),
            Some(AssemblyExportFormat::ThreeMf)
        );
        assert_eq!(
            AssemblyExportFormat::from_path(Path::new("test.stl")),
            Some(AssemblyExportFormat::StlMerged)
        );
        assert_eq!(AssemblyExportFormat::from_path(Path::new("test.obj")), None);
    }

    #[test]
    fn test_transform_helpers() {
        // Test identity detection
        let identity = Isometry3::identity();
        assert!(is_identity_transform(&identity));

        // Test non-identity with translation
        let translated = Isometry3::translation(1.0, 0.0, 0.0);
        assert!(!is_identity_transform(&translated));

        // Test transform matrix generation
        let matrix_str = transform_to_3mf_matrix(&translated);
        // Should have 12 numbers: 3x4 affine matrix
        let parts: Vec<&str> = matrix_str.split_whitespace().collect();
        assert_eq!(parts.len(), 12);
    }

    #[test]
    fn test_escape_functions() {
        // XML escaping
        assert_eq!(escape_xml("a < b"), "a &lt; b");
        assert_eq!(escape_xml("a & b"), "a &amp; b");
        assert_eq!(escape_xml("\"test\""), "&quot;test&quot;");

        // CSV escaping
        assert_eq!(escape_csv("simple"), "simple");
        assert_eq!(escape_csv("with,comma"), "\"with,comma\"");
        assert_eq!(escape_csv("with\"quote"), "\"with\"\"quote\"");
    }

    #[test]
    fn test_sanitize_filename() {
        assert_eq!(sanitize_filename("normal_name"), "normal_name");
        assert_eq!(sanitize_filename("with/slash"), "with_slash");
        assert_eq!(sanitize_filename("with:colon"), "with_colon");
        assert_eq!(
            sanitize_filename("with*star?question"),
            "with_star_question"
        );
    }
}

//! Mesh file I/O for STL, OBJ, and 3MF formats.

use std::fs::File;
use std::io::{BufReader, BufWriter, Read, Write};
use std::path::Path;

use tracing::{debug, info, warn};

use crate::error::{MeshError, MeshResult};
use crate::validate::{ValidationOptions, validate_mesh_data};
use crate::{Mesh, Vertex};

/// Supported mesh file formats.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MeshFormat {
    Stl,
    Obj,
    ThreeMf,
    Ply,
}

impl MeshFormat {
    /// Detect format from file extension.
    pub fn from_path(path: &Path) -> Option<Self> {
        path.extension()
            .and_then(|ext| ext.to_str())
            .map(|ext| ext.to_lowercase())
            .and_then(|ext| match ext.as_str() {
                "stl" => Some(MeshFormat::Stl),
                "obj" => Some(MeshFormat::Obj),
                "3mf" => Some(MeshFormat::ThreeMf),
                "ply" => Some(MeshFormat::Ply),
                _ => None,
            })
    }
}

/// Load a mesh from file, auto-detecting format from extension.
pub fn load_mesh(path: &Path) -> MeshResult<Mesh> {
    let format = MeshFormat::from_path(path).ok_or_else(|| MeshError::UnsupportedFormat {
        extension: path.extension().and_then(|e| e.to_str()).map(String::from),
    })?;

    info!("Loading mesh from {:?} (format: {:?})", path, format);

    let mesh = match format {
        MeshFormat::Stl => load_stl(path)?,
        MeshFormat::Obj => load_obj(path)?,
        MeshFormat::ThreeMf => load_3mf(path)?,
        MeshFormat::Ply => load_ply(path)?,
    };

    // Log basic stats
    if let Some((min, max)) = mesh.bounds() {
        let dims = max - min;
        info!(
            "Loaded mesh: {} vertices, {} faces",
            mesh.vertex_count(),
            mesh.face_count()
        );
        debug!(
            "Bounding box: [{:.1}, {:.1}, {:.1}] to [{:.1}, {:.1}, {:.1}]",
            min.x, min.y, min.z, max.x, max.y, max.z
        );
        debug!("Dimensions: {:.1} x {:.1} x {:.1}", dims.x, dims.y, dims.z);

        // Warn if dimensions seem unusually small or large
        let max_dim = dims.x.max(dims.y).max(dims.z);
        if max_dim < 0.1 {
            warn!(
                "Mesh largest dimension is {:.6} - may need scaling",
                max_dim
            );
        }
    }

    if mesh.vertices.is_empty() || mesh.faces.is_empty() {
        return Err(MeshError::EmptyMesh {
            details: "mesh has no vertices or faces".to_string(),
        });
    }

    // Validate mesh data (check for invalid indices and coordinates)
    validate_mesh_data(&mesh, &ValidationOptions::default())?;

    Ok(mesh)
}

/// Load mesh from STL file (binary or ASCII).
fn load_stl(path: &Path) -> MeshResult<Mesh> {
    let file = File::open(path).map_err(|e| MeshError::IoRead {
        path: path.to_path_buf(),
        source: e,
    })?;
    let mut reader = BufReader::new(file);

    // stl_io::read_stl returns an IndexedMesh with vertices and indexed faces
    let stl = stl_io::read_stl(&mut reader).map_err(|e| MeshError::ParseError {
        path: path.to_path_buf(),
        details: e.to_string(),
    })?;

    debug!(
        "STL contains {} vertices, {} triangles",
        stl.vertices.len(),
        stl.faces.len()
    );

    // Convert stl_io types to our types
    let mut mesh = Mesh::with_capacity(stl.vertices.len(), stl.faces.len());

    // Convert vertices (stl_io::Vertex is Vector<f32> with .0 being [f32; 3])
    for v in &stl.vertices {
        mesh.vertices.push(Vertex::from_coords(
            v.0[0] as f64,
            v.0[1] as f64,
            v.0[2] as f64,
        ));
    }

    // Convert faces (stl_io::IndexedTriangle has .vertices: [usize; 3])
    for face in &stl.faces {
        let indices = [
            face.vertices[0] as u32,
            face.vertices[1] as u32,
            face.vertices[2] as u32,
        ];

        // Skip degenerate triangles
        if indices[0] != indices[1] && indices[1] != indices[2] && indices[0] != indices[2] {
            mesh.faces.push(indices);
        }
    }

    debug!(
        "Converted mesh: {} vertices, {} faces",
        mesh.vertices.len(),
        mesh.faces.len()
    );

    Ok(mesh)
}

/// Load mesh from OBJ file.
fn load_obj(path: &Path) -> MeshResult<Mesh> {
    let (models, _materials) = tobj::load_obj(
        path,
        &tobj::LoadOptions {
            triangulate: true,
            single_index: true,
            ..Default::default()
        },
    )
    .map_err(|e| MeshError::ParseError {
        path: path.to_path_buf(),
        details: e.to_string(),
    })?;

    if models.is_empty() {
        return Err(MeshError::EmptyMesh {
            details: "OBJ file contains no models".to_string(),
        });
    }

    // Merge all models into single mesh
    let mut mesh = Mesh::new();
    let mut vertex_offset = 0u32;

    for model in &models {
        debug!("OBJ model '{}': loading", model.name);

        let obj_mesh = &model.mesh;

        // Add vertices
        for chunk in obj_mesh.positions.chunks(3) {
            if chunk.len() == 3 {
                mesh.vertices.push(Vertex::from_coords(
                    chunk[0] as f64,
                    chunk[1] as f64,
                    chunk[2] as f64,
                ));
            }
        }

        // Add faces (indices are per-model, need offset)
        for chunk in obj_mesh.indices.chunks(3) {
            if chunk.len() == 3 {
                mesh.faces.push([
                    chunk[0] + vertex_offset,
                    chunk[1] + vertex_offset,
                    chunk[2] + vertex_offset,
                ]);
            }
        }

        vertex_offset = mesh.vertices.len() as u32;
    }

    debug!(
        "OBJ loaded: {} vertices, {} faces from {} models",
        mesh.vertices.len(),
        mesh.faces.len(),
        models.len()
    );

    Ok(mesh)
}

/// Load mesh from PLY file (ASCII or binary).
///
/// PLY (Polygon File Format, also known as Stanford Triangle Format) is widely
/// used in 3D scanning and point cloud processing. This function supports:
/// - ASCII format
/// - Binary little-endian format
/// - Binary big-endian format
///
/// The loader expects `vertex` elements with `x`, `y`, `z` properties and
/// `face` elements with a `vertex_indices` list property.
fn load_ply(path: &Path) -> MeshResult<Mesh> {
    use ply_rs::parser::Parser;
    use ply_rs::ply::Property;

    let file = File::open(path).map_err(|e| MeshError::IoRead {
        path: path.to_path_buf(),
        source: e,
    })?;
    let mut reader = BufReader::new(file);

    // Create parser for default element type
    let parser = Parser::<ply_rs::ply::DefaultElement>::new();

    // Parse the PLY file
    let ply = parser
        .read_ply(&mut reader)
        .map_err(|e| MeshError::ParseError {
            path: path.to_path_buf(),
            details: format!("PLY parse error: {:?}", e),
        })?;

    let mut mesh = Mesh::new();

    // Extract vertices
    if let Some(vertices) = ply.payload.get("vertex") {
        for vertex_element in vertices {
            let x = get_ply_float(vertex_element.get("x"), "x", path)?;
            let y = get_ply_float(vertex_element.get("y"), "y", path)?;
            let z = get_ply_float(vertex_element.get("z"), "z", path)?;

            let mut vertex = Vertex::from_coords(x, y, z);

            // Try to load normals if present
            if let (Some(nx), Some(ny), Some(nz)) = (
                vertex_element.get("nx"),
                vertex_element.get("ny"),
                vertex_element.get("nz"),
            ) && let (Ok(nx), Ok(ny), Ok(nz)) = (
                get_ply_float(Some(nx), "nx", path),
                get_ply_float(Some(ny), "ny", path),
                get_ply_float(Some(nz), "nz", path),
            ) {
                vertex.normal = Some(nalgebra::Vector3::new(nx, ny, nz));
            }

            // Try to load vertex colors if present (red, green, blue)
            if let (Some(r), Some(g), Some(b)) = (
                vertex_element.get("red"),
                vertex_element.get("green"),
                vertex_element.get("blue"),
            ) && let (Ok(r), Ok(g), Ok(b)) = (
                get_ply_u8(Some(r)),
                get_ply_u8(Some(g)),
                get_ply_u8(Some(b)),
            ) {
                vertex.color = Some(crate::VertexColor::new(r, g, b));
            }

            mesh.vertices.push(vertex);
        }
    }

    // Extract faces
    if let Some(faces) = ply.payload.get("face") {
        for face_element in faces {
            // Face indices can be stored under various names
            let indices = face_element
                .get("vertex_indices")
                .or_else(|| face_element.get("vertex_index"));

            if let Some(Property::ListInt(indices)) = indices {
                // Triangulate if necessary (fan triangulation for polygons)
                if indices.len() >= 3 {
                    for i in 1..indices.len() - 1 {
                        mesh.faces.push([
                            indices[0] as u32,
                            indices[i] as u32,
                            indices[i + 1] as u32,
                        ]);
                    }
                }
            } else if let Some(Property::ListUInt(indices)) = indices {
                if indices.len() >= 3 {
                    for i in 1..indices.len() - 1 {
                        mesh.faces.push([indices[0], indices[i], indices[i + 1]]);
                    }
                }
            } else if let Some(Property::ListUChar(indices)) = indices
                && indices.len() >= 3
            {
                for i in 1..indices.len() - 1 {
                    mesh.faces
                        .push([indices[0] as u32, indices[i] as u32, indices[i + 1] as u32]);
                }
            }
        }
    }

    debug!(
        "PLY loaded: {} vertices, {} faces",
        mesh.vertices.len(),
        mesh.faces.len()
    );

    Ok(mesh)
}

/// Helper to extract a float value from a PLY property.
fn get_ply_float(prop: Option<&ply_rs::ply::Property>, name: &str, path: &Path) -> MeshResult<f64> {
    use ply_rs::ply::Property;

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
            path: path.to_path_buf(),
            details: format!("Missing or invalid PLY property: {}", name),
        }),
    }
}

/// Helper to extract a u8 value from a PLY property (for colors).
fn get_ply_u8(prop: Option<&ply_rs::ply::Property>) -> Result<u8, ()> {
    use ply_rs::ply::Property;

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

/// Save mesh to file, auto-detecting format from extension.
pub fn save_mesh(mesh: &Mesh, path: &Path) -> MeshResult<()> {
    let format = MeshFormat::from_path(path).ok_or_else(|| MeshError::UnsupportedFormat {
        extension: path.extension().and_then(|e| e.to_str()).map(String::from),
    })?;

    match format {
        MeshFormat::Stl => save_stl(mesh, path),
        MeshFormat::Obj => save_obj(mesh, path),
        MeshFormat::ThreeMf => save_3mf(mesh, path),
        MeshFormat::Ply => save_ply(mesh, path),
    }
}

/// Save mesh to STL file (binary format).
pub fn save_stl(mesh: &Mesh, path: &Path) -> MeshResult<()> {
    info!("Saving mesh to {:?}", path);

    let file = File::create(path).map_err(|e| MeshError::IoWrite {
        path: path.to_path_buf(),
        source: e,
    })?;
    let mut writer = BufWriter::new(file);

    // Build stl_io triangles
    let triangles: Vec<stl_io::Triangle> = mesh
        .faces
        .iter()
        .map(|&[i0, i1, i2]| {
            let v0 = &mesh.vertices[i0 as usize].position;
            let v1 = &mesh.vertices[i1 as usize].position;
            let v2 = &mesh.vertices[i2 as usize].position;

            stl_io::Triangle {
                normal: stl_io::Normal::new([0.0, 0.0, 0.0]), // Readers recompute
                vertices: [
                    stl_io::Vertex::new([v0.x as f32, v0.y as f32, v0.z as f32]),
                    stl_io::Vertex::new([v1.x as f32, v1.y as f32, v1.z as f32]),
                    stl_io::Vertex::new([v2.x as f32, v2.y as f32, v2.z as f32]),
                ],
            }
        })
        .collect();

    stl_io::write_stl(&mut writer, triangles.iter()).map_err(|e| MeshError::IoWrite {
        path: path.to_path_buf(),
        source: std::io::Error::other(e.to_string()),
    })?;

    writer.flush().map_err(|e| MeshError::IoWrite {
        path: path.to_path_buf(),
        source: e,
    })?;

    info!("Saved {} triangles to {:?}", mesh.face_count(), path);

    Ok(())
}

/// Save mesh to OBJ file (ASCII format).
///
/// OBJ format preserves vertex indices exactly, making it ideal for debugging
/// pipeline stages where vertex tracking is important. Unlike STL which
/// duplicates vertices per-triangle, OBJ maintains the indexed mesh structure.
///
/// The output includes:
/// - Vertex positions (`v x y z`)
/// - Vertex normals if present (`vn nx ny nz`)
/// - Face indices (`f v1 v2 v3` or `f v1//n1 v2//n2 v3//n3` with normals)
/// - Comments with tag and offset info for debugging
pub fn save_obj(mesh: &Mesh, path: &Path) -> MeshResult<()> {
    info!("Saving mesh to {:?} (OBJ format)", path);

    let file = File::create(path).map_err(|e| MeshError::IoWrite {
        path: path.to_path_buf(),
        source: e,
    })?;
    let mut writer = BufWriter::new(file);

    // Header comment
    writeln!(writer, "# OBJ file exported by mesh-repair").map_err(|e| MeshError::IoWrite {
        path: path.to_path_buf(),
        source: e,
    })?;
    writeln!(writer, "# Vertices: {}", mesh.vertices.len()).map_err(|e| MeshError::IoWrite {
        path: path.to_path_buf(),
        source: e,
    })?;
    writeln!(writer, "# Faces: {}", mesh.faces.len()).map_err(|e| MeshError::IoWrite {
        path: path.to_path_buf(),
        source: e,
    })?;
    writeln!(writer).map_err(|e| MeshError::IoWrite {
        path: path.to_path_buf(),
        source: e,
    })?;

    // Check if we have normals
    let has_normals = mesh.vertices.iter().any(|v| v.normal.is_some());

    // Write vertices
    for (i, v) in mesh.vertices.iter().enumerate() {
        // Write position
        writeln!(
            writer,
            "v {:.6} {:.6} {:.6}",
            v.position.x, v.position.y, v.position.z
        )
        .map_err(|e| MeshError::IoWrite {
            path: path.to_path_buf(),
            source: e,
        })?;

        // Add debug comment with vertex attributes (tag, offset)
        if v.tag.is_some() || v.offset.is_some() {
            let tag_str = v.tag.map_or("none".to_string(), |z| format!("{}", z));
            let offset_str = v.offset.map_or("none".to_string(), |c| format!("{:.3}", c));
            writeln!(writer, "# v{} tag={} offset={}", i, tag_str, offset_str).map_err(|e| {
                MeshError::IoWrite {
                    path: path.to_path_buf(),
                    source: e,
                }
            })?;
        }
    }

    // Write normals if present
    if has_normals {
        writeln!(writer).map_err(|e| MeshError::IoWrite {
            path: path.to_path_buf(),
            source: e,
        })?;
        writeln!(writer, "# Vertex normals").map_err(|e| MeshError::IoWrite {
            path: path.to_path_buf(),
            source: e,
        })?;

        for v in &mesh.vertices {
            if let Some(n) = &v.normal {
                writeln!(writer, "vn {:.6} {:.6} {:.6}", n.x, n.y, n.z).map_err(|e| {
                    MeshError::IoWrite {
                        path: path.to_path_buf(),
                        source: e,
                    }
                })?;
            } else {
                // Write zero normal as placeholder to maintain index correspondence
                writeln!(writer, "vn 0 0 0").map_err(|e| MeshError::IoWrite {
                    path: path.to_path_buf(),
                    source: e,
                })?;
            }
        }
    }

    // Write faces
    writeln!(writer).map_err(|e| MeshError::IoWrite {
        path: path.to_path_buf(),
        source: e,
    })?;
    writeln!(writer, "# Faces").map_err(|e| MeshError::IoWrite {
        path: path.to_path_buf(),
        source: e,
    })?;

    for face in &mesh.faces {
        // OBJ uses 1-based indexing
        let i0 = face[0] + 1;
        let i1 = face[1] + 1;
        let i2 = face[2] + 1;

        if has_normals {
            // Format: f v1//n1 v2//n2 v3//n3 (no texture coords)
            writeln!(writer, "f {}//{} {}//{} {}//{}", i0, i0, i1, i1, i2, i2).map_err(|e| {
                MeshError::IoWrite {
                    path: path.to_path_buf(),
                    source: e,
                }
            })?;
        } else {
            writeln!(writer, "f {} {} {}", i0, i1, i2).map_err(|e| MeshError::IoWrite {
                path: path.to_path_buf(),
                source: e,
            })?;
        }
    }

    writer.flush().map_err(|e| MeshError::IoWrite {
        path: path.to_path_buf(),
        source: e,
    })?;

    info!(
        "Saved {} vertices and {} faces to {:?}",
        mesh.vertices.len(),
        mesh.faces.len(),
        path
    );

    Ok(())
}

/// Load mesh from 3MF file.
///
/// 3MF is a ZIP archive containing XML files. The mesh data is in
/// 3D/3dmodel.model as indexed vertices and triangles.
fn load_3mf(path: &Path) -> MeshResult<Mesh> {
    let file = File::open(path).map_err(|e| MeshError::IoRead {
        path: path.to_path_buf(),
        source: e,
    })?;

    let mut archive = zip::ZipArchive::new(file).map_err(|e| MeshError::ParseError {
        path: path.to_path_buf(),
        details: format!("Invalid 3MF archive: {}", e),
    })?;

    // Find the model file (usually 3D/3dmodel.model)
    let model_path = find_3mf_model_path(&mut archive)?;

    let mut model_file = archive
        .by_name(&model_path)
        .map_err(|e| MeshError::ParseError {
            path: path.to_path_buf(),
            details: format!("Cannot open model file '{}': {}", model_path, e),
        })?;

    let mut xml_content = String::new();
    model_file
        .read_to_string(&mut xml_content)
        .map_err(|e| MeshError::IoRead {
            path: path.to_path_buf(),
            source: e,
        })?;

    parse_3mf_model(&xml_content, path)
}

/// Find the model file path in a 3MF archive.
fn find_3mf_model_path(archive: &mut zip::ZipArchive<File>) -> MeshResult<String> {
    // Common locations for the model file
    let candidates = ["3D/3dmodel.model", "3d/3dmodel.model", "3D/3DModel.model"];

    for candidate in candidates {
        if archive.by_name(candidate).is_ok() {
            return Ok(candidate.to_string());
        }
    }

    // Search for any .model file
    for i in 0..archive.len() {
        if let Ok(file) = archive.by_index(i) {
            let name = file.name().to_lowercase();
            if name.ends_with(".model") {
                return Ok(file.name().to_string());
            }
        }
    }

    Err(MeshError::ParseError {
        path: std::path::PathBuf::new(),
        details: "No model file found in 3MF archive".to_string(),
    })
}

/// Parse 3MF model XML content.
fn parse_3mf_model(xml: &str, path: &Path) -> MeshResult<Mesh> {
    use quick_xml::Reader;
    use quick_xml::events::Event;

    let mut reader = Reader::from_str(xml);
    reader.config_mut().trim_text(true);

    let mut mesh = Mesh::new();
    let mut in_vertices = false;
    let mut in_triangles = false;

    loop {
        match reader.read_event() {
            Ok(Event::Start(ref e)) | Ok(Event::Empty(ref e)) => {
                let local_name = e.local_name();
                match local_name.as_ref() {
                    b"vertices" => in_vertices = true,
                    b"triangles" => in_triangles = true,
                    b"vertex" if in_vertices => {
                        let mut x = 0.0f64;
                        let mut y = 0.0f64;
                        let mut z = 0.0f64;

                        for attr in e.attributes().flatten() {
                            let value = String::from_utf8_lossy(&attr.value);
                            match attr.key.local_name().as_ref() {
                                b"x" => x = value.parse().unwrap_or(0.0),
                                b"y" => y = value.parse().unwrap_or(0.0),
                                b"z" => z = value.parse().unwrap_or(0.0),
                                _ => {}
                            }
                        }
                        mesh.vertices.push(Vertex::from_coords(x, y, z));
                    }
                    b"triangle" if in_triangles => {
                        let mut v1 = 0u32;
                        let mut v2 = 0u32;
                        let mut v3 = 0u32;

                        for attr in e.attributes().flatten() {
                            let value = String::from_utf8_lossy(&attr.value);
                            match attr.key.local_name().as_ref() {
                                b"v1" => v1 = value.parse().unwrap_or(0),
                                b"v2" => v2 = value.parse().unwrap_or(0),
                                b"v3" => v3 = value.parse().unwrap_or(0),
                                _ => {}
                            }
                        }
                        mesh.faces.push([v1, v2, v3]);
                    }
                    _ => {}
                }
            }
            Ok(Event::End(ref e)) => {
                let local_name = e.local_name();
                match local_name.as_ref() {
                    b"vertices" => in_vertices = false,
                    b"triangles" => in_triangles = false,
                    _ => {}
                }
            }
            Ok(Event::Eof) => break,
            Err(e) => {
                return Err(MeshError::ParseError {
                    path: path.to_path_buf(),
                    details: format!("XML parse error: {}", e),
                });
            }
            _ => {}
        }
    }

    debug!(
        "3MF loaded: {} vertices, {} faces",
        mesh.vertices.len(),
        mesh.faces.len()
    );

    Ok(mesh)
}

/// Save mesh to 3MF file.
///
/// 3MF is a modern mesh format that:
/// - Preserves vertex indexing exactly (no deduplication issues)
/// - Uses ZIP compression for smaller files
/// - Is widely supported by slicers (PrusaSlicer, Cura, etc.)
/// - Stores units as millimeters by default
pub fn save_3mf(mesh: &Mesh, path: &Path) -> MeshResult<()> {
    info!("Saving mesh to {:?} (3MF format)", path);

    let file = File::create(path).map_err(|e| MeshError::IoWrite {
        path: path.to_path_buf(),
        source: e,
    })?;

    let mut zip = zip::ZipWriter::new(file);
    let options = zip::write::SimpleFileOptions::default()
        .compression_method(zip::CompressionMethod::Deflated);

    // Write content types file (required by 3MF spec)
    zip.start_file("[Content_Types].xml", options)
        .map_err(|e| MeshError::IoWrite {
            path: path.to_path_buf(),
            source: std::io::Error::other(e.to_string()),
        })?;
    zip.write_all(CONTENT_TYPES_XML.as_bytes())
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
    zip.write_all(RELS_XML.as_bytes())
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

    let model_xml = generate_3mf_model_xml(mesh);
    zip.write_all(model_xml.as_bytes())
        .map_err(|e| MeshError::IoWrite {
            path: path.to_path_buf(),
            source: e,
        })?;

    zip.finish().map_err(|e| MeshError::IoWrite {
        path: path.to_path_buf(),
        source: std::io::Error::other(e.to_string()),
    })?;

    info!(
        "Saved {} vertices and {} faces to {:?} (3MF)",
        mesh.vertices.len(),
        mesh.faces.len(),
        path
    );

    Ok(())
}

/// Generate 3MF model XML content.
fn generate_3mf_model_xml(mesh: &Mesh) -> String {
    let mut xml = String::with_capacity(mesh.vertices.len() * 50 + mesh.faces.len() * 40);

    // XML header and model element
    xml.push_str(
        r#"<?xml version="1.0" encoding="UTF-8"?>
<model unit="millimeter" xmlns="http://schemas.microsoft.com/3dmanufacturing/core/2015/02">
  <resources>
    <object id="1" type="model">
      <mesh>
        <vertices>
"#,
    );

    // Write vertices
    for v in &mesh.vertices {
        xml.push_str(&format!(
            "          <vertex x=\"{:.6}\" y=\"{:.6}\" z=\"{:.6}\"/>\n",
            v.position.x, v.position.y, v.position.z
        ));
    }

    xml.push_str("        </vertices>\n        <triangles>\n");

    // Write triangles
    for face in &mesh.faces {
        xml.push_str(&format!(
            "          <triangle v1=\"{}\" v2=\"{}\" v3=\"{}\"/>\n",
            face[0], face[1], face[2]
        ));
    }

    xml.push_str(
        r#"        </triangles>
      </mesh>
    </object>
  </resources>
  <build>
    <item objectid="1"/>
  </build>
</model>
"#,
    );

    xml
}

/// 3MF Content Types XML (required by spec).
const CONTENT_TYPES_XML: &str = r#"<?xml version="1.0" encoding="UTF-8"?>
<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types">
  <Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/>
  <Default Extension="model" ContentType="application/vnd.ms-package.3dmanufacturing-3dmodel+xml"/>
</Types>
"#;

/// 3MF Relationships XML (required by spec).
const RELS_XML: &str = r#"<?xml version="1.0" encoding="UTF-8"?>
<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">
  <Relationship Target="/3D/3dmodel.model" Id="rel0" Type="http://schemas.microsoft.com/3dmanufacturing/2013/01/3dmodel"/>
</Relationships>
"#;

// ============================================================================
// 3MF Extension Types
// ============================================================================

/// Beam cap mode for 3MF beam lattice extension.
///
/// Per the 3MF Beam Lattice Extension specification, beams can have different
/// cap geometries at their endpoints.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub enum BeamCap {
    /// Spherical cap (full sphere at endpoint).
    #[default]
    Sphere,
    /// Hemispherical cap (half sphere extending outward).
    Hemisphere,
    /// Flat/butt cap (no cap geometry).
    Butt,
}

impl BeamCap {
    /// Convert to 3MF XML attribute value.
    pub fn as_3mf_str(&self) -> &'static str {
        match self {
            BeamCap::Sphere => "sphere",
            BeamCap::Hemisphere => "hemisphere",
            BeamCap::Butt => "butt",
        }
    }
}

/// A single beam in a beam lattice structure.
///
/// Represents a conical frustum connecting two vertices with potentially
/// different radii at each end.
#[derive(Debug, Clone)]
pub struct Beam {
    /// Index of the first vertex.
    pub v1: u32,
    /// Index of the second vertex.
    pub v2: u32,
    /// Radius at the first vertex (mm).
    pub r1: f64,
    /// Radius at the second vertex (mm).
    pub r2: f64,
    /// Cap mode at the first vertex.
    pub cap1: BeamCap,
    /// Cap mode at the second vertex.
    pub cap2: BeamCap,
}

impl Beam {
    /// Create a new beam with uniform radius and default caps.
    pub fn new(v1: u32, v2: u32, radius: f64) -> Self {
        Self {
            v1,
            v2,
            r1: radius,
            r2: radius,
            cap1: BeamCap::default(),
            cap2: BeamCap::default(),
        }
    }

    /// Create a new beam with varying radii (tapered).
    pub fn tapered(v1: u32, v2: u32, r1: f64, r2: f64) -> Self {
        Self {
            v1,
            v2,
            r1,
            r2,
            cap1: BeamCap::default(),
            cap2: BeamCap::default(),
        }
    }

    /// Set cap modes for both endpoints.
    pub fn with_caps(mut self, cap1: BeamCap, cap2: BeamCap) -> Self {
        self.cap1 = cap1;
        self.cap2 = cap2;
        self
    }
}

/// A set of beams for organizational purposes.
///
/// Beam sets allow grouping beams for editing, selection, or material assignment.
/// They do not affect geometry.
#[derive(Debug, Clone, Default)]
pub struct BeamSet {
    /// Optional human-readable name.
    pub name: Option<String>,
    /// Optional unique identifier.
    pub identifier: Option<String>,
    /// Indices into the parent BeamLatticeData::beams vector.
    pub beam_indices: Vec<usize>,
}

/// Complete beam lattice data for 3MF export.
///
/// This structure holds the raw beam definitions that can be exported using
/// the 3MF Beam Lattice Extension, providing a more efficient representation
/// than triangulated mesh geometry for lattice structures.
#[derive(Debug, Clone)]
pub struct BeamLatticeData {
    /// Shared vertices (node positions) referenced by beams.
    pub vertices: Vec<nalgebra::Point3<f64>>,
    /// Beam definitions.
    pub beams: Vec<Beam>,
    /// Optional beam sets for grouping.
    pub beam_sets: Vec<BeamSet>,
    /// Default radius for beams (mm).
    pub default_radius: f64,
    /// Minimum segment length (for validation).
    pub min_length: f64,
    /// Default cap mode.
    pub default_cap: BeamCap,
}

impl Default for BeamLatticeData {
    fn default() -> Self {
        Self {
            vertices: Vec::new(),
            beams: Vec::new(),
            beam_sets: Vec::new(),
            default_radius: 0.5,
            min_length: 0.0001,
            default_cap: BeamCap::Sphere,
        }
    }
}

impl BeamLatticeData {
    /// Create new beam lattice data with specified default radius.
    pub fn new(default_radius: f64) -> Self {
        Self {
            default_radius,
            ..Default::default()
        }
    }

    /// Add a vertex and return its index.
    pub fn add_vertex(&mut self, point: nalgebra::Point3<f64>) -> u32 {
        let idx = self.vertices.len() as u32;
        self.vertices.push(point);
        idx
    }

    /// Add a beam between two existing vertices.
    pub fn add_beam(&mut self, v1: u32, v2: u32, radius: f64) -> &mut Self {
        self.beams.push(Beam::new(v1, v2, radius));
        self
    }

    /// Add a beam with the default radius.
    pub fn add_beam_default(&mut self, v1: u32, v2: u32) -> &mut Self {
        self.beams.push(Beam::new(v1, v2, self.default_radius));
        self
    }

    /// Check if the lattice data is empty.
    pub fn is_empty(&self) -> bool {
        self.beams.is_empty()
    }

    /// Get the number of beams.
    pub fn beam_count(&self) -> usize {
        self.beams.len()
    }
}

/// A color group for the 3MF Materials Extension.
///
/// Color groups allow per-vertex or per-face color assignment using
/// RGBA values. Each triangle can reference different colors at each vertex
/// for smooth color gradients.
#[derive(Debug, Clone)]
pub struct ColorGroup {
    /// Resource ID for this color group.
    pub id: u32,
    /// Colors in RGBA format (0-255 per channel).
    pub colors: Vec<(u8, u8, u8, u8)>,
}

impl ColorGroup {
    /// Create a new color group with the given resource ID.
    pub fn new(id: u32) -> Self {
        Self {
            id,
            colors: Vec::new(),
        }
    }

    /// Add a color and return its index within this group.
    pub fn add_color(&mut self, r: u8, g: u8, b: u8, a: u8) -> usize {
        let idx = self.colors.len();
        self.colors.push((r, g, b, a));
        idx
    }

    /// Add an opaque RGB color.
    pub fn add_rgb(&mut self, r: u8, g: u8, b: u8) -> usize {
        self.add_color(r, g, b, 255)
    }
}

/// Per-triangle color assignment for ColorGroup.
#[derive(Debug, Clone)]
pub struct TriangleColors {
    /// Color group resource ID (pid).
    pub color_group_id: u32,
    /// Color index for vertex 1.
    pub p1: usize,
    /// Color index for vertex 2.
    pub p2: usize,
    /// Color index for vertex 3.
    pub p3: usize,
}

/// Parameters for 3MF export with extensions.
///
/// This struct controls which 3MF extensions are enabled and provides data
/// for each extension type.
#[derive(Debug, Clone, Default)]
pub struct ThreeMfExportParams {
    /// Material zones to include in the export (basematerials extension).
    pub material_zones: Vec<crate::region::MaterialZone>,
    /// Whether to include mesh regions as separate components.
    pub include_regions: bool,
    /// Region map to export (if include_regions is true).
    pub region_map: Option<crate::region::RegionMap>,

    // ---- Beam Lattice Extension ----
    /// Beam lattice data for the 3MF Beam Lattice Extension.
    /// When set, exports lattice as beams instead of triangulated mesh.
    pub beam_lattice: Option<BeamLatticeData>,

    // ---- Color Group Extension ----
    /// Color groups for per-vertex/per-face coloring.
    pub color_groups: Vec<ColorGroup>,
    /// Per-triangle color assignments (maps triangle index to TriangleColors).
    pub triangle_colors: std::collections::HashMap<usize, TriangleColors>,

    // ---- Production Extension ----
    /// Whether to generate UUIDs for production extension.
    /// Requires the `3mf-production` feature.
    pub generate_uuids: bool,
}

impl ThreeMfExportParams {
    /// Create new export params with material zones.
    pub fn with_materials(zones: Vec<crate::region::MaterialZone>) -> Self {
        Self {
            material_zones: zones,
            ..Default::default()
        }
    }

    /// Create new export params with regions.
    pub fn with_regions(region_map: crate::region::RegionMap) -> Self {
        Self {
            include_regions: true,
            region_map: Some(region_map),
            ..Default::default()
        }
    }

    /// Add a material zone.
    pub fn add_material_zone(mut self, zone: crate::region::MaterialZone) -> Self {
        self.material_zones.push(zone);
        self
    }

    /// Set beam lattice data for export.
    pub fn with_beam_lattice(mut self, beam_lattice: BeamLatticeData) -> Self {
        self.beam_lattice = Some(beam_lattice);
        self
    }

    /// Add a color group.
    pub fn add_color_group(mut self, color_group: ColorGroup) -> Self {
        self.color_groups.push(color_group);
        self
    }

    /// Enable UUID generation for production extension.
    pub fn with_uuids(mut self, enable: bool) -> Self {
        self.generate_uuids = enable;
        self
    }

    /// Check if any extensions are enabled.
    pub fn has_extensions(&self) -> bool {
        !self.material_zones.is_empty()
            || self.beam_lattice.is_some()
            || !self.color_groups.is_empty()
            || self.generate_uuids
    }

    /// Check if beam lattice extension is enabled.
    pub fn has_beam_lattice(&self) -> bool {
        self.beam_lattice.as_ref().is_some_and(|bl| !bl.is_empty())
    }

    /// Check if color group extension is enabled.
    pub fn has_color_groups(&self) -> bool {
        !self.color_groups.is_empty()
    }
}

/// Result from loading a 3MF file with materials.
#[derive(Debug)]
pub struct ThreeMfLoadResult {
    /// The loaded mesh.
    pub mesh: Mesh,
    /// Material zones parsed from the file.
    pub material_zones: Vec<crate::region::MaterialZone>,
    /// Per-triangle material indices (index into material_zones).
    pub triangle_materials: Vec<Option<usize>>,
}

/// Save mesh to 3MF file with material zones.
///
/// This function exports the mesh with the 3MF Materials and Properties Extension,
/// allowing per-triangle material assignments based on the provided material zones.
///
/// # Arguments
/// * `mesh` - The mesh to save
/// * `path` - Output file path
/// * `params` - Export parameters including material zones
///
/// # Example
/// ```no_run
/// use mesh_repair::{Mesh, save_3mf_with_materials, ThreeMfExportParams};
/// use mesh_repair::region::{MaterialZone, MeshRegion};
///
/// let mesh = Mesh::new();
/// let region = MeshRegion::from_faces("heel", vec![0, 1, 2]);
/// let zone = MaterialZone::new(region, "TPU-95A")
///     .with_color(255, 128, 0)
///     .with_shore_hardness(95.0);
///
/// let params = ThreeMfExportParams::with_materials(vec![zone]);
/// save_3mf_with_materials(&mesh, std::path::Path::new("output.3mf"), &params).unwrap();
/// ```
pub fn save_3mf_with_materials(
    mesh: &Mesh,
    path: &Path,
    params: &ThreeMfExportParams,
) -> MeshResult<()> {
    info!(
        "Saving mesh to {:?} (3MF format with {} material zones)",
        path,
        params.material_zones.len()
    );

    let file = File::create(path).map_err(|e| MeshError::IoWrite {
        path: path.to_path_buf(),
        source: e,
    })?;

    let mut zip = zip::ZipWriter::new(file);
    let options = zip::write::SimpleFileOptions::default()
        .compression_method(zip::CompressionMethod::Deflated);

    // Write content types file (required by 3MF spec)
    zip.start_file("[Content_Types].xml", options)
        .map_err(|e| MeshError::IoWrite {
            path: path.to_path_buf(),
            source: std::io::Error::other(e.to_string()),
        })?;
    zip.write_all(CONTENT_TYPES_XML.as_bytes())
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
    zip.write_all(RELS_XML.as_bytes())
        .map_err(|e| MeshError::IoWrite {
            path: path.to_path_buf(),
            source: e,
        })?;

    // Write the model file with materials
    zip.start_file("3D/3dmodel.model", options)
        .map_err(|e| MeshError::IoWrite {
            path: path.to_path_buf(),
            source: std::io::Error::other(e.to_string()),
        })?;

    let model_xml = generate_3mf_model_xml_with_materials(mesh, params);
    zip.write_all(model_xml.as_bytes())
        .map_err(|e| MeshError::IoWrite {
            path: path.to_path_buf(),
            source: e,
        })?;

    zip.finish().map_err(|e| MeshError::IoWrite {
        path: path.to_path_buf(),
        source: std::io::Error::other(e.to_string()),
    })?;

    info!(
        "Saved {} vertices, {} faces, {} materials to {:?} (3MF)",
        mesh.vertices.len(),
        mesh.faces.len(),
        params.material_zones.len(),
        path
    );

    Ok(())
}

/// Generate 3MF model XML with materials extension.
fn generate_3mf_model_xml_with_materials(mesh: &Mesh, params: &ThreeMfExportParams) -> String {
    let has_materials = !params.material_zones.is_empty();

    let mut xml = String::with_capacity(mesh.vertices.len() * 50 + mesh.faces.len() * 50);

    // XML header and model element with materials namespace if needed
    if has_materials {
        xml.push_str(r#"<?xml version="1.0" encoding="UTF-8"?>
<model unit="millimeter" xmlns="http://schemas.microsoft.com/3dmanufacturing/core/2015/02" xmlns:m="http://schemas.microsoft.com/3dmanufacturing/material/2015/02">
  <resources>
"#);

        // Write basematerials resource
        xml.push_str("    <basematerials id=\"1\">\n");
        for zone in &params.material_zones {
            let (r, g, b) = zone.properties.color.unwrap_or((128, 128, 128));
            // 3MF uses sRGB hex color format
            xml.push_str(&format!(
                "      <base name=\"{}\" displaycolor=\"#{:02X}{:02X}{:02X}\"/>\n",
                escape_xml(&zone.material_name),
                r,
                g,
                b
            ));
        }
        xml.push_str("    </basematerials>\n");

        // Write mesh object with material references
        xml.push_str(
            r#"    <object id="2" type="model" pid="1" pindex="0">
      <mesh>
        <vertices>
"#,
        );
    } else {
        xml.push_str(
            r#"<?xml version="1.0" encoding="UTF-8"?>
<model unit="millimeter" xmlns="http://schemas.microsoft.com/3dmanufacturing/core/2015/02">
  <resources>
    <object id="1" type="model">
      <mesh>
        <vertices>
"#,
        );
    }

    // Write vertices
    for v in &mesh.vertices {
        xml.push_str(&format!(
            "          <vertex x=\"{:.6}\" y=\"{:.6}\" z=\"{:.6}\"/>\n",
            v.position.x, v.position.y, v.position.z
        ));
    }

    xml.push_str("        </vertices>\n        <triangles>\n");

    // Build face-to-material map
    let face_materials = build_face_material_map(mesh, params);

    // Write triangles with material references
    for (face_idx, face) in mesh.faces.iter().enumerate() {
        if has_materials {
            if let Some(mat_idx) = face_materials.get(&(face_idx as u32)) {
                // Triangle with material: pid references the basematerials group,
                // p1 is the material index within that group
                xml.push_str(&format!(
                    "          <triangle v1=\"{}\" v2=\"{}\" v3=\"{}\" pid=\"1\" p1=\"{}\"/>\n",
                    face[0], face[1], face[2], mat_idx
                ));
            } else {
                // Triangle without material assignment uses default (first material)
                xml.push_str(&format!(
                    "          <triangle v1=\"{}\" v2=\"{}\" v3=\"{}\" pid=\"1\" p1=\"0\"/>\n",
                    face[0], face[1], face[2]
                ));
            }
        } else {
            xml.push_str(&format!(
                "          <triangle v1=\"{}\" v2=\"{}\" v3=\"{}\"/>\n",
                face[0], face[1], face[2]
            ));
        }
    }

    if has_materials {
        xml.push_str(
            r#"        </triangles>
      </mesh>
    </object>
  </resources>
  <build>
    <item objectid="2"/>
  </build>
</model>
"#,
        );
    } else {
        xml.push_str(
            r#"        </triangles>
      </mesh>
    </object>
  </resources>
  <build>
    <item objectid="1"/>
  </build>
</model>
"#,
        );
    }

    xml
}

/// Build a map from face index to material zone index.
fn build_face_material_map(
    mesh: &Mesh,
    params: &ThreeMfExportParams,
) -> std::collections::HashMap<u32, usize> {
    use std::collections::HashMap;

    let mut face_materials: HashMap<u32, usize> = HashMap::new();

    for (mat_idx, zone) in params.material_zones.iter().enumerate() {
        // If the zone has faces directly assigned, use those
        if !zone.region.faces.is_empty() {
            for &face_idx in &zone.region.faces {
                face_materials.insert(face_idx, mat_idx);
            }
        } else if !zone.region.vertices.is_empty() {
            // Otherwise, find faces that have ALL vertices in this region
            for (face_idx, face) in mesh.faces.iter().enumerate() {
                let all_in_region = face.iter().all(|&v| zone.region.vertices.contains(&v));
                if all_in_region {
                    face_materials.insert(face_idx as u32, mat_idx);
                }
            }
        }
    }

    face_materials
}

/// Escape special XML characters.
fn escape_xml(s: &str) -> String {
    s.replace('&', "&amp;")
        .replace('<', "&lt;")
        .replace('>', "&gt;")
        .replace('"', "&quot;")
        .replace('\'', "&apos;")
}

// ============================================================================
// Extended 3MF Export with All Extensions
// ============================================================================

/// 3MF namespace constants.
mod threemf_namespaces {
    pub const CORE: &str = "http://schemas.microsoft.com/3dmanufacturing/core/2015/02";
    pub const MATERIALS: &str = "http://schemas.microsoft.com/3dmanufacturing/material/2015/02";
    pub const BEAMLATTICE: &str =
        "http://schemas.microsoft.com/3dmanufacturing/beamlattice/2017/02";
    pub const PRODUCTION: &str = "http://schemas.microsoft.com/3dmanufacturing/production/2015/06";
}

/// Save mesh to 3MF file with full extension support.
///
/// This is the unified export function that supports all 3MF extensions:
/// - Materials extension (basematerials, colorgroup)
/// - Beam lattice extension
/// - Production extension (UUIDs)
///
/// # Arguments
/// * `mesh` - The mesh to save
/// * `path` - Output file path
/// * `params` - Export parameters controlling which extensions to use
///
/// # Example
/// ```no_run
/// use mesh_repair::{Mesh, save_3mf_extended, ThreeMfExportParams, BeamLatticeData};
///
/// let mesh = Mesh::new();
/// let mut params = ThreeMfExportParams::default();
///
/// // Add beam lattice data
/// let mut beam_lattice = BeamLatticeData::new(0.5);
/// let v1 = beam_lattice.add_vertex(nalgebra::Point3::new(0.0, 0.0, 0.0));
/// let v2 = beam_lattice.add_vertex(nalgebra::Point3::new(10.0, 0.0, 0.0));
/// beam_lattice.add_beam_default(v1, v2);
/// params = params.with_beam_lattice(beam_lattice);
///
/// save_3mf_extended(&mesh, std::path::Path::new("output.3mf"), &params).unwrap();
/// ```
pub fn save_3mf_extended(mesh: &Mesh, path: &Path, params: &ThreeMfExportParams) -> MeshResult<()> {
    info!(
        "Saving mesh to {:?} (3MF extended format, beam_lattice={}, color_groups={}, uuids={})",
        path,
        params.has_beam_lattice(),
        params.has_color_groups(),
        params.generate_uuids
    );

    let file = File::create(path).map_err(|e| MeshError::IoWrite {
        path: path.to_path_buf(),
        source: e,
    })?;

    let mut zip = zip::ZipWriter::new(file);
    let options = zip::write::SimpleFileOptions::default()
        .compression_method(zip::CompressionMethod::Deflated);

    // Write content types file
    zip.start_file("[Content_Types].xml", options)
        .map_err(|e| MeshError::IoWrite {
            path: path.to_path_buf(),
            source: std::io::Error::other(e.to_string()),
        })?;
    zip.write_all(CONTENT_TYPES_XML.as_bytes())
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
    zip.write_all(RELS_XML.as_bytes())
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

    let model_xml = generate_3mf_extended_xml(mesh, params);
    zip.write_all(model_xml.as_bytes())
        .map_err(|e| MeshError::IoWrite {
            path: path.to_path_buf(),
            source: e,
        })?;

    zip.finish().map_err(|e| MeshError::IoWrite {
        path: path.to_path_buf(),
        source: std::io::Error::other(e.to_string()),
    })?;

    info!(
        "Saved mesh to {:?} (vertices={}, faces={}, beams={})",
        path,
        mesh.vertices.len(),
        mesh.faces.len(),
        params.beam_lattice.as_ref().map_or(0, |bl| bl.beam_count())
    );

    Ok(())
}

/// Generate 3MF model XML with all extension support.
fn generate_3mf_extended_xml(mesh: &Mesh, params: &ThreeMfExportParams) -> String {
    let has_materials = !params.material_zones.is_empty();
    let has_beam_lattice = params.has_beam_lattice();
    let has_color_groups = params.has_color_groups();
    let has_production = params.generate_uuids;

    let mut xml = String::with_capacity(
        mesh.vertices.len() * 50
            + mesh.faces.len() * 50
            + params
                .beam_lattice
                .as_ref()
                .map_or(0, |bl| bl.beams.len() * 80),
    );

    // Build namespace declarations
    let mut namespaces = format!("xmlns=\"{}\"", threemf_namespaces::CORE);
    if has_materials || has_color_groups {
        namespaces.push_str(&format!(" xmlns:m=\"{}\"", threemf_namespaces::MATERIALS));
    }
    if has_beam_lattice {
        namespaces.push_str(&format!(" xmlns:b=\"{}\"", threemf_namespaces::BEAMLATTICE));
    }
    if has_production {
        namespaces.push_str(&format!(" xmlns:p=\"{}\"", threemf_namespaces::PRODUCTION));
    }

    // XML header and model element
    xml.push_str(&format!(
        "<?xml version=\"1.0\" encoding=\"UTF-8\"?>\n<model unit=\"millimeter\" {}>\n",
        namespaces
    ));

    // Generate build UUID if production extension is enabled
    #[cfg(feature = "3mf-production")]
    let build_uuid = if has_production {
        Some(uuid::Uuid::new_v4().to_string())
    } else {
        None
    };
    #[cfg(not(feature = "3mf-production"))]
    let build_uuid: Option<String> = None;

    // Resources section
    xml.push_str("  <resources>\n");

    // Write basematerials if present
    if has_materials {
        xml.push_str("    <basematerials id=\"1\">\n");
        for zone in &params.material_zones {
            let (r, g, b) = zone.properties.color.unwrap_or((128, 128, 128));
            xml.push_str(&format!(
                "      <base name=\"{}\" displaycolor=\"#{:02X}{:02X}{:02X}\"/>\n",
                escape_xml(&zone.material_name),
                r,
                g,
                b
            ));
        }
        xml.push_str("    </basematerials>\n");
    }

    // Write color groups if present
    if has_color_groups {
        for cg in &params.color_groups {
            xml.push_str(&format!("    <m:colorgroup id=\"{}\">\n", cg.id));
            for (r, g, b, a) in &cg.colors {
                xml.push_str(&format!(
                    "      <m:color color=\"#{:02X}{:02X}{:02X}{:02X}\"/>\n",
                    r, g, b, a
                ));
            }
            xml.push_str("    </m:colorgroup>\n");
        }
    }

    // Determine object ID and generate UUID if needed
    let object_id = if has_materials { 2 } else { 1 };

    #[cfg(feature = "3mf-production")]
    let object_uuid = if has_production {
        Some(uuid::Uuid::new_v4().to_string())
    } else {
        None
    };
    #[cfg(not(feature = "3mf-production"))]
    let object_uuid: Option<String> = None;

    // Write object element
    if has_materials {
        if let Some(ref uuid) = object_uuid {
            xml.push_str(&format!(
                "    <object id=\"{}\" type=\"model\" pid=\"1\" pindex=\"0\" p:UUID=\"{}\">\n",
                object_id, uuid
            ));
        } else {
            xml.push_str(&format!(
                "    <object id=\"{}\" type=\"model\" pid=\"1\" pindex=\"0\">\n",
                object_id
            ));
        }
    } else if let Some(ref uuid) = object_uuid {
        xml.push_str(&format!(
            "    <object id=\"{}\" type=\"model\" p:UUID=\"{}\">\n",
            object_id, uuid
        ));
    } else {
        xml.push_str(&format!(
            "    <object id=\"{}\" type=\"model\">\n",
            object_id
        ));
    }

    // Mesh element
    xml.push_str("      <mesh>\n");

    // Write mesh vertices
    xml.push_str("        <vertices>\n");
    for v in &mesh.vertices {
        xml.push_str(&format!(
            "          <vertex x=\"{:.6}\" y=\"{:.6}\" z=\"{:.6}\"/>\n",
            v.position.x, v.position.y, v.position.z
        ));
    }
    xml.push_str("        </vertices>\n");

    // Write triangles
    xml.push_str("        <triangles>\n");
    let face_materials = build_face_material_map(mesh, params);
    for (face_idx, face) in mesh.faces.iter().enumerate() {
        // Check for color group assignment
        if let Some(tri_colors) = params.triangle_colors.get(&face_idx) {
            xml.push_str(&format!(
                "          <triangle v1=\"{}\" v2=\"{}\" v3=\"{}\" pid=\"{}\" p1=\"{}\" p2=\"{}\" p3=\"{}\"/>\n",
                face[0], face[1], face[2],
                tri_colors.color_group_id,
                tri_colors.p1, tri_colors.p2, tri_colors.p3
            ));
        } else if has_materials {
            if let Some(mat_idx) = face_materials.get(&(face_idx as u32)) {
                xml.push_str(&format!(
                    "          <triangle v1=\"{}\" v2=\"{}\" v3=\"{}\" pid=\"1\" p1=\"{}\"/>\n",
                    face[0], face[1], face[2], mat_idx
                ));
            } else {
                xml.push_str(&format!(
                    "          <triangle v1=\"{}\" v2=\"{}\" v3=\"{}\" pid=\"1\" p1=\"0\"/>\n",
                    face[0], face[1], face[2]
                ));
            }
        } else {
            xml.push_str(&format!(
                "          <triangle v1=\"{}\" v2=\"{}\" v3=\"{}\"/>\n",
                face[0], face[1], face[2]
            ));
        }
    }
    xml.push_str("        </triangles>\n");

    // Write beam lattice if present
    if let Some(ref beam_lattice) = params.beam_lattice {
        xml.push_str(&generate_beamlattice_xml(beam_lattice));
    }

    xml.push_str("      </mesh>\n");
    xml.push_str("    </object>\n");
    xml.push_str("  </resources>\n");

    // Build section
    if let Some(ref uuid) = build_uuid {
        xml.push_str(&format!("  <build p:UUID=\"{}\">\n", uuid));
    } else {
        xml.push_str("  <build>\n");
    }

    #[cfg(feature = "3mf-production")]
    let item_uuid = if has_production {
        Some(uuid::Uuid::new_v4().to_string())
    } else {
        None
    };
    #[cfg(not(feature = "3mf-production"))]
    let item_uuid: Option<String> = None;

    if let Some(ref uuid) = item_uuid {
        xml.push_str(&format!(
            "    <item objectid=\"{}\" p:UUID=\"{}\"/>\n",
            object_id, uuid
        ));
    } else {
        xml.push_str(&format!("    <item objectid=\"{}\"/>\n", object_id));
    }

    xml.push_str("  </build>\n");
    xml.push_str("</model>\n");

    xml
}

/// Generate beam lattice XML fragment.
fn generate_beamlattice_xml(beam_lattice: &BeamLatticeData) -> String {
    let mut xml = String::with_capacity(beam_lattice.beams.len() * 80 + 500);

    // Open beamlattice element with attributes
    xml.push_str(&format!(
        "        <b:beamlattice radius=\"{:.6}\" minlength=\"{:.6}\" cap=\"{}\"",
        beam_lattice.default_radius,
        beam_lattice.min_length,
        beam_lattice.default_cap.as_3mf_str()
    ));

    // Check if we need separate vertices (beam lattice uses its own vertex list)
    if !beam_lattice.vertices.is_empty() {
        xml.push_str(">\n");

        // Write beam lattice vertices
        xml.push_str("          <b:vertices>\n");
        for v in &beam_lattice.vertices {
            xml.push_str(&format!(
                "            <b:vertex x=\"{:.6}\" y=\"{:.6}\" z=\"{:.6}\"/>\n",
                v.x, v.y, v.z
            ));
        }
        xml.push_str("          </b:vertices>\n");

        // Write beams
        xml.push_str("          <b:beams>\n");
        for beam in &beam_lattice.beams {
            // Only include optional attributes if they differ from defaults
            let mut beam_attrs = format!("v1=\"{}\" v2=\"{}\"", beam.v1, beam.v2);

            // Add radii if they differ from default
            if (beam.r1 - beam_lattice.default_radius).abs() > 1e-9
                || (beam.r2 - beam_lattice.default_radius).abs() > 1e-9
            {
                beam_attrs.push_str(&format!(" r1=\"{:.6}\" r2=\"{:.6}\"", beam.r1, beam.r2));
            }

            // Add caps if they differ from default
            if beam.cap1 != beam_lattice.default_cap {
                beam_attrs.push_str(&format!(" cap1=\"{}\"", beam.cap1.as_3mf_str()));
            }
            if beam.cap2 != beam_lattice.default_cap {
                beam_attrs.push_str(&format!(" cap2=\"{}\"", beam.cap2.as_3mf_str()));
            }

            xml.push_str(&format!("            <b:beam {}/>\n", beam_attrs));
        }
        xml.push_str("          </b:beams>\n");

        // Write beam sets if present
        if !beam_lattice.beam_sets.is_empty() {
            xml.push_str("          <b:beamsets>\n");
            for beam_set in &beam_lattice.beam_sets {
                let mut set_attrs = String::new();
                if let Some(ref name) = beam_set.name {
                    set_attrs.push_str(&format!(" name=\"{}\"", escape_xml(name)));
                }
                if let Some(ref id) = beam_set.identifier {
                    set_attrs.push_str(&format!(" identifier=\"{}\"", escape_xml(id)));
                }
                xml.push_str(&format!("            <b:beamset{}>\n", set_attrs));
                for &idx in &beam_set.beam_indices {
                    xml.push_str(&format!("              <b:ref index=\"{}\"/>\n", idx));
                }
                xml.push_str("            </b:beamset>\n");
            }
            xml.push_str("          </b:beamsets>\n");
        }

        xml.push_str("        </b:beamlattice>\n");
    } else {
        // Empty beam lattice (just close the element)
        xml.push_str("/>\n");
    }

    xml
}

/// Load 3MF file with material zone information.
///
/// This function parses the 3MF Materials and Properties Extension to extract
/// per-triangle material assignments and reconstruct material zones.
///
/// # Arguments
/// * `path` - Path to the 3MF file
///
/// # Returns
/// A `ThreeMfLoadResult` containing the mesh, material zones, and per-triangle
/// material assignments.
pub fn load_3mf_with_materials(path: &Path) -> MeshResult<ThreeMfLoadResult> {
    use quick_xml::Reader;
    use quick_xml::events::Event;

    info!("Loading 3MF with materials from {:?}", path);

    let file = File::open(path).map_err(|e| MeshError::IoRead {
        path: path.to_path_buf(),
        source: e,
    })?;

    let mut archive = zip::ZipArchive::new(file).map_err(|e| MeshError::ParseError {
        path: path.to_path_buf(),
        details: format!("Invalid ZIP archive: {}", e),
    })?;

    // Find and read the 3D model file into a string
    let mut model_content = String::new();
    {
        let mut model_file =
            archive
                .by_name("3D/3dmodel.model")
                .map_err(|_| MeshError::ParseError {
                    path: path.to_path_buf(),
                    details: "3MF archive missing 3D/3dmodel.model".to_string(),
                })?;
        model_file
            .read_to_string(&mut model_content)
            .map_err(|e| MeshError::IoRead {
                path: path.to_path_buf(),
                source: e,
            })?;
    }

    let mut reader = Reader::from_str(&model_content);
    reader.config_mut().trim_text(true);

    let mut mesh = Mesh::new();

    // Material parsing state
    #[allow(clippy::type_complexity)]
    let mut base_materials: Vec<(String, Option<(u8, u8, u8)>)> = Vec::new(); // (name, color)
    let mut in_basematerials = false;
    let mut basematerials_id: Option<String> = None;

    // Triangle material assignments
    let mut triangle_materials: Vec<Option<usize>> = Vec::new();

    // Parse XML
    loop {
        match reader.read_event() {
            Ok(Event::Empty(ref e)) | Ok(Event::Start(ref e)) => {
                let local_name = e.local_name();
                match local_name.as_ref() {
                    b"vertex" => {
                        let mut x = 0.0_f64;
                        let mut y = 0.0_f64;
                        let mut z = 0.0_f64;

                        for attr in e.attributes().flatten() {
                            let value = String::from_utf8_lossy(&attr.value);
                            match attr.key.local_name().as_ref() {
                                b"x" => x = value.parse().unwrap_or(0.0),
                                b"y" => y = value.parse().unwrap_or(0.0),
                                b"z" => z = value.parse().unwrap_or(0.0),
                                _ => {}
                            }
                        }

                        mesh.vertices.push(Vertex::from_coords(x, y, z));
                    }
                    b"triangle" => {
                        let mut v1 = 0_u32;
                        let mut v2 = 0_u32;
                        let mut v3 = 0_u32;
                        let mut pid: Option<String> = None;
                        let mut p1: Option<usize> = None;

                        for attr in e.attributes().flatten() {
                            let value = String::from_utf8_lossy(&attr.value);
                            match attr.key.local_name().as_ref() {
                                b"v1" => v1 = value.parse().unwrap_or(0),
                                b"v2" => v2 = value.parse().unwrap_or(0),
                                b"v3" => v3 = value.parse().unwrap_or(0),
                                b"pid" => pid = Some(value.to_string()),
                                b"p1" => p1 = value.parse().ok(),
                                _ => {}
                            }
                        }

                        mesh.faces.push([v1, v2, v3]);

                        // Track material assignment
                        if pid.is_some() && basematerials_id.is_some() && p1.is_some() {
                            if pid == basematerials_id {
                                triangle_materials.push(p1);
                            } else {
                                triangle_materials.push(None);
                            }
                        } else {
                            triangle_materials.push(None);
                        }
                    }
                    b"basematerials" => {
                        in_basematerials = true;
                        for attr in e.attributes().flatten() {
                            let value = String::from_utf8_lossy(&attr.value);
                            if attr.key.local_name().as_ref() == b"id" {
                                basematerials_id = Some(value.to_string());
                            }
                        }
                    }
                    b"base" if in_basematerials => {
                        let mut name = String::new();
                        let mut color: Option<(u8, u8, u8)> = None;

                        for attr in e.attributes().flatten() {
                            let value = String::from_utf8_lossy(&attr.value);
                            match attr.key.local_name().as_ref() {
                                b"name" => name = value.to_string(),
                                b"displaycolor" => {
                                    color = parse_hex_color(&value);
                                }
                                _ => {}
                            }
                        }

                        base_materials.push((name, color));
                    }
                    _ => {}
                }
            }
            Ok(Event::End(ref e)) => {
                if e.local_name().as_ref() == b"basematerials" {
                    in_basematerials = false;
                }
            }
            Ok(Event::Eof) => break,
            Err(e) => {
                return Err(MeshError::ParseError {
                    path: path.to_path_buf(),
                    details: format!("XML parse error: {}", e),
                });
            }
            _ => {}
        }
    }

    // Build material zones from parsed data
    let mut material_zones = Vec::new();
    for (mat_idx, (name, color)) in base_materials.iter().enumerate() {
        // Collect faces assigned to this material
        let face_indices: Vec<u32> = triangle_materials
            .iter()
            .enumerate()
            .filter_map(|(face_idx, mat)| {
                if *mat == Some(mat_idx) {
                    Some(face_idx as u32)
                } else {
                    None
                }
            })
            .collect();

        if !face_indices.is_empty() || mat_idx < base_materials.len() {
            let region = crate::region::MeshRegion::from_faces(name.clone(), face_indices);
            let mut zone = crate::region::MaterialZone::new(region, name.clone());
            if let Some((r, g, b)) = color {
                zone = zone.with_color(*r, *g, *b);
            }
            material_zones.push(zone);
        }
    }

    debug!(
        "3MF loaded with materials: {} vertices, {} faces, {} materials",
        mesh.vertices.len(),
        mesh.faces.len(),
        material_zones.len()
    );

    Ok(ThreeMfLoadResult {
        mesh,
        material_zones,
        triangle_materials,
    })
}

/// Parse a hex color string like "#FF8000" to (r, g, b).
fn parse_hex_color(s: &str) -> Option<(u8, u8, u8)> {
    let s = s.trim_start_matches('#');
    if s.len() != 6 {
        return None;
    }

    let r = u8::from_str_radix(&s[0..2], 16).ok()?;
    let g = u8::from_str_radix(&s[2..4], 16).ok()?;
    let b = u8::from_str_radix(&s[4..6], 16).ok()?;

    Some((r, g, b))
}

/// Save mesh to PLY file (ASCII format for maximum compatibility).
///
/// PLY (Polygon File Format) is widely supported by 3D scanning software,
/// point cloud libraries (PCL), and mesh processing tools like MeshLab.
///
/// The output includes:
/// - Vertex positions (x, y, z as float32)
/// - Vertex normals if present (nx, ny, nz as float32)
/// - Face vertex indices (as a list property)
///
/// Uses ASCII format for maximum compatibility across tools.
/// For binary format (smaller files), use `save_ply_binary()`.
pub fn save_ply(mesh: &Mesh, path: &Path) -> MeshResult<()> {
    // Default to ASCII for maximum compatibility
    save_ply_ascii(mesh, path)
}

/// Save mesh to PLY file (binary little-endian format).
///
/// Binary format is more compact but may have compatibility issues with some tools.
/// Use `save_ply()` (ASCII) if you encounter issues.
pub fn save_ply_binary(mesh: &Mesh, path: &Path) -> MeshResult<()> {
    use ply_rs::ply::{
        Addable, DefaultElement, ElementDef, Encoding, Ply, Property, PropertyDef, PropertyType,
        ScalarType,
    };
    use ply_rs::writer::Writer;

    info!("Saving mesh to {:?} (PLY binary format)", path);

    let mut ply = Ply::<DefaultElement>::new();

    // Set encoding to binary little-endian for efficiency
    ply.header.encoding = Encoding::BinaryLittleEndian;

    // Check if we have normals and colors
    let has_normals = mesh.vertices.iter().any(|v| v.normal.is_some());
    let has_colors = mesh.vertices.iter().any(|v| v.color.is_some());

    // Define vertex element
    let mut vertex_def = ElementDef::new("vertex".to_string());
    vertex_def.properties.add(PropertyDef::new(
        "x".to_string(),
        PropertyType::Scalar(ScalarType::Float),
    ));
    vertex_def.properties.add(PropertyDef::new(
        "y".to_string(),
        PropertyType::Scalar(ScalarType::Float),
    ));
    vertex_def.properties.add(PropertyDef::new(
        "z".to_string(),
        PropertyType::Scalar(ScalarType::Float),
    ));
    if has_normals {
        vertex_def.properties.add(PropertyDef::new(
            "nx".to_string(),
            PropertyType::Scalar(ScalarType::Float),
        ));
        vertex_def.properties.add(PropertyDef::new(
            "ny".to_string(),
            PropertyType::Scalar(ScalarType::Float),
        ));
        vertex_def.properties.add(PropertyDef::new(
            "nz".to_string(),
            PropertyType::Scalar(ScalarType::Float),
        ));
    }
    if has_colors {
        vertex_def.properties.add(PropertyDef::new(
            "red".to_string(),
            PropertyType::Scalar(ScalarType::UChar),
        ));
        vertex_def.properties.add(PropertyDef::new(
            "green".to_string(),
            PropertyType::Scalar(ScalarType::UChar),
        ));
        vertex_def.properties.add(PropertyDef::new(
            "blue".to_string(),
            PropertyType::Scalar(ScalarType::UChar),
        ));
    }
    vertex_def.count = mesh.vertices.len();
    ply.header.elements.add(vertex_def);

    // Define face element
    let mut face_def = ElementDef::new("face".to_string());
    face_def.properties.add(PropertyDef::new(
        "vertex_indices".to_string(),
        PropertyType::List(ScalarType::UChar, ScalarType::Int),
    ));
    face_def.count = mesh.faces.len();
    ply.header.elements.add(face_def);

    // Add vertex data
    let mut vertices_payload: Vec<DefaultElement> = Vec::with_capacity(mesh.vertices.len());
    for v in &mesh.vertices {
        let mut element = DefaultElement::new();
        element.insert("x".to_string(), Property::Float(v.position.x as f32));
        element.insert("y".to_string(), Property::Float(v.position.y as f32));
        element.insert("z".to_string(), Property::Float(v.position.z as f32));
        if has_normals {
            let n = v.normal.unwrap_or(nalgebra::Vector3::new(0.0, 0.0, 0.0));
            element.insert("nx".to_string(), Property::Float(n.x as f32));
            element.insert("ny".to_string(), Property::Float(n.y as f32));
            element.insert("nz".to_string(), Property::Float(n.z as f32));
        }
        if has_colors {
            let c = v.color.unwrap_or(crate::VertexColor::new(255, 255, 255));
            element.insert("red".to_string(), Property::UChar(c.r));
            element.insert("green".to_string(), Property::UChar(c.g));
            element.insert("blue".to_string(), Property::UChar(c.b));
        }
        vertices_payload.push(element);
    }
    ply.payload.insert("vertex".to_string(), vertices_payload);

    // Add face data
    let mut faces_payload: Vec<DefaultElement> = Vec::with_capacity(mesh.faces.len());
    for face in &mesh.faces {
        let mut element = DefaultElement::new();
        element.insert(
            "vertex_indices".to_string(),
            Property::ListInt(vec![face[0] as i32, face[1] as i32, face[2] as i32]),
        );
        faces_payload.push(element);
    }
    ply.payload.insert("face".to_string(), faces_payload);

    // Ensure header counts match payload (required for ply-rs)
    ply.make_consistent().map_err(|e| MeshError::IoWrite {
        path: path.to_path_buf(),
        source: std::io::Error::other(format!("PLY consistency error: {:?}", e)),
    })?;

    // Write to file
    let file = File::create(path).map_err(|e| MeshError::IoWrite {
        path: path.to_path_buf(),
        source: e,
    })?;
    let mut writer = BufWriter::new(file);

    let ply_writer = Writer::new();
    ply_writer
        .write_ply(&mut writer, &mut ply)
        .map_err(|e| MeshError::IoWrite {
            path: path.to_path_buf(),
            source: std::io::Error::other(format!("PLY write error: {:?}", e)),
        })?;

    writer.flush().map_err(|e| MeshError::IoWrite {
        path: path.to_path_buf(),
        source: e,
    })?;

    info!(
        "Saved {} vertices and {} faces to {:?} (PLY binary)",
        mesh.vertices.len(),
        mesh.faces.len(),
        path
    );

    Ok(())
}

/// Save mesh to ASCII PLY file.
///
/// ASCII format is human-readable but larger than binary.
/// Useful for debugging or when binary format is not supported.
pub fn save_ply_ascii(mesh: &Mesh, path: &Path) -> MeshResult<()> {
    use ply_rs::ply::{
        Addable, DefaultElement, ElementDef, Encoding, Ply, Property, PropertyDef, PropertyType,
        ScalarType,
    };
    use ply_rs::writer::Writer;

    info!("Saving mesh to {:?} (PLY ASCII format)", path);

    let mut ply = Ply::<DefaultElement>::new();

    // Set encoding to ASCII
    ply.header.encoding = Encoding::Ascii;

    // Check if we have normals and colors
    let has_normals = mesh.vertices.iter().any(|v| v.normal.is_some());
    let has_colors = mesh.vertices.iter().any(|v| v.color.is_some());

    // Define vertex element
    let mut vertex_def = ElementDef::new("vertex".to_string());
    vertex_def.properties.add(PropertyDef::new(
        "x".to_string(),
        PropertyType::Scalar(ScalarType::Float),
    ));
    vertex_def.properties.add(PropertyDef::new(
        "y".to_string(),
        PropertyType::Scalar(ScalarType::Float),
    ));
    vertex_def.properties.add(PropertyDef::new(
        "z".to_string(),
        PropertyType::Scalar(ScalarType::Float),
    ));
    if has_normals {
        vertex_def.properties.add(PropertyDef::new(
            "nx".to_string(),
            PropertyType::Scalar(ScalarType::Float),
        ));
        vertex_def.properties.add(PropertyDef::new(
            "ny".to_string(),
            PropertyType::Scalar(ScalarType::Float),
        ));
        vertex_def.properties.add(PropertyDef::new(
            "nz".to_string(),
            PropertyType::Scalar(ScalarType::Float),
        ));
    }
    if has_colors {
        vertex_def.properties.add(PropertyDef::new(
            "red".to_string(),
            PropertyType::Scalar(ScalarType::UChar),
        ));
        vertex_def.properties.add(PropertyDef::new(
            "green".to_string(),
            PropertyType::Scalar(ScalarType::UChar),
        ));
        vertex_def.properties.add(PropertyDef::new(
            "blue".to_string(),
            PropertyType::Scalar(ScalarType::UChar),
        ));
    }
    vertex_def.count = mesh.vertices.len();
    ply.header.elements.add(vertex_def);

    // Define face element
    let mut face_def = ElementDef::new("face".to_string());
    face_def.properties.add(PropertyDef::new(
        "vertex_indices".to_string(),
        PropertyType::List(ScalarType::UChar, ScalarType::Int),
    ));
    face_def.count = mesh.faces.len();
    ply.header.elements.add(face_def);

    // Add vertex data
    let mut vertices_payload: Vec<DefaultElement> = Vec::with_capacity(mesh.vertices.len());
    for v in &mesh.vertices {
        let mut element = DefaultElement::new();
        element.insert("x".to_string(), Property::Float(v.position.x as f32));
        element.insert("y".to_string(), Property::Float(v.position.y as f32));
        element.insert("z".to_string(), Property::Float(v.position.z as f32));
        if has_normals {
            let n = v.normal.unwrap_or(nalgebra::Vector3::new(0.0, 0.0, 0.0));
            element.insert("nx".to_string(), Property::Float(n.x as f32));
            element.insert("ny".to_string(), Property::Float(n.y as f32));
            element.insert("nz".to_string(), Property::Float(n.z as f32));
        }
        if has_colors {
            let c = v.color.unwrap_or(crate::VertexColor::new(255, 255, 255));
            element.insert("red".to_string(), Property::UChar(c.r));
            element.insert("green".to_string(), Property::UChar(c.g));
            element.insert("blue".to_string(), Property::UChar(c.b));
        }
        vertices_payload.push(element);
    }
    ply.payload.insert("vertex".to_string(), vertices_payload);

    // Add face data
    let mut faces_payload: Vec<DefaultElement> = Vec::with_capacity(mesh.faces.len());
    for face in &mesh.faces {
        let mut element = DefaultElement::new();
        element.insert(
            "vertex_indices".to_string(),
            Property::ListInt(vec![face[0] as i32, face[1] as i32, face[2] as i32]),
        );
        faces_payload.push(element);
    }
    ply.payload.insert("face".to_string(), faces_payload);

    // Ensure header counts match payload (required for ply-rs)
    ply.make_consistent().map_err(|e| MeshError::IoWrite {
        path: path.to_path_buf(),
        source: std::io::Error::other(format!("PLY consistency error: {:?}", e)),
    })?;

    // Write to file
    let file = File::create(path).map_err(|e| MeshError::IoWrite {
        path: path.to_path_buf(),
        source: e,
    })?;
    let mut writer = BufWriter::new(file);

    let ply_writer = Writer::new();
    ply_writer
        .write_ply(&mut writer, &mut ply)
        .map_err(|e| MeshError::IoWrite {
            path: path.to_path_buf(),
            source: std::io::Error::other(format!("PLY write error: {:?}", e)),
        })?;

    writer.flush().map_err(|e| MeshError::IoWrite {
        path: path.to_path_buf(),
        source: e,
    })?;

    info!(
        "Saved {} vertices and {} faces to {:?} (PLY ASCII)",
        mesh.vertices.len(),
        mesh.faces.len(),
        path
    );

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use nalgebra::Point3;
    use std::io::Write;
    use tempfile::NamedTempFile;

    fn create_test_stl() -> NamedTempFile {
        let mut file = NamedTempFile::with_suffix(".stl").unwrap();

        // ASCII STL with a single triangle
        writeln!(file, "solid test").unwrap();
        writeln!(file, "  facet normal 0 0 1").unwrap();
        writeln!(file, "    outer loop").unwrap();
        writeln!(file, "      vertex 0 0 0").unwrap();
        writeln!(file, "      vertex 100 0 0").unwrap();
        writeln!(file, "      vertex 0 100 0").unwrap();
        writeln!(file, "    endloop").unwrap();
        writeln!(file, "  endfacet").unwrap();
        writeln!(file, "endsolid test").unwrap();

        file
    }

    #[test]
    fn test_format_detection() {
        assert_eq!(
            MeshFormat::from_path(Path::new("test.stl")),
            Some(MeshFormat::Stl)
        );
        assert_eq!(
            MeshFormat::from_path(Path::new("test.STL")),
            Some(MeshFormat::Stl)
        );
        assert_eq!(
            MeshFormat::from_path(Path::new("test.obj")),
            Some(MeshFormat::Obj)
        );
        assert_eq!(
            MeshFormat::from_path(Path::new("test.ply")),
            Some(MeshFormat::Ply)
        );
        assert_eq!(
            MeshFormat::from_path(Path::new("test.PLY")),
            Some(MeshFormat::Ply)
        );
        assert_eq!(MeshFormat::from_path(Path::new("test.xyz")), None);
    }

    #[test]
    fn test_load_stl() {
        let file = create_test_stl();
        let mesh = load_mesh(file.path()).expect("should load");

        assert_eq!(mesh.vertex_count(), 3);
        assert_eq!(mesh.face_count(), 1);

        let (min, max) = mesh.bounds().unwrap();
        assert_eq!(min, Point3::new(0.0, 0.0, 0.0));
        assert_eq!(max, Point3::new(100.0, 100.0, 0.0));
    }

    #[test]
    fn test_save_and_reload_stl() {
        // Create a simple mesh
        let mut mesh = Mesh::new();
        mesh.vertices.push(Vertex::from_coords(0.0, 0.0, 0.0));
        mesh.vertices.push(Vertex::from_coords(10.0, 0.0, 0.0));
        mesh.vertices.push(Vertex::from_coords(0.0, 10.0, 0.0));
        mesh.vertices.push(Vertex::from_coords(0.0, 0.0, 10.0));
        mesh.faces.push([0, 1, 2]);
        mesh.faces.push([0, 2, 3]);
        mesh.faces.push([0, 3, 1]);
        mesh.faces.push([1, 3, 2]);

        // Save to temp file
        let file = NamedTempFile::with_suffix(".stl").unwrap();
        save_stl(&mesh, file.path()).expect("should save");

        // Reload
        let reloaded = load_mesh(file.path()).expect("should reload");

        assert_eq!(reloaded.vertex_count(), 4);
        assert_eq!(reloaded.face_count(), 4);
    }

    #[test]
    fn test_save_and_reload_obj() {
        use nalgebra::Vector3;

        // Create a simple mesh with vertex attributes
        let mut mesh = Mesh::new();

        // Add vertices with attributes
        let mut v0 = Vertex::from_coords(0.0, 0.0, 0.0);
        v0.normal = Some(Vector3::new(0.0, 0.0, 1.0));
        v0.tag = Some(1);
        v0.offset = Some(2.5);
        mesh.vertices.push(v0);

        let mut v1 = Vertex::from_coords(10.0, 0.0, 0.0);
        v1.normal = Some(Vector3::new(1.0, 0.0, 0.0));
        v1.tag = Some(2);
        v1.offset = Some(3.0);
        mesh.vertices.push(v1);

        let mut v2 = Vertex::from_coords(0.0, 10.0, 0.0);
        v2.normal = Some(Vector3::new(0.0, 1.0, 0.0));
        v2.tag = Some(3);
        v2.offset = Some(2.0);
        mesh.vertices.push(v2);

        let mut v3 = Vertex::from_coords(0.0, 0.0, 10.0);
        v3.normal = Some(Vector3::new(-1.0, 0.0, 0.0));
        mesh.vertices.push(v3);

        mesh.faces.push([0, 1, 2]);
        mesh.faces.push([0, 2, 3]);
        mesh.faces.push([0, 3, 1]);
        mesh.faces.push([1, 3, 2]);

        // Save to OBJ
        let file = NamedTempFile::with_suffix(".obj").unwrap();
        save_obj(&mesh, file.path()).expect("should save");

        // Reload
        let reloaded = load_mesh(file.path()).expect("should reload");

        // OBJ preserves exact vertex count and order
        assert_eq!(reloaded.vertex_count(), 4);
        assert_eq!(reloaded.face_count(), 4);

        // Verify vertex positions are preserved exactly
        for (i, (orig, loaded)) in mesh
            .vertices
            .iter()
            .zip(reloaded.vertices.iter())
            .enumerate()
        {
            let pos_diff = (orig.position - loaded.position).norm();
            assert!(
                pos_diff < 1e-5,
                "Vertex {} position mismatch: {:?} vs {:?}",
                i,
                orig.position,
                loaded.position
            );
        }

        // Verify face indices are preserved
        for (i, (orig, loaded)) in mesh.faces.iter().zip(reloaded.faces.iter()).enumerate() {
            assert_eq!(orig, loaded, "Face {} indices mismatch", i);
        }
    }

    #[test]
    fn test_obj_vertex_index_preservation() {
        // This test specifically verifies that OBJ preserves vertex indices
        // unlike STL which re-orders vertices during save/load

        let mut mesh = Mesh::new();

        // Create vertices in a specific order
        for i in 0..10 {
            mesh.vertices.push(Vertex::from_coords(
                i as f64 * 10.0,
                (i % 3) as f64 * 5.0,
                (i / 3) as f64 * 7.0,
            ));
        }

        // Create some faces referencing specific vertices
        mesh.faces.push([0, 1, 2]);
        mesh.faces.push([3, 4, 5]);
        mesh.faces.push([6, 7, 8]);
        mesh.faces.push([0, 5, 9]);

        // Save and reload OBJ
        let obj_file = NamedTempFile::with_suffix(".obj").unwrap();
        save_obj(&mesh, obj_file.path()).expect("should save obj");
        let obj_reloaded = load_mesh(obj_file.path()).expect("should reload obj");

        // Save and reload STL for comparison
        let stl_file = NamedTempFile::with_suffix(".stl").unwrap();
        save_stl(&mesh, stl_file.path()).expect("should save stl");
        let _stl_reloaded = load_mesh(stl_file.path()).expect("should reload stl");

        // OBJ should preserve exact vertex count
        assert_eq!(
            obj_reloaded.vertex_count(),
            mesh.vertex_count(),
            "OBJ should preserve vertex count"
        );

        // STL may have different vertex count due to deduplication
        // (it duplicates vertices per-triangle, then deduplicates)

        // OBJ should preserve exact face indices
        for (i, (orig, loaded)) in mesh.faces.iter().zip(obj_reloaded.faces.iter()).enumerate() {
            assert_eq!(orig, loaded, "OBJ face {} indices should match", i);
        }

        // Verify we can track a specific vertex through OBJ save/load
        let target_vertex_idx = 5;
        let orig_pos = mesh.vertices[target_vertex_idx].position;
        let loaded_pos = obj_reloaded.vertices[target_vertex_idx].position;
        let diff = (orig_pos - loaded_pos).norm();
        assert!(
            diff < 1e-5,
            "Vertex {} should be at same index after OBJ reload",
            target_vertex_idx
        );
    }

    fn create_test_ply_ascii() -> NamedTempFile {
        let mut file = NamedTempFile::with_suffix(".ply").unwrap();

        // ASCII PLY with a single triangle
        writeln!(file, "ply").unwrap();
        writeln!(file, "format ascii 1.0").unwrap();
        writeln!(file, "element vertex 3").unwrap();
        writeln!(file, "property float x").unwrap();
        writeln!(file, "property float y").unwrap();
        writeln!(file, "property float z").unwrap();
        writeln!(file, "element face 1").unwrap();
        writeln!(file, "property list uchar int vertex_indices").unwrap();
        writeln!(file, "end_header").unwrap();
        writeln!(file, "0 0 0").unwrap();
        writeln!(file, "100 0 0").unwrap();
        writeln!(file, "0 100 0").unwrap();
        writeln!(file, "3 0 1 2").unwrap();

        file
    }

    #[test]
    fn test_load_ply_ascii() {
        let file = create_test_ply_ascii();
        let mesh = load_mesh(file.path()).expect("should load PLY");

        assert_eq!(mesh.vertex_count(), 3);
        assert_eq!(mesh.face_count(), 1);

        let (min, max) = mesh.bounds().unwrap();
        assert_eq!(min, Point3::new(0.0, 0.0, 0.0));
        assert_eq!(max, Point3::new(100.0, 100.0, 0.0));
    }

    #[test]
    fn test_save_and_reload_ply() {
        // Create a simple mesh
        let mut mesh = Mesh::new();
        mesh.vertices.push(Vertex::from_coords(0.0, 0.0, 0.0));
        mesh.vertices.push(Vertex::from_coords(10.0, 0.0, 0.0));
        mesh.vertices.push(Vertex::from_coords(0.0, 10.0, 0.0));
        mesh.vertices.push(Vertex::from_coords(0.0, 0.0, 10.0));
        mesh.faces.push([0, 1, 2]);
        mesh.faces.push([0, 2, 3]);
        mesh.faces.push([0, 3, 1]);
        mesh.faces.push([1, 3, 2]);

        // Save to temp file (default format = ASCII)
        let file = NamedTempFile::with_suffix(".ply").unwrap();
        save_ply(&mesh, file.path()).expect("should save PLY");

        // Reload
        let reloaded = load_mesh(file.path()).expect("should reload PLY");

        assert_eq!(reloaded.vertex_count(), 4);
        assert_eq!(reloaded.face_count(), 4);

        // Verify vertex positions are preserved
        for (i, (orig, loaded)) in mesh
            .vertices
            .iter()
            .zip(reloaded.vertices.iter())
            .enumerate()
        {
            let pos_diff = (orig.position - loaded.position).norm();
            assert!(
                pos_diff < 1e-5,
                "PLY vertex {} position mismatch: {:?} vs {:?}",
                i,
                orig.position,
                loaded.position
            );
        }

        // Verify face indices are preserved
        for (i, (orig, loaded)) in mesh.faces.iter().zip(reloaded.faces.iter()).enumerate() {
            assert_eq!(orig, loaded, "PLY face {} indices mismatch", i);
        }
    }

    #[test]
    fn test_save_and_reload_ply_explicit_ascii() {
        // Create a simple mesh
        let mut mesh = Mesh::new();
        mesh.vertices.push(Vertex::from_coords(0.0, 0.0, 0.0));
        mesh.vertices.push(Vertex::from_coords(10.0, 0.0, 0.0));
        mesh.vertices.push(Vertex::from_coords(0.0, 10.0, 0.0));
        mesh.vertices.push(Vertex::from_coords(0.0, 0.0, 10.0));
        mesh.faces.push([0, 1, 2]);
        mesh.faces.push([0, 2, 3]);
        mesh.faces.push([0, 3, 1]);
        mesh.faces.push([1, 3, 2]);

        // Save to temp file (ASCII format)
        let file = NamedTempFile::with_suffix(".ply").unwrap();
        save_ply_ascii(&mesh, file.path()).expect("should save PLY ASCII");

        // Reload
        let reloaded = load_mesh(file.path()).expect("should reload PLY ASCII");

        assert_eq!(reloaded.vertex_count(), 4);
        assert_eq!(reloaded.face_count(), 4);

        // Verify vertex positions are preserved
        for (i, (orig, loaded)) in mesh
            .vertices
            .iter()
            .zip(reloaded.vertices.iter())
            .enumerate()
        {
            let pos_diff = (orig.position - loaded.position).norm();
            assert!(
                pos_diff < 1e-5,
                "PLY ASCII vertex {} position mismatch: {:?} vs {:?}",
                i,
                orig.position,
                loaded.position
            );
        }
    }

    #[test]
    fn test_ply_with_normals() {
        use nalgebra::Vector3;

        // Create a mesh with normals
        let mut mesh = Mesh::new();

        let mut v0 = Vertex::from_coords(0.0, 0.0, 0.0);
        v0.normal = Some(Vector3::new(0.0, 0.0, 1.0));
        mesh.vertices.push(v0);

        let mut v1 = Vertex::from_coords(10.0, 0.0, 0.0);
        v1.normal = Some(Vector3::new(1.0, 0.0, 0.0));
        mesh.vertices.push(v1);

        let mut v2 = Vertex::from_coords(0.0, 10.0, 0.0);
        v2.normal = Some(Vector3::new(0.0, 1.0, 0.0));
        mesh.vertices.push(v2);

        mesh.faces.push([0, 1, 2]);

        // Save to PLY (binary)
        let file = NamedTempFile::with_suffix(".ply").unwrap();
        save_ply(&mesh, file.path()).expect("should save PLY with normals");

        // Reload
        let reloaded = load_mesh(file.path()).expect("should reload PLY with normals");

        assert_eq!(reloaded.vertex_count(), 3);
        assert_eq!(reloaded.face_count(), 1);

        // Verify normals are preserved
        for (i, (orig, loaded)) in mesh
            .vertices
            .iter()
            .zip(reloaded.vertices.iter())
            .enumerate()
        {
            let orig_n = orig.normal.expect("original should have normal");
            let loaded_n = loaded.normal.expect("loaded should have normal");
            let diff = (orig_n - loaded_n).norm();
            assert!(
                diff < 1e-5,
                "PLY vertex {} normal mismatch: {:?} vs {:?}",
                i,
                orig_n,
                loaded_n
            );
        }
    }

    #[test]
    fn test_ply_vertex_index_preservation() {
        // Verify PLY preserves exact vertex indices like OBJ
        let mut mesh = Mesh::new();

        // Create vertices in a specific order
        for i in 0..10 {
            mesh.vertices.push(Vertex::from_coords(
                i as f64 * 10.0,
                (i % 3) as f64 * 5.0,
                (i / 3) as f64 * 7.0,
            ));
        }

        // Create faces referencing specific vertices
        mesh.faces.push([0, 1, 2]);
        mesh.faces.push([3, 4, 5]);
        mesh.faces.push([6, 7, 8]);
        mesh.faces.push([0, 5, 9]);

        // Save and reload PLY
        let file = NamedTempFile::with_suffix(".ply").unwrap();
        save_ply(&mesh, file.path()).expect("should save PLY");
        let reloaded = load_mesh(file.path()).expect("should reload PLY");

        // PLY should preserve exact vertex count
        assert_eq!(
            reloaded.vertex_count(),
            mesh.vertex_count(),
            "PLY should preserve vertex count"
        );

        // PLY should preserve exact face indices
        for (i, (orig, loaded)) in mesh.faces.iter().zip(reloaded.faces.iter()).enumerate() {
            assert_eq!(orig, loaded, "PLY face {} indices should match", i);
        }

        // Verify specific vertex tracking
        let target_vertex_idx = 5;
        let orig_pos = mesh.vertices[target_vertex_idx].position;
        let loaded_pos = reloaded.vertices[target_vertex_idx].position;
        let diff = (orig_pos - loaded_pos).norm();
        assert!(
            diff < 1e-5,
            "Vertex {} should be at same index after PLY reload",
            target_vertex_idx
        );
    }

    #[test]
    fn test_ply_with_colors() {
        use crate::VertexColor;

        // Create a mesh with vertex colors
        let mut mesh = Mesh::new();

        let mut v0 = Vertex::from_coords(0.0, 0.0, 0.0);
        v0.color = Some(VertexColor::new(255, 0, 0)); // Red
        mesh.vertices.push(v0);

        let mut v1 = Vertex::from_coords(10.0, 0.0, 0.0);
        v1.color = Some(VertexColor::new(0, 255, 0)); // Green
        mesh.vertices.push(v1);

        let mut v2 = Vertex::from_coords(0.0, 10.0, 0.0);
        v2.color = Some(VertexColor::new(0, 0, 255)); // Blue
        mesh.vertices.push(v2);

        mesh.faces.push([0, 1, 2]);

        // Save to PLY (ASCII)
        let file = NamedTempFile::with_suffix(".ply").unwrap();
        save_ply(&mesh, file.path()).expect("should save PLY with colors");

        // Reload
        let reloaded = load_mesh(file.path()).expect("should reload PLY with colors");

        assert_eq!(reloaded.vertex_count(), 3);
        assert_eq!(reloaded.face_count(), 1);

        // Verify colors are preserved
        let c0 = reloaded.vertices[0].color.expect("v0 should have color");
        assert_eq!(c0.r, 255);
        assert_eq!(c0.g, 0);
        assert_eq!(c0.b, 0);

        let c1 = reloaded.vertices[1].color.expect("v1 should have color");
        assert_eq!(c1.r, 0);
        assert_eq!(c1.g, 255);
        assert_eq!(c1.b, 0);

        let c2 = reloaded.vertices[2].color.expect("v2 should have color");
        assert_eq!(c2.r, 0);
        assert_eq!(c2.g, 0);
        assert_eq!(c2.b, 255);
    }

    #[test]
    fn test_ply_with_colors_and_normals() {
        use crate::VertexColor;
        use nalgebra::Vector3;

        // Create a mesh with both vertex colors and normals
        let mut mesh = Mesh::new();

        let mut v0 = Vertex::from_coords(0.0, 0.0, 0.0);
        v0.color = Some(VertexColor::new(128, 64, 32));
        v0.normal = Some(Vector3::new(0.0, 0.0, 1.0));
        mesh.vertices.push(v0);

        let mut v1 = Vertex::from_coords(10.0, 0.0, 0.0);
        v1.color = Some(VertexColor::new(200, 100, 50));
        v1.normal = Some(Vector3::new(1.0, 0.0, 0.0));
        mesh.vertices.push(v1);

        let mut v2 = Vertex::from_coords(0.0, 10.0, 0.0);
        v2.color = Some(VertexColor::new(50, 150, 250));
        v2.normal = Some(Vector3::new(0.0, 1.0, 0.0));
        mesh.vertices.push(v2);

        mesh.faces.push([0, 1, 2]);

        // Save to PLY
        let file = NamedTempFile::with_suffix(".ply").unwrap();
        save_ply(&mesh, file.path()).expect("should save PLY with colors and normals");

        // Reload
        let reloaded = load_mesh(file.path()).expect("should reload PLY with colors and normals");

        // Verify both colors and normals are preserved
        for (i, (orig, loaded)) in mesh
            .vertices
            .iter()
            .zip(reloaded.vertices.iter())
            .enumerate()
        {
            // Check color
            let orig_c = orig.color.expect("original should have color");
            let loaded_c = loaded.color.expect("loaded should have color");
            assert_eq!(orig_c, loaded_c, "PLY vertex {} color mismatch", i);

            // Check normal
            let orig_n = orig.normal.expect("original should have normal");
            let loaded_n = loaded.normal.expect("loaded should have normal");
            let diff = (orig_n - loaded_n).norm();
            assert!(diff < 1e-5, "PLY vertex {} normal mismatch", i);
        }
    }

    #[test]
    fn test_3mf_with_materials_roundtrip() {
        use crate::region::{MaterialZone, MeshRegion};

        // Create a simple cube mesh (8 vertices, 12 faces)
        let mut mesh = Mesh::new();
        // Bottom face vertices
        mesh.vertices.push(Vertex::from_coords(0.0, 0.0, 0.0));
        mesh.vertices.push(Vertex::from_coords(10.0, 0.0, 0.0));
        mesh.vertices.push(Vertex::from_coords(10.0, 10.0, 0.0));
        mesh.vertices.push(Vertex::from_coords(0.0, 10.0, 0.0));
        // Top face vertices
        mesh.vertices.push(Vertex::from_coords(0.0, 0.0, 10.0));
        mesh.vertices.push(Vertex::from_coords(10.0, 0.0, 10.0));
        mesh.vertices.push(Vertex::from_coords(10.0, 10.0, 10.0));
        mesh.vertices.push(Vertex::from_coords(0.0, 10.0, 10.0));

        // 12 triangles for the cube
        // Bottom face (z=0)
        mesh.faces.push([0, 2, 1]);
        mesh.faces.push([0, 3, 2]);
        // Top face (z=10)
        mesh.faces.push([4, 5, 6]);
        mesh.faces.push([4, 6, 7]);
        // Front face (y=0)
        mesh.faces.push([0, 1, 5]);
        mesh.faces.push([0, 5, 4]);
        // Back face (y=10)
        mesh.faces.push([2, 3, 7]);
        mesh.faces.push([2, 7, 6]);
        // Left face (x=0)
        mesh.faces.push([0, 4, 7]);
        mesh.faces.push([0, 7, 3]);
        // Right face (x=10)
        mesh.faces.push([1, 2, 6]);
        mesh.faces.push([1, 6, 5]);

        // Create material zones
        // Bottom faces (indices 0, 1) - Blue "Base" material
        let bottom_region = MeshRegion::from_faces("bottom", vec![0, 1]);
        let bottom_zone = MaterialZone::new(bottom_region, "Base-Material")
            .with_color(0, 0, 255)
            .with_shore_hardness(80.0);

        // Top faces (indices 2, 3) - Red "Top" material
        let top_region = MeshRegion::from_faces("top", vec![2, 3]);
        let top_zone = MaterialZone::new(top_region, "Top-Material")
            .with_color(255, 0, 0)
            .with_flexibility(0.5);

        // Side faces (indices 4-11) - Green "Side" material
        let side_region = MeshRegion::from_faces("sides", vec![4, 5, 6, 7, 8, 9, 10, 11]);
        let side_zone = MaterialZone::new(side_region, "Side-Material")
            .with_color(0, 255, 0)
            .with_density(1.2);

        let params = ThreeMfExportParams::with_materials(vec![bottom_zone, top_zone, side_zone]);

        // Save to temp file
        let file = NamedTempFile::with_suffix(".3mf").unwrap();
        save_3mf_with_materials(&mesh, file.path(), &params)
            .expect("should save 3MF with materials");

        // Reload
        let result =
            load_3mf_with_materials(file.path()).expect("should reload 3MF with materials");

        // Verify mesh geometry
        assert_eq!(result.mesh.vertex_count(), 8, "vertex count should match");
        assert_eq!(result.mesh.face_count(), 12, "face count should match");

        // Verify materials
        assert_eq!(
            result.material_zones.len(),
            3,
            "should have 3 material zones"
        );

        // Check material names and colors
        let names: Vec<&str> = result
            .material_zones
            .iter()
            .map(|z| z.material_name.as_str())
            .collect();
        assert!(names.contains(&"Base-Material"));
        assert!(names.contains(&"Top-Material"));
        assert!(names.contains(&"Side-Material"));

        // Verify triangle material assignments
        assert_eq!(
            result.triangle_materials.len(),
            12,
            "should have 12 triangle materials"
        );

        // Bottom faces should be material 0 (Base-Material)
        assert_eq!(result.triangle_materials[0], Some(0));
        assert_eq!(result.triangle_materials[1], Some(0));

        // Top faces should be material 1 (Top-Material)
        assert_eq!(result.triangle_materials[2], Some(1));
        assert_eq!(result.triangle_materials[3], Some(1));

        // Side faces should be material 2 (Side-Material)
        for i in 4..12 {
            assert_eq!(
                result.triangle_materials[i],
                Some(2),
                "face {} should have Side-Material",
                i
            );
        }
    }

    #[test]
    fn test_3mf_with_materials_color_roundtrip() {
        use crate::region::{MaterialZone, MeshRegion};

        // Create a simple triangle mesh
        let mut mesh = Mesh::new();
        mesh.vertices.push(Vertex::from_coords(0.0, 0.0, 0.0));
        mesh.vertices.push(Vertex::from_coords(10.0, 0.0, 0.0));
        mesh.vertices.push(Vertex::from_coords(5.0, 10.0, 0.0));
        mesh.faces.push([0, 1, 2]);

        // Create material with specific color
        let region = MeshRegion::from_faces("colored", vec![0]);
        let zone = MaterialZone::new(region, "Orange-TPU").with_color(255, 128, 0);

        let params = ThreeMfExportParams::with_materials(vec![zone]);

        // Save and reload
        let file = NamedTempFile::with_suffix(".3mf").unwrap();
        save_3mf_with_materials(&mesh, file.path(), &params).expect("should save");

        let result = load_3mf_with_materials(file.path()).expect("should reload");

        // Verify color is preserved
        assert_eq!(result.material_zones.len(), 1);
        let loaded_zone = &result.material_zones[0];
        assert_eq!(loaded_zone.material_name, "Orange-TPU");
        assert_eq!(loaded_zone.properties.color, Some((255, 128, 0)));
    }

    #[test]
    fn test_3mf_without_materials_backward_compatible() {
        // Create a simple mesh
        let mut mesh = Mesh::new();
        mesh.vertices.push(Vertex::from_coords(0.0, 0.0, 0.0));
        mesh.vertices.push(Vertex::from_coords(10.0, 0.0, 0.0));
        mesh.vertices.push(Vertex::from_coords(5.0, 10.0, 0.0));
        mesh.faces.push([0, 1, 2]);

        // Save with empty params (no materials)
        let params = ThreeMfExportParams::default();
        let file = NamedTempFile::with_suffix(".3mf").unwrap();
        save_3mf_with_materials(&mesh, file.path(), &params).expect("should save");

        // Should be loadable with both functions
        let basic_mesh = load_mesh(file.path()).expect("should load with basic function");
        assert_eq!(basic_mesh.vertex_count(), 3);
        assert_eq!(basic_mesh.face_count(), 1);

        let result =
            load_3mf_with_materials(file.path()).expect("should load with materials function");
        assert_eq!(result.mesh.vertex_count(), 3);
        assert_eq!(result.mesh.face_count(), 1);
        assert!(result.material_zones.is_empty(), "should have no materials");
    }

    #[test]
    fn test_parse_hex_color() {
        assert_eq!(parse_hex_color("#FF8000"), Some((255, 128, 0)));
        assert_eq!(parse_hex_color("FF8000"), Some((255, 128, 0)));
        assert_eq!(parse_hex_color("#000000"), Some((0, 0, 0)));
        assert_eq!(parse_hex_color("#FFFFFF"), Some((255, 255, 255)));
        assert_eq!(parse_hex_color("#abc"), None); // Too short
        assert_eq!(parse_hex_color(""), None);
    }

    #[test]
    fn test_escape_xml() {
        assert_eq!(escape_xml("hello"), "hello");
        assert_eq!(escape_xml("a & b"), "a &amp; b");
        assert_eq!(escape_xml("<tag>"), "&lt;tag&gt;");
        assert_eq!(escape_xml("\"quoted\""), "&quot;quoted&quot;");
        assert_eq!(escape_xml("it's"), "it&apos;s");
    }

    #[test]
    fn test_3mf_with_vertex_based_region() {
        use crate::region::{MaterialZone, MeshRegion};

        // Create a mesh with 2 triangles sharing vertices
        let mut mesh = Mesh::new();
        mesh.vertices.push(Vertex::from_coords(0.0, 0.0, 0.0)); // 0
        mesh.vertices.push(Vertex::from_coords(10.0, 0.0, 0.0)); // 1
        mesh.vertices.push(Vertex::from_coords(5.0, 10.0, 0.0)); // 2
        mesh.vertices.push(Vertex::from_coords(10.0, 10.0, 0.0)); // 3

        mesh.faces.push([0, 1, 2]); // face 0: vertices 0, 1, 2
        mesh.faces.push([1, 3, 2]); // face 1: vertices 1, 3, 2

        // Create a region using vertices (not faces)
        // Only face 0 has ALL its vertices in the region
        let region = MeshRegion::from_vertices("left-triangle", vec![0, 1, 2]);
        let zone = MaterialZone::new(region, "Left-Material").with_color(255, 0, 0);

        let params = ThreeMfExportParams::with_materials(vec![zone]);

        // Save and reload
        let file = NamedTempFile::with_suffix(".3mf").unwrap();
        save_3mf_with_materials(&mesh, file.path(), &params).expect("should save");

        let result = load_3mf_with_materials(file.path()).expect("should reload");

        // Face 0 should have material 0, face 1 should have default (0) since only face-based regions are loaded
        assert_eq!(result.triangle_materials[0], Some(0));
        // Face 1 will have material 0 as default since it's not in the region but defaults to first material
        assert_eq!(result.triangle_materials[1], Some(0));
    }

    #[test]
    fn test_3mf_extended_with_beam_lattice() {
        use nalgebra::Point3;

        // Create a simple mesh
        let mut mesh = Mesh::new();
        mesh.vertices.push(Vertex::from_coords(0.0, 0.0, 0.0));
        mesh.vertices.push(Vertex::from_coords(10.0, 0.0, 0.0));
        mesh.vertices.push(Vertex::from_coords(5.0, 10.0, 0.0));
        mesh.faces.push([0, 1, 2]);

        // Create beam lattice data
        let mut beam_lattice = BeamLatticeData::new(0.5);
        let v1 = beam_lattice.add_vertex(Point3::new(0.0, 0.0, 0.0));
        let v2 = beam_lattice.add_vertex(Point3::new(10.0, 0.0, 0.0));
        let v3 = beam_lattice.add_vertex(Point3::new(5.0, 10.0, 0.0));
        let v4 = beam_lattice.add_vertex(Point3::new(5.0, 5.0, 10.0));

        beam_lattice.add_beam_default(v1, v2);
        beam_lattice.add_beam_default(v2, v3);
        beam_lattice.add_beam_default(v3, v1);
        beam_lattice.add_beam_default(v1, v4);
        beam_lattice.add_beam_default(v2, v4);
        beam_lattice.add_beam_default(v3, v4);

        let params = ThreeMfExportParams::default().with_beam_lattice(beam_lattice);

        // Save the 3MF
        let file = NamedTempFile::with_suffix(".3mf").unwrap();
        save_3mf_extended(&mesh, file.path(), &params).expect("should save 3mf with beam lattice");

        // Verify the file was created and can be read as a zip
        let zip_file = std::fs::File::open(file.path()).expect("should open file");
        let mut archive = zip::ZipArchive::new(zip_file).expect("should be valid zip");

        // Read the model XML
        let mut model_content = String::new();
        archive
            .by_name("3D/3dmodel.model")
            .expect("should have model file")
            .read_to_string(&mut model_content)
            .expect("should read model");

        // Verify beam lattice namespace is present
        assert!(
            model_content.contains("xmlns:b="),
            "should have beam lattice namespace"
        );
        assert!(
            model_content.contains("beamlattice"),
            "should have beamlattice element"
        );
        assert!(
            model_content.contains("<b:beam"),
            "should have beam elements"
        );
        assert!(
            model_content.contains("<b:vertex"),
            "should have vertex elements"
        );

        // Verify beam count - we added 6 beams
        let beam_count = model_content.matches("<b:beam").count();
        assert!(
            beam_count >= 6,
            "should have at least 6 beams, found {}",
            beam_count
        );
    }

    #[test]
    fn test_3mf_extended_with_color_groups() {
        // Create a simple mesh with 2 triangles
        let mut mesh = Mesh::new();
        mesh.vertices.push(Vertex::from_coords(0.0, 0.0, 0.0));
        mesh.vertices.push(Vertex::from_coords(10.0, 0.0, 0.0));
        mesh.vertices.push(Vertex::from_coords(5.0, 10.0, 0.0));
        mesh.vertices.push(Vertex::from_coords(15.0, 10.0, 0.0));
        mesh.faces.push([0, 1, 2]);
        mesh.faces.push([1, 3, 2]);

        // Create color group
        let mut color_group = ColorGroup::new(2);
        let red = color_group.add_rgb(255, 0, 0);
        let green = color_group.add_rgb(0, 255, 0);
        let blue = color_group.add_rgb(0, 0, 255);

        let mut params = ThreeMfExportParams::default().add_color_group(color_group);

        // Assign colors to first triangle
        params.triangle_colors.insert(
            0,
            TriangleColors {
                color_group_id: 2,
                p1: red,
                p2: green,
                p3: blue,
            },
        );

        // Save the 3MF
        let file = NamedTempFile::with_suffix(".3mf").unwrap();
        save_3mf_extended(&mesh, file.path(), &params).expect("should save 3mf with color groups");

        // Verify the file content
        let zip_file = std::fs::File::open(file.path()).expect("should open file");
        let mut archive = zip::ZipArchive::new(zip_file).expect("should be valid zip");

        let mut model_content = String::new();
        archive
            .by_name("3D/3dmodel.model")
            .expect("should have model file")
            .read_to_string(&mut model_content)
            .expect("should read model");

        // Verify color group elements
        assert!(
            model_content.contains("xmlns:m="),
            "should have materials namespace"
        );
        assert!(
            model_content.contains("<m:colorgroup"),
            "should have colorgroup element"
        );
        assert!(
            model_content.contains("<m:color"),
            "should have color elements"
        );
        assert!(model_content.contains("#FF0000FF"), "should have red color");
        assert!(
            model_content.contains("#00FF00FF"),
            "should have green color"
        );
        assert!(
            model_content.contains("#0000FFFF"),
            "should have blue color"
        );

        // Verify per-vertex color assignment on first triangle
        assert!(
            model_content.contains("p1=\"0\" p2=\"1\" p3=\"2\""),
            "should have per-vertex color indices"
        );
    }

    #[test]
    #[cfg(feature = "3mf-production")]
    fn test_3mf_extended_with_production_uuids() {
        // Create a simple mesh
        let mut mesh = Mesh::new();
        mesh.vertices.push(Vertex::from_coords(0.0, 0.0, 0.0));
        mesh.vertices.push(Vertex::from_coords(10.0, 0.0, 0.0));
        mesh.vertices.push(Vertex::from_coords(5.0, 10.0, 0.0));
        mesh.faces.push([0, 1, 2]);

        let params = ThreeMfExportParams::default().with_uuids(true);

        // Save the 3MF
        let file = NamedTempFile::with_suffix(".3mf").unwrap();
        save_3mf_extended(&mesh, file.path(), &params).expect("should save 3mf with UUIDs");

        // Verify the file content
        let zip_file = std::fs::File::open(file.path()).expect("should open file");
        let mut archive = zip::ZipArchive::new(zip_file).expect("should be valid zip");

        let mut model_content = String::new();
        archive
            .by_name("3D/3dmodel.model")
            .expect("should have model file")
            .read_to_string(&mut model_content)
            .expect("should read model");

        // Verify production namespace and UUID attributes
        assert!(
            model_content.contains("xmlns:p="),
            "should have production namespace"
        );
        assert!(
            model_content.contains("p:UUID="),
            "should have UUID attributes"
        );

        // Count UUID occurrences - should have at least 3 (build, object, item)
        let uuid_count = model_content.matches("p:UUID=").count();
        assert!(
            uuid_count >= 3,
            "should have at least 3 UUIDs, found {}",
            uuid_count
        );
    }

    #[test]
    fn test_beam_lattice_from_cubic_generation() {
        use crate::lattice::{LatticeParams, generate_lattice};
        use nalgebra::Point3;

        // Generate a small cubic lattice with beam data preservation
        let params = LatticeParams::cubic(5.0).with_beam_export(true);

        let bounds = (Point3::new(0.0, 0.0, 0.0), Point3::new(10.0, 10.0, 10.0));

        let result = generate_lattice(&params, bounds);

        // Should have beam data
        assert!(result.beam_data.is_some(), "should preserve beam data");

        let beam_data = result.beam_data.unwrap();
        assert!(!beam_data.vertices.is_empty(), "should have vertices");
        assert!(!beam_data.beams.is_empty(), "should have beams");

        // For a 2x2x2 cell cubic lattice, we should have:
        // - 3x3x3 = 27 vertices (corners)
        // - 3 struts per direction per layer = significant number of beams
        assert!(
            beam_data.vertices.len() >= 8,
            "should have at least corner vertices"
        );
        assert!(
            beam_data.beams.len() >= 12,
            "should have at least edge beams"
        );

        // Export to 3MF with beam lattice
        let mut mesh = Mesh::new();
        mesh.vertices.push(Vertex::from_coords(0.0, 0.0, 0.0));
        mesh.vertices.push(Vertex::from_coords(10.0, 0.0, 0.0));
        mesh.vertices.push(Vertex::from_coords(5.0, 10.0, 0.0));
        mesh.faces.push([0, 1, 2]);

        let export_params = ThreeMfExportParams::default().with_beam_lattice(beam_data);

        let file = NamedTempFile::with_suffix(".3mf").unwrap();
        save_3mf_extended(&mesh, file.path(), &export_params)
            .expect("should export lattice with beam data");

        // Verify file is valid
        let zip_file = std::fs::File::open(file.path()).expect("should open file");
        let archive = zip::ZipArchive::new(zip_file).expect("should be valid zip");
        assert!(archive.len() >= 3, "should have required 3MF files");
    }
}

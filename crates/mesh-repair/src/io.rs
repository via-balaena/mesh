//! Mesh file I/O for STL, OBJ, and 3MF formats.

use std::fs::File;
use std::io::{BufReader, BufWriter, Read, Write};
use std::path::Path;

use tracing::{debug, info, warn};

use crate::error::{MeshError, MeshResult};
use crate::validate::{validate_mesh_data, ValidationOptions};
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
        debug!(
            "Dimensions: {:.1} x {:.1} x {:.1}",
            dims.x, dims.y, dims.z
        );

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
    let ply = parser.read_ply(&mut reader).map_err(|e| MeshError::ParseError {
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
            ) {
                if let (Ok(nx), Ok(ny), Ok(nz)) = (
                    get_ply_float(Some(nx), "nx", path),
                    get_ply_float(Some(ny), "ny", path),
                    get_ply_float(Some(nz), "nz", path),
                ) {
                    vertex.normal = Some(nalgebra::Vector3::new(nx, ny, nz));
                }
            }

            // Try to load vertex colors if present (red, green, blue)
            if let (Some(r), Some(g), Some(b)) = (
                vertex_element.get("red"),
                vertex_element.get("green"),
                vertex_element.get("blue"),
            ) {
                if let (Ok(r), Ok(g), Ok(b)) = (
                    get_ply_u8(Some(r)),
                    get_ply_u8(Some(g)),
                    get_ply_u8(Some(b)),
                ) {
                    vertex.color = Some(crate::VertexColor::new(r, g, b));
                }
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
                        mesh.faces.push([
                            indices[0] as u32,
                            indices[i] as u32,
                            indices[i + 1] as u32,
                        ]);
                    }
                }
            } else if let Some(Property::ListUChar(indices)) = indices {
                if indices.len() >= 3 {
                    for i in 1..indices.len() - 1 {
                        mesh.faces.push([
                            indices[0] as u32,
                            indices[i] as u32,
                            indices[i + 1] as u32,
                        ]);
                    }
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
fn get_ply_float(
    prop: Option<&ply_rs::ply::Property>,
    name: &str,
    path: &Path,
) -> MeshResult<f64> {
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
        source: std::io::Error::new(std::io::ErrorKind::Other, e.to_string()),
    })?;

    writer.flush().map_err(|e| MeshError::IoWrite {
        path: path.to_path_buf(),
        source: e,
    })?;

    info!(
        "Saved {} triangles to {:?}",
        mesh.face_count(),
        path
    );

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
        writeln!(writer, "v {:.6} {:.6} {:.6}", v.position.x, v.position.y, v.position.z)
            .map_err(|e| MeshError::IoWrite {
                path: path.to_path_buf(),
                source: e,
            })?;

        // Add debug comment with vertex attributes (tag, offset)
        if v.tag.is_some() || v.offset.is_some() {
            let tag_str = v.tag.map_or("none".to_string(), |z| format!("{}", z));
            let offset_str = v.offset.map_or("none".to_string(), |c| format!("{:.3}", c));
            writeln!(writer, "# v{} tag={} offset={}", i, tag_str, offset_str)
                .map_err(|e| MeshError::IoWrite {
                    path: path.to_path_buf(),
                    source: e,
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
                writeln!(writer, "vn {:.6} {:.6} {:.6}", n.x, n.y, n.z)
                    .map_err(|e| MeshError::IoWrite {
                        path: path.to_path_buf(),
                        source: e,
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
            writeln!(writer, "f {}//{} {}//{} {}//{}", i0, i0, i1, i1, i2, i2)
                .map_err(|e| MeshError::IoWrite {
                    path: path.to_path_buf(),
                    source: e,
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

    let mut model_file = archive.by_name(&model_path).map_err(|e| MeshError::ParseError {
        path: path.to_path_buf(),
        details: format!("Cannot open model file '{}': {}", model_path, e),
    })?;

    let mut xml_content = String::new();
    model_file.read_to_string(&mut xml_content).map_err(|e| MeshError::IoRead {
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
    use quick_xml::events::Event;
    use quick_xml::Reader;

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
    zip.start_file("[Content_Types].xml", options).map_err(|e| MeshError::IoWrite {
        path: path.to_path_buf(),
        source: std::io::Error::new(std::io::ErrorKind::Other, e.to_string()),
    })?;
    zip.write_all(CONTENT_TYPES_XML.as_bytes()).map_err(|e| MeshError::IoWrite {
        path: path.to_path_buf(),
        source: e,
    })?;

    // Write relationships file
    zip.start_file("_rels/.rels", options).map_err(|e| MeshError::IoWrite {
        path: path.to_path_buf(),
        source: std::io::Error::new(std::io::ErrorKind::Other, e.to_string()),
    })?;
    zip.write_all(RELS_XML.as_bytes()).map_err(|e| MeshError::IoWrite {
        path: path.to_path_buf(),
        source: e,
    })?;

    // Write the model file
    zip.start_file("3D/3dmodel.model", options).map_err(|e| MeshError::IoWrite {
        path: path.to_path_buf(),
        source: std::io::Error::new(std::io::ErrorKind::Other, e.to_string()),
    })?;

    let model_xml = generate_3mf_model_xml(mesh);
    zip.write_all(model_xml.as_bytes()).map_err(|e| MeshError::IoWrite {
        path: path.to_path_buf(),
        source: e,
    })?;

    zip.finish().map_err(|e| MeshError::IoWrite {
        path: path.to_path_buf(),
        source: std::io::Error::new(std::io::ErrorKind::Other, e.to_string()),
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
    xml.push_str(r#"<?xml version="1.0" encoding="UTF-8"?>
<model unit="millimeter" xmlns="http://schemas.microsoft.com/3dmanufacturing/core/2015/02">
  <resources>
    <object id="1" type="model">
      <mesh>
        <vertices>
"#);

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

    xml.push_str(r#"        </triangles>
      </mesh>
    </object>
  </resources>
  <build>
    <item objectid="1"/>
  </build>
</model>
"#);

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
    vertex_def.properties.add(PropertyDef::new("x".to_string(), PropertyType::Scalar(ScalarType::Float)));
    vertex_def.properties.add(PropertyDef::new("y".to_string(), PropertyType::Scalar(ScalarType::Float)));
    vertex_def.properties.add(PropertyDef::new("z".to_string(), PropertyType::Scalar(ScalarType::Float)));
    if has_normals {
        vertex_def.properties.add(PropertyDef::new("nx".to_string(), PropertyType::Scalar(ScalarType::Float)));
        vertex_def.properties.add(PropertyDef::new("ny".to_string(), PropertyType::Scalar(ScalarType::Float)));
        vertex_def.properties.add(PropertyDef::new("nz".to_string(), PropertyType::Scalar(ScalarType::Float)));
    }
    if has_colors {
        vertex_def.properties.add(PropertyDef::new("red".to_string(), PropertyType::Scalar(ScalarType::UChar)));
        vertex_def.properties.add(PropertyDef::new("green".to_string(), PropertyType::Scalar(ScalarType::UChar)));
        vertex_def.properties.add(PropertyDef::new("blue".to_string(), PropertyType::Scalar(ScalarType::UChar)));
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
        source: std::io::Error::new(std::io::ErrorKind::Other, format!("PLY consistency error: {:?}", e)),
    })?;

    // Write to file
    let file = File::create(path).map_err(|e| MeshError::IoWrite {
        path: path.to_path_buf(),
        source: e,
    })?;
    let mut writer = BufWriter::new(file);

    let ply_writer = Writer::new();
    ply_writer.write_ply(&mut writer, &mut ply).map_err(|e| MeshError::IoWrite {
        path: path.to_path_buf(),
        source: std::io::Error::new(std::io::ErrorKind::Other, format!("PLY write error: {:?}", e)),
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
    vertex_def.properties.add(PropertyDef::new("x".to_string(), PropertyType::Scalar(ScalarType::Float)));
    vertex_def.properties.add(PropertyDef::new("y".to_string(), PropertyType::Scalar(ScalarType::Float)));
    vertex_def.properties.add(PropertyDef::new("z".to_string(), PropertyType::Scalar(ScalarType::Float)));
    if has_normals {
        vertex_def.properties.add(PropertyDef::new("nx".to_string(), PropertyType::Scalar(ScalarType::Float)));
        vertex_def.properties.add(PropertyDef::new("ny".to_string(), PropertyType::Scalar(ScalarType::Float)));
        vertex_def.properties.add(PropertyDef::new("nz".to_string(), PropertyType::Scalar(ScalarType::Float)));
    }
    if has_colors {
        vertex_def.properties.add(PropertyDef::new("red".to_string(), PropertyType::Scalar(ScalarType::UChar)));
        vertex_def.properties.add(PropertyDef::new("green".to_string(), PropertyType::Scalar(ScalarType::UChar)));
        vertex_def.properties.add(PropertyDef::new("blue".to_string(), PropertyType::Scalar(ScalarType::UChar)));
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
        source: std::io::Error::new(std::io::ErrorKind::Other, format!("PLY consistency error: {:?}", e)),
    })?;

    // Write to file
    let file = File::create(path).map_err(|e| MeshError::IoWrite {
        path: path.to_path_buf(),
        source: e,
    })?;
    let mut writer = BufWriter::new(file);

    let ply_writer = Writer::new();
    ply_writer.write_ply(&mut writer, &mut ply).map_err(|e| MeshError::IoWrite {
        path: path.to_path_buf(),
        source: std::io::Error::new(std::io::ErrorKind::Other, format!("PLY write error: {:?}", e)),
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
        for (i, (orig, loaded)) in mesh.vertices.iter().zip(reloaded.vertices.iter()).enumerate() {
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
        for (i, (orig, loaded)) in mesh.vertices.iter().zip(reloaded.vertices.iter()).enumerate() {
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
        for (i, (orig, loaded)) in mesh.vertices.iter().zip(reloaded.vertices.iter()).enumerate() {
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
        for (i, (orig, loaded)) in mesh.vertices.iter().zip(reloaded.vertices.iter()).enumerate() {
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
        for (i, (orig, loaded)) in mesh.vertices.iter().zip(reloaded.vertices.iter()).enumerate() {
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
}

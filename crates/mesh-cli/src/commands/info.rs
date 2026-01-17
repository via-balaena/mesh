//! mesh info command - display mesh statistics.

use std::path::Path;

use anyhow::{Context, Result};
use colored::Colorize;
use mesh_repair::Mesh;
use serde::Serialize;

use crate::{Cli, OutputFormat, output};

#[derive(Serialize)]
struct MeshInfo {
    path: String,
    vertices: usize,
    faces: usize,
    #[serde(skip_serializing_if = "Option::is_none")]
    bounds: Option<BoundsInfo>,
    #[serde(skip_serializing_if = "Option::is_none")]
    volume: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    surface_area: Option<f64>,
    has_normals: bool,
    has_colors: bool,
    components: usize,
}

#[derive(Serialize)]
struct BoundsInfo {
    min: [f64; 3],
    max: [f64; 3],
    dimensions: [f64; 3],
}

pub fn run(input: &Path, detailed: bool, cli: &Cli) -> Result<()> {
    let mesh =
        Mesh::load(input).with_context(|| format!("Failed to load mesh from {:?}", input))?;

    let components = mesh.find_components();
    let bounds = mesh.bounds().map(|(min, max)| {
        let dims = max - min;
        BoundsInfo {
            min: [min.x, min.y, min.z],
            max: [max.x, max.y, max.z],
            dimensions: [dims.x, dims.y, dims.z],
        }
    });

    let volume = if detailed { Some(mesh.volume()) } else { None };
    let surface_area = if detailed {
        Some(mesh.surface_area())
    } else {
        None
    };

    let info = MeshInfo {
        path: input.display().to_string(),
        vertices: mesh.vertex_count(),
        faces: mesh.face_count(),
        bounds,
        volume,
        surface_area,
        has_normals: mesh.vertices.iter().any(|v| v.normal.is_some()),
        has_colors: mesh.vertices.iter().any(|v| v.color.is_some()),
        components: components.component_count,
    };

    match cli.format {
        OutputFormat::Json => {
            output::print(&info, cli.format, cli.quiet);
        }
        OutputFormat::Text => {
            if !cli.quiet {
                println!("{}", "Mesh Information".bold().underline());
                println!("  {}: {}", "File".cyan(), input.display());
                println!("  {}: {}", "Vertices".cyan(), info.vertices);
                println!("  {}: {}", "Faces".cyan(), info.faces);
                println!("  {}: {}", "Components".cyan(), info.components);

                if let Some(ref b) = info.bounds {
                    println!(
                        "  {}: {:.2} x {:.2} x {:.2} mm",
                        "Dimensions".cyan(),
                        b.dimensions[0],
                        b.dimensions[1],
                        b.dimensions[2]
                    );
                    println!(
                        "  {}: ({:.2}, {:.2}, {:.2})",
                        "Min bounds".cyan(),
                        b.min[0],
                        b.min[1],
                        b.min[2]
                    );
                    println!(
                        "  {}: ({:.2}, {:.2}, {:.2})",
                        "Max bounds".cyan(),
                        b.max[0],
                        b.max[1],
                        b.max[2]
                    );
                }

                if let Some(vol) = info.volume {
                    println!("  {}: {:.2} mm³", "Volume".cyan(), vol);
                }
                if let Some(area) = info.surface_area {
                    println!("  {}: {:.2} mm²", "Surface area".cyan(), area);
                }

                println!(
                    "  {}: {}",
                    "Has normals".cyan(),
                    if info.has_normals { "yes" } else { "no" }
                );
                println!(
                    "  {}: {}",
                    "Has colors".cyan(),
                    if info.has_colors { "yes" } else { "no" }
                );
            }
        }
    }

    Ok(())
}

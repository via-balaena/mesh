//! mesh convert command - convert between mesh formats.

use std::path::Path;

use anyhow::{Context, Result};
use colored::Colorize;
use mesh_repair::Mesh;
use serde::Serialize;

use crate::{Cli, OutputFormat, output};

#[derive(Serialize)]
struct ConvertResult {
    input: String,
    output: String,
    input_format: String,
    output_format: String,
    vertices: usize,
    faces: usize,
}

pub fn run(input: &Path, output_path: &Path, _ascii: bool, cli: &Cli) -> Result<()> {
    let mesh =
        Mesh::load(input).with_context(|| format!("Failed to load mesh from {:?}", input))?;

    let input_ext = input
        .extension()
        .and_then(|e| e.to_str())
        .unwrap_or("unknown")
        .to_lowercase();

    let output_ext = output_path
        .extension()
        .and_then(|e| e.to_str())
        .unwrap_or("unknown")
        .to_lowercase();

    // Save in target format
    // Note: ascii flag would be used for STL/PLY output if we had dedicated functions
    mesh.save(output_path)
        .with_context(|| format!("Failed to save mesh to {:?}", output_path))?;

    let result = ConvertResult {
        input: input.display().to_string(),
        output: output_path.display().to_string(),
        input_format: input_ext,
        output_format: output_ext,
        vertices: mesh.vertex_count(),
        faces: mesh.face_count(),
    };

    match cli.format {
        OutputFormat::Json => {
            output::print(&result, cli.format, cli.quiet);
        }
        OutputFormat::Text => {
            if !cli.quiet {
                output::success(
                    &format!("Converted {} to {}", input.display(), output_path.display()),
                    cli.format,
                    cli.quiet,
                );
                println!(
                    "  {}: {} → {}",
                    "Format".cyan(),
                    result.input_format.to_uppercase(),
                    result.output_format.to_uppercase()
                );
                println!(
                    "  {}: {} vertices, {} faces",
                    "Size".cyan(),
                    result.vertices,
                    result.faces
                );
            }
        }
    }

    Ok(())
}

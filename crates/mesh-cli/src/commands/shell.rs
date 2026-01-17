//! mesh shell command - generate shell/offset surface.

use std::path::Path;

use anyhow::{Context, Result};
use colored::Colorize;
use mesh_repair::Mesh;
use mesh_shell::{SdfOffsetParams, ShellParams, apply_sdf_offset, generate_shell};
use serde::Serialize;

use crate::{Cli, OutputFormat, ShellDirection, output};

#[derive(Serialize)]
struct ShellResult {
    input: String,
    output: String,
    success: bool,
    thickness: f64,
    direction: String,
    input_faces: usize,
    output_faces: usize,
}

pub fn run(
    input: &Path,
    output_path: &Path,
    thickness: f64,
    _resolution: usize,
    direction: ShellDirection,
    cli: &Cli,
) -> Result<()> {
    let mut mesh =
        Mesh::load(input).with_context(|| format!("Failed to load mesh from {:?}", input))?;

    let input_faces = mesh.face_count();

    // Set offset values on all vertices based on direction
    let offset_value: f32 = match direction {
        ShellDirection::Outward => thickness as f32,
        ShellDirection::Inward => -(thickness as f32),
        ShellDirection::Both => thickness as f32, // For both, we'll need both surfaces
    };

    for v in &mut mesh.vertices {
        v.offset = Some(offset_value);
    }

    if !cli.quiet {
        output::info(
            &format!("Generating {:.2}mm shell...", thickness),
            cli.format,
            cli.quiet,
        );
    }

    // Apply SDF offset to create the offset surface
    let sdf_params = SdfOffsetParams::default();
    let offset_result = apply_sdf_offset(&mesh, &sdf_params)
        .map_err(|e| anyhow::anyhow!("SDF offset failed: {:?}", e))?;

    // Generate shell with walls
    let shell_params = ShellParams::default();
    let (shell_mesh, _stats) = generate_shell(&offset_result.mesh, &shell_params);

    // Save output
    shell_mesh
        .save(output_path)
        .with_context(|| format!("Failed to save shell mesh to {:?}", output_path))?;

    let direction_str = match direction {
        ShellDirection::Outward => "outward",
        ShellDirection::Inward => "inward",
        ShellDirection::Both => "both",
    };

    let result = ShellResult {
        input: input.display().to_string(),
        output: output_path.display().to_string(),
        success: true,
        thickness,
        direction: direction_str.to_string(),
        input_faces,
        output_faces: shell_mesh.face_count(),
    };

    match cli.format {
        OutputFormat::Json => {
            output::print(&result, cli.format, cli.quiet);
        }
        OutputFormat::Text => {
            if !cli.quiet {
                output::success(
                    &format!("Shell mesh saved to {}", output_path.display()),
                    cli.format,
                    cli.quiet,
                );
                println!(
                    "  {}: {:.2}mm {}",
                    "Thickness".cyan(),
                    thickness,
                    direction_str
                );
                println!(
                    "  {}: {} → {} faces",
                    "Faces".cyan(),
                    result.input_faces,
                    result.output_faces
                );
            }
        }
    }

    Ok(())
}

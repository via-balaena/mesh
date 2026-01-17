//! mesh decimate command - simplify mesh by reducing triangles.

use std::path::Path;

use anyhow::{Context, Result};
use colored::Colorize;
use mesh_repair::{DecimateParams, Mesh};
use serde::Serialize;

use crate::{Cli, OutputFormat, output};

#[derive(Serialize)]
struct DecimateResult {
    input: String,
    output: String,
    success: bool,
    original_triangles: usize,
    final_triangles: usize,
    reduction_ratio: f64,
    collapses_performed: usize,
}

pub fn run(
    input: &Path,
    output_path: &Path,
    ratio: Option<f64>,
    count: Option<usize>,
    preserve_boundary: bool,
    cli: &Cli,
) -> Result<()> {
    let mesh =
        Mesh::load(input).with_context(|| format!("Failed to load mesh from {:?}", input))?;

    // Determine decimation parameters
    let params = if let Some(target_count) = count {
        let mut p = DecimateParams::with_target_triangles(target_count);
        p.preserve_boundary = preserve_boundary;
        p
    } else {
        let target_ratio = ratio.unwrap_or(0.5);
        let mut p = DecimateParams::with_target_ratio(target_ratio);
        p.preserve_boundary = preserve_boundary;
        p
    };

    if !cli.quiet {
        output::info(
            &format!("Decimating mesh ({} triangles)...", mesh.face_count()),
            cli.format,
            cli.quiet,
        );
    }

    // Perform decimation
    let decimate_result = mesh.decimate_with_params(&params);

    // Save output
    decimate_result
        .mesh
        .save(output_path)
        .with_context(|| format!("Failed to save decimated mesh to {:?}", output_path))?;

    let reduction =
        1.0 - (decimate_result.final_triangles as f64 / decimate_result.original_triangles as f64);

    let result = DecimateResult {
        input: input.display().to_string(),
        output: output_path.display().to_string(),
        success: true,
        original_triangles: decimate_result.original_triangles,
        final_triangles: decimate_result.final_triangles,
        reduction_ratio: reduction,
        collapses_performed: decimate_result.collapses_performed,
    };

    match cli.format {
        OutputFormat::Json => {
            output::print(&result, cli.format, cli.quiet);
        }
        OutputFormat::Text => {
            if !cli.quiet {
                output::success(
                    &format!("Decimated mesh saved to {}", output_path.display()),
                    cli.format,
                    cli.quiet,
                );
                println!(
                    "  {}: {} → {} triangles ({:.1}% reduction)",
                    "Triangles".cyan(),
                    result.original_triangles,
                    result.final_triangles,
                    result.reduction_ratio * 100.0
                );
                println!(
                    "  {}: {} edge collapses",
                    "Operations".cyan(),
                    result.collapses_performed
                );
            }
        }
    }

    Ok(())
}

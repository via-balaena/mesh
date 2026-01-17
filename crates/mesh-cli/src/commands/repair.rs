//! mesh repair command - fix common mesh issues.

use std::path::Path;

use anyhow::{Context, Result};
use colored::Colorize;
use mesh_repair::{Mesh, RepairParams};
use serde::Serialize;

use crate::{Cli, OutputFormat, output};

#[derive(Serialize)]
struct RepairResult {
    input: String,
    output: String,
    success: bool,
    input_vertices: usize,
    input_faces: usize,
    output_vertices: usize,
    output_faces: usize,
    holes_filled: usize,
    degenerates_removed: usize,
}

#[allow(clippy::too_many_arguments)]
pub fn run(
    input: &Path,
    output_path: &Path,
    for_printing: bool,
    for_scan: bool,
    for_cad: bool,
    max_hole_edges: Option<usize>,
    weld_tolerance: Option<f64>,
    cli: &Cli,
) -> Result<()> {
    let mut mesh =
        Mesh::load(input).with_context(|| format!("Failed to load mesh from {:?}", input))?;

    let input_vertices = mesh.vertex_count();
    let input_faces = mesh.face_count();

    // Determine repair parameters
    let mut params = if for_printing {
        RepairParams::for_printing()
    } else if for_scan {
        RepairParams::for_scans()
    } else if for_cad {
        RepairParams::for_cad()
    } else {
        RepairParams::default()
    };

    // Apply custom options
    if let Some(max_edges) = max_hole_edges {
        params.max_hole_edges = max_edges;
    }
    if let Some(tolerance) = weld_tolerance {
        params.weld_epsilon = tolerance;
    }

    // Get initial state for comparison
    let initial_report = mesh.validate();
    let initial_holes = initial_report.boundary_edge_count;

    // Perform repair
    mesh.repair_with_config(&params)
        .with_context(|| "Repair operation failed")?;

    // Get final state
    let final_report = mesh.validate();
    let final_holes = final_report.boundary_edge_count;

    // Estimate repairs made
    let holes_filled = if initial_holes > final_holes {
        (initial_holes - final_holes) / 2 // Rough estimate
    } else {
        0
    };
    let degenerates_removed = if input_faces > mesh.face_count() {
        input_faces - mesh.face_count()
    } else {
        0
    };

    // Save output
    mesh.save(output_path)
        .with_context(|| format!("Failed to save repaired mesh to {:?}", output_path))?;

    let result = RepairResult {
        input: input.display().to_string(),
        output: output_path.display().to_string(),
        success: true,
        input_vertices,
        input_faces,
        output_vertices: mesh.vertex_count(),
        output_faces: mesh.face_count(),
        holes_filled,
        degenerates_removed,
    };

    match cli.format {
        OutputFormat::Json => {
            output::print(&result, cli.format, cli.quiet);
        }
        OutputFormat::Text => {
            if !cli.quiet {
                output::success(
                    &format!("Repaired mesh saved to {}", output_path.display()),
                    cli.format,
                    cli.quiet,
                );
                println!(
                    "  {}: {} → {} vertices",
                    "Vertices".cyan(),
                    result.input_vertices,
                    result.output_vertices
                );
                println!(
                    "  {}: {} → {} faces",
                    "Faces".cyan(),
                    result.input_faces,
                    result.output_faces
                );
                if result.holes_filled > 0 {
                    println!(
                        "  {}: ~{} holes filled",
                        "Repairs".green(),
                        result.holes_filled
                    );
                }
                if result.degenerates_removed > 0 {
                    println!(
                        "  {}: {} degenerate faces removed",
                        "Cleanup".green(),
                        result.degenerates_removed
                    );
                }
            }
        }
    }

    Ok(())
}

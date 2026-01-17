//! mesh slice command - slice mesh for 3D printing preview.

use std::path::Path;

use anyhow::{Context, Result};
use colored::Colorize;
use mesh_repair::{Mesh, SliceParams, calculate_layer_stats, slice_mesh};
use serde::Serialize;

use crate::{Cli, OutputFormat, output};

#[derive(Serialize)]
struct SliceInfo {
    path: String,
    layer_height: f64,
    total_layers: usize,
    total_height: f64,
    estimated_print_time_minutes: f64,
    estimated_filament_mm: f64,
    #[serde(skip_serializing_if = "Option::is_none")]
    stats: Option<LayerStatsInfo>,
    #[serde(skip_serializing_if = "Option::is_none")]
    layer_detail: Option<LayerDetailInfo>,
}

#[derive(Serialize)]
struct LayerStatsInfo {
    min_area: f64,
    max_area: f64,
    avg_area: f64,
    min_perimeter: f64,
    max_perimeter: f64,
    avg_perimeter: f64,
    max_islands: usize,
}

#[derive(Serialize)]
struct LayerDetailInfo {
    layer_number: usize,
    z_height: f64,
    area: f64,
    perimeter: f64,
    contour_count: usize,
}

pub fn run(
    input: &Path,
    layer_height: f64,
    layer_num: Option<usize>,
    detailed: bool,
    cli: &Cli,
) -> Result<()> {
    let mesh =
        Mesh::load(input).with_context(|| format!("Failed to load mesh from {:?}", input))?;

    let params = SliceParams {
        layer_height,
        ..SliceParams::default()
    };

    if !cli.quiet {
        output::info("Slicing mesh...", cli.format, cli.quiet);
    }

    let result = slice_mesh(&mesh, &params);

    let stats = if detailed {
        let s = calculate_layer_stats(&result);
        Some(LayerStatsInfo {
            min_area: s.min_area,
            max_area: s.max_area,
            avg_area: s.avg_area,
            min_perimeter: s.min_perimeter,
            max_perimeter: s.max_perimeter,
            avg_perimeter: s.avg_perimeter,
            max_islands: s.max_islands,
        })
    } else {
        None
    };

    let layer_detail = layer_num.and_then(|n| {
        result.layers.get(n).map(|layer| LayerDetailInfo {
            layer_number: n,
            z_height: layer.z_height,
            area: layer.area,
            perimeter: layer.perimeter,
            contour_count: layer.contours.len(),
        })
    });

    let info = SliceInfo {
        path: input.display().to_string(),
        layer_height,
        total_layers: result.layer_count,
        total_height: result.total_height,
        estimated_print_time_minutes: result.estimated_print_time,
        estimated_filament_mm: result.estimated_filament_length,
        stats,
        layer_detail,
    };

    match cli.format {
        OutputFormat::Json => {
            output::print(&info, cli.format, cli.quiet);
        }
        OutputFormat::Text => {
            if !cli.quiet {
                println!("{}", "Slice Information".bold().underline());
                println!("  {}: {}", "File".cyan(), input.display());
                println!("  {}: {:.2}mm", "Layer height".cyan(), layer_height);
                println!("  {}: {}", "Total layers".cyan(), info.total_layers);
                println!("  {}: {:.2}mm", "Total height".cyan(), info.total_height);
                println!(
                    "  {}: {:.1} minutes",
                    "Est. print time".cyan(),
                    info.estimated_print_time_minutes
                );
                println!(
                    "  {}: {:.1}mm ({:.2}m)",
                    "Est. filament".cyan(),
                    info.estimated_filament_mm,
                    info.estimated_filament_mm / 1000.0
                );

                if let Some(ref s) = info.stats {
                    println!("\n{}", "Layer Statistics:".bold());
                    println!(
                        "  {}: {:.2} - {:.2} mm² (avg: {:.2})",
                        "Area".cyan(),
                        s.min_area,
                        s.max_area,
                        s.avg_area
                    );
                    println!(
                        "  {}: {:.2} - {:.2} mm (avg: {:.2})",
                        "Perimeter".cyan(),
                        s.min_perimeter,
                        s.max_perimeter,
                        s.avg_perimeter
                    );
                    println!("  {}: {}", "Max islands".cyan(), s.max_islands);
                }

                if let Some(ref l) = info.layer_detail {
                    println!("\n{}", format!("Layer {}:", l.layer_number).bold());
                    println!("  {}: {:.2}mm", "Z height".cyan(), l.z_height);
                    println!("  {}: {:.2}mm²", "Area".cyan(), l.area);
                    println!("  {}: {:.2}mm", "Perimeter".cyan(), l.perimeter);
                    println!("  {}: {}", "Contours".cyan(), l.contour_count);
                }
            }
        }
    }

    Ok(())
}

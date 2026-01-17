//! mesh validate command - check mesh for issues.

use std::path::Path;

use anyhow::{Context, Result};
use colored::Colorize;
use mesh_repair::{Mesh, PrinterConfig};
use serde::Serialize;

use crate::{Cli, OutputFormat, output};

#[derive(Serialize)]
struct ValidationResult {
    path: String,
    valid: bool,
    issues: Vec<IssueInfo>,
    #[serde(skip_serializing_if = "Option::is_none")]
    printability: Option<PrintabilityInfo>,
}

#[derive(Serialize)]
struct IssueInfo {
    category: String,
    message: String,
    severity: String,
}

#[derive(Serialize)]
struct PrintabilityInfo {
    printable: bool,
    score: f64,
    thin_walls: usize,
    overhangs: usize,
    support_volume: f64,
}

pub fn run(input: &Path, check_printable: bool, min_thickness: f64, cli: &Cli) -> Result<()> {
    let mesh =
        Mesh::load(input).with_context(|| format!("Failed to load mesh from {:?}", input))?;

    let report = mesh.validate();
    let mut issues = Vec::new();

    // Collect validation issues
    if !report.is_watertight {
        issues.push(IssueInfo {
            category: "topology".to_string(),
            message: format!(
                "Mesh is not watertight ({} boundary edges)",
                report.boundary_edge_count
            ),
            severity: "error".to_string(),
        });
    }

    if !report.is_manifold {
        issues.push(IssueInfo {
            category: "topology".to_string(),
            message: format!(
                "Mesh is not manifold ({} non-manifold edges)",
                report.non_manifold_edge_count
            ),
            severity: "error".to_string(),
        });
    }

    if report.is_inside_out {
        issues.push(IssueInfo {
            category: "winding".to_string(),
            message: "Mesh appears to be inside-out".to_string(),
            severity: "warning".to_string(),
        });
    }

    // Note: degenerate face check requires a separate analysis
    // The basic validation doesn't include this

    // Check printability if requested
    let printability = if check_printable {
        let mut config = PrinterConfig::fdm_default();
        config.min_wall_thickness = min_thickness;
        let validation = mesh_repair::validate_for_printing(&mesh, &config);

        Some(PrintabilityInfo {
            printable: validation.printable,
            score: validation.score,
            thin_walls: validation.thin_walls.len(),
            overhangs: validation.overhangs.len(),
            support_volume: validation.estimated_support_volume,
        })
    } else {
        None
    };

    let valid = issues.iter().all(|i| i.severity != "error")
        && printability.as_ref().is_none_or(|p| p.printable);

    let result = ValidationResult {
        path: input.display().to_string(),
        valid,
        issues,
        printability,
    };

    match cli.format {
        OutputFormat::Json => {
            output::print(&result, cli.format, cli.quiet);
        }
        OutputFormat::Text => {
            if !cli.quiet {
                println!("{}", "Validation Report".bold().underline());
                println!("  {}: {}", "File".cyan(), input.display());

                if result.valid {
                    println!("  {}: {}", "Status".cyan(), "Valid".green().bold());
                } else {
                    println!("  {}: {}", "Status".cyan(), "Issues found".red().bold());
                }

                if !result.issues.is_empty() {
                    println!("\n{}", "Issues:".bold());
                    for issue in &result.issues {
                        let icon = match issue.severity.as_str() {
                            "error" => "✗".red(),
                            "warning" => "⚠".yellow(),
                            _ => "ℹ".blue(),
                        };
                        println!("  {} [{}] {}", icon, issue.category, issue.message);
                    }
                }

                if let Some(ref p) = result.printability {
                    println!("\n{}", "Printability:".bold());
                    println!(
                        "  {}: {}",
                        "Printable".cyan(),
                        if p.printable {
                            "Yes".green().bold()
                        } else {
                            "No".red().bold()
                        }
                    );
                    println!("  {}: {:.0}%", "Score".cyan(), p.score * 100.0);
                    if p.thin_walls > 0 {
                        println!("  {}: {} regions", "Thin walls".yellow(), p.thin_walls);
                    }
                    if p.overhangs > 0 {
                        println!("  {}: {} regions", "Overhangs".yellow(), p.overhangs);
                    }
                    if p.support_volume > 0.0 {
                        println!(
                            "  {}: {:.2} mm³",
                            "Est. support volume".cyan(),
                            p.support_volume
                        );
                    }
                }
            }
        }
    }

    // Exit with error code if invalid
    if !result.valid {
        std::process::exit(1);
    }

    Ok(())
}

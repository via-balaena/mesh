//! mesh-cli: Command-line interface for mesh processing and repair.
//!
//! This tool provides access to mesh-repair and mesh-shell functionality
//! from the command line, suitable for scripting and CI/CD pipelines.
//!
//! # Logging
//!
//! Set the `RUST_LOG` environment variable to control log output:
//! - `RUST_LOG=mesh_repair=info` - Basic operation logging
//! - `RUST_LOG=mesh_repair=debug` - Detailed progress logging
//! - `RUST_LOG=mesh_repair::timing=debug` - Performance timing
//! - `RUST_LOG=debug` - All debug output
//!
//! # Example
//!
//! ```bash
//! # Basic repair with info logging
//! RUST_LOG=mesh_repair=info mesh repair input.stl -o output.stl
//!
//! # Debug output for troubleshooting
//! RUST_LOG=debug mesh validate input.stl
//! ```

use std::path::PathBuf;

use anyhow::Result;
use clap::{Parser, Subcommand, ValueEnum};
use colored::Colorize;
use tracing_subscriber::{EnvFilter, fmt, prelude::*};

mod commands;
mod output;

use commands::{convert, decimate, info, repair, shell, slice, validate};

/// mesh - A command-line tool for mesh processing and repair.
///
/// Process, repair, and transform 3D meshes for manufacturing and 3D printing.
#[derive(Parser)]
#[command(name = "mesh")]
#[command(author, version, about, long_about = None)]
struct Cli {
    #[command(subcommand)]
    command: Commands,

    /// Output format for results
    #[arg(long, global = true, default_value = "text")]
    format: OutputFormat,

    /// Suppress all non-error output
    #[arg(long, short, global = true)]
    quiet: bool,

    /// Increase output verbosity (-v for info, -vv for debug, -vvv for trace)
    #[arg(long, short, global = true, action = clap::ArgAction::Count)]
    verbose: u8,
}

#[derive(Clone, Copy, ValueEnum)]
pub enum OutputFormat {
    /// Human-readable text output
    Text,
    /// JSON output for scripting
    Json,
}

#[derive(Subcommand)]
enum Commands {
    /// Display mesh statistics and information
    Info {
        /// Input mesh file
        input: PathBuf,

        /// Show detailed vertex/face statistics
        #[arg(long)]
        detailed: bool,
    },

    /// Validate mesh for issues and printability
    Validate {
        /// Input mesh file
        input: PathBuf,

        /// Check printability requirements
        #[arg(long)]
        printable: bool,

        /// Minimum wall thickness for print validation (mm)
        #[arg(long, default_value = "0.8")]
        min_thickness: f64,
    },

    /// Repair common mesh issues
    Repair {
        /// Input mesh file
        input: PathBuf,

        /// Output file path
        #[arg(short, long)]
        output: PathBuf,

        /// Use printing-optimized repair settings
        #[arg(long)]
        for_printing: bool,

        /// Use scan-optimized repair settings
        #[arg(long)]
        for_scan: bool,

        /// Use CAD-optimized repair settings
        #[arg(long)]
        for_cad: bool,

        /// Fill holes up to this edge count
        #[arg(long)]
        max_hole_edges: Option<usize>,

        /// Vertex welding tolerance
        #[arg(long)]
        weld_tolerance: Option<f64>,
    },

    /// Generate a shell (offset surface) from a mesh
    Shell {
        /// Input mesh file
        input: PathBuf,

        /// Output file path
        #[arg(short, long)]
        output: PathBuf,

        /// Shell thickness in mm
        #[arg(long, short, default_value = "2.0")]
        thickness: f64,

        /// Grid resolution for SDF computation
        #[arg(long, default_value = "128")]
        resolution: usize,

        /// Direction of offset (outward, inward, or both)
        #[arg(long, default_value = "outward")]
        direction: ShellDirection,
    },

    /// Decimate (simplify) a mesh to reduce triangle count
    Decimate {
        /// Input mesh file
        input: PathBuf,

        /// Output file path
        #[arg(short, long)]
        output: PathBuf,

        /// Target ratio (0.0-1.0) of original triangles to keep
        #[arg(long, conflicts_with = "count")]
        ratio: Option<f64>,

        /// Target number of triangles
        #[arg(long, conflicts_with = "ratio")]
        count: Option<usize>,

        /// Preserve mesh boundary edges
        #[arg(long)]
        preserve_boundary: bool,
    },

    /// Convert mesh between formats
    Convert {
        /// Input mesh file
        input: PathBuf,

        /// Output file path (format determined by extension)
        #[arg(short, long)]
        output: PathBuf,

        /// Use ASCII format when available (STL, PLY)
        #[arg(long)]
        ascii: bool,
    },

    /// Slice mesh for 3D printing preview
    Slice {
        /// Input mesh file
        input: PathBuf,

        /// Layer height in mm
        #[arg(long, default_value = "0.2")]
        layer_height: f64,

        /// Show specific layer number
        #[arg(long)]
        layer: Option<usize>,

        /// Show detailed per-layer statistics
        #[arg(long)]
        detailed: bool,
    },
}

#[derive(Clone, Copy, ValueEnum)]
pub enum ShellDirection {
    /// Offset outward (make larger)
    Outward,
    /// Offset inward (make smaller)
    Inward,
    /// Offset both directions (create solid shell)
    Both,
}

/// Initialize the tracing subscriber based on verbosity level.
fn init_tracing(verbose: u8, quiet: bool) {
    // If quiet, don't initialize any tracing
    if quiet {
        return;
    }

    // Determine log level based on verbosity flag
    // Check RUST_LOG first, then fall back to -v flags
    let filter = if std::env::var("RUST_LOG").is_ok() {
        EnvFilter::from_default_env()
    } else {
        let level = match verbose {
            0 => "warn",
            1 => "mesh_repair=info,mesh_shell=info",
            2 => "mesh_repair=debug,mesh_shell=debug",
            _ => "trace",
        };
        EnvFilter::try_new(level).unwrap_or_else(|_| EnvFilter::new("warn"))
    };

    // Initialize the subscriber
    tracing_subscriber::registry()
        .with(fmt::layer().with_writer(std::io::stderr).compact())
        .with(filter)
        .init();
}

fn main() -> Result<()> {
    // Install miette's panic hook for better error display
    // This makes panics show nicer error reports in development
    #[cfg(debug_assertions)]
    miette::set_panic_hook();

    let cli = Cli::parse();

    // Initialize tracing
    init_tracing(cli.verbose, cli.quiet);

    let result = match &cli.command {
        Commands::Info { input, detailed } => info::run(input, *detailed, &cli),
        Commands::Validate {
            input,
            printable,
            min_thickness,
        } => validate::run(input, *printable, *min_thickness, &cli),
        Commands::Repair {
            input,
            output,
            for_printing,
            for_scan,
            for_cad,
            max_hole_edges,
            weld_tolerance,
        } => repair::run(
            input,
            output,
            *for_printing,
            *for_scan,
            *for_cad,
            *max_hole_edges,
            *weld_tolerance,
            &cli,
        ),
        Commands::Shell {
            input,
            output,
            thickness,
            resolution,
            direction,
        } => shell::run(input, output, *thickness, *resolution, *direction, &cli),
        Commands::Decimate {
            input,
            output,
            ratio,
            count,
            preserve_boundary,
        } => decimate::run(input, output, *ratio, *count, *preserve_boundary, &cli),
        Commands::Convert {
            input,
            output,
            ascii,
        } => convert::run(input, output, *ascii, &cli),
        Commands::Slice {
            input,
            layer_height,
            layer,
            detailed,
        } => slice::run(input, *layer_height, *layer, *detailed, &cli),
    };

    if let Err(e) = &result {
        if !cli.quiet {
            // Check if the error is a miette Diagnostic for enhanced display
            if let Some(mesh_err) = e.downcast_ref::<mesh_repair::MeshError>() {
                // Display error with code and help text
                eprintln!("{}: {}", "Error".red().bold(), mesh_err);
                eprintln!("  {}: {}", "Code".cyan(), mesh_err.code());
                eprintln!(
                    "  {}: {}",
                    "Suggestion".green(),
                    mesh_err.recovery_suggestion()
                );
                if let Some(location) = mesh_err.location() {
                    eprintln!("  {}: {}", "Location".yellow(), location);
                }
            } else {
                // Fall back to standard error display
                eprintln!("{}: {}", "Error".red().bold(), e);
                for cause in e.chain().skip(1) {
                    eprintln!("  {}: {}", "Caused by".yellow(), cause);
                }
            }
        }
        std::process::exit(1);
    }

    Ok(())
}

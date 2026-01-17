//! Benchmarks for mesh-shell operations.
//!
//! Run with: cargo bench -p mesh-shell
//!
//! To compare against baseline:
//! 1. First run: cargo bench -p mesh-shell -- --save-baseline main
//! 2. After changes: cargo bench -p mesh-shell -- --baseline main

use criterion::{BenchmarkId, Criterion, Throughput, black_box, criterion_group, criterion_main};
use mesh_repair::{Mesh, Vertex};
use mesh_shell::{ShellParams, WallGenerationMethod, generate_shell};

// =============================================================================
// Test Mesh Generation
// =============================================================================

/// Create a unit cube mesh (12 triangles).
fn create_cube() -> Mesh {
    let mut mesh = Mesh::new();

    let verts = [
        [-0.5, -0.5, -0.5],
        [0.5, -0.5, -0.5],
        [0.5, 0.5, -0.5],
        [-0.5, 0.5, -0.5],
        [-0.5, -0.5, 0.5],
        [0.5, -0.5, 0.5],
        [0.5, 0.5, 0.5],
        [-0.5, 0.5, 0.5],
    ];

    for v in &verts {
        mesh.vertices.push(Vertex::from_coords(v[0], v[1], v[2]));
    }

    let faces = [
        [0, 1, 2],
        [0, 2, 3],
        [4, 6, 5],
        [4, 7, 6],
        [0, 4, 5],
        [0, 5, 1],
        [2, 6, 7],
        [2, 7, 3],
        [0, 3, 7],
        [0, 7, 4],
        [1, 5, 6],
        [1, 6, 2],
    ];

    for f in &faces {
        mesh.faces.push([f[0] as u32, f[1] as u32, f[2] as u32]);
    }

    mesh
}

/// Create an icosphere mesh with specified subdivision level.
fn create_sphere(subdivisions: u32) -> Mesh {
    let mut mesh = Mesh::new();

    let phi = (1.0 + 5.0_f64.sqrt()) / 2.0;
    let a = 1.0;
    let b = 1.0 / phi;

    let ico_verts = [
        [0.0, b, -a],
        [b, a, 0.0],
        [-b, a, 0.0],
        [0.0, b, a],
        [0.0, -b, a],
        [-a, 0.0, b],
        [0.0, -b, -a],
        [a, 0.0, -b],
        [a, 0.0, b],
        [-a, 0.0, -b],
        [b, -a, 0.0],
        [-b, -a, 0.0],
    ];

    for v in &ico_verts {
        let len = (v[0] * v[0] + v[1] * v[1] + v[2] * v[2]).sqrt();
        mesh.vertices
            .push(Vertex::from_coords(v[0] / len, v[1] / len, v[2] / len));
    }

    let ico_faces: [[u32; 3]; 20] = [
        [0, 1, 2],
        [3, 2, 1],
        [3, 4, 5],
        [3, 8, 4],
        [0, 6, 7],
        [0, 9, 6],
        [4, 10, 11],
        [6, 11, 10],
        [2, 5, 9],
        [11, 9, 5],
        [1, 7, 8],
        [10, 8, 7],
        [3, 5, 2],
        [3, 1, 8],
        [0, 2, 9],
        [0, 7, 1],
        [6, 9, 11],
        [6, 10, 7],
        [4, 11, 5],
        [4, 8, 10],
    ];

    for f in &ico_faces {
        mesh.faces.push(*f);
    }

    for _ in 0..subdivisions {
        mesh = subdivide_sphere(&mesh);
    }

    mesh
}

fn subdivide_sphere(mesh: &Mesh) -> Mesh {
    use std::collections::HashMap;

    let mut new_mesh = Mesh::new();
    new_mesh.vertices = mesh.vertices.clone();

    let mut edge_midpoints: HashMap<(u32, u32), u32> = HashMap::new();

    let mut get_midpoint = |v1: u32, v2: u32, vertices: &mut Vec<Vertex>| -> u32 {
        let key = if v1 < v2 { (v1, v2) } else { (v2, v1) };

        if let Some(&idx) = edge_midpoints.get(&key) {
            return idx;
        }

        let p1 = &vertices[v1 as usize];
        let p2 = &vertices[v2 as usize];

        let mx = (p1.position.x + p2.position.x) / 2.0;
        let my = (p1.position.y + p2.position.y) / 2.0;
        let mz = (p1.position.z + p2.position.z) / 2.0;
        let len = (mx * mx + my * my + mz * mz).sqrt();

        let idx = vertices.len() as u32;
        vertices.push(Vertex::from_coords(mx / len, my / len, mz / len));
        edge_midpoints.insert(key, idx);
        idx
    };

    for face in &mesh.faces {
        let v0 = face[0];
        let v1 = face[1];
        let v2 = face[2];

        let m01 = get_midpoint(v0, v1, &mut new_mesh.vertices);
        let m12 = get_midpoint(v1, v2, &mut new_mesh.vertices);
        let m20 = get_midpoint(v2, v0, &mut new_mesh.vertices);

        new_mesh.faces.push([v0, m01, m20]);
        new_mesh.faces.push([v1, m12, m01]);
        new_mesh.faces.push([v2, m20, m12]);
        new_mesh.faces.push([m01, m12, m20]);
    }

    new_mesh
}

// =============================================================================
// Shell Generation Benchmarks
// =============================================================================

fn bench_shell_generation(c: &mut Criterion) {
    let mut group = c.benchmark_group("ShellGeneration");
    group.sample_size(10); // Shell generation is slow

    let test_cases = [
        ("cube_12tri", create_cube()),
        ("sphere_80tri", create_sphere(1)),
        ("sphere_320tri", create_sphere(2)),
    ];

    let thicknesses = [0.1, 0.2];

    for (name, mesh) in &test_cases {
        for &thickness in &thicknesses {
            group.throughput(Throughput::Elements(mesh.faces.len() as u64));

            let params = ShellParams {
                wall_thickness_mm: thickness,
                wall_generation_method: WallGenerationMethod::Normal,
                ..Default::default()
            };

            group.bench_with_input(
                BenchmarkId::new(format!("shell_t{}", thickness), name),
                &(mesh, &params),
                |b, (mesh, params)| b.iter(|| generate_shell(black_box(mesh), black_box(params))),
            );
        }
    }

    group.finish();
}

// =============================================================================
// SDF Method Benchmarks
// =============================================================================

fn bench_sdf_shell(c: &mut Criterion) {
    let mut group = c.benchmark_group("SdfShell");
    group.sample_size(10);

    let cube = create_cube();
    let sphere = create_sphere(1);

    // Normal method vs SDF method comparison
    let normal_params = ShellParams {
        wall_thickness_mm: 0.1,
        wall_generation_method: WallGenerationMethod::Normal,
        ..Default::default()
    };

    let sdf_params = ShellParams {
        wall_thickness_mm: 0.1,
        wall_generation_method: WallGenerationMethod::Sdf,
        sdf_voxel_size_mm: 0.05,
        ..Default::default()
    };

    group.bench_function("cube_normal", |b| {
        b.iter(|| generate_shell(black_box(&cube), black_box(&normal_params)))
    });

    group.bench_function("cube_sdf", |b| {
        b.iter(|| generate_shell(black_box(&cube), black_box(&sdf_params)))
    });

    group.bench_function("sphere_80tri_normal", |b| {
        b.iter(|| generate_shell(black_box(&sphere), black_box(&normal_params)))
    });

    group.bench_function("sphere_80tri_sdf", |b| {
        b.iter(|| generate_shell(black_box(&sphere), black_box(&sdf_params)))
    });

    group.finish();
}

// =============================================================================
// Memory Estimation Benchmarks
// =============================================================================

fn bench_estimation(c: &mut Criterion) {
    let mut group = c.benchmark_group("Estimation");

    let test_cases = [
        ("cube", create_cube()),
        ("sphere_320tri", create_sphere(2)),
        ("sphere_1280tri", create_sphere(3)),
    ];

    for (name, mesh) in &test_cases {
        let params = ShellParams::default();

        group.bench_with_input(
            BenchmarkId::new("estimate_grid_size", name),
            &(&mesh, &params),
            |b, (mesh, params)| {
                b.iter(|| {
                    // Estimate without actually computing
                    let bounds = mesh.bounds().unwrap();
                    let size = bounds.1 - bounds.0;
                    let voxel_size = params.sdf_voxel_size_mm;
                    let dims = [
                        (size.x / voxel_size).ceil() as usize + 4,
                        (size.y / voxel_size).ceil() as usize + 4,
                        (size.z / voxel_size).ceil() as usize + 4,
                    ];
                    black_box(dims[0] * dims[1] * dims[2])
                })
            },
        );
    }

    group.finish();
}

// =============================================================================
// Criterion Setup
// =============================================================================

criterion_group!(
    benches,
    bench_shell_generation,
    bench_sdf_shell,
    bench_estimation,
);

criterion_main!(benches);

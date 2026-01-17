//! Benchmarks for mesh-repair operations.
//!
//! Run with: cargo bench -p mesh-repair
//!
//! To compare against baseline:
//! 1. First run: cargo bench -p mesh-repair -- --save-baseline main
//! 2. After changes: cargo bench -p mesh-repair -- --baseline main

use criterion::{BenchmarkId, Criterion, Throughput, black_box, criterion_group, criterion_main};
use mesh_repair::{DecimateParams, Mesh, RemeshParams, Vertex};

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
        [0, 2, 3], // front
        [4, 6, 5],
        [4, 7, 6], // back
        [0, 4, 5],
        [0, 5, 1], // bottom
        [2, 6, 7],
        [2, 7, 3], // top
        [0, 3, 7],
        [0, 7, 4], // left
        [1, 5, 6],
        [1, 6, 2], // right
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

/// Create a mesh with holes (open cube missing one face).
fn create_open_cube() -> Mesh {
    let mut mesh = create_cube();
    mesh.faces.pop(); // Remove one face to create a hole
    mesh.faces.pop();
    mesh
}

// =============================================================================
// Validation Benchmarks
// =============================================================================

fn bench_validation(c: &mut Criterion) {
    let mut group = c.benchmark_group("Validation");

    let test_cases = [
        ("cube_12tri", create_cube()),
        ("sphere_80tri", create_sphere(1)),
        ("sphere_320tri", create_sphere(2)),
        ("sphere_1280tri", create_sphere(3)),
        ("sphere_5120tri", create_sphere(4)),
    ];

    for (name, mesh) in &test_cases {
        group.throughput(Throughput::Elements(mesh.faces.len() as u64));

        group.bench_with_input(BenchmarkId::new("validate", name), mesh, |b, mesh| {
            b.iter(|| mesh_repair::validate_mesh(black_box(mesh)))
        });
    }

    group.finish();
}

// =============================================================================
// Repair Benchmarks
// =============================================================================

fn bench_repair(c: &mut Criterion) {
    let mut group = c.benchmark_group("Repair");

    let test_cases = [
        ("cube_12tri", create_cube()),
        ("sphere_320tri", create_sphere(2)),
        ("sphere_1280tri", create_sphere(3)),
    ];

    for (name, mesh) in &test_cases {
        group.throughput(Throughput::Elements(mesh.faces.len() as u64));

        group.bench_with_input(BenchmarkId::new("fix_winding", name), mesh, |b, mesh| {
            let mut m = mesh.clone();
            b.iter(|| {
                let _ = mesh_repair::fix_winding_order(&mut m);
            })
        });

        group.bench_with_input(
            BenchmarkId::new("remove_degenerate", name),
            mesh,
            |b, mesh| {
                let mut m = mesh.clone();
                b.iter(|| {
                    mesh_repair::remove_degenerate_triangles(&mut m, 1e-10);
                })
            },
        );

        group.bench_with_input(BenchmarkId::new("weld_vertices", name), mesh, |b, mesh| {
            let mut m = mesh.clone();
            b.iter(|| {
                mesh_repair::weld_vertices(&mut m, 1e-6);
            })
        });
    }

    group.finish();
}

// =============================================================================
// Hole Filling Benchmarks
// =============================================================================

fn bench_hole_filling(c: &mut Criterion) {
    let mut group = c.benchmark_group("HoleFilling");

    // Create meshes with holes
    let open_cube = create_open_cube();

    group.bench_function("fill_holes_cube", |b| {
        let mut mesh = open_cube.clone();
        b.iter(|| {
            let _ = mesh_repair::fill_holes(&mut mesh);
        })
    });

    group.finish();
}

// =============================================================================
// Decimation Benchmarks
// =============================================================================

fn bench_decimation(c: &mut Criterion) {
    let mut group = c.benchmark_group("Decimation");
    group.sample_size(20); // Decimation is slower, reduce samples

    let test_cases = [
        ("sphere_320tri", create_sphere(2)),
        ("sphere_1280tri", create_sphere(3)),
        ("sphere_5120tri", create_sphere(4)),
    ];

    for (name, mesh) in &test_cases {
        let target = mesh.faces.len() / 2; // Reduce by 50%

        group.throughput(Throughput::Elements(mesh.faces.len() as u64));

        group.bench_with_input(
            BenchmarkId::new("decimate_50pct", name),
            &(mesh, target),
            |b, (mesh, target)| {
                let params = DecimateParams::with_target_triangles(*target);
                b.iter(|| mesh_repair::decimate_mesh(black_box(mesh), black_box(&params)))
            },
        );
    }

    group.finish();
}

// =============================================================================
// Remeshing Benchmarks
// =============================================================================

fn bench_remeshing(c: &mut Criterion) {
    let mut group = c.benchmark_group("Remeshing");
    group.sample_size(10); // Remeshing is very slow

    let test_cases = [
        ("sphere_320tri", create_sphere(2)),
        ("sphere_1280tri", create_sphere(3)),
    ];

    for (name, mesh) in &test_cases {
        let target_edge_length = 0.1;

        group.throughput(Throughput::Elements(mesh.faces.len() as u64));

        group.bench_with_input(BenchmarkId::new("isotropic", name), mesh, |b, mesh| {
            let params = RemeshParams {
                target_edge_length: Some(target_edge_length),
                iterations: 3,
                ..Default::default()
            };
            b.iter(|| mesh_repair::remesh_isotropic(black_box(mesh), black_box(&params)))
        });
    }

    group.finish();
}

// =============================================================================
// I/O Benchmarks
// =============================================================================

fn bench_io(c: &mut Criterion) {
    let mut group = c.benchmark_group("IO");

    let sphere = create_sphere(4); // ~5k triangles

    // Write to temp file for reading benchmarks
    let temp_dir = std::env::temp_dir();
    let stl_path = temp_dir.join("bench_sphere.stl");
    let obj_path = temp_dir.join("bench_sphere.obj");

    let _ = mesh_repair::save_mesh(&sphere, &stl_path);
    let _ = mesh_repair::save_mesh(&sphere, &obj_path);

    group.throughput(Throughput::Elements(sphere.faces.len() as u64));

    group.bench_function("load_stl", |b| {
        b.iter(|| mesh_repair::load_mesh(black_box(&stl_path)))
    });

    group.bench_function("load_obj", |b| {
        b.iter(|| mesh_repair::load_mesh(black_box(&obj_path)))
    });

    group.bench_function("save_stl", |b| {
        let out_path = temp_dir.join("bench_out.stl");
        b.iter(|| mesh_repair::save_mesh(black_box(&sphere), black_box(&out_path)))
    });

    group.finish();

    // Cleanup
    let _ = std::fs::remove_file(&stl_path);
    let _ = std::fs::remove_file(&obj_path);
}

// =============================================================================
// Intersection Detection Benchmarks
// =============================================================================

fn bench_intersection(c: &mut Criterion) {
    let mut group = c.benchmark_group("Intersection");

    let test_cases = [
        ("sphere_320tri", create_sphere(2)),
        ("sphere_1280tri", create_sphere(3)),
    ];

    for (name, mesh) in &test_cases {
        group.throughput(Throughput::Elements(mesh.faces.len() as u64));

        group.bench_with_input(
            BenchmarkId::new("detect_self_intersection", name),
            mesh,
            |b, mesh| {
                let params = mesh_repair::intersect::IntersectionParams::default();
                b.iter(|| {
                    mesh_repair::intersect::detect_self_intersections(
                        black_box(mesh),
                        black_box(&params),
                    )
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
    bench_validation,
    bench_repair,
    bench_hole_filling,
    bench_decimation,
    bench_remeshing,
    bench_io,
    bench_intersection,
);

criterion_main!(benches);

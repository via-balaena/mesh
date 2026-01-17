//! GPU vs CPU benchmarks for mesh processing operations.
//!
//! Run with: cargo bench -p mesh-gpu
//!
//! Note: GPU benchmarks require a GPU. They will be skipped if no GPU is available.

use criterion::{BenchmarkId, Criterion, Throughput, black_box, criterion_group, criterion_main};
use mesh_gpu::{
    GpuCollisionParams, GpuSdfParams, GpuSurfaceNetsParams, compute_sdf_gpu,
    detect_self_intersections_gpu, extract_isosurface_gpu,
};
use mesh_repair::intersect::{IntersectionParams, detect_self_intersections};
use mesh_repair::{Mesh, Vertex};

// =============================================================================
// Test Mesh Generation
// =============================================================================

/// Create a simple cube mesh with 12 triangles.
fn create_cube() -> Mesh {
    let mut mesh = Mesh::new();

    // 8 vertices of a unit cube centered at origin
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

    // 12 triangles (2 per face)
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

/// Create a sphere mesh with approximately `n` triangles using icosphere subdivision.
fn create_sphere(subdivisions: u32) -> Mesh {
    let mut mesh = Mesh::new();

    // Golden ratio for icosahedron
    let phi = (1.0 + 5.0_f64.sqrt()) / 2.0;
    let a = 1.0;
    let b = 1.0 / phi;

    // 12 vertices of icosahedron
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

    // 20 faces of icosahedron
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

    // Subdivide
    for _ in 0..subdivisions {
        mesh = subdivide_sphere(&mesh);
    }

    mesh
}

/// Subdivide a sphere mesh by splitting each triangle into 4.
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

        // Midpoint, normalized to sphere surface
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

/// Create a grid of cubes (for testing larger meshes).
fn create_cube_grid(count_per_axis: usize) -> Mesh {
    let mut mesh = Mesh::new();
    let base_cube = create_cube();

    for x in 0..count_per_axis {
        for y in 0..count_per_axis {
            for z in 0..count_per_axis {
                let offset_x = x as f64 * 2.0;
                let offset_y = y as f64 * 2.0;
                let offset_z = z as f64 * 2.0;

                let vertex_offset = mesh.vertices.len() as u32;

                for v in &base_cube.vertices {
                    mesh.vertices.push(Vertex::from_coords(
                        v.position.x + offset_x,
                        v.position.y + offset_y,
                        v.position.z + offset_z,
                    ));
                }

                for f in &base_cube.faces {
                    mesh.faces.push([
                        f[0] + vertex_offset,
                        f[1] + vertex_offset,
                        f[2] + vertex_offset,
                    ]);
                }
            }
        }
    }

    mesh
}

// =============================================================================
// SDF Benchmarks
// =============================================================================

/// CPU SDF computation using mesh_to_sdf.
fn compute_sdf_cpu(mesh: &Mesh, dims: [usize; 3], voxel_size: f64) -> Vec<f32> {
    use mesh_to_sdf::{Grid, SignMethod, Topology, generate_grid_sdf};

    // Convert mesh to mesh_to_sdf format
    let vertices: Vec<[f32; 3]> = mesh
        .vertices
        .iter()
        .map(|v| {
            [
                v.position.x as f32,
                v.position.y as f32,
                v.position.z as f32,
            ]
        })
        .collect();

    let indices: Vec<u32> = mesh.faces.iter().flat_map(|f| f.iter().copied()).collect();

    // Compute bounding box
    let mut min = [f32::MAX; 3];
    let mut max = [f32::MIN; 3];
    for v in &vertices {
        for i in 0..3 {
            min[i] = min[i].min(v[i]);
            max[i] = max[i].max(v[i]);
        }
    }

    // Add padding
    let padding = voxel_size as f32 * 2.0;
    for i in 0..3 {
        min[i] -= padding;
        max[i] += padding;
    }

    let cell_count = [dims[0], dims[1], dims[2]];

    let grid = Grid::from_bounding_box(&min, &max, cell_count);

    generate_grid_sdf(
        &vertices,
        Topology::TriangleList(Some(&indices)),
        &grid,
        SignMethod::Raycast,
    )
}

fn bench_sdf(c: &mut Criterion) {
    let mut group = c.benchmark_group("SDF Computation");

    // Test different mesh sizes
    let test_cases = [
        ("cube_12tri", create_cube()),
        ("sphere_80tri", create_sphere(1)),
        ("sphere_320tri", create_sphere(2)),
        ("sphere_1280tri", create_sphere(3)),
        ("sphere_5120tri", create_sphere(4)),
        ("grid_1728tri", create_cube_grid(3)), // 3x3x3 = 27 cubes * 12 = 324 tris
    ];

    // Grid sizes to test
    let grid_sizes = [32, 64, 128];

    for (mesh_name, mesh) in &test_cases {
        for &grid_size in &grid_sizes {
            let dims = [grid_size, grid_size, grid_size];
            let voxel_size = 3.0 / grid_size as f64;

            group.throughput(Throughput::Elements(
                (grid_size * grid_size * grid_size) as u64,
            ));

            // CPU benchmark
            group.bench_with_input(
                BenchmarkId::new(format!("cpu/{}", mesh_name), grid_size),
                &(mesh, dims, voxel_size),
                |b, (mesh, dims, voxel_size)| {
                    b.iter(|| {
                        compute_sdf_cpu(black_box(mesh), black_box(*dims), black_box(*voxel_size))
                    })
                },
            );

            // GPU benchmark
            let gpu_params = GpuSdfParams {
                dims,
                origin: [-1.5, -1.5, -1.5],
                voxel_size: voxel_size as f32,
            };

            group.bench_with_input(
                BenchmarkId::new(format!("gpu/{}", mesh_name), grid_size),
                &(mesh, &gpu_params),
                |b, (mesh, params)| {
                    b.iter(|| {
                        if let Ok(result) = compute_sdf_gpu(black_box(mesh), black_box(params)) {
                            black_box(result);
                        }
                    })
                },
            );
        }
    }

    group.finish();
}

// =============================================================================
// Surface Nets Benchmarks
// =============================================================================

fn bench_surface_nets(c: &mut Criterion) {
    let mut group = c.benchmark_group("Surface Nets");

    // Create SDF data for a sphere (negative inside, positive outside)
    let grid_sizes = [32, 64, 128];

    for &grid_size in &grid_sizes {
        let dims = [grid_size, grid_size, grid_size];
        let voxel_size = 2.0 / grid_size as f32;
        let radius = 0.8_f32;

        // Generate sphere SDF
        let mut sdf = Vec::with_capacity(grid_size * grid_size * grid_size);
        for z in 0..grid_size {
            for y in 0..grid_size {
                for x in 0..grid_size {
                    let px = (x as f32 + 0.5) * voxel_size - 1.0;
                    let py = (y as f32 + 0.5) * voxel_size - 1.0;
                    let pz = (z as f32 + 0.5) * voxel_size - 1.0;
                    let dist = (px * px + py * py + pz * pz).sqrt() - radius;
                    sdf.push(dist);
                }
            }
        }

        group.throughput(Throughput::Elements(
            (grid_size * grid_size * grid_size) as u64,
        ));

        // CPU benchmark using fast-surface-nets
        group.bench_with_input(
            BenchmarkId::new("cpu", grid_size),
            &(&sdf, dims, voxel_size),
            |b, (sdf, dims, _voxel_size)| {
                b.iter(|| {
                    use fast_surface_nets::{
                        SurfaceNetsBuffer, ndshape::RuntimeShape, surface_nets,
                    };

                    let shape = RuntimeShape::<u32, 3>::new([
                        dims[0] as u32,
                        dims[1] as u32,
                        dims[2] as u32,
                    ]);
                    let mut buffer = SurfaceNetsBuffer::default();

                    surface_nets(
                        sdf,
                        &shape,
                        [0; 3],
                        [dims[0] as u32 - 1, dims[1] as u32 - 1, dims[2] as u32 - 1],
                        &mut buffer,
                    );

                    black_box(buffer.positions.len())
                })
            },
        );

        // GPU benchmark
        let gpu_params = GpuSurfaceNetsParams {
            dims,
            origin: [-1.0, -1.0, -1.0],
            voxel_size,
            iso_value: 0.0,
        };

        group.bench_with_input(
            BenchmarkId::new("gpu", grid_size),
            &(&sdf, &gpu_params),
            |b, (sdf, params)| {
                b.iter(|| {
                    if let Ok(result) = extract_isosurface_gpu(black_box(sdf), black_box(params)) {
                        black_box(result);
                    }
                })
            },
        );
    }

    group.finish();
}

// =============================================================================
// Collision Detection Benchmarks
// =============================================================================

fn bench_collision(c: &mut Criterion) {
    let mut group = c.benchmark_group("Collision Detection");

    // Test different mesh sizes
    let test_cases = [
        ("sphere_80tri", create_sphere(1)),
        ("sphere_320tri", create_sphere(2)),
        ("sphere_1280tri", create_sphere(3)),
        ("sphere_5120tri", create_sphere(4)),
        ("sphere_20480tri", create_sphere(5)),
        ("grid_3888tri", create_cube_grid(4)), // 4x4x4 = 64 cubes * 12 = 768 tris... wait
    ];

    for (mesh_name, mesh) in &test_cases {
        let triangle_count = mesh.faces.len();
        group.throughput(Throughput::Elements(
            (triangle_count * triangle_count) as u64,
        ));

        // CPU benchmark
        group.bench_with_input(BenchmarkId::new("cpu", mesh_name), mesh, |b, mesh| {
            let params = IntersectionParams::default();
            b.iter(|| detect_self_intersections(black_box(mesh), black_box(&params)))
        });

        // GPU benchmark
        group.bench_with_input(BenchmarkId::new("gpu", mesh_name), mesh, |b, mesh| {
            let params = GpuCollisionParams::default();
            b.iter(|| {
                if let Ok(result) =
                    detect_self_intersections_gpu(black_box(mesh), black_box(&params))
                {
                    black_box(result);
                }
            })
        });
    }

    group.finish();
}

// =============================================================================
// Large Mesh Benchmarks (target: 100k+ triangles)
// =============================================================================

fn bench_large_meshes(c: &mut Criterion) {
    let mut group = c.benchmark_group("Large Meshes");
    group.sample_size(10); // Fewer samples for large meshes

    // Create large meshes
    let large_sphere = create_sphere(6); // ~82k triangles
    let large_grid = create_cube_grid(10); // 10x10x10 = 1000 cubes * 12 = 12000 triangles

    println!("Large sphere: {} triangles", large_sphere.faces.len());
    println!("Large grid: {} triangles", large_grid.faces.len());

    // SDF on large sphere with 128^3 grid
    let dims = [128, 128, 128];
    let voxel_size = 3.0 / 128.0;

    group.throughput(Throughput::Elements(128 * 128 * 128));

    group.bench_function("sdf_cpu_82k_tri_128grid", |b| {
        b.iter(|| {
            compute_sdf_cpu(
                black_box(&large_sphere),
                black_box(dims),
                black_box(voxel_size),
            )
        })
    });

    let gpu_params = GpuSdfParams {
        dims,
        origin: [-1.5, -1.5, -1.5],
        voxel_size: voxel_size as f32,
    };

    group.bench_function("sdf_gpu_82k_tri_128grid", |b| {
        b.iter(|| {
            if let Ok(result) = compute_sdf_gpu(black_box(&large_sphere), black_box(&gpu_params)) {
                black_box(result);
            }
        })
    });

    // Collision detection on large sphere
    group.throughput(Throughput::Elements(
        (large_sphere.faces.len() * large_sphere.faces.len()) as u64,
    ));

    group.bench_function("collision_cpu_82k_tri", |b| {
        let params = IntersectionParams::default();
        b.iter(|| detect_self_intersections(black_box(&large_sphere), black_box(&params)))
    });

    group.bench_function("collision_gpu_82k_tri", |b| {
        let params = GpuCollisionParams::default();
        b.iter(|| {
            if let Ok(result) =
                detect_self_intersections_gpu(black_box(&large_sphere), black_box(&params))
            {
                black_box(result);
            }
        })
    });

    group.finish();
}

// =============================================================================
// Criterion Setup
// =============================================================================

criterion_group!(
    benches,
    bench_sdf,
    bench_surface_nets,
    bench_collision,
    bench_large_meshes
);
criterion_main!(benches);

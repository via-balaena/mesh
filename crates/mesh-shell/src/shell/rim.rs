//! Rim generation for connecting inner and outer shell surfaces.
//!
//! This module handles the generation of rim faces that connect the inner and outer
//! surfaces of a shell at boundary edges. It properly handles:
//! - Multiple boundary loops (e.g., ventilation holes in helmets)
//! - Boundary topology validation
//! - Non-manifold boundary vertex detection

use hashbrown::{HashMap, HashSet};
use tracing::{debug, info, warn};

use mesh_repair::{Mesh, MeshAdjacency};

/// A boundary loop representing a single closed boundary in the mesh.
#[derive(Debug, Clone)]
pub struct BoundaryLoop {
    /// Ordered list of vertex indices forming the loop.
    /// The loop is closed: the last vertex connects back to the first.
    pub vertices: Vec<u32>,
}

impl BoundaryLoop {
    /// Number of edges (and vertices) in the loop.
    pub fn edge_count(&self) -> usize {
        self.vertices.len()
    }

    /// Get edges as (v0, v1) pairs in loop order.
    pub fn edges(&self) -> impl Iterator<Item = (u32, u32)> + '_ {
        let n = self.vertices.len();
        (0..n).map(move |i| (self.vertices[i], self.vertices[(i + 1) % n]))
    }
}

/// Result of boundary topology analysis.
#[derive(Debug, Clone)]
pub struct BoundaryAnalysis {
    /// Closed boundary loops found.
    pub loops: Vec<BoundaryLoop>,
    /// Vertices that appear in more than one boundary loop (non-manifold boundary vertices).
    pub non_manifold_vertices: Vec<u32>,
    /// Boundary edges that couldn't be assigned to any loop (malformed boundaries).
    pub orphan_edges: Vec<(u32, u32)>,
    /// Total number of boundary edges.
    pub total_boundary_edges: usize,
    /// Whether the boundary topology is valid for rim generation.
    pub is_valid: bool,
}

impl BoundaryAnalysis {
    /// Check if there are any issues with the boundary topology.
    pub fn has_issues(&self) -> bool {
        !self.non_manifold_vertices.is_empty() || !self.orphan_edges.is_empty()
    }
}

impl std::fmt::Display for BoundaryAnalysis {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        writeln!(f, "Boundary Analysis:")?;
        writeln!(f, "  Total boundary edges: {}", self.total_boundary_edges)?;
        writeln!(f, "  Closed loops: {}", self.loops.len())?;

        if !self.loops.is_empty() {
            let sizes: Vec<usize> = self.loops.iter().map(|l| l.edge_count()).collect();
            writeln!(f, "  Loop sizes: {:?}", sizes)?;
        }

        if !self.non_manifold_vertices.is_empty() {
            writeln!(
                f,
                "  Non-manifold boundary vertices: {:?}",
                self.non_manifold_vertices
            )?;
        }

        if !self.orphan_edges.is_empty() {
            writeln!(
                f,
                "  Orphan edges: {} (couldn't form loops)",
                self.orphan_edges.len()
            )?;
        }

        writeln!(
            f,
            "  Valid for rim generation: {}",
            if self.is_valid { "yes" } else { "NO" }
        )?;

        Ok(())
    }
}

/// Result of rim generation.
#[derive(Debug, Clone)]
pub struct RimResult {
    /// Generated rim faces.
    pub faces: Vec<[u32; 3]>,
    /// Boundary analysis used for generation.
    pub boundary_analysis: BoundaryAnalysis,
    /// Number of loops that had rims generated.
    pub loops_processed: usize,
    /// Warnings encountered during generation.
    pub warnings: Vec<String>,
}

impl RimResult {
    /// Check if rim generation was successful.
    pub fn is_success(&self) -> bool {
        self.boundary_analysis.is_valid && self.warnings.is_empty()
    }
}

impl std::fmt::Display for RimResult {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        writeln!(f, "Rim Generation Result:")?;
        writeln!(f, "  Faces generated: {}", self.faces.len())?;
        writeln!(f, "  Loops processed: {}", self.loops_processed)?;

        if !self.warnings.is_empty() {
            writeln!(f, "  Warnings:")?;
            for warning in &self.warnings {
                writeln!(f, "    - {}", warning)?;
            }
        }

        Ok(())
    }
}

/// Analyze boundary topology of a mesh.
///
/// Detects all boundary loops and identifies any topology issues such as
/// non-manifold boundary vertices or malformed boundaries.
pub fn analyze_boundary(mesh: &Mesh) -> BoundaryAnalysis {
    let adjacency = MeshAdjacency::build(&mesh.faces);
    analyze_boundary_with_adjacency(&adjacency)
}

/// Analyze boundary topology using pre-built adjacency.
pub fn analyze_boundary_with_adjacency(adjacency: &MeshAdjacency) -> BoundaryAnalysis {
    let boundary_edges: Vec<(u32, u32)> = adjacency.boundary_edges().collect();
    let total_boundary_edges = boundary_edges.len();

    if boundary_edges.is_empty() {
        return BoundaryAnalysis {
            loops: Vec::new(),
            non_manifold_vertices: Vec::new(),
            orphan_edges: Vec::new(),
            total_boundary_edges: 0,
            is_valid: true, // No boundaries is valid (watertight mesh)
        };
    }

    debug!("Analyzing {} boundary edges", boundary_edges.len());

    // Build adjacency for boundary edges (vertex -> neighboring vertices on boundary)
    let mut boundary_neighbors: HashMap<u32, Vec<u32>> = HashMap::new();
    for &(a, b) in &boundary_edges {
        boundary_neighbors.entry(a).or_default().push(b);
        boundary_neighbors.entry(b).or_default().push(a);
    }

    // Check for non-manifold boundary vertices (vertices with != 2 boundary neighbors)
    let mut non_manifold_vertices = Vec::new();
    for (&vertex, neighbors) in &boundary_neighbors {
        if neighbors.len() != 2 {
            non_manifold_vertices.push(vertex);
            debug!(
                "Non-manifold boundary vertex {}: {} neighbors (expected 2)",
                vertex,
                neighbors.len()
            );
        }
    }

    // Trace closed loops
    let mut visited_edges: HashSet<(u32, u32)> = HashSet::new();
    let mut loops = Vec::new();
    let mut orphan_edges = Vec::new();

    for &(start_a, start_b) in &boundary_edges {
        let edge_key = if start_a < start_b {
            (start_a, start_b)
        } else {
            (start_b, start_a)
        };
        if visited_edges.contains(&edge_key) {
            continue;
        }

        // Try to trace a loop starting from this edge
        match trace_boundary_loop(start_a, start_b, &boundary_neighbors, &mut visited_edges) {
            Some(loop_vertices) => {
                if loop_vertices.len() >= 3 {
                    loops.push(BoundaryLoop {
                        vertices: loop_vertices,
                    });
                }
            }
            None => {
                // Couldn't form a closed loop from this edge
                orphan_edges.push((start_a, start_b));
                visited_edges.insert(edge_key);
            }
        }
    }

    // Sort loops by size (largest first) for predictable ordering
    loops.sort_by_key(|b| std::cmp::Reverse(b.edge_count()));

    let is_valid = non_manifold_vertices.is_empty() && orphan_edges.is_empty();

    info!(
        "Boundary analysis: {} loops, {} non-manifold vertices, {} orphan edges",
        loops.len(),
        non_manifold_vertices.len(),
        orphan_edges.len()
    );

    BoundaryAnalysis {
        loops,
        non_manifold_vertices,
        orphan_edges,
        total_boundary_edges,
        is_valid,
    }
}

/// Trace a boundary loop starting from edge (start, next).
/// Returns the ordered vertex list if a closed loop is found, None otherwise.
fn trace_boundary_loop(
    start: u32,
    next: u32,
    boundary_neighbors: &HashMap<u32, Vec<u32>>,
    visited_edges: &mut HashSet<(u32, u32)>,
) -> Option<Vec<u32>> {
    let mut loop_vertices = vec![start];
    let mut current = next;
    let mut prev = start;

    // Mark the starting edge as visited
    let edge_key = if start < next {
        (start, next)
    } else {
        (next, start)
    };
    visited_edges.insert(edge_key);

    loop {
        loop_vertices.push(current);

        // Find next vertex (not the one we came from)
        let neighbors = boundary_neighbors.get(&current)?;

        let next_vertex = neighbors.iter().find(|&&n| n != prev);

        match next_vertex {
            Some(&n) => {
                // Mark this edge as visited
                let edge_key = if current < n {
                    (current, n)
                } else {
                    (n, current)
                };

                if n == start {
                    // Loop closed successfully
                    visited_edges.insert(edge_key);
                    return Some(loop_vertices);
                }

                if visited_edges.contains(&edge_key) {
                    // Hit an already-visited edge without closing the loop
                    // This indicates a malformed boundary
                    return None;
                }

                visited_edges.insert(edge_key);
                prev = current;
                current = n;
            }
            None => {
                // Dead end - malformed boundary
                return None;
            }
        }

        // Safety check: prevent infinite loops
        if loop_vertices.len() > boundary_neighbors.len() + 1 {
            warn!("Boundary loop tracing exceeded expected length, aborting");
            return None;
        }
    }
}

/// Generate rim faces to connect inner and outer surfaces at boundaries.
///
/// This is the simple API that generates rims for all boundary loops.
///
/// # Arguments
/// * `inner_mesh` - The inner surface mesh
/// * `vertex_offset` - Offset to add to vertex indices for the outer surface
///   (typically equal to the number of vertices in the inner mesh)
///
/// # Returns
/// A tuple of (rim faces, total boundary edge count).
pub fn generate_rim(inner_mesh: &Mesh, vertex_offset: usize) -> (Vec<[u32; 3]>, usize) {
    let result = generate_rim_advanced(inner_mesh, vertex_offset);
    (result.faces, result.boundary_analysis.total_boundary_edges)
}

/// Generate rim faces with full boundary analysis and reporting.
///
/// This function:
/// 1. Analyzes boundary topology to find all loops
/// 2. Validates the boundary (checks for non-manifold vertices)
/// 3. Generates rim faces for each valid loop
/// 4. Reports any issues encountered
///
/// # Arguments
/// * `inner_mesh` - The inner surface mesh
/// * `vertex_offset` - Offset to add to vertex indices for the outer surface
///
/// # Returns
/// A `RimResult` with generated faces and analysis details.
pub fn generate_rim_advanced(inner_mesh: &Mesh, vertex_offset: usize) -> RimResult {
    let analysis = analyze_boundary(inner_mesh);

    if analysis.total_boundary_edges == 0 {
        return RimResult {
            faces: Vec::new(),
            boundary_analysis: analysis,
            loops_processed: 0,
            warnings: Vec::new(),
        };
    }

    let mut faces = Vec::new();
    let mut warnings = Vec::new();
    let n = vertex_offset as u32;

    // Warn about topology issues but continue processing valid loops
    if !analysis.non_manifold_vertices.is_empty() {
        warnings.push(format!(
            "Found {} non-manifold boundary vertices: {:?}",
            analysis.non_manifold_vertices.len(),
            analysis.non_manifold_vertices
        ));
    }

    if !analysis.orphan_edges.is_empty() {
        warnings.push(format!(
            "Found {} orphan boundary edges that couldn't form closed loops",
            analysis.orphan_edges.len()
        ));
    }

    // Generate rim faces for each loop
    for (loop_idx, boundary_loop) in analysis.loops.iter().enumerate() {
        debug!(
            "Generating rim for loop {} ({} edges)",
            loop_idx,
            boundary_loop.edge_count()
        );

        for (v0, v1) in boundary_loop.edges() {
            // Create two triangles to connect inner edge to outer edge
            // Inner edge: (v0, v1)
            // Outer edge: (v0+n, v1+n)
            //
            // Rim triangles (normals pointing outward):
            // Triangle 1: v1 -> v0 -> v0+n
            // Triangle 2: v1 -> v0+n -> v1+n
            faces.push([v1, v0, v0 + n]);
            faces.push([v1, v0 + n, v1 + n]);
        }
    }

    info!(
        "Generated {} rim faces for {} boundary loops",
        faces.len(),
        analysis.loops.len()
    );

    RimResult {
        faces,
        loops_processed: analysis.loops.len(),
        boundary_analysis: analysis,
        warnings,
    }
}

/// Validate that a mesh's boundaries are suitable for rim generation.
///
/// Returns Ok(analysis) if valid, Err with description if not.
pub fn validate_boundary_for_rim(mesh: &Mesh) -> Result<BoundaryAnalysis, String> {
    let analysis = analyze_boundary(mesh);

    if !analysis.is_valid {
        let mut issues = Vec::new();

        if !analysis.non_manifold_vertices.is_empty() {
            issues.push(format!(
                "{} non-manifold boundary vertices",
                analysis.non_manifold_vertices.len()
            ));
        }

        if !analysis.orphan_edges.is_empty() {
            issues.push(format!(
                "{} orphan boundary edges",
                analysis.orphan_edges.len()
            ));
        }

        return Err(format!("Invalid boundary topology: {}", issues.join(", ")));
    }

    Ok(analysis)
}

/// Generate rim faces for SDF-based shell where inner and outer surfaces
/// have different vertex counts and no 1:1 correspondence.
///
/// This function finds the closest outer boundary vertex for each inner
/// boundary vertex using spatial matching, then generates triangles to
/// connect the boundaries.
///
/// # Arguments
/// * `inner_mesh` - The inner surface mesh
/// * `outer_mesh` - The outer surface mesh (from SDF extraction)
/// * `outer_vertex_offset` - Offset to add to outer vertex indices
///
/// # Returns
/// A tuple of (rim faces, total boundary edge count).
pub fn generate_rim_for_sdf_shell(
    inner_mesh: &Mesh,
    outer_mesh: &Mesh,
    outer_vertex_offset: usize,
) -> (Vec<[u32; 3]>, usize) {
    use kiddo::KdTree;

    // Analyze boundaries of both meshes
    let inner_analysis = analyze_boundary(inner_mesh);
    let outer_analysis = analyze_boundary(outer_mesh);

    if inner_analysis.total_boundary_edges == 0 {
        return (Vec::new(), 0);
    }

    // If outer has no boundary (closed mesh from SDF), we can't connect
    // This shouldn't happen for open shells, but handle gracefully
    if outer_analysis.total_boundary_edges == 0 {
        warn!("Outer mesh has no boundary edges, cannot generate rim");
        return (Vec::new(), inner_analysis.total_boundary_edges);
    }

    debug!(
        "Generating SDF rim: inner loops={}, outer loops={}",
        inner_analysis.loops.len(),
        outer_analysis.loops.len()
    );

    // Build KD-tree of outer boundary vertices for fast nearest-neighbor lookup
    let mut outer_boundary_vertices: HashSet<u32> = HashSet::new();
    for outer_loop in &outer_analysis.loops {
        for &v in &outer_loop.vertices {
            outer_boundary_vertices.insert(v);
        }
    }

    let mut kdtree: KdTree<f64, 3> = KdTree::new();
    for &v in &outer_boundary_vertices {
        let pos = &outer_mesh.vertices[v as usize].position;
        kdtree.add(&[pos.x, pos.y, pos.z], v as u64);
    }

    let mut faces = Vec::new();
    let offset = outer_vertex_offset as u32;

    // For each inner boundary loop, generate rim connecting to outer boundary
    for inner_loop in &inner_analysis.loops {
        let vertices = &inner_loop.vertices;
        let n = vertices.len();

        if n < 3 {
            continue;
        }

        // For each edge in the inner boundary, find corresponding outer vertices
        // and create triangles to connect them
        for i in 0..n {
            let v0 = vertices[i];
            let v1 = vertices[(i + 1) % n];

            // Find closest outer boundary vertex for each inner vertex
            let pos0 = &inner_mesh.vertices[v0 as usize].position;
            let pos1 = &inner_mesh.vertices[v1 as usize].position;

            let nearest0 = kdtree.nearest_one::<kiddo::SquaredEuclidean>(&[pos0.x, pos0.y, pos0.z]);
            let nearest1 = kdtree.nearest_one::<kiddo::SquaredEuclidean>(&[pos1.x, pos1.y, pos1.z]);

            let outer_v0 = nearest0.item as u32 + offset;
            let outer_v1 = nearest1.item as u32 + offset;

            // Create triangles connecting inner edge to outer vertices
            // Triangle winding is set so normals point outward from the rim
            if outer_v0 != outer_v1 {
                // Quad case: inner edge connects to two different outer vertices
                // Create two triangles forming a quad
                faces.push([v1, v0, outer_v0]);
                faces.push([v1, outer_v0, outer_v1]);
            } else {
                // Degenerate case: both inner vertices map to same outer vertex
                // Create a single triangle
                faces.push([v1, v0, outer_v0]);
            }
        }
    }

    info!(
        "Generated {} SDF rim faces for {} inner boundary loops",
        faces.len(),
        inner_analysis.loops.len()
    );

    (faces, inner_analysis.total_boundary_edges)
}

#[cfg(test)]
mod tests {
    use super::*;
    use mesh_repair::Vertex;

    fn create_open_square() -> Mesh {
        // A simple square (4 vertices, 2 triangles, 4 boundary edges)
        let mut mesh = Mesh::new();

        mesh.vertices.push(Vertex::from_coords(0.0, 0.0, 0.0));
        mesh.vertices.push(Vertex::from_coords(10.0, 0.0, 0.0));
        mesh.vertices.push(Vertex::from_coords(10.0, 10.0, 0.0));
        mesh.vertices.push(Vertex::from_coords(0.0, 10.0, 0.0));

        mesh.faces.push([0, 1, 2]);
        mesh.faces.push([0, 2, 3]);

        mesh
    }

    fn create_mesh_with_two_holes() -> Mesh {
        // A mesh with two separate boundary loops
        // Imagine a rectangular plate with two circular holes cut out
        // Simplified as two separate triangulated regions

        let mut mesh = Mesh::new();

        // Outer boundary (square)
        mesh.vertices.push(Vertex::from_coords(0.0, 0.0, 0.0)); // 0
        mesh.vertices.push(Vertex::from_coords(20.0, 0.0, 0.0)); // 1
        mesh.vertices.push(Vertex::from_coords(20.0, 10.0, 0.0)); // 2
        mesh.vertices.push(Vertex::from_coords(0.0, 10.0, 0.0)); // 3

        // First hole (triangle shape at left)
        mesh.vertices.push(Vertex::from_coords(3.0, 3.0, 0.0)); // 4
        mesh.vertices.push(Vertex::from_coords(5.0, 3.0, 0.0)); // 5
        mesh.vertices.push(Vertex::from_coords(4.0, 6.0, 0.0)); // 6

        // Second hole (triangle shape at right)
        mesh.vertices.push(Vertex::from_coords(15.0, 3.0, 0.0)); // 7
        mesh.vertices.push(Vertex::from_coords(17.0, 3.0, 0.0)); // 8
        mesh.vertices.push(Vertex::from_coords(16.0, 6.0, 0.0)); // 9

        // Triangulate the outer boundary (simplified - just corners)
        mesh.faces.push([0, 1, 2]);
        mesh.faces.push([0, 2, 3]);

        // Add faces around first hole (creates a triangular boundary)
        // These faces connect the hole boundary to nearby vertices
        mesh.faces.push([4, 6, 5]); // Inner triangle for first hole

        // Add faces around second hole
        mesh.faces.push([7, 9, 8]); // Inner triangle for second hole

        mesh
    }

    fn create_mesh_with_single_hole() -> Mesh {
        // Open box missing top face
        let mut mesh = Mesh::new();

        // Bottom corners
        mesh.vertices.push(Vertex::from_coords(0.0, 0.0, 0.0));
        mesh.vertices.push(Vertex::from_coords(10.0, 0.0, 0.0));
        mesh.vertices.push(Vertex::from_coords(10.0, 10.0, 0.0));
        mesh.vertices.push(Vertex::from_coords(0.0, 10.0, 0.0));
        // Top corners
        mesh.vertices.push(Vertex::from_coords(0.0, 0.0, 10.0));
        mesh.vertices.push(Vertex::from_coords(10.0, 0.0, 10.0));
        mesh.vertices.push(Vertex::from_coords(10.0, 10.0, 10.0));
        mesh.vertices.push(Vertex::from_coords(0.0, 10.0, 10.0));

        // Bottom (2 triangles)
        mesh.faces.push([0, 2, 1]);
        mesh.faces.push([0, 3, 2]);
        // Front
        mesh.faces.push([0, 1, 5]);
        mesh.faces.push([0, 5, 4]);
        // Back
        mesh.faces.push([2, 3, 7]);
        mesh.faces.push([2, 7, 6]);
        // Left
        mesh.faces.push([0, 4, 7]);
        mesh.faces.push([0, 7, 3]);
        // Right
        mesh.faces.push([1, 2, 6]);
        mesh.faces.push([1, 6, 5]);
        // Top is OPEN - creates a 4-edge boundary loop

        mesh
    }

    #[test]
    fn test_generate_rim() {
        let mesh = create_open_square();
        let (rim_faces, boundary_count) = generate_rim(&mesh, 4);

        // 4 boundary edges * 2 triangles per edge = 8 faces
        assert_eq!(boundary_count, 4);
        assert_eq!(rim_faces.len(), 8);
    }

    #[test]
    fn test_analyze_boundary_single_loop() {
        let mesh = create_open_square();
        let analysis = analyze_boundary(&mesh);

        assert_eq!(analysis.loops.len(), 1);
        assert_eq!(analysis.loops[0].edge_count(), 4);
        assert!(analysis.non_manifold_vertices.is_empty());
        assert!(analysis.orphan_edges.is_empty());
        assert!(analysis.is_valid);
    }

    #[test]
    fn test_analyze_boundary_open_box() {
        let mesh = create_mesh_with_single_hole();
        let analysis = analyze_boundary(&mesh);

        assert_eq!(analysis.loops.len(), 1);
        assert_eq!(analysis.loops[0].edge_count(), 4); // Square hole at top
        assert!(analysis.is_valid);
    }

    #[test]
    fn test_analyze_boundary_multiple_holes() {
        let mesh = create_mesh_with_two_holes();
        let analysis = analyze_boundary(&mesh);

        // Should find multiple boundary loops
        assert!(
            analysis.loops.len() >= 2,
            "Expected at least 2 loops, got {}",
            analysis.loops.len()
        );
    }

    #[test]
    fn test_generate_rim_advanced() {
        let mesh = create_open_square();
        let result = generate_rim_advanced(&mesh, 4);

        assert_eq!(result.faces.len(), 8);
        assert_eq!(result.loops_processed, 1);
        assert!(result.warnings.is_empty());
        assert!(result.is_success());
    }

    #[test]
    fn test_watertight_mesh_no_boundary() {
        // Tetrahedron (watertight)
        let mut mesh = Mesh::new();
        mesh.vertices.push(Vertex::from_coords(0.0, 0.0, 0.0));
        mesh.vertices.push(Vertex::from_coords(1.0, 0.0, 0.0));
        mesh.vertices.push(Vertex::from_coords(0.5, 1.0, 0.0));
        mesh.vertices.push(Vertex::from_coords(0.5, 0.5, 1.0));

        mesh.faces.push([0, 2, 1]);
        mesh.faces.push([0, 1, 3]);
        mesh.faces.push([1, 2, 3]);
        mesh.faces.push([2, 0, 3]);

        let analysis = analyze_boundary(&mesh);

        assert_eq!(analysis.loops.len(), 0);
        assert_eq!(analysis.total_boundary_edges, 0);
        assert!(analysis.is_valid);
    }

    #[test]
    fn test_validate_boundary_for_rim_valid() {
        let mesh = create_open_square();
        let result = validate_boundary_for_rim(&mesh);

        assert!(result.is_ok());
        let analysis = result.unwrap();
        assert_eq!(analysis.loops.len(), 1);
    }

    #[test]
    fn test_boundary_loop_edges_iterator() {
        let boundary = BoundaryLoop {
            vertices: vec![0, 1, 2, 3],
        };

        let edges: Vec<_> = boundary.edges().collect();
        assert_eq!(edges.len(), 4);
        assert_eq!(edges[0], (0, 1));
        assert_eq!(edges[1], (1, 2));
        assert_eq!(edges[2], (2, 3));
        assert_eq!(edges[3], (3, 0)); // Closes back to start
    }

    #[test]
    fn test_boundary_analysis_display() {
        let mesh = create_open_square();
        let analysis = analyze_boundary(&mesh);
        let output = format!("{}", analysis);

        assert!(output.contains("Boundary Analysis"));
        assert!(output.contains("Total boundary edges: 4"));
        assert!(output.contains("Closed loops: 1"));
        assert!(output.contains("Valid for rim generation: yes"));
    }

    #[test]
    fn test_rim_result_display() {
        let mesh = create_open_square();
        let result = generate_rim_advanced(&mesh, 4);
        let output = format!("{}", result);

        assert!(output.contains("Rim Generation Result"));
        assert!(output.contains("Faces generated: 8"));
        assert!(output.contains("Loops processed: 1"));
    }
}

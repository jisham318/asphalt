//! Converts 3D model files into Roblox's `.mesh` format (version 2.00), so they
//! can be referenced from Studio's content folder without being uploaded.
//!
//! Every mesh in the file is flattened into a single mesh with its node
//! transforms applied. Units are converted so that one meter is one stud (OBJ
//! files have no units, so their coordinates are used as-is).

use anyhow::{Context, anyhow, bail};
use binrw::BinWrite;
use log::warn;
use rbx_mesh::mesh::{Face2, Mesh2, Revision2, Vertex2, VertexId2, Vertices2};
use std::{collections::HashMap, io::Cursor};

#[derive(Debug, Clone, Copy)]
pub enum MeshSource {
    Fbx,
    Obj,
    Gltf,
}

pub fn convert_to_mesh(data: &[u8], source: MeshSource) -> anyhow::Result<Vec<u8>> {
    let builder = match source {
        MeshSource::Fbx => from_ufbx(data, ufbx::FileFormat::Fbx)?,
        MeshSource::Obj => from_ufbx(data, ufbx::FileFormat::Obj)?,
        MeshSource::Gltf => from_gltf(data)?,
    };

    builder.finish()
}

// The tangent Roblox writes for meshes without tangent data
const DEFAULT_TANGENT: [i8; 4] = [0, 0, -128, 127];

#[derive(Default)]
struct MeshBuilder {
    vertices: Vec<Vertex2>,
    faces: Vec<Face2>,
    vertex_ids: HashMap<[u32; 8], u32>,
}

impl MeshBuilder {
    /// Adds a vertex, reusing an identical one if it already exists
    fn vertex(&mut self, pos: [f32; 3], norm: [f32; 3], tex: [f32; 2]) -> anyhow::Result<u32> {
        let key = [
            pos[0], pos[1], pos[2], norm[0], norm[1], norm[2], tex[0], tex[1],
        ]
        .map(f32::to_bits);

        if let Some(&id) = self.vertex_ids.get(&key) {
            return Ok(id);
        }

        let id = u32::try_from(self.vertices.len()).context("Mesh has too many vertices")?;
        self.vertices.push(Vertex2 {
            pos,
            norm,
            tex,
            tangent: DEFAULT_TANGENT,
            color: [255; 4],
        });
        self.vertex_ids.insert(key, id);

        Ok(id)
    }

    fn triangle(&mut self, [a, b, c]: [u32; 3]) {
        self.faces
            .push(Face2([VertexId2(a), VertexId2(b), VertexId2(c)]));
    }

    fn finish(self) -> anyhow::Result<Vec<u8>> {
        if self.faces.is_empty() {
            bail!("Model doesn't contain any triangles");
        }

        let mesh = Mesh2 {
            revision: Revision2::Version200,
            vertices: Vertices2::Full(self.vertices),
            faces: self.faces,
        };

        let mut writer = Cursor::new(Vec::new());
        mesh.write_le(&mut writer).context("Failed to write mesh")?;

        Ok(writer.into_inner())
    }
}

fn from_ufbx(data: &[u8], file_format: ufbx::FileFormat) -> anyhow::Result<MeshBuilder> {
    let opts = ufbx::LoadOpts {
        file_format,
        ignore_animation: true,
        ignore_embedded: true,
        generate_missing_normals: true,
        space_conversion: ufbx::SpaceConversion::ModifyGeometry,
        target_axes: ufbx::CoordinateAxes::right_handed_y_up(),
        target_unit_meters: 1.0,
        obj_axes: ufbx::CoordinateAxes::right_handed_y_up(),
        obj_unit_meters: 1.0,
        ..Default::default()
    };

    let scene = ufbx::load_memory(data, opts)
        .map_err(|e| anyhow!("Failed to load model: {}", e.description.as_ref()))?;

    let mut builder = MeshBuilder::default();
    let mut indices = Vec::new();

    for node in &scene.nodes {
        let Some(mesh) = &node.mesh else { continue };

        let matrix = &node.geometry_to_world;
        let normal_matrix = ufbx::matrix_for_normals(matrix);
        let mirrored = determinant(matrix) < 0.0;

        for &face in &mesh.faces {
            ufbx::triangulate_face_vec(&mut indices, mesh, face);

            for tri in indices.chunks_exact(3) {
                let mut ids = [0; 3];
                for (id, &index) in ids.iter_mut().zip(tri) {
                    let index = index as usize;

                    let pos = ufbx::transform_position(matrix, mesh.vertex_position[index]);
                    let norm = if mesh.vertex_normal.exists {
                        let norm =
                            ufbx::transform_direction(&normal_matrix, mesh.vertex_normal[index]);
                        normalize([norm.x as f32, norm.y as f32, norm.z as f32])
                    } else {
                        [0.0; 3]
                    };
                    // Roblox meshes put the UV origin at the top left
                    let tex = if mesh.vertex_uv.exists {
                        let uv = mesh.vertex_uv[index];
                        [uv.x as f32, 1.0 - uv.y as f32]
                    } else {
                        [0.0; 2]
                    };

                    *id = builder.vertex([pos.x as f32, pos.y as f32, pos.z as f32], norm, tex)?;
                }

                if mirrored {
                    ids.swap(1, 2);
                }
                builder.triangle(ids);
            }
        }
    }

    Ok(builder)
}

fn determinant(m: &ufbx::Matrix) -> f64 {
    m.m00 * (m.m11 * m.m22 - m.m12 * m.m21) - m.m01 * (m.m10 * m.m22 - m.m12 * m.m20)
        + m.m02 * (m.m10 * m.m21 - m.m11 * m.m20)
}

type Mat4 = [[f32; 4]; 4];

const IDENTITY: Mat4 = [
    [1.0, 0.0, 0.0, 0.0],
    [0.0, 1.0, 0.0, 0.0],
    [0.0, 0.0, 1.0, 0.0],
    [0.0, 0.0, 0.0, 1.0],
];

fn from_gltf(data: &[u8]) -> anyhow::Result<MeshBuilder> {
    let gltf::Gltf { document, blob } = gltf::Gltf::from_slice(data)?;
    let buffers = gltf::import_buffers(&document, None, blob)
        .context("Failed to load buffers (external .bin files aren't supported, use .glb)")?;

    let scene = document
        .default_scene()
        .or_else(|| document.scenes().next())
        .context("Model doesn't contain any scenes")?;

    let mut builder = MeshBuilder::default();
    for node in scene.nodes() {
        add_gltf_node(&mut builder, &node, &IDENTITY, &buffers)?;
    }

    Ok(builder)
}

fn add_gltf_node(
    builder: &mut MeshBuilder,
    node: &gltf::Node,
    parent: &Mat4,
    buffers: &[gltf::buffer::Data],
) -> anyhow::Result<()> {
    let matrix = mat4_mul(parent, &node.transform().matrix());

    if let Some(mesh) = node.mesh() {
        // Normals are transformed by the cofactor matrix, which is the inverse
        // transpose scaled by the determinant
        let [x, y, z] = [0, 1, 2].map(|i| [matrix[i][0], matrix[i][1], matrix[i][2]]);
        let determinant = dot(x, cross(y, z));
        let cofactor = [cross(y, z), cross(z, x), cross(x, y)];
        let mirrored = determinant < 0.0;

        for primitive in mesh.primitives() {
            if primitive.mode() != gltf::mesh::Mode::Triangles {
                warn!("Skipping a primitive that isn't made of triangles");
                continue;
            }

            let reader = primitive.reader(|buffer| Some(&buffers[buffer.index()]));

            let positions: Vec<[f32; 3]> = reader
                .read_positions()
                .context("Primitive has no positions")?
                .map(|pos| transform_point(&matrix, pos))
                .collect();
            let normals: Option<Vec<[f32; 3]>> = reader.read_normals().map(|normals| {
                normals
                    .map(|norm| {
                        let norm = add(
                            add(scale(cofactor[0], norm[0]), scale(cofactor[1], norm[1])),
                            scale(cofactor[2], norm[2]),
                        );
                        normalize(if mirrored { scale(norm, -1.0) } else { norm })
                    })
                    .collect()
            });
            let tex_coords: Option<Vec<[f32; 2]>> = reader
                .read_tex_coords(0)
                .map(|tex_coords| tex_coords.into_f32().collect());
            let indices: Vec<u32> = match reader.read_indices() {
                Some(indices) => indices.into_u32().collect(),
                None => (0..u32::try_from(positions.len())?).collect(),
            };

            for tri in indices.chunks_exact(3) {
                let tri = [tri[0], tri[1], tri[2]].map(|i| i as usize);
                if tri.iter().any(|&i| i >= positions.len()) {
                    bail!("Primitive has an out of bounds index");
                }

                let flat_normal = normalize(cross(
                    sub(positions[tri[1]], positions[tri[0]]),
                    sub(positions[tri[2]], positions[tri[0]]),
                ));

                let mut ids = [0; 3];
                for (id, i) in ids.iter_mut().zip(tri) {
                    let norm = normals.as_ref().map_or(flat_normal, |n| n[i]);
                    let tex = tex_coords.as_ref().map_or([0.0; 2], |t| t[i]);
                    *id = builder.vertex(positions[i], norm, tex)?;
                }

                if mirrored {
                    ids.swap(1, 2);
                }
                builder.triangle(ids);
            }
        }
    }

    for child in node.children() {
        add_gltf_node(builder, &child, &matrix, buffers)?;
    }

    Ok(())
}

/// Multiplies two column-major matrices
fn mat4_mul(a: &Mat4, b: &Mat4) -> Mat4 {
    let mut out = [[0.0; 4]; 4];
    for (col, out_col) in out.iter_mut().enumerate() {
        for (row, value) in out_col.iter_mut().enumerate() {
            *value = (0..4).map(|k| a[k][row] * b[col][k]).sum();
        }
    }
    out
}

fn transform_point(m: &Mat4, [x, y, z]: [f32; 3]) -> [f32; 3] {
    [0, 1, 2].map(|row| m[0][row] * x + m[1][row] * y + m[2][row] * z + m[3][row])
}

fn add(a: [f32; 3], b: [f32; 3]) -> [f32; 3] {
    [a[0] + b[0], a[1] + b[1], a[2] + b[2]]
}

fn sub(a: [f32; 3], b: [f32; 3]) -> [f32; 3] {
    [a[0] - b[0], a[1] - b[1], a[2] - b[2]]
}

fn scale(a: [f32; 3], s: f32) -> [f32; 3] {
    a.map(|v| v * s)
}

fn dot(a: [f32; 3], b: [f32; 3]) -> f32 {
    a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
}

fn cross(a: [f32; 3], b: [f32; 3]) -> [f32; 3] {
    [
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    ]
}

fn normalize(a: [f32; 3]) -> [f32; 3] {
    let len = dot(a, a).sqrt();
    if len > 0.0 { scale(a, 1.0 / len) } else { a }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rbx_mesh::mesh::Mesh;

    const QUAD_OBJ: &str = "\
v 0 0 0
v 1 0 0
v 1 1 0
v 0 1 0
vt 0 0
vt 1 0
vt 1 1
vt 0 1
vn 0 0 1
f 1/1/1 2/2/1 3/3/1 4/4/1
";

    fn read_back(data: &[u8]) -> Mesh2 {
        match rbx_mesh::read_mesh_versioned(Cursor::new(data)).unwrap() {
            Mesh::V2(mesh) => mesh,
            other => panic!("expected a version 2.00 mesh, got {other:?}"),
        }
    }

    fn full_vertices(mesh: &Mesh2) -> &[Vertex2] {
        match &mesh.vertices {
            Vertices2::Full(vertices) => vertices,
            Vertices2::Truncated(_) => panic!("expected full vertices"),
        }
    }

    #[test]
    fn writes_v2_header() {
        let data = convert_to_mesh(QUAD_OBJ.as_bytes(), MeshSource::Obj).unwrap();

        assert!(data.starts_with(b"version 2.00\n"));
        // header size, vertex size, face size, vertex count, face count
        assert_eq!(&data[13..16], &[12, 0, 40]);
        assert_eq!(data[16], 12);
        assert_eq!(&data[17..25], &[4, 0, 0, 0, 2, 0, 0, 0]);
        assert_eq!(data.len(), 25 + 4 * 40 + 2 * 12);
    }

    #[test]
    fn converts_obj_quad() {
        let data = convert_to_mesh(QUAD_OBJ.as_bytes(), MeshSource::Obj).unwrap();
        let mesh = read_back(&data);
        let vertices = full_vertices(&mesh);

        assert_eq!(vertices.len(), 4);
        assert_eq!(mesh.faces.len(), 2);

        for vertex in vertices {
            assert_eq!(vertex.norm, [0.0, 0.0, 1.0]);
            // V is flipped
            assert_eq!(vertex.tex[1], 1.0 - vertex.pos[1]);
        }
    }

    #[test]
    fn converts_gltf_triangle() {
        // A single triangle with no normals, translated along X
        let positions: [[f32; 3]; 3] = [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]];
        let bin: Vec<u8> = positions
            .iter()
            .flatten()
            .flat_map(|v| v.to_le_bytes())
            .collect();
        let json = format!(
            r#"{{
                "asset": {{ "version": "2.0" }},
                "scene": 0,
                "scenes": [{{ "nodes": [0] }}],
                "nodes": [{{ "mesh": 0, "translation": [5, 0, 0] }}],
                "meshes": [{{ "primitives": [{{ "attributes": {{ "POSITION": 0 }} }}] }}],
                "accessors": [{{ "bufferView": 0, "componentType": 5126, "count": 3, "type": "VEC3", "min": [0, 0, 0], "max": [1, 1, 0] }}],
                "bufferViews": [{{ "buffer": 0, "byteLength": 36 }}],
                "buffers": [{{ "byteLength": 36, "uri": "data:application/octet-stream;base64,{}" }}]
            }}"#,
            base64_encode(&bin)
        );

        let data = convert_to_mesh(json.as_bytes(), MeshSource::Gltf).unwrap();
        let mesh = read_back(&data);
        let vertices = full_vertices(&mesh);

        assert_eq!(mesh.faces.len(), 1);
        assert_eq!(vertices[0].pos, [5.0, 0.0, 0.0]);
        assert_eq!(vertices[1].pos, [6.0, 0.0, 0.0]);
        assert!(vertices.iter().all(|v| v.norm == [0.0, 0.0, 1.0]));
    }

    #[test]
    fn rejects_empty_model() {
        assert!(convert_to_mesh(b"v 0 0 0\n", MeshSource::Obj).is_err());
    }

    fn base64_encode(data: &[u8]) -> String {
        const ALPHABET: &[u8] = b"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
        let mut out = String::new();
        for chunk in data.chunks(3) {
            let n = chunk
                .iter()
                .enumerate()
                .fold(0u32, |n, (i, &b)| n | (b as u32) << (16 - 8 * i));
            for i in 0..4 {
                if i <= chunk.len() {
                    out.push(ALPHABET[(n >> (18 - 6 * i) & 63) as usize] as char);
                } else {
                    out.push('=');
                }
            }
        }
        out
    }
}

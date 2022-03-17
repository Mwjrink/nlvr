use super::{mesh::Mesh, texture::*};
// use ash::vk;
use cgmath::Matrix4;
// use cmreader::reader;

pub struct Renderable {
    // pub texture_index: Option<u32>,
    // pub texture: Option<Texture>,
    //
    // pub vertex_buffer_ptr: BuffPtr,
    // pub vertex_count: usize,
    // //
    // pub index_buffer_ptr: BuffPtr,
    // pub index_count: usize,
    // TODO TEMP
    // pub mesh: Mesh,
    pub meshes: Vec<Mesh>,
    // TODO: TEMP
    // pub materials: Vec<Material>,
    pub texture: Option<Texture>,
    // TODO TEMP
    pub texture_index: Option<u32>,
    //
    pub asset_path: String,
    //
    pub instances: Vec<Matrix4<f32>>,
}

impl Renderable {
    pub fn update(&mut self, instance_index: usize, transform: Matrix4<f32>) {
        self.instances[instance_index] = transform;
    }

    pub fn create_instance(&mut self, transform: Matrix4<f32>) -> usize {
        let index = self.instances.len();
        self.instances.push(transform);
        index
    }
}

impl Renderable {
    //     fn from_file<T: UBO + Copy>(render_instance: &mut RenderInstance<T>, path: String) -> Self {
    //         let ctree = reader::read(&path);
    //
    //         // ctree.load_all();
    //
    //         let roots = ctree.roots;
    //
    //         let root_cluster_0 = ctree.get_cluster(&roots[0]);
    //
    //         let root_mesh_0 = Mesh {
    //
    //         };
    //
    //         let root_cluster_1 = ctree.get_cluster(&roots[0]);
    //         let root_mesh_1 = Mesh {
    //
    //         };
    //
    //         let texture = Texture {
    //
    //         };
    //
    //         upload_vertices
    //
    //         upload_indices
    //
    //         Self {
    //             vec![root_mesh_0, root_mesh_1],
    //             texture,
    //             asset_path: path,
    //             instances: vec![Matrix4::identity()],
    //         }
    //     }
}

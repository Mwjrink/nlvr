use std::{convert::TryInto, fs, fs::File, io::Read, mem};

// TODO: make these not pub, have access methods
pub struct Model {
    pub albedo: Texture,
    // pub gloss:    Texture,
    // pub specular: Texture,
    pub x:      Vec<f32,>,
    pub y:      Vec<f32,>,
    pub z:      Vec<f32,>,
    pub u:      Vec<f32,>,
    pub v:      Vec<f32,>,
}

pub struct Texture {
    pub raw_pixels: Vec<u8,>,
    pub width:      u32,
    pub height:     u32,
}

fn to_x4<T,>(array: &[T],) -> &[T; 4] {
    array
        .try_into()
        .expect("to_x4 has found a chunk with an incorrect length",)
}

fn load_texture(file_name: &str,) -> Texture {
    let mut file = File::open(file_name,).unwrap();

    let mut size_buf = vec![0u8; 8];
    file.read_exact(&mut size_buf,).unwrap();

    let width = u32::from_le_bytes(size_buf[0..4].try_into().unwrap(),);
    let height = u32::from_le_bytes(size_buf[4..8].try_into().unwrap(),);

    let mut raw_pixels = Vec::with_capacity(width as usize * height as usize * mem::size_of::<u32,>(),);
    file.read_to_end(&mut raw_pixels,).unwrap();

    Texture {
        raw_pixels,
        width,
        height,
    }
}

fn load_mesh_texture(file_name: &str,) -> Option<Vec<f32,>,> {
    let temp_vec: Vec<u8,> = fs::read(file_name,).unwrap_or(vec![],);
    Some(temp_vec.chunks(4,).map(|i| f32::from_le_bytes(*to_x4(i,),),).collect(),)
}

impl Model {
    pub fn load(file_name: &str,) -> Self {
        // necessary
        let albedo = load_texture(format!("{}.nlvoa", file_name).as_str(),);

        let x: Vec<f32,> = load_mesh_texture(format!("{}.nlvox", file_name).as_str(),).unwrap();
        let y: Vec<f32,> = load_mesh_texture(format!("{}.nlvoy", file_name).as_str(),).unwrap();
        let z: Vec<f32,> = load_mesh_texture(format!("{}.nlvoz", file_name).as_str(),).unwrap();

        let u: Vec<f32,> = load_mesh_texture(format!("{}.nlvou", file_name).as_str(),).unwrap();
        let v: Vec<f32,> = load_mesh_texture(format!("{}.nlvov", file_name).as_str(),).unwrap();

        // unecessary, load_texture?
        // let gloss = load_mesh_texture(format!("{}.nlvog", file_name).as_str(),).unwrap_or_else(|| vec![],);
        // let specular = load_mesh_texture(format!("{}.nlvos", file_name).as_str(),).unwrap_or_else(|| vec![],);

        Self {
            albedo,
            // gloss,
            // specular,
            x,
            y,
            z,
            //
            u,
            v,
        }
        //
    }
}

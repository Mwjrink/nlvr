use cgmath;

pub type Transform = cgmath::Decomposed<cgmath::Vector3<f32,>, cgmath::Quaternion,>;

impl Component for Transform {
    //
}

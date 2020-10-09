use super::syncobjects::*;
use ash::Device;

pub struct InFlightFrames {
    pub sync_objects:  Vec<SyncObjects,>,
    pub current_frame: usize,
}

impl InFlightFrames {
    pub fn new(sync_objects: Vec<SyncObjects,>,) -> Self {
        Self {
            sync_objects,
            current_frame: 0,
        }
    }

    pub fn destroy(&self, device: &Device,) {
        self.sync_objects.iter().for_each(|o| o.destroy(&device,),);
    }
}

impl Iterator for InFlightFrames {
    type Item = SyncObjects;

    fn next(&mut self,) -> Option<Self::Item,> {
        let next = self.sync_objects[self.current_frame];

        self.current_frame = (self.current_frame + 1) % self.sync_objects.len();

        Some(next,)
    }
}

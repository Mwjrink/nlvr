use std::sync::atomic::{AtomicUsize, Ordering};

struct Wrap<T,> {
    element: T,
    key:     usize,
}

pub struct SwapVec<T,> {
    active_count: usize,
    underlying:   Vec<Wrap<T,>,>, // [0..active_count..] elements from 0 to active_count are active
    map:          HashMap<usize, usize,>,
    current:      AtomicUsize,
}

impl<T,> SwapVec<T,> {
    pub fn push(&self, element: &T,) {
        self.underlying.push(Wrap {
            element,
            // probably make this release
            key: self.current.fetch_add(1, Ordering::Relaxed,),
        },);
    }

    pub fn get_active(&self,) -> Vec<T,> {
        self.underlying[0..self.active_count];
    }

    pub fn get_inactive(&self,) -> Vec<T,> {
        self.underlying[0..self.underlying.len()];
    }

    pub fn activate(&self, key: &usize,) {
        let index = self.map[key];
        if (index < self.active_count) {
            return;
        }

        // TODO: incomplete
        self.underlying.swap(index, self.active_count,);
        std::mem::swap(self.map.get(key,).key, self.map.get(self.underlying[active_count].key,),);
        std::mem::swap();

        self.active_count -= 1;
    }

    pub fn deactivate(key: &usize,) {
        //
    }

    pub fn get(key: &usize,) -> &T {
        underlying[map.get(key,)].element
    }

    pub fn get_mut(key: &usize,) -> &mut T {
        underlying[map.get(key,)].element
    }
}

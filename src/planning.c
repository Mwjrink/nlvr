struct VkDrawIndexedIndirectCommand {
    uint32_t    indexCount;
    uint32_t    instanceCount;
    uint32_t    firstIndex;
    int32_t     vertexOffset;
    uint32_t    firstInstance;
};

struct InstanceData {
    glm::vec3 pos;
    glm::vec3 rot;
    float scale;
    uint32_t texIndex;
};

struct uniformData {
    glm::mat4 projection;
    glm::mat4 view;
};

struct {
    vks::Buffer scene;
};

struct pipelines {
    VkPipeline plants;
    VkPipeline ground;
    VkPipeline skysphere;
};

struct vex<T> {
    _backing: Vec<T>,
    _active: usize,

    cull () {}
    uncull () {}
    ...
    &[T] active () {}
}

vex<Renderables>.active() ->
GPU_Compute {
    input {
        renderable [] {
            buffer:         vkBuffer,
            vertex_count:   u32,
            vertex_offset:  i32,
            index_count:    u32,
            index_offset:   u32,
            texture_index:  u32,
            instance_count: u32,
            model:          mat4,
        },
        renderable_count:   u64,
        model:              mat4[renderable.sum(ren -> ren.instance_count)],
        // projection:      mat4,
        // view:            mat4,
        projview:           mat4, // projection * view
        indexVertexBuffer:  vkBuffer,
    }

    return {
        VkDrawIndexedIndirectCommand {
            indexCount:    indices_count,
            instanceCount: instance_count,
            firstIndex:    0,
            vertexOffset:  vertex_offset,
            firstInstance: 0, // VUID-VkDrawIndexedIndirectCommand-firstInstance-00554
                              // If the drawIndirectFirstInstance feature is not enabled, firstInstance must be 0
        },
        ObjData [] {
            texture_index: u32,
        },
        InstanceData [] {
            model:         model,
        },
        CommonData {
            projview: projection * view,
        }
    }
}

renderable {
    void cull(); // removes from the draw call but keeps the data loaded in the gpu/vertex buffer
    void unload(); // unloads and removes/marks the old data for rewriting from the gpu memory (need to somehow defrag the data)
}
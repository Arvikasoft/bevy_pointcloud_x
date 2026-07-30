use bevy_core_pipeline::FullscreenShader;
use bevy_ecs::prelude::*;
use bevy_render::render_resource::{
    binding_types::{texture_2d, texture_2d_multisampled},
    *,
};
use bevy_shader::ShaderDefVal;

use crate::render::{
    normalize_pass::EyeDomeLightingUniformBindgroupLayout, NORMALIZE_SHADER_HANDLE,
};

#[derive(PartialEq, Eq, Hash, Clone)]
pub struct NormalizePassPipelineKey {
    pub samples: u32,
    pub use_edl: bool,
    pub edl_neighbour_count: u32,
    /// TurboTwin vendor patch, re-derived for bevy 0.19: specialize the color
    /// target on the view target's actual format. The normalize pass writes to
    /// the camera's view target, which is `Rgba16Float` for HDR views (e.g.
    /// cameras with `Atmosphere`) and the surface format otherwise; the upstream
    /// hardcoded `bevy_default()` (`Rgba8UnormSrgb`) fails wgpu validation on
    /// both HDR views and Bgra8 surfaces.
    ///
    /// 0.18 keyed this on a `hdr: bool` and picked `ViewTarget::TEXTURE_FORMAT_HDR`
    /// vs `bevy_default()`. 0.19 exposes [`ExtractedView::target_format`], which
    /// is the format itself — strictly better, and `BevyDefault` is now
    /// deprecated for exactly this reason.
    ///
    /// [`ExtractedView::target_format`]: bevy_render::view::ExtractedView::target_format
    pub target_format: TextureFormat,
}

#[derive(Component)]
pub struct NormalizePassPipelineId(pub CachedRenderPipelineId);

#[derive(Resource)]
pub struct NormalizePassPipeline {
    pub layout: BindGroupLayoutDescriptor,
    pub layout_msaa: BindGroupLayoutDescriptor,
    pub edl_layout: BindGroupLayoutDescriptor,
    fullscreen_shader: FullscreenShader,
}

impl FromWorld for NormalizePassPipeline {
    fn from_world(world: &mut World) -> Self {
        let layout = BindGroupLayoutDescriptor {
            label: "pcl_normalize_pass_bind_group_layout".into(),
            entries: BindGroupLayoutEntries::sequential(
                ShaderStages::FRAGMENT,
                (
                    // The texture containing the depth
                    // Binding Depth buffer is not supported in WASM/WebGL
                    texture_2d(TextureSampleType::Float { filterable: false }),
                    // The texture containing the rendered point cloud (rgb = weighted sum, a = sum of weights)
                    texture_2d(TextureSampleType::Float { filterable: false }),
                ),
            )
            .to_vec(),
        };
        let layout_msaa = BindGroupLayoutDescriptor {
            label: "pcl_normalize_pass_bind_group_layout_msaa".into(),
            entries: BindGroupLayoutEntries::sequential(
                ShaderStages::FRAGMENT,
                (
                    // The texture containing the mask
                    // We could transmit the complete depth as f32, but we don't need
                    // Binding Depth buffer is not supported in WASM/WebGL
                    texture_2d_multisampled(TextureSampleType::Float { filterable: false }),
                    // The texture containing the rendered point cloud (rgb = weighted sum, a = sum of weights)
                    texture_2d_multisampled(TextureSampleType::Float { filterable: false }),
                ),
            )
            .to_vec(),
        };
        let edl_layout = world
            .resource::<EyeDomeLightingUniformBindgroupLayout>()
            .layout
            .clone();

        Self {
            layout,
            layout_msaa,
            edl_layout,
            fullscreen_shader: world.resource::<FullscreenShader>().clone(),
        }
    }
}

impl SpecializedRenderPipeline for NormalizePassPipeline {
    type Key = NormalizePassPipelineKey;

    fn specialize(&self, key: Self::Key) -> RenderPipelineDescriptor {
        let mut layout = match key.samples {
            1 => vec![self.layout.clone()],
            _ => vec![self.layout_msaa.clone()],
        };

        let mut shader_defs = Vec::new();
        if key.samples > 1 {
            shader_defs.push("MULTISAMPLED".into());
        }
        if key.use_edl {
            layout.push(self.edl_layout.clone());
            shader_defs.push("USE_EDL".into());
            shader_defs.push(ShaderDefVal::UInt(
                "NEIGHBOUR_COUNT".into(),
                key.edl_neighbour_count,
            ));
        }

        RenderPipelineDescriptor {
            label: Some("pcl_normalize_pass_pipeline".into()),
            layout,
            // This will setup a fullscreen triangle for the vertex state
            vertex: self.fullscreen_shader.to_vertex_state(),
            fragment: Some(FragmentState {
                shader: NORMALIZE_SHADER_HANDLE,
                shader_defs,
                entry_point: Some("fragment".into()),
                targets: vec![Some(ColorTargetState {
                    // TurboTwin vendor patch: match the view target format.
                    format: key.target_format,
                    blend: None,
                    write_mask: ColorWrites::ALL,
                })],
            }),
            primitive: PrimitiveState::default(),
            depth_stencil: Some(DepthStencilState {
                format: TextureFormat::Depth32Float,
                // Do not write the depth, because it has already been written in the depth pass, and we don't have access to it
                depth_write_enabled: Some(false),
                depth_compare: Some(CompareFunction::Always),
                stencil: Default::default(),
                bias: Default::default(),
            }),
            multisample: MultisampleState {
                count: key.samples,
                mask: !0,
                alpha_to_coverage_enabled: false,
            },
            immediate_size: 0,
            zero_initialize_workgroup_memory: false,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Guards the vendor patch: the normalize pass must get a *separate*
    /// pipeline per view-target format, or wgpu rejects the pass on HDR views.
    /// If someone drops `target_format` from the key (or stops hashing it),
    /// `SpecializedRenderPipelines` hands out one pipeline for every view again.
    #[test]
    fn key_distinguishes_view_target_formats() {
        let base = NormalizePassPipelineKey {
            samples: 1,
            use_edl: true,
            edl_neighbour_count: 4,
            target_format: TextureFormat::Rgba8UnormSrgb,
        };
        let hdr = NormalizePassPipelineKey {
            target_format: TextureFormat::Rgba16Float,
            ..base.clone()
        };
        assert!(base != hdr);

        // `SpecializedRenderPipelines` keys a HashMap on this, so the format has
        // to reach the hash too.
        use core::hash::BuildHasher;
        let state = std::collections::hash_map::RandomState::new();
        assert_ne!(state.hash_one(&base), state.hash_one(&hdr));
    }
}

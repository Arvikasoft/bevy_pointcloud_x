use bevy_ecs::prelude::*;
use bevy_render::{
    render_resource::*,
    renderer::{RenderContext, ViewQuery},
    view::{ViewDepthTexture, ViewTarget},
};

use crate::render::normalize_pass::{
    eye_dome_lighting::NormalizePassEdlBindgroup, pipeline::NormalizePassPipelineId,
    texture::NormalizePassBindGroup,
};

/// bevy 0.19: was `ViewNode for NormalizePassNode` behind `NormalizePassLabel`.
/// See [`crate::render::depth_pass::node::pcl_depth_pass`].
pub fn pcl_normalize_pass(
    pipeline_cache: Res<PipelineCache>,
    view: ViewQuery<(
        &ViewTarget,
        &ViewDepthTexture,
        &NormalizePassBindGroup,
        Option<&NormalizePassEdlBindgroup>,
        &NormalizePassPipelineId,
    )>,
    mut ctx: RenderContext,
) {
    let (view_target, depth, textures_bind_group, edl_bind_group, pipeline_id) = view.into_inner();

    let Some(pipeline) = pipeline_cache.get_render_pipeline(pipeline_id.0) else {
        return;
    };

    let mut render_pass = ctx.begin_tracked_render_pass(RenderPassDescriptor {
        label: Some("pcl_normalize_pass"),
        color_attachments: &[Some(view_target.get_color_attachment())],
        depth_stencil_attachment: Some(depth.get_attachment(StoreOp::Store)),
        timestamp_writes: None,
        occlusion_query_set: None,
        multiview_mask: None,
    });

    render_pass.set_render_pipeline(pipeline);
    render_pass.set_bind_group(0, &textures_bind_group.value, &[]);

    if let Some(edl_bind_group) = edl_bind_group {
        render_pass.set_bind_group(1, &edl_bind_group.value, &[]);
    }
    render_pass.draw(0..3, 0..1);
}

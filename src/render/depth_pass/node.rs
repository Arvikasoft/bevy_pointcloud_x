use bevy_ecs::prelude::*;
use bevy_log::prelude::*;
use bevy_render::{
    camera::ExtractedCamera,
    render_phase::ViewBinnedRenderPhases,
    render_resource::{RenderPassDescriptor, StoreOp},
    renderer::{RenderContext, ViewQuery},
    view::{ExtractedView, ViewDepthTexture},
};

use crate::render::depth_pass::{phase::PointCloud3dDepthPhase, texture::ViewDepthPrepassTextures};

/// bevy 0.19 replaced the render graph with ECS schedules: this used to be a
/// `ViewNode` (`DepthPassNode` behind `DepthPassLabel`), now it is a plain
/// system registered into the `Core3d` schedule. `ViewQuery` supplies what
/// `ViewNode::ViewQuery` used to, `RenderContext` is a system param, and
/// ordering is expressed with `.after`/`.before` on the pass systems instead of
/// graph edges.
pub fn pcl_depth_pass(
    world: &World,
    view: ViewQuery<(
        &ExtractedCamera,
        &ExtractedView,
        &ViewDepthTexture,
        &ViewDepthPrepassTextures,
    )>,
    point_cloud_3d_phases: Res<ViewBinnedRenderPhases<PointCloud3dDepthPhase>>,
    mut ctx: RenderContext,
) {
    let view_entity = view.entity();
    let (camera, extracted_view, view_depth_texture, view_prepass_textures) = view.into_inner();

    // Get the phase for the current view running our pass
    let Some(point_cloud_3d_phase) =
        point_cloud_3d_phases.get(&extracted_view.retained_view_entity)
    else {
        info!("no pointcloud phase");
        return;
    };

    let color_attachments = vec![view_prepass_textures
        .depth
        .as_ref()
        .map(|attribute_texture| attribute_texture.get_attachment())];

    let mut render_pass = ctx.begin_tracked_render_pass(RenderPassDescriptor {
        label: Some("pcl_depth_pass"),
        // In the depth phase, we write a depth mask to reuse it in normalize pass
        // We could use the depth buffer, but with WebGL we can't bind it in a shader
        color_attachments: &color_attachments,
        // We store the depth for usage in attribute pass
        depth_stencil_attachment: Some(view_depth_texture.get_attachment(StoreOp::Store)),
        timestamp_writes: None,
        occlusion_query_set: None,
        multiview_mask: None,
    });

    if let Some(viewport) = camera.viewport.as_ref() {
        render_pass.set_camera_viewport(viewport);
    }

    if let Err(err) = point_cloud_3d_phase.render(&mut render_pass, world, view_entity) {
        error!("Error encountered while rendering the point cloud depth phase {err:?}");
    }
}

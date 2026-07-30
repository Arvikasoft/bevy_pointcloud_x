use bevy_ecs::prelude::*;
use bevy_log::{error, prelude::*};
use bevy_render::{
    camera::ExtractedCamera,
    render_phase::ViewBinnedRenderPhases,
    render_resource::{RenderPassDescriptor, StoreOp},
    renderer::{RenderContext, ViewQuery},
    view::{ExtractedView, ViewDepthTexture},
};

use super::texture::ViewAttributePrepassTextures;
use crate::render::attribute_pass::phase::PointCloud3dAttributePhase;

/// bevy 0.19: was `ViewNode for AttributePassNode` behind `AttributePassLabel`.
/// See [`crate::render::depth_pass::node::pcl_depth_pass`].
pub fn pcl_attribute_pass(
    world: &World,
    view: ViewQuery<(
        &ExtractedCamera,
        &ExtractedView,
        &ViewDepthTexture,
        &ViewAttributePrepassTextures,
    )>,
    point_cloud_3d_phases: Res<ViewBinnedRenderPhases<PointCloud3dAttributePhase>>,
    mut ctx: RenderContext,
) {
    let view_entity = view.entity();
    let (camera, extracted_view, depth, view_prepass_textures) = view.into_inner();

    // Get the phase for the current view running our pass
    let Some(point_cloud_3d_phase) =
        point_cloud_3d_phases.get(&extracted_view.retained_view_entity)
    else {
        info!("no pointcloud phase");
        return;
    };

    let color_attachments = vec![view_prepass_textures
        .attribute
        .as_ref()
        .map(|attribute_texture| attribute_texture.get_attachment())];

    let mut render_pass = ctx.begin_tracked_render_pass(RenderPassDescriptor {
        label: Some("pcl_attribute_pass"),
        color_attachments: &color_attachments,
        depth_stencil_attachment: Some(depth.get_attachment(StoreOp::Store)),
        timestamp_writes: None,
        occlusion_query_set: None,
        multiview_mask: None,
    });

    if let Some(viewport) = camera.viewport.as_ref() {
        render_pass.set_camera_viewport(viewport);
    }

    if let Err(err) = point_cloud_3d_phase.render(&mut render_pass, world, view_entity) {
        error!("Error encountered while rendering the point cloud attribute phase {err:?}");
    }
}

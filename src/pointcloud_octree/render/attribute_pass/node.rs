use bevy_ecs::prelude::*;
#[cfg(feature = "trace")]
use bevy_log::info_span;
use bevy_log::{error, prelude::*};
use bevy_render::{
    camera::ExtractedCamera,
    render_phase::BinnedPhaseItem,
    render_resource::{RenderPassDescriptor, StoreOp},
    renderer::{RenderContext, ViewQuery},
    view::{ExtractedView, ViewDepthTexture},
};

use crate::{
    pointcloud_octree::render::phase::ViewOctreeNodesRenderAttributePhases,
    render::attribute_pass::texture::ViewAttributePrepassTextures,
};

/// bevy 0.19: was `ViewNode for AttributePassOctreeNode<BPI>` behind
/// `AttributePassOctreeLabel`. See
/// [`crate::render::depth_pass::node::pcl_depth_pass`].
pub fn pcl_octree_attribute_pass<BPI: BinnedPhaseItem>(
    world: &World,
    view: ViewQuery<(
        &ExtractedCamera,
        &ExtractedView,
        &ViewDepthTexture,
        &ViewAttributePrepassTextures,
    )>,
    point_cloud_3d_phases: Res<ViewOctreeNodesRenderAttributePhases<BPI>>,
    mut ctx: RenderContext,
) {
    #[cfg(feature = "trace")]
    let _span = info_span!("pcl_octree_attribute_pass").entered();

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
        label: Some("pcl_octree_attribute_pass"),
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

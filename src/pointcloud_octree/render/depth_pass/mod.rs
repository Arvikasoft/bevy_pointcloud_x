pub mod node;

use bevy_app::prelude::*;
use bevy_camera::{Camera, Camera3d};
use bevy_core_pipeline::Core3d;
use bevy_ecs::{change_detection::Tick, prelude::*};
use bevy_log::prelude::*;
use bevy_pbr::{SetMeshViewBindGroup, ViewKeyCache};
use bevy_platform::collections::HashSet;
use bevy_render::{
    batching::gpu_preprocessing::GpuPreprocessingSupport,
    render_phase::{AddRenderCommand, DrawFunctions, SetItemPipeline},
    render_resource::{PipelineCache, SpecializedRenderPipelines},
    sync_world::MainEntity,
    view::{ExtractedView, RetainedViewEntity},
    Extract, ExtractSchedule, Render, RenderApp, RenderSystems,
};
use node::pcl_octree_depth_pass;

use super::phase::PointCloudOctree3dBinKey;
#[cfg(feature = "webgl")]
use crate::pointcloud_octree::render::draw::DrawPointCloudOctree;
#[cfg(not(feature = "webgl"))]
use crate::pointcloud_octree::render::draw::DrawPointCloudOctreeIndirect;
use crate::{
    octree::extract::render::components::RenderVisibleOctreeNodes,
    pointcloud_octree::{
        asset::data::PointCloudNodeData,
        component::PointCloudOctree3d,
        render::{
            data::SetPointCloudOctree3dUniformGroup,
            draw::{SetPointCloudOctreeNodeUniformGroup, SetRenderOctreeUniformGroup},
            phase::{PointCloudOctree3dNodePhase, ViewOctreeNodesRenderDepthPhases},
            prepare::SetVisibleNodesTexture,
        },
    },
    render::{
        depth_pass::{
            node::pcl_depth_pass,
            pipeline::{DepthPipeline, DepthPipelineKey},
            texture::prepare_depth_pass_textures,
        },
        material::SetPointCloudMaterialGroup,
        phase::PointCloud3dBatchSetKey,
        PointCloudPass, PointCloudRenderMode, PointCloudRenderModeOpt,
    },
};

pub struct DepthPassPlugin;
impl Plugin for DepthPassPlugin {
    fn build(&self, app: &mut App) {
        // We need to get the render app from the main app
        let Some(render_app) = app.get_sub_app_mut(RenderApp) else {
            return;
        };
        render_app
            .init_resource::<DrawFunctions<PointCloudOctree3dNodePhase>>()
            .init_resource::<ViewOctreeNodesRenderDepthPhases<PointCloudOctree3dNodePhase>>()
            .add_render_command::<PointCloudOctree3dNodePhase, DrawDepthPass>()
            .init_resource::<SpecializedRenderPipelines<DepthPipeline>>()
            .add_systems(ExtractSchedule, extract_camera_phases)
            .add_systems(
                Render,
                (
                    prepare_depth_pass_textures.in_set(RenderSystems::PrepareResources),
                    queue_depth_pass.in_set(RenderSystems::QueueMeshes),
                ),
            )
            .add_systems(
                Core3d,
                pcl_octree_depth_pass::<PointCloudOctree3dNodePhase>
                    .in_set(PointCloudPass::Depth)
                    // Both depth passes write the same `ViewDepthPrepassTextures`
                    // attachment, whose Clear-vs-Load is decided by whichever
                    // calls `get_attachment()` first in wall-clock order while
                    // command buffers submit in system order. Unordered, the
                    // clear can land on top of the other pass's output. Vacuous
                    // when the simple renderer is not installed.
                    .after(pcl_depth_pass),
            );
    }
}

// We will reuse render commands already defined by bevy to draw a 3d mesh
#[cfg(not(feature = "webgl"))]
type DrawDepthPass = (
    SetItemPipeline,
    SetMeshViewBindGroup<0>,
    SetPointCloudOctree3dUniformGroup<1>,
    SetPointCloudMaterialGroup<2>,
    SetVisibleNodesTexture<3>,
    SetPointCloudOctreeNodeUniformGroup<4>,
    SetRenderOctreeUniformGroup<5>,
    DrawPointCloudOctreeIndirect,
);

#[cfg(feature = "webgl")]
type DrawDepthPass = (
    SetItemPipeline,
    SetMeshViewBindGroup<0>,
    SetPointCloudOctree3dUniformGroup<1>,
    SetPointCloudMaterialGroup<2>,
    SetVisibleNodesTexture<3>,
    SetPointCloudOctreeNodeUniformGroup<4>,
    SetRenderOctreeUniformGroup<5>,
    DrawPointCloudOctree,
);

fn extract_camera_phases(
    mut pointcloud3d_phases: ResMut<ViewOctreeNodesRenderDepthPhases<PointCloudOctree3dNodePhase>>,
    cameras: Extract<Query<(Entity, &Camera), With<Camera3d>>>,
    mut live_entities: Local<HashSet<RetainedViewEntity>>,
    _gpu_preprocessing_support: Res<GpuPreprocessingSupport>,
) {
    #[cfg(feature = "trace")]
    let _span = info_span!("extract_camera_phases", name = "depth").entered();
    live_entities.clear();
    for (main_entity, camera) in &cameras {
        if !camera.is_active {
            continue;
        }

        // This is the main camera, so we use the first subview index (0)
        let retained_view_entity = RetainedViewEntity::new(main_entity.into(), None, 0);

        pointcloud3d_phases.prepare_for_new_frame(retained_view_entity);

        live_entities.insert(retained_view_entity);
    }

    // Clear out all dead views.
    pointcloud3d_phases.retain(|camera_entity, _| live_entities.contains(camera_entity));
}

#[allow(clippy::too_many_arguments)]
#[allow(clippy::type_complexity)]
fn queue_depth_pass(
    custom_draw_functions: Res<DrawFunctions<PointCloudOctree3dNodePhase>>,
    mut pipelines: ResMut<SpecializedRenderPipelines<DepthPipeline>>,
    pipeline_cache: Res<PipelineCache>,
    custom_draw_pipeline: Res<DepthPipeline>,
    point_cloud_octrees_3d: Query<&PointCloudOctree3d>,
    mut custom_render_phases: ResMut<ViewOctreeNodesRenderDepthPhases<PointCloudOctree3dNodePhase>>,
    view_key_cache: Res<ViewKeyCache>,
    mut views: Query<(
        &ExtractedView,
        &RenderVisibleOctreeNodes<PointCloudNodeData, PointCloudOctree3d>,
        Option<&PointCloudRenderMode>,
    )>,
    main_entities: Query<&MainEntity>,
    mut next_tick: Local<Tick>,
) {
    for (view, visible_entities, point_cloud_render_mode) in &mut views {
        let Some(custom_phase) = custom_render_phases.get_mut(&view.retained_view_entity) else {
            continue;
        };
        let draw_custom = custom_draw_functions.read().id::<DrawDepthPass>();

        // See the simple path's `queue_depth_pass`: read the view key, never
        // rebuild it — the mesh-view bind group layout is derived from all 16
        // of its bits, and a missing one is a silent wgpu validation failure.
        let Some(&view_key) = view_key_cache.get(&view.retained_view_entity) else {
            continue;
        };

        let depth_key = DepthPipelineKey {
            mesh_key: view_key,
            use_edl: point_cloud_render_mode.use_edl(),
            is_octree: true,
        };

        let pipeline_id = pipelines.specialize(&pipeline_cache, &custom_draw_pipeline, depth_key);

        // Since our phase can work on any 3d mesh we can reuse the default mesh 3d filter
        for (render_entity, _) in &visible_entities.octrees {
            let Ok(main_entity) = main_entities.get(*render_entity) else {
                warn!("Render entity not found, skipping.");
                continue;
            };
            let Ok(point_cloud_octree_3d) = point_cloud_octrees_3d.get(*render_entity) else {
                warn!("point_cloud_octree_3d missing");
                continue;
            };

            // Bump the change tick in order to force Bevy to rebuild the bin.
            let this_tick = next_tick.get() + 1;
            next_tick.set(this_tick);

            // Add the render phase
            custom_phase.add(
                PointCloud3dBatchSetKey {
                    pipeline: pipeline_id,
                    draw_function: draw_custom,
                },
                PointCloudOctree3dBinKey {
                    asset_id: point_cloud_octree_3d.0.id(),
                },
                (*render_entity, *main_entity),
            );
        }
    }
}

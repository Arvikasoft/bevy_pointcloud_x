pub mod node;
pub mod phase;
pub mod pipeline;
pub mod texture;

use bevy_app::prelude::*;
use bevy_camera::{Camera, Camera3d};
use bevy_core_pipeline::Core3d;
use bevy_ecs::prelude::*;
use bevy_log::prelude::*;
use bevy_pbr::{MeshPipelineSystems, SetMeshViewBindGroup, ViewKeyCache};
use bevy_platform::collections::HashSet;
use bevy_render::{
    batching::gpu_preprocessing::{GpuPreprocessingMode, GpuPreprocessingSupport},
    init_gpu_resource,
    render_phase::{
        AddRenderCommand, BinnedRenderPhaseType, DrawFunctions, InputUniformIndex, SetItemPipeline,
        ViewBinnedRenderPhases,
    },
    render_resource::{PipelineCache, SpecializedRenderPipelines},
    view::{ExtractedView, NoIndirectDrawing, RenderVisibleEntities, RetainedViewEntity},
    Extract, ExtractSchedule, Render, RenderApp, RenderStartup, RenderSystems,
};
use phase::PointCloud3dDepthPhase;

use crate::{
    point_cloud::PointCloud3d,
    render::{
        depth_pass::{
            node::pcl_depth_pass,
            pipeline::{DepthPipeline, DepthPipelineKey},
            texture::{prepare_depth_pass_textures, DepthPassLayout},
        },
        draw::DrawPointCloud,
        material::SetPointCloudMaterialGroup,
        phase::{PointCloud3dBatchSetKey, PointCloud3dBinKey},
        point_cloud_uniform::SetPointCloudUniformGroup,
        PointCloudPass, PointCloudRenderMode, PointCloudRenderModeOpt,
    },
};

pub struct DepthPassPlugin;
impl Plugin for DepthPassPlugin {
    fn build(&self, app: &mut App) {
        // app.add_plugins(SortedRenderPhasePlugin::<DepthPass3d, MeshPipeline>::new(
        //     RenderDebugFlags::default(),
        // ));

        // We need to get the render app from the main app
        let Some(render_app) = app.get_sub_app_mut(RenderApp) else {
            return;
        };
        render_app
            .init_resource::<DrawFunctions<PointCloud3dDepthPhase>>()
            .init_resource::<ViewBinnedRenderPhases<PointCloud3dDepthPhase>>()
            .add_render_command::<PointCloud3dDepthPhase, DrawDepthPass>()
            .init_resource::<SpecializedRenderPipelines<DepthPipeline>>()
            .add_systems(ExtractSchedule, extract_camera_phases)
            .add_systems(
                Render,
                (
                    prepare_depth_pass_textures.in_set(RenderSystems::PrepareResources),
                    queue_depth_pass.in_set(RenderSystems::QueueMeshes),
                ),
            )
            .add_systems(Core3d, pcl_depth_pass.in_set(PointCloudPass::Depth));
    }

    fn finish(&self, app: &mut App) {
        let Some(render_app) = app.get_sub_app_mut(RenderApp) else {
            return;
        };
        // The layout needs the RenderDevice, which only exists once plugins are
        // initialized.
        render_app.init_resource::<DepthPassLayout>().add_systems(
            // 0.19 moved `MeshPipeline` init out of `Plugin::finish` into
            // `RenderStartup` (bevy_pbr render/mesh.rs ~L319), so a `FromWorld`
            // pipeline reading it can no longer be built in `finish` — it panics
            // with "Requested resource does not exist". Init it in `RenderStartup`
            // ordered after the set that builds `MeshPipeline`, never on
            // registration order. Same shape as bevy's own wireframe pipeline.
            RenderStartup,
            init_gpu_resource::<DepthPipeline>.after(MeshPipelineSystems),
        );
    }
}

// We will reuse render commands already defined by bevy to draw a 3d mesh
type DrawDepthPass = (
    SetItemPipeline,
    SetMeshViewBindGroup<0>,
    SetPointCloudUniformGroup<1>,
    SetPointCloudMaterialGroup<2>,
    DrawPointCloud,
);

#[allow(clippy::type_complexity)]
fn extract_camera_phases(
    mut pointcloud3d_phases: ResMut<ViewBinnedRenderPhases<PointCloud3dDepthPhase>>,
    cameras: Extract<Query<(Entity, &Camera, Has<NoIndirectDrawing>), With<Camera3d>>>,
    mut live_entities: Local<HashSet<RetainedViewEntity>>,
    gpu_preprocessing_support: Res<GpuPreprocessingSupport>,
) {
    live_entities.clear();
    for (main_entity, camera, no_indirect_drawing) in &cameras {
        if !camera.is_active {
            continue;
        }

        // If GPU culling is in use, use it (and indirect mode); otherwise, just
        // preprocess the meshes.
        let gpu_preprocessing_mode = gpu_preprocessing_support.min(if !no_indirect_drawing {
            GpuPreprocessingMode::Culling
        } else {
            GpuPreprocessingMode::PreprocessingOnly
        });

        // This is the main camera, so we use the first subview index (0)
        let retained_view_entity = RetainedViewEntity::new(main_entity.into(), None, 0);

        pointcloud3d_phases.prepare_for_new_frame(retained_view_entity, gpu_preprocessing_mode);

        // clear phases between each iteration
        pointcloud3d_phases
            .get_mut(&retained_view_entity)
            .unwrap()
            .non_mesh_items
            .clear();

        live_entities.insert(retained_view_entity);
    }

    // Clear out all dead views.
    pointcloud3d_phases.retain(|camera_entity, _| live_entities.contains(camera_entity));
}

#[allow(clippy::too_many_arguments, clippy::type_complexity)]
fn queue_depth_pass(
    custom_draw_functions: Res<DrawFunctions<PointCloud3dDepthPhase>>,
    mut pipelines: ResMut<SpecializedRenderPipelines<DepthPipeline>>,
    pipeline_cache: Res<PipelineCache>,
    custom_draw_pipeline: Res<DepthPipeline>,
    point_clouds_3d: Query<&PointCloud3d>,
    mut custom_render_phases: ResMut<ViewBinnedRenderPhases<PointCloud3dDepthPhase>>,
    view_key_cache: Res<ViewKeyCache>,
    mut views: Query<(
        &ExtractedView,
        &RenderVisibleEntities,
        Option<&PointCloudRenderMode>,
    )>,
) {
    for (view, visible_entities, point_cloud_render_mode) in &mut views {
        let Some(custom_phase) = custom_render_phases.get_mut(&view.retained_view_entity) else {
            continue;
        };
        let draw_custom = custom_draw_functions.read().id::<DrawDepthPass>();

        // NEVER hand-assemble this key. `MeshPipelineViewLayoutKey::from` maps 16
        // bits of it into the mesh-view bind group *layout*, and
        // `prepare_mesh_view_bind_groups` derives the matching bind group from the
        // same view state — miss a bit (TONEMAP_IN_SHADER, ENVIRONMENT_MAP, SSAO,
        // SSR, IRRADIANCE_VOLUME, CONTACT_SHADOWS, DISTANCE_FOG, ATMOSPHERE …) and
        // `SetMeshViewBindGroup` fails wgpu validation, i.e. the points silently
        // never draw. `ViewKeyCache` is bevy's answer to exactly this (it is what
        // bevy's own 3D gizmo pass reads); it is filled in `RenderSystems::
        // PrepareAssets`, before `QueueMeshes`.
        let Some(&view_key) = view_key_cache.get(&view.retained_view_entity) else {
            continue;
        };

        let depth_key = DepthPipelineKey {
            mesh_key: view_key,
            use_edl: point_cloud_render_mode.use_edl(),
            is_octree: false,
        };

        let pipeline_id = pipelines.specialize(&pipeline_cache, &custom_draw_pipeline, depth_key);

        // bevy 0.19: `RenderVisibleEntities::iter::<C>()` is gone; the per-class
        // lists live in `RenderVisibleEntitiesClass`, which also carries the
        // change lists.
        let Some(visible) = visible_entities.get::<PointCloud3d>() else {
            continue;
        };
        // Drop entities that went invisible this frame out of the bin cache.
        // `extract_camera_phases` clears `non_mesh_items` every frame, but 0.19
        // no longer tick-sweeps `cached_entity_bin_keys` — the change list is how
        // that is done now.
        for (_, main_entity) in &visible.removed_entities {
            custom_phase.remove(*main_entity);
        }

        // `iter_visible()` (not `entities_cpu_culling`) is the 0.18
        // `RenderVisibleEntities::iter::<C>()` equivalent — the CPU-culled list
        // alone drops anything carrying `NoCpuCulling`.
        for (render_entity, main_entity) in visible.iter_visible() {
            let Ok(point_cloud_3d) = point_clouds_3d.get(*render_entity) else {
                warn!("point_cloud_3d missing");
                continue;
            };

            // At this point we have all the data we need to create a phase item and add it to our
            // phase
            custom_phase.add(
                PointCloud3dBatchSetKey {
                    pipeline: pipeline_id,
                    draw_function: draw_custom,
                },
                PointCloud3dBinKey {
                    asset_id: point_cloud_3d.0.id(),
                },
                (*render_entity, *main_entity),
                InputUniformIndex::default(),
                BinnedRenderPhaseType::NonMesh,
            );
        }
    }
}

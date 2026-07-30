mod eye_dome_lighting;
pub mod node;
mod pipeline;
mod texture;

use bevy_app::prelude::*;
use bevy_core_pipeline::Core3d;
use bevy_ecs::prelude::*;
use bevy_render::{
    render_resource::{PipelineCache, SpecializedRenderPipelines},
    view::{ExtractedView, Msaa},
    Render, RenderApp, RenderSystems,
};
use node::pcl_normalize_pass;
use pipeline::NormalizePassPipeline;

use crate::render::{
    eye_dome_lighting::EyeDomeLightingUniformBindgroupLayout,
    normalize_pass::{
        eye_dome_lighting::prepare_normalize_pass_edl_bind_groups,
        pipeline::{NormalizePassPipelineId, NormalizePassPipelineKey},
        texture::prepare_normalize_pass_bind_groups,
    },
    PointCloudPass, PointCloudRenderMode, PointCloudRenderModeOpt,
};

pub struct NormalizePassPlugin;

impl Plugin for NormalizePassPlugin {
    fn build(&self, app: &mut App) {
        let Some(render_app) = app.get_sub_app_mut(RenderApp) else {
            return;
        };

        render_app
            .init_resource::<SpecializedRenderPipelines<NormalizePassPipeline>>()
            .add_systems(
                Render,
                (
                    prepare_normalize_pass_pipelines.in_set(RenderSystems::Prepare),
                    prepare_normalize_pass_bind_groups.in_set(RenderSystems::PrepareBindGroups),
                    prepare_normalize_pass_edl_bind_groups.in_set(RenderSystems::PrepareBindGroups),
                ),
            )
            .add_systems(Core3d, pcl_normalize_pass.in_set(PointCloudPass::Normalize));
    }

    fn finish(&self, app: &mut App) {
        let Some(render_app) = app.get_sub_app_mut(RenderApp) else {
            return;
        };

        render_app
            // Init this resource here because I need it here
            .init_resource::<EyeDomeLightingUniformBindgroupLayout>()
            // Initialize the pipeline
            .init_resource::<NormalizePassPipeline>();
    }
}

fn prepare_normalize_pass_pipelines(
    mut commands: Commands,
    pipeline_cache: Res<PipelineCache>,
    mut pipelines: ResMut<SpecializedRenderPipelines<NormalizePassPipeline>>,
    pipeline: Res<NormalizePassPipeline>,
    views: Query<(Entity, &ExtractedView, &Msaa, Option<&PointCloudRenderMode>)>,
) {
    for (entity, view, msaa, point_cloud_render_mode) in &views {
        let pipeline_id = pipelines.specialize(
            &pipeline_cache,
            &pipeline,
            NormalizePassPipelineKey {
                samples: msaa.samples(),
                use_edl: point_cloud_render_mode.use_edl(),
                edl_neighbour_count: point_cloud_render_mode.edl_neighbour_count(),
                // TurboTwin vendor patch: see NormalizePassPipelineKey::target_format.
                target_format: view.target_format,
            },
        );

        commands
            .entity(entity)
            .insert(NormalizePassPipelineId(pipeline_id));
    }
}

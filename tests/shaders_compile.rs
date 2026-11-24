//! Builds the point-cloud render pipelines on the machine's GPU adapter and
//! checks that the WESL shaders compile. Rust builds fine when a shader is
//! wrong: the error only shows when a pipeline is created.
//!
//! The cameras cover EDL on and off and MSAA on and off for the simple
//! renderer. The octree path's IS_OCTREE variants of the depth and attribute
//! pipelines are specialized directly with the same view keys (there is no
//! octree in the scene), so their shaders compile too, but no octree is drawn.
//! Without an adapter the test prints a note and passes.
#![cfg(not(target_arch = "wasm32"))]

use std::panic::{catch_unwind, resume_unwind, AssertUnwindSafe};

use bevy::{
    app::{App, PluginsState},
    asset::Assets,
    camera::{Camera3d, RenderTarget},
    ecs::world::Mut,
    image::Image,
    math::Vec3,
    pbr::ViewKeyCache,
    render::{
        render_resource::{
            CachedPipelineState, PipelineCache, PipelineDescriptor, SpecializedRenderPipelines,
            TextureFormat,
        },
        view::{Msaa, Tonemapping},
        RenderApp, RenderPlugin,
    },
    transform::components::Transform,
    window::{ExitCondition, WindowPlugin},
    DefaultPlugins,
};
use bevy_app::PluginGroup;
use bevy_pointcloud_x::{
    point_cloud::{PointCloud, PointCloud3d, PointCloudData},
    point_cloud_material::{PointCloudMaterial, PointCloudMaterial3d},
    pointcloud_octree::PointCloudOctreePlugin,
    render::{
        attribute_pass::pipeline::{AttributePassPipeline, AttributePipelineKey},
        depth_pass::pipeline::{DepthPipeline, DepthPipelineKey},
        PointCloudRenderMode,
    },
    PointCloudPlugin,
};

const FRAMES: usize = 8;

fn build_app() -> App {
    let mut app = App::new();
    app.add_plugins((
        DefaultPlugins
            .set(WindowPlugin {
                primary_window: None,
                exit_condition: ExitCondition::DontExit,
                ..Default::default()
            })
            // Compile each pipeline in the frame that first asks for it, so the
            // state is final when the test reads it.
            .set(RenderPlugin {
                synchronous_pipeline_compilation: true,
                ..Default::default()
            }),
        PointCloudPlugin,
        PointCloudOctreePlugin,
    ));
    app
}

fn spawn_scene(app: &mut App) {
    let world = app.world_mut();
    let cloud = world.resource_mut::<Assets<PointCloud>>().add(PointCloud {
        points: (0..16)
            .map(|i| PointCloudData {
                position: Vec3::new(i as f32 * 0.1 - 0.8, 0.0, 0.0),
                point_size: 0.05,
                color: [1.0, 0.5, 0.2, 1.0],
            })
            .collect(),
    });
    let material = world
        .resource_mut::<Assets<PointCloudMaterial>>()
        .add(PointCloudMaterial {
            point_size: 4.0,
            min_point_size: 1.0,
            max_point_size: 16.0,
        });
    world.spawn((
        PointCloud3d(cloud),
        PointCloudMaterial3d(material),
        Transform::default(),
    ));

    for (msaa, edl) in [
        (Msaa::Off, false),
        (Msaa::Off, true),
        (Msaa::Sample4, false),
        (Msaa::Sample4, true),
    ] {
        let target = world
            .resource_mut::<Assets<Image>>()
            .add(Image::new_target_texture(
                32,
                32,
                TextureFormat::Rgba8UnormSrgb,
                None,
            ));
        world.spawn((
            Camera3d::default(),
            RenderTarget::Image(target.into()),
            // No tonemapping LUTs in this build.
            Tonemapping::None,
            msaa,
            PointCloudRenderMode {
                use_edl: edl,
                ..Default::default()
            },
            Transform::from_xyz(0.0, 0.0, 3.0).looking_at(Vec3::ZERO, Vec3::Y),
        ));
    }
}

#[test]
fn point_cloud_shaders_compile() {
    let built = catch_unwind(AssertUnwindSafe(build_app));
    let mut app = match built {
        Ok(app) => app,
        Err(panic) => {
            let message = panic
                .downcast_ref::<String>()
                .map(String::as_str)
                .or_else(|| panic.downcast_ref::<&str>().copied())
                .unwrap_or_default();
            if message.contains("Unable to find a GPU") {
                println!("note: no GPU adapter on this machine, shader compile check skipped");
                return;
            }
            resume_unwind(panic);
        }
    };

    while app.plugins_state() == PluginsState::Adding {
        bevy::tasks::tick_global_task_pools_on_main_thread();
    }
    app.finish();
    app.cleanup();

    spawn_scene(&mut app);
    for _ in 0..FRAMES {
        app.update();
    }
    specialize_octree_variants(&mut app);
    for _ in 0..FRAMES {
        app.update();
    }

    let cache = app.sub_app(RenderApp).world().resource::<PipelineCache>();
    let mut ours = Vec::new();
    let mut octree = Vec::new();
    for pipeline in cache.pipelines() {
        let label = match &pipeline.descriptor {
            PipelineDescriptor::RenderPipelineDescriptor(d) => d.label.clone(),
            PipelineDescriptor::ComputePipelineDescriptor(d) => d.label.clone(),
            PipelineDescriptor::MeshPipelineDescriptor(d) => d.label.clone(),
        }
        .map(|l| l.to_string())
        .unwrap_or_default();
        if let CachedPipelineState::Err(err) = &pipeline.state {
            panic!("pipeline {label:?} failed: {err}");
        }
        if let PipelineDescriptor::RenderPipelineDescriptor(d) = &pipeline.descriptor
            && d.vertex.shader_defs.contains(&"IS_OCTREE".into())
        {
            octree.push(label.clone());
        }
        if label.starts_with("pcl_") {
            // A shader whose import does not resolve stays queued for good,
            // so ours must be ready, not only free of errors.
            assert!(
                matches!(pipeline.state, CachedPipelineState::Ok(_)),
                "pipeline {label:?} is not ready: {:?}",
                pipeline.state
            );
            ours.push(label);
        }
    }
    ours.sort();
    println!("point-cloud pipelines compiled: {ours:?}");

    // Depth, attribute and normalize, each for the simple and the octree path
    // where it has one, over the EDL and MSAA variants above.
    for name in [
        "pcl_depth_pass_pipeline",
        "pcl_attribute_pass_pipeline",
        "pcl_normalize_pass_pipeline",
    ] {
        assert!(ours.iter().any(|l| l == name), "no {name} was built");
    }
    // The IS_OCTREE variants of depth and attribute (ready, checked above).
    octree.sort();
    println!("octree variants compiled: {octree:?}");
    for name in ["pcl_depth_pass_pipeline", "pcl_attribute_pass_pipeline"] {
        assert!(
            octree.iter().any(|l| l == name),
            "no IS_OCTREE {name} was built"
        );
    }
}

/// The octree queue systems specialize the depth and attribute pipelines with
/// `is_octree` only for a view that sees an octree, and this scene has none.
/// Specialize them the same way here, with each view's key, so the IS_OCTREE
/// shader variants are compiled too.
fn specialize_octree_variants(app: &mut App) {
    let world = app.sub_app_mut(RenderApp).world_mut();
    let keys: Vec<_> = world.resource::<ViewKeyCache>().values().copied().collect();
    assert!(
        !keys.is_empty(),
        "no view key to specialize the octree variants with"
    );
    world.resource_scope(
        |world, mut depth: Mut<SpecializedRenderPipelines<DepthPipeline>>| {
            for &mesh_key in &keys {
                for use_edl in [false, true] {
                    let key = DepthPipelineKey {
                        mesh_key,
                        use_edl,
                        is_octree: true,
                    };
                    depth.specialize(
                        world.resource::<PipelineCache>(),
                        world.resource::<DepthPipeline>(),
                        key,
                    );
                }
            }
        },
    );
    world.resource_scope(
        |world, mut attribute: Mut<SpecializedRenderPipelines<AttributePassPipeline>>| {
            for &mesh_key in &keys {
                let key = AttributePipelineKey {
                    mesh_key,
                    is_octree: true,
                };
                attribute.specialize(
                    world.resource::<PipelineCache>(),
                    world.resource::<AttributePassPipeline>(),
                    key,
                );
            }
        },
    );
}

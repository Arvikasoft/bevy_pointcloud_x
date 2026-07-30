use bevy_ecs::query::QueryItem;
use bevy_render::{extract_component::ExtractComponent, sync_component::SyncComponent};
use bevy_transform::prelude::GlobalTransform;

use crate::{
    point_cloud::PointCloud3d, point_cloud_material::PointCloudMaterial3d,
    render::point_cloud_uniform::PointCloudUniform,
};

// bevy 0.19 split the render-world cleanup contract out of `ExtractComponent`:
// `Target` is what gets removed from the render world when `PointCloud3d` is
// removed from the main world, so it mirrors `Out`.
impl SyncComponent for PointCloud3d {
    type Target = (PointCloud3d, PointCloudUniform, PointCloudMaterial3d);
}

impl ExtractComponent for PointCloud3d {
    type QueryData = (
        &'static PointCloud3d,
        &'static GlobalTransform,
        &'static PointCloudMaterial3d,
    );
    type QueryFilter = ();
    type Out = (PointCloud3d, PointCloudUniform, PointCloudMaterial3d);

    fn extract_component(
        (point_cloud_3d, global_transform, point_cloud_material_3d): QueryItem<
            '_,
            '_,
            Self::QueryData,
        >,
    ) -> Option<Self::Out> {
        let custom_uniform = PointCloudUniform {
            world_from_local: global_transform.to_matrix(),
        };
        Some((
            point_cloud_3d.clone(),
            custom_uniform,
            point_cloud_material_3d.clone(),
        ))
    }
}

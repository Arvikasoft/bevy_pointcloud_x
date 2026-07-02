# bevy_pointcloud_x

Potree-style point-cloud rendering for [Bevy](https://bevyengine.org) — EDL
shading, octree LOD, LAS/LAZ + PLY loaders, an in-memory `PointCloud` asset.

This is a crates.io-publishable fork of
[rlamarche/bevy_pointcloud](https://github.com/rlamarche/bevy_pointcloud)
(vendored at rev `4c16ead`, MIT).

## Why a fork under a different name?

Upstream is git-only: its Potree/COPC streaming-loader features depend on
git-hosted crates (`potree-rs`, `copc-streaming`), which crates.io forbids in
published crates — so it cannot be published under its own name, and we leave
the `bevy_pointcloud` name to its author. This fork:

- **removes the `potree` / `copc` / `wasm_worker` features** (and their git
  dependencies). Everything else remains: the `PointCloud` asset, the octree
  renderer, EDL shading, the LAS/LAZ and PLY loaders.
- **carries WebGPU/MSAA correctness patches** (grep `TurboTwin vendor patch`):
  - the normalize pass specializes its color target on the view's HDR-ness
    (upstream hardcoded the default format, failing wgpu validation on
    `Rgba16Float` views);
  - f16 intermediates in the depth/attribute passes — WebGPU rejects float32
    color formats for multisampling and blending, which native Vulkan
    tolerates;
  - the depth/attribute passes match the view's MSAA sample count;
  - a screen-space minimum point size in the simple (non-octree) path so
    distant points don't drop below one raster sample and vanish.

Maintained by [Arvikasoft](https://github.com/Arvikasoft) as the point-tile
renderer behind
[`bevy_3d_tiles`](https://github.com/Arvikasoft/bevy_3d_tiles). The patches
are offered upstream; if upstream publishes to crates.io one day, prefer it.

## Bevy compatibility

| `bevy_pointcloud_x` | Bevy |
|---|---|
| 0.1 | 0.18 |

---

## Upstream README

WASM Demo site : https://rlamarche.github.io/bevy_pointcloud/

A Bevy plugin to render point clouds.

Roadmap :

- [x] Render point cloud using a shader
- [x] Implement rendering described in https://www.cg.tuwien.ac.at/research/publications/2016/SCHUETZ-2016-POT/SCHUETZ-2016-POT-thesis.pdf
- [x] Read Potree files in https://www.cg.tuwien.ac.at/research/publications/2016/SCHUETZ-2016-POT/SCHUETZ-2016-POT-thesis.pdf
- [x] Efficiently load/unload chunks based on camera location
- [x] Support for COPC Laz point clouds

*(The Potree/COPC streaming loaders are disabled in this fork — see above.
Use upstream for those.)*

## Credits

- Rendering techniques and shaders adapted from [Potree](https://github.com/potree/potree)
  Copyright (c) 2011-2020, Markus Schütz
  Licensed under the BSD 2-Clause License (see THIRD_PARTY_LICENCES.md).

- Potree itself is based on prior research and open source projects.
  Notably:
    - Multi-resolution octree algorithms developed at the Vienna University of Technology (Michael Wimmer, Claus Scheiblauer, Scanopy Project).
    - Shaders inspired by CloudCompare (EDL), plas.io (LAS/LAZ support), and other point cloud tools.

- The rendering pipeline implementation is based on Bevy's render pipelines
  ([Bevy Engine](https://bevyengine.org), MIT or Apache-2.0).

- COPC Parsing from https://github.com/360-geo/copc (`copc-streaming` crate)

## Acknowledgements

This project would not have been possible without the amazing work of the
open-source and research communities. In particular, upstream acknowledges:

- [Potree](https://github.com/potree/potree) by Markus Schütz, which inspired the
  rendering techniques and some shader code used here.
- [Bevy Engine](https://bevyengine.org), whose rendering pipelines served as a
  foundation for the Rust-side implementation.
- The Vienna University of Technology (Michael Wimmer, Claus Scheiblauer) for their
  research on multi-resolution octrees in the [Scanopy Project](http://www.cg.tuwien.ac.at/research/projects/Scanopy/).
- [CloudCompare](http://www.danielgm.net/cc/) for the EDL shader approach.
- [plas.io](http://plas.io/) for early contributions to LAS/LAZ point cloud rendering.

## Contributors (upstream)

- [Romain Lamarche](https://github.com/rlamarche) (author)
- [François Mockers](https://github.com/mockersf)
- [Tin Lai](https://github.com/soraxas)

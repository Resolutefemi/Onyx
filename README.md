<div align="center">

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="assets/brand/onyx-logo-lockup-dark.svg">
  <img src="assets/brand/onyx-logo-lockup.svg" width="520" alt="Onyx Engine logo"/>
</picture>

# ONYX ENGINE

**Pure Rust · AI-Native · Next-Gen Polyglot 2D/3D Game Engine**

Sub-5ms Agentic MCP Control · Bevy-class ECS · wgpu Renderer · WASM Component Model Scripting

![Status](https://img.shields.io/badge/status-pre--alpha--design-black)
![License](https://img.shields.io/badge/license-MIT-black)
![Language](https://img.shields.io/badge/core-100%25_Rust-black)

</div>

---

## What is Onyx?

Onyx Engine is an open-source, pure-Rust 2D/3D game engine engineered to challenge AAA engines like Unreal Engine in visual capability and raw performance, while surpassing Bevy and Godot in developer velocity, polyglot scripting, and AI-native automation. Onyx exposes its entire simulation loop, asset pipeline, and ECS scene hierarchy via a zero-latency MCP (Model Context Protocol) server — allowing AI agents to execute batch mutations and playtest scenes in under 5 milliseconds.

> **Status: pre-alpha — M0 and M1 (ECS Core) complete, PRD §1 fully delivered; M2 (Renderer) complete for its milestone scope; M3 (MCP Control Plane) next.** PRD §1 now covers the full core architecture: archetype SoA `World` with change detection and a scheduler that dispatches read-only system batches on real threads, a reversible command bus with undo/redo, transaction grouping and branching snapshots, bump/frame-scratch/slab allocators with deterministic budgets, a worker job pool, the scene & prefab document layer (nested prefabs, variant overrides, stable entity id paths, JSON round-trips), and `#[derive(OnyxReflect)]` with JSON + tagged binary serialization. M2 has its foundation, first GPU work and the resource layer: `wgpu` 30 adapter/device init with graceful headless degradation, a resource-driven render graph (inferred producer→consumer edges, cycle detection), a real WGSL triangle pipeline with a deterministic pixel contract, single-submission graph execution with per-target readback, bind groups (uniform + texture + sampler), a vertexless textured-quad pipeline with a pixel-exact round-trip contract and linear-space tinting, 4× MSAA with resolve, per-target MSAA inside the graph (multisampled attachments resolving into readback textures), and a parallel wave-based executor whose output is bit-identical to the sequential one, plus sampling passes inside the graph (a node's declared read resource binds as a texture through the quad bind group — pixel-exact copies, linear-space tinting, uv sub-windows), a format-agnostic executor (per-format pipelines, BGRA readback swizzle), windowed present (the swapchain frame is an external graph target — sRGB format + Fifo vsync policy, acquire → render the backbuffer node → present), forward PBR (a Cook-Torrance GGX material pass whose WGSL shader and CPU mirror share the exact operation sequence, so the shaded color is predictable byte for byte on every backend), the 2D sprite suite: batched sprites with rects, uv windows and linear tints through a CPU-mirrored viewport camera, post-processing screen adjustments (Godot `Environment`-style brightness/contrast/saturation through a bit-for-bit CPU-mirrored shader chain), and the separable tent blur (a 3-tap `[0.25, 0.5, 0.25]` pass along a texel direction; two chained nodes form the glow/bloom base) (see [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md)). The full subsystem-by-subsystem specification lives in [`docs/PRD.md`](docs/PRD.md) (see the [roadmap](#roadmap)).

## Why Onyx?

| vs. | The gap | Onyx's answer |
| --- | --- | --- |
| **Unreal Engine 5** | Multi-gigabyte C++ monoliths, slow compile times, manual UI clicking for every iteration | Lightweight multi-threaded Rust core, and programmatic sub-5ms MCP control instead of GUI clicking |
| **Bevy** | Lightning-fast ECS, but editor, scripting and stylized rendering are bring-your-own | Built-in tactile visual editor, native WASM multi-language scripting, out-of-the-box stylized Anime-PBR shaders |
| **Godot 4** | Object-oriented scene tree hits single-thread bottlenecks on high-entity open worlds | Lock-free archetype-based ECS with data-oriented, parallel-by-default systems |

## Architecture Reference Templates

Onyx stands on the shoulders of proven open-source projects, adopting one architectural component from each:

| Reference Project | Primary Language | Architectural Component Adopted |
| --- | --- | --- |
| [bevy](https://github.com/bevyengine/bevy) | Rust | Entity Component System (ECS) architecture & lock-free system scheduler |
| [godot](https://github.com/godotengine/godot) | C++ / GDScript | Scene tree serialization formats and visual editor layout paradigms |
| [fyrox](https://github.com/FyroxEngine/Fyrox) | Rust | Scene graph serialization patterns and built-in Rust UI inspector components |
| [wgpu](https://github.com/gfx-rs/wgpu) | Rust | Cross-platform WebGPU graphics abstraction over Vulkan, Metal and DX12 |

## Core Technical Stack

- **Core Engine Runtime** — 100% Rust, `bevy_ecs` for high-performance entity management
- **Graphics & Shading** — `wgpu` + raw Vulkan (`ash`): meshlet culling, GPU-driven rendering, custom Anime-PBR cel shading
- **Polyglot Scripting** — sandboxed `wasmtime` engine using the WebAssembly Component Model (WIT)
- **Physics** — `rapier2d` / `rapier3d`: character controllers, ragdolls, raycasting
- **AI MCP Control Plane** — native JSON-RPC tool server (`rmcp`) exposing 50+ atomic commands over IPC/stdio

## Feature Map

The [Master Feature Specification](docs/PRD.md) details all 16 subsystems. Highlights:

- **ECS & Memory** — archetype SoA storage, lock-free parallel scheduler, change detection, reflection macros, unified reversible command bus with shared human/AI undo-redo
- **Rendering** — deferred PBR / Forward+ / mobile / Anime-PBR / ray-traced pipelines, GPU-driven meshlet virtualization, Hi-Z occlusion culling, full post-processing stack
- **World Building** — clipmap-LOD terrain with erosion brushes, GPU foliage scattering, FFT oceans, world partition & f64 floating-origin for solar-system scale, node-based PCG (WFC, L-systems)
- **Physics** — Rapier rigid bodies + constraints, kinematic character & vehicle controllers, cloth, spring bones, Voronoi destruction
- **Animation** — GPU skinning, blend spaces, Motion Matching, FABRIK/CCD/full-body IK, Control Rig, retargeting
- **VFX & Audio** — compute-shader particle system (Niagara-class), deferred decals, HRTF spatial audio, acoustic raytracing, beat-synced adaptive music
- **UI** — Flexbox/Grid layout (Taffy), MSDF text with HarfBuzz shaping, full widget library, world-space diegetic UI
- **AI & Gameplay** — Recast/Detour navmesh, behavior trees / GOAP / EQS, first-class dialogue, quest, inventory & ability systems
- **Networking** — QUIC/ENet/Renet transports, client prediction & rollback netcode, headless dedicated server target
- **Polyglot Scripting** — Rust / Python / TypeScript / C++ / Go / C# / Zig / Lua / Swift compiled to sandboxed `.wasm` components with sub-100ms hot-reload
- **Editor & Tools** — gizmos, inspector, sequencer, Tracy flamegraphs, in-game console, automated headless regression playtests
- **Asset Pipeline & XR** — glTF/VRM/FBX/USD import, Basis Universal compression, OpenXR runtime, 1-click builds to desktop / mobile / web / console HALs

## The AI Control Plane (`mcp`)

Every feature above is exposed to AI agents as typed JSON-Schema MCP tools, live world-state resources, and multi-step workflow prompts:

```jsonc
// batch_execute: 100+ mutations in a single frame (<5 ms)
{
  "method": "tools/call",
  "params": {
    "name": "batch_execute",
    "arguments": {
      "commands": [
        { "tool": "spawn_terrain",    "args": { "size": 512, "layers": 8 } },
        { "tool": "rig_character",    "args": { "asset": "hero.vrm" } },
        { "tool": "set_light_preset", "args": { "preset": "golden_hour" } },
        { "tool": "step_physics_frames", "args": { "count": 60 } }
      ],
      "mode": "atomic"
    }
  }
}
```

Key properties:

- **Self-describing registry** — every engine subsystem automatically becomes a typed MCP tool (`tools/list`), world state readable via `resources/list`, workflows via `prompts/list`
- **Batched atomic RPC** — 100+ terrain, entity, lighting, shader and script mutations in one frame, zero GUI overhead
- **Headless simulation** — agents run `step_frames(60)`, query exact raycasts/collisions/bounds mathematically, or pull compressed viewport/depth/segmentation renders for visual QA
- **Human + AI co-editing** — live activity log, ghost-preview approval mode, auto-pilot mode, and shared Ctrl+Z undo/redo between human and AI

## Polyglot Scripting

| Language | Toolchain | Notes |
| --- | --- | --- |
| Rust | `cargo-component` | Zero-overhead, also available as native plugins |
| Python | `componentize-python` / RustPython | Gameplay scripting without bindings boilerplate |
| TypeScript / JS | `jco` | Full WASM Component Model support |
| C / C++ | `wit-bindgen`, Clang | Legacy code reuse |
| Go | `wasm-tools`, TinyGo | |
| C# | `wasmtime dotnet` | |
| Zig / Lua / Swift | `wit-bindgen` | |

## Roadmap

| Milestone | Scope |
| --- | --- |
| **M0 — Scaffold** | Cargo workspace, CI, brand, PRD sign-off ✅ *(complete)* |
| **M1 — ECS Core** | Archetype ECS, scheduler, command bus, reflection & serialization, memory & threading, scene & prefab ✅ *(PRD §1 fully delivered — sparse-set side tables consciously deferred: unit-struct components already tag entities through archetypes; MessagePack deferred to the asset pipeline (§15); typed component application into the World lands with the editor's dynamic type bridge (M5))* |
| **M2 — Renderer** | wgpu HAL, forward PBR, 2D suite, post-processing ✅ *(complete — wgpu 30 adapter/device init with graceful NoAdapter, resource-driven DAG graph with inferred edges + cycle detection, WGSL triangle pipeline with pixel-contract validation, single-submission graph executor, bind groups + uniform buffers + RGBA textures + samplers, vertexless textured quad with pixel-exact round-trip, 4× MSAA with resolve, per-target MSAA inside the graph, a parallel wave-based multi-encoder executor bit-identical to the sequential one, sampling passes inside the graph (declared reads bind as textures), a format-agnostic executor (per-format pipelines, BGRA swizzle), windowed present (swapchain frame as external graph target, sRGB + Fifo policy), forward PBR (Cook-Torrance GGX material pass, CPU-mirrored bit for bit, MSAA-capable, feeds sampling passes), the 2D sprite suite (`PassKind::Sprites2d`: batched rects/uv/tints, one batch → one draw call, 1:1 viewport camera CPU-mirrored bit for bit, submission-order overdraw, per-target MSAA, full-viewport sprite = pixel-exact copy), and post-processing screen adjustments (`PassKind::Adjust`: brightness/contrast/saturation in a fixed order with Rec. 709 luma, `post::adjust` CPU mirror ≡ WGSL bit for bit, [0, 8] param validation, per-target MSAA, identity = pixel-exact copy), and the separable tent blur (`PassKind::Blur`: 3-tap `[0.25, 0.5, 0.25]` along a texel direction, `post::tent3` CPU mirror ≡ WGSL bit for bit, two chained nodes = the separable glow/bloom base, per-target MSAA, blur of a solid = pixel-exact copy))* |
| **M3 — MCP Control Plane** | `rmcp` JSON-RPC server, `batch_execute`, headless sim, undo/redo bridge |
| **M4 — Physics & Animation** | Rapier integration, character controller, skeletal animation & blending |
| **M5 — Editor** | Visual editor, inspector, gizmos, prefab system |
| **M6 — Polyglot & Assets** | WASM component runtime, hot-reload, asset pipeline, platform targets |

## Contributing

The engine is in early implementation (M0 and M1 complete; M2 renderer in progress) — the most valuable contributions right now are feedback on [`docs/PRD.md`](docs/PRD.md), reference implementations of isolated subsystems (see the [architecture templates](#architecture-reference-templates)), and MCP tool-schema review from AI-tooling developers. For build instructions and the workspace layout, see [`CONTRIBUTING.md`](CONTRIBUTING.md) and [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md).

## License

Onyx Engine is released under the [MIT License](LICENSE).

# Onyx Engine — Architecture (M2 in progress)

This document maps the Cargo workspace onto the subsystem catalog in
[`PRD.md`](PRD.md) and fixes the rules that later milestones build on. The
workspace stayed dependency-free through M0/M1; M2 (renderer) introduced the
first external dependencies — `wgpu` and `pollster` in
`[workspace.dependencies]` — exactly when its milestone started, so CI is
green from day one and the remaining heavyweight dependencies (`wasmtime`,
`rapier`, `rmcp`) land when their milestones start.

## Workspace layout

```text
Onyx/
├── crates/
│   ├── onyx/          umbrella crate + `onyx` CLI (info, sim, render,  (PRD §1)
│   │                  --version)
│   ├── onyx-core/     app loop, time, unified reversible command bus   (PRD §1)
│   ├── onyx-ecs/      archetype SoA ECS: world, queries, change        (PRD §1)
│   │                  detection, parallel-batch schedule, plugin
│   ├── onyx-reflect/  reflection: schemas, JSON, type registry         (PRD §1)
│   ├── onyx-reflect-macro/  #[derive(OnyxReflect)] (no-dep proc macro)
│   ├── onyx-render/   wgpu 30 HAL, resource-driven DAG graph, WGSL     (PRD §2–4)
│   │                  triangle pipeline, offscreen graph executor;
│   │                  PBR + 2D pipelines in M2
│   ├── onyx-mcp/      sub-5ms AI control plane, batched atomic RPC     (PRD §16)
│   └── onyx-script/   polyglot WASM component runtime                  (PRD §13)
├── docs/              PRD and this document
├── assets/brand/      logos, wordmark, app icon set
└── .github/workflows/ CI: fmt, clippy, build + test on Linux/Windows/macOS
```

## Crate map

| Crate | PRD scope | Owned milestone | Current contents |
| --- | --- | --- | --- |
| `onyx-core` | §1 Core Architecture | M1 | `App` runner + `Plugin` model, `Time` (variable delta + fixed-step accrual), `EngineCommand` / `CommandBus` with transaction grouping, replay log, shared undo-redo and branching `HistoryMark` snapshots; `memory` — `Bump` arena (alignment-checked, `Copy`-only), double-buffered `FrameScratch`, generational `Slab` with ABA-safe keys; `jobs` — `JobPool` workers with counter-based `JobHandle` joins and chunked `parallel_for`; `scene` — `SceneDocument` / `PrefabRegistry`: nested prefab instantiation, variant overrides, stable `parent/child` id paths, JSON round-trips |
| `onyx-ecs` | §1 Core Architecture | M1 | `World`: dense SoA archetype storage with entity migration and row patching; `Entity` generational handles; `view`/`view2` (+ `_mut`) queries; change detection (`iter_changed`, `take_removed`); `Schedule` with access-declared `System`s, same-stage conflict rejection and parallel read-only batches (`reader_system` + scoped threads); `EcsPlugin` fixed-timestep driver |
| `onyx-reflect` | §1 Core Architecture | M1 | `Reflect` trait + `#[derive(OnyxReflect)]` (via `onyx-reflect-macro`), `FieldValue` / `Schema`, `to_json` / `from_json` (flat scalar objects), `json` — full JSON DOM (nested objects/arrays, depth-limited parser, order-preserving objects), `binary` — compact tagged `FieldValue` serialization, `TypeRegistry` with `schemas_json` export seeding MCP tool schemas |
| `onyx-reflect-macro` | §1 Core Architecture | M1 | Hand-rolled derive on the built-in `proc_macro` crate — no `syn`/`quote` — for non-generic structs with named scalar fields (`f32`, `f64`, `u32`, `u64`, `i32`, `i64`, `bool`, `String`) |
| `onyx-render` | §2–4 Rendering | M2 | `Renderer::try_new` — wgpu 30 instance/adapter/device/queue with backend filtering and graceful `RenderError::NoAdapter` on GPU-less machines; `RenderGraph` — nodes declare resource reads/writes, edges inferred (producer→consumer, conflicting writers by declaration order), Kahn topological order + cycle detection; `pipeline` — inlined WGSL triangle shader, `Vertex` layout + explicit byte packing, `create_triangle_pipeline`; `executor::execute_offscreen` — one encoder, passes in graph order, single submission, per-target readback; `render_offscreen` clear pass with sRGB-verified readback; `Backend` enum, `RendererConfig`, `PipelineId` |
| `onyx-mcp` | §16 AI Control Plane | M3 | `ToolSpec` + `ToolRegistry` (`tools/list` shape), `BatchCommand` / `BatchResult` (`batch_execute` semantics), batch validation |
| `onyx-script` | §13 Polyglot Scripting | M6 | `ScriptLanguage` (11 targets), `ScriptRuntimeConfig` (hot-reload, fuel limit, instance cap) |
| `onyx` | composition | M0 → | re-exports of all subsystems, `ENGINE_NAME` / `ENGINE_VERSION`, `onyx` CLI (`info`, `sim` motion benchmark with reflected JSON sample, `render` GPU probe with offscreen graph checksums, `--version`) |

## Dependency direction

```text
            ┌─────────── onyx (umbrella + CLI) ───────────┐
            │        depends on every subsystem           │
            ▼                                             ▼
  onyx-core ←── onyx-ecs        onyx-render   onyx-mcp   onyx-script
  (base layer)  (uses core)     (may use      (uses     (may use
  onyx-reflect (standalone,     core+ecs)     core)     core+ecs)
  re-exports the macro)
```

Rules:

1. Subsystem crates may depend on `onyx-core` (and `onyx-ecs`) — never on the
   `onyx` umbrella crate. `onyx-reflect` depends on nothing but its macro and
   is usable from any crate, including the editor (M5) and MCP (M3).
2. The `onyx` umbrella crate depends on every subsystem and on nothing else
   heavyweight; it stays the composition + CLI layer.
3. New external dependencies are added to `[workspace.dependencies]` and
   inherited via `.workspace = true`, so versions resolve once for the whole
   workspace.
4. A crate's public API that is still a stub must say *which milestone*
   replaces it, with a link to the PRD section (see any M0 module doc).

## Milestone mapping

| Milestone | Status / what exists | What lands there |
| --- | --- | --- |
| **M0 — Scaffold** ✅ | Workspace, CI, brand, PRD | — |
| **M1 — ECS Core** ✅ | SoA archetype storage ✅, entity migration ✅, `view`/`view2` queries ✅, change detection ✅, scheduler with conflict rejection + parallel read-only batches ✅, command-bus transactions + replay ✅, branching `HistoryMark` snapshots ✅, `Bump`/`FrameScratch`/`Slab` allocators ✅, `JobPool` with `parallel_for` ✅, scene & prefab document layer (nested prefabs, overrides, stable id paths, JSON) ✅, `#[derive(OnyxReflect)]` + JSON + JSON DOM + tagged binary ✅ | Deferred by decision: sparse-set side tables (unit-struct components already tag entities via archetypes), MessagePack (asset pipeline §15), typed dynamic component application (editor bridge M5) |
| **M2 — Renderer** 🚧 | wgpu 30 adapter/device init + graceful `NoAdapter` ✅, resource-driven DAG graph (inferred write→read edges, conflict ordering, cycle detection) ✅, WGSL triangle pipeline + sRGB pixel contract ✅, single-submission graph executor with per-target readback ✅ | Remaining: surfaces & present, bind groups & resource sampling, parallel multi-encoder execution, forward PBR, 2D suite, post-processing stack |
| **M3 — MCP Control Plane** | `onyx-mcp` registry + batch shapes | `rmcp` JSON-RPC server, atomic `batch_execute` <5 ms, headless sim, undo/redo bridge to `onyx-core` |
| **M4 — Physics & Animation** | crate TBD (`onyx-physics`) | Rapier 2D/3D integration, character controller, skeletal animation & blending |
| **M5 — Editor** | crate TBD (`onyx-editor`) | Visual editor, inspector, gizmos, prefab system |
| **M6 — Polyglot & Assets** | `onyx-script` config surface | wasmtime + WIT component runtime, hot-reload, asset pipeline, platform targets |

## Conventions

- **Naming**: crates `onyx-*`, types PascalCase, functions snake_case; stable
  machine-readable ids (`Backend::as_str`, `Stage::as_str`,
  `ScriptLanguage::id`, `EngineCommand::name`) double as MCP-facing names.
- **Errors**: fallible public APIs return `Result`/`Option`; engine internals
  avoid panics outside tests and explicit `expect` on truly-impossible states.
- **Testing**: every public type ships with unit tests; the `onyx` CLI has
  integration tests (`crates/onyx/tests/cli.rs`) that CI runs as a smoke check.
- **MSRV**: `rust-version = "1.82"` is the floor; `rust-toolchain.toml` pins
  the stable channel contributors and CI resolve. Bump one minor at a time.
- **Formatting/lint**: `cargo fmt` (rustfmt.toml) and
  `cargo clippy --workspace --all-targets -- -D warnings` are enforced by CI.

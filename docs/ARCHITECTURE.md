# Onyx Engine — Architecture (M1, first increment)

This document maps the Cargo workspace onto the subsystem catalog in
[`PRD.md`](PRD.md) and fixes the rules that later milestones build on. The
workspace is deliberately dependency-free: every crate compiles with `cargo`
alone, so CI is green from day one and heavy dependencies (`wgpu`, `wasmtime`,
`rapier`, `rmcp`) are introduced exactly when their milestone starts.

## Workspace layout

```text
Onyx/
├── crates/
│   ├── onyx/          umbrella crate + `onyx` CLI (info, sim, --version)
│   ├── onyx-core/     app loop, time, unified reversible command bus   (PRD §1)
│   ├── onyx-ecs/      archetype SoA ECS: world, queries, change        (PRD §1)
│   │                  detection, schedule, fixed-timestep plugin
│   ├── onyx-render/   wgpu HAL, render graph, PBR + 2D pipelines       (PRD §2–4)
│   ├── onyx-mcp/      sub-5ms AI control plane, batched atomic RPC     (PRD §16)
│   └── onyx-script/   polyglot WASM component runtime                  (PRD §13)
├── docs/              PRD and this document
├── assets/brand/      logos, wordmark, app icon set
└── .github/workflows/ CI: fmt, clippy, build + test on Linux/Windows/macOS
```

## Crate map

| Crate | PRD scope | Owned milestone | Current contents |
| --- | --- | --- | --- |
| `onyx-core` | §1 Core Architecture | M1 | `App` runner + `Plugin` model, `Time` (variable delta + fixed-step accrual), `EngineCommand` / `CommandBus` with transaction grouping, replay log and shared undo-redo |
| `onyx-ecs` | §1 Core Architecture | M1 | `World`: dense SoA archetype storage with entity migration and row patching; `Entity` generational handles; `view`/`view2` (+ `_mut`) queries; change detection (`iter_changed`, `take_removed`); `Schedule` with access-declared `System`s and same-stage conflict rejection; `EcsPlugin` fixed-timestep driver |
| `onyx-render` | §2–4 Rendering | M2 | `Backend` enum (Vulkan/DX12/Metal/WebGPU), `RendererConfig`, `PipelineId` |
| `onyx-mcp` | §16 AI Control Plane | M3 | `ToolSpec` + `ToolRegistry` (`tools/list` shape), `BatchCommand` / `BatchResult` (`batch_execute` semantics), batch validation |
| `onyx-script` | §13 Polyglot Scripting | M6 | `ScriptLanguage` (11 targets), `ScriptRuntimeConfig` (hot-reload, fuel limit, instance cap) |
| `onyx` | composition | M0 → | re-exports of all subsystems, `ENGINE_NAME` / `ENGINE_VERSION`, `onyx` CLI (`info`, `sim` motion benchmark with deterministic checksum, `--version`) |

## Dependency direction

```text
            ┌─────────── onyx (umbrella + CLI) ───────────┐
            │        depends on every subsystem           │
            ▼                                             ▼
  onyx-core ←── onyx-ecs        onyx-render   onyx-mcp   onyx-script
  (base layer)  (uses core)     (may use      (uses     (may use
                                 core+ecs)     core)     core+ecs)
```

Rules:

1. Subsystem crates may depend on `onyx-core` (and, once M1 lands, on
   `onyx-ecs`) — never on the `onyx` umbrella crate.
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
| **M1 — ECS Core** 🚧 | SoA archetype storage ✅, entity migration ✅, `view`/`view2` queries ✅, change detection ✅, access-declared scheduler with conflict rejection ✅, command-bus transactions + replay ✅ | Remaining: sparse-set tags for high-churn components, lock-free parallel dispatch of independent systems, `#[derive(OnyxReflect)]` reflection, serialization |
| **M2 — Renderer** | `onyx-render` config surface | wgpu HAL, forward PBR, 2D suite, post-processing stack |
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

# Onyx Engine — Architecture (M0)

This document maps the M0 Cargo workspace onto the subsystem catalog in
[`PRD.md`](PRD.md) and fixes the rules that M1+ builds on. The workspace is
deliberately dependency-free at M0: every crate compiles with `cargo` alone, so
CI is green from day one and heavy dependencies (`wgpu`, `wasmtime`, `rapier`,
`rmcp`) are introduced exactly when their milestone starts.

## Workspace layout

```text
Onyx/
├── crates/
│   ├── onyx/          umbrella crate + `onyx` CLI (info, --version)
│   ├── onyx-core/     app loop, time, unified reversible command bus   (PRD §1)
│   ├── onyx-ecs/      archetype ECS seed: entities, components,        (PRD §1)
│   │                  archetype signatures, schedule stages
│   ├── onyx-render/   wgpu HAL, render graph, PBR + 2D pipelines       (PRD §2–4)
│   ├── onyx-mcp/      sub-5ms AI control plane, batched atomic RPC     (PRD §16)
│   └── onyx-script/   polyglot WASM component runtime                  (PRD §13)
├── docs/              PRD and this document
├── assets/brand/      logos, wordmark, app icon set
└── .github/workflows/ CI: fmt, clippy, build + test on Linux/Windows/macOS
```

## Crate map

| Crate | PRD scope | Owned milestone | M0 contents |
| --- | --- | --- | --- |
| `onyx-core` | §1 Core Architecture | M0 → M1 | `App` runner + `Plugin` model, `Time` (variable delta + fixed-step accrual), `EngineCommand` / `CommandBus` shared undo-redo history |
| `onyx-ecs` | §1 Core Architecture | M0 → M1 | `Entity` generational handles + `Entities` allocator, `Component` marker + `ComponentRegistry`, `ArchetypeSignature`, `Stage` / `Schedule` |
| `onyx-render` | §2–4 Rendering | M0 → M2 | `Backend` enum (Vulkan/DX12/Metal/WebGPU), `RendererConfig`, `PipelineId` |
| `onyx-mcp` | §16 AI Control Plane | M0 → M3 | `ToolSpec` + `ToolRegistry` (`tools/list` shape), `BatchCommand` / `BatchResult` (`batch_execute` semantics), batch validation |
| `onyx-script` | §13 Polyglot Scripting | M0 → M6 | `ScriptLanguage` (11 targets), `ScriptRuntimeConfig` (hot-reload, fuel limit, instance cap) |
| `onyx` | composition | M0 | re-exports of all subsystems, `ENGINE_NAME` / `ENGINE_VERSION`, `onyx` CLI with `info`, `--version`, exit-code contract |

## Dependency direction

```text
            ┌─────────── onyx (umbrella + CLI) ───────────┐
            │        depends on every subsystem           │
            ▼                                             ▼
  onyx-core ←── onyx-ecs        onyx-render   onyx-mcp   onyx-script
  (base layer)  (may use core)   (may use      (uses     (may use
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

| Milestone | What exists at M0 | What lands there |
| --- | --- | --- |
| **M0 — Scaffold** ✅ | This workspace, CI, brand, PRD | — |
| **M1 — ECS Core** | `onyx-ecs` seed types | SoA archetype storage, sparse-set tags, lock-free parallel scheduler, change detection, `#[derive(OnyxReflect)]`, serialization |
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

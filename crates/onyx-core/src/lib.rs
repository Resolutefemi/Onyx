//! # onyx-core
//!
//! Core runtime plumbing for the Onyx Engine (`docs/PRD.md` §1):
//!
//! - [`app`] — the headless application runner and `Plugin` registration model.
//! - [`time`] — the engine clock: variable delta plus fixed-timestep accrual.
//! - [`command_bus`] — every engine mutation (human or AI) as a reversible
//!   [`EngineCommand`] routed through one shared undo/redo history with
//!   transaction grouping, deterministic replay and branching
//!   [`HistoryMark`] snapshots.
//! - [`memory`] — `Bump` arena, double-buffered [`memory::FrameScratch`] and
//!   generational [`memory::Slab`] storage with used/capacity introspection
//!   for deterministic memory budgets.
//! - [`jobs`] — a fixed worker [`JobPool`] with counter-based
//!   [`JobHandle`] joins and chunked `parallel_for` dispatch.
//! - [`scene`] — the Hybrid Scene & Prefab System document layer: nested
//!   prefab instantiation, variant overrides, stable entity id paths and
//!   JSON round-trips (typed component application lands with the editor's
//!   dynamic type bridge, M5).
//!
//! The archetype entity storage and parallel scheduler live in `onyx-ecs`
//! (M1); reflection & serialization live in `onyx-reflect`; the MCP bridge
//! that feeds the command bus from AI agents lands in `onyx-mcp` (M3).

pub mod app;
pub mod command_bus;
pub mod jobs;
pub mod memory;
pub mod scene;
pub mod time;

pub use app::{App, Plugin};
pub use command_bus::{CommandBus, EngineCommand, HistoryMark};
pub use jobs::{JobHandle, JobPool};
pub use memory::{Bump, FrameScratch, Slab, SlabKey};
pub use scene::{
    ComponentMap, EntityTemplate, Prefab, PrefabRegistry, ResolvedEntity, ResolvedScene,
    SceneDocument, SceneError,
};
pub use time::{Time, FIXED_TIMESTEP_HZ, FIXED_TIMESTEP_SECS};

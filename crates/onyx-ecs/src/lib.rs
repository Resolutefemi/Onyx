//! # onyx-ecs
//!
//! Archetype Entity Component System for the Onyx Engine (`docs/PRD.md` §1).
//!
//! M1 (first increment) lands the storage and scheduling core:
//!
//! - [`world::World`] — dense SoA archetype storage: [`world::Entity`]
//!   handles, component insert/remove/despawn with archetype migration,
//!   `view`/`view2` (+ `_mut`) queries, change detection
//!   ([`world::World::iter_changed`], [`world::World::take_removed`]),
//! - [`schedule`] — stage-ordered [`schedule::System`]s with declared access
//!   and same-stage conflict rejection; parallel dispatch of independent
//!   systems is the next M1 increment,
//! - [`plugin::EcsPlugin`] — drives the schedule on the fixed timestep inside
//!   `onyx-core`'s `App` loop.
//!
//! Still ahead in M1: sparse-set side tables for high-churn tags, the
//! lock-free parallel executor, and `#[derive(OnyxReflect)]` reflection
//! driving serialization and MCP schemas.

pub mod archetype;
pub mod column;
pub mod component;
pub mod entity;
pub mod plugin;
pub mod schedule;
pub mod world;

pub use archetype::{Archetype, ArchetypeSignature};
pub use column::{Column, Ticks, TypedColumn};
pub use component::{Component, ComponentRegistry};
pub use entity::{Entities, Entity};
pub use plugin::EcsPlugin;
pub use schedule::{
    system, FnSystem, Schedule, ScheduleConflictError, Stage, System, SystemAccess,
};
pub use world::{EntityLocation, World};

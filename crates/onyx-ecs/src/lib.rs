//! # onyx-ecs
//!
//! Archetype Entity Component System for the Onyx Engine (`docs/PRD.md` §1).
//!
//! M1 lands the storage and scheduling core:
//!
//! - [`world::World`] — dense SoA archetype storage: [`world::Entity`]
//!   handles, component insert/remove/despawn with archetype migration,
//!   `view`/`view2` (+ `_mut`) queries, change detection
//!   ([`world::World::iter_changed`], [`world::World::take_removed`]),
//! - [`schedule`] — stage-ordered [`schedule::System`]s with declared access
//!   and same-stage conflict rejection; consecutive read-only systems
//!   ([`schedule::reader_system`]) dispatch in parallel batches on scoped
//!   threads between the writers,
//! - [`plugin::EcsPlugin`] — drives the schedule on the fixed timestep inside
//!   `onyx-core`'s `App` loop.
//!
//! M1 reflection & serialization live in the separate `onyx-reflect` crate
//! (`#[derive(OnyxReflect)]`, JSON, type registry). Sparse-set side tables
//! for high-churn tags are consciously deferred: unit-struct components
//! already tag entities through archetypes.

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
    reader_system, system, FnReaderSystem, FnSystem, Schedule, ScheduleConflictError, Stage,
    System, SystemAccess,
};
pub use world::{EntityLocation, World};

//! # onyx-ecs
//!
//! Archetype Entity Component System for the Onyx Engine (`docs/PRD.md` §1).
//!
//! M0 seeds the identity and typing layer — [`entity::Entity`], the
//! [`component`] registry, [`archetype::ArchetypeSignature`] and
//! [`schedule::Stage`] — so subsystem crates can already name their data.
//! M1 replaces these seeds with the full implementation:
//!
//! - cache-friendly SoA archetype storage with sparse-set side tables,
//! - lock-free parallel scheduler with automatic dependency resolution,
//! - change detection (`Changed<T>`, `Added<T>`, `Removed<T>`),
//! - `#[derive(OnyxReflect)]` reflection driving serialization and MCP schemas.

pub mod archetype;
pub mod component;
pub mod entity;
pub mod schedule;

pub use archetype::ArchetypeSignature;
pub use component::{Component, ComponentRegistry};
pub use entity::{Entities, Entity};
pub use schedule::{Schedule, Stage};

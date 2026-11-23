//! # onyx-core
//!
//! Core runtime plumbing for the Onyx Engine:
//!
//! - [`app`] — the headless application runner and `Plugin` registration model.
//! - [`time`] — the engine clock: variable delta plus fixed-timestep accrual.
//! - [`command_bus`] — every engine mutation (human or AI) as a reversible
//!   [`EngineCommand`] routed through one shared undo/redo history.
//!
//! This crate implements the foundations of `docs/PRD.md` §1 (Core
//! Architecture, ECS & Memory Management). The archetype entity storage and
//! parallel scheduler live in `onyx-ecs` (M1); the MCP bridge that feeds the
//! command bus from AI agents lands in `onyx-mcp` (M3).

pub mod app;
pub mod command_bus;
pub mod time;

pub use app::{App, Plugin};
pub use command_bus::{CommandBus, EngineCommand};
pub use time::{Time, FIXED_TIMESTEP_HZ, FIXED_TIMESTEP_SECS};

//! # onyx-mcp
//!
//! The native Sub-5ms MCP AI Control Plane ("The AI Dictionary") —
//! `docs/PRD.md` §16.
//!
//! M0 ships the tool-registry data model: [`ToolSpec`], the
//! [`ToolRegistry`] and the [`BatchCommand`]/[`BatchResult`] shapes behind
//! `batch_execute`. M3 lands the real transport:
//!
//! - `rmcp` JSON-RPC server over IPC/stdio (`tools/list`, `resources/list`,
//!   `prompts/list`),
//! - atomic `batch_execute` of 100+ mutations inside one frame (<5 ms),
//! - headless simulation queries (`step_frames`, raycasts, bounds),
//! - ghost-preview approval mode and the shared undo/redo bridge into
//!   `onyx-core`'s `CommandBus`.

pub mod tools;

pub use tools::{BatchCommand, BatchResult, ToolRegistry, ToolSpec};

//! # onyx-mcp
//!
//! The native Sub-5ms MCP AI Control Plane ("The AI Dictionary") —
//! `docs/PRD.md` §16.
//!
//! M3's first increment lands the control-plane core:
//!
//! - [`rpc`] — the JSON-RPC 2.0 envelope (parse/invalid-request/method-
//!   not-found/invalid-params error codes) the transport will carry,
//! - [`tools`] — the self-describing registry data model (M0): every
//!   engine capability as a typed [`ToolSpec`] with a JSON-Schema
//!   `input_schema`,
//! - [`plane`] — [`ControlPlane`], binding the registry to `onyx-core`'s
//!   [`onyx_core::CommandBus`]: `tools/list`, atomic `batch_execute` (one
//!   transaction per batch — a 100-command AI batch is one shared
//!   Ctrl+Z), `history/undo`/`redo`/`snapshot`/`restore` (the shared
//!   undo/redo bridge over branching marks), `world/state` and the live
//!   activity log of every call,
//! - [`scene_tools`] — the first real [`crate::plane::ToolFactory`]-backed
//!   commands (`spawn_entity`, `despawn_entity`, `set_component`)
//!   mutating a shared [`onyx_core::SceneDocument`] through reversible
//!   apply/revert pairs.
//!
//! Still ahead in M3: the `rmcp` JSON-RPC transport over IPC/stdio
//! (`tools/list`, `resources/list`, `prompts/list`), headless simulation
//! queries (`step_frames`, raycasts, bounds), and ghost-preview approval
//! mode.

pub mod plane;
pub mod rpc;
pub mod scene_tools;
pub mod tools;

pub use plane::{ControlPlane, OpOutcome, ToolFactory, ACTIVITY_CAPACITY};
pub use rpc::{
    RpcError, RpcRequest, INTERNAL_ERROR, INVALID_PARAMS, INVALID_REQUEST, METHOD_NOT_FOUND,
    PARSE_ERROR,
};
pub use scene_tools::{
    DespawnEntity, SetComponent, SharedScene, SpawnEntity, TOOL_DESPAWN_ENTITY, TOOL_SET_COMPONENT,
    TOOL_SPAWN_ENTITY,
};
pub use tools::{BatchCommand, BatchResult, ToolRegistry, ToolSpec};

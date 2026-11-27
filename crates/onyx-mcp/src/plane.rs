//! The control-plane core: binds the self-describing [`ToolRegistry`] to
//! `onyx-core`'s [`CommandBus`] and speaks JSON-RPC 2.0 in-process
//! (`docs/PRD.md` §16). This is what makes shared human/AI undo-redo and
//! batched atomic RPC real: [`ControlPlane::batch_execute`] routes every
//! command through the bus — atomically inside one transaction when
//! requested — so a 100-command AI batch is ONE Ctrl+Z for the human, and
//! every call lands in the live activity log.

use std::cell::RefCell;
use std::collections::BTreeMap;
use std::rc::Rc;

use serde_json::{json, Map, Value};

use onyx_core::{CommandBus, EngineCommand, HistoryMark, SceneDocument};

use crate::rpc::{failure, success, RpcError, RpcRequest, INVALID_PARAMS, METHOD_NOT_FOUND};
use crate::scene_tools::{
    self, SharedScene, TOOL_DESPAWN_ENTITY, TOOL_SET_COMPONENT, TOOL_SPAWN_ENTITY,
};
use crate::tools::{BatchCommand, BatchResult, ToolRegistry, ToolSpec};

/// The activity log keeps this many most-recent entries.
pub const ACTIVITY_CAPACITY: usize = 128;

/// One recorded control-plane call (PRD §16, "Live Activity Log showing
/// every AI tool call").
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ActivityEntry {
    /// Monotonic sequence number, starting at 1.
    pub seq: u64,
    /// The JSON-RPC method invoked (`?` when the envelope was unparseable).
    pub method: String,
    /// Whether the call produced a result (`true`) or an error.
    pub ok: bool,
    /// One-line outcome summary.
    pub summary: String,
}

/// Per-command outcome inside [`ControlPlane::batch_execute`].
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct OpOutcome {
    /// The tool id.
    pub tool: String,
    /// Whether the command ended applied (`false` = rejected, or rolled
    /// back with an aborted atomic batch).
    pub applied: bool,
    /// Why the command did not apply (`None` when applied).
    pub reason: Option<String>,
}

/// A tool factory: turn `args_json` into a concrete [`EngineCommand`].
pub type ToolFactory = Rc<dyn Fn(&str) -> Result<Box<dyn EngineCommand>, String>>;

/// The in-process MCP control plane over the engine's command bus.
pub struct ControlPlane {
    bus: CommandBus,
    registry: ToolRegistry,
    factories: BTreeMap<String, ToolFactory>,
    marks: BTreeMap<String, HistoryMark>,
    activity: Vec<ActivityEntry>,
    scene: Option<SharedScene>,
    seq: u64,
    mark_seq: u64,
}

impl Default for ControlPlane {
    fn default() -> Self {
        Self::new()
    }
}

impl ControlPlane {
    /// An empty plane: no tools, no scene.
    #[must_use]
    pub fn new() -> Self {
        Self {
            bus: CommandBus::new(),
            registry: ToolRegistry::new(),
            factories: BTreeMap::new(),
            marks: BTreeMap::new(),
            activity: Vec::new(),
            scene: None,
            seq: 0,
            mark_seq: 0,
        }
    }

    /// A plane with a fresh empty scene named `name` attached as the live
    /// world state (`world/state` reads it, the built-in scene tools
    /// mutate it).
    #[must_use]
    pub fn with_scene(name: impl Into<String>) -> Self {
        let mut plane = Self::new();
        plane.scene = Some(Rc::new(RefCell::new(SceneDocument::new(name))));
        plane
    }

    /// Register `spec` with `factory`. Returns `false` (no change) when a
    /// tool with the same name already exists or `input_schema` is not
    /// valid JSON — a registered tool always describes itself.
    pub fn register_tool(&mut self, spec: ToolSpec, factory: ToolFactory) -> bool {
        if serde_json::from_str::<Value>(spec.input_schema).is_err() {
            return false;
        }
        let name = spec.name;
        if !self.registry.register(spec) {
            return false;
        }
        self.factories.insert(name.to_owned(), factory);
        true
    }

    /// Register the built-in scene tools (`spawn_entity`, `despawn_entity`,
    /// `set_component`) against the attached scene. Returns `false` when no
    /// scene is attached or any tool name was already registered.
    pub fn register_scene_tools(&mut self) -> bool {
        let Some(scene) = self.scene.clone() else {
            return false;
        };
        let mut ok = true;
        ok &= self.register_tool(TOOL_SPAWN_ENTITY, scene_tools::spawn_factory(&scene));
        ok &= self.register_tool(TOOL_DESPAWN_ENTITY, scene_tools::despawn_factory(&scene));
        ok &= self.register_tool(
            TOOL_SET_COMPONENT,
            scene_tools::set_component_factory(&scene),
        );
        ok
    }

    /// The tool registry (registration order preserved for `tools/list`).
    #[must_use]
    pub fn registry(&self) -> &ToolRegistry {
        &self.registry
    }

    /// The command bus (undo/redo depths, replay log).
    #[must_use]
    pub fn bus(&self) -> &CommandBus {
        &self.bus
    }

    /// The retained activity entries, oldest first.
    #[must_use]
    pub fn activity(&self) -> &[ActivityEntry] {
        &self.activity
    }

    /// Execute `commands` through the bus. With `atomic`, all commands run
    /// inside ONE transaction: the first failure — unknown tool, bad args,
    /// engine refusal — rolls the batch back, `applied` ends at 0 and
    /// `rejected` at the batch size, while a committed batch is a single
    /// undoable unit (one shared Ctrl+Z). Without `atomic`, every command
    /// stands alone: failures skip, successes keep.
    pub fn batch_execute(
        &mut self,
        commands: &[BatchCommand],
        atomic: bool,
    ) -> (BatchResult, Vec<OpOutcome>) {
        let mut result = BatchResult::default();
        let mut outcomes: Vec<OpOutcome> = Vec::with_capacity(commands.len());
        if atomic {
            self.bus.begin_transaction();
        }
        for command in commands {
            if result.aborted {
                outcomes.push(OpOutcome {
                    tool: command.tool.clone(),
                    applied: false,
                    reason: Some("batch aborted".to_owned()),
                });
                continue;
            }
            let Some(factory) = self.factories.get(command.tool.as_str()) else {
                if atomic {
                    result.aborted = true;
                }
                outcomes.push(OpOutcome {
                    tool: command.tool.clone(),
                    applied: false,
                    reason: Some(format!("unknown tool '{}'", command.tool)),
                });
                continue;
            };
            let outcome = match factory(&command.args_json) {
                Ok(engine_command) => {
                    if self.bus.execute(engine_command) {
                        OpOutcome {
                            tool: command.tool.clone(),
                            applied: true,
                            reason: None,
                        }
                    } else {
                        if atomic {
                            result.aborted = true;
                        }
                        OpOutcome {
                            tool: command.tool.clone(),
                            applied: false,
                            reason: Some("engine refused the command".to_owned()),
                        }
                    }
                }
                Err(reason) => {
                    if atomic {
                        result.aborted = true;
                    }
                    OpOutcome {
                        tool: command.tool.clone(),
                        applied: false,
                        reason: Some(reason),
                    }
                }
            };
            outcomes.push(outcome);
        }
        if atomic {
            if result.aborted {
                self.bus.rollback_transaction();
                // The M0 contract: in atomic mode nothing survives an abort.
                for outcome in &mut outcomes {
                    outcome.applied = false;
                    if outcome.reason.is_none() {
                        outcome.reason = Some("rolled back: atomic abort".to_owned());
                    }
                }
            } else {
                self.bus.commit_transaction();
            }
        }
        if result.aborted {
            result.applied = 0;
            result.rejected = commands.len();
        } else {
            result.applied = outcomes.iter().filter(|outcome| outcome.applied).count();
            result.rejected = outcomes.len() - result.applied;
        }
        (result, outcomes)
    }

    /// Shared undo: revert the last history unit (one command or one
    /// committed batch).
    pub fn undo(&mut self) -> bool {
        self.bus.undo()
    }

    /// Shared redo.
    pub fn redo(&mut self) -> bool {
        self.bus.redo()
    }

    /// Record a named position in the undo history (auto-named when `name`
    /// is `None`); returns the name.
    pub fn snapshot(&mut self, name: Option<&str>) -> String {
        self.mark_seq += 1;
        let name = name.map_or_else(|| format!("mark-{}", self.mark_seq), str::to_owned);
        self.marks.insert(name.clone(), self.bus.snapshot());
        name
    }

    /// Restore a named snapshot (branching: the redo stack is cleared);
    /// returns the reverted unit count, `None` for an unknown name.
    pub fn restore(&mut self, name: &str) -> Option<usize> {
        let mark = *self.marks.get(name)?;
        Some(self.bus.restore(mark))
    }

    /// Handle one JSON-RPC 2.0 request string; returns the response JSON
    /// string. Every call — parseable or not — lands in the activity log.
    #[must_use]
    pub fn handle(&mut self, request_json: &str) -> String {
        match crate::rpc::parse_request(request_json) {
            Err(error) => {
                self.record("?", false, &error.message);
                failure(&Value::Null, &error).to_string()
            }
            Ok(request) => {
                let outcome = self.dispatch(&request);
                let response = match &outcome {
                    Ok(result) => success(&request.id, result.clone()),
                    Err(error) => failure(&request.id, error),
                };
                let (ok, summary) = match &outcome {
                    Ok(result) => (true, summarize(result)),
                    Err(error) => (false, error.message.clone()),
                };
                self.record(&request.method, ok, summary);
                response.to_string()
            }
        }
    }

    /// The live world state as JSON: scene name and every entity's id and
    /// inline components (an unattached scene reads as `null` name, no
    /// entities).
    #[must_use]
    pub fn world_state(&self) -> Value {
        let Some(scene) = &self.scene else {
            return json!({ "name": Value::Null, "entities": [] });
        };
        let scene = scene.borrow();
        let entities: Vec<Value> = scene
            .entities
            .iter()
            .map(|entity| {
                let mut components = Map::new();
                for (key, value) in &entity.components {
                    components.insert(key.clone(), scene_tools::field_value_to_json(value));
                }
                json!({ "id": entity.id, "components": Value::Object(components) })
            })
            .collect();
        json!({ "name": scene.name, "entities": entities })
    }

    fn record(&mut self, method: &str, ok: bool, summary: impl Into<String>) {
        self.seq += 1;
        self.activity.push(ActivityEntry {
            seq: self.seq,
            method: method.to_owned(),
            ok,
            summary: summary.into(),
        });
        if self.activity.len() > ACTIVITY_CAPACITY {
            self.activity.remove(0);
        }
    }

    fn dispatch(&mut self, request: &RpcRequest) -> Result<Value, RpcError> {
        match request.method.as_str() {
            "tools/list" => Ok(self.tools_list()),
            "batch_execute" => self.batch_rpc(&request.params),
            "history/undo" => {
                let changed = self.bus.undo();
                Ok(self.history_status(changed))
            }
            "history/redo" => {
                let changed = self.bus.redo();
                Ok(self.history_status(changed))
            }
            "history/snapshot" => Ok(self.snapshot_rpc(&request.params)),
            "history/restore" => self.restore_rpc(&request.params),
            "world/state" => Ok(self.world_state()),
            "activity/log" => Ok(self.activity_rpc(&request.params)),
            other => Err(RpcError::new(
                METHOD_NOT_FOUND,
                format!("unknown method '{other}'"),
            )),
        }
    }

    fn tools_list(&self) -> Value {
        let tools: Vec<Value> = self
            .registry
            .list()
            .iter()
            .map(|tool| {
                let schema: Value = serde_json::from_str(tool.input_schema)
                    .expect("schemas validated at registration");
                json!({
                    "name": tool.name,
                    "description": tool.description,
                    "inputSchema": schema,
                })
            })
            .collect();
        json!({ "tools": tools })
    }

    fn batch_rpc(&mut self, params: &Value) -> Result<Value, RpcError> {
        let entries = params
            .get("commands")
            .ok_or_else(|| RpcError::new(INVALID_PARAMS, "params.commands is required"))?
            .as_array()
            .ok_or_else(|| RpcError::new(INVALID_PARAMS, "params.commands must be an array"))?;
        // PRD §16: "Batched Atomic RPC" — atomic is the default.
        let atomic = params
            .get("atomic")
            .and_then(Value::as_bool)
            .unwrap_or(true);
        let mut commands = Vec::with_capacity(entries.len());
        for (index, entry) in entries.iter().enumerate() {
            let object = entry.as_object().ok_or_else(|| {
                RpcError::new(
                    INVALID_PARAMS,
                    format!("commands[{index}] must be an object"),
                )
            })?;
            let tool = object
                .get("tool")
                .and_then(Value::as_str)
                .ok_or_else(|| {
                    RpcError::new(
                        INVALID_PARAMS,
                        format!("commands[{index}].tool must be a string"),
                    )
                })?
                .to_owned();
            let args_json = object
                .get("args_json")
                .and_then(Value::as_str)
                .unwrap_or("{}")
                .to_owned();
            commands.push(BatchCommand { tool, args_json });
        }
        let (result, outcomes) = self.batch_execute(&commands, atomic);
        let operations: Vec<Value> = outcomes
            .iter()
            .map(|outcome| {
                json!({
                    "tool": outcome.tool,
                    "applied": outcome.applied,
                    "reason": outcome.reason,
                })
            })
            .collect();
        Ok(json!({
            "applied": result.applied,
            "rejected": result.rejected,
            "aborted": result.aborted,
            "total": result.total(),
            "fully_applied": result.fully_applied(),
            "operations": operations,
        }))
    }

    fn history_status(&self, changed: bool) -> Value {
        json!({
            "changed": changed,
            "undo_depth": self.bus.undo_depth(),
            "redo_depth": self.bus.redo_depth(),
        })
    }

    fn snapshot_rpc(&mut self, params: &Value) -> Value {
        let name = params.get("name").and_then(Value::as_str);
        let name = self.snapshot(name);
        json!({ "name": name, "undo_depth": self.bus.undo_depth() })
    }

    fn restore_rpc(&mut self, params: &Value) -> Result<Value, RpcError> {
        let name = params
            .get("name")
            .and_then(Value::as_str)
            .ok_or_else(|| RpcError::new(INVALID_PARAMS, "params.name is required"))?;
        let Some(restored) = self.restore(name) else {
            return Err(RpcError::new(
                INVALID_PARAMS,
                format!("unknown snapshot '{name}'"),
            ));
        };
        Ok(json!({
            "name": name,
            "restored": restored,
            "undo_depth": self.bus.undo_depth(),
            "redo_depth": self.bus.redo_depth(),
        }))
    }

    fn activity_rpc(&self, params: &Value) -> Value {
        let limit = params
            .get("limit")
            .and_then(Value::as_u64)
            .map_or(self.activity.len(), |limit| {
                (limit as usize).min(self.activity.len())
            });
        let entries: Vec<Value> = self.activity[self.activity.len() - limit..]
            .iter()
            .map(|entry| {
                json!({
                    "seq": entry.seq,
                    "method": entry.method,
                    "ok": entry.ok,
                    "summary": entry.summary,
                })
            })
            .collect();
        json!({ "entries": entries })
    }
}

/// Short activity-log summary for one result value.
fn summarize(result: &Value) -> String {
    if let (Some(applied), Some(rejected)) = (
        result.get("applied").and_then(Value::as_u64),
        result.get("rejected").and_then(Value::as_u64),
    ) {
        return format!("batch: {applied} applied, {rejected} rejected");
    }
    if let Some(tools) = result.get("tools").and_then(Value::as_array) {
        return format!("listed {} tool(s)", tools.len());
    }
    if let Some(changed) = result.get("changed").and_then(Value::as_bool) {
        return if changed {
            "history moved"
        } else {
            "history unchanged"
        }
        .to_owned();
    }
    if let Some(restored) = result.get("restored").and_then(Value::as_u64) {
        return format!("restored {restored} unit(s)");
    }
    "ok".to_owned()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn plane() -> ControlPlane {
        let mut plane = ControlPlane::with_scene("level-1");
        assert!(plane.register_scene_tools());
        plane
    }

    fn call(plane: &mut ControlPlane, method: &str, params: Value) -> Value {
        let request = json!({ "jsonrpc": "2.0", "id": 1, "method": method, "params": params });
        let response: Value =
            serde_json::from_str(&plane.handle(&request.to_string())).expect("response JSON");
        assert_eq!(response["id"], 1);
        response
    }

    #[test]
    fn tools_list_describes_registered_tools() {
        let mut plane = plane();
        let response = call(&mut plane, "tools/list", json!({}));
        let tools = response["result"]["tools"].as_array().expect("tools");
        assert_eq!(tools.len(), 3);
        let names: Vec<&str> = tools
            .iter()
            .map(|tool| tool["name"].as_str().expect("name"))
            .collect();
        assert_eq!(names, ["spawn_entity", "despawn_entity", "set_component"]);
        for tool in tools {
            assert!(tool["inputSchema"].is_object(), "schema parses as JSON");
            assert!(!tool["description"].as_str().expect("desc").is_empty());
        }
    }

    #[test]
    fn atomic_batch_is_all_or_nothing_and_one_undo_unit() {
        let mut plane = plane();
        // First failure (unknown tool) aborts: NOTHING applied, bus clean.
        let batch = json!({
            "commands": [
                { "tool": "spawn_entity", "args_json": "{\"id\":\"a\"}" },
                { "tool": "spawn_entity", "args_json": "{\"id\":\"b\"}" },
                { "tool": "mystery", "args_json": "{}" },
            ]
        });
        let response = call(&mut plane, "batch_execute", batch);
        let result = &response["result"];
        assert_eq!(result["aborted"], true);
        assert_eq!(result["applied"], 0, "M0 contract: 0 applied on abort");
        assert_eq!(result["rejected"], 3);
        assert_eq!(plane.bus().undo_depth(), 0, "rollback leaves no history");
        assert_eq!(
            plane.world_state()["entities"]
                .as_array()
                .expect("entities")
                .len(),
            0
        );

        // A clean atomic batch commits as ONE undoable unit.
        let batch = json!({
            "commands": [
                { "tool": "spawn_entity", "args_json": "{\"id\":\"a\"}" },
                { "tool": "spawn_entity", "args_json": "{\"id\":\"b\",\"args\":0}" },
            ]
        });
        let response = call(&mut plane, "batch_execute", batch);
        let result = &response["result"];
        assert_eq!(result["aborted"], false);
        assert_eq!(result["applied"], 2);
        assert_eq!(result["fully_applied"], true);
        assert_eq!(
            plane.bus().undo_depth(),
            1,
            "one committed batch = one unit"
        );

        // One shared Ctrl+Z reverts the whole batch.
        let response = call(&mut plane, "history/undo", json!({}));
        assert_eq!(response["result"]["changed"], true);
        assert_eq!(response["result"]["undo_depth"], 0);
        assert_eq!(
            plane.world_state()["entities"]
                .as_array()
                .expect("entities")
                .len(),
            0
        );
        let response = call(&mut plane, "history/redo", json!({}));
        assert_eq!(response["result"]["changed"], true);
        assert_eq!(
            plane.world_state()["entities"]
                .as_array()
                .expect("entities")
                .len(),
            2
        );
    }

    #[test]
    fn atomic_batch_rolls_back_apply_refusals() {
        let mut plane = plane();
        // 'a' exists already: the first command applies, the second is
        // refused, the atomic batch rolls the first one back.
        let batch = json!({
            "commands": [
                { "tool": "spawn_entity", "args_json": "{\"id\":\"a\"}" },
                { "tool": "spawn_entity", "args_json": "{\"id\":\"a\"}" },
            ]
        });
        let response = call(&mut plane, "batch_execute", batch);
        assert_eq!(response["result"]["aborted"], true);
        let reasons: Vec<&str> = response["result"]["operations"]
            .as_array()
            .expect("ops")
            .iter()
            .map(|op| op["reason"].as_str().unwrap_or(""))
            .collect();
        assert!(reasons[0].contains("rolled back"), "reasons: {reasons:?}");
        assert!(reasons[1].contains("refused"), "reasons: {reasons:?}");
        assert_eq!(
            plane.world_state()["entities"]
                .as_array()
                .expect("entities")
                .len(),
            0
        );
    }

    #[test]
    fn non_atomic_batch_applies_what_it_can() {
        let mut plane = plane();
        let batch = json!({
            "atomic": false,
            "commands": [
                { "tool": "spawn_entity", "args_json": "{\"id\":\"a\"}" },
                { "tool": "mystery", "args_json": "{}" },
            ]
        });
        let response = call(&mut plane, "batch_execute", batch);
        let result = &response["result"];
        assert_eq!(result["applied"], 1);
        assert_eq!(result["rejected"], 1);
        assert_eq!(result["aborted"], false);
        assert_eq!(
            plane.world_state()["entities"]
                .as_array()
                .expect("entities")
                .len(),
            1
        );
        assert_eq!(plane.bus().undo_depth(), 1);
    }

    #[test]
    fn snapshot_restore_branches_history() {
        let mut plane = plane();
        let name = call(&mut plane, "history/snapshot", json!({}))["result"]["name"]
            .as_str()
            .expect("auto name")
            .to_owned();
        call(
            &mut plane,
            "batch_execute",
            json!({
                "commands": [{ "tool": "spawn_entity", "args_json": "{\"id\":\"a\"}" }]
            }),
        );
        assert_eq!(
            plane.world_state()["entities"]
                .as_array()
                .expect("entities")
                .len(),
            1
        );
        let response = call(&mut plane, "history/restore", json!({ "name": name }));
        assert_eq!(response["result"]["restored"], 1);
        assert_eq!(
            plane.world_state()["entities"]
                .as_array()
                .expect("entities")
                .len(),
            0
        );
        let response = call(&mut plane, "history/restore", json!({ "name": "nope" }));
        assert_eq!(response["error"]["code"], INVALID_PARAMS);
    }

    #[test]
    fn world_state_reflects_components() {
        let mut plane = plane();
        // Two separate atomic batches: the spawn is one undo unit, the set
        // another — undoing the second restores the previous value.
        call(
            &mut plane,
            "batch_execute",
            json!({
                "commands": [
                    { "tool": "spawn_entity", "args_json": "{\"id\":\"hero\",\"components\":{\"hp\":42,\"tag\":\"knight\",\"alive\":true}}" },
                ]
            }),
        );
        call(
            &mut plane,
            "batch_execute",
            json!({
                "commands": [
                    { "tool": "set_component", "args_json": "{\"id\":\"hero\",\"key\":\"hp\",\"value\":99}" },
                ]
            }),
        );
        let state = plane.world_state();
        assert_eq!(state["name"], "level-1");
        let entities = state["entities"].as_array().expect("entities");
        assert_eq!(entities.len(), 1);
        assert_eq!(entities[0]["id"], "hero");
        assert_eq!(entities[0]["components"]["hp"], 99.0);
        assert_eq!(entities[0]["components"]["tag"], "knight");
        assert_eq!(entities[0]["components"]["alive"], true);
        // Undoing the set restores 42.
        call(&mut plane, "history/undo", json!({}));
        let state = plane.world_state();
        assert_eq!(
            state["entities"][0]["components"]["hp"], 42.0,
            "undo restores the previous component value"
        );
    }

    #[test]
    fn activity_log_records_every_call() {
        let mut plane = plane();
        let _ = call(&mut plane, "tools/list", json!({}));
        let _ = call(
            &mut plane,
            "batch_execute",
            json!({
                "commands": [{ "tool": "spawn_entity", "args_json": "{\"id\":\"a\"}" }]
            }),
        );
        let _ = call(&mut plane, "mystery/method", json!({}));
        let response = call(&mut plane, "activity/log", json!({}));
        let entries = response["result"]["entries"].as_array().expect("entries");
        assert_eq!(entries.len(), 3);
        assert_eq!(entries[0]["method"], "tools/list");
        assert_eq!(entries[0]["ok"], true);
        assert_eq!(entries[2]["method"], "mystery/method");
        assert_eq!(entries[2]["ok"], false, "failures are logged too");
        assert_eq!(entries[1]["summary"], "batch: 1 applied, 0 rejected");

        // limit keeps the tail; seq stays monotonic (the previous
        // activity/log call itself is the newest entry).
        let response = call(&mut plane, "activity/log", json!({ "limit": 1 }));
        let entries = response["result"]["entries"].as_array().expect("entries");
        assert_eq!(entries.len(), 1);
        assert_eq!(entries[0]["seq"], 4);
    }

    #[test]
    fn broken_envelopes_get_json_rpc_errors() {
        let mut plane = plane();
        let response: Value = serde_json::from_str(&plane.handle("not json")).expect("response");
        assert_eq!(response["error"]["code"], -32700);
        let response: Value =
            serde_json::from_str(&plane.handle(r#"{"jsonrpc":"2.0"}"#)).expect("response");
        assert_eq!(response["error"]["code"], -32600);
        let response = call(&mut plane, "no/such/method", json!({}));
        assert_eq!(response["error"]["code"], -32601);
        let response = call(&mut plane, "batch_execute", json!({}));
        assert_eq!(response["error"]["code"], INVALID_PARAMS);
        // The unparseable calls are in the activity log too.
        let response = call(&mut plane, "activity/log", json!({}));
        let entries = response["result"]["entries"].as_array().expect("entries");
        assert!(entries.iter().any(|entry| entry["method"] == "?"), "logged");
    }

    #[test]
    fn register_tool_rejects_duplicates_and_bad_schemas() {
        let mut plane = ControlPlane::new();
        let scene = SharedScene::new(RefCell::new(SceneDocument::new("s")));
        assert!(plane.register_tool(TOOL_SPAWN_ENTITY, scene_tools::spawn_factory(&scene)));
        assert!(!plane.register_tool(TOOL_SPAWN_ENTITY, scene_tools::spawn_factory(&scene)));
        let broken = ToolSpec {
            name: "broken",
            description: "bad schema",
            input_schema: "{not json",
        };
        assert!(!plane.register_tool(broken, scene_tools::spawn_factory(&scene)));
        assert!(plane.registry().get("broken").is_none());
    }
}

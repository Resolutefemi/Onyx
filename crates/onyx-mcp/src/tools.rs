//! Self-describing tool registry and batch RPC shapes (`docs/PRD.md` §16).

/// Typed description of one MCP tool, mirroring a `tools/list` entry.
///
/// Every engine subsystem registers its commands here, which is what makes the
/// control plane self-describing: agents discover capabilities at runtime
/// instead of hard-coding them.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ToolSpec {
    /// Stable tool id (snake_case), shared with `EngineCommand::name`.
    pub name: &'static str,
    /// One-line capability description shown to agents.
    pub description: &'static str,
    /// JSON-Schema text describing the tool's `arguments` object.
    pub input_schema: &'static str,
}

/// Registry of every tool the engine exposes to AI agents.
#[derive(Debug, Default)]
pub struct ToolRegistry {
    tools: Vec<ToolSpec>,
}

impl ToolRegistry {
    /// An empty registry.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Register `spec`. Returns `false` and keeps the registry unchanged if a
    /// tool with the same name already exists.
    pub fn register(&mut self, spec: ToolSpec) -> bool {
        if self.get(spec.name).is_some() {
            return false;
        }
        self.tools.push(spec);
        true
    }

    /// All registered tools, in registration order.
    #[must_use]
    pub fn list(&self) -> &[ToolSpec] {
        &self.tools
    }

    /// Look up a tool by id.
    #[must_use]
    pub fn get(&self, name: &str) -> Option<&ToolSpec> {
        self.tools.iter().find(|tool| tool.name == name)
    }

    /// Number of registered tools.
    #[must_use]
    pub fn len(&self) -> usize {
        self.tools.len()
    }

    /// Whether the registry is empty.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.tools.is_empty()
    }

    /// Validate a `batch_execute` request against the registry.
    ///
    /// M3 replaces this with real execution routed through the command bus;
    /// the semantics are fixed now so agents can build against them: with
    /// `atomic` set, the first unknown tool marks the whole batch as aborted.
    #[must_use]
    pub fn validate_batch(&self, commands: &[BatchCommand], atomic: bool) -> BatchResult {
        let mut result = BatchResult::default();
        for command in commands {
            if result.aborted {
                result.rejected += 1;
            } else if self.get(command.tool).is_some() {
                result.applied += 1;
            } else {
                result.rejected += 1;
                if atomic {
                    result.aborted = true;
                }
            }
        }
        result
    }
}

/// One command inside a `batch_execute` call (`docs/PRD.md` §16).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct BatchCommand {
    /// Target tool id; must exist in the [`ToolRegistry`].
    pub tool: &'static str,
    /// JSON-encoded arguments, matching the tool's `input_schema`.
    pub args_json: String,
}

/// Outcome of a `batch_execute` call.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct BatchResult {
    /// Commands applied (in atomic mode: 0 when aborted).
    pub applied: usize,
    /// Commands rejected (unknown tool, schema mismatch, engine refusal).
    pub rejected: usize,
    /// Atomic-mode flag: the batch failed as a whole and nothing was applied.
    pub aborted: bool,
}

impl BatchResult {
    /// Total commands processed.
    #[must_use]
    pub fn total(&self) -> usize {
        self.applied + self.rejected
    }

    /// Whether every command in the batch was applied.
    #[must_use]
    pub fn fully_applied(&self) -> bool {
        !self.aborted && self.rejected == 0
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn spawn_entity() -> ToolSpec {
        ToolSpec {
            name: "spawn_entity",
            description: "Spawn an entity with the given component set",
            input_schema: "{\"type\":\"object\"}",
        }
    }

    #[test]
    fn registry_rejects_duplicate_names() {
        let mut registry = ToolRegistry::new();
        assert!(registry.register(spawn_entity()));
        assert!(!registry.register(spawn_entity()));
        assert_eq!(registry.len(), 1);
        assert_eq!(
            registry.get("spawn_entity").map(|tool| tool.name),
            Some("spawn_entity")
        );
        assert_eq!(registry.get("missing"), None);
    }

    #[test]
    fn batch_validation_tracks_applied_and_rejected() {
        let mut registry = ToolRegistry::new();
        registry.register(spawn_entity());

        let commands = vec![
            BatchCommand {
                tool: "spawn_entity",
                args_json: "{}".to_owned(),
            },
            BatchCommand {
                tool: "spawn_entity",
                args_json: "{}".to_owned(),
            },
            BatchCommand {
                tool: "unknown_tool",
                args_json: "{}".to_owned(),
            },
        ];

        let non_atomic = registry.validate_batch(&commands, false);
        assert_eq!(non_atomic.applied, 2);
        assert_eq!(non_atomic.rejected, 1);
        assert!(!non_atomic.aborted);
        assert!(!non_atomic.fully_applied());

        let atomic = registry.validate_batch(&commands, true);
        assert_eq!(
            atomic.applied, 2,
            "commands before the failure still validate"
        );
        assert_eq!(atomic.rejected, 1);
        assert!(atomic.aborted);
    }
}

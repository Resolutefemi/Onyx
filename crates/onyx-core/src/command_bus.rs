//! Unified Command Bus & History (`docs/PRD.md` §1).
//!
//! Every engine mutation — whether issued by a human editor action or an AI
//! agent through MCP (`docs/PRD.md` §16) — is wrapped in an [`EngineCommand`]
//! and routed through the [`CommandBus`]. That single choke point is what makes
//! shared human/AI undo-redo, transaction grouping and deterministic replay
//! possible later; M0 ships the reversible history itself.

use std::fmt::Debug;

/// A reversible engine mutation.
///
/// Implementors own whatever state they need to apply and revert themselves;
/// the bus only tracks ordering and history. `name` doubles as the MCP tool id
/// once the command is exposed to agents (M3).
pub trait EngineCommand: Debug {
    /// Stable machine-readable command name.
    fn name(&self) -> &'static str;

    /// Apply the mutation. Returns `false` if it was rejected.
    fn apply(&mut self) -> bool;

    /// Revert a previously applied mutation. Returns `false` if it cannot be
    /// reverted (the bus then drops it from the redo stack).
    fn revert(&mut self) -> bool;
}

/// Shared human/AI undo-redo history (PRD §1, "Unified Command Bus & History").
#[derive(Debug, Default)]
pub struct CommandBus {
    applied: Vec<Box<dyn EngineCommand>>,
    undone: Vec<Box<dyn EngineCommand>>,
}

impl CommandBus {
    /// An empty history.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Apply `command` and push it onto the undo stack. A new execution
    /// invalidates the redo branch, mirroring editor undo semantics.
    pub fn execute(&mut self, mut command: Box<dyn EngineCommand>) -> bool {
        if command.apply() {
            self.applied.push(command);
            self.undone.clear();
            true
        } else {
            false
        }
    }

    /// Undo the most recent applied command.
    pub fn undo(&mut self) -> bool {
        match self.applied.pop() {
            Some(mut command) => {
                let reverted = command.revert();
                if reverted {
                    self.undone.push(command);
                }
                reverted
            }
            None => false,
        }
    }

    /// Redo the most recently undone command.
    pub fn redo(&mut self) -> bool {
        match self.undone.pop() {
            Some(mut command) => {
                let applied = command.apply();
                if applied {
                    self.applied.push(command);
                }
                applied
            }
            None => false,
        }
    }

    /// Number of commands available for undo.
    #[must_use]
    pub fn undo_depth(&self) -> usize {
        self.applied.len()
    }

    /// Number of commands available for redo.
    #[must_use]
    pub fn redo_depth(&self) -> usize {
        self.undone.len()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[derive(Debug)]
    struct SetValue {
        value: Option<u32>,
    }

    impl SetValue {
        fn new() -> Self {
            Self { value: None }
        }
    }

    impl EngineCommand for SetValue {
        fn name(&self) -> &'static str {
            "set_value"
        }
        fn apply(&mut self) -> bool {
            self.value = Some(42);
            true
        }
        fn revert(&mut self) -> bool {
            self.value = None;
            true
        }
    }

    #[test]
    fn execute_undo_redo_round_trip() {
        let mut bus = CommandBus::new();
        assert_eq!(bus.undo_depth(), 0);
        assert!(!bus.undo(), "undo on empty history must fail");

        assert!(bus.execute(Box::new(SetValue::new())));
        assert_eq!(bus.undo_depth(), 1);

        assert!(bus.undo());
        assert_eq!(bus.undo_depth(), 0);
        assert_eq!(bus.redo_depth(), 1);

        assert!(bus.redo());
        assert_eq!(bus.undo_depth(), 1);
        assert_eq!(bus.redo_depth(), 0);
    }

    #[test]
    fn new_execution_clears_redo_branch() {
        let mut bus = CommandBus::new();
        bus.execute(Box::new(SetValue::new()));
        bus.undo();
        assert_eq!(bus.redo_depth(), 1);

        bus.execute(Box::new(SetValue::new()));
        assert_eq!(bus.redo_depth(), 0, "redo branch must be invalidated");
        assert_eq!(bus.undo_depth(), 1);
    }
}

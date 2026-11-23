//! Unified Command Bus & History (`docs/PRD.md` §1).
//!
//! Every engine mutation — whether issued by a human editor action or an AI
//! agent through MCP (`docs/PRD.md` §16) — is wrapped in an [`EngineCommand`]
//! and routed through the [`CommandBus`]. That single choke point is what makes
//! shared human/AI undo-redo, transaction grouping and deterministic replay
//! work: commands executed inside a transaction commit as ONE undoable unit,
//! and [`CommandBus::replay_log`] records the applied command names in order.

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
    /// reverted (the bus then drops the unit from the redo stack).
    fn revert(&mut self) -> bool;
}

/// Shared human/AI undo-redo history with transaction grouping
/// (PRD §1, "Unified Command Bus & History").
///
/// History units are groups of commands: a single [`CommandBus::execute`]
/// forms a one-command group; a committed transaction forms one group that a
/// single `undo` reverts in reverse order.
#[derive(Debug, Default)]
pub struct CommandBus {
    /// Undoable units, oldest first; each unit is one command or one
    /// committed transaction.
    applied: Vec<Vec<Box<dyn EngineCommand>>>,
    /// Undone units, newest first.
    undone: Vec<Vec<Box<dyn EngineCommand>>>,
    /// Commands accumulated inside an open transaction.
    pending: Vec<Box<dyn EngineCommand>>,
    /// Whether a transaction window is open.
    transaction_open: bool,
}

impl CommandBus {
    /// An empty history.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Apply `command`. Outside a transaction it becomes its own undoable
    /// unit and invalidates the redo branch; inside a transaction it joins
    /// the pending group.
    pub fn execute(&mut self, mut command: Box<dyn EngineCommand>) -> bool {
        if command.apply() {
            self.undone.clear();
            if self.transaction_open {
                self.pending.push(command);
            } else {
                self.applied.push(vec![command]);
            }
            true
        } else {
            false
        }
    }

    /// Open a transaction: subsequent [`CommandBus::execute`] calls accumulate
    /// into one undoable unit until [`CommandBus::commit_transaction`] or
    /// [`CommandBus::rollback_transaction`].
    ///
    /// # Panics
    /// If a transaction is already open (nesting is not supported).
    pub fn begin_transaction(&mut self) {
        assert!(
            !self.transaction_open,
            "nested transactions are not supported"
        );
        self.transaction_open = true;
    }

    /// Commit the open transaction as a single undoable unit. Returns the
    /// number of commands committed (an empty transaction pushes nothing).
    ///
    /// # Panics
    /// If no transaction is open.
    pub fn commit_transaction(&mut self) -> usize {
        assert!(self.transaction_open, "no transaction open");
        self.transaction_open = false;
        let group = std::mem::take(&mut self.pending);
        let count = group.len();
        if count > 0 {
            self.applied.push(group);
        }
        count
    }

    /// Revert and discard the open transaction, in reverse application order.
    /// Returns the number of commands reverted.
    ///
    /// # Panics
    /// If no transaction is open.
    pub fn rollback_transaction(&mut self) -> usize {
        assert!(self.transaction_open, "no transaction open");
        self.transaction_open = false;
        let mut group = std::mem::take(&mut self.pending);
        let mut reverted = 0;
        while let Some(mut command) = group.pop() {
            if command.revert() {
                reverted += 1;
            }
        }
        reverted
    }

    /// Undo the most recent unit (single command or committed transaction) in
    /// reverse order. Returns `false` when the history is empty or a revert
    /// failed (the unit is then discarded).
    pub fn undo(&mut self) -> bool {
        let Some(mut group) = self.applied.pop() else {
            return false;
        };
        let mut ok = true;
        while let Some(mut command) = group.pop() {
            ok &= command.revert();
        }
        if ok {
            self.undone.push(group);
            true
        } else {
            false
        }
    }

    /// Redo the most recently undone unit, re-applying commands in execution
    /// order. Returns `false` when there is nothing to redo or an apply
    /// failed (the unit is then discarded).
    pub fn redo(&mut self) -> bool {
        let Some(mut group) = self.undone.pop() else {
            return false;
        };
        let mut ok = true;
        for command in &mut group {
            ok &= command.apply();
        }
        if ok {
            self.applied.push(group);
            true
        } else {
            false
        }
    }

    /// Number of undoable units available.
    #[must_use]
    pub fn undo_depth(&self) -> usize {
        self.applied.len()
    }

    /// Number of redoable units available.
    #[must_use]
    pub fn redo_depth(&self) -> usize {
        self.undone.len()
    }

    /// Names of every applied command, in execution order — the deterministic
    /// replay log (PRD §1).
    #[must_use]
    pub fn replay_log(&self) -> Vec<&'static str> {
        self.applied
            .iter()
            .flatten()
            .map(|command| command.name())
            .collect()
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

    #[test]
    fn transaction_commits_as_single_undo_unit() {
        let mut bus = CommandBus::new();
        bus.begin_transaction();
        assert!(bus.execute(Box::new(SetValue::new())));
        assert!(bus.execute(Box::new(SetValue::new())));
        assert_eq!(bus.undo_depth(), 0, "pending commands are not yet history");

        assert_eq!(bus.commit_transaction(), 2);
        assert_eq!(bus.undo_depth(), 1, "the whole transaction is one unit");
        assert_eq!(bus.replay_log(), vec!["set_value", "set_value"]);

        assert!(bus.undo());
        assert_eq!(bus.undo_depth(), 0);
        assert_eq!(bus.redo_depth(), 1);
        assert!(bus.redo());
        assert_eq!(bus.undo_depth(), 1);
    }

    #[test]
    fn rollback_discards_pending_commands() {
        let mut bus = CommandBus::new();
        bus.begin_transaction();
        assert!(bus.execute(Box::new(SetValue::new())));
        assert_eq!(bus.rollback_transaction(), 1);
        assert_eq!(bus.undo_depth(), 0);
        assert_eq!(bus.redo_depth(), 0);
        assert!(bus.replay_log().is_empty());
    }

    #[test]
    fn empty_transaction_commits_nothing() {
        let mut bus = CommandBus::new();
        bus.begin_transaction();
        assert_eq!(bus.commit_transaction(), 0);
        assert_eq!(bus.undo_depth(), 0);
    }

    #[test]
    #[should_panic(expected = "nested transactions are not supported")]
    fn nested_transactions_are_rejected() {
        let mut bus = CommandBus::new();
        bus.begin_transaction();
        bus.begin_transaction();
    }
}

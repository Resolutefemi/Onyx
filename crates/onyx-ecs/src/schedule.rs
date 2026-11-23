//! Schedule stages (`docs/PRD.md` §1, "system sets" and "lock-free parallel
//! system scheduler").
//!
//! M0 fixes the canonical stage order; M1 resolves per-stage system sets into
//! a parallel dependency graph on top of the work-stealing pool.

/// Canonical execution stages, in dispatch order.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum Stage {
    /// Setup and input ingestion.
    First,
    /// Deterministic fixed-rate simulation ticks (physics, gameplay rules).
    FixedUpdate,
    /// Per-frame preparation (animation sampling, culling lists).
    PreUpdate,
    /// Main gameplay systems.
    Update,
    /// Presentation-adjacent work (render extraction, UI sync).
    PostUpdate,
    /// Teardown and end-of-frame bookkeeping.
    Last,
}

impl Stage {
    /// Stable machine-readable stage name.
    #[must_use]
    pub fn as_str(&self) -> &'static str {
        match self {
            Self::First => "first",
            Self::FixedUpdate => "fixed_update",
            Self::PreUpdate => "pre_update",
            Self::Update => "update",
            Self::PostUpdate => "post_update",
            Self::Last => "last",
        }
    }
}

/// An ordered list of [`Stage`]s forming a schedule.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct Schedule {
    stages: Vec<Stage>,
}

impl Schedule {
    /// The canonical engine schedule: every stage in dispatch order.
    #[must_use]
    pub fn standard() -> Self {
        Self {
            stages: vec![
                Stage::First,
                Stage::FixedUpdate,
                Stage::PreUpdate,
                Stage::Update,
                Stage::PostUpdate,
                Stage::Last,
            ],
        }
    }

    /// Stages in dispatch order.
    #[must_use]
    pub fn stages(&self) -> &[Stage] {
        &self.stages
    }

    /// Dispatch position of `stage`, if present.
    #[must_use]
    pub fn position_of(&self, stage: Stage) -> Option<usize> {
        self.stages.iter().position(|s| *s == stage)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn standard_schedule_orders_all_stages() {
        let schedule = Schedule::standard();
        assert_eq!(schedule.stages().len(), 6);
        assert_eq!(schedule.position_of(Stage::FixedUpdate), Some(1));
        assert!(
            schedule.stages().windows(2).all(|pair| pair[0] < pair[1]),
            "stages must ascend in dispatch order"
        );
    }
}

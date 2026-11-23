//! System definition, access declaration and the stage scheduler
//! (`docs/PRD.md` §1, "system sets" and "lock-free parallel system scheduler").
//!
//! Systems declare what they read and write via [`SystemAccess`]; the
//! schedule rejects two systems in the same stage whose access overlaps on a
//! write (write/write or write/read). The same declarations will drive the
//! parallel executor — independent systems dispatched across the work-stealing
//! pool — in the next M1 increment.

use std::any::TypeId;
use std::collections::HashMap;
use std::fmt;

use crate::component::Component;
use crate::world::World;

/// What a system reads and writes, declared up front.
///
/// Reads and writes are recorded as `(TypeId, type name)` pairs so conflict
/// errors can name the offending component.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct SystemAccess {
    reads: Vec<(TypeId, &'static str)>,
    writes: Vec<(TypeId, &'static str)>,
}

impl SystemAccess {
    /// Empty access (touching no components).
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Declare that the system reads component `T`.
    #[must_use]
    pub fn read<T: Component>(mut self) -> Self {
        let entry = (TypeId::of::<T>(), std::any::type_name::<T>());
        if !self.reads.contains(&entry) {
            self.reads.push(entry);
        }
        self
    }

    /// Declare that the system writes component `T` (implies read).
    #[must_use]
    pub fn write<T: Component>(mut self) -> Self {
        let entry = (TypeId::of::<T>(), std::any::type_name::<T>());
        if !self.writes.contains(&entry) {
            self.writes.push(entry);
        }
        self
    }

    /// First component (by name) on which `self` conflicts with `other`:
    /// write/write or write/read overlap. Read/read never conflicts.
    #[must_use]
    pub fn conflicts(&self, other: &SystemAccess) -> Option<&'static str> {
        for (id, name) in &self.writes {
            if other.writes.iter().any(|(oid, _)| oid == id) {
                return Some(name);
            }
            if other.reads.iter().any(|(oid, _)| oid == id) {
                return Some(name);
            }
        }
        for (id, name) in &self.reads {
            if other.writes.iter().any(|(oid, _)| oid == id) {
                return Some(name);
            }
        }
        None
    }
}

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

/// A unit of work run by the scheduler.
pub trait System: Send {
    /// Stable human-readable system name.
    fn name(&self) -> &'static str;

    /// Declared component access (conflict detection now, parallel dispatch
    /// in the next M1 increment).
    fn access(&self) -> SystemAccess;

    /// Execute the system against the world.
    fn run(&mut self, world: &mut World);
}

/// Closure-backed [`System`].
#[allow(clippy::type_complexity)]
pub struct FnSystem<F: FnMut(&mut World) + Send> {
    name: &'static str,
    access: SystemAccess,
    run: F,
}

impl<F: FnMut(&mut World) + Send + 'static> System for FnSystem<F> {
    fn name(&self) -> &'static str {
        self.name
    }

    fn access(&self) -> SystemAccess {
        self.access.clone()
    }

    fn run(&mut self, world: &mut World) {
        (self.run)(world);
    }
}

/// Build a [`System`] from a closure with declared access.
#[must_use]
#[allow(clippy::type_complexity)]
pub fn system<F>(name: &'static str, access: SystemAccess, run: F) -> FnSystem<F>
where
    F: FnMut(&mut World) + Send + 'static,
{
    FnSystem { name, access, run }
}

/// Two systems in the same stage declare conflicting access.
#[derive(Debug)]
pub struct ScheduleConflictError {
    /// Stage where the conflict occurred.
    pub stage: Stage,
    /// Name of the already-registered system.
    pub first: &'static str,
    /// Name of the rejected system.
    pub second: &'static str,
    /// Component type (debug name) they conflict on.
    pub component: &'static str,
}

impl fmt::Display for ScheduleConflictError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "systems '{}' and '{}' conflict on component '{}' in stage '{}'",
            self.first,
            self.second,
            self.component,
            self.stage.as_str()
        )
    }
}

impl std::error::Error for ScheduleConflictError {}

/// Stage-ordered system schedule (PRD §1). Systems run in canonical stage
/// order; within a stage, access conflicts are rejected at registration time
/// so the ordering is always well-defined (and later, parallelizable).
#[derive(Default)]
pub struct Schedule {
    stages: Vec<Stage>,
    systems: HashMap<Stage, Vec<Box<dyn System>>>,
}

impl Schedule {
    /// An empty schedule with no stages.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

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
            systems: HashMap::new(),
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

    /// Total number of registered systems across all stages.
    #[must_use]
    pub fn system_count(&self) -> usize {
        self.systems.values().map(Vec::len).sum()
    }

    fn ensure_stage(&mut self, stage: Stage) {
        if !self.stages.contains(&stage) {
            self.stages.push(stage);
            self.stages.sort_unstable();
        }
    }

    /// Register `system` in `stage`, rejecting same-stage access conflicts.
    pub fn try_add_system(
        &mut self,
        stage: Stage,
        system: Box<dyn System>,
    ) -> Result<(), ScheduleConflictError> {
        if let Some(existing) = self.systems.get(&stage) {
            let access = system.access();
            for other in existing {
                if let Some(component) = access.conflicts(&other.access()) {
                    return Err(ScheduleConflictError {
                        stage,
                        first: other.name(),
                        second: system.name(),
                        component,
                    });
                }
            }
        }
        self.ensure_stage(stage);
        self.systems.entry(stage).or_default().push(system);
        Ok(())
    }

    /// Panicking variant of [`Schedule::try_add_system`].
    ///
    /// # Panics
    /// If `system` conflicts with another system already registered in
    /// `stage`.
    pub fn add_system(&mut self, stage: Stage, system: Box<dyn System>) -> &mut Self {
        if let Err(error) = self.try_add_system(stage, system) {
            panic!("conflicting system registration: {error}");
        }
        self
    }

    /// Builder variant: `Schedule::standard().with_system(Stage::Update, my_system)`.
    ///
    /// # Panics
    /// Same as [`Schedule::add_system`].
    #[must_use]
    pub fn with_system(mut self, stage: Stage, system: impl System + 'static) -> Self {
        self.add_system(stage, Box::new(system));
        self
    }

    /// Run every registered system in canonical stage order.
    pub fn run(&mut self, world: &mut World) {
        for stage in &self.stages {
            if let Some(systems) = self.systems.get_mut(stage) {
                for system in systems {
                    system.run(world);
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::{Arc, Mutex};

    struct Alpha;
    struct Beta;

    fn noop_system(name: &'static str, access: SystemAccess) -> Box<dyn System> {
        Box::new(system(name, access, |_world: &mut World| {}))
    }

    #[test]
    fn standard_schedule_orders_all_stages() {
        let schedule = Schedule::standard();
        assert_eq!(schedule.stages().len(), 6);
        assert_eq!(schedule.position_of(Stage::FixedUpdate), Some(1));
        assert!(
            schedule.stages().windows(2).all(|pair| pair[0] < pair[1]),
            "stages must ascend in dispatch order"
        );
        assert_eq!(schedule.system_count(), 0);
    }

    #[test]
    fn conflicting_writes_are_rejected() {
        let mut schedule = Schedule::new();
        schedule
            .try_add_system(
                Stage::Update,
                noop_system("writer_a", SystemAccess::new().write::<Alpha>()),
            )
            .expect("first registration");

        let error = schedule
            .try_add_system(
                Stage::Update,
                noop_system("writer_b", SystemAccess::new().write::<Alpha>()),
            )
            .expect_err("write/write conflict must be rejected");
        assert_eq!(error.first, "writer_a");
        assert_eq!(error.second, "writer_b");
        assert!(error.to_string().contains("Alpha"));
        assert_eq!(schedule.system_count(), 1);
    }

    #[test]
    fn write_read_conflicts_but_read_read_does_not() {
        let mut schedule = Schedule::new();
        schedule
            .try_add_system(
                Stage::Update,
                noop_system("alpha_reader_a", SystemAccess::new().read::<Alpha>()),
            )
            .expect("registered");

        schedule
            .try_add_system(
                Stage::Update,
                noop_system("alpha_reader_b", SystemAccess::new().read::<Alpha>()),
            )
            .expect("read/read must be allowed");
        assert!(
            schedule
                .try_add_system(
                    Stage::Update,
                    noop_system("alpha_writer", SystemAccess::new().write::<Alpha>()),
                )
                .is_err(),
            "write/read conflict must be rejected"
        );
        schedule
            .try_add_system(
                Stage::Update,
                noop_system("beta_writer", SystemAccess::new().write::<Beta>()),
            )
            .expect("disjoint components must be allowed");
        assert!(
            schedule
                .try_add_system(
                    Stage::Update,
                    noop_system("beta_reader", SystemAccess::new().read::<Beta>()),
                )
                .is_err(),
            "read of a written component must be rejected"
        );
        assert_eq!(schedule.system_count(), 3);
    }

    #[test]
    fn run_executes_stages_in_canonical_order() {
        let order = Arc::new(Mutex::new(Vec::new()));
        let mut schedule = Schedule::new();

        let late = Arc::clone(&order);
        schedule
            .try_add_system(
                Stage::Last,
                Box::new(system(
                    "late",
                    SystemAccess::new(),
                    move |_world: &mut World| {
                        late.lock().expect("lock").push("late");
                    },
                )),
            )
            .expect("registered");

        let early = Arc::clone(&order);
        schedule
            .try_add_system(
                Stage::Update,
                Box::new(system(
                    "early",
                    SystemAccess::new(),
                    move |_world: &mut World| {
                        early.lock().expect("lock").push("early");
                    },
                )),
            )
            .expect("registered");

        let mut world = World::new();
        schedule.run(&mut world);
        assert_eq!(*order.lock().expect("lock"), vec!["early", "late"]);
    }
}

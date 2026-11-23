//! System definition, access declaration and the stage scheduler
//! (`docs/PRD.md` §1, "system sets" and "lock-free parallel system scheduler").
//!
//! Systems declare what they read and write via [`SystemAccess`]; the
//! schedule rejects two systems in the same stage whose access overlaps on a
//! write (write/write or write/read). The same declarations drive the
//! executor: consecutive read-only systems ([`reader_system`]) form parallel
//! batches that run concurrently on scoped threads against a shared
//! [`World`], while writing systems execute one at a time between batches,
//! preserving declaration order.

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

    /// Declared component access: drives same-stage conflict detection and
    /// the reader/writer batching that decides what runs in parallel.
    fn access(&self) -> SystemAccess;

    /// Whether the system executes against a *shared* world and is eligible
    /// for parallel reader batches. Only reader systems built with
    /// [`reader_system`] are readers; the conservative default is `false`
    /// (exclusive access), so writer-style systems with empty access are
    /// never misclassified.
    fn is_reader(&self) -> bool {
        false
    }

    /// Execute the system against the world with exclusive access. Writers
    /// implement this; reader systems built with [`reader_system`] panic here
    /// (the scheduler never calls it on them).
    fn run(&mut self, world: &mut World);

    /// Execute the system against a shared world. Reader systems implement
    /// this and run concurrently inside a batch; writer systems panic here.
    fn run_shared(&mut self, world: &World) {
        let _ = world;
        panic!(
            "system '{}' declares read-only access but only implements run(&mut World); build it with onyx_ecs::reader_system",
            self.name()
        );
    }
}

/// Closure-backed writing [`System`] (`FnMut(&mut World)`).
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

/// Closure-backed read-only [`System`] (`FnMut(&World)`), dispatched in
/// parallel batches by the scheduler.
#[allow(clippy::type_complexity)]
pub struct FnReaderSystem<F: FnMut(&World) + Send> {
    name: &'static str,
    access: SystemAccess,
    run: F,
}

impl<F: FnMut(&World) + Send + 'static> System for FnReaderSystem<F> {
    fn name(&self) -> &'static str {
        self.name
    }

    fn access(&self) -> SystemAccess {
        self.access.clone()
    }

    fn is_reader(&self) -> bool {
        true
    }

    fn run(&mut self, _world: &mut World) {
        panic!(
            "'{}' is a reader system and must not be run with exclusive access",
            self.name
        );
    }

    fn run_shared(&mut self, world: &World) {
        (self.run)(world);
    }
}

/// Build a writing [`System`] from a closure with declared access.
#[must_use]
#[allow(clippy::type_complexity)]
pub fn system<F>(name: &'static str, access: SystemAccess, run: F) -> FnSystem<F>
where
    F: FnMut(&mut World) + Send + 'static,
{
    FnSystem { name, access, run }
}

/// Build a read-only [`System`] from a closure over a shared [`World`].
/// Reader systems run concurrently in scheduler batches; the access
/// declaration must not contain writes.
///
/// # Panics
/// If `access` declares any writes.
#[must_use]
#[allow(clippy::type_complexity)]
pub fn reader_system<F>(name: &'static str, access: SystemAccess, run: F) -> FnReaderSystem<F>
where
    F: FnMut(&World) + Send + 'static,
{
    assert!(
        access.writes.is_empty(),
        "reader_system '{name}' must declare reads only"
    );
    FnReaderSystem { name, access, run }
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

/// Run a run of consecutive reader systems against a shared world. A lone
/// reader executes inline (no thread spawn); larger batches fan out across
/// scoped threads.
fn run_reader_batch(systems: &mut [Box<dyn System>], world: &World) {
    if systems.len() == 1 {
        systems[0].run_shared(world);
        return;
    }
    std::thread::scope(|scope| {
        for system in systems.iter_mut() {
            scope.spawn(move || system.run_shared(world));
        }
    });
}

/// Stage-ordered system schedule (PRD §1). Systems run in canonical stage
/// order; within a stage, access conflicts are rejected at registration time.
/// Runs of consecutive reader systems are dispatched as parallel batches
/// between the writers, whose relative order is preserved.
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
    /// Works for both writers ([`system`]) and readers ([`reader_system`]).
    ///
    /// # Panics
    /// Same as [`Schedule::add_system`].
    #[must_use]
    pub fn with_system(mut self, stage: Stage, system: impl System + 'static) -> Self {
        self.add_system(stage, Box::new(system));
        self
    }

    /// Run every registered system in canonical stage order.
    ///
    /// Consecutive reader systems within a stage form one parallel batch:
    /// they execute concurrently against a shared [`World`] on scoped
    /// threads. Writers run one at a time in declaration order, flushing the
    /// pending batch first. Reader order inside a batch is nondeterministic,
    /// so readers must not have externally ordered side effects.
    pub fn run(&mut self, world: &mut World) {
        for stage in &self.stages {
            let Some(systems) = self.systems.get_mut(stage) else {
                continue;
            };
            let mut batch_start = 0usize;
            for index in 0..systems.len() {
                if !systems[index].is_reader() {
                    if index > batch_start {
                        run_reader_batch(&mut systems[batch_start..index], world);
                    }
                    systems[index].run(world);
                    batch_start = index + 1;
                }
            }
            if batch_start < systems.len() {
                run_reader_batch(&mut systems[batch_start..], world);
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::{Arc, Mutex};

    #[derive(Debug, Clone, Copy, PartialEq)]
    struct Alpha(f32);
    #[derive(Debug, Clone, Copy, PartialEq)]
    struct Beta(f32);

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

    #[test]
    fn reader_batches_run_before_the_next_writer() {
        let log = Arc::new(Mutex::new(Vec::<&'static str>::new()));
        let mut world = World::new();
        let entity = world.spawn();
        world.insert(entity, Alpha(0.0));
        world.insert(entity, Beta(0.0));

        let r1_log = Arc::clone(&log);
        let r2_log = Arc::clone(&log);
        let writer_log = Arc::clone(&log);
        let r3_log = Arc::clone(&log);

        let mut schedule = Schedule::new();
        schedule
            .try_add_system(
                Stage::Update,
                Box::new(reader_system(
                    "r1",
                    SystemAccess::new().read::<Alpha>(),
                    move |_world: &World| {
                        r1_log.lock().expect("lock").push("r1");
                    },
                )),
            )
            .expect("registered");
        schedule
            .try_add_system(
                Stage::Update,
                Box::new(reader_system(
                    "r2",
                    SystemAccess::new().read::<Alpha>(),
                    move |_world: &World| {
                        r2_log.lock().expect("lock").push("r2");
                    },
                )),
            )
            .expect("registered");
        schedule
            .try_add_system(
                Stage::Update,
                Box::new(system(
                    "writer",
                    SystemAccess::new().write::<Beta>(),
                    move |world: &mut World| {
                        writer_log.lock().expect("lock").push("writer");
                        if let Some(beta) = world.get_mut::<Beta>(entity) {
                            beta.0 += 1.0;
                        }
                    },
                )),
            )
            .expect("registered");
        schedule
            .try_add_system(
                Stage::Update,
                Box::new(reader_system(
                    "r3",
                    SystemAccess::new().read::<Alpha>(),
                    move |_world: &World| {
                        r3_log.lock().expect("lock").push("r3");
                    },
                )),
            )
            .expect("registered");

        schedule.run(&mut world);

        let order = log.lock().expect("lock").clone();
        let writer_index = order
            .iter()
            .position(|name| *name == "writer")
            .expect("writer ran");
        let r1_index = order.iter().position(|name| *name == "r1").expect("r1 ran");
        let r2_index = order.iter().position(|name| *name == "r2").expect("r2 ran");
        let r3_index = order.iter().position(|name| *name == "r3").expect("r3 ran");
        assert!(
            r1_index < writer_index && r2_index < writer_index,
            "the pending reader batch must flush before the writer runs"
        );
        assert!(
            r3_index > writer_index,
            "systems declared after the writer must run after it"
        );
        assert_eq!(world.get::<Beta>(entity), Some(&Beta(1.0)));
    }

    #[test]
    fn consecutive_readers_share_one_parallel_batch() {
        let hits = Arc::new(Mutex::new(Vec::<&'static str>::new()));
        let names = ["reader_0", "reader_1", "reader_2", "reader_3"];
        let mut schedule = Schedule::new();
        for name in names {
            let hits_in_system = Arc::clone(&hits);
            schedule
                .try_add_system(
                    Stage::Update,
                    Box::new(reader_system(
                        name,
                        SystemAccess::new().read::<Alpha>(),
                        move |_world: &World| {
                            hits_in_system.lock().expect("lock").push(name);
                        },
                    )),
                )
                .expect("registered");
        }

        let mut world = World::new();
        schedule.run(&mut world);

        let mut ran = hits.lock().expect("lock").clone();
        ran.sort_unstable();
        assert_eq!(ran, vec!["reader_0", "reader_1", "reader_2", "reader_3"]);
    }

    #[test]
    #[should_panic(expected = "must declare reads only")]
    fn reader_system_rejects_write_access() {
        let _ = reader_system(
            "bad",
            SystemAccess::new().write::<Alpha>(),
            |_world: &World| {},
        );
    }
}

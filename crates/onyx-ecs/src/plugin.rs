//! Bridges the ECS into the engine loop (`onyx-core::App`).
//!
//! [`EcsPlugin`] owns a [`World`] and a [`Schedule`] and drives the schedule
//! on the fixed timestep (1/60 s) every frame, so simulation stays
//! deterministic regardless of render framerate (PRD §1 fixed-timestep
//! schedules).

use std::fmt;

use onyx_core::{Plugin, Time, FIXED_TIMESTEP_SECS};

use crate::schedule::Schedule;
use crate::world::World;

/// Owns a [`World`] and a [`Schedule`]; runs the schedule on every fixed tick.
pub struct EcsPlugin {
    world: World,
    schedule: Schedule,
    accumulator: f64,
    ticks_run: usize,
}

impl EcsPlugin {
    /// A plugin with an empty world and the given schedule.
    #[must_use]
    pub fn new(schedule: Schedule) -> Self {
        Self {
            world: World::new(),
            schedule,
            accumulator: 0.0,
            ticks_run: 0,
        }
    }

    /// A plugin over an existing world (pre-populated scenes, tests).
    #[must_use]
    pub fn with_world(world: World, schedule: Schedule) -> Self {
        Self {
            world,
            schedule,
            accumulator: 0.0,
            ticks_run: 0,
        }
    }

    /// Read-only view of the simulated world.
    #[must_use]
    pub fn world(&self) -> &World {
        &self.world
    }

    /// Mutable view of the simulated world (scene setup, tests, MCP bridge).
    pub fn world_mut(&mut self) -> &mut World {
        &mut self.world
    }

    /// Number of fixed ticks simulated so far.
    #[must_use]
    pub fn ticks_run(&self) -> usize {
        self.ticks_run
    }
}

impl Plugin for EcsPlugin {
    fn name(&self) -> &'static str {
        "onyx_ecs"
    }

    fn update(&mut self, time: &Time) {
        self.accumulator += time.delta;
        while self.accumulator >= FIXED_TIMESTEP_SECS {
            self.accumulator -= FIXED_TIMESTEP_SECS;
            self.world.tick();
            self.schedule.run(&mut self.world);
            self.ticks_run += 1;
        }
    }
}

impl fmt::Debug for EcsPlugin {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("EcsPlugin")
            .field("entities", &self.world.entity_count())
            .field("systems", &self.schedule.system_count())
            .field("ticks_run", &self.ticks_run)
            .finish()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::schedule::{system, Stage, SystemAccess};
    use std::sync::{Arc, Mutex};

    #[derive(Debug, Clone, Copy, PartialEq)]
    struct Counter(u32);

    #[test]
    fn plugin_drives_schedule_on_fixed_timestep() {
        let hits = Arc::new(Mutex::new(0usize));
        let hits_in_system = Arc::clone(&hits);
        let schedule = Schedule::standard().with_system(
            Stage::Update,
            system(
                "count_ticks",
                SystemAccess::new(),
                move |_world: &mut World| {
                    *hits_in_system.lock().expect("lock") += 1;
                },
            ),
        );
        let mut app = onyx_core::App::new().with_plugin(EcsPlugin::new(schedule));
        app.set_frame_budget(Some(3));
        app.run(FIXED_TIMESTEP_SECS);
        // One fixed tick per frame (dt == 1/60 s) → three system runs.
        assert_eq!(app.frame(), 3);
        assert_eq!(*hits.lock().expect("lock"), 3);
    }

    #[test]
    fn plugin_accumulates_multiple_ticks_per_frame() {
        let mut world = World::new();
        let entity = world.spawn();
        world.insert(entity, Counter(0));

        let schedule = Schedule::standard().with_system(
            Stage::Update,
            system(
                "increment",
                SystemAccess::new().write::<Counter>(),
                move |world: &mut World| {
                    for (_entity, counter) in world.view_mut::<Counter>() {
                        counter.0 += 1;
                    }
                },
            ),
        );
        let mut plugin = EcsPlugin::with_world(world, schedule);

        let mut time = Time::new();
        time.advance(FIXED_TIMESTEP_SECS * 2.0);
        plugin.update(&time);

        assert_eq!(plugin.ticks_run(), 2);
        assert_eq!(plugin.world().get::<Counter>(entity), Some(&Counter(2)));
    }
}

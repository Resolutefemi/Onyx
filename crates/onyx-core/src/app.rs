//! Minimal application runner and plugin model.
//!
//! The full stage/parallel scheduler arrives with the archetype ECS in M1
//! (`docs/PRD.md` §1, "lock-free parallel system scheduler"). M0 ships the
//! frame loop, the clock wiring and plugin registration so every subsystem
//! crate has a host to attach to.

use crate::time::Time;

/// A unit of engine functionality hooked into the app lifecycle. Plugin-style
/// registration mirrors the Bevy reference architecture (see the architecture
/// reference table in README.md).
pub trait Plugin: std::fmt::Debug {
    /// Stable human-readable plugin name.
    fn name(&self) -> &'static str;

    /// Called once, at registration time, before the loop starts.
    fn build(&mut self, _app: &mut App) {}

    /// Called once per simulated frame, after the clock advanced.
    fn update(&mut self, _time: &Time) {}
}

/// The engine application: owns plugins, the clock and the frame budget.
#[derive(Debug)]
pub struct App {
    name: &'static str,
    plugins: Vec<Box<dyn Plugin>>,
    time: Time,
    /// Headless frame budget; `None` runs until aborted (windowed / MCP-driven
    /// runtimes replace this with real lifecycle control in M3+).
    max_frames: Option<u64>,
}

impl Default for App {
    fn default() -> Self {
        Self::new()
    }
}

impl App {
    /// A fresh app named after the engine.
    #[must_use]
    pub fn new() -> Self {
        Self {
            name: "Onyx",
            plugins: Vec::new(),
            time: Time::new(),
            max_frames: None,
        }
    }

    /// Application name.
    #[must_use]
    pub fn name(&self) -> &'static str {
        self.name
    }

    /// Snapshot of the engine clock.
    #[must_use]
    pub fn time(&self) -> &Time {
        &self.time
    }

    /// Frames simulated so far.
    #[must_use]
    pub fn frame(&self) -> u64 {
        self.time.frame
    }

    /// Builder-style plugin registration (calls [`Plugin::build`] immediately).
    ///
    /// # Panics
    /// Never panics; `build` callbacks are expected to be infallible in M0.
    pub fn add_plugin(&mut self, mut plugin: impl Plugin + 'static) -> &mut Self {
        plugin.build(self);
        self.plugins.push(Box::new(plugin));
        self
    }

    /// Builder variant of [`App::add_plugin`].
    #[must_use]
    pub fn with_plugin(mut self, plugin: impl Plugin + 'static) -> Self {
        self.add_plugin(plugin);
        self
    }

    /// Set the headless frame budget (`None` = unlimited).
    pub fn set_frame_budget(&mut self, max_frames: Option<u64>) -> &mut Self {
        self.max_frames = max_frames;
        self
    }

    /// Simulate exactly one frame.
    pub fn step(&mut self, delta_seconds: f64) {
        self.time.advance(delta_seconds);
        for plugin in &mut self.plugins {
            plugin.update(&self.time);
        }
    }

    /// Run frames until the budget (if any) is exhausted.
    pub fn run(&mut self, delta_seconds: f64) {
        while !self.frame_budget_reached() {
            self.step(delta_seconds);
        }
    }

    fn frame_budget_reached(&self) -> bool {
        matches!(self.max_frames, Some(max) if self.time.frame >= max)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::time::FIXED_TIMESTEP_SECS;

    #[derive(Debug)]
    struct CountPlugin {
        updates: usize,
    }

    impl Plugin for CountPlugin {
        fn name(&self) -> &'static str {
            "count"
        }
        fn update(&mut self, _time: &Time) {
            self.updates += 1;
        }
    }

    #[test]
    fn runs_headless_frame_budget() {
        let mut app = App::new().with_plugin(CountPlugin { updates: 0 });
        app.set_frame_budget(Some(10));
        app.run(FIXED_TIMESTEP_SECS);
        assert_eq!(app.frame(), 10);
        assert!((app.time().elapsed - 10.0 * FIXED_TIMESTEP_SECS).abs() < 1e-9);
    }

    #[test]
    fn plugins_are_built_at_registration() {
        let mut app = App::new();
        assert_eq!(app.name(), "Onyx");
        app.add_plugin(CountPlugin { updates: 0 });
        app.step(1.0 / 30.0);
        assert_eq!(app.frame(), 1);
    }
}

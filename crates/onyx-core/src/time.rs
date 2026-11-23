//! Engine clock model: variable-timestep delta plus fixed-timestep accrual
//! (`docs/PRD.md` §1, "fixed-timestep vs. variable-timestep schedules").

/// Target fixed simulation rate in ticks per second. Gameplay, physics and
/// deterministic replay accumulate against this rate.
pub const FIXED_TIMESTEP_HZ: u32 = 60;

/// Duration of one fixed simulation tick, in seconds.
pub const FIXED_TIMESTEP_SECS: f64 = 1.0 / FIXED_TIMESTEP_HZ as f64;

/// The engine clock. Advanced once per frame by the [`crate::app::App`] runner.
#[derive(Debug, Clone, Copy, PartialEq, Default)]
pub struct Time {
    /// Seconds since engine start.
    pub elapsed: f64,
    /// Delta seconds of the frame currently being simulated.
    pub delta: f64,
    /// Number of frames simulated since engine start.
    pub frame: u64,
}

impl Time {
    /// A clock at time zero.
    #[must_use]
    pub const fn new() -> Self {
        Self {
            elapsed: 0.0,
            delta: 0.0,
            frame: 0,
        }
    }

    /// Advance the clock by `delta_seconds` for one variable-timestep frame.
    pub fn advance(&mut self, delta_seconds: f64) {
        self.elapsed += delta_seconds;
        self.delta = delta_seconds;
        self.frame = self.frame.saturating_add(1);
    }

    /// Feed the frame delta into a fixed-timestep accumulator and return how
    /// many whole ticks should be simulated this frame. Excess remainder stays
    /// in `accumulator`, keeping simulation deterministic regardless of render
    /// framerate.
    pub fn fixed_step_ticks(&self, accumulator: &mut f64) -> u32 {
        *accumulator += self.delta;
        let mut ticks = 0;
        while *accumulator >= FIXED_TIMESTEP_SECS {
            *accumulator -= FIXED_TIMESTEP_SECS;
            ticks += 1;
        }
        ticks
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn advance_counts_frames_and_elapsed() {
        let mut time = Time::new();
        time.advance(0.5);
        time.advance(0.25);
        assert_eq!(time.frame, 2);
        assert!((time.elapsed - 0.75).abs() < 1e-9);
        assert!((time.delta - 0.25).abs() < 1e-9);
    }

    #[test]
    fn fixed_steps_accumulate_deterministically() {
        let mut time = Time::new();
        let mut accumulator = 0.0;
        // A delta exactly equal to the fixed step drains one tick and leaves
        // no remainder.
        for _ in 0..3 {
            time.advance(FIXED_TIMESTEP_SECS);
            assert_eq!(time.fixed_step_ticks(&mut accumulator), 1);
        }
        assert_eq!(accumulator, 0.0);
        // A 2x-step frame drains two ticks.
        time.advance(2.0 * FIXED_TIMESTEP_SECS);
        assert_eq!(time.fixed_step_ticks(&mut accumulator), 2);
        assert!(accumulator.abs() < FIXED_TIMESTEP_SECS);
    }
}

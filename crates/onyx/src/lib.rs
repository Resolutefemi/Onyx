//! # onyx
//!
//! Umbrella crate for the Onyx Engine: it pulls in every subsystem crate,
//! ships the `onyx` CLI (`info`, `sim`, `--version`) and provides the
//! [`run_sim`] benchmark demo. Subsystems stay independently usable — see
//! `docs/ARCHITECTURE.md` for the dependency rules.

use onyx_core::FIXED_TIMESTEP_SECS;
use onyx_ecs::{system, Schedule, Stage, SystemAccess, World};
use onyx_reflect::OnyxReflect;

pub const ENGINE_NAME: &str = "Onyx Engine";

/// Version of the engine workspace this build came from.
pub const ENGINE_VERSION: &str = env!("CARGO_PKG_VERSION");

/// Subsystem crates compiled into this build, with their PRD scope.
/// Rendered by `onyx info` and asserted by tests.
pub const CRATES: &[(&str, &str)] = &[
    (
        "onyx-core",
        "app loop, time, reversible command bus with transactions (PRD S1)",
    ),
    (
        "onyx-ecs",
        "archetype SoA ECS: world, queries, change detection, schedule (PRD S1)",
    ),
    (
        "onyx-reflect",
        "reflection, JSON serialization, type registry (PRD S1)",
    ),
    (
        "onyx-render",
        "wgpu HAL, render graph, PBR + 2D pipelines (PRD S2-S4)",
    ),
    (
        "onyx-mcp",
        "sub-5ms AI control plane, batched atomic RPC (PRD S16)",
    ),
    ("onyx-script", "polyglot WASM component runtime (PRD S13)"),
];

pub use onyx_core;
pub use onyx_ecs;
pub use onyx_mcp;
pub use onyx_reflect;
pub use onyx_render;
pub use onyx_script;

/// Demo simulation component: a body's position.
#[derive(Debug, Clone, Copy, PartialEq, Default, OnyxReflect)]
pub struct SimPosition {
    pub x: f32,
    pub y: f32,
}

/// Demo simulation component: a body's velocity.
#[derive(Debug, Clone, Copy, PartialEq, Default, OnyxReflect)]
pub struct SimVelocity {
    pub vx: f32,
    pub vy: f32,
}

/// Result of [`run_sim`].
#[derive(Debug, Clone, PartialEq)]
pub struct SimReport {
    /// Bodies simulated.
    pub entities: u64,
    /// Frames requested.
    pub frames: u64,
    /// Fixed ticks actually simulated (== frames at dt = 1/60 s).
    pub ticks: u64,
    /// Wall-clock duration of the whole run in milliseconds.
    pub elapsed_ms: f64,
    /// Sum of all x positions after the run (deterministic checksum).
    pub checksum: f64,
    /// JSON serialization of the first body's position (reflection demo).
    pub sample_json: String,
}

/// Run a fixed-timestep motion simulation: `entity_count` bodies integrating
/// velocity into position every 1/60 s tick for `frames` frames.
///
/// Exercises the M1 ECS end-to-end — spawn, insert with archetype migration,
/// schedule execution and `view2_mut` iteration — and reports a deterministic
/// checksum so regressions surface as checksum changes.
pub fn run_sim(entity_count: u32, frames: u64) -> SimReport {
    let start = std::time::Instant::now();
    let mut world = World::new();
    world.register_component::<SimPosition>("sim_position");
    world.register_component::<SimVelocity>("sim_velocity");

    for i in 0..entity_count {
        let entity = world.spawn();
        world.insert(
            entity,
            SimPosition {
                x: i as f32,
                y: (i % 7) as f32,
            },
        );
        world.insert(entity, SimVelocity { vx: 1.0, vy: 0.5 });
    }

    let dt = FIXED_TIMESTEP_SECS as f32;
    let integrate = system(
        "integrate_motion",
        SystemAccess::new()
            .write::<SimPosition>()
            .read::<SimVelocity>(),
        move |world: &mut World| {
            for (_entity, position, velocity) in world.view2_mut::<SimPosition, SimVelocity>() {
                position.x += velocity.vx * dt;
                position.y += velocity.vy * dt;
            }
        },
    );
    let mut schedule = Schedule::standard();
    schedule.add_system(Stage::Update, Box::new(integrate));

    let mut ticks = 0u64;
    for _ in 0..frames {
        world.tick();
        schedule.run(&mut world);
        ticks += 1;
    }

    let checksum: f64 = world
        .view::<SimPosition>()
        .map(|(_entity, position)| position.x as f64)
        .sum();
    let sample_json = world
        .view::<SimPosition>()
        .next()
        .map(|(_entity, position)| onyx_reflect::to_json(position))
        .unwrap_or_else(|| "{}".to_string());
    SimReport {
        entities: u64::from(entity_count),
        frames,
        ticks,
        elapsed_ms: start.elapsed().as_secs_f64() * 1000.0,
        checksum,
        sample_json,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn crate_map_lists_all_subsystems() {
        assert_eq!(CRATES.len(), 6);
        let names: Vec<&str> = CRATES.iter().map(|(name, _)| *name).collect();
        assert!(names.contains(&"onyx-core"));
        assert!(names.contains(&"onyx-ecs"));
        assert!(names.contains(&"onyx-reflect"));
        assert!(names.contains(&"onyx-render"));
        assert!(names.contains(&"onyx-mcp"));
        assert!(names.contains(&"onyx-script"));
    }

    #[test]
    fn version_is_semver() {
        let parts: Vec<&str> = ENGINE_VERSION.split('.').collect();
        assert_eq!(parts.len(), 3);
        assert!(parts.iter().all(|part| part.parse::<u64>().is_ok()));
    }

    #[test]
    fn sim_is_deterministic() {
        let first = run_sim(64, 10);
        let second = run_sim(64, 10);
        assert_eq!(first.ticks, 10);
        assert_eq!(first.entities, 64);
        assert_eq!(
            first.checksum, second.checksum,
            "same inputs, same checksum"
        );
    }

    #[test]
    fn sim_checksum_grows_with_simulation_length() {
        let short = run_sim(16, 2);
        let long = run_sim(16, 6);
        assert!(long.checksum > short.checksum, "bodies only move +x");
    }

    #[test]
    fn sim_sample_json_round_trips() {
        let report = run_sim(8, 10);
        let position: SimPosition =
            onyx_reflect::from_json(&report.sample_json).expect("valid sample JSON");
        // Entity 0 starts at x=0, y=0 and integrates vx=1, vy=0.5 for 10 ticks.
        assert!(
            (position.x - 10.0 / 60.0).abs() < 1e-4,
            "unexpected x: {}",
            position.x
        );
        assert!(
            (position.y - 10.0 * 0.5 / 60.0).abs() < 1e-4,
            "unexpected y: {}",
            position.y
        );
    }

    #[test]
    fn sim_components_export_reflection_schemas() {
        let mut registry = onyx_reflect::TypeRegistry::new();
        assert!(registry.register::<SimPosition>());
        assert!(registry.register::<SimVelocity>());
        assert!(!registry.register::<SimPosition>());
        let json = registry.schemas_json();
        assert!(json.contains("\"type_name\":\"SimPosition\""), "{json}");
        assert!(json.contains("\"name\":\"x\",\"type\":\"f32\""), "{json}");
        assert!(json.contains("\"type_name\":\"SimVelocity\""), "{json}");
    }
}

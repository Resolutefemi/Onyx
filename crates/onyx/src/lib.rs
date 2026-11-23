//! # onyx
//!
//! Umbrella crate for the Onyx Engine: it pulls in every subsystem crate and
//! ships the `onyx` CLI (`info`, `--version`). Subsystems stay independently
//! usable — see `docs/ARCHITECTURE.md` for the dependency rules.

pub const ENGINE_NAME: &str = "Onyx Engine";

/// Version of the engine workspace this build came from.
pub const ENGINE_VERSION: &str = env!("CARGO_PKG_VERSION");

/// Subsystem crates compiled into this build, with their PRD scope.
/// Rendered by `onyx info` and asserted by tests.
pub const CRATES: &[(&str, &str)] = &[
    (
        "onyx-core",
        "app loop, time, unified reversible command bus (PRD S1)",
    ),
    (
        "onyx-ecs",
        "archetype ECS seed: entities, components, schedules (PRD S1)",
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
pub use onyx_render;
pub use onyx_script;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn crate_map_lists_all_subsystems() {
        assert_eq!(CRATES.len(), 5);
        let names: Vec<&str> = CRATES.iter().map(|(name, _)| *name).collect();
        assert!(names.contains(&"onyx-core"));
        assert!(names.contains(&"onyx-ecs"));
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
}

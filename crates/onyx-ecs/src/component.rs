//! Component typing and the runtime type registry (`docs/PRD.md` §1,
//! "Reflection & Serialization Engine").
//!
//! M0 accepts every `Send + Sync + 'static` type as a component so crates can
//! prototype freely. M1 narrows this to types that opt in via
//! `#[derive(OnyxReflect)]`, which also emits the schemas consumed by the
//! Inspector UI and MCP tool descriptions.

use std::any::{Any, TypeId};
use std::collections::HashMap;

/// Marker trait for component data stored in archetype tables.
pub trait Component: Any + Send + Sync {}

// M0: blanket impl keeps prototyping frictionless. M1 replaces it with the
// `#[derive(OnyxReflect)]` opt-in (PRD §1 reflection macros).
impl<T: Any + Send + Sync> Component for T {}

/// Runtime registry mapping component types to stable names.
///
/// Drives schema generation for the Inspector UI and the MCP tool registry
/// (`docs/PRD.md` §16) once reflection lands in M1.
#[derive(Debug, Default)]
pub struct ComponentRegistry {
    names: HashMap<TypeId, &'static str>,
    order: Vec<TypeId>,
}

impl ComponentRegistry {
    /// An empty registry.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Register component type `T` under `name`. Returns `false` if `T` was
    /// already registered.
    pub fn register<T: Component>(&mut self, name: &'static str) -> bool {
        let id = TypeId::of::<T>();
        if self.names.contains_key(&id) {
            return false;
        }
        self.names.insert(id, name);
        self.order.push(id);
        true
    }

    /// Stable name of component type `T`, if registered.
    #[must_use]
    pub fn name_of<T: Component>(&self) -> Option<&'static str> {
        self.names.get(&TypeId::of::<T>()).copied()
    }

    /// Registered component names in registration order (deterministic output
    /// for schemas and logs).
    #[must_use]
    pub fn type_names(&self) -> Vec<&'static str> {
        self.order
            .iter()
            .filter_map(|id| self.names.get(id).copied())
            .collect()
    }

    /// Number of registered component types.
    #[must_use]
    pub fn len(&self) -> usize {
        self.order.len()
    }

    /// Whether the registry is empty.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.order.is_empty()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    struct Position;
    struct Velocity;
    struct Health;

    #[test]
    fn registers_and_reports_component_types() {
        let mut registry = ComponentRegistry::new();
        assert!(registry.register::<Position>("position"));
        assert!(registry.register::<Velocity>("velocity"));
        assert!(
            !registry.register::<Position>("position_again"),
            "duplicate must fail"
        );

        assert_eq!(registry.name_of::<Position>(), Some("position"));
        assert_eq!(registry.name_of::<Health>(), None);
        assert_eq!(registry.len(), 2);
        assert_eq!(registry.type_names(), vec!["position", "velocity"]);
        assert!(!registry.is_empty());
    }
}

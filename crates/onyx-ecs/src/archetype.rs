//! Archetype identity (`docs/PRD.md` §1, "Archetype Entity Component System").
//!
//! An archetype is identified by the ordered set of component types its
//! entities hold. M0 ships the signature type; M1 attaches the SoA storage
//! tables and sparse-set side channels for high-churn tag components.

use std::any::TypeId;

use crate::component::Component;

/// The ordered, de-duplicated set of component types an entity holds.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct ArchetypeSignature {
    type_ids: Vec<TypeId>,
}

impl ArchetypeSignature {
    /// The empty signature (entities with no components).
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Builder: add component type `T`, keeping the set sorted for O(log n)
    /// lookups and stable archetype identity.
    #[must_use]
    pub fn with<T: Component>(mut self) -> Self {
        let id = TypeId::of::<T>();
        if let Err(position) = self.type_ids.binary_search(&id) {
            self.type_ids.insert(position, id);
        }
        self
    }

    /// Whether entities in this archetype hold component type `T`.
    #[must_use]
    pub fn contains<T: Component>(&self) -> bool {
        self.type_ids.binary_search(&TypeId::of::<T>()).is_ok()
    }

    /// Number of component types in the signature.
    #[must_use]
    pub fn len(&self) -> usize {
        self.type_ids.len()
    }

    /// Whether the signature is empty.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.type_ids.is_empty()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    struct Position;
    struct Velocity;
    struct Health;

    #[test]
    fn signature_is_sorted_and_deduplicated() {
        let signature = ArchetypeSignature::new()
            .with::<Velocity>()
            .with::<Position>();
        assert_eq!(signature.len(), 2);
        assert!(signature.contains::<Position>());
        assert!(signature.contains::<Velocity>());
        assert!(!signature.contains::<Health>());

        let readded = signature.clone().with::<Position>();
        assert_eq!(readded.len(), 2, "duplicate insertion must be a no-op");
        assert_eq!(signature, readded, "identity must be order-insensitive");
    }
}

//! Archetype identity and SoA tables (`docs/PRD.md` §1, "Archetype Entity
//! Component System").
//!
//! An archetype is identified by the sorted set of component types its
//! entities hold ([`ArchetypeSignature`]) and stores them as parallel
//! type-erased columns ([`Archetype`]). Entities migrate between archetypes
//! whenever their component set changes; the swapped-in row's handle stays
//! valid because the [`World`](crate::world) patches its location.

use std::any::TypeId;

use crate::column::Column;
use crate::component::Component;
use crate::entity::Entity;

/// The ordered, de-duplicated set of component types an entity holds.
///
/// Sorted insertion gives O(log n) membership tests and a canonical
/// `Hash`/`Eq` identity, so the [`World`](crate::world) can map signatures to
/// archetype slots deterministically.
#[derive(Debug, Clone, Default, PartialEq, Eq, Hash)]
pub struct ArchetypeSignature {
    type_ids: Vec<TypeId>,
}

impl ArchetypeSignature {
    /// The empty signature (entities with no components).
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Builder: add component type `T`, keeping the set sorted.
    #[must_use]
    pub fn with<T: Component>(mut self) -> Self {
        let id = TypeId::of::<T>();
        if let Err(position) = self.type_ids.binary_search(&id) {
            self.type_ids.insert(position, id);
        }
        self
    }

    /// Builder: remove component type `T`, preserving sort order.
    #[must_use]
    pub fn without<T: Component>(mut self) -> Self {
        self.type_ids.retain(|id| *id != TypeId::of::<T>());
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

/// One archetype table: entities sharing an exact component set, stored as
/// parallel type-erased columns (SoA). Column lengths always equal the entity
/// count.
pub struct Archetype {
    signature: ArchetypeSignature,
    /// Columns sorted ascending by `TypeId`: binary-search lookups plus
    /// stable indices, so two columns can be borrowed disjointly via
    /// `split_at_mut`.
    columns: Vec<(TypeId, Box<dyn Column>)>,
    entities: Vec<Entity>,
}

impl Archetype {
    pub(crate) fn new(signature: ArchetypeSignature) -> Self {
        Self {
            signature,
            columns: Vec::new(),
            entities: Vec::new(),
        }
    }

    /// The component set identifying this table.
    #[must_use]
    pub fn signature(&self) -> &ArchetypeSignature {
        &self.signature
    }

    /// Entities in this archetype, row-aligned with every column.
    #[must_use]
    pub fn entities(&self) -> &[Entity] {
        &self.entities
    }

    /// Number of entities (== every column's length).
    #[must_use]
    pub fn len(&self) -> usize {
        self.entities.len()
    }

    /// Whether the table holds no entities.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.entities.is_empty()
    }

    fn column_index(&self, type_id: &TypeId) -> Option<usize> {
        self.columns
            .binary_search_by(|(id, _)| id.cmp(type_id))
            .ok()
    }

    pub(crate) fn column<T: Component>(&self) -> Option<&crate::column::TypedColumn<T>> {
        let index = self.column_index(&TypeId::of::<T>())?;
        self.columns[index]
            .1
            .as_any()
            .downcast_ref::<crate::column::TypedColumn<T>>()
    }

    pub(crate) fn column_mut<T: Component>(
        &mut self,
    ) -> Option<&mut crate::column::TypedColumn<T>> {
        let index = self.column_index(&TypeId::of::<T>())?;
        self.columns[index]
            .1
            .as_any_mut()
            .downcast_mut::<crate::column::TypedColumn<T>>()
    }

    pub(crate) fn column_dynamic(&self, type_id: &TypeId) -> Option<&dyn Column> {
        self.column_index(type_id)
            .map(|index| self.columns[index].1.as_ref())
    }

    pub(crate) fn column_dynamic_mut(&mut self, type_id: &TypeId) -> Option<&mut dyn Column> {
        match self.column_index(type_id) {
            Some(index) => Some(self.columns[index].1.as_mut()),
            None => None,
        }
    }

    /// Simultaneous mutable access to two typed columns: the sorted column
    /// list is split at the higher index, giving provably disjoint borrows.
    pub(crate) fn columns_mut_pair<A: Component, B: Component>(
        &mut self,
    ) -> Option<(
        &mut crate::column::TypedColumn<A>,
        &mut crate::column::TypedColumn<B>,
    )> {
        let index_a = self.column_index(&TypeId::of::<A>())?;
        let index_b = self.column_index(&TypeId::of::<B>())?;
        debug_assert_ne!(index_a, index_b, "distinct component types");
        let (first, second) = if index_a < index_b {
            (index_a, index_b)
        } else {
            (index_b, index_a)
        };
        let (left, right) = self.columns.split_at_mut(second);
        let (a_slot, b_slot) = if index_a < index_b {
            (&mut left[first].1, &mut right[0].1)
        } else {
            (&mut right[0].1, &mut left[first].1)
        };
        Some((
            a_slot
                .as_any_mut()
                .downcast_mut::<crate::column::TypedColumn<A>>()
                .expect("typed column A"),
            b_slot
                .as_any_mut()
                .downcast_mut::<crate::column::TypedColumn<B>>()
                .expect("typed column B"),
        ))
    }

    pub(crate) fn insert_column(&mut self, type_id: TypeId, column: Box<dyn Column>) {
        match self.columns.binary_search_by(|(id, _)| id.cmp(&type_id)) {
            Ok(index) => self.columns[index].1 = column,
            Err(position) => self.columns.insert(position, (type_id, column)),
        }
    }

    pub(crate) fn column_type_ids(&self) -> Vec<TypeId> {
        self.columns.iter().map(|(id, _)| *id).collect()
    }

    pub(crate) fn push_entity(&mut self, entity: Entity) {
        self.entities.push(entity);
    }

    /// Swap-remove the entity row only; column rows are handled by the caller
    /// (migration moves them out first, despawn swaps them in the same step).
    /// Returns the entity that was swapped into `row`, if any.
    pub(crate) fn remove_entity_at(&mut self, row: usize) -> Option<Entity> {
        let swapped = if row + 1 < self.entities.len() {
            Some(self.entities[self.entities.len() - 1])
        } else {
            None
        };
        self.entities.swap_remove(row);
        swapped
    }

    /// Swap-remove one row in every column (used on despawn).
    pub(crate) fn swap_remove_all_columns(&mut self, row: usize) {
        for (_, column) in self.columns.iter_mut() {
            column.swap_remove(row);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::column::{Ticks, TypedColumn};

    struct Position(f32);
    struct Velocity(f32);
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

    #[test]
    fn without_removes_only_the_requested_type() {
        let signature = ArchetypeSignature::new()
            .with::<Position>()
            .with::<Velocity>();
        let reduced = signature.clone().without::<Velocity>();
        assert_eq!(reduced.len(), 1);
        assert!(reduced.contains::<Position>());
        assert!(!reduced.contains::<Velocity>());
        assert_eq!(
            signature.without::<Health>().len(),
            2,
            "removing an absent type must be a no-op"
        );
    }

    #[test]
    fn archetype_rows_stay_aligned_across_migration() {
        let mut a = Archetype::new(ArchetypeSignature::new().with::<Position>());
        let mut b = Archetype::new(
            ArchetypeSignature::new()
                .with::<Position>()
                .with::<Velocity>(),
        );
        let e1 = Entity {
            index: 0,
            generation: 0,
        };
        let e2 = Entity {
            index: 1,
            generation: 0,
        };
        a.push_entity(e1);
        a.push_entity(e2);
        a.insert_column(
            TypeId::of::<Position>(),
            Box::new(TypedColumn::<Position>::new()),
        );
        for _ in 0..2 {
            a.column_mut::<Position>().expect("column").push(
                Position(0.0),
                Ticks {
                    added: 0,
                    changed: 0,
                },
            );
        }

        // Migrate e1 (row 0) into b: value moves, e2's row compacts to 0.
        b.push_entity(e1);
        b.insert_column(
            TypeId::of::<Position>(),
            Box::new(TypedColumn::<Position>::new()),
        );
        let src = a
            .column_dynamic_mut(&TypeId::of::<Position>())
            .expect("column");
        let dst = b
            .column_dynamic_mut(&TypeId::of::<Position>())
            .expect("column");
        src.move_to(dst, 0);
        assert_eq!(a.remove_entity_at(0), Some(e2), "e2 swapped into row 0");

        assert_eq!(a.len(), 1);
        assert_eq!(a.entities()[0], e2);
        assert_eq!(b.len(), 1);
        assert_eq!(b.entities()[0], e1);
        assert!(b.column::<Position>().is_some());
    }

    #[test]
    fn swap_remove_all_columns_compacts_every_column() {
        let mut archetype = Archetype::new(ArchetypeSignature::new().with::<Position>());
        archetype.insert_column(
            TypeId::of::<Position>(),
            Box::new(TypedColumn::<Position>::new()),
        );
        for index in 0..3u32 {
            archetype.push_entity(Entity {
                index,
                generation: 0,
            });
        }
        for _ in 0..3 {
            archetype.column_mut::<Position>().expect("column").push(
                Position(0.0),
                Ticks {
                    added: 0,
                    changed: 0,
                },
            );
        }

        archetype.swap_remove_all_columns(1);
        archetype.remove_entity_at(1);

        assert_eq!(archetype.len(), 2);
        assert_eq!(
            archetype.column::<Position>().expect("column").len(),
            2,
            "column and entity rows must stay aligned"
        );
    }

    #[test]
    fn columns_mut_pair_yields_disjoint_typed_columns() {
        let mut archetype = Archetype::new(
            ArchetypeSignature::new()
                .with::<Position>()
                .with::<Velocity>(),
        );
        archetype.insert_column(
            TypeId::of::<Position>(),
            Box::new(TypedColumn::<Position>::new()),
        );
        archetype.insert_column(
            TypeId::of::<Velocity>(),
            Box::new(TypedColumn::<Velocity>::new()),
        );
        archetype.push_entity(Entity {
            index: 0,
            generation: 0,
        });
        archetype
            .column_mut::<Position>()
            .expect("position column")
            .push(
                Position(0.0),
                Ticks {
                    added: 0,
                    changed: 0,
                },
            );
        archetype
            .column_mut::<Velocity>()
            .expect("velocity column")
            .push(
                Velocity(0.0),
                Ticks {
                    added: 0,
                    changed: 0,
                },
            );

        let (positions, velocities) = archetype
            .columns_mut_pair::<Position, Velocity>()
            .expect("both columns exist");
        positions.data_mut()[0].0 = 7.0;
        velocities.data_mut()[0].0 = 9.0;
        assert_eq!(
            archetype.column::<Position>().expect("col").data()[0].0,
            7.0
        );
        assert_eq!(
            archetype.column::<Velocity>().expect("col").data()[0].0,
            9.0
        );
        assert!(archetype.columns_mut_pair::<Position, Health>().is_none());
    }
}

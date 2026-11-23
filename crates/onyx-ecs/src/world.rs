//! The `World`: archetype-based entity storage, component insertion/removal,
//! queries and change detection (`docs/PRD.md` §1).
//!
//! Storage layout: entities live in [`Archetype`] tables keyed by their exact
//! component set; inserting or removing a component migrates the entity's row
//! between tables (swap-remove compaction keeps columns dense). Handles are
//! generational (`Entity`), so recycling can never alias a stale reference.
//!
//! Change detection: the world owns a monotonically increasing tick
//! ([`World::tick`]); mutations stamp [`Ticks`] into the affected column row.
//! Queries: [`World::view`]/[`World::view2`] (+ `_mut`) iterate archetypes
//! directly — no per-entity filtering — and [`World::iter_changed`],
//! [`World::take_removed`] expose `Changed<T>` / `Removed<T>` semantics.

use std::any::TypeId;
use std::collections::HashMap;

use crate::archetype::{Archetype, ArchetypeSignature};
use crate::column::{Column, Ticks, TypedColumn};
use crate::component::{Component, ComponentRegistry};
use crate::entity::{Entities, Entity};

/// Where an entity's row lives: archetype slot index + row index.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct EntityLocation {
    /// Slot index into `World::archetypes`.
    pub archetype: usize,
    /// Row index inside the archetype (entities + every column).
    pub row: usize,
}

/// Dense archetype-based entity storage.
pub struct World {
    entities: Entities,
    archetypes: Vec<Option<Archetype>>,
    free_archetypes: Vec<usize>,
    signature_index: HashMap<ArchetypeSignature, usize>,
    locations: Vec<Option<EntityLocation>>,
    component_registry: ComponentRegistry,
    current_tick: u64,
    removed_events: HashMap<TypeId, Vec<(Entity, u64)>>,
}

impl Default for World {
    fn default() -> Self {
        Self::new()
    }
}

impl World {
    /// An empty world.
    #[must_use]
    pub fn new() -> Self {
        Self {
            entities: Entities::new(),
            archetypes: Vec::new(),
            free_archetypes: Vec::new(),
            signature_index: HashMap::new(),
            locations: Vec::new(),
            component_registry: ComponentRegistry::new(),
            current_tick: 0,
            removed_events: HashMap::new(),
        }
    }

    /// Advance the world's change tick. Systems run after `tick()`, so their
    /// mutations stamp the new tick and `iter_changed(since)` sees exactly the
    /// mutations of ticks greater than `since`.
    pub fn tick(&mut self) {
        self.current_tick += 1;
    }

    /// The current change tick.
    #[must_use]
    pub fn current_tick(&self) -> u64 {
        self.current_tick
    }

    /// Register a stable display name for component type `T` (used later by
    /// reflection schemas and MCP tool descriptions). Returns `false` if `T`
    /// was already registered.
    pub fn register_component<T: Component>(&mut self, name: &'static str) -> bool {
        self.component_registry.register::<T>(name)
    }

    /// Registered display name of component type `T`.
    #[must_use]
    pub fn component_name<T: Component>(&self) -> Option<&'static str> {
        self.component_registry.name_of::<T>()
    }

    /// Number of live entities.
    #[must_use]
    pub fn entity_count(&self) -> usize {
        self.entities.len()
    }

    /// Number of live archetype tables.
    #[must_use]
    pub fn archetype_count(&self) -> usize {
        self.archetypes
            .iter()
            .filter_map(|slot| slot.as_ref())
            .count()
    }

    /// Whether `entity` refers to a live entity.
    #[must_use]
    pub fn is_alive(&self, entity: Entity) -> bool {
        self.entities.is_alive(entity)
    }

    fn ensure_archetype(&mut self, signature: &ArchetypeSignature) -> usize {
        if let Some(&index) = self.signature_index.get(signature) {
            return index;
        }
        let archetype = Archetype::new(signature.clone());
        let index = match self.free_archetypes.pop() {
            Some(free) => {
                self.archetypes[free] = Some(archetype);
                free
            }
            None => {
                self.archetypes.push(Some(archetype));
                self.archetypes.len() - 1
            }
        };
        self.signature_index.insert(signature.clone(), index);
        index
    }

    /// Return an emptied archetype slot to the free list.
    fn maybe_drop_archetype(&mut self, index: usize) {
        let droppable = self
            .archetypes
            .get(index)
            .is_some_and(|slot| slot.as_ref().is_some_and(Archetype::is_empty));
        if droppable {
            if let Some(archetype) = self.archetypes[index].take() {
                self.signature_index.remove(archetype.signature());
                self.free_archetypes.push(index);
            }
        }
    }

    fn patch_swapped(&mut self, swapped: Option<Entity>, archetype: usize, row: usize) {
        if let Some(moved) = swapped {
            if let Some(location) = self.locations.get_mut(moved.index as usize) {
                *location = Some(EntityLocation { archetype, row });
            }
        }
    }

    /// Spawn a new entity with no components.
    pub fn spawn(&mut self) -> Entity {
        let entity = self.entities.spawn();
        let index = self.ensure_archetype(&ArchetypeSignature::new());
        if self.locations.len() <= entity.index as usize {
            self.locations.resize(entity.index as usize + 1, None);
        }
        let archetype = self.archetypes[index].as_mut().expect("just ensured");
        archetype.push_entity(entity);
        let row = archetype.len() - 1;
        self.locations[entity.index as usize] = Some(EntityLocation {
            archetype: index,
            row,
        });
        entity
    }

    /// Remove an entity and all its components. Stale handles are ignored
    /// (returns `false`).
    pub fn despawn(&mut self, entity: Entity) -> bool {
        if !self.entities.is_alive(entity) {
            return false;
        }
        let location =
            self.locations[entity.index as usize].expect("alive entities have locations");
        let current = self.current_tick;
        let swapped = {
            let archetype = self.archetypes[location.archetype]
                .as_mut()
                .expect("location points at live archetype");
            for type_id in archetype.column_type_ids() {
                self.removed_events
                    .entry(type_id)
                    .or_default()
                    .push((entity, current));
            }
            archetype.swap_remove_all_columns(location.row);
            archetype.remove_entity_at(location.row)
        };
        self.patch_swapped(swapped, location.archetype, location.row);
        self.locations[entity.index as usize] = None;
        debug_assert!(self.entities.kill(entity), "despawn of a live entity");
        self.maybe_drop_archetype(location.archetype);
        true
    }

    /// Attach (or overwrite) component `C` on `entity`, migrating the entity
    /// to the archetype matching the new component set.
    ///
    /// # Panics
    /// If `entity` is stale (despawned or recycled).
    pub fn insert<C: Component>(&mut self, entity: Entity, component: C) {
        if !self.entities.is_alive(entity) {
            panic!("insert on dead/stale entity {entity:?}");
        }
        let current = self.current_tick;
        let location =
            self.locations[entity.index as usize].expect("alive entities have locations");

        // Overwrite in place when the archetype already holds C.
        if self.archetypes[location.archetype]
            .as_ref()
            .is_some_and(|a| a.signature().contains::<C>())
        {
            let archetype = self.archetypes[location.archetype]
                .as_mut()
                .expect("checked above");
            let column = archetype
                .column_mut::<C>()
                .expect("signature contains C, column must exist");
            column.data_mut()[location.row] = component;
            column.ticks_mut()[location.row].changed = current;
            return;
        }

        let target_signature = self.archetypes[location.archetype]
            .as_ref()
            .expect("location points at live archetype")
            .signature()
            .clone()
            .with::<C>();
        let target = self.ensure_archetype(&target_signature);
        debug_assert_ne!(target, location.archetype, "signature must differ");

        let (swapped, target_row) = {
            let (from, to): (&mut Archetype, &mut Archetype) = if location.archetype < target {
                let (left, right) = self.archetypes.split_at_mut(target);
                (
                    left[location.archetype]
                        .as_mut()
                        .expect("live from archetype"),
                    right[0].as_mut().expect("just ensured to archetype"),
                )
            } else {
                let (left, right) = self.archetypes.split_at_mut(location.archetype);
                (
                    right[0].as_mut().expect("live from archetype"),
                    left[target].as_mut().expect("just ensured to archetype"),
                )
            };

            to.push_entity(entity);
            let target_row = to.len() - 1;

            // Create missing destination columns, then migrate the row data.
            let blanks: Vec<(TypeId, Box<dyn Column>)> = from
                .column_type_ids()
                .into_iter()
                .filter_map(|type_id| {
                    from.column_dynamic(&type_id)
                        .map(|column| (type_id, column.blank()))
                })
                .collect();
            for (type_id, blank) in blanks {
                if to.column_dynamic(&type_id).is_none() {
                    to.insert_column(type_id, blank);
                }
            }
            for type_id in from.column_type_ids() {
                let src = from.column_dynamic_mut(&type_id).expect("column exists");
                let dst = to.column_dynamic_mut(&type_id).expect("just created");
                src.move_to(dst, location.row);
            }

            if to.column::<C>().is_none() {
                to.insert_column(TypeId::of::<C>(), Box::new(TypedColumn::<C>::new()));
            }
            to.column_mut::<C>().expect("column just ensured").push(
                component,
                Ticks {
                    added: current,
                    changed: current,
                },
            );

            let swapped = from.remove_entity_at(location.row);
            (swapped, target_row)
        };
        self.locations[entity.index as usize] = Some(EntityLocation {
            archetype: target,
            row: target_row,
        });
        self.patch_swapped(swapped, location.archetype, location.row);
        self.maybe_drop_archetype(location.archetype);
    }

    /// Detach component `C` from `entity`, returning it. `None` if the entity
    /// does not hold `C`.
    ///
    /// # Panics
    /// If `entity` is stale (despawned or recycled).
    pub fn remove<C: Component>(&mut self, entity: Entity) -> Option<C> {
        if !self.entities.is_alive(entity) {
            panic!("remove on dead/stale entity {entity:?}");
        }
        let location =
            self.locations[entity.index as usize].expect("alive entities have locations");
        let (holds_c, target_signature) = {
            let archetype = self.archetypes[location.archetype]
                .as_ref()
                .expect("location points at live archetype");
            (
                archetype.signature().contains::<C>(),
                archetype.signature().clone().without::<C>(),
            )
        };
        if !holds_c {
            return None;
        }
        let target = self.ensure_archetype(&target_signature);
        let (removed, swapped, target_row) = {
            let (from, to): (&mut Archetype, &mut Archetype) = if location.archetype < target {
                let (left, right) = self.archetypes.split_at_mut(target);
                (
                    left[location.archetype]
                        .as_mut()
                        .expect("live from archetype"),
                    right[0].as_mut().expect("just ensured to archetype"),
                )
            } else {
                let (left, right) = self.archetypes.split_at_mut(location.archetype);
                (
                    right[0].as_mut().expect("live from archetype"),
                    left[target].as_mut().expect("just ensured to archetype"),
                )
            };

            to.push_entity(entity);
            let target_row = to.len() - 1;

            let removed = from
                .column_mut::<C>()
                .expect("holds C, column must exist")
                .remove_at(location.row)
                .expect("row valid");

            // Migrate every remaining column into the reduced archetype.
            let others: Vec<TypeId> = from
                .column_type_ids()
                .into_iter()
                .filter(|type_id| *type_id != TypeId::of::<C>())
                .collect();
            let blanks: Vec<(TypeId, Box<dyn Column>)> = others
                .iter()
                .filter_map(|type_id| {
                    from.column_dynamic(type_id)
                        .map(|column| (*type_id, column.blank()))
                })
                .collect();
            for (type_id, blank) in blanks {
                if to.column_dynamic(&type_id).is_none() {
                    to.insert_column(type_id, blank);
                }
            }
            for type_id in &others {
                let src = from.column_dynamic_mut(type_id).expect("column exists");
                let dst = to.column_dynamic_mut(type_id).expect("just created");
                src.move_to(dst, location.row);
            }

            let swapped = from.remove_entity_at(location.row);
            (removed, swapped, target_row)
        };
        self.locations[entity.index as usize] = Some(EntityLocation {
            archetype: target,
            row: target_row,
        });
        self.patch_swapped(swapped, location.archetype, location.row);
        self.maybe_drop_archetype(location.archetype);
        Some(removed)
    }

    /// Read component `C` of `entity`.
    #[must_use]
    pub fn get<C: Component>(&self, entity: Entity) -> Option<&C> {
        let location = self
            .locations
            .get(entity.index as usize)
            .copied()
            .flatten()?;
        let archetype = self.archetypes.get(location.archetype)?.as_ref()?;
        archetype
            .column::<C>()
            .and_then(|column| column.data().get(location.row))
    }

    /// Mutate component `C` of `entity`, stamping the current tick as changed.
    pub fn get_mut<C: Component>(&mut self, entity: Entity) -> Option<&mut C> {
        let location = self
            .locations
            .get(entity.index as usize)
            .copied()
            .flatten()?;
        let current = self.current_tick;
        let archetype = self.archetypes.get_mut(location.archetype)?.as_mut()?;
        if !archetype.signature().contains::<C>() {
            return None;
        }
        let column = archetype.column_mut::<C>().expect("signature contains C");
        column.ticks_mut()[location.row].changed = current;
        column.data_mut().get_mut(location.row)
    }

    /// Whether `entity` holds component `C`.
    #[must_use]
    pub fn has<C: Component>(&self, entity: Entity) -> bool {
        self.get::<C>(entity).is_some()
    }

    /// Iterate every entity holding component `A`.
    pub fn view<'w, A: Component>(&'w self) -> impl Iterator<Item = (Entity, &'w A)> + 'w {
        self.archetypes
            .iter()
            .flatten()
            .filter(|archetype| archetype.signature().contains::<A>())
            .flat_map(|archetype| {
                let column = archetype.column::<A>().expect("signature contains A");
                archetype
                    .entities()
                    .iter()
                    .copied()
                    .zip(column.data().iter())
            })
    }

    /// Mutably iterate every entity holding component `A`; each row's `changed`
    /// tick is stamped with the current tick (write-intent semantics).
    pub fn view_mut<'w, A: Component>(
        &'w mut self,
    ) -> impl Iterator<Item = (Entity, &'w mut A)> + 'w {
        let current = self.current_tick;
        self.archetypes
            .iter_mut()
            .flatten()
            .filter(|archetype| archetype.signature().contains::<A>())
            .flat_map(move |archetype| {
                let entities: Vec<Entity> = archetype.entities().to_vec();
                let column = archetype.column_mut::<A>().expect("signature contains A");
                for ticks in column.ticks_mut() {
                    ticks.changed = current;
                }
                entities.into_iter().zip(column.data_mut().iter_mut())
            })
    }

    /// Iterate every entity holding both `A` and `B`.
    pub fn view2<'w, A: Component, B: Component>(
        &'w self,
    ) -> impl Iterator<Item = (Entity, &'w A, &'w B)> + 'w {
        self.archetypes
            .iter()
            .flatten()
            .filter(|archetype| {
                archetype.signature().contains::<A>() && archetype.signature().contains::<B>()
            })
            .flat_map(|archetype| {
                let a = archetype.column::<A>().expect("contains A").data();
                let b = archetype.column::<B>().expect("contains B").data();
                let entities = archetype.entities();
                entities
                    .iter()
                    .copied()
                    .zip(a.iter())
                    .zip(b.iter())
                    .map(|((entity, a), b)| (entity, a, b))
            })
    }

    /// Mutably iterate every entity holding both `A` and `B`; both columns'
    /// `changed` ticks are stamped with the current tick.
    pub fn view2_mut<'w, A: Component, B: Component>(
        &'w mut self,
    ) -> impl Iterator<Item = (Entity, &'w mut A, &'w mut B)> + 'w {
        let current = self.current_tick;
        self.archetypes
            .iter_mut()
            .flatten()
            .filter(|archetype| {
                archetype.signature().contains::<A>() && archetype.signature().contains::<B>()
            })
            .flat_map(move |archetype| {
                let entities: Vec<Entity> = archetype.entities().to_vec();
                let (a_column, b_column) = archetype
                    .columns_mut_pair::<A, B>()
                    .expect("filtered to contain both");
                for ticks in a_column.ticks_mut() {
                    ticks.changed = current;
                }
                for ticks in b_column.ticks_mut() {
                    ticks.changed = current;
                }
                entities
                    .into_iter()
                    .zip(a_column.data_mut().iter_mut())
                    .zip(b_column.data_mut().iter_mut())
                    .map(|((entity, a), b)| (entity, a, b))
            })
    }

    /// Iterate entities whose component `C` changed after `since_tick`
    /// (`Changed<T>` semantics, PRD §1).
    pub fn iter_changed<'w, C: Component>(
        &'w self,
        since_tick: u64,
    ) -> impl Iterator<Item = (Entity, &'w C)> + 'w {
        self.archetypes
            .iter()
            .flatten()
            .filter(|archetype| archetype.signature().contains::<C>())
            .flat_map(move |archetype| {
                let column = archetype.column::<C>().expect("contains C");
                let data = column.data();
                let ticks = column.ticks();
                archetype
                    .entities()
                    .iter()
                    .copied()
                    .zip(data.iter())
                    .zip(ticks.iter())
                    .filter(move |entry| entry.1.changed > since_tick)
                    .map(|entry| (entry.0 .0, entry.0 .1))
            })
    }

    /// Drain the removal log for component type `C`: every entity that had `C`
    /// removed or despawned since the last drain, with the tick it happened
    /// (`Removed<T>` semantics, PRD §1).
    pub fn take_removed<C: Component>(&mut self) -> Vec<(Entity, u64)> {
        self.removed_events
            .remove(&TypeId::of::<C>())
            .unwrap_or_default()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[derive(Debug, Clone, Copy, PartialEq)]
    struct Position {
        x: f32,
        y: f32,
    }

    #[derive(Debug, Clone, Copy, PartialEq)]
    struct Velocity {
        dx: f32,
        dy: f32,
    }

    #[test]
    fn spawn_insert_query_round_trip() {
        let mut world = World::new();
        let e1 = world.spawn();
        world.insert(e1, Position { x: 1.0, y: 2.0 });
        let e2 = world.spawn();
        world.insert(e2, Position { x: 3.0, y: 4.0 });
        world.insert(e2, Velocity { dx: 0.5, dy: -0.5 });

        assert_eq!(
            world.get::<Position>(e1),
            Some(&Position { x: 1.0, y: 2.0 })
        );
        assert_eq!(world.entity_count(), 2);
        assert_eq!(
            world.archetype_count(),
            2,
            "Position and Position+Velocity archetypes"
        );
        assert_eq!(world.view::<Position>().count(), 2);
        assert_eq!(world.view2::<Position, Velocity>().count(), 1);
        assert!(world
            .view2::<Position, Velocity>()
            .all(|(entity, _, _)| entity == e2));
    }

    #[test]
    fn overwrite_keeps_row_and_bumps_changed() {
        let mut world = World::new();
        world.tick();
        let entity = world.spawn();
        world.insert(entity, Position { x: 1.0, y: 0.0 });
        let before = world.current_tick();

        world.tick();
        world.insert(entity, Position { x: 9.0, y: 0.0 });

        assert_eq!(
            world.get::<Position>(entity),
            Some(&Position { x: 9.0, y: 0.0 })
        );
        assert_eq!(world.view::<Position>().count(), 1);
        assert_eq!(world.archetype_count(), 1, "overwrite must not migrate");
        let changed: Vec<Entity> = world
            .iter_changed::<Position>(before)
            .map(|(entity, _)| entity)
            .collect();
        assert_eq!(changed, vec![entity], "overwrite must stamp changed tick");
    }

    #[test]
    fn remove_migrates_between_archetypes() {
        let mut world = World::new();
        let e1 = world.spawn();
        world.insert(e1, Position { x: 1.0, y: 0.0 });
        world.insert(e1, Velocity { dx: 1.0, dy: 1.0 });
        let e2 = world.spawn();
        world.insert(e2, Position { x: 2.0, y: 0.0 });

        let removed = world.remove::<Velocity>(e1);
        assert_eq!(removed, Some(Velocity { dx: 1.0, dy: 1.0 }));
        assert!(!world.has::<Velocity>(e1));
        assert!(world.has::<Position>(e1));
        assert_eq!(world.view::<Position>().count(), 2);
        assert_eq!(world.view2::<Position, Velocity>().count(), 0);
        assert_eq!(
            world.archetype_count(),
            1,
            "emptied archetype must be dropped"
        );
    }

    #[test]
    fn despawn_patches_swapped_rows_and_logs_removal() {
        let mut world = World::new();
        let e1 = world.spawn();
        world.insert(e1, Position { x: 1.0, y: 0.0 });
        let e2 = world.spawn();
        world.insert(e2, Position { x: 2.0, y: 0.0 });
        let e3 = world.spawn();
        world.insert(e3, Position { x: 3.0, y: 0.0 });

        assert!(world.despawn(e1));
        assert!(!world.despawn(e1), "double despawn must be rejected");
        assert!(!world.is_alive(e1));

        // e3 was swapped into e1's old row; its data must still resolve.
        assert_eq!(
            world.get::<Position>(e3),
            Some(&Position { x: 3.0, y: 0.0 })
        );
        assert_eq!(world.view::<Position>().count(), 2);
        assert_eq!(world.entity_count(), 2);

        let removed = world.take_removed::<Position>();
        assert_eq!(removed.len(), 1);
        assert_eq!(removed[0].0, e1);
        assert!(
            world.take_removed::<Position>().is_empty(),
            "drain semantics"
        );
    }

    #[test]
    fn iter_changed_sees_only_later_ticks() {
        let mut world = World::new();
        world.tick();
        let e1 = world.spawn();
        world.insert(e1, Position { x: 1.0, y: 0.0 });
        let e2 = world.spawn();
        world.insert(e2, Position { x: 2.0, y: 0.0 });
        let after_spawn = world.current_tick();

        world.tick();
        world.get_mut::<Position>(e1).expect("position").x = 10.0;

        let changed: Vec<Entity> = world
            .iter_changed::<Position>(after_spawn)
            .map(|(entity, _)| entity)
            .collect();
        assert_eq!(changed, vec![e1], "only e1 mutated after the mark");
        assert_eq!(world.iter_changed::<Position>(after_spawn).count(), 1);
    }

    #[test]
    fn component_registry_records_names() {
        let mut world = World::new();
        assert!(world.register_component::<Position>("position"));
        assert!(!world.register_component::<Position>("position_again"));
        assert_eq!(world.component_name::<Position>(), Some("position"));
        assert_eq!(world.component_name::<Velocity>(), None);
    }

    #[test]
    #[should_panic(expected = "insert on dead/stale entity")]
    fn insert_on_stale_handle_panics() {
        let mut world = World::new();
        let entity = world.spawn();
        assert!(world.despawn(entity));
        world.insert(entity, Position { x: 0.0, y: 0.0 });
    }

    #[test]
    #[should_panic(expected = "remove on dead/stale entity")]
    fn remove_on_stale_handle_panics() {
        let mut world = World::new();
        let entity = world.spawn();
        world.insert(entity, Position { x: 0.0, y: 0.0 });
        assert!(world.despawn(entity));
        let _ = world.remove::<Position>(entity);
    }
}

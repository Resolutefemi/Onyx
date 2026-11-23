//! Entity handles and generational allocation (`docs/PRD.md` §1).
//!
//! Entity identity uses a generational index: each slot carries a generation
//! counter that increments on every kill, so stale handles held by scripts,
//! the editor or AI agents can never alias a recycled entity.

/// A lightweight entity handle: slot index plus generation.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct Entity {
    /// Slot index into archetype tables.
    pub index: u32,
    /// Generation of the slot at handle-creation time (even = live slot).
    pub generation: u32,
}

/// Allocator handing out [`Entity`] handles and recycling killed slots.
///
/// Live slots carry an even generation; killing flips it to odd and queues the
/// index for reuse, where the next spawn flips it back to even. A stale handle
/// therefore never matches the current generation.
#[derive(Debug, Default)]
pub struct Entities {
    generations: Vec<u32>,
    free: Vec<u32>,
    alive_count: usize,
}

impl Entities {
    /// An empty allocator.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Allocate a fresh entity handle, recycling a killed slot when available.
    pub fn spawn(&mut self) -> Entity {
        let entity = match self.free.pop() {
            Some(index) => {
                let generation = self.generations[index as usize] + 1;
                self.generations[index as usize] = generation;
                Entity { index, generation }
            }
            None => {
                let index = self.generations.len() as u32;
                self.generations.push(0);
                Entity {
                    index,
                    generation: 0,
                }
            }
        };
        self.alive_count += 1;
        entity
    }

    /// Kill `entity` and queue its slot for recycling. Returns `false` if the
    /// handle is stale (already killed or recycled) or out of range.
    pub fn kill(&mut self, entity: Entity) -> bool {
        if !self.is_alive(entity) {
            return false;
        }
        self.generations[entity.index as usize] += 1;
        self.free.push(entity.index);
        self.alive_count -= 1;
        true
    }

    /// Whether `entity` refers to a currently live slot.
    #[must_use]
    pub fn is_alive(&self, entity: Entity) -> bool {
        match self.generations.get(entity.index as usize) {
            Some(&generation) => {
                generation == entity.generation && entity.generation.is_multiple_of(2)
            }
            None => false,
        }
    }

    /// Number of live entities.
    #[must_use]
    pub fn len(&self) -> usize {
        self.alive_count
    }

    /// Whether no entity is alive.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.alive_count == 0
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn spawn_allocates_unique_indices() {
        let mut entities = Entities::new();
        let a = entities.spawn();
        let b = entities.spawn();
        let c = entities.spawn();
        assert_ne!(a.index, b.index);
        assert_ne!(b.index, c.index);
        assert_eq!(entities.len(), 3);
        assert!(entities.is_alive(b));
    }

    #[test]
    fn kill_recycles_slots_with_bumped_generation() {
        let mut entities = Entities::new();
        let b = entities.spawn();
        let b_stale = b;
        assert!(entities.kill(b));
        assert!(!entities.is_alive(b_stale), "stale handle must be dead");
        assert!(!entities.kill(b_stale), "double kill must be rejected");

        let recycled = entities.spawn();
        assert_eq!(recycled.index, b.index, "slot must be reused");
        assert_ne!(recycled.generation, b_stale.generation);
        assert!(entities.is_alive(recycled));
        assert!(!entities.is_alive(b_stale));
        assert_eq!(entities.len(), 1);
    }
}

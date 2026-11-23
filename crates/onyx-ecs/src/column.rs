//! Type-erased SoA columns with change-detection ticks (`docs/PRD.md` §1).
//!
//! Archetypes store entity data as parallel columns, one per component type.
//! [`Column`] is the type-erased interface that lets the [`World`](crate::world)
//! migrate rows between archetypes without knowing `T`; [`TypedColumn<T>`] is
//! the dense implementation: a `Vec<T>` plus per-row [`Ticks`].

use std::any::{Any, TypeId};

use crate::component::Component;

/// Per-element change bookkeeping.
///
/// `added` is the world tick at which the component was attached; `changed`
/// the tick of its most recent mutation (write access through the ECS bumps
/// it). Change queries compare these against a `since` tick.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Ticks {
    /// World tick at which the component was attached to its entity.
    pub added: u64,
    /// World tick of the most recent mutation of the value.
    pub changed: u64,
}

/// Type-erased component column owned by an archetype.
pub trait Column: Send + Sync {
    /// `TypeId` of the stored component type.
    fn component_type(&self) -> TypeId;

    /// Number of rows (must equal the owning archetype's entity count).
    fn len(&self) -> usize;

    /// Whether the column holds no rows.
    fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Remove row `index`, moving the last row into its place so the column
    /// stays aligned with its archetype's entity list.
    fn swap_remove(&mut self, index: usize);

    /// Move row `index` out of `self` and append it to `dst` (archetype
    /// migration). Panics if `dst` holds a different component type.
    fn move_to(&mut self, dst: &mut dyn Column, index: usize);

    /// A fresh, empty column of the same component type.
    fn blank(&self) -> Box<dyn Column>;

    fn as_any(&self) -> &dyn Any;
    fn as_any_mut(&mut self) -> &mut dyn Any;
}

/// Dense typed storage: a `Vec<T>` plus per-row [`Ticks`].
#[derive(Debug)]
pub struct TypedColumn<T: Component> {
    data: Vec<T>,
    ticks: Vec<Ticks>,
}

impl<T: Component> TypedColumn<T> {
    /// An empty column.
    pub fn new() -> Self {
        Self {
            data: Vec::new(),
            ticks: Vec::new(),
        }
    }

    /// Append a row.
    pub fn push(&mut self, value: T, ticks: Ticks) {
        self.data.push(value);
        self.ticks.push(ticks);
    }

    /// Swap-remove row `index`, returning its value.
    pub fn remove_at(&mut self, row: usize) -> Option<T> {
        if row >= self.data.len() {
            return None;
        }
        let value = self.data.swap_remove(row);
        self.ticks.swap_remove(row);
        Some(value)
    }

    /// The stored values, row-aligned with the archetype's entities.
    pub fn data(&self) -> &[T] {
        &self.data
    }

    /// Mutable row values; the caller is responsible for bumping [`Ticks`]
    /// when a mutation should be observed.
    pub fn data_mut(&mut self) -> &mut [T] {
        &mut self.data
    }

    /// Per-row change bookkeeping.
    pub fn ticks(&self) -> &[Ticks] {
        &self.ticks
    }

    /// Mutable per-row change bookkeeping.
    pub fn ticks_mut(&mut self) -> &mut [Ticks] {
        &mut self.ticks
    }
}

impl<T: Component> Default for TypedColumn<T> {
    fn default() -> Self {
        Self::new()
    }
}

impl<T: Component> Column for TypedColumn<T> {
    fn component_type(&self) -> TypeId {
        TypeId::of::<T>()
    }

    fn len(&self) -> usize {
        self.data.len()
    }

    fn swap_remove(&mut self, index: usize) {
        self.data.swap_remove(index);
        self.ticks.swap_remove(index);
    }

    fn move_to(&mut self, dst: &mut dyn Column, index: usize) {
        let typed = dst
            .as_any_mut()
            .downcast_mut::<TypedColumn<T>>()
            .expect("column type mismatch in archetype migration");
        let value = self.data.swap_remove(index);
        let ticks = self.ticks.swap_remove(index);
        typed.data.push(value);
        typed.ticks.push(ticks);
    }

    fn blank(&self) -> Box<dyn Column> {
        Box::new(TypedColumn::<T>::new())
    }

    fn as_any(&self) -> &dyn Any {
        self
    }

    fn as_any_mut(&mut self) -> &mut dyn Any {
        self
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[derive(Debug, Clone, PartialEq)]
    struct Hp(i32);

    #[test]
    fn push_remove_and_move_keep_rows_aligned() {
        let mut a = TypedColumn::<Hp>::new();
        let mut b = TypedColumn::<Hp>::new();
        a.push(
            Hp(1),
            Ticks {
                added: 1,
                changed: 1,
            },
        );
        a.push(
            Hp(2),
            Ticks {
                added: 1,
                changed: 2,
            },
        );
        a.push(
            Hp(3),
            Ticks {
                added: 2,
                changed: 3,
            },
        );

        a.move_to(&mut b, 1);
        assert_eq!(a.data(), &[Hp(1), Hp(3)]);
        assert_eq!(b.data(), &[Hp(2)]);
        assert_eq!(b.ticks()[0].changed, 2, "ticks travel with the row");

        assert_eq!(a.remove_at(0), Some(Hp(1)));
        assert_eq!(a.data(), &[Hp(3)]);
        assert_eq!(a.ticks()[0].changed, 3);
        assert_eq!(a.remove_at(9), None);
        assert_eq!(a.len(), 1);
        assert_eq!(a.component_type(), TypeId::of::<Hp>());
    }
}

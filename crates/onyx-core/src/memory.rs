//! Memory & threading primitives (`docs/PRD.md` §1, "Memory & Threading").
//!
//! - [`Bump`] — monotonic arena allocator: allocations are sequential and the
//!   whole arena resets in O(1). Values must be `Copy`, which keeps `reset`
//!   destructor-free and the API sound.
//! - [`FrameScratch`] — double-buffered bump for per-frame scratch memory:
//!   [`FrameScratch::flip`] retires everything allocated last frame.
//! - [`Slab`] — generational slab storage: stable [`SlabKey`] handles whose
//!   generations invalidate stale references after removal (ABA-safe).
//!
//! Every allocator exposes its used/capacity byte counts so platforms can
//! enforce the deterministic memory budgets the PRD requires.

/// Monotonic arena allocator. `alloc` returns a reference borrowed from the
/// arena; the borrow checker guarantees the arena cannot be `reset` while any
/// allocation reference is alive, so use-after-reset is not expressible in
/// safe code.
///
/// The flip side of that safety: allocation references are **exclusive** —
/// only one may be live at a time (the arena is borrowed mutably). Use a
/// value immediately, or reach for [`Slab`] when many live objects need
/// stable handles. A lifetime-parameterized arena (bumpalo-style, many live
/// references) lands with the job-system integration.
///
/// Chunks are `u64`-backed, so allocations with alignment up to 8 bytes
/// (every scalar the reflect layer models) are supported.
#[derive(Debug)]
pub struct Bump {
    chunks: Vec<Chunk>,
    chunk_size: usize,
}

#[derive(Debug)]
struct Chunk {
    data: Box<[u64]>,
    /// Bytes consumed so far (offset from the chunk base).
    used: usize,
}

impl Bump {
    /// An empty arena that grows in `chunk_size`-byte chunks (minimum 64).
    #[must_use]
    pub fn new(chunk_size: usize) -> Self {
        Self {
            chunks: Vec::new(),
            chunk_size: chunk_size.max(64),
        }
    }

    /// Allocate `value` in the arena and return a mutable reference.
    ///
    /// The returned reference borrows the arena: no `reset` can run while it
    /// is live, and the region handed out is unique because the head only
    /// moves forward within a `&mut self` call sequence.
    ///
    /// # Panics
    /// If `T`'s alignment exceeds 8 bytes (see the module docs).
    pub fn alloc<T: Copy>(&mut self, value: T) -> &mut T {
        let (size, align) = (std::mem::size_of::<T>(), std::mem::align_of::<T>());
        assert!(align <= 8, "Bump supports alignments up to 8 bytes");

        let fits = self.chunks.last().is_some_and(|chunk| {
            let aligned = (chunk.used + (align - 1)) & !(align - 1);
            aligned + size <= chunk.data.len() * 8
        });
        if !fits {
            let words = size.div_ceil(8).max(self.chunk_size.div_ceil(8));
            self.chunks.push(Chunk {
                data: vec![0u64; words].into_boxed_slice(),
                used: 0,
            });
        }
        let chunk = self.chunks.last_mut().expect("a chunk now exists");
        let aligned = (chunk.used + (align - 1)) & !(align - 1);
        // SAFETY: `aligned + size` was bounds-checked against the chunk's
        // byte capacity above, and the u64 backing aligns the base to 8.
        unsafe { Self::write(chunk, aligned, value) }
    }

    /// # Safety
    /// `offset` must be within `chunk`'s byte length and leave room for
    /// `size_of::<T>()` bytes; alignment is guaranteed by the `u64` backing.
    unsafe fn write<T: Copy>(chunk: &mut Chunk, offset: usize, value: T) -> &mut T {
        let base = chunk.data.as_mut_ptr().cast::<u8>();
        // SAFETY: `offset + size` was bounds-checked against the chunk's byte
        // capacity by the caller, and the u64 backing aligns the base to 8.
        let ptr = unsafe { base.add(offset).cast::<T>() };
        // SAFETY: the region is arena-private and was never written before
        // (the head only moves forward), so there is no aliased reference.
        unsafe { ptr.write(value) };
        chunk.used = offset + std::mem::size_of::<T>();
        // SAFETY: the pointer is valid, aligned, and exclusively borrowed for
        // the lifetime of `&mut self` (enforced by the borrow checker).
        unsafe { &mut *ptr }
    }

    /// Return every chunk's memory to the pool. All outstanding references
    /// must have ended (the borrow checker enforces this).
    pub fn reset(&mut self) {
        for chunk in &mut self.chunks {
            chunk.used = 0;
        }
    }

    /// Bytes currently allocated across all chunks (excluding padding).
    #[must_use]
    pub fn used(&self) -> usize {
        self.chunks.iter().map(|chunk| chunk.used).sum()
    }

    /// Total backing capacity across all chunks, in bytes.
    #[must_use]
    pub fn capacity(&self) -> usize {
        self.chunks.iter().map(|chunk| chunk.data.len() * 8).sum()
    }
}

/// Double-buffered per-frame scratch arena: allocations go to the active
/// buffer; [`flip`](FrameScratch::flip) retires last frame's allocations and
/// starts a fresh frame, keeping the previous frame readable until the next
/// flip (useful for transient read/write pairs like culling results).
#[derive(Debug)]
pub struct FrameScratch {
    buffers: [Bump; 2],
    active: usize,
}

impl FrameScratch {
    /// Two buffers of `chunk_size` bytes each.
    #[must_use]
    pub fn new(chunk_size: usize) -> Self {
        Self {
            buffers: [Bump::new(chunk_size), Bump::new(chunk_size)],
            active: 0,
        }
    }

    /// Allocate in the active frame's buffer.
    ///
    /// # Panics
    /// If `T`'s alignment exceeds 8 bytes.
    pub fn alloc<T: Copy>(&mut self, value: T) -> &mut T {
        self.buffers[self.active].alloc(value)
    }

    /// End the current frame: the next allocations go into the other buffer,
    /// which is reset first. Allocations from two frames ago are retired.
    pub fn flip(&mut self) {
        self.active = 1 - self.active;
        self.buffers[self.active].reset();
    }

    /// Bytes allocated in the current frame.
    #[must_use]
    pub fn used(&self) -> usize {
        self.buffers[self.active].used()
    }

    /// Total backing capacity of both buffers, in bytes.
    #[must_use]
    pub fn capacity(&self) -> usize {
        self.buffers[0].capacity() + self.buffers[1].capacity()
    }
}

/// Stable handle into a [`Slab`]. `generation` invalidates keys whose slot
/// was removed and reused (ABA protection).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct SlabKey {
    index: u32,
    generation: u32,
}

#[derive(Debug)]
struct Slot<T> {
    value: Option<T>,
    generation: u32,
}

/// Generational slab storage: dense `Vec` backing with a free list, stable
/// keys, and generation-checked access — the standard backing store for
/// handle-based subsystems (asset pools, audio voices, render objects).
#[derive(Debug)]
pub struct Slab<T> {
    slots: Vec<Slot<T>>,
    free: Vec<u32>,
    len: usize,
}

impl<T> Default for Slab<T> {
    fn default() -> Self {
        Self {
            slots: Vec::new(),
            free: Vec::new(),
            len: 0,
        }
    }
}

impl<T> Slab<T> {
    /// An empty slab.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Insert `value`, reusing a freed slot when available. The returned key
    /// stays valid until this value is removed.
    pub fn insert(&mut self, value: T) -> SlabKey {
        self.len += 1;
        if let Some(index) = self.free.pop() {
            let slot = &mut self.slots[index as usize];
            debug_assert!(slot.value.is_none());
            slot.value = Some(value);
            SlabKey {
                index,
                generation: slot.generation,
            }
        } else {
            let index = u32::try_from(self.slots.len()).expect("slab overflow");
            self.slots.push(Slot {
                value: Some(value),
                generation: 0,
            });
            SlabKey {
                index,
                generation: 0,
            }
        }
    }

    /// The value behind `key`, if it is still live.
    #[must_use]
    pub fn get(&self, key: SlabKey) -> Option<&T> {
        let slot = self.slots.get(key.index as usize)?;
        if slot.generation != key.generation {
            return None;
        }
        slot.value.as_ref()
    }

    /// Mutable access behind `key`, if it is still live.
    pub fn get_mut(&mut self, key: SlabKey) -> Option<&mut T> {
        let slot = self.slots.get_mut(key.index as usize)?;
        if slot.generation != key.generation {
            return None;
        }
        slot.value.as_mut()
    }

    /// Whether `key` still refers to a live value.
    #[must_use]
    pub fn contains(&self, key: SlabKey) -> bool {
        self.get(key).is_some()
    }

    /// Remove and return the value behind `key`. The slot's generation
    /// advances, invalidating every outstanding key to it.
    pub fn remove(&mut self, key: SlabKey) -> Option<T> {
        let slot = self.slots.get_mut(key.index as usize)?;
        if slot.generation != key.generation {
            return None;
        }
        slot.generation = slot.generation.wrapping_add(1);
        let value = slot.value.take()?;
        self.free.push(key.index);
        self.len -= 1;
        Some(value)
    }

    /// Number of live values.
    #[must_use]
    pub fn len(&self) -> usize {
        self.len
    }

    /// Whether the slab holds no live values.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    /// Iterate live values with their current keys, in slot order.
    pub fn iter(&self) -> impl Iterator<Item = (SlabKey, &T)> {
        self.slots.iter().enumerate().filter_map(|(index, slot)| {
            let value = slot.value.as_ref()?;
            Some((
                SlabKey {
                    index: index as u32,
                    generation: slot.generation,
                },
                value,
            ))
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn bump_honors_alignment_across_allocations() {
        let mut bump = Bump::new(64);
        assert_eq!(*bump.alloc(7u32), 7);
        let second = bump.alloc(1.5f64);
        let second_addr = std::ptr::from_ref(second) as usize;
        assert_eq!(second_addr % 8, 0, "f64 must be 8-aligned");
        assert_eq!(*second, 1.5);
        assert!(*bump.alloc(true));
        // 4 (u32) + 4 padding + 8 (f64) + 1 (bool), sequential head.
        assert_eq!(bump.used(), 17);
        assert!(bump.capacity() >= bump.used());
    }

    #[test]
    fn bump_grows_chunks_and_resets_in_place() {
        let mut bump = Bump::new(64);
        for i in 0..200u32 {
            *bump.alloc(i) += 1;
        }
        assert_eq!(bump.used(), 200 * 4);
        assert!(bump.capacity() >= 800, "must have grown past one chunk");
        bump.reset();
        assert_eq!(bump.used(), 0);
        assert!(bump.capacity() >= 800, "chunks are reused, not freed");
        let value = bump.alloc(42u64);
        assert_eq!(*value, 42);
    }

    #[test]
    fn frame_scratch_flips_between_buffers() {
        let mut scratch = FrameScratch::new(64);
        *scratch.alloc(1u32) = 10;
        assert_eq!(scratch.used(), 4);
        scratch.flip();
        assert_eq!(scratch.used(), 0, "new frame starts fresh");
        *scratch.alloc(2u64) = 20;
        assert_eq!(scratch.used(), 8);
        // Flipping to the buffer that held two frames ago resets it: frame A
        // is retired here, frame B still lives in the other buffer.
        scratch.flip();
        assert_eq!(scratch.used(), 0);
        scratch.flip();
        assert_eq!(scratch.used(), 0, "frame B retired one flip later");
        assert!(scratch.capacity() >= 128);
    }

    #[test]
    fn slab_keys_survive_removal_and_detect_reuse() {
        let mut slab = Slab::new();
        let first = slab.insert("alpha");
        let second = slab.insert("beta");
        assert_eq!(slab.get(first), Some(&"alpha"));
        assert_eq!(slab.len(), 2);

        assert_eq!(slab.remove(first), Some("alpha"));
        assert_eq!(slab.get(first), None);
        assert!(!slab.contains(first));
        assert_eq!(slab.len(), 1);

        let third = slab.insert("gamma");
        assert_eq!(third.index, first.index, "freed slot is reused");
        assert_ne!(third.generation, first.generation, "generation advances");
        assert_eq!(slab.get(first), None, "stale key must not see the reuse");
        assert_eq!(slab.get(third), Some(&"gamma"));
        assert_eq!(slab.get(second), Some(&"beta"));
    }

    #[test]
    fn slab_mut_access_and_iteration() {
        let mut slab = Slab::new();
        let key = slab.insert(vec![1, 2, 3]);
        slab.get_mut(key).expect("live").push(4);
        assert_eq!(slab.get(key), Some(&vec![1, 2, 3, 4]));
        let collected: Vec<_> = slab.iter().map(|(k, v)| (k, v.len())).collect();
        assert_eq!(collected, vec![(key, 4)]);
    }

    #[test]
    fn slab_remove_with_stale_key_is_noop() {
        let mut slab = Slab::new();
        let key = slab.insert(1u8);
        slab.remove(key);
        assert_eq!(slab.remove(key), None, "double remove rejected");
        assert_eq!(
            slab.remove(SlabKey {
                index: 99,
                generation: 0
            }),
            None
        );
    }
}

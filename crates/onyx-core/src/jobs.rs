//! Job pool (`docs/PRD.md` §1, "Memory & Threading": job system /
//! work-stealing thread pool).
//!
//! A fixed worker pool pulls boxed jobs from a shared inject queue and parks
//! on a condvar when idle; [`JobHandle`] joins by counter, so any number of
//! submissions can share one wait. The ECS read-batch path keeps its own
//! scoped-thread dispatch (onyx-ecs); this pool serves general engine jobs.
//! Lock-free per-worker deques and fiber tasks land with the fiber scheduler
//! — the queue is mutex-protected for now, which is correct and fast enough
//! for coarse-grained engine work.

use std::collections::VecDeque;
use std::sync::{Arc, Condvar, Mutex};
use std::thread::JoinHandle;

type Job = Box<dyn FnOnce() + Send + 'static>;

#[derive(Default)]
struct QueueState {
    jobs: VecDeque<Job>,
    shutdown: bool,
}

struct Counter {
    remaining: Mutex<usize>,
    idle: Condvar,
}

impl Counter {
    fn new(remaining: usize) -> Arc<Self> {
        Arc::new(Self {
            remaining: Mutex::new(remaining),
            idle: Condvar::new(),
        })
    }

    fn wait(&self) {
        let mut remaining = self.remaining.lock().expect("counter poisoned");
        while *remaining > 0 {
            remaining = self.idle.wait(remaining).expect("counter poisoned");
        }
    }

    fn complete(&self) {
        let mut remaining = self.remaining.lock().expect("counter poisoned");
        *remaining -= 1;
        if *remaining == 0 {
            self.idle.notify_all();
        }
    }
}

/// Join handle for one or more submitted jobs. Jobs detached from their
/// handle still run to completion; joining waits until every job the handle
/// tracks has finished.
#[derive(Clone)]
pub struct JobHandle {
    counter: Arc<Counter>,
}

impl JobHandle {
    /// Block until every tracked job has finished.
    pub fn join(&self) {
        self.counter.wait();
    }

    /// Whether every tracked job has finished without blocking.
    #[must_use]
    pub fn is_complete(&self) -> bool {
        *self.counter.remaining.lock().expect("counter poisoned") == 0
    }
}

/// A fixed-size worker pool executing boxed jobs from a shared queue.
pub struct JobPool {
    state: Arc<(Mutex<QueueState>, Condvar)>,
    workers: Vec<JoinHandle<()>>,
    threads: usize,
}

impl JobPool {
    /// Spawn `threads` workers (`threads` clamped to at least 1).
    #[must_use]
    pub fn new(threads: usize) -> Self {
        let threads = threads.max(1);
        let state = Arc::new((Mutex::new(QueueState::default()), Condvar::new()));
        let mut workers = Vec::with_capacity(threads);
        for _ in 0..threads {
            let state = Arc::clone(&state);
            workers.push(
                std::thread::Builder::new()
                    .name("onyx-job".to_string())
                    .spawn(move || worker_loop(&state))
                    .expect("spawn job worker"),
            );
        }
        Self {
            state,
            workers,
            threads,
        }
    }

    /// Number of worker threads.
    #[must_use]
    pub fn worker_count(&self) -> usize {
        self.threads
    }

    /// Submit `job`; the returned handle joins it.
    pub fn submit<F>(&self, job: F) -> JobHandle
    where
        F: FnOnce() + Send + 'static,
    {
        let counter = Counter::new(1);
        self.enqueue(&counter, job);
        JobHandle { counter }
    }

    /// Run `f(index)` for every index in `0..len`, split into `chunk`-sized
    /// batches dispatched across the pool. One handle tracks all batches.
    ///
    /// # Panics
    /// A panicking batch poisons the closure's internal state the same way a
    /// panicking thread would; the join still returns.
    pub fn parallel_for<F>(&self, len: usize, chunk: usize, f: F) -> JobHandle
    where
        F: Fn(usize) + Send + Sync + 'static,
    {
        let chunk = chunk.max(1);
        let f = Arc::new(f);
        let batches = len.div_ceil(chunk);
        let counter = Counter::new(batches);
        for start in (0..len).step_by(chunk) {
            let f = Arc::clone(&f);
            let end = (start + chunk).min(len);
            self.enqueue(&counter, move || {
                for index in start..end {
                    f(index);
                }
            });
        }
        JobHandle { counter }
    }

    fn enqueue(&self, counter: &Arc<Counter>, job: impl FnOnce() + Send + 'static) {
        let counter = Arc::clone(counter);
        let (state, signal) = &*self.state;
        state
            .lock()
            .expect("job queue poisoned")
            .jobs
            .push_back(Box::new(move || {
                job();
                counter.complete();
            }));
        signal.notify_one();
    }
}

impl Drop for JobPool {
    fn drop(&mut self) {
        let (state, signal) = &*self.state;
        {
            let mut guard = state.lock().expect("job queue poisoned");
            guard.shutdown = true;
            guard.jobs.clear();
        }
        signal.notify_all();
        for worker in self.workers.drain(..) {
            let _ = worker.join();
        }
    }
}

fn worker_loop(state: &(Mutex<QueueState>, Condvar)) {
    let (mutex, signal) = state;
    loop {
        let job = {
            let mut guard = mutex.lock().expect("job queue poisoned");
            while guard.jobs.is_empty() && !guard.shutdown {
                guard = signal.wait(guard).expect("job queue poisoned");
            }
            match guard.jobs.pop_front() {
                Some(job) => job,
                None => return, // shutdown with empty queue
            }
        };
        job();
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::{AtomicUsize, Ordering};

    #[test]
    fn submitted_jobs_all_run_and_join() {
        let pool = JobPool::new(4);
        let done = Arc::new(AtomicUsize::new(0));
        let mut handles = Vec::new();
        for _ in 0..256 {
            let done = Arc::clone(&done);
            handles.push(pool.submit(move || {
                done.fetch_add(1, Ordering::Relaxed);
            }));
        }
        for handle in &handles {
            handle.join();
        }
        assert_eq!(done.load(Ordering::Relaxed), 256);
        assert!(handles.first().expect("handle").is_complete());
    }

    #[test]
    fn parallel_for_covers_every_index_exactly_once() {
        let pool = JobPool::new(4);
        let seen = Arc::new(Mutex::new(Vec::<usize>::new()));
        let handle = {
            let seen = Arc::clone(&seen);
            pool.parallel_for(1000, 7, move |index| {
                seen.lock().expect("not poisoned").push(index);
            })
        };
        handle.join();
        let mut seen = seen.lock().expect("not poisoned").clone();
        seen.sort_unstable();
        let expected: Vec<usize> = (0..1000).collect();
        assert_eq!(seen, expected);
    }

    #[test]
    fn pool_is_reusable_across_waves() {
        let pool = JobPool::new(2);
        for wave in 0..3 {
            let done = Arc::new(AtomicUsize::new(0));
            let mut handles = Vec::new();
            for _ in 0..32 {
                let done = Arc::clone(&done);
                handles.push(pool.submit(move || {
                    done.fetch_add(wave, Ordering::Relaxed);
                }));
            }
            for handle in handles {
                handle.join();
            }
            assert_eq!(done.load(Ordering::Relaxed), 32 * wave);
        }
        assert_eq!(pool.worker_count(), 2);
    }

    #[test]
    fn zero_thread_request_still_yields_one_worker() {
        let pool = JobPool::new(0);
        assert_eq!(pool.worker_count(), 1);
        let handle = pool.submit(|| ());
        handle.join();
        assert!(handle.is_complete());
    }
}

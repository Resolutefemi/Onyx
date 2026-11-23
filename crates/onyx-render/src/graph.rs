//! DAG render graph with resource-driven dependency inference
//! (`docs/PRD.md` §2).
//!
//! Nodes declare stable names and which render **resources** they read and
//! write; [`RenderGraph::execution_order`] infers the edges — a writer runs
//! before any reader of the same resource, conflicting writers run in
//! declaration order — and resolves a deterministic Kahn topological order
//! (ties break by insertion order). Cycles through resources are rejected.
//!
//! This is the scheduling half of the PRD's automatic-barrier story: the
//! graph fixes pass order, and wgpu inserts the actual GPU barriers from real
//! resource usage within a submission (see [`crate::executor`]).

use std::fmt;

/// A named node in the render graph declaring resource access.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RenderGraphNode {
    /// Stable node name (unique within a graph).
    pub name: &'static str,
    /// Render resources this node consumes (textures, buffers, …).
    pub reads: &'static [&'static str],
    /// Render resources this node produces or overwrites.
    pub writes: &'static [&'static str],
}

/// Render graph construction and scheduling failures.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum GraphError {
    /// A node with this name is already registered.
    DuplicateNode {
        /// The offending name.
        name: String,
    },
    /// Resource dependencies form a cycle; these nodes could never be
    /// ordered.
    Cycle {
        /// Names of nodes caught in the cycle.
        unresolved: Vec<String>,
    },
}

impl fmt::Display for GraphError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::DuplicateNode { name } => write!(f, "duplicate render node: {name}"),
            Self::Cycle { unresolved } => {
                write!(f, "render graph cycle involving: {}", unresolved.join(", "))
            }
        }
    }
}

impl std::error::Error for GraphError {}

/// A directed acyclic graph of render nodes.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct RenderGraph {
    nodes: Vec<RenderGraphNode>,
}

impl RenderGraph {
    /// An empty graph.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Number of registered nodes.
    #[must_use]
    pub fn node_count(&self) -> usize {
        self.nodes.len()
    }

    /// Registered nodes, in insertion order.
    #[must_use]
    pub fn nodes(&self) -> &[RenderGraphNode] {
        &self.nodes
    }

    /// Register a node. Names must be unique; edges are inferred from
    /// resource declarations at scheduling time, not at registration.
    ///
    /// # Errors
    /// [`GraphError::DuplicateNode`] when the name is taken.
    pub fn try_add_node(&mut self, node: RenderGraphNode) -> Result<(), GraphError> {
        if self.nodes.iter().any(|existing| existing.name == node.name) {
            return Err(GraphError::DuplicateNode {
                name: node.name.to_string(),
            });
        }
        self.nodes.push(node);
        Ok(())
    }

    /// Kahn topological order of node names, with edges inferred from
    /// resources:
    ///
    /// 1. a node **writing** a resource runs before any node **reading** it
    ///    (producer before consumer),
    /// 2. two nodes **writing** the same resource conflict; insertion order
    ///    breaks the tie (later passes overwrite earlier ones),
    /// 3. a node reading what it writes in the same pass is feedback — the
    ///    self-edge is ignored (ping-pong resources land with the real
    ///    pipelines).
    ///
    /// Ties break by insertion order, so equal-priority nodes schedule
    /// deterministically.
    ///
    /// # Errors
    /// [`GraphError::Cycle`] when resource dependencies are not a DAG; the
    /// error lists the nodes that could never be ordered.
    pub fn execution_order(&self) -> Result<Vec<&'static str>, GraphError> {
        let count = self.nodes.len();
        let (mut in_degree, dependents) = self.build_edges();

        // `ready` acts as a FIFO seeded in insertion order.
        let mut ready: Vec<usize> = (0..count).filter(|i| in_degree[*i] == 0).collect();
        let mut cursor = 0;
        let mut order = Vec::with_capacity(count);
        while cursor < ready.len() {
            let index = ready[cursor];
            cursor += 1;
            order.push(self.nodes[index].name);
            for &dependent in &dependents[index] {
                in_degree[dependent] -= 1;
                if in_degree[dependent] == 0 {
                    ready.push(dependent);
                }
            }
        }

        if order.len() < count {
            let unresolved = self
                .nodes
                .iter()
                .filter(|node| !order.contains(&node.name))
                .map(|node| node.name.to_string())
                .collect();
            return Err(GraphError::Cycle { unresolved });
        }
        Ok(order)
    }

    /// Wavefront decomposition of the order [`RenderGraph::execution_order`]
    /// produces: wave 0 holds every node with no producers, and a node lands
    /// in the first wave after all its producers. Nodes within one wave never
    /// conflict on a written resource (edges force such pairs into different
    /// waves), so a scheduler may encode each wave's nodes concurrently —
    /// that is what [`crate::executor::execute_offscreen_parallel`] does.
    /// Within a wave (and across waves) order follows insertion order.
    ///
    /// # Errors
    /// [`GraphError::Cycle`] — same detection as
    /// [`RenderGraph::execution_order`].
    pub fn execution_waves(&self) -> Result<Vec<Vec<&'static str>>, GraphError> {
        let count = self.nodes.len();
        let (mut in_degree, dependents) = self.build_edges();
        let mut wave_of = vec![0usize; count];

        let mut ready: Vec<usize> = (0..count).filter(|i| in_degree[*i] == 0).collect();
        let mut cursor = 0;
        let mut order = Vec::with_capacity(count);
        while cursor < ready.len() {
            let index = ready[cursor];
            cursor += 1;
            order.push(self.nodes[index].name);
            for &dependent in &dependents[index] {
                // The consumer lands after its latest producer's wave.
                wave_of[dependent] = wave_of[dependent].max(wave_of[index] + 1);
                in_degree[dependent] -= 1;
                if in_degree[dependent] == 0 {
                    ready.push(dependent);
                }
            }
        }

        if order.len() < count {
            let unresolved = self
                .nodes
                .iter()
                .filter(|node| !order.contains(&node.name))
                .map(|node| node.name.to_string())
                .collect();
            return Err(GraphError::Cycle { unresolved });
        }

        // Group by wave; indices run in insertion order, so within-wave (and
        // cross-wave) order stays deterministic. Wave numbers cannot skip:
        // a node at wave N always has a producer at wave N-1.
        let max_wave = wave_of.iter().copied().max().unwrap_or(0);
        let mut waves: Vec<Vec<&'static str>> = vec![Vec::new(); max_wave + 1];
        for index in 0..count {
            waves[wave_of[index]].push(self.nodes[index].name);
        }
        waves.retain(|wave| !wave.is_empty());
        Ok(waves)
    }

    /// Inferred edge graph shared by [`RenderGraph::execution_order`] and
    /// [`RenderGraph::execution_waves`]: in-degrees and, per node, the
    /// indices it feeds.
    fn build_edges(&self) -> (Vec<usize>, Vec<Vec<usize>>) {
        let count = self.nodes.len();
        let mut in_degree = vec![0usize; count];
        let mut dependents: Vec<Vec<usize>> = vec![Vec::new(); count];

        for a in 0..count {
            for b in 0..count {
                if a == b {
                    continue; // rule 3: ignore self edges
                }
                // Rule 2 (write-write conflict): exactly one edge, by
                // declaration order — later passes overwrite earlier ones.
                // Rule 1 (write-read): producer before consumer. Mutual feed
                // pairs get both edges and are caught as cycles upstream.
                let shared_write = self.nodes[a]
                    .writes
                    .iter()
                    .any(|resource| self.nodes[b].writes.contains(resource));
                let a_feeds_b = self.nodes[a]
                    .writes
                    .iter()
                    .any(|resource| self.nodes[b].reads.contains(resource));
                let a_before_b = if shared_write { a < b } else { a_feeds_b };
                if a_before_b {
                    in_degree[b] += 1;
                    dependents[a].push(b);
                }
            }
        }
        (in_degree, dependents)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn node(
        name: &'static str,
        reads: &'static [&'static str],
        writes: &'static [&'static str],
    ) -> RenderGraphNode {
        RenderGraphNode {
            name,
            reads,
            writes,
        }
    }

    fn graph(nodes: &[RenderGraphNode]) -> RenderGraph {
        let mut graph = RenderGraph::new();
        for node in nodes {
            graph
                .try_add_node(node.clone())
                .expect("test graph is valid");
        }
        graph
    }

    #[test]
    fn empty_graph_orders_trivially() {
        let graph = RenderGraph::new();
        assert_eq!(graph.node_count(), 0);
        assert_eq!(
            graph.execution_order().expect("empty order"),
            Vec::<&str>::new()
        );
    }

    #[test]
    fn resource_chain_orders_writer_before_reader() {
        let graph = graph(&[
            node("shadow_map", &[], &["shadows"]),
            node("opaque", &["shadows"], &["color"]),
            node("post", &["color"], &[]),
        ]);
        assert_eq!(
            graph.execution_order().expect("acyclic"),
            vec!["shadow_map", "opaque", "post"]
        );
    }

    #[test]
    fn independent_nodes_keep_insertion_order() {
        let graph = graph(&[
            node("c", &[], &[]),
            node("a", &[], &[]),
            node("b", &[], &[]),
        ]);
        assert_eq!(
            graph.execution_order().expect("acyclic"),
            vec!["c", "a", "b"]
        );
    }

    #[test]
    fn conflicting_writers_run_in_declaration_order() {
        let first = graph(&[node("a", &[], &["t"]), node("b", &[], &["t"])]);
        assert_eq!(first.execution_order().expect("acyclic"), vec!["a", "b"]);
        let second = graph(&[node("b", &[], &["t"]), node("a", &[], &["t"])]);
        assert_eq!(second.execution_order().expect("acyclic"), vec!["b", "a"]);
    }

    #[test]
    fn diamond_joins_after_both_parents() {
        let graph = graph(&[
            node("root", &[], &["t"]),
            node("left", &["t"], &["m1"]),
            node("right", &["t"], &["m2"]),
            node("merge", &["m1", "m2"], &[]),
        ]);
        let order = graph.execution_order().expect("acyclic");
        assert_eq!(order.first(), Some(&"root"));
        assert_eq!(order.last(), Some(&"merge"));
        assert_eq!(order.len(), 4);
    }

    #[test]
    fn resource_cycle_is_detected() {
        let cyclic = graph(&[node("a", &["t2"], &["t1"]), node("b", &["t1"], &["t2"])]);
        let error = cyclic.execution_order().expect_err("cycle");
        assert!(error.to_string().contains("cycle"));
        let GraphError::Cycle { unresolved } = error else {
            panic!("expected cycle error");
        };
        assert_eq!(unresolved.len(), 2);
    }

    #[test]
    fn waves_split_independent_nodes_and_join_after_parents() {
        let diamond = graph(&[
            node("root", &[], &["t"]),
            node("left", &["t"], &["m1"]),
            node("right", &["t"], &["m2"]),
            node("merge", &["m1", "m2"], &[]),
        ]);
        let waves = diamond.execution_waves().expect("acyclic");
        assert_eq!(waves.len(), 3);
        assert_eq!(waves[0], vec!["root"]);
        assert_eq!(waves[1].len(), 2, "left and right run in one wave");
        assert!(waves[1].contains(&"left") && waves[1].contains(&"right"));
        assert_eq!(waves[2], vec!["merge"]);
    }

    #[test]
    fn conflicting_writers_never_share_a_wave() {
        let first = graph(&[node("a", &[], &["t"]), node("b", &[], &["t"])]);
        let waves = first.execution_waves().expect("acyclic");
        assert_eq!(waves, vec![vec!["a"], vec!["b"]]);
        let second = graph(&[node("b", &[], &["t"]), node("a", &[], &["t"])]);
        let waves = second.execution_waves().expect("acyclic");
        assert_eq!(waves, vec![vec!["b"], vec!["a"]]);
    }

    #[test]
    fn fully_independent_nodes_form_one_wave() {
        let graph = graph(&[
            node("c", &[], &[]),
            node("a", &[], &[]),
            node("b", &[], &[]),
        ]);
        let waves = graph.execution_waves().expect("acyclic");
        assert_eq!(waves.len(), 1);
        assert_eq!(
            waves[0],
            vec!["c", "a", "b"],
            "insertion order within a wave"
        );
    }

    #[test]
    fn deep_chain_deepens_the_waves() {
        let graph = graph(&[
            node("first", &[], &["t"]),
            node("second", &["t"], &["u"]),
            node("third", &["u"], &[]),
        ]);
        let waves = graph.execution_waves().expect("acyclic");
        assert_eq!(waves, vec![vec!["first"], vec!["second"], vec!["third"]]);
    }

    #[test]
    fn waves_reject_cycles_like_execution_order() {
        let cyclic = graph(&[node("a", &["t2"], &["t1"]), node("b", &["t1"], &["t2"])]);
        assert!(cyclic.execution_waves().is_err());
    }

    #[test]
    fn self_edges_are_ignored_as_feedback() {
        let graph = graph(&[node("feedback", &["t"], &["t"])]);
        assert_eq!(
            graph.execution_order().expect("self edge ignored"),
            vec!["feedback"]
        );
    }

    #[test]
    fn duplicate_names_are_rejected() {
        let mut graph = RenderGraph::new();
        graph
            .try_add_node(node("opaque", &[], &["color"]))
            .expect("first");
        let error = graph
            .try_add_node(node("opaque", &[], &["color"]))
            .expect_err("duplicate");
        assert_eq!(
            error,
            GraphError::DuplicateNode {
                name: "opaque".to_string()
            }
        );
        assert!(error.to_string().contains("opaque"));
    }
}

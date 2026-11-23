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
                // pairs get both edges and are caught as cycles below.
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

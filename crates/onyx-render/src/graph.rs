//! Minimal DAG render graph for the Onyx renderer (`docs/PRD.md` §2).
//!
//! Nodes carry stable names and explicit dependencies; [`RenderGraph::
//! execution_order`] resolves a deterministic Kahn topological order (ties
//! break by insertion order) and rejects cycles. Resource-driven automatic
//! barrier insertion and parallel pass execution land with the real passes
//! later in M2 — this module fixes the scheduling skeleton they build on.

use std::fmt;

/// A named node in the render graph with declared dependencies.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RenderGraphNode {
    /// Stable node name (unique within a graph).
    pub name: &'static str,
    /// Names of nodes that must run before this one.
    pub after: Vec<&'static str>,
}

/// Render graph construction and scheduling failures.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum GraphError {
    /// A node with this name is already registered.
    DuplicateNode {
        /// The offending name.
        name: String,
    },
    /// A dependency refers to a node that has not been registered.
    UnknownDependency {
        /// Node declaring the dependency.
        node: String,
        /// Unregistered dependency name.
        dependency: String,
    },
    /// Dependencies form a cycle; these nodes could never be ordered.
    Cycle {
        /// Names of nodes caught in the cycle.
        unresolved: Vec<String>,
    },
}

impl fmt::Display for GraphError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::DuplicateNode { name } => write!(f, "duplicate render node: {name}"),
            Self::UnknownDependency { node, dependency } => {
                write!(
                    f,
                    "node '{node}' depends on unregistered node '{dependency}'"
                )
            }
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

    /// Register a node. Dependencies must already be registered (declaration
    /// order is declaration order), and names must be unique.
    ///
    /// # Errors
    /// [`GraphError::DuplicateNode`] or [`GraphError::UnknownDependency`].
    pub fn try_add_node(&mut self, node: RenderGraphNode) -> Result<(), GraphError> {
        if self.nodes.iter().any(|existing| existing.name == node.name) {
            return Err(GraphError::DuplicateNode {
                name: node.name.to_string(),
            });
        }
        for dependency in &node.after {
            if !self
                .nodes
                .iter()
                .any(|existing| existing.name == *dependency)
            {
                return Err(GraphError::UnknownDependency {
                    node: node.name.to_string(),
                    dependency: (*dependency).to_string(),
                });
            }
        }
        self.nodes.push(node);
        Ok(())
    }

    /// Kahn topological order of node names. Ties break by insertion order,
    /// so equal-priority nodes schedule deterministically.
    ///
    /// # Errors
    /// [`GraphError::Cycle`] when dependencies are not a DAG; the error lists
    /// the nodes that could never be ordered.
    pub fn execution_order(&self) -> Result<Vec<&'static str>, GraphError> {
        let count = self.nodes.len();
        let mut in_degree = vec![0usize; count];
        let mut dependents: Vec<Vec<usize>> = vec![Vec::new(); count];
        for (index, node) in self.nodes.iter().enumerate() {
            for dependency in &node.after {
                if let Some(dependency_index) =
                    self.nodes.iter().position(|n| n.name == *dependency)
                {
                    in_degree[index] += 1;
                    dependents[dependency_index].push(index);
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

    fn node(name: &'static str, after: &[&'static str]) -> RenderGraphNode {
        RenderGraphNode {
            name,
            after: after.to_vec(),
        }
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
    fn linear_chain_respects_dependencies() {
        let mut graph = RenderGraph::new();
        graph
            .try_add_node(node("shadow_map", &[]))
            .expect("registered");
        graph
            .try_add_node(node("opaque", &["shadow_map"]))
            .expect("registered");
        graph
            .try_add_node(node("post", &["opaque"]))
            .expect("registered");
        assert_eq!(
            graph.execution_order().expect("acyclic"),
            vec!["shadow_map", "opaque", "post"]
        );
    }

    #[test]
    fn independent_nodes_keep_insertion_order() {
        let mut graph = RenderGraph::new();
        for name in ["c", "a", "b"] {
            graph.try_add_node(node(name, &[])).expect("registered");
        }
        assert_eq!(
            graph.execution_order().expect("acyclic"),
            vec!["c", "a", "b"]
        );
    }

    #[test]
    fn diamond_joins_after_both_parents() {
        let mut graph = RenderGraph::new();
        graph.try_add_node(node("root", &[])).expect("registered");
        graph
            .try_add_node(node("left", &["root"]))
            .expect("registered");
        graph
            .try_add_node(node("right", &["root"]))
            .expect("registered");
        graph
            .try_add_node(node("merge", &["left", "right"]))
            .expect("registered");
        let order = graph.execution_order().expect("acyclic");
        assert_eq!(order.first(), Some(&"root"));
        assert_eq!(order.last(), Some(&"merge"));
        assert_eq!(order.len(), 4);
    }

    #[test]
    fn cycles_are_rejected_with_the_unresolved_nodes() {
        let mut graph = RenderGraph::new();
        graph
            .try_add_node(node("a", &["b"]))
            .expect_err("b not registered yet");
        graph.try_add_node(node("b", &[])).expect("registered");
        graph.try_add_node(node("a", &["b"])).expect("registered");
        // Now form the cycle: a → b already, add c → a with b → c impossible
        // through try_add (deps must pre-exist), so build b depending on a
        // through a fresh graph instead.
        let mut cyclic = RenderGraph::new();
        cyclic
            .try_add_node(node("x", &["y"]))
            .expect_err("y missing");
        let _ = cyclic;
        // Build a genuine cycle by re-registering names with back edges is
        // impossible via try_add_node (forward deps rejected), so simulate a
        // legacy mutation path: construct nodes directly and sort.
        let hand_built = RenderGraph {
            nodes: vec![node("p", &["q"]), node("q", &["p"])],
        };
        let error = hand_built.execution_order().expect_err("cycle");
        let GraphError::Cycle { unresolved } = error else {
            panic!("expected cycle error");
        };
        assert_eq!(unresolved.len(), 2);
    }

    #[test]
    fn duplicate_names_are_rejected() {
        let mut graph = RenderGraph::new();
        graph.try_add_node(node("opaque", &[])).expect("first");
        let error = graph
            .try_add_node(node("opaque", &[]))
            .expect_err("duplicate");
        assert_eq!(
            error,
            GraphError::DuplicateNode {
                name: "opaque".to_string()
            }
        );
    }

    #[test]
    fn unknown_dependencies_are_rejected() {
        let mut graph = RenderGraph::new();
        let error = graph
            .try_add_node(node("post", &["missing_pass"]))
            .expect_err("unknown dependency");
        assert_eq!(
            error,
            GraphError::UnknownDependency {
                node: "post".to_string(),
                dependency: "missing_pass".to_string(),
            }
        );
        assert!(error.to_string().contains("missing_pass"));
    }
}

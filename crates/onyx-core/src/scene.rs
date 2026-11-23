//! Hybrid Scene & Prefab System (`docs/PRD.md` §1) — the document layer.
//!
//! [`SceneDocument`] and [`PrefabRegistry`] own scene *structure*: nested
//! prefab instantiation, override merging (variants), stable entity IDs
//! (`"parent/child"` paths) and JSON round-trips. Component values are
//! reflect scalars ([`ComponentMap`]). Applying them into the typed ECS
//! `World` requires the editor's dynamic type bridge (M5) — this layer is
//! the format both humans and AI agents (MCP, §16) read and write.

use std::collections::{BTreeMap, HashSet};
use std::fmt;

use onyx_reflect::json::JsonValue;
use onyx_reflect::FieldValue;

/// Component map for one entity: field name → reflect scalar.
pub type ComponentMap = BTreeMap<String, FieldValue>;

/// An entity template: either inline components or a prefab reference with
/// overrides, plus child templates. Ids are stable within a document and
/// become `"parent/child"` paths on instantiation.
#[derive(Debug, Clone, PartialEq, Default)]
pub struct EntityTemplate {
    /// Stable entity id (unique within the scene).
    pub id: String,
    /// Prefab to instantiate, if any.
    pub prefab: Option<String>,
    /// Inline components (merged last, so inline wins over prefab output).
    pub components: ComponentMap,
    /// Variant overrides patching the prefab's components (merge #2).
    pub overrides: ComponentMap,
    /// Child entities appended below the prefab/inline content.
    pub children: Vec<EntityTemplate>,
}

impl EntityTemplate {
    /// An inline entity with no prefab reference.
    #[must_use]
    pub fn leaf(id: impl Into<String>, components: ComponentMap) -> Self {
        Self {
            id: id.into(),
            prefab: None,
            components,
            overrides: ComponentMap::new(),
            children: Vec::new(),
        }
    }

    /// A prefab instance whose components are patched by `overrides`.
    #[must_use]
    pub fn instance(
        id: impl Into<String>,
        prefab: impl Into<String>,
        overrides: ComponentMap,
    ) -> Self {
        Self {
            id: id.into(),
            prefab: Some(prefab.into()),
            components: ComponentMap::new(),
            overrides,
            children: Vec::new(),
        }
    }

    fn resolve(
        &self,
        registry: &PrefabRegistry,
        stack: &mut Vec<String>,
        parent_path: &str,
    ) -> Result<ResolvedEntity, SceneError> {
        let id = if parent_path.is_empty() {
            self.id.clone()
        } else {
            format!("{parent_path}/{}", self.id)
        };
        let (mut components, mut children) = match &self.prefab {
            None => (self.components.clone(), Vec::new()),
            Some(prefab_name) => {
                if stack.iter().any(|name| name == prefab_name) {
                    return Err(SceneError::PrefabCycle {
                        prefab: prefab_name.clone(),
                    });
                }
                let prefab =
                    registry
                        .prefab(prefab_name)
                        .ok_or_else(|| SceneError::UnknownPrefab {
                            name: prefab_name.clone(),
                        })?;
                stack.push(prefab_name.clone());
                let inner = prefab.root.resolve(registry, stack, "");
                stack.pop();
                let mut inner = inner?;
                // The prefab's root maps onto this instance's id (standard
                // prefab semantics); descendants swap the old root prefix
                // for the instance's path.
                let old_root = inner.id.clone();
                inner.id = id.clone();
                for child in &mut inner.children {
                    re_root(child, &old_root, &id);
                }
                (inner.components, inner.children)
            }
        };
        // Merge order: prefab components → variant overrides → inline.
        for (field, value) in &self.overrides {
            components.insert(field.clone(), value.clone());
        }
        for (field, value) in &self.components {
            components.insert(field.clone(), value.clone());
        }
        for child in &self.children {
            children.push(child.resolve(registry, stack, &id)?);
        }
        Ok(ResolvedEntity {
            id,
            components,
            children,
        })
    }
}

/// A named prefab: a reusable entity tree root.
#[derive(Debug, Clone, PartialEq)]
pub struct Prefab {
    /// Prefab name (unique within a registry).
    pub name: String,
    /// The prefab's entity tree.
    pub root: EntityTemplate,
}

/// Registry of prefabs a scene can reference.
#[derive(Debug, Clone, PartialEq, Default)]
pub struct PrefabRegistry {
    prefabs: BTreeMap<String, Prefab>,
}

impl PrefabRegistry {
    /// An empty registry.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Register a prefab. Returns `false` if the name is taken.
    pub fn register(&mut self, prefab: Prefab) -> bool {
        if self.prefabs.contains_key(&prefab.name) {
            return false;
        }
        self.prefabs.insert(prefab.name.clone(), prefab);
        true
    }

    /// The prefab registered under `name`.
    #[must_use]
    pub fn prefab(&self, name: &str) -> Option<&Prefab> {
        self.prefabs.get(name)
    }

    /// Registered prefab names, in name order.
    #[must_use]
    pub fn names(&self) -> Vec<&str> {
        self.prefabs.keys().map(String::as_str).collect()
    }
}

/// Re-root `entity` and all descendants: ids prefixed with `old_root` swap
/// it for `new_root`.
fn re_root(entity: &mut ResolvedEntity, old_root: &str, new_root: &str) {
    if let Some(tail) = entity.id.strip_prefix(old_root) {
        entity.id = format!("{new_root}{tail}");
    } else {
        entity.id = format!("{new_root}/{}", entity.id);
    }
    let parent = entity.id.clone();
    for child in &mut entity.children {
        re_root(child, old_root, &parent);
    }
}

/// A fully instantiated entity (no prefab references remain).
#[derive(Debug, Clone, PartialEq)]
pub struct ResolvedEntity {
    /// Stable id path (`"parent/child"`).
    pub id: String,
    /// Merged component map.
    pub components: ComponentMap,
    /// Instantiated children.
    pub children: Vec<ResolvedEntity>,
}

impl ResolvedEntity {
    /// Depth-first `(id path, components)` pairs — the order entities spawn
    /// in when a scene is applied to the world.
    #[must_use]
    pub fn flatten(&self) -> Vec<(&str, &ComponentMap)> {
        let mut out = vec![(self.id.as_str(), &self.components)];
        for child in &self.children {
            out.extend(child.flatten());
        }
        out
    }
}

/// A scene document: named top-level entity templates.
#[derive(Debug, Clone, PartialEq, Default)]
pub struct SceneDocument {
    /// Scene name.
    pub name: String,
    /// Top-level entity templates, in document order.
    pub entities: Vec<EntityTemplate>,
}

impl SceneDocument {
    /// An empty scene with `name`.
    #[must_use]
    pub fn new(name: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            entities: Vec::new(),
        }
    }

    /// Instantiate every entity against `registry`, merging overrides and
    /// deriving stable id paths.
    ///
    /// # Errors
    /// [`SceneError::UnknownPrefab`], [`SceneError::PrefabCycle`],
    /// [`SceneError::DuplicateEntityId`].
    pub fn instantiate(&self, registry: &PrefabRegistry) -> Result<ResolvedScene, SceneError> {
        let mut stack = Vec::new();
        let entities = self
            .entities
            .iter()
            .map(|entity| entity.resolve(registry, &mut stack, ""))
            .collect::<Result<Vec<_>, _>>()?;
        let resolved = ResolvedScene {
            name: self.name.clone(),
            entities,
        };
        let mut seen = HashSet::new();
        for (id, _) in resolved.flatten() {
            if !seen.insert(id.to_string()) {
                return Err(SceneError::DuplicateEntityId { id: id.to_string() });
            }
        }
        Ok(resolved)
    }

    /// Serialize to compact JSON (order-preserving, round-trippable).
    #[must_use]
    pub fn to_json(&self) -> String {
        self.to_json_value().to_json()
    }

    fn to_json_value(&self) -> JsonValue {
        JsonValue::Object(vec![
            ("name".to_string(), JsonValue::String(self.name.clone())),
            (
                "entities".to_string(),
                JsonValue::Array(self.entities.iter().map(template_json).collect()),
            ),
        ])
    }

    /// Parse a scene document from JSON.
    ///
    /// # Errors
    /// [`SceneError::InvalidJson`] for malformed input,
    /// [`SceneError::UnsupportedValue`] for non-scalar component values.
    pub fn from_json(input: &str) -> Result<SceneDocument, SceneError> {
        let value =
            JsonValue::parse(input).map_err(|error| SceneError::InvalidJson(error.to_string()))?;
        let name = value
            .get("name")
            .and_then(JsonValue::as_str)
            .ok_or_else(|| SceneError::InvalidJson("missing scene name".to_string()))?
            .to_string();
        let entities = value
            .get("entities")
            .and_then(JsonValue::as_array)
            .ok_or_else(|| SceneError::InvalidJson("missing entities".to_string()))?
            .iter()
            .map(parse_template)
            .collect::<Result<Vec<_>, _>>()?;
        Ok(SceneDocument { name, entities })
    }
}

/// A fully instantiated scene.
#[derive(Debug, Clone, PartialEq)]
pub struct ResolvedScene {
    /// Scene name.
    pub name: String,
    /// Instantiated entity trees.
    pub entities: Vec<ResolvedEntity>,
}

impl ResolvedScene {
    /// Every entity in depth-first spawn order.
    #[must_use]
    pub fn flatten(&self) -> Vec<(&str, &ComponentMap)> {
        self.entities
            .iter()
            .flat_map(ResolvedEntity::flatten)
            .collect()
    }
}

/// Scene & prefab failures.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum SceneError {
    /// A prefab reference names nothing in the registry.
    UnknownPrefab {
        /// The missing prefab name.
        name: String,
    },
    /// Prefab references form a cycle.
    PrefabCycle {
        /// The prefab that closes the cycle.
        prefab: String,
    },
    /// Two entities claim the same stable id.
    DuplicateEntityId {
        /// The duplicated id path.
        id: String,
    },
    /// A component value is not a JSON scalar.
    UnsupportedValue(String),
    /// The document is malformed.
    InvalidJson(String),
}

impl fmt::Display for SceneError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::UnknownPrefab { name } => write!(f, "unknown prefab: {name}"),
            Self::PrefabCycle { prefab } => write!(f, "prefab cycle through: {prefab}"),
            Self::DuplicateEntityId { id } => write!(f, "duplicate entity id: {id}"),
            Self::UnsupportedValue(detail) => write!(f, "unsupported value: {detail}"),
            Self::InvalidJson(detail) => write!(f, "invalid scene JSON: {detail}"),
        }
    }
}

impl std::error::Error for SceneError {}

fn component_value(value: &JsonValue) -> Result<FieldValue, SceneError> {
    match value {
        JsonValue::Number(n) => Ok(FieldValue::F64(*n)),
        JsonValue::Bool(b) => Ok(FieldValue::Bool(*b)),
        JsonValue::String(s) => Ok(FieldValue::String(s.clone())),
        JsonValue::Null | JsonValue::Array(_) | JsonValue::Object(_) => Err(
            SceneError::UnsupportedValue("component values must be scalars".to_string()),
        ),
    }
}

fn component_map(value: &JsonValue) -> Result<ComponentMap, SceneError> {
    let pairs = value
        .as_object()
        .ok_or_else(|| SceneError::InvalidJson("components must be an object".to_string()))?;
    pairs
        .iter()
        .map(|(key, value)| Ok((key.clone(), component_value(value)?)))
        .collect()
}

fn template_json(template: &EntityTemplate) -> JsonValue {
    let components = |map: &ComponentMap| {
        JsonValue::Object(
            map.iter()
                .map(|(key, value)| (key.clone(), field_json(value)))
                .collect(),
        )
    };
    JsonValue::Object(vec![
        ("id".to_string(), JsonValue::String(template.id.clone())),
        (
            "prefab".to_string(),
            template
                .prefab
                .clone()
                .map(JsonValue::String)
                .unwrap_or(JsonValue::Null),
        ),
        ("components".to_string(), components(&template.components)),
        ("overrides".to_string(), components(&template.overrides)),
        (
            "children".to_string(),
            JsonValue::Array(template.children.iter().map(template_json).collect()),
        ),
    ])
}

fn field_json(value: &FieldValue) -> JsonValue {
    match value {
        FieldValue::F32(v) => JsonValue::Number(f64::from(*v)),
        FieldValue::F64(v) => JsonValue::Number(*v),
        FieldValue::U32(v) => JsonValue::Number(f64::from(*v)),
        FieldValue::U64(v) => JsonValue::Number(*v as f64),
        FieldValue::I32(v) => JsonValue::Number(f64::from(*v)),
        FieldValue::I64(v) => JsonValue::Number(*v as f64),
        FieldValue::Bool(v) => JsonValue::Bool(*v),
        FieldValue::String(v) => JsonValue::String(v.clone()),
    }
}

fn parse_template(value: &JsonValue) -> Result<EntityTemplate, SceneError> {
    let id = value
        .get("id")
        .and_then(JsonValue::as_str)
        .ok_or_else(|| SceneError::InvalidJson("entity missing id".to_string()))?
        .to_string();
    let prefab = match value.get("prefab") {
        Some(JsonValue::String(name)) => Some(name.clone()),
        Some(JsonValue::Null) | None => None,
        Some(_) => {
            return Err(SceneError::InvalidJson(
                "prefab must be a string or null".to_string(),
            ))
        }
    };
    let components = match value.get("components") {
        Some(components) => component_map(components)?,
        None => ComponentMap::new(),
    };
    let overrides = match value.get("overrides") {
        Some(overrides) => component_map(overrides)?,
        None => ComponentMap::new(),
    };
    let children = match value.get("children") {
        Some(JsonValue::Array(children)) => children
            .iter()
            .map(parse_template)
            .collect::<Result<Vec<_>, _>>()?,
        Some(JsonValue::Null) | None => Vec::new(),
        Some(_) => {
            return Err(SceneError::InvalidJson(
                "children must be an array".to_string(),
            ))
        }
    };
    Ok(EntityTemplate {
        id,
        prefab,
        components,
        overrides,
        children,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_prefab_registry() -> PrefabRegistry {
        let mut registry = PrefabRegistry::new();
        // "tree": trunk with two nested leaf children; the second child is
        // a prefab instance itself (nested prefab composition).
        let mut leaf = EntityTemplate::leaf(
            "leaf",
            [
                ("kind".to_string(), FieldValue::String("oak".to_string())),
                ("size".to_string(), FieldValue::F64(1.0)),
            ]
            .into_iter()
            .collect(),
        );
        leaf.children.push(EntityTemplate::leaf(
            "fruit",
            [("ripe".to_string(), FieldValue::Bool(true))]
                .into_iter()
                .collect(),
        ));
        registry.register(Prefab {
            name: "tree".to_string(),
            root: leaf,
        });
        registry
    }

    #[test]
    fn inline_entities_resolve_verbatim() {
        let mut scene = SceneDocument::new("empty");
        scene.entities.push(EntityTemplate::leaf(
            "hero",
            [("hp".to_string(), FieldValue::F64(100.0))]
                .into_iter()
                .collect(),
        ));
        let resolved = scene.instantiate(&PrefabRegistry::new()).expect("resolves");
        let flat = resolved.flatten();
        assert_eq!(flat.len(), 1);
        assert_eq!(flat[0].0, "hero");
        assert_eq!(flat[0].1.get("hp"), Some(&FieldValue::F64(100.0)));
    }

    #[test]
    fn prefab_instance_merges_overrides_and_children() {
        let registry = make_prefab_registry();
        let mut scene = SceneDocument::new("forest");
        let mut overrides = ComponentMap::new();
        overrides.insert("size".to_string(), FieldValue::F64(2.5));
        let mut instance = EntityTemplate::instance("tree1", "tree", overrides);
        instance.children.push(EntityTemplate::leaf(
            "swing",
            [(
                "attach".to_string(),
                FieldValue::String("trunk".to_string()),
            )]
            .into_iter()
            .collect(),
        ));
        scene.entities.push(instance);

        let resolved = scene.instantiate(&registry).expect("resolves");
        let flat = resolved.flatten();
        let ids: Vec<&str> = flat.iter().map(|(id, _)| *id).collect();
        assert_eq!(
            ids,
            vec!["tree1", "tree1/fruit", "tree1/swing"],
            "stable id paths; the prefab root maps onto the instance id",
        );
        let (_, components) = flat[0];
        assert_eq!(
            components.get("size"),
            Some(&FieldValue::F64(2.5)),
            "override wins over prefab value"
        );
        assert_eq!(
            components.get("kind"),
            Some(&FieldValue::String("oak".to_string())),
            "prefab fields survive"
        );
    }

    #[test]
    fn cycles_between_prefabs_are_rejected() {
        let mut registry = PrefabRegistry::new();
        registry.register(Prefab {
            name: "a".to_string(),
            root: EntityTemplate::instance("inner", "b", ComponentMap::new()),
        });
        registry.register(Prefab {
            name: "b".to_string(),
            root: EntityTemplate::instance("inner", "a", ComponentMap::new()),
        });
        let mut scene = SceneDocument::new("cycle");
        scene
            .entities
            .push(EntityTemplate::instance("outer", "a", ComponentMap::new()));
        assert_eq!(
            scene.instantiate(&registry),
            Err(SceneError::PrefabCycle {
                prefab: "a".to_string()
            })
        );
    }

    #[test]
    fn unknown_prefabs_and_duplicate_ids_are_rejected() {
        let mut scene = SceneDocument::new("broken");
        scene
            .entities
            .push(EntityTemplate::instance("x", "ghost", ComponentMap::new()));
        assert_eq!(
            scene.instantiate(&PrefabRegistry::new()),
            Err(SceneError::UnknownPrefab {
                name: "ghost".to_string()
            })
        );

        let mut scene = SceneDocument::new("dupes");
        scene
            .entities
            .push(EntityTemplate::leaf("twin", ComponentMap::new()));
        scene
            .entities
            .push(EntityTemplate::leaf("twin", ComponentMap::new()));
        assert_eq!(
            scene.instantiate(&PrefabRegistry::new()),
            Err(SceneError::DuplicateEntityId {
                id: "twin".to_string()
            })
        );
    }

    #[test]
    fn duplicate_prefab_names_are_rejected() {
        let mut registry = PrefabRegistry::new();
        assert!(registry.register(Prefab {
            name: "p".to_string(),
            root: EntityTemplate::leaf("x", ComponentMap::new()),
        }));
        assert!(!registry.register(Prefab {
            name: "p".to_string(),
            root: EntityTemplate::leaf("y", ComponentMap::new()),
        }));
        assert_eq!(registry.names(), vec!["p"]);
    }

    #[test]
    fn json_round_trip_preserves_structure() {
        let registry = make_prefab_registry();
        let mut scene = SceneDocument::new("forest");
        let mut overrides = ComponentMap::new();
        overrides.insert("size".to_string(), FieldValue::F64(2.5));
        let mut instance = EntityTemplate::instance("tree1", "tree", overrides);
        instance.children.push(EntityTemplate::leaf(
            "swing",
            [(
                "attach".to_string(),
                FieldValue::String("trunk".to_string()),
            )]
            .into_iter()
            .collect(),
        ));
        scene.entities.push(instance);
        scene.entities.push(EntityTemplate::leaf(
            "rock",
            [("mass".to_string(), FieldValue::F64(42.0))]
                .into_iter()
                .collect(),
        ));

        let json = scene.to_json();
        let parsed = SceneDocument::from_json(&json).expect("round trip");
        assert_eq!(parsed, scene);
        let resolved = parsed.instantiate(&registry).expect("resolves");
        assert_eq!(resolved.flatten().len(), 4);
    }

    #[test]
    fn non_scalar_component_values_are_rejected() {
        let error = SceneDocument::from_json(
            r#"{"name":"s","entities":[{"id":"e","components":{"nested":{"x":1}}}]}"#,
        )
        .expect_err("nested object is not a scalar");
        assert!(matches!(error, SceneError::UnsupportedValue(_)));
        assert!(SceneDocument::from_json("nope").is_err());
    }
}

//! Built-in scene tools: the first real [`EngineCommand`] implementations
//! on the control plane (PRD §16 pillar 1 — engine features exposed as
//! typed JSON-Schema tools). They mutate a shared [`SceneDocument`]
//! through reversible apply/revert pairs, so every AI mutation joins the
//! same undo history as human edits.

use std::cell::RefCell;
use std::rc::Rc;

use serde_json::{json, Value};

use onyx_core::{ComponentMap, EngineCommand, EntityTemplate, SceneDocument};
use onyx_reflect::FieldValue;

use crate::plane::ToolFactory;
use crate::tools::ToolSpec;

/// Shared scene handle the built-in tools close over.
pub type SharedScene = Rc<RefCell<SceneDocument>>;

/// `spawn_entity` tool spec.
pub const TOOL_SPAWN_ENTITY: ToolSpec = ToolSpec {
    name: "spawn_entity",
    description:
        "Append a leaf entity with the given id and optional component set; refused if the id already exists",
    input_schema: r#"{"type":"object","properties":{"id":{"type":"string"},"components":{"type":"object"}},"required":["id"],"additionalProperties":false}"#,
};

/// `despawn_entity` tool spec.
pub const TOOL_DESPAWN_ENTITY: ToolSpec = ToolSpec {
    name: "despawn_entity",
    description:
        "Remove the entity with the given id, remembering it for undo; refused if the id does not exist",
    input_schema: r#"{"type":"object","properties":{"id":{"type":"string"}},"required":["id"],"additionalProperties":false}"#,
};

/// `set_component` tool spec.
pub const TOOL_SET_COMPONENT: ToolSpec = ToolSpec {
    name: "set_component",
    description:
        "Set one component field on an entity (number, boolean or string); undo restores the previous value",
    input_schema: r#"{"type":"object","properties":{"id":{"type":"string"},"key":{"type":"string"},"value":{"type":["number","boolean","string"]}},"required":["id","key","value"],"additionalProperties":false}"#,
};

/// Map a JSON value to a scalar [`FieldValue`] (numbers become `F64`).
/// Objects and arrays are rejected in this increment.
///
/// # Errors
/// A message naming the unsupported value.
pub fn json_to_field_value(value: &Value) -> Result<FieldValue, String> {
    match value {
        Value::Number(number) => number
            .as_f64()
            .map(FieldValue::F64)
            .ok_or_else(|| "number out of range".to_owned()),
        Value::Bool(flag) => Ok(FieldValue::Bool(*flag)),
        Value::String(text) => Ok(FieldValue::String(text.clone())),
        other => Err(format!("unsupported component value: {other}")),
    }
}

/// Map a [`FieldValue`] back to JSON (`f32`/`f64` NaN and infinities
/// serialize as `null`, mirroring `serde_json`).
#[must_use]
pub fn field_value_to_json(value: &FieldValue) -> Value {
    match value {
        FieldValue::F32(v) => json!(v),
        FieldValue::F64(v) => json!(v),
        FieldValue::U32(v) => json!(v),
        FieldValue::U64(v) => json!(v),
        FieldValue::I32(v) => json!(v),
        FieldValue::I64(v) => json!(v),
        FieldValue::Bool(v) => json!(v),
        FieldValue::String(v) => json!(v),
    }
}

/// `spawn_entity`: append a leaf entity, refused when the id exists.
#[derive(Debug)]
pub struct SpawnEntity {
    scene: SharedScene,
    id: String,
    components: ComponentMap,
    applied: bool,
}

impl SpawnEntity {
    /// A spawn of `id` with `components` against `scene`.
    #[must_use]
    pub fn new(scene: SharedScene, id: impl Into<String>, components: ComponentMap) -> Self {
        Self {
            scene,
            id: id.into(),
            components,
            applied: false,
        }
    }
}

impl EngineCommand for SpawnEntity {
    fn name(&self) -> &'static str {
        TOOL_SPAWN_ENTITY.name
    }

    fn apply(&mut self) -> bool {
        let mut scene = self.scene.borrow_mut();
        if scene.entities.iter().any(|entity| entity.id == self.id) {
            return false;
        }
        scene.entities.push(EntityTemplate::leaf(
            self.id.clone(),
            self.components.clone(),
        ));
        self.applied = true;
        true
    }

    fn revert(&mut self) -> bool {
        if !self.applied {
            return false;
        }
        let mut scene = self.scene.borrow_mut();
        scene.entities.retain(|entity| entity.id != self.id);
        self.applied = false;
        true
    }
}

/// `despawn_entity`: remove an entity by id, restoring it at the same
/// position on revert.
#[derive(Debug)]
pub struct DespawnEntity {
    scene: SharedScene,
    id: String,
    removed: Option<(usize, EntityTemplate)>,
}

impl DespawnEntity {
    /// A despawn of `id` against `scene`.
    #[must_use]
    pub fn new(scene: SharedScene, id: impl Into<String>) -> Self {
        Self {
            scene,
            id: id.into(),
            removed: None,
        }
    }
}

impl EngineCommand for DespawnEntity {
    fn name(&self) -> &'static str {
        TOOL_DESPAWN_ENTITY.name
    }

    fn apply(&mut self) -> bool {
        let mut scene = self.scene.borrow_mut();
        let Some(position) = scene
            .entities
            .iter()
            .position(|entity| entity.id == self.id)
        else {
            return false;
        };
        self.removed = Some((position, scene.entities.remove(position)));
        true
    }

    fn revert(&mut self) -> bool {
        let Some((position, entity)) = self.removed.take() else {
            return false;
        };
        let mut scene = self.scene.borrow_mut();
        let position = position.min(scene.entities.len());
        scene.entities.insert(position, entity);
        true
    }
}

/// `set_component`: set one field, restoring the previous value (or its
/// absence) on revert.
#[derive(Debug)]
pub struct SetComponent {
    scene: SharedScene,
    id: String,
    key: String,
    value: FieldValue,
    previous: Option<FieldValue>,
    had_previous: bool,
    applied: bool,
}

impl SetComponent {
    /// A set of `key = value` on entity `id` against `scene`.
    #[must_use]
    pub fn new(
        scene: SharedScene,
        id: impl Into<String>,
        key: impl Into<String>,
        value: FieldValue,
    ) -> Self {
        Self {
            scene,
            id: id.into(),
            key: key.into(),
            value,
            previous: None,
            had_previous: false,
            applied: false,
        }
    }
}

impl EngineCommand for SetComponent {
    fn name(&self) -> &'static str {
        TOOL_SET_COMPONENT.name
    }

    fn apply(&mut self) -> bool {
        let mut scene = self.scene.borrow_mut();
        let Some(entity) = scene
            .entities
            .iter_mut()
            .find(|entity| entity.id == self.id)
        else {
            return false;
        };
        self.had_previous = entity.components.contains_key(&self.key);
        self.previous = entity.components.get(&self.key).cloned();
        entity
            .components
            .insert(self.key.clone(), self.value.clone());
        self.applied = true;
        true
    }

    fn revert(&mut self) -> bool {
        if !self.applied {
            return false;
        }
        let mut scene = self.scene.borrow_mut();
        let Some(entity) = scene
            .entities
            .iter_mut()
            .find(|entity| entity.id == self.id)
        else {
            return false;
        };
        if self.had_previous {
            if let Some(previous) = &self.previous {
                entity.components.insert(self.key.clone(), previous.clone());
            }
        } else {
            entity.components.remove(&self.key);
        }
        self.applied = false;
        true
    }
}

/// Parse `spawn_entity` arguments: `{"id": ..., "components": {...}}`
/// (components optional, defaulting to empty; values map through
/// [`json_to_field_value`]).
///
/// # Errors
/// A message naming the first invalid argument.
pub fn parse_spawn_args(args_json: &str) -> Result<(String, ComponentMap), String> {
    let value: Value =
        serde_json::from_str(args_json).map_err(|error| format!("invalid args: {error}"))?;
    let object = value.as_object().ok_or("args must be an object")?;
    let id = object
        .get("id")
        .and_then(Value::as_str)
        .ok_or("id must be a string")?
        .to_owned();
    let mut components = ComponentMap::new();
    if let Some(entries) = object.get("components") {
        let entries = entries.as_object().ok_or("components must be an object")?;
        for (key, item) in entries {
            components.insert(key.clone(), json_to_field_value(item)?);
        }
    }
    Ok((id, components))
}

/// Parse `despawn_entity` arguments: `{"id": ...}`.
///
/// # Errors
/// A message naming the first invalid argument.
pub fn parse_despawn_args(args_json: &str) -> Result<String, String> {
    let value: Value =
        serde_json::from_str(args_json).map_err(|error| format!("invalid args: {error}"))?;
    value
        .as_object()
        .and_then(|object| object.get("id"))
        .and_then(Value::as_str)
        .map(str::to_owned)
        .ok_or_else(|| "id must be a string".to_owned())
}

/// Parse `set_component` arguments:
/// `{"id": ..., "key": ..., "value": <scalar>}`.
///
/// # Errors
/// A message naming the first invalid argument.
pub fn parse_set_component_args(args_json: &str) -> Result<(String, String, FieldValue), String> {
    let value: Value =
        serde_json::from_str(args_json).map_err(|error| format!("invalid args: {error}"))?;
    let object = value.as_object().ok_or("args must be an object")?;
    let id = object
        .get("id")
        .and_then(Value::as_str)
        .ok_or("id must be a string")?
        .to_owned();
    let key = object
        .get("key")
        .and_then(Value::as_str)
        .ok_or("key must be a string")?
        .to_owned();
    let value = object
        .get("value")
        .ok_or_else(|| "value is required".to_owned())
        .and_then(json_to_field_value)?;
    Ok((id, key, value))
}

/// The `spawn_entity` factory bound to `scene`.
#[must_use]
pub fn spawn_factory(scene: &SharedScene) -> ToolFactory {
    let scene = scene.clone();
    Rc::new(move |args_json| {
        let (id, components) = parse_spawn_args(args_json)?;
        Ok(Box::new(SpawnEntity::new(scene.clone(), id, components)))
    })
}

/// The `despawn_entity` factory bound to `scene`.
#[must_use]
pub fn despawn_factory(scene: &SharedScene) -> ToolFactory {
    let scene = scene.clone();
    Rc::new(move |args_json| {
        let id = parse_despawn_args(args_json)?;
        Ok(Box::new(DespawnEntity::new(scene.clone(), id)))
    })
}

/// The `set_component` factory bound to `scene`.
#[must_use]
pub fn set_component_factory(scene: &SharedScene) -> ToolFactory {
    let scene = scene.clone();
    Rc::new(move |args_json| {
        let (id, key, value) = parse_set_component_args(args_json)?;
        Ok(Box::new(SetComponent::new(scene.clone(), id, key, value)))
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn scene_with(id: &str) -> SharedScene {
        let scene = SharedScene::new(RefCell::new(SceneDocument::new("test")));
        scene
            .borrow_mut()
            .entities
            .push(EntityTemplate::leaf(id, ComponentMap::new()));
        scene
    }

    #[test]
    fn spawn_refuses_duplicates_and_reverts() {
        let scene = scene_with("a");
        let mut command = SpawnEntity::new(scene.clone(), "a", ComponentMap::new());
        assert!(!command.apply(), "duplicate id refused");
        let mut command = SpawnEntity::new(scene.clone(), "b", ComponentMap::new());
        assert!(command.apply());
        assert_eq!(scene.borrow().entities.len(), 2);
        assert!(command.revert());
        assert_eq!(scene.borrow().entities.len(), 1);
        assert!(!command.revert(), "nothing left to revert");
    }

    #[test]
    fn despawn_restores_position_on_revert() {
        let scene = scene_with("first");
        scene
            .borrow_mut()
            .entities
            .push(EntityTemplate::leaf("second", ComponentMap::new()));
        let mut command = DespawnEntity::new(scene.clone(), "first");
        assert!(command.apply());
        assert_eq!(scene.borrow().entities[0].id, "second");
        assert!(command.revert());
        assert_eq!(scene.borrow().entities[0].id, "first");
        // Reverting made the id exist again; a genuinely missing id refuses.
        let mut missing = DespawnEntity::new(scene.clone(), "third");
        assert!(!missing.apply(), "missing id refused");
    }

    #[test]
    fn set_component_restores_previous_and_absence() {
        let scene = scene_with("hero");
        scene
            .borrow_mut()
            .entities
            .last_mut()
            .expect("entity")
            .components
            .insert("hp".to_owned(), FieldValue::F64(10.0));

        let mut command = SetComponent::new(scene.clone(), "hero", "hp", FieldValue::F64(99.0));
        assert!(command.apply());
        assert!(command.revert());
        assert_eq!(
            scene.borrow().entities[0].components.get("hp"),
            Some(&FieldValue::F64(10.0))
        );

        let mut insert = SetComponent::new(
            scene.clone(),
            "hero",
            "name",
            FieldValue::String("knight".to_owned()),
        );
        assert!(insert.apply());
        let mut remove = SetComponent::new(scene.clone(), "hero", "hp", FieldValue::F64(1.0));
        assert!(remove.apply());
        assert!(insert.revert(), "newly inserted key is removed on revert");
        assert!(!scene.borrow().entities[0].components.contains_key("name"));
    }

    #[test]
    fn args_parse_and_reject() {
        let (id, components) =
            parse_spawn_args(r#"{"id":"a","components":{"hp":5,"alive":true,"tag":"goblin"}}"#)
                .expect("spawn args");
        assert_eq!(id, "a");
        assert_eq!(components.get("hp"), Some(&FieldValue::F64(5.0)));
        assert_eq!(components.get("alive"), Some(&FieldValue::Bool(true)));
        assert_eq!(
            components.get("tag"),
            Some(&FieldValue::String("goblin".to_owned()))
        );
        assert!(parse_spawn_args("{}").is_err(), "id is required");
        assert!(parse_spawn_args(r#"{"id":"a","components":{"bad":{"x":1}}}"#).is_err());

        assert_eq!(parse_despawn_args(r#"{"id":"a"}"#).as_deref(), Ok("a"));
        let (id, key, value) =
            parse_set_component_args(r#"{"id":"a","key":"hp","value":7}"#).expect("set args");
        assert_eq!((id.as_str(), key.as_str()), ("a", "hp"));
        assert_eq!(value, FieldValue::F64(7.0));
    }

    #[test]
    fn field_values_round_trip_through_json() {
        // Numbers normalize to F64 across the JSON boundary.
        for value in [
            FieldValue::F32(0.5),
            FieldValue::F64(2.5),
            FieldValue::U32(7),
            FieldValue::U64(9),
            FieldValue::I32(-3),
            FieldValue::I64(-5),
        ] {
            let json = field_value_to_json(&value);
            let back = json_to_field_value(&json).expect("scalar round trip");
            assert_eq!(back.type_name(), "f64", "value {value:?} → {back:?}");
        }
        assert_eq!(
            json_to_field_value(&json!(2.5)),
            Ok(FieldValue::F64(2.5)),
            "numeric value is preserved"
        );
        assert_eq!(
            json_to_field_value(&field_value_to_json(&FieldValue::Bool(true)))
                .expect("bool")
                .type_name(),
            "bool"
        );
        assert_eq!(
            json_to_field_value(&field_value_to_json(&FieldValue::String("x".to_owned())))
                .expect("string")
                .type_name(),
            "String"
        );
    }
}

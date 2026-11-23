//! # onyx-reflect
//!
//! Reflection and serialization for the Onyx Engine (`docs/PRD.md` §1).
//!
//! Data-oriented components derive [`OnyxReflect`] and gain a compile-time
//! [`Schema`], dynamic get/set by field name ([`FieldValue`]), JSON
//! serialization ([`to_json`] / [`from_json`]) and a [`TypeRegistry`] whose
//! `schemas_json` export seeds the MCP tool-schema layer (PRD §16). The
//! editor inspector (M5) builds on the same registry.
//!
//! The derive supports non-generic structs with named scalar fields: `f32`,
//! `f64`, `u32`, `u64`, `i32`, `i64`, `bool` and `String`.

pub use onyx_reflect_macro::OnyxReflect;

/// Lets the derive-generated `onyx_reflect::…` paths resolve inside this
/// crate itself (the unit tests use the derive).
extern crate self as onyx_reflect;

use std::fmt;

/// A reflected scalar value (PRD §1 reflection layer).
#[derive(Debug, Clone, PartialEq)]
pub enum FieldValue {
    /// 32-bit float.
    F32(f32),
    /// 64-bit float.
    F64(f64),
    /// Unsigned 32-bit integer.
    U32(u32),
    /// Unsigned 64-bit integer.
    U64(u64),
    /// Signed 32-bit integer.
    I32(i32),
    /// Signed 64-bit integer.
    I64(i64),
    /// Boolean.
    Bool(bool),
    /// Owned string.
    String(String),
}

impl FieldValue {
    /// Type name of the stored value (matches [`FieldSchema::type_name`]).
    #[must_use]
    pub fn type_name(&self) -> &'static str {
        match self {
            Self::F32(_) => "f32",
            Self::F64(_) => "f64",
            Self::U32(_) => "u32",
            Self::U64(_) => "u64",
            Self::I32(_) => "i32",
            Self::I64(_) => "i64",
            Self::Bool(_) => "bool",
            Self::String(_) => "String",
        }
    }

    /// JSON literal for embedding inside an object: strings are quoted and
    /// escaped, non-finite floats become `null`.
    #[must_use]
    pub fn to_json_fragment(&self) -> String {
        match self {
            Self::F32(value) => json_number(f64::from(*value)),
            Self::F64(value) => json_number(*value),
            Self::U32(value) => format!("{value}"),
            Self::U64(value) => format!("{value}"),
            Self::I32(value) => format!("{value}"),
            Self::I64(value) => format!("{value}"),
            Self::Bool(value) => format!("{value}"),
            Self::String(value) => json_string_literal(value),
        }
    }
}

/// One field of a reflected type.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct FieldSchema {
    /// Field name.
    pub name: &'static str,
    /// Leaf type name (`"f32"`, `"String"`, ...).
    pub type_name: &'static str,
}

/// Compile-time-known schema of a reflected type.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Schema {
    /// Struct name.
    pub type_name: &'static str,
    /// Fields in declaration order.
    pub fields: Vec<FieldSchema>,
}

impl Schema {
    /// The schema entry for `name`, if the type has such a field.
    #[must_use]
    pub fn field(&self, name: &str) -> Option<&FieldSchema> {
        self.fields.iter().find(|field| field.name == name)
    }
}

/// Reflection for data-oriented structs (PRD §1). Derive with
/// [`OnyxReflect`]; [`schema`] is compile-time known and therefore not
/// callable through `dyn Reflect`, while get/set are object-safe.
pub trait Reflect {
    /// The static schema of the implementing type.
    fn schema() -> Schema
    where
        Self: Sized;

    /// Read a field by name.
    fn get(&self, field: &str) -> Option<FieldValue>;

    /// Write a field by name with a strictly matching [`FieldValue`] variant.
    /// Returns `false` for unknown fields or type mismatches.
    fn set(&mut self, field: &str, value: &FieldValue) -> bool;
}

/// Serialize a reflected value to JSON. Field order follows the schema.
#[must_use]
pub fn to_json<T: Reflect>(value: &T) -> String {
    let schema = T::schema();
    let mut out = String::from("{");
    for (index, field) in schema.fields.iter().enumerate() {
        if index > 0 {
            out.push(',');
        }
        out.push_str(&json_string_literal(field.name));
        out.push(':');
        let fragment = value
            .get(field.name)
            .map(|v| v.to_json_fragment())
            .unwrap_or_else(|| "null".to_string());
        out.push_str(&fragment);
    }
    out.push('}');
    out
}

/// Deserialize a reflected value from a flat JSON object (scalar fields
/// only). Requires `T: Default`: unknown JSON fields are rejected, missing
/// fields keep their `Default` values.
///
/// # Errors
/// [`ReflectError::InvalidJson`] for malformed input,
/// [`ReflectError::UnknownField`] for keys not in the schema,
/// [`ReflectError::TypeMismatch`] when a literal does not parse as the
/// declared field type.
pub fn from_json<T: Reflect + Default>(json: &str) -> Result<T, ReflectError> {
    let schema = T::schema();
    let mut value = T::default();
    for (key, fragment) in parse_object(json)? {
        let Some(field) = schema.field(&key) else {
            return Err(ReflectError::UnknownField(format!(
                "{}::{key}",
                schema.type_name
            )));
        };
        let typed = match field.type_name {
            "f32" => FieldValue::F32(parse_number(&fragment, field.type_name, &key)?),
            "f64" => FieldValue::F64(parse_number(&fragment, field.type_name, &key)?),
            "u32" => FieldValue::U32(parse_number(&fragment, field.type_name, &key)?),
            "u64" => FieldValue::U64(parse_number(&fragment, field.type_name, &key)?),
            "i32" => FieldValue::I32(parse_number(&fragment, field.type_name, &key)?),
            "i64" => FieldValue::I64(parse_number(&fragment, field.type_name, &key)?),
            "bool" => match fragment.as_str() {
                "true" => FieldValue::Bool(true),
                "false" => FieldValue::Bool(false),
                _ => {
                    return Err(ReflectError::TypeMismatch {
                        field: key,
                        expected: field.type_name,
                    })
                }
            },
            "String" => FieldValue::String(unquote(&fragment)?),
            other => {
                return Err(ReflectError::UnsupportedValue(format!(
                    "field type {other}"
                )))
            }
        };
        if !value.set(&key, &typed) {
            return Err(ReflectError::TypeMismatch {
                field: key,
                expected: field.type_name,
            });
        }
    }
    Ok(value)
}

/// Serialization / deserialization failures (PRD §1).
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ReflectError {
    /// The input is not a flat JSON object of scalar values.
    InvalidJson(String),
    /// A value cannot be represented (e.g. `\u` escapes).
    UnsupportedValue(String),
    /// JSON key not present in the target schema.
    UnknownField(String),
    /// A literal does not match the declared field type.
    TypeMismatch {
        /// Offending field name.
        field: String,
        /// Declared leaf type name.
        expected: &'static str,
    },
}

impl fmt::Display for ReflectError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidJson(detail) => write!(f, "invalid JSON: {detail}"),
            Self::UnsupportedValue(detail) => write!(f, "unsupported value: {detail}"),
            Self::UnknownField(name) => write!(f, "unknown field: {name}"),
            Self::TypeMismatch { field, expected } => {
                write!(f, "field '{field}' does not parse as {expected}")
            }
        }
    }
}

impl std::error::Error for ReflectError {}

/// Registry of reflected type schemas, in registration order. `schemas_json`
/// seeds the MCP tool-schema export (PRD §16) and later the editor inspector.
#[derive(Debug, Clone, Default)]
pub struct TypeRegistry {
    schemas: Vec<Schema>,
}

impl TypeRegistry {
    /// An empty registry.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Register the schema of `T`. Returns `false` if the type name is
    /// already registered.
    pub fn register<T: Reflect>(&mut self) -> bool {
        let schema = T::schema();
        if self
            .schemas
            .iter()
            .any(|existing| existing.type_name == schema.type_name)
        {
            return false;
        }
        self.schemas.push(schema);
        true
    }

    /// Registered schema for `type_name`.
    #[must_use]
    pub fn schema_of(&self, type_name: &str) -> Option<&Schema> {
        self.schemas
            .iter()
            .find(|schema| schema.type_name == type_name)
    }

    /// All registered schemas, in registration order.
    #[must_use]
    pub fn schemas(&self) -> &[Schema] {
        &self.schemas
    }

    /// JSON array of every registered schema — the shape MCP tool schemas
    /// and the future editor inspector are generated from.
    #[must_use]
    pub fn schemas_json(&self) -> String {
        let mut out = String::from("[");
        for (index, schema) in self.schemas.iter().enumerate() {
            if index > 0 {
                out.push(',');
            }
            out.push('{');
            out.push_str(&json_string_literal("type_name"));
            out.push(':');
            out.push_str(&json_string_literal(schema.type_name));
            out.push(',');
            out.push_str(&json_string_literal("fields"));
            out.push_str(":[");
            for (field_index, field) in schema.fields.iter().enumerate() {
                if field_index > 0 {
                    out.push(',');
                }
                out.push('{');
                out.push_str(&json_string_literal("name"));
                out.push(':');
                out.push_str(&json_string_literal(field.name));
                out.push(',');
                out.push_str(&json_string_literal("type"));
                out.push(':');
                out.push_str(&json_string_literal(field.type_name));
                out.push('}');
            }
            out.push_str("]}");
        }
        out.push(']');
        out
    }
}

/// Quoted, escaped JSON string literal.
fn json_string_literal(value: &str) -> String {
    let mut out = String::with_capacity(value.len() + 2);
    out.push('"');
    for character in value.chars() {
        match character {
            '"' => out.push_str("\\\""),
            '\\' => out.push_str("\\\\"),
            '\n' => out.push_str("\\n"),
            '\r' => out.push_str("\\r"),
            '\t' => out.push_str("\\t"),
            control if (control as u32) < 0x20 => {
                out.push_str(&format!("\\u{:04x}", control as u32));
            }
            other => out.push(other),
        }
    }
    out.push('"');
    out
}

/// Finite JSON number, or `null` for non-finite floats (JSON has no NaN/inf).
fn json_number(value: f64) -> String {
    if value.is_finite() {
        format!("{value}")
    } else {
        "null".to_string()
    }
}

fn type_mismatch(field: &str, expected: &'static str) -> ReflectError {
    ReflectError::TypeMismatch {
        field: field.to_string(),
        expected,
    }
}

fn parse_number<T: std::str::FromStr>(
    fragment: &str,
    expected: &'static str,
    field: &str,
) -> Result<T, ReflectError> {
    fragment
        .trim()
        .parse::<T>()
        .map_err(|_| type_mismatch(field, expected))
}

/// Strip the surrounding quotes of a JSON string literal and unescape it.
fn unquote(fragment: &str) -> Result<String, ReflectError> {
    let chars: Vec<char> = fragment.chars().collect();
    let (value, _) = parse_string(&chars, 0)?;
    Ok(value)
}

/// Parse a flat JSON object into `(key, raw value fragment)` pairs. Fragments
/// are scalar JSON literals (no nested objects or arrays).
fn parse_object(json: &str) -> Result<Vec<(String, String)>, ReflectError> {
    let chars: Vec<char> = json.chars().collect();
    let mut cursor = skip_ws(&chars, 0);
    if cursor >= chars.len() || chars[cursor] != '{' {
        return Err(ReflectError::InvalidJson("expected '{'".to_string()));
    }
    cursor = skip_ws(&chars, cursor + 1);
    if cursor < chars.len() && chars[cursor] == '}' {
        return Ok(Vec::new());
    }

    let mut pairs = Vec::new();
    loop {
        cursor = skip_ws(&chars, cursor);
        let (key, next) = parse_string(&chars, cursor)
            .map_err(|_| ReflectError::InvalidJson("expected object key".to_string()))?;
        cursor = skip_ws(&chars, next);
        if cursor >= chars.len() || chars[cursor] != ':' {
            return Err(ReflectError::InvalidJson(
                "expected ':' after key".to_string(),
            ));
        }
        cursor = skip_ws(&chars, cursor + 1);
        let (fragment, next) = parse_value_fragment(&chars, cursor)?;
        pairs.push((key, fragment));
        cursor = skip_ws(&chars, next);
        match chars.get(cursor) {
            Some(',') => cursor = skip_ws(&chars, cursor + 1),
            Some('}') => return Ok(pairs),
            _ => return Err(ReflectError::InvalidJson("expected ',' or '}'".to_string())),
        }
    }
}

fn skip_ws(chars: &[char], mut cursor: usize) -> usize {
    while cursor < chars.len() && matches!(chars[cursor], ' ' | '\t' | '\n' | '\r') {
        cursor += 1;
    }
    cursor
}

/// Parse a JSON string literal, unescaping escapes. `\u` escapes are
/// rejected ([`ReflectError::UnsupportedValue`]).
fn parse_string(chars: &[char], start: usize) -> Result<(String, usize), ReflectError> {
    if start >= chars.len() || chars[start] != '"' {
        return Err(ReflectError::InvalidJson("expected string".to_string()));
    }
    let mut out = String::new();
    let mut cursor = start + 1;
    while cursor < chars.len() {
        match chars[cursor] {
            '"' => return Ok((out, cursor + 1)),
            '\\' => {
                cursor += 1;
                let Some(escaped) = chars.get(cursor) else {
                    return Err(ReflectError::InvalidJson("unterminated escape".to_string()));
                };
                match escaped {
                    '"' => out.push('"'),
                    '\\' => out.push('\\'),
                    '/' => out.push('/'),
                    'n' => out.push('\n'),
                    'r' => out.push('\r'),
                    't' => out.push('\t'),
                    'u' => {
                        return Err(ReflectError::UnsupportedValue(
                            "\\u escapes are not supported".to_string(),
                        ))
                    }
                    other => {
                        return Err(ReflectError::InvalidJson(format!(
                            "unknown escape \\{other}"
                        )))
                    }
                }
                cursor += 1;
            }
            character => {
                out.push(character);
                cursor += 1;
            }
        }
    }
    Err(ReflectError::InvalidJson("unterminated string".to_string()))
}

/// Parse a scalar JSON value (string, number or boolean) into a raw fragment.
fn parse_value_fragment(chars: &[char], start: usize) -> Result<(String, usize), ReflectError> {
    if start >= chars.len() {
        return Err(ReflectError::InvalidJson(
            "unexpected end of input".to_string(),
        ));
    }
    if chars[start] == '"' {
        let (value, next) = parse_string(chars, start)?;
        return Ok((json_string_literal(&value), next));
    }
    let mut cursor = start;
    while cursor < chars.len() && !matches!(chars[cursor], ',' | '}' | ' ' | '\t' | '\n' | '\r') {
        if matches!(chars[cursor], '{' | '[') {
            return Err(ReflectError::UnsupportedValue(
                "nested objects and arrays are not supported".to_string(),
            ));
        }
        cursor += 1;
    }
    if cursor == start {
        return Err(ReflectError::InvalidJson("empty value".to_string()));
    }
    let fragment: String = chars[start..cursor].iter().collect();
    Ok((fragment, cursor))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[derive(Debug, Clone, PartialEq, Default, OnyxReflect)]
    struct Sample {
        x: f32,
        count: u32,
        label: String,
        active: bool,
    }

    #[derive(Debug, Clone, Copy, PartialEq, Default, OnyxReflect)]
    struct Point {
        x: f64,
        y: f64,
    }

    #[test]
    fn derived_schema_matches_declaration() {
        let schema = Sample::schema();
        assert_eq!(schema.type_name, "Sample");
        let names: Vec<&str> = schema.fields.iter().map(|field| field.name).collect();
        assert_eq!(names, vec!["x", "count", "label", "active"]);
        assert_eq!(schema.field("count").expect("count").type_name, "u32");
        assert!(schema.field("missing").is_none());
    }

    #[test]
    fn dynamic_get_and_set() {
        let mut sample = Sample {
            x: 1.5,
            count: 3,
            label: "hi".to_string(),
            active: true,
        };
        assert_eq!(sample.get("x"), Some(FieldValue::F32(1.5)));
        assert_eq!(sample.get("nope"), None);
        assert!(sample.set("count", &FieldValue::U32(7)));
        assert_eq!(sample.count, 7);
        assert!(
            !sample.set("count", &FieldValue::F32(1.0)),
            "strict variant matching must reject type mismatches"
        );
        assert!(!sample.set("nope", &FieldValue::U32(1)));
        assert!(sample.set("label", &FieldValue::String("changed".to_string())));
        assert_eq!(sample.label, "changed");
    }

    #[test]
    fn json_round_trip_with_escaping() {
        let sample = Sample {
            x: -0.25,
            count: 42,
            label: "quote \" back \\ newline \n tab \t".to_string(),
            active: true,
        };
        let json = to_json(&sample);
        assert!(json.contains("\"label\":\"quote \\\" back \\\\ newline \\n tab \\t\""));
        let parsed: Sample = from_json(&json).expect("round trip");
        assert_eq!(parsed, sample);
    }

    #[test]
    fn json_number_formatting() {
        let point = Point { x: 3.0, y: -0.25 };
        assert_eq!(to_json(&point), "{\"x\":3,\"y\":-0.25}");
        let parsed: Point = from_json("{\"x\": 3, \"y\":-0.25}").expect("parse");
        assert_eq!(parsed, point);
    }

    #[test]
    fn from_json_rejects_unknown_fields_and_bad_types() {
        assert_eq!(
            from_json::<Point>("{\"x\": 1.0, \"z\": 2.0}"),
            Err(ReflectError::UnknownField("Point::z".to_string()))
        );
        assert!(matches!(
            from_json::<Point>("{\"x\": \"not a number\"}"),
            Err(ReflectError::TypeMismatch { field, expected: "f64" }) if field == "x"
        ));
        assert!(from_json::<Point>("not json at all").is_err());
    }

    #[test]
    fn reflection_works_through_trait_objects() {
        let boxed: Box<dyn Reflect> = Box::new(Sample {
            x: 2.0,
            count: 1,
            label: String::new(),
            active: false,
        });
        assert_eq!(boxed.get("x"), Some(FieldValue::F32(2.0)));
        let mut boxed = boxed;
        assert!(boxed.set("count", &FieldValue::U32(9)));
    }

    #[test]
    fn registry_records_and_exports_schemas() {
        let mut registry = TypeRegistry::new();
        assert!(registry.register::<Sample>());
        assert!(registry.register::<Point>());
        assert!(
            !registry.register::<Sample>(),
            "duplicate registration rejected"
        );
        assert_eq!(registry.schemas().len(), 2);
        assert!(registry.schema_of("Point").is_some());
        let json = registry.schemas_json();
        assert!(json.contains("\"type_name\":\"Sample\""));
        assert!(json.contains("\"type_name\":\"Point\""));
        assert!(json.contains("\"name\":\"count\",\"type\":\"u32\""));
    }
}

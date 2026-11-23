//! Minimal JSON document model + parser (`docs/PRD.md` §1).
//!
//! Scenes, save games and the asset pipeline (§15) need a real DOM —
//! nested objects and arrays — beyond the flat scalar objects the reflect
//! layer handles. Dependency-free recursive-descent parser, order-preserving
//! objects, compact serializer, and a nesting depth limit for untrusted
//! input.

use std::fmt;

/// Maximum nesting depth accepted by [`JsonValue::parse`] (guards untrusted
/// scene/asset files against stack exhaustion).
const MAX_DEPTH: usize = 64;

/// A JSON value (PRD §1 serialization layer).
#[derive(Debug, Clone, PartialEq)]
pub enum JsonValue {
    /// `null`.
    Null,
    /// `true` / `false`.
    Bool(bool),
    /// Any JSON number, stored as f64.
    Number(f64),
    /// A string.
    String(String),
    /// An ordered array.
    Array(Vec<JsonValue>),
    /// An object; pairs keep their document order (duplicate keys keep the
    /// last occurrence on lookup).
    Object(Vec<(String, JsonValue)>),
}

impl JsonValue {
    /// Parse a complete JSON document.
    ///
    /// # Errors
    /// [`JsonError`] for malformed input, trailing content, or nesting past
    /// [`MAX_DEPTH`].
    pub fn parse(input: &str) -> Result<JsonValue, JsonError> {
        let mut parser = Parser {
            chars: input.chars().collect(),
            cursor: 0,
        };
        parser.skip_ws();
        let value = parser.parse_value(0)?;
        parser.skip_ws();
        if parser.cursor < parser.chars.len() {
            return Err(JsonError::TrailingContent(parser.cursor));
        }
        Ok(value)
    }

    /// Compact serialization.
    #[must_use]
    pub fn to_json(&self) -> String {
        let mut out = String::new();
        self.write(&mut out);
        out
    }

    fn write(&self, out: &mut String) {
        match self {
            Self::Null => out.push_str("null"),
            Self::Bool(value) => out.push_str(if *value { "true" } else { "false" }),
            Self::Number(value) => {
                if value.is_finite() {
                    out.push_str(&format!("{value}"));
                } else {
                    out.push_str("null");
                }
            }
            Self::String(value) => write_escaped(value, out),
            Self::Array(items) => {
                out.push('[');
                for (index, item) in items.iter().enumerate() {
                    if index > 0 {
                        out.push(',');
                    }
                    item.write(out);
                }
                out.push(']');
            }
            Self::Object(pairs) => {
                out.push('{');
                for (index, (key, value)) in pairs.iter().enumerate() {
                    if index > 0 {
                        out.push(',');
                    }
                    write_escaped(key, out);
                    out.push(':');
                    value.write(out);
                }
                out.push('}');
            }
        }
    }

    /// The value under `key` if this is an object.
    #[must_use]
    pub fn get(&self, key: &str) -> Option<&JsonValue> {
        match self {
            Self::Object(pairs) => pairs
                .iter()
                .rev()
                .find(|(name, _)| name == key)
                .map(|(_, value)| value),
            _ => None,
        }
    }

    /// String contents, if this is a string.
    #[must_use]
    pub fn as_str(&self) -> Option<&str> {
        match self {
            Self::String(value) => Some(value),
            _ => None,
        }
    }

    /// Number contents, if this is a number.
    #[must_use]
    pub fn as_f64(&self) -> Option<f64> {
        match self {
            Self::Number(value) => Some(*value),
            _ => None,
        }
    }

    /// Boolean contents, if this is a boolean.
    #[must_use]
    pub fn as_bool(&self) -> Option<bool> {
        match self {
            Self::Bool(value) => Some(*value),
            _ => None,
        }
    }

    /// Array items, if this is an array.
    #[must_use]
    pub fn as_array(&self) -> Option<&[JsonValue]> {
        match self {
            Self::Array(items) => Some(items),
            _ => None,
        }
    }

    /// Object pairs, if this is an object.
    #[must_use]
    pub fn as_object(&self) -> Option<&[(String, JsonValue)]> {
        match self {
            Self::Object(pairs) => Some(pairs),
            _ => None,
        }
    }
}

/// JSON parsing failures.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum JsonError {
    /// A character appeared where `expected` was required.
    UnexpectedCharacter {
        /// Byte-ish position (character index).
        position: usize,
        /// The offending character.
        found: char,
        /// What was expected.
        expected: &'static str,
    },
    /// Input ended before a complete value.
    UnexpectedEnd,
    /// Content followed the first complete value.
    TrailingContent(usize),
    /// Nesting exceeded [`MAX_DEPTH`].
    NestingTooDeep,
    /// A string escape is invalid or an escape sequence is unsupported.
    InvalidEscape(String),
}

impl fmt::Display for JsonError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::UnexpectedCharacter {
                position,
                found,
                expected,
            } => write!(f, "unexpected '{found}' at {position}: expected {expected}"),
            Self::UnexpectedEnd => write!(f, "unexpected end of input"),
            Self::TrailingContent(position) => {
                write!(f, "trailing content after value at {position}")
            }
            Self::NestingTooDeep => write!(f, "nesting exceeds {MAX_DEPTH} levels"),
            Self::InvalidEscape(detail) => write!(f, "invalid escape: {detail}"),
        }
    }
}

impl std::error::Error for JsonError {}

struct Parser {
    chars: Vec<char>,
    cursor: usize,
}

impl Parser {
    fn skip_ws(&mut self) {
        while self
            .chars
            .get(self.cursor)
            .is_some_and(|c| matches!(c, ' ' | '\t' | '\n' | '\r'))
        {
            self.cursor += 1;
        }
    }

    fn peek(&self) -> Option<char> {
        self.chars.get(self.cursor).copied()
    }

    fn expect(&mut self, expected: char, what: &'static str) -> Result<(), JsonError> {
        match self.peek() {
            Some(found) if found == expected => {
                self.cursor += 1;
                Ok(())
            }
            Some(found) => Err(JsonError::UnexpectedCharacter {
                position: self.cursor,
                found,
                expected: what,
            }),
            None => Err(JsonError::UnexpectedEnd),
        }
    }

    fn parse_value(&mut self, depth: usize) -> Result<JsonValue, JsonError> {
        if depth > MAX_DEPTH {
            return Err(JsonError::NestingTooDeep);
        }
        match self.peek() {
            Some('{') => self.parse_object(depth),
            Some('[') => self.parse_array(depth),
            Some('"') => Ok(JsonValue::String(self.parse_string()?)),
            Some('t') => self.parse_literal("true", JsonValue::Bool(true)),
            Some('f') => self.parse_literal("false", JsonValue::Bool(false)),
            Some('n') => self.parse_literal("null", JsonValue::Null),
            Some(c) if c == '-' || c.is_ascii_digit() => self.parse_number(),
            Some(found) => Err(JsonError::UnexpectedCharacter {
                position: self.cursor,
                found,
                expected: "a JSON value",
            }),
            None => Err(JsonError::UnexpectedEnd),
        }
    }

    fn parse_literal(&mut self, word: &str, value: JsonValue) -> Result<JsonValue, JsonError> {
        for expected in word.chars() {
            match self.peek() {
                Some(found) if found == expected => self.cursor += 1,
                Some(found) => {
                    return Err(JsonError::UnexpectedCharacter {
                        position: self.cursor,
                        found,
                        expected: "literal",
                    })
                }
                None => return Err(JsonError::UnexpectedEnd),
            }
        }
        Ok(value)
    }

    fn parse_object(&mut self, depth: usize) -> Result<JsonValue, JsonError> {
        self.expect('{', "'{'")?;
        let mut pairs = Vec::new();
        self.skip_ws();
        if self.peek() == Some('}') {
            self.cursor += 1;
            return Ok(JsonValue::Object(pairs));
        }
        loop {
            self.skip_ws();
            if self.peek() != Some('"') {
                return match self.peek() {
                    Some(found) => Err(JsonError::UnexpectedCharacter {
                        position: self.cursor,
                        found,
                        expected: "an object key",
                    }),
                    None => Err(JsonError::UnexpectedEnd),
                };
            }
            let key = self.parse_string()?;
            self.skip_ws();
            self.expect(':', "':'")?;
            self.skip_ws();
            let value = self.parse_value(depth + 1)?;
            pairs.push((key, value));
            self.skip_ws();
            match self.peek() {
                Some(',') => self.cursor += 1,
                Some('}') => {
                    self.cursor += 1;
                    return Ok(JsonValue::Object(pairs));
                }
                Some(found) => {
                    return Err(JsonError::UnexpectedCharacter {
                        position: self.cursor,
                        found,
                        expected: "',' or '}'",
                    })
                }
                None => return Err(JsonError::UnexpectedEnd),
            }
        }
    }

    fn parse_array(&mut self, depth: usize) -> Result<JsonValue, JsonError> {
        self.expect('[', "'['")?;
        let mut items = Vec::new();
        self.skip_ws();
        if self.peek() == Some(']') {
            self.cursor += 1;
            return Ok(JsonValue::Array(items));
        }
        loop {
            self.skip_ws();
            items.push(self.parse_value(depth + 1)?);
            self.skip_ws();
            match self.peek() {
                Some(',') => self.cursor += 1,
                Some(']') => {
                    self.cursor += 1;
                    return Ok(JsonValue::Array(items));
                }
                Some(found) => {
                    return Err(JsonError::UnexpectedCharacter {
                        position: self.cursor,
                        found,
                        expected: "',' or ']'",
                    })
                }
                None => return Err(JsonError::UnexpectedEnd),
            }
        }
    }

    fn parse_string(&mut self) -> Result<String, JsonError> {
        self.expect('"', "a string")?;
        let mut out = String::new();
        loop {
            match self.peek() {
                Some('"') => {
                    self.cursor += 1;
                    return Ok(out);
                }
                Some('\\') => {
                    self.cursor += 1;
                    match self.peek() {
                        Some('"') => out.push('"'),
                        Some('\\') => out.push('\\'),
                        Some('/') => out.push('/'),
                        Some('b') => out.push('\u{8}'),
                        Some('f') => out.push('\u{c}'),
                        Some('n') => out.push('\n'),
                        Some('r') => out.push('\r'),
                        Some('t') => out.push('\t'),
                        Some('u') => {
                            self.cursor += 1;
                            let first = self.parse_hex4()?;
                            let code = if (0xD800..0xDC00).contains(&first) {
                                // High surrogate: require a following low surrogate.
                                if self.peek() != Some('\\') {
                                    return Err(JsonError::InvalidEscape(
                                        "lone high surrogate".to_string(),
                                    ));
                                }
                                self.cursor += 1;
                                if self.peek() != Some('u') {
                                    return Err(JsonError::InvalidEscape(
                                        "lone high surrogate".to_string(),
                                    ));
                                }
                                self.cursor += 1;
                                let second = self.parse_hex4()?;
                                if !(0xDC00..0xE000).contains(&second) {
                                    return Err(JsonError::InvalidEscape(
                                        "invalid low surrogate".to_string(),
                                    ));
                                }
                                0x1_0000 + ((first - 0xD800) << 10) + (second - 0xDC00)
                            } else if (0xDC00..0xE000).contains(&first) {
                                return Err(JsonError::InvalidEscape(
                                    "lone low surrogate".to_string(),
                                ));
                            } else {
                                first
                            };
                            out.push(char::from_u32(code).ok_or_else(|| {
                                JsonError::InvalidEscape("invalid code point".to_string())
                            })?);
                            // The \u sequence already consumed everything up
                            // to (not including) the next character; cancel
                            // the shared advance below.
                            self.cursor -= 1;
                        }
                        Some(found) => return Err(JsonError::InvalidEscape(format!("\\{found}"))),
                        None => return Err(JsonError::UnexpectedEnd),
                    }
                    self.cursor += 1;
                }
                Some(character) => {
                    out.push(character);
                    self.cursor += 1;
                }
                None => return Err(JsonError::UnexpectedEnd),
            }
        }
    }

    fn parse_hex4(&mut self) -> Result<u32, JsonError> {
        let mut value = 0u32;
        for _ in 0..4 {
            let digit = match self.peek() {
                Some(c) if c.is_ascii_hexdigit() => {
                    self.cursor += 1;
                    c.to_digit(16).expect("hexdigit checked")
                }
                Some(found) => {
                    return Err(JsonError::InvalidEscape(format!("\\u…{found}")));
                }
                None => return Err(JsonError::UnexpectedEnd),
            };
            value = value * 16 + digit;
        }
        Ok(value)
    }

    fn parse_number(&mut self) -> Result<JsonValue, JsonError> {
        let start = self.cursor;
        while let Some(c) = self.peek() {
            if c.is_ascii_digit() || matches!(c, '-' | '+' | '.' | 'e' | 'E') {
                self.cursor += 1;
            } else {
                break;
            }
        }
        let text: String = self.chars[start..self.cursor].iter().collect();
        let value = text
            .parse::<f64>()
            .map_err(|_| JsonError::UnexpectedCharacter {
                position: start,
                found: self.chars[start],
                expected: "a JSON number",
            })?;
        Ok(JsonValue::Number(value))
    }
}

/// Quoted, escaped JSON string literal.
fn write_escaped(value: &str, out: &mut String) {
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
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parse_nested_document_with_all_types() {
        let input = r#"{
            "name": "level1",
            "hp": 100.5,
            "flags": [true, false, null],
            "nested": { "x": -1, "tags": ["a", "b"] }
        }"#;
        let value = JsonValue::parse(input).expect("valid document");
        assert_eq!(
            value.get("name").and_then(JsonValue::as_str),
            Some("level1")
        );
        assert_eq!(value.get("hp").and_then(JsonValue::as_f64), Some(100.5));
        let flags = value
            .get("flags")
            .and_then(JsonValue::as_array)
            .expect("flags");
        assert_eq!(flags.len(), 3);
        assert_eq!(flags[0].as_bool(), Some(true));
        assert!(flags[2].as_f64().is_none() && flags[2].as_str().is_none());
        let x = value
            .get("nested")
            .and_then(|n| n.get("x"))
            .and_then(JsonValue::as_f64);
        assert_eq!(x, Some(-1.0));
        let tags = value
            .get("nested")
            .and_then(|n| n.get("tags"))
            .and_then(JsonValue::as_array)
            .expect("tags");
        assert_eq!(tags.len(), 2);
    }

    #[test]
    fn parse_string_escapes_including_unicode() {
        let parsed = JsonValue::parse(r#""quote\" back\\ newline\n emoji \uD83D\uDE00""#)
            .expect("valid string");
        assert_eq!(
            parsed.as_str(),
            Some("quote\" back\\ newline\n emoji \u{1F600}")
        );
    }

    #[test]
    fn parse_rejects_lone_surrogates_and_bad_escapes() {
        assert!(matches!(
            JsonValue::parse(r#""bad \uD83D tail""#),
            Err(JsonError::InvalidEscape(_))
        ));
        assert!(matches!(
            JsonValue::parse(r#""bad \q""#),
            Err(JsonError::InvalidEscape(_))
        ));
    }

    #[test]
    fn round_trip_through_compact_serializer() {
        let original = JsonValue::parse(
            r#"{"name":"hero \"quoted\"","values":[1,-2.5,3e2],"empty":{},"list":[]}"#,
        )
        .expect("valid");
        let reparsed = JsonValue::parse(&original.to_json()).expect("round trip");
        assert_eq!(original, reparsed);
        assert!(original.to_json().contains("\"values\":[1,-2.5,300"));
    }

    #[test]
    fn non_finite_numbers_serialize_as_null() {
        let value = JsonValue::Number(f64::NAN);
        assert_eq!(value.to_json(), "null");
    }

    #[test]
    fn malformed_documents_report_positions() {
        assert!(matches!(
            JsonValue::parse("{"),
            Err(JsonError::UnexpectedEnd)
        ));
        assert!(matches!(
            JsonValue::parse("{\"a\":1} extra"),
            Err(JsonError::TrailingContent(_))
        ));
        assert!(matches!(
            JsonValue::parse("[1 2]"),
            Err(JsonError::UnexpectedCharacter {
                expected: "',' or ']'",
                ..
            })
        ));
    }

    #[test]
    fn nesting_beyond_the_limit_is_rejected() {
        let deep = "[".repeat(MAX_DEPTH + 2) + &"]".repeat(MAX_DEPTH + 2);
        assert!(matches!(
            JsonValue::parse(&deep),
            Err(JsonError::NestingTooDeep)
        ));
        let ok = "[".repeat(MAX_DEPTH) + &"]".repeat(MAX_DEPTH);
        assert!(JsonValue::parse(&ok).is_ok(), "exactly MAX_DEPTH passes");
    }

    #[test]
    fn duplicate_keys_keep_the_last_occurrence() {
        let value = JsonValue::parse(r#"{"a":1,"a":2}"#).expect("valid");
        assert_eq!(value.get("a").and_then(JsonValue::as_f64), Some(2.0));
    }
}

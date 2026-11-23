//! `#[derive(OnyxReflect)]` for the `onyx-reflect` runtime (`docs/PRD.md` §1).
//!
//! The macro is hand-rolled on the built-in `proc_macro` crate — no `syn`,
//! no `quote` — keeping the whole workspace dependency-free. It supports
//! non-generic structs with named fields whose types are OnyxReflect leaf
//! scalars: `f32`, `f64`, `u32`, `u64`, `i32`, `i64`, `bool` and `String`.
//! Anything else fails with a `compile_error!` naming the offending field.

use proc_macro::{Delimiter, TokenStream, TokenTree};

/// Derives `onyx_reflect::Reflect` for a named-field struct of scalar fields.
#[proc_macro_derive(OnyxReflect)]
pub fn derive_onyx_reflect(input: TokenStream) -> TokenStream {
    match parse_struct(input).and_then(|structure| expand(&structure)) {
        Ok(output) => output,
        Err(error) => compile_error(&error),
    }
}

struct ReflectStruct {
    name: String,
    /// (field name, leaf type name) pairs in declaration order.
    fields: Vec<(String, String)>,
}

fn compile_error(message: &str) -> TokenStream {
    format!("compile_error!({message:?});")
        .parse()
        .expect("valid compile_error invocation")
}

fn parse_struct(input: TokenStream) -> Result<ReflectStruct, String> {
    let tokens: Vec<TokenTree> = input.into_iter().collect();

    // Scan past attributes, visibility and anything else up to `struct`.
    let mut index = 0;
    while index < tokens.len() {
        if let TokenTree::Ident(ident) = &tokens[index] {
            if ident.to_string() == "struct" {
                break;
            }
        }
        index += 1;
    }
    if index >= tokens.len() {
        return Err("OnyxReflect can only be derived for structs".to_string());
    }

    let name = match tokens.get(index + 1) {
        Some(TokenTree::Ident(ident)) => ident.to_string(),
        _ => return Err("expected struct name after `struct`".to_string()),
    };
    if let Some(TokenTree::Punct(punct)) = tokens.get(index + 2) {
        if punct.as_char() == '<' {
            return Err(format!(
                "OnyxReflect does not support generic structs (`{name}`)"
            ));
        }
    }

    let body = tokens[index + 2..]
        .iter()
        .find_map(|token| match token {
            TokenTree::Group(group) if group.delimiter() == Delimiter::Brace => {
                Some(group.stream())
            }
            _ => None,
        })
        .ok_or_else(|| format!("expected a named-field struct body for `{name}`"))?;

    let body_tokens: Vec<TokenTree> = body.into_iter().collect();
    let supported = ["f32", "f64", "u32", "u64", "i32", "i64", "bool", "String"];
    let mut fields = Vec::new();
    let mut cursor = 0;
    while cursor < body_tokens.len() {
        match &body_tokens[cursor] {
            // Attribute: `#` followed by a bracketed group.
            TokenTree::Punct(punct) if punct.as_char() == '#' => {
                cursor += 2;
            }
            // Visibility: `pub` with an optional `(...)` scope.
            TokenTree::Ident(ident) if ident.to_string() == "pub" => {
                cursor += 1;
                if let Some(TokenTree::Group(group)) = body_tokens.get(cursor) {
                    if group.delimiter() == Delimiter::Parenthesis {
                        cursor += 1;
                    }
                }
            }
            TokenTree::Ident(ident) => {
                let field = ident.to_string();
                match body_tokens.get(cursor + 1) {
                    Some(TokenTree::Punct(punct)) if punct.as_char() == ':' => {}
                    _ => return Err(format!("expected ':' after field `{field}` in `{name}`")),
                }
                let mut type_parts = Vec::new();
                let mut scan = cursor + 2;
                while scan < body_tokens.len() {
                    match &body_tokens[scan] {
                        TokenTree::Punct(punct) if punct.as_char() == ',' => break,
                        token => type_parts.push(token.to_string()),
                    }
                    scan += 1;
                }
                let joined = type_parts.concat();
                let leaf = joined.rsplit("::").next().unwrap_or(&joined).to_string();
                if !supported.contains(&leaf.as_str()) {
                    return Err(format!(
                        "field `{field}` of type `{joined}` is not a supported OnyxReflect leaf type (f32, f64, u32, u64, i32, i64, bool, String)"
                    ));
                }
                fields.push((field, leaf));
                cursor = scan + 1;
            }
            other => {
                return Err(format!("unexpected token in struct `{name}` body: {other}"));
            }
        }
    }

    if fields.is_empty() {
        return Err(format!(
            "OnyxReflect needs at least one named field on `{name}`"
        ));
    }
    Ok(ReflectStruct { name, fields })
}

/// `FieldValue` variant name for a leaf type (`bool` maps to `Bool`, `f32`
/// maps to `F32`, ...; `String` maps to itself).
fn variant_name(leaf: &str) -> &str {
    match leaf {
        "f32" => "F32",
        "f64" => "F64",
        "u32" => "U32",
        "u64" => "U64",
        "i32" => "I32",
        "i64" => "I64",
        "bool" => "Bool",
        "String" => "String",
        other => other,
    }
}

fn expand(structure: &ReflectStruct) -> Result<TokenStream, String> {
    let name = &structure.name;

    let schema_fields = structure
        .fields
        .iter()
        .map(|(field, leaf)| {
            format!(
                "                onyx_reflect::FieldSchema {{ name: \"{field}\", type_name: \"{leaf}\" }},"
            )
        })
        .collect::<Vec<_>>()
        .join("\n");

    let get_arms = structure
        .fields
        .iter()
        .map(|(field, leaf)| {
            let expression = match leaf.as_str() {
                "String" => format!("onyx_reflect::FieldValue::String(self.{field}.clone())"),
                other => {
                    format!(
                        "onyx_reflect::FieldValue::{}(self.{field})",
                        variant_name(other)
                    )
                }
            };
            format!("            \"{field}\" => Some({expression}),")
        })
        .collect::<Vec<_>>()
        .join("\n");

    let set_arms = structure
        .fields
        .iter()
        .map(|(field, leaf)| {
            // `value` is `&FieldValue`, so bindings are references: scalars
            // are Copy and assign via `*inner`; `String` needs a clone.
            let assignment = match leaf.as_str() {
                "String" => format!("self.{field} = inner.clone();"),
                _ => format!("self.{field} = *inner;"),
            };
            format!(
                "            \"{field}\" => match value {{\n                onyx_reflect::FieldValue::{}(inner) => {{ {assignment} true }}\n                _ => false,\n            }},",
                variant_name(leaf)
            )
        })
        .collect::<Vec<_>>()
        .join("\n");

    let mut code = String::new();
    code.push_str(&format!("impl onyx_reflect::Reflect for {name} {{\n"));
    code.push_str("    fn schema() -> onyx_reflect::Schema {\n");
    code.push_str("        onyx_reflect::Schema {\n");
    code.push_str(&format!("            type_name: \"{name}\",\n"));
    code.push_str("            fields: vec![\n");
    code.push_str(&schema_fields);
    code.push_str("\n            ],\n");
    code.push_str("        }\n");
    code.push_str("    }\n");
    code.push_str("    fn get(&self, field: &str) -> Option<onyx_reflect::FieldValue> {\n");
    code.push_str("        match field {\n");
    code.push_str(&get_arms);
    code.push_str("\n            _ => None,\n");
    code.push_str("        }\n");
    code.push_str("    }\n");
    code.push_str(
        "    fn set(&mut self, field: &str, value: &onyx_reflect::FieldValue) -> bool {\n",
    );
    code.push_str("        match field {\n");
    code.push_str(&set_arms);
    code.push_str("\n            _ => false,\n");
    code.push_str("        }\n");
    code.push_str("    }\n");
    code.push_str("}\n");

    code.parse()
        .map_err(|_| format!("failed to expand OnyxReflect for `{name}`"))
}

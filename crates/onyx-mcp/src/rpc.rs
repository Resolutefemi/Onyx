//! JSON-RPC 2.0 envelope for the control plane (`docs/PRD.md` §16).
//!
//! The plane speaks JSON-RPC 2.0 in-process so the M3 transport (`rmcp`)
//! can map MCP tool calls onto it one-to-one; the envelope parsing and
//! error codes are tested here without any transport.

use serde_json::{json, Map, Value};

/// JSON-RPC 2.0: parse error (-32700).
pub const PARSE_ERROR: i64 = -32700;
/// JSON-RPC 2.0: invalid request envelope (-32600).
pub const INVALID_REQUEST: i64 = -32600;
/// JSON-RPC 2.0: unknown method (-32601).
pub const METHOD_NOT_FOUND: i64 = -32601;
/// JSON-RPC 2.0: invalid method params (-32602).
pub const INVALID_PARAMS: i64 = -32602;
/// JSON-RPC 2.0: engine-side failure (-32603).
pub const INTERNAL_ERROR: i64 = -32603;

/// A JSON-RPC 2.0 error object.
#[derive(Debug, Clone, PartialEq)]
pub struct RpcError {
    /// Error code (see the [`PARSE_ERROR`]-family constants).
    pub code: i64,
    /// One-line human-readable message.
    pub message: String,
    /// Optional structured detail.
    pub data: Option<Value>,
}

impl RpcError {
    /// An error with a message and no data.
    #[must_use]
    pub fn new(code: i64, message: impl Into<String>) -> Self {
        Self {
            code,
            message: message.into(),
            data: None,
        }
    }

    /// Serialize to the JSON-RPC `error` object.
    #[must_use]
    pub fn to_value(&self) -> Value {
        let mut object = Map::new();
        object.insert("code".into(), json!(self.code));
        object.insert("message".into(), json!(self.message));
        if let Some(data) = &self.data {
            object.insert("data".into(), data.clone());
        }
        Value::Object(object)
    }
}

/// A parsed request: id, method, params (default `Null`).
#[derive(Debug, Clone, PartialEq)]
pub struct RpcRequest {
    /// The request id, echoed back in the response.
    pub id: Value,
    /// The method name.
    pub method: String,
    /// The method params, `Null` when absent.
    pub params: Value,
}

/// Parse a request string: valid JSON, `jsonrpc: "2.0"`, string method,
/// present id. Notifications (no id) are rejected — the plane is
/// request/response only.
///
/// # Errors
/// [`RpcError`] with the JSON-RPC code naming the first broken rule.
pub fn parse_request(input: &str) -> Result<RpcRequest, RpcError> {
    let value: Value = serde_json::from_str(input)
        .map_err(|_| RpcError::new(PARSE_ERROR, "request is not valid JSON"))?;
    parse_request_value(&value)
}

/// Parse an already-decoded request object.
///
/// # Errors
/// [`RpcError`] with the JSON-RPC code naming the first broken rule.
pub fn parse_request_value(value: &Value) -> Result<RpcRequest, RpcError> {
    let object = value
        .as_object()
        .ok_or_else(|| RpcError::new(INVALID_REQUEST, "request must be an object"))?;
    if object.get("jsonrpc").and_then(Value::as_str) != Some("2.0") {
        return Err(RpcError::new(INVALID_REQUEST, "jsonrpc must be \"2.0\""));
    }
    let method = object
        .get("method")
        .and_then(Value::as_str)
        .ok_or_else(|| RpcError::new(INVALID_REQUEST, "method must be a string"))?;
    let id = object.get("id").cloned().ok_or_else(|| {
        RpcError::new(
            INVALID_REQUEST,
            "notifications are not supported: id is required",
        )
    })?;
    Ok(RpcRequest {
        id,
        method: method.to_owned(),
        params: object.get("params").cloned().unwrap_or(Value::Null),
    })
}

/// A success response for `id`.
#[must_use]
pub fn success(id: &Value, result: Value) -> Value {
    json!({ "jsonrpc": "2.0", "id": id, "result": result })
}

/// An error response for `id`.
#[must_use]
pub fn failure(id: &Value, error: &RpcError) -> Value {
    json!({ "jsonrpc": "2.0", "id": id, "error": error.to_value() })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parse_accepts_well_formed_requests() {
        let request = parse_request(
            r#"{"jsonrpc":"2.0","id":7,"method":"batch_execute","params":{"atomic":true}}"#,
        )
        .expect("valid");
        assert_eq!(request.id, json!(7));
        assert_eq!(request.method, "batch_execute");
        assert_eq!(request.params, json!({"atomic": true}));
    }

    #[test]
    fn parse_defaults_params_to_null() {
        let request =
            parse_request(r#"{"jsonrpc":"2.0","id":"a","method":"tools/list"}"#).expect("valid");
        assert_eq!(request.params, Value::Null);
        assert_eq!(request.id, json!("a"));
    }

    #[test]
    fn parse_rejects_broken_envelopes_with_the_right_codes() {
        let cases: [(&str, i64); 5] = [
            ("not json", PARSE_ERROR),
            ("[]", INVALID_REQUEST),
            (r#"{"id":1,"method":"x"}"#, INVALID_REQUEST),
            (r#"{"jsonrpc":"1.0","id":1,"method":"x"}"#, INVALID_REQUEST),
            (r#"{"jsonrpc":"2.0","id":1}"#, INVALID_REQUEST),
        ];
        for (input, code) in cases {
            let error = parse_request(input).expect_err(input);
            assert_eq!(error.code, code, "input {input}");
        }
    }

    #[test]
    fn envelopes_round_trip_through_values() {
        let error = RpcError::new(INVALID_PARAMS, "bad params");
        let response = failure(&json!(3), &error);
        assert_eq!(response["jsonrpc"], "2.0");
        assert_eq!(response["id"], 3);
        assert_eq!(response["error"]["code"], INVALID_PARAMS);
        assert_eq!(response["error"]["message"], "bad params");

        let response = success(&json!(3), json!({"ok": true}));
        assert_eq!(response["result"]["ok"], true);
        assert!(response.get("error").is_none());
    }
}

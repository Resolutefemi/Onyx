//! Tagged binary serialization for [`FieldValue`] (`docs/PRD.md` §1,
//! "automatic JSON/MessagePack/Binary serialization").
//!
//! A compact self-describing format: one tag byte per value, little-endian
//! payloads. It serves deterministic replay logs and save-game slots now;
//! MessagePack lands with the asset pipeline (§15) where interop with
//! external tools demands it.

use std::collections::BTreeMap;

use super::{FieldValue, ReflectError};

const TAG_F32: u8 = 0;
const TAG_F64: u8 = 1;
const TAG_U32: u8 = 2;
const TAG_U64: u8 = 3;
const TAG_I32: u8 = 4;
const TAG_I64: u8 = 5;
const TAG_BOOL: u8 = 6;
const TAG_STRING: u8 = 7;

/// Serialize one value.
#[must_use]
pub fn to_binary(value: &FieldValue) -> Vec<u8> {
    let mut out = Vec::new();
    write_value(&mut out, value);
    out
}

/// Parse one value from the start of `bytes`; trailing bytes are ignored
/// (maps encode a count up front, so this stays total).
///
/// # Errors
/// [`ReflectError::InvalidBinary`] on truncation or an unknown tag.
pub fn from_binary(bytes: &[u8]) -> Result<FieldValue, ReflectError> {
    let (value, _) = read_value(bytes, 0)?;
    Ok(value)
}

/// Serialize a name→value map: `u32` count, then (name, value) pairs.
#[must_use]
pub fn encode_map(map: &BTreeMap<String, FieldValue>) -> Vec<u8> {
    let mut out = Vec::new();
    out.extend_from_slice(&u32::try_from(map.len()).unwrap_or(u32::MAX).to_le_bytes());
    for (name, value) in map {
        write_string(&mut out, name);
        write_value(&mut out, value);
    }
    out
}

/// Parse a map encoded by [`encode_map`]; the whole input must be consumed.
///
/// # Errors
/// [`ReflectError::InvalidBinary`] on truncation, an unknown tag, or
/// trailing bytes.
pub fn decode_map(bytes: &[u8]) -> Result<BTreeMap<String, FieldValue>, ReflectError> {
    let (count, mut pos) = read_u32(bytes, 0)?;
    let mut map = BTreeMap::new();
    for _ in 0..count {
        let (name, next) = read_string(bytes, pos)?;
        pos = next;
        let (value, next) = read_value(bytes, pos)?;
        pos = next;
        map.insert(name, value);
    }
    if pos != bytes.len() {
        return Err(ReflectError::InvalidBinary(
            "trailing bytes after map".to_string(),
        ));
    }
    Ok(map)
}

fn write_value(out: &mut Vec<u8>, value: &FieldValue) {
    match value {
        FieldValue::F32(v) => {
            out.push(TAG_F32);
            out.extend_from_slice(&v.to_le_bytes());
        }
        FieldValue::F64(v) => {
            out.push(TAG_F64);
            out.extend_from_slice(&v.to_le_bytes());
        }
        FieldValue::U32(v) => {
            out.push(TAG_U32);
            out.extend_from_slice(&v.to_le_bytes());
        }
        FieldValue::U64(v) => {
            out.push(TAG_U64);
            out.extend_from_slice(&v.to_le_bytes());
        }
        FieldValue::I32(v) => {
            out.push(TAG_I32);
            out.extend_from_slice(&v.to_le_bytes());
        }
        FieldValue::I64(v) => {
            out.push(TAG_I64);
            out.extend_from_slice(&v.to_le_bytes());
        }
        FieldValue::Bool(v) => {
            out.push(TAG_BOOL);
            out.push(u8::from(*v));
        }
        FieldValue::String(v) => {
            out.push(TAG_STRING);
            write_string(out, v);
        }
    }
}

fn write_string(out: &mut Vec<u8>, value: &str) {
    out.extend_from_slice(&u32::try_from(value.len()).unwrap_or(u32::MAX).to_le_bytes());
    out.extend_from_slice(value.as_bytes());
}

fn read_u32(bytes: &[u8], pos: usize) -> Result<(u32, usize), ReflectError> {
    let end = pos
        .checked_add(4)
        .ok_or_else(|| ReflectError::InvalidBinary("position overflow".to_string()))?;
    if end > bytes.len() {
        return Err(ReflectError::InvalidBinary("truncated u32".to_string()));
    }
    let raw: [u8; 4] = bytes[pos..end]
        .try_into()
        .map_err(|_| ReflectError::InvalidBinary("u32 slice".to_string()))?;
    Ok((u32::from_le_bytes(raw), end))
}

fn read_string(bytes: &[u8], pos: usize) -> Result<(String, usize), ReflectError> {
    let (len, mut pos) = read_u32(bytes, pos)?;
    let end = pos
        .checked_add(len as usize)
        .ok_or_else(|| ReflectError::InvalidBinary("string length overflow".to_string()))?;
    if end > bytes.len() {
        return Err(ReflectError::InvalidBinary("truncated string".to_string()));
    }
    let value = String::from_utf8(bytes[pos..end].to_vec())
        .map_err(|_| ReflectError::InvalidBinary("non-utf8 string".to_string()))?;
    pos = end;
    Ok((value, pos))
}

fn read_value(bytes: &[u8], pos: usize) -> Result<(FieldValue, usize), ReflectError> {
    let Some(&tag) = bytes.get(pos) else {
        return Err(ReflectError::InvalidBinary("missing tag".to_string()));
    };
    let pos = pos + 1;
    let fixed = |size: usize| -> Result<(&[u8], usize), ReflectError> {
        let end = pos + size;
        if end > bytes.len() {
            return Err(ReflectError::InvalidBinary("truncated payload".to_string()));
        }
        Ok((&bytes[pos..end], end))
    };
    let le = |slice: &[u8]| -> [u8; 8] {
        let mut out = [0u8; 8];
        out[..slice.len()].copy_from_slice(slice);
        out
    };
    match tag {
        TAG_F32 => {
            let (payload, next) = fixed(4)?;
            Ok((
                FieldValue::F32(f32::from_le_bytes(
                    le(payload)[..4].try_into().expect("4 bytes"),
                )),
                next,
            ))
        }
        TAG_F64 => {
            let (payload, next) = fixed(8)?;
            Ok((FieldValue::F64(f64::from_le_bytes(le(payload))), next))
        }
        TAG_U32 => {
            let (payload, next) = fixed(4)?;
            Ok((
                FieldValue::U32(u32::from_le_bytes(
                    le(payload)[..4].try_into().expect("4 bytes"),
                )),
                next,
            ))
        }
        TAG_U64 => {
            let (payload, next) = fixed(8)?;
            Ok((FieldValue::U64(u64::from_le_bytes(le(payload))), next))
        }
        TAG_I32 => {
            let (payload, next) = fixed(4)?;
            Ok((
                FieldValue::I32(i32::from_le_bytes(
                    le(payload)[..4].try_into().expect("4 bytes"),
                )),
                next,
            ))
        }
        TAG_I64 => {
            let (payload, next) = fixed(8)?;
            Ok((FieldValue::I64(i64::from_le_bytes(le(payload))), next))
        }
        TAG_BOOL => {
            let (payload, next) = fixed(1)?;
            Ok((FieldValue::Bool(payload[0] != 0), next))
        }
        TAG_STRING => {
            let (value, next) = read_string(bytes, pos)?;
            Ok((FieldValue::String(value), next))
        }
        other => Err(ReflectError::InvalidBinary(format!("unknown tag {other}"))),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn every_variant_round_trips() {
        let values = [
            FieldValue::F32(1.5),
            FieldValue::F64(-2.25e10),
            FieldValue::U32(u32::MAX),
            FieldValue::U64(u64::MAX),
            FieldValue::I32(i32::MIN),
            FieldValue::I64(i64::MIN),
            FieldValue::Bool(true),
            FieldValue::Bool(false),
            FieldValue::String("héllo \u{1F600}".to_string()),
        ];
        for value in values {
            let bytes = to_binary(&value);
            assert_eq!(from_binary(&bytes).expect("round trip"), value);
        }
    }

    #[test]
    fn tags_and_widths_are_exact() {
        assert_eq!(to_binary(&FieldValue::U32(1)), vec![TAG_U32, 1, 0, 0, 0]);
        assert_eq!(to_binary(&FieldValue::Bool(true)), vec![TAG_BOOL, 1]);
        assert_eq!(
            to_binary(&FieldValue::String("ab".to_string())).len(),
            1 + 4 + 2
        );
    }

    #[test]
    fn map_round_trips_deterministically() {
        let mut map = BTreeMap::new();
        map.insert("hp".to_string(), FieldValue::F64(100.0));
        map.insert("name".to_string(), FieldValue::String("hero".to_string()));
        map.insert("alive".to_string(), FieldValue::Bool(true));
        let bytes = encode_map(&map);
        assert_eq!(decode_map(&bytes).expect("round trip"), map);
    }

    #[test]
    fn truncated_and_unknown_payloads_are_rejected() {
        let bytes = to_binary(&FieldValue::U64(7));
        assert!(from_binary(&bytes[..3]).is_err(), "truncated payload");
        assert!(from_binary(&[9, 0, 0]).is_err(), "unknown tag");
        let mut map_bytes = encode_map(&BTreeMap::from([("a".to_string(), FieldValue::U32(1))]));
        map_bytes.push(0);
        assert!(decode_map(&map_bytes).is_err(), "trailing byte");
    }
}

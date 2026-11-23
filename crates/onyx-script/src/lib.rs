//! # onyx-script
//!
//! Polyglot scripting for the Onyx Engine — `docs/PRD.md` §13.
//!
//! M0 defines the language model and runtime configuration knobs that other
//! crates can already reference. M6 lands the implementation:
//!
//! - `wasmtime` + WebAssembly Component Model (WIT) execution,
//! - per-language toolchains (`cargo-component`, `componentize-python`, `jco`,
//!   `wit-bindgen`, TinyGo, ...) compiled to sandboxed `.wasm` components,
//! - sub-100 ms hot-reload,
//! - zero-overhead native Rust plugins registered directly as ECS systems.

/// Languages targeting the sandboxed `.wasm` component runtime.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub enum ScriptLanguage {
    /// Rust (also available as a native plugin target).
    Rust,
    /// Python.
    Python,
    /// TypeScript.
    TypeScript,
    /// JavaScript.
    JavaScript,
    /// C.
    C,
    /// C++.
    Cpp,
    /// Go.
    Go,
    /// C#.
    CSharp,
    /// Zig.
    Zig,
    /// Lua.
    Lua,
    /// Swift.
    Swift,
}

impl ScriptLanguage {
    /// Human-readable language name.
    #[must_use]
    pub fn as_str(&self) -> &'static str {
        match self {
            Self::Rust => "Rust",
            Self::Python => "Python",
            Self::TypeScript => "TypeScript",
            Self::JavaScript => "JavaScript",
            Self::C => "C",
            Self::Cpp => "C++",
            Self::Go => "Go",
            Self::CSharp => "C#",
            Self::Zig => "Zig",
            Self::Lua => "Lua",
            Self::Swift => "Swift",
        }
    }

    /// Kebab-case id used in WIT world and package naming.
    #[must_use]
    pub fn id(&self) -> &'static str {
        match self {
            Self::Rust => "rust",
            Self::Python => "python",
            Self::TypeScript => "typescript",
            Self::JavaScript => "javascript",
            Self::C => "c",
            Self::Cpp => "cpp",
            Self::Go => "go",
            Self::CSharp => "csharp",
            Self::Zig => "zig",
            Self::Lua => "lua",
            Self::Swift => "swift",
        }
    }

    /// Every supported language.
    #[must_use]
    pub fn all() -> &'static [ScriptLanguage] {
        &[
            Self::Rust,
            Self::Python,
            Self::TypeScript,
            Self::JavaScript,
            Self::C,
            Self::Cpp,
            Self::Go,
            Self::CSharp,
            Self::Zig,
            Self::Lua,
            Self::Swift,
        ]
    }
}

/// Runtime configuration for the WASM component host.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ScriptRuntimeConfig {
    /// Languages compiled into this build.
    pub languages: Vec<ScriptLanguage>,
    /// Whether `.wasm` components hot-reload from disk (sub-100 ms target).
    pub hot_reload: bool,
    /// wasmtime fuel limit per call, bounding runaway scripts; `None` = off.
    pub fuel_limit: Option<u64>,
    /// Maximum concurrently instantiated components.
    pub max_instances: u32,
}

impl Default for ScriptRuntimeConfig {
    fn default() -> Self {
        Self {
            languages: ScriptLanguage::all().to_vec(),
            hot_reload: true,
            fuel_limit: None,
            max_instances: 256,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn language_ids_are_unique() {
        let all = ScriptLanguage::all();
        let mut ids: Vec<&str> = all.iter().map(ScriptLanguage::id).collect();
        ids.sort_unstable();
        let unique = ids.len();
        ids.dedup();
        assert_eq!(unique, ids.len(), "ids must be unique");
        assert_eq!(all.len(), 11, "PRD §13 lists ten languages plus JavaScript");
    }

    #[test]
    fn default_config_enables_every_language() {
        let config = ScriptRuntimeConfig::default();
        assert_eq!(config.languages.len(), ScriptLanguage::all().len());
        assert!(config.hot_reload);
        assert_eq!(config.max_instances, 256);
    }
}

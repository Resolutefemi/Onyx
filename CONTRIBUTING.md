# Contributing to Onyx Engine

Thanks for helping build Onyx! The engine is in **pre-alpha (M0 scaffold
complete)** — the most valuable contributions right now are feedback on
[`docs/PRD.md`](docs/PRD.md), MCP tool-schema review from AI-tooling
developers, and reference implementations of isolated subsystems.

## Prerequisites

- Rust (stable channel) via [rustup](https://rustup.rs). `rust-toolchain.toml`
  pins the exact channel; any `cargo` command inside the repo resolves it.
- No other system dependencies: the M0 workspace is dependency-free.

## Build & test

```sh
cargo build --workspace --all-targets   # compile everything, including tests
cargo test --workspace                  # unit + CLI integration tests
cargo clippy --workspace --all-targets -- -D warnings   # lint gate (CI)
cargo fmt --all -- --check              # format gate (CI)
```

CI (`.github/workflows/ci.yml`) runs fmt + clippy on Linux and
build + test on Linux, Windows and macOS. A PR that fails any of these locally
will fail CI, so run them before pushing.

## Where things live

See [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md) for the full crate map,
dependency rules and milestone plan. In short:

- `crates/onyx-core` — app loop, time, command bus (PRD §1)
- `crates/onyx-ecs` — ECS seed types (PRD §1)
- `crates/onyx-render` — renderer config surface (PRD §2–4)
- `crates/onyx-mcp` — AI control plane data model (PRD §16)
- `crates/onyx-script` — polyglot scripting config (PRD §13)
- `crates/onyx` — umbrella crate + `onyx` CLI

## Working on a milestone

1. Pick (or claim) a milestone item from the README roadmap.
2. Keep new dependencies in `[workspace.dependencies]` (root `Cargo.toml`) and
   inherit with `.workspace = true`.
3. Stub APIs must document which milestone replaces them and link the PRD
   section — no unexplained placeholders.
4. Every public type gets unit tests; CLI behavior gets integration tests.
5. Commit messages: short imperative summary, e.g.
   `M0 scaffold: Cargo workspace (6 crates), GitHub Actions CI, architecture docs`.

## PRD feedback

The PRD is the single source of truth for scope. Open an issue referencing the
section number (e.g. "§16: batch_execute should cap at N commands per frame")
or propose edits directly — PRD sign-off is part of M0.

## License

By contributing you agree that your contributions are licensed under the MIT
License (see [LICENSE](LICENSE)).

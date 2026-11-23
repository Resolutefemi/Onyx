use std::process::Command;

#[test]
fn cli_prints_engine_version() {
    let output = Command::new(env!("CARGO_BIN_EXE_onyx"))
        .arg("--version")
        .output()
        .expect("failed to run onyx --version");
    assert!(output.status.success());
    let stdout = String::from_utf8_lossy(&output.stdout);
    assert!(
        stdout.contains("Onyx Engine"),
        "unexpected output: {stdout}"
    );
}

#[test]
fn cli_info_lists_workspace_crates() {
    let output = Command::new(env!("CARGO_BIN_EXE_onyx"))
        .arg("info")
        .output()
        .expect("failed to run onyx info");
    assert!(output.status.success());
    let stdout = String::from_utf8_lossy(&output.stdout);
    assert!(stdout.contains("onyx-core"), "unexpected output: {stdout}");
    assert!(stdout.contains("onyx-mcp"), "unexpected output: {stdout}");
}

#[test]
fn cli_rejects_unknown_commands() {
    let output = Command::new(env!("CARGO_BIN_EXE_onyx"))
        .arg("definitely-not-a-command")
        .output()
        .expect("failed to run onyx");
    assert_eq!(output.status.code(), Some(2));
}

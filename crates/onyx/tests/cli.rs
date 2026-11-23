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
fn cli_sim_runs_and_reports() {
    let output = Command::new(env!("CARGO_BIN_EXE_onyx"))
        .args(["sim", "--entities", "50", "--frames", "10"])
        .output()
        .expect("failed to run onyx sim");
    assert!(output.status.success());
    let stdout = String::from_utf8_lossy(&output.stdout);
    assert!(
        stdout.contains("50 entities x 10 frames"),
        "unexpected: {stdout}"
    );
    assert!(stdout.contains("ticks"), "unexpected: {stdout}");
    assert!(stdout.contains("checksum"), "unexpected: {stdout}");
    assert!(stdout.contains("sample"), "unexpected: {stdout}");
    assert!(
        stdout.contains("\"x\""),
        "reflected position JSON missing from sim output: {stdout}"
    );
}

#[test]
fn cli_sim_rejects_unknown_flags() {
    let output = Command::new(env!("CARGO_BIN_EXE_onyx"))
        .args(["sim", "--bogus"])
        .output()
        .expect("failed to run onyx sim");
    assert_eq!(output.status.code(), Some(2));
}

#[test]
fn cli_rejects_unknown_commands() {
    let output = Command::new(env!("CARGO_BIN_EXE_onyx"))
        .arg("definitely-not-a-command")
        .output()
        .expect("failed to run onyx");
    assert_eq!(output.status.code(), Some(2));
}

//! `onyx` — the Onyx Engine CLI.

use onyx::{CRATES, ENGINE_NAME, ENGINE_VERSION};

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    match args.first().map(String::as_str) {
        Some("info") => print_info(),
        Some("-V" | "--version") => println!("{ENGINE_NAME} {ENGINE_VERSION}"),
        Some("-h" | "--help") => print_help(),
        Some(other) => {
            eprintln!("unknown command: {other}");
            print_help();
            std::process::exit(2);
        }
        None => print_help(),
    }
}

fn print_info() {
    println!("{ENGINE_NAME} v{ENGINE_VERSION}");
    println!("status: pre-alpha (M0 scaffold complete)");
    println!();
    println!("Workspace crates:");
    for (name, scope) in CRATES {
        println!("  {name:<14} {scope}");
    }
}

fn print_help() {
    println!("usage: onyx [info | --version | --help]");
    println!();
    println!("  info       print engine version and workspace crate map");
    println!("  --version  print the engine version");
    println!("  --help     print this help");
}

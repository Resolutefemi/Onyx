//! `onyx` — the Onyx Engine CLI.

use onyx::{CRATES, ENGINE_NAME, ENGINE_VERSION};

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    match args.first().map(String::as_str) {
        Some("info") => print_info(),
        Some("sim") => run_sim_command(&args[1..]),
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
    println!("status: pre-alpha (M0 and M1 complete; renderer next)");
    println!();
    println!("Workspace crates:");
    for (name, scope) in CRATES {
        println!("  {name:<14} {scope}");
    }
}

fn run_sim_command(rest: &[String]) {
    let mut entities: u32 = 1_000;
    let mut frames: u64 = 600;
    let mut index = 0;
    while index < rest.len() {
        match rest[index].as_str() {
            "--entities" => {
                index += 1;
                entities = rest
                    .get(index)
                    .and_then(|v| v.parse().ok())
                    .unwrap_or_else(|| {
                        eprintln!("--entities needs a number");
                        std::process::exit(2);
                    });
            }
            "--frames" => {
                index += 1;
                frames = rest
                    .get(index)
                    .and_then(|v| v.parse().ok())
                    .unwrap_or_else(|| {
                        eprintln!("--frames needs a number");
                        std::process::exit(2);
                    });
            }
            other => {
                eprintln!("unknown sim flag: {other}");
                print_help();
                std::process::exit(2);
            }
        }
        index += 1;
    }

    let report = onyx::run_sim(entities, frames);
    println!("{ENGINE_NAME} sim: {entities} entities x {frames} frames");
    println!("  ticks      {}", report.ticks);
    println!("  checksum   {:.3}", report.checksum);
    println!("  sample     {}", report.sample_json);
    println!("  elapsed    {:.1} ms", report.elapsed_ms);
}

fn print_help() {
    println!("usage: onyx [info | sim | --version | --help]");
    println!();
    println!("  info       print engine version and workspace crate map");
    println!("  sim        run the archetype-ECS motion benchmark");
    println!("             (--entities N, --frames F; defaults 1000 x 600)");
    println!("  --version  print the engine version");
    println!("  --help     print this help");
}

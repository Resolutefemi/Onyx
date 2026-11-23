//! `onyx` — the Onyx Engine CLI.

use onyx::{
    render_offscreen, RenderGraph, RenderGraphNode, Renderer, RendererConfig, CRATES, ENGINE_NAME,
    ENGINE_VERSION,
};

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    match args.first().map(String::as_str) {
        Some("info") => print_info(),
        Some("sim") => run_sim_command(&args[1..]),
        Some("render") => run_render_command(),
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
    println!("status: pre-alpha (M0 and M1 complete; M2 renderer in progress)");
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

fn run_render_command() {
    println!("{ENGINE_NAME} render probe");
    match Renderer::try_new(&RendererConfig::default()) {
        Ok(renderer) => {
            let info = renderer.adapter_info();
            println!("  adapter    {}", info.name);
            println!("  backend    {:?}", info.backend);

            let mut graph = RenderGraph::new();
            let declarations: [(&str, &[&str]); 4] = [
                ("red", &[]),
                ("green", &[]),
                ("blue", &[]),
                ("composite", &["red", "green", "blue"]),
            ];
            for (name, after) in declarations {
                graph
                    .try_add_node(RenderGraphNode {
                        name,
                        after: after.to_vec(),
                    })
                    .expect("demo nodes are unique and declared in order");
            }
            let order = graph.execution_order().expect("demo graph is acyclic");
            println!("  graph      {} nodes → {order:?}", graph.node_count());

            let colors: [(&str, [f64; 4]); 4] = [
                ("red", [0.8, 0.1, 0.1, 1.0]),
                ("green", [0.1, 0.8, 0.2, 1.0]),
                ("blue", [0.15, 0.25, 0.85, 1.0]),
                ("composite", [0.5, 0.5, 0.5, 1.0]),
            ];
            for (name, color) in colors {
                match render_offscreen(&renderer, 64, 64, color) {
                    Ok(frame) => println!(
                        "  node {name:<10} 64x64 clear checksum {:016x}",
                        frame.checksum()
                    ),
                    Err(error) => println!("  node {name:<10} render failed: {error}"),
                }
            }
        }
        Err(error) => {
            println!("  adapter    unavailable: {error}");
            println!("  offscreen  skipped (headless machine or no GPU driver)");
        }
    }
}

fn print_help() {
    println!("usage: onyx [info | sim | render | --version | --help]");
    println!();
    println!("  info       print engine version and workspace crate map");
    println!("  sim        run the archetype-ECS motion benchmark");
    println!("             (--entities N, --frames F; defaults 1000 x 600)");
    println!("  render     probe the GPU, render an offscreen graph");
    println!("  --version  print the engine version");
    println!("  --help     print this help");
}

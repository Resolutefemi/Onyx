//! `onyx` — the Onyx Engine CLI.

use onyx::{
    execute_offscreen, PassKind, RenderGraph, RenderGraphNode, Renderer, RendererConfig,
    ResourceSpec, CRATES, ENGINE_NAME, ENGINE_VERSION,
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
            for node in [
                RenderGraphNode {
                    name: "triangle",
                    reads: &[],
                    writes: &["scene"],
                },
                RenderGraphNode {
                    name: "clear",
                    reads: &[],
                    writes: &["post"],
                },
            ] {
                graph.try_add_node(node).expect("demo nodes are unique");
            }
            let order = graph.execution_order().expect("demo graph is acyclic");
            println!(
                "  graph      {} nodes → {order:?} (2 targets)",
                graph.node_count()
            );

            let resources = [
                ResourceSpec::new("scene", 64, 64),
                ResourceSpec::new("post", 64, 64),
            ];
            let kinds = |name: &str| match name {
                "triangle" => Some(PassKind::Triangle {
                    triangle: [1.0, 0.35, 0.15, 1.0],
                    background: [0.05, 0.05, 0.1, 1.0],
                }),
                "clear" => Some(PassKind::Clear([0.1, 0.1, 0.35, 1.0])),
                _ => None,
            };
            match execute_offscreen(&renderer, &graph, &resources, &kinds) {
                Ok(frames) => {
                    for (name, frame) in frames {
                        println!(
                            "  target {name:<10} {}x{} checksum {:016x}",
                            frame.width,
                            frame.height,
                            frame.checksum()
                        );
                    }
                }
                Err(error) => println!("  execute    failed: {error}"),
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

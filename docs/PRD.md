# Onyx Engine — Master Feature Specification

**Pure Rust / AI-Native — Next-Gen Polyglot 2D/3D Game Engine with Sub-5ms Agentic MCP Control**

> Exhaustive, subsystem-by-subsystem master feature catalog — combining the full capability set of Unreal Engine, Unity, Godot 4, and Bevy, plus Onyx's native Sub-5ms MCP AI Control Plane and Polyglot WASM runtime.

---

## Table of Contents

1. [Core Architecture, ECS & Memory Management](#1-core-architecture-ecs--memory-management)
2. [Rendering & Graphics Pipeline](#2-rendering--graphics-pipeline)
3. [Lighting, Shadows & Atmosphere](#3-lighting-shadows--atmosphere)
4. [Materials, Shaders & Post-Processing](#4-materials-shaders--post-processing)
5. [World Building, Terrain & Open-World Streaming](#5-world-building-terrain--open-world-streaming)
6. [Physics & Simulation](#6-physics--simulation)
7. [Animation, Rigging & Characters](#7-animation-rigging--characters)
8. [Visual Effects (VFX) & Particles](#8-visual-effects-vfx--particles)
9. [Audio & Acoustics](#9-audio--acoustics)
10. [UI, HUD & Text System](#10-ui-hud--text-system)
11. [Gameplay, AI & Story Mode Systems](#11-gameplay-ai--story-mode-systems)
12. [Networking & Multiplayer](#12-networking--multiplayer)
13. [Polyglot "Any-Language" Scripting (WASM + Native)](#13-polyglot-any-language-scripting-wasm--native)
14. [Visual Editor, Cinematics & Developer Tools](#14-visual-editor-cinematics--developer-tools)
15. [Asset Pipeline, XR & Multi-Platform Export](#15-asset-pipeline-xr--multi-platform-export)
16. [Native Sub-5ms MCP AI Control Plane ("The AI Dictionary")](#16-native-sub-5ms-mcp-ai-control-plane-the-ai-dictionary)

---

## 1. Core Architecture, ECS & Memory Management

**Archetype Entity Component System (ECS):** Cache-friendly SoA (Structure of Arrays) storage, lock-free parallel system scheduler, automatic dependency graph resolution, change detection (`Changed<T>`, `Added<T>`, `Removed<T>`), system sets, fixed-timestep vs. variable-timestep schedules, observer/event hooks, sparse-set components for high-churn tags.

**Hybrid Scene & Prefab System:** Nested prefabs, prefab variants/overrides, scene instancing, live hot-reloading of `.onyx` / `.ron` / `.json` scenes, entity ID preservation for save games, additive scene loading, async background scene streaming.

**Reflection & Serialization Engine:** Compile-time Rust reflection macros (`#[derive(OnyxReflect)]`), runtime type registry, dynamic property getter/setter, automatic JSON/MessagePack/Binary serialization, schema generation for Inspector UI and MCP tools.

**Memory & Threading:** Custom arena/slab/bump allocators, frame-scratch allocators, job system / work-stealing thread pool, fiber/coroutine task scheduling, zero-copy asset streaming, deterministic memory budgets per platform (mobile/console/PC).

**Unified Command Bus & History:** Every engine mutation (human or AI) routed through reversible `EngineCommand` structs, unlimited Undo/Redo stack, transaction grouping, branching history snapshots, deterministic replay log.

## 2. Rendering & Graphics Pipeline

**Hardware Abstraction Layer (HAL):** wgpu cross-platform backend (Vulkan, DirectX 12, Metal, WebGPU) + raw `ash` (Vulkan) fast-path + pluggable console HAL traits (PlayStation AGC, Xbox GDK, Switch NVN).

**Render Pipelines:** Deferred PBR, Forward+ (clustered shading), Mobile Forward (tile-based), Stylized Anime-PBR pipeline, Raytracing / Hardware Path Tracing pipeline, Custom Programmable Render Graph (DAG-based parallel passes).

**GPU-Driven Geometry:** Virtualized geometry / Meshlets (Nanite-equivalent), Mesh Shaders & Task Shaders, GPU frustum culling, Hi-Z occlusion culling, indirect multi-draw (`multi_draw_indirect`), hardware tessellation, automatic LOD & HLOD generation, instanced static/skinned mesh batching.

**2D Rendering Suite:** Batch sprite renderer, 9-slice sprites, sprite sheets & dynamic texture atlases, 2D skeletal/mesh deformation, 2D normal-mapped lighting & shadow casters, tilemap renderer with auto-tiling & animated tiles, signed-distance-field (SDF) vector shapes, parallax scrolling layers, pixel-perfect camera snapping.

**Camera Systems:** Perspective & Orthographic cameras, physical camera parameters (sensor size, focal length, aperture/f-stop, ISO, shutter speed), split-screen multi-camera, render-to-texture / SubViewport portals, oblique frustum clipping, Cinemachine-style virtual cameras (follow, orbit, look-at, rail/dolly, spring-arm collision, camera shake, smooth blending).

## 3. Lighting, Shadows & Atmosphere

**Light Types:** Directional (Sun/Moon), Point, Spot (with IES profiles), Area/Rect lights, Tube lights, Emissive mesh lights, Sky/Environment lights, Light Cookies/Projectors.

**Global Illumination (GI):** Real-time Software & Hardware Raytraced GI (Lumen-equivalent), Screen-Space Global Illumination (SSGI), Lightmap Baking (CPU/GPU path-traced), Light Probes / Irradiance Volumes, Reflection Probes (Box/Sphere/Planar), Dynamic Sky Atmosphere & Aerial Perspective.

**Shadow Systems:** Cascaded Shadow Maps (CSM), Virtual Shadow Maps (VSM), Contact Shadows, Raytraced Soft Shadows, Percentage-Closer Soft Shadows (PCSS), Distance Field Shadows, Per-Object Shadow Caching, 2D Shadow Polygons.

**Atmosphere & Weather:** Physically based Rayleigh/Mie sky scattering, volumetric clouds (raymarched), exponential height fog, volumetric fog with light shafts (god rays), dynamic time-of-day cycle, procedural rain/snow/wind/lightning weather controller.

## 4. Materials, Shaders & Post-Processing

**Material Models:** Standard PBR Metallic-Roughness, Specular-Glossiness, Clear Coat (car paint/lacquer), Subsurface Scattering (skin/wax/foliage), Anisotropic (brushed metal/hair), Transmission/Refraction (glass/water), Cloth/Sheen, Hair (Marschner/Kajiya-Kay), Eye shader, Anime Cel/Toon PBR (ramp shading, rim light, matcap, inverted-hull & screen-space outlines, stylized hair specular).

**Shader Authoring:** Native WGSL / GLSL / HLSL / SPIR-V support, Visual Node-Based Shader Graph (math, noise, UV distortion, triplanar mapping, vertex displacement, parallax occlusion mapping, custom code nodes), live shader hot-reloading, shader variant stripping & pipeline state caching (PSO cache) to prevent stutter.

**Post-Processing Stack:** ACES / AgX / Neutral / Filmic Tonemapping, HDR10/Dolby Vision output, Bloom (physically based dual-kawase), Auto-Exposure (histogram eye adaptation), Screen-Space Reflections (SSR), Screen-Space Ambient Occlusion (GTAO/HBAO+/SSAO), Depth of Field (Bokeh), Motion Blur (per-object & camera), Chromatic Aberration, Vignette, Film Grain, Color Grading (3D LUTs, lift/gamma/gain, curves), Anti-Aliasing (TAA, FXAA, SMAA, MSAA), Upscaling hooks (FSR, DLSS, XeSS, MetalFX).

## 5. World Building, Terrain & Open-World Streaming

**Terrain System:** Heightmap & Voxel/Cave-capable terrain, GPU clipmap/quadtree LOD, multi-layer texture splatting (16+ layers with height-blending), non-destructive sculpt brushes (raise, lower, smooth, flatten, terrace, hydraulic/thermal erosion, noise stamp), spline-based roads/rivers/cliffs with automatic terrain conformation, terrain holes.

**Foliage & Environment Scattering:** GPU-instanced grass/trees/rocks, procedural biome placement rules (slope, altitude, moisture, exclusion masks), interactive grass bending (trails from players/explosions), wind vertex animation, billboard/impostor LODs.

**Water & Fluids:** Gerstner/FFT ocean waves, buoyancy physics, underwater caustics & absorption, foam shoreline detection, interactive ripples, river spline flowmaps, real-time 2D/3D fluid simulation.

**Open-World Partitioning:** Grid-based World Partition, Data Layers (day/night, quest states, destroyed city states), One-File-Per-Actor (OFPA) for conflict-free team/AI editing, Floating-Point Origin Rebasing (double-precision f64 world coordinates for massive solar-system scale), Seamless Level Streaming Volumes.

**Procedural Content Generation (PCG):** Node-based PCG graph for cities, dungeons, forests, wave-function collapse (WFC), L-systems, procedural buildings, and CSG (Constructive Solid Geometry) greyboxing brushes.

## 6. Physics & Simulation

**Rigid Body & Collision (Rapier / Jolt):** 2D & 3D rigid bodies (Dynamic, Kinematic, Static), Continuous Collision Detection (CCD), collision primitives (Box, Sphere, Capsule, Cylinder, Cone, Convex Hull, Triangle Mesh, Heightfield, Compound shapes), collision layers/masks, trigger volumes/sensors, sleeping/island optimization.

**Constraints & Joints:** Fixed, Revolute/Hinge, Prismatic/Slider, Spherical/Ball-Socket, Universal, 6-DOF Generic, Spring/Damper, Distance, Rope/Chain, Breakable joints, Motors & Limits.

**Specialized Controllers:** Kinematic Character Controller (slope limits, step-up height, ground snapping, moving platforms, wall-sliding, dashing, swimming, climbing), Arcade & Realistic Raycast Vehicle Physics (suspension, tire friction curves, differential, gears, drift), Ragdoll physics with active/powered ragdoll blending.

**Soft Bodies & Destruction:** Real-time Cloth simulation (capes, skirts, flags with wind & collider capsules), Spring Bones / Jiggle Physics (anime hair, accessories, tails), Destructible Meshes (Voronoi real-time fracturing, debris pooling), Spatial Queries (raycast, shapecast, sphere/box overlap, sweep tests).

## 7. Animation, Rigging & Characters

**Skeletal Animation:** Hierarchical bone rigs, glTF/VRM/FBX animation import, GPU skinning, dual-quaternion skinning (prevents candy-wrapper collapse), morph targets / blend shapes (facial expressions, lip-sync), root motion extraction, animation compression.

**Animation State Machines & Blending:** Hierarchical Animation State Machines (ASM), 1D & 2D Blend Spaces (walk/run/strafe), layered/masked additive blending (aiming upper body while running lower body), crossfade transitions, inertialization blending, Motion Matching (AAA fluid locomotion).

**Procedural & Inverse Kinematics (IK):** Two-bone IK, FABRIK, CCD IK, Full-Body IK (FBIK), automated foot placement on slopes/stairs, hand-to-weapon/ledge snapping, head/eye Look-At tracking, Control Rig (in-engine keyframing & constraints), Animation Retargeting across different skeletons.

**Animation Events & Curves:** Frame-accurate animation notifies (footstep audio, sword hitbox enable/disable, VFX spawn), custom float curves driven by animations.

## 8. Visual Effects (VFX) & Particles

**GPU Particle System (Niagara-equivalent):** Millions of GPU-simulated particles, compute-shader emitters, node-based particle behavior graph, ribbon/trail renderers (sword slashes, bullet tracers), mesh particles, decal particles, flipbook/sub-UV sprite sheets, soft particles (depth fade), lit & shadow-casting particles.

**Forces & Collisions:** Gravity, curl noise turbulence, vortex, point attractors, vector fields, depth-buffer screen-space collision, physics world collision, event generators (spawn sub-emitters on death/collision).

**Decals:** Deferred PBR decals (bullet holes, scorch marks, blood, graffiti, puddles), normal/roughness-only decals, angle-fade & distance-fade.

## 9. Audio & Acoustics

**Audio Engine:** Multi-threaded lock-free audio mixer, streaming (OGG/FLAC/MP3/Opus) & in-memory (WAV) playback, polyphony & voice-stealing priority, pitch/volume randomization containers.

**3D Spatial Audio:** HRTF binaural spatialization, distance attenuation curves, Doppler effect, occlusion/obstruction raycasting (muffled sound behind walls), dynamic Reverb Zones, real-time acoustic raytracing.

**DSP Effects & Adaptive Music:** Parametric EQ, Low/High/Band-pass filters, Compressor/Limiter, Chorus, Flanger, Delay, Convolution Reverb, Sidechain ducking, beat-synced interactive music stems (Exploration ↔ Combat seamless crossfades), procedural synth/MetaSounds node graph, microphone capture & voice chat bus.

## 10. UI, HUD & Text System

**Onyx UI Theme (binding):** every piece of Onyx-authored UI — editor, inspector, gizmos, HUD widget defaults, debug overlays and CLI output — is strict monochrome: white background (`#FFFFFF`), near-black foreground (`#0A0A0A`), grayscale-only hierarchy (no accent colors, no hues anywhere). Selection, state and emphasis are expressed through weight, shape and grayscale contrast, never color; lines are 1px hairlines at 1× DPI and typography stays a clean geometric sans on a fixed size grid. The only color in an Onyx session comes from the user's own content.

**Layout Engine:** Flexbox & CSS-Grid layout engine (Taffy/Yoga), responsive anchoring, aspect-ratio containers, safe-area insets (mobile notches), scroll views, virtualized lists (for 10,000+ inventory items), drag-and-drop framework, DPI scaling.

**Widget Library:** Button, Toggle, Checkbox, Radio, Slider, SpinBox, ProgressBar, TextInput (single/multi-line with IME), RichText (BBCode/Markdown inline icons & animations), Dropdown/ComboBox, TabContainer, TreeView, ColorPicker, Modal/Popup, Tooltip, ContextMenu, Minimap, Radial Wheel.

**Text & Localization:** MSDF (Multi-channel Signed Distance Field) crisp vector fonts, HarfBuzz text shaping (Arabic, Japanese, Chinese, Korean, emoji, RTL), fallback font chains, dynamic localization tables (CSV/JSON/PO), pluralization & live language switching.

**3D & Diegetic UI:** World-space UI canvases (health bars, floating damage numbers, in-world computer terminals), UI shaders & 9-slice theming, full Gamepad/Keyboard focus navigation & screen-reader accessibility.

## 11. Gameplay, AI & Story Mode Systems

**Navigation & Pathfinding:** Automatic Recast/Detour NavMesh generation (2D & 3D), dynamic runtime NavMesh obstacles & carving, NavMesh Links (jumps, ladders, teleports), A* / Dijkstra / Flow-Field pathfinding (for 10,000+ RTS/horde units), RVO/ORCA local crowd avoidance.

**Game AI Frameworks:** Visual Behavior Trees, Finite State Machines (FSM), Utility AI scoring, GOAP (Goal-Oriented Action Planning), Environment Query System (EQS—find cover, line-of-sight, flank positions), AI Perception (vision cones, hearing radius, damage/touch memory), Smart Objects (NPCs sitting on benches, forging weapons).

**Input System:** Action-based Input Mapping (rebindable at runtime), Keyboard, Mouse, Gamepad (Xbox, DualSense haptics/adaptive triggers, Switch Pro), Multi-Touch & Virtual Joysticks, Gyro/Accelerometer, Input Buffering & Combo/Chord detection, Local Co-op multi-device routing.

**Story Mode & RPG Suite (First-Class Built-In):** Visual Branching Dialogue Graph (conditions, variables, voice-over clips, portrait emotions), Quest & Objective State Machine, Inventory & Equipment Grid, Ability/Skill & Cooldown System (GAS-equivalent: buffs, debuffs, elemental reactions, damage pipelines), Save/Load Slot Manager (JSON/Binary + screenshot thumbnails + cloud sync).

## 12. Networking & Multiplayer

**Transport Layer:** UDP (QUIC / ENet / Renet), WebSockets, WebRTC (browser P2P + NAT traversal/STUN/TURN), Bluetooth / Wi-Fi Direct LAN discovery.

**Replication & Sync:** Authoritative Client-Server, Listen Server, P2P, Entity/Component state replication, delta compression, bit-packing, interest management / relevancy culling (only sync nearby players), RPCs (Reliable/Unreliable).

**Netcode Models:** Client-Side Prediction & Server Reconciliation (FPS/Action RPG), Lag Compensation (server rewind hitboxes), Deterministic Rollback Netcode (GGPO-style for fighting/arena games), Snapshot Interpolation.

**Online Services:** Lobby & Matchmaking hooks, Dedicated Headless Server build target (no GPU required), Anti-Cheat validation hooks, Voice/Text Chat channels.

## 13. Polyglot "Any-Language" Scripting (WASM + Native)

**WebAssembly Component Model (wasmtime + WIT):** Write gameplay in Rust, Python, TypeScript/JavaScript, C, C++, Go, C#, Zig, Lua, or Swift, compiled to sandboxed `.wasm` components with near-native speed and sub-100ms hot-reloading.

**Native Rust Plugins:** Zero-overhead direct ECS system registration for engine-level extensions.

**Visual Scripting Graph:** Node-based Blueprint/State-Graph editor that compiles directly to deterministic bytecode/WASM—usable by non-coders and readable/editable by AI agents.

## 14. Visual Editor, Cinematics & Developer Tools

**Viewport & Gizmos:** Translate/Rotate/Scale gizmos (Local/World space), Grid/Vertex/Surface snapping, Wireframe/Unlit/Albedo/Normals/Overdraw/Collision debug view modes, Free-fly & Orbit camera, Bookmarked camera views, Multi-viewport quad view.

**Inspector & Outliner:** Fuzzy search, tag/layer filtering, drag-and-drop reparenting, multi-object editing, live runtime inspection & editing during Play-In-Editor (PIE), Live Pause & Frame-Step.

**Cinematics & Sequencer:** Non-linear timeline editor for cutscenes, camera tracks, skeletal animation tracks, audio/dialogue tracks, material/light keyframing, event triggers, offline 4K/60fps video/image-sequence movie renderer.

**Profiling & Debugging:** Real-time FPS/Frame-time graph, CPU Flamegraph profiler (Tracy integration), GPU Pass timing profiler, Memory allocation tracker, Network packet/bandwidth monitor, Physics collider & NavMesh debug drawer, In-game Developer Console (~), Automated Headless Playtest & Regression Runner.

## 15. Asset Pipeline, XR & Multi-Platform Export

**Asset Importers:** glTF 2.0 / GLB, VRM (anime avatars), FBX, OBJ, USD/USDZ, PNG/JPG/WebP/EXR/HDR/KTX2/DDS, WAV/OGG/FLAC/MP3, TTF/OTF, Tiled (.tmx) / LDtk (.ldtk) 2D maps, Godot .tscn importer.

**Asset Optimization:** Automatic mipmap generation, GPU texture compression (BC7, ASTC, ETC2, Basis Universal), mesh optimization (meshoptimizer vertex cache/overdraw), asset bundling & encryption (.pak archives), incremental content patching / DLC bundles.

**XR (VR / AR):** OpenXR native runtime, stereo multiview rendering, foveated rendering, hand tracking, controller haptics, passthrough AR.

**1-Click Platform Targets:** Windows (.exe/.msi), Linux (AppImage/Flatpak), macOS (Universal .app), Android (.apk/.aab + AdMob/Billing SDK hooks), iOS (.ipa Xcode project), Web (WebAssembly + WebGPU), Console HAL (PS5, Xbox Series X|S, Switch).

## 16. Native Sub-5ms MCP AI Control Plane ("The AI Dictionary")

**Self-Describing Tool Registry (`mcp-2026-07-28`):** Every single feature in Sections 1–15 automatically exposed as typed JSON-Schema MCP tools (`tools/list`), live readable world state (`resources/list`), and multi-step workflow macros (`prompts/list`).

**Batched Atomic RPC (`batch_execute`):** Execute 100+ terrain, entity, lighting, shader, and script mutations in one single frame (<5ms) — zero browser/GUI clicking overhead.

**Headless Simulation & Spatial Queries:** AI agents run `step_frames(60)`, query exact physics collisions, raycasts, and entity bounds mathematically, or pull compressed viewport/depth/segmentation renders for visual QA.

**Human + AI Co-Editing:** Live Activity Log showing every AI tool call, Ghost-Preview approval mode, Auto-Pilot mode, and shared Ctrl+Z undo/redo between human and AI.


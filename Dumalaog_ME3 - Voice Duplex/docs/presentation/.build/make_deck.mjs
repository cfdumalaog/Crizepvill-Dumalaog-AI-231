import fs from "node:fs/promises";
import path from "node:path";
import { pathToFileURL } from "node:url";
import { Presentation, PresentationFile } from "@oai/artifact-tool";

const skillDir = "C:/Users/danda/.codex/plugins/cache/openai-primary-runtime/presentations/26.1022.11436/skills/presentations";
const workspaceDir = "C:/Users/danda/Desktop/MEng AI Notebooks/AI 222 231/AI 231/Dumalaog_ME3 - Voice Duplex";
const tmpDir = path.join(workspaceDir, "docs/presentation/.build");
const finalPath = path.join(workspaceDir, "docs/presentation/ME3_Voice_Duplex_Presentation_v6.pptx");

const C = {
  ink: "#F7FAFC",
  muted: "#A8B8CA",
  navy: "#081827",
  navy2: "#102A43",
  panel: "#14324A",
  teal: "#5EEAD4",
  blue: "#60A5FA",
  green: "#A7F3D0",
  amber: "#FCD34D",
  red: "#FDA4AF",
  white: "#FFFFFF",
  line: "#31536C",
};
const FONT = "Arial";

async function writeBlob(filePath, blob) {
  await fs.writeFile(filePath, new Uint8Array(await blob.arrayBuffer()));
}

function rect(slide, position, fill, lineFill = "none", geometry = "roundRect", name) {
  return slide.shapes.add({
    geometry,
    name,
    position,
    fill,
    line: { style: "solid", fill: lineFill, width: lineFill === "none" ? 0 : 1 },
    borderRadius: geometry === "roundRect" ? "rounded-xl" : undefined,
  });
}

function text(slide, value, position, style = {}, name) {
  const shape = slide.shapes.add({
    geometry: "textbox",
    name,
    position,
    fill: "none",
    line: { style: "solid", fill: "none", width: 0 },
  });
  shape.text = value;
  shape.text.style = {
    typeface: FONT,
    fontSize: 18,
    color: C.ink,
    ...style,
  };
  return shape;
}

function line(slide, left, top, width, color = C.line, arrow = true, name) {
  return slide.shapes.add({
    geometry: "connector",
    kind: "straight",
    name,
    position: { left, top, width, height: 0 },
    line: { style: "solid", fill: color, width: 2 },
    tail: arrow ? { type: "arrow", width: "med", length: "med" } : { type: "none" },
  });
}

function footer(slide, page) {
  text(slide, `ME3 VOICE DUPLEX  •  QWEN TOOL BRAIN  •  ${page}/7`,
    { left: 64, top: 682, width: 420, height: 18 },
    { fontSize: 11, color: C.muted, bold: true });
}

function notes(slide, body) {
  slide.speakerNotes.text = `${body}\n\n[Sources]\nAI 231/Dumalaog_ME3 - Voice Duplex/README.md\nAI 231/Dumalaog_ME3 - Voice Duplex/docs/ME3_MANUAL_ELI5_AND_DEEP.md`;
  slide.speakerNotes.setVisible(true);
}

function title(slide, kicker, heading, subheading) {
  text(slide, kicker.toUpperCase(), { left: 64, top: 42, width: 500, height: 24 },
    { fontSize: 13, color: C.teal, bold: true });
  text(slide, heading, { left: 64, top: 76, width: 1120, height: 60 },
    { fontSize: 36, color: C.ink, bold: true });
  if (subheading) {
    text(slide, subheading, { left: 64, top: 140, width: 1040, height: 36 },
      { fontSize: 19, color: C.muted });
  }
}

function bulletList(slide, items, position, color = C.ink, size = 19, gap = 38) {
  items.forEach((item, index) => {
    text(slide, `•  ${item}`, {
      left: position.left,
      top: position.top + index * gap,
      width: position.width,
      height: gap + 6,
    }, { fontSize: size, color });
  });
}

async function build() {
  await fs.mkdir(tmpDir, { recursive: true });
  await fs.mkdir(path.dirname(finalPath), { recursive: true });

  const presentation = Presentation.create({ slideSize: { width: 1280, height: 720 } });

  // 1. Cover
  {
    const slide = presentation.slides.add();
    slide.background.fill = C.navy;
    rect(slide, { left: 820, top: 20, width: 420, height: 420 }, C.navy2, "none", "ellipse", "cover-orbit");
    rect(slide, { left: 890, top: 62, width: 300, height: 300 }, "none", C.teal, "ellipse", "cover-ring");
    rect(slide, { left: 972, top: 144, width: 136, height: 136 }, C.teal, "none", "ellipse", "cover-core");
    text(slide, "ME3 VOICE DUPLEX", { left: 72, top: 74, width: 600, height: 28 },
      { fontSize: 16, color: C.teal, bold: true });
    text(slide, "From voice request\nto verified action", { left: 72, top: 145, width: 660, height: 154 },
      { fontSize: 54, color: C.ink, bold: true });
    text(slide, "A Raspberry Pi full-duplex agent with a Qwen tool-using brain", { left: 76, top: 330, width: 650, height: 42 },
      { fontSize: 24, color: C.muted });
    rect(slide, { left: 76, top: 430, width: 570, height: 80 }, C.panel, "none", "roundRect", "cover-callout");
    text(slide, "No task-specific training required", { left: 102, top: 448, width: 500, height: 26 },
      { fontSize: 22, color: C.green, bold: true });
    text(slide, "Inference + typed tools + verification", { left: 102, top: 478, width: 500, height: 22 },
      { fontSize: 16, color: C.muted });
    text(slide, "Crizepvill F. Dumalaog  •  Mapua University  •  2026", { left: 76, top: 630, width: 700, height: 24 },
      { fontSize: 14, color: C.muted });
    notes(slide, "Open with the central idea: ME3 is an action-capable voice interface. The model does not need to be trained for each task; it needs a disciplined tool surface and evidence that the requested work happened.");
  }

  // 2. Experience sequence
  {
    const slide = presentation.slides.add();
    slide.background.fill = C.navy;
    title(slide, "The experience", "A spoken request becomes a checked result", "The user stays in conversation while the model reasons and acts through bounded tools.");
    const xs = [80, 350, 620, 890];
    const labels = [
      ["01", "Speak", "“Build me a web app”", C.teal],
      ["02", "Plan", "Qwen selects tool calls", C.blue],
      ["03", "Act", "Files, web, commands, jobs", C.amber],
      ["04", "Prove", "Read-back + verification", C.green],
    ];
    // Connectors first so they remain behind the nodes.
    line(slide, 236, 330, 100, C.teal, true, "sequence-1");
    line(slide, 506, 330, 100, C.blue, true, "sequence-2");
    line(slide, 776, 330, 100, C.amber, true, "sequence-3");
    labels.forEach(([num, head, body, color], i) => {
      rect(slide, { left: xs[i], top: 250, width: 190, height: 160 }, C.panel, color, "roundRect", `sequence-node-${i}`);
      text(slide, num, { left: xs[i] + 18, top: 270, width: 50, height: 26 }, { fontSize: 14, color, bold: true });
      text(slide, head, { left: xs[i] + 18, top: 304, width: 150, height: 32 }, { fontSize: 25, color: C.ink, bold: true });
      text(slide, body, { left: xs[i] + 18, top: 346, width: 152, height: 42 }, { fontSize: 16, color: C.muted });
    });
    rect(slide, { left: 180, top: 488, width: 920, height: 86 }, C.navy2, "none", "roundRect", "sequence-caption");
    text(slide, "Full duplex means listening continues during playback.", { left: 212, top: 508, width: 870, height: 28 },
      { fontSize: 23, color: C.ink, bold: true });
    text(slide, "Speech during TTS cancels playback and starts a new turn.", { left: 212, top: 542, width: 870, height: 22 },
      { fontSize: 17, color: C.muted });
    footer(slide, 2);
    notes(slide, "Walk left to right. The important change from a voice chatbot is the middle: Qwen is not only writing an answer; it is choosing typed operations, receiving their results, and continuing until the task is complete or needs clarification.");
  }

  // 3. Architecture
  {
    const slide = presentation.slides.add();
    slide.background.fill = C.navy;
    title(slide, "Architecture", "Local speech, remote reasoning, owned actions", "Speech stays local; Qwen reasons remotely; typed tools own side effects.");
    // Unattached connectors first.
    line(slide, 245, 334, 75, C.teal, true, "arch-local-to-qwen");
    line(slide, 565, 334, 75, C.blue, true, "arch-qwen-to-tools");
    line(slide, 885, 334, 75, C.green, true, "arch-tools-to-response");
    const nodes = [
      { x: 70, color: C.teal, head: "LOCAL AUDIO", body: "Mic → VAD → ASR\nTTS → speaker", foot: "Pi / Windows" },
      { x: 390, color: C.blue, head: "QWEN BRAIN", body: "Chat completions\ntool_choice = auto", foot: "vLLM API" },
      { x: 710, color: C.amber, head: "TOOL REGISTRY", body: "Files • web • commands\nbackground tasks", foot: "ME3 policy" },
      { x: 1030, color: C.green, head: "EVIDENCE", body: "Tool result →\nspoken response", foot: "auditable" },
    ];
    nodes.forEach((node, i) => {
      rect(slide, { left: node.x, top: 240, width: 180, height: 190 }, C.panel, node.color, "roundRect", `arch-node-${i}`);
      rect(slide, { left: node.x + 20, top: 264, width: 48, height: 6 }, node.color, "none", "rect", `arch-accent-${i}`);
      text(slide, node.head, { left: node.x + 20, top: 288, width: 144, height: 24 }, { fontSize: 14, color: node.color, bold: true });
      text(slide, node.body, { left: node.x + 20, top: 330, width: 144, height: 70 }, { fontSize: 17, color: C.ink, bold: true });
      text(slide, node.foot, { left: node.x + 20, top: 404, width: 144, height: 18 }, { fontSize: 13, color: C.muted });
    });
    rect(slide, { left: 170, top: 500, width: 940, height: 72 }, C.navy2, "none", "roundRect", "arch-principle");
    text(slide, "One bounded registry is shared by direct Qwen calls and MCP wrappers.", { left: 205, top: 520, width: 870, height: 26 },
      { fontSize: 21, color: C.ink, bold: true });
    footer(slide, 3);
    notes(slide, "Use this slide to separate responsibilities. VAD, ASR, and TTS are local. Qwen is remote and performs inference. The registry owns what may happen. The final answer is grounded in structured tool results, not just model confidence.");
  }

  // 4. Web app flow
  {
    const slide = presentation.slides.add();
    slide.background.fill = C.navy;
    title(slide, "Tool call in action", "“Build a web app” is an executable request", "The same loop can start from text; the voice path connects to the Pi audio pipeline.");
    rect(slide, { left: 70, top: 220, width: 290, height: 300 }, C.panel, C.teal, "roundRect", "webapp-request");
    text(slide, "VOICE REQUEST", { left: 98, top: 248, width: 220, height: 24 }, { fontSize: 14, color: C.teal, bold: true });
    text(slide, "“Build a\nweb app.”", { left: 98, top: 298, width: 230, height: 92 }, { fontSize: 32, color: C.ink, bold: true });
    text(slide, "Qwen turns intent\ninto tool calls.", { left: 98, top: 430, width: 220, height: 44 }, { fontSize: 17, color: C.muted });
    line(slide, 360, 366, 65, C.teal, true, "webapp-to-calls");
    rect(slide, { left: 440, top: 220, width: 350, height: 300 }, C.panel, C.blue, "roundRect", "webapp-calls");
    text(slide, "TOOL LOOP", { left: 468, top: 248, width: 220, height: 24 }, { fontSize: 14, color: C.blue, bold: true });
    text(slide, "list_files\nwrite_file\nread_file\nrun_command", { left: 468, top: 300, width: 260, height: 132 }, { fontSize: 24, color: C.ink, bold: true });
    text(slide, "Each call returns JSON evidence.", { left: 468, top: 466, width: 290, height: 24 }, { fontSize: 16, color: C.muted });
    line(slide, 790, 366, 65, C.blue, true, "calls-to-files");
    rect(slide, { left: 870, top: 220, width: 340, height: 300 }, C.panel, C.green, "roundRect", "webapp-result");
    text(slide, "PROJECT RESULT", { left: 898, top: 248, width: 220, height: 24 }, { fontSize: 14, color: C.green, bold: true });
    text(slide, "agent_workspace/\n  voice_webapp/\n    index.html\n    styles.css\n    app.js", { left: 898, top: 296, width: 270, height: 142 }, { fontSize: 21, color: C.ink, bold: true });
    text(slide, "Verified by a later list + read pass.", { left: 898, top: 468, width: 270, height: 24 }, { fontSize: 16, color: C.muted });
    footer(slide, 4);
    notes(slide, "This is the concrete capability the project needed. A spoken request can create real project files. The model then reads them back and reports what exists. The current demonstration is a local, dependency-free static web app; it does not silently publish to the internet.");
  }

  // 5. Capability boundary
  {
    const slide = presentation.slides.add();
    slide.background.fill = C.navy;
    title(slide, "Capability boundary", "Onit-like usefulness through typed, reviewable tools", "Broad task coverage, with a safer default surface for an always-listening device.");
    const rows = [
      ["Conversation", "Qwen chat + multi-step tool loop", C.teal],
      ["Project work", "list / read / write / search files", C.blue],
      ["Web", "public URL fetch + web search", C.green],
      ["Execution", "allowlisted commands + background jobs", C.amber],
      ["Operations", "system info + timers + MCP bridge", C.red],
    ];
    rows.forEach(([left, right, color], i) => {
      const y = 214 + i * 64;
      rect(slide, { left: 80, top: y, width: 190, height: 44 }, C.panel, color, "roundRect", `cap-left-${i}`);
      text(slide, left, { left: 102, top: y + 10, width: 145, height: 23 }, { fontSize: 18, color, bold: true });
      text(slide, right, { left: 310, top: y + 10, width: 560, height: 23 }, { fontSize: 18, color: C.ink });
    });
    rect(slide, { left: 910, top: 214, width: 300, height: 300 }, C.navy2, C.amber, "roundRect", "boundary-callout");
    text(slide, "PARITY ≠\nUNRESTRICTED\nSHELL", { left: 934, top: 252, width: 252, height: 110 }, { fontSize: 27, color: C.amber, bold: true });
    text(slide, "Paths stay inside ME3.\nSecrets and model caches are\nblocked. Destructive patterns\nare rejected.", { left: 944, top: 394, width: 232, height: 86 }, { fontSize: 16, color: C.muted });
    text(slide, "Capability is added as a typed tool with an observable result.", { left: 82, top: 574, width: 820, height: 26 }, { fontSize: 20, color: C.ink, bold: true });
    footer(slide, 5);
    notes(slide, "Explain the interpretation of parity. The objective is that useful tasks Onit can perform are represented by explicit tools: files, web, commands, jobs, system information, and the existing MCP services. That does not mean granting an always-listening Pi arbitrary shell access or hidden credentials.");
  }

  // 6. Verification
  {
    const slide = presentation.slides.add();
    slide.background.fill = C.navy;
    title(slide, "Evidence", "The implementation is verified in layers", "Each layer answers a different question; a local pass is not silently promoted to Pi or benchmark evidence.");
    rect(slide, { left: 76, top: 214, width: 520, height: 360 }, C.panel, C.green, "roundRect", "verified-now");
    text(slide, "VERIFIED NOW", { left: 108, top: 244, width: 300, height: 24 }, { fontSize: 14, color: C.green, bold: true });
    bulletList(slide, [
      "Pi deployed (transfer hash verified against the release manifest)",
      "Pi --check passes; Qwen key provisioned (mode 600)",
      "Pi text E2E: Qwen built + served a web app, URL LAN-verified",
      "Dev box: Qwen web-app file creation + read-back; 4 tooling tests",
      "All 4 MCP services + PortAudio smoke + v1.5 contract checks",
    ], { left: 108, top: 292, width: 440 }, C.ink, 18, 47);
    rect(slide, { left: 650, top: 214, width: 554, height: 360 }, C.panel, C.amber, "roundRect", "next-gates");
    text(slide, "REMAINING GATES", { left: 682, top: 244, width: 300, height: 24 }, { fontSize: 14, color: C.amber, bold: true });
    bulletList(slide, [
      "Pi live audio run on installed hardware (0 audio devices now)",
      "Clean, unaudited FULL-DUPLEX-BENCH v1.5 capture",
      "HTTPS endpoint when an administrator provides one",
    ], { left: 682, top: 292, width: 474 }, C.ink, 18, 55);
    text(slide, "No training run is in the critical path.", { left: 682, top: 510, width: 480, height: 26 }, { fontSize: 20, color: C.green, bold: true });
    footer(slide, 6);
    notes(slide, "This slide protects the evidence boundary. The Pi now runs the verified release with a provisioned, mode-600 Qwen key, and a text-mode Qwen run built and served a web app on the Pi (LAN-verified). The remaining gates are physical Pi audio and a clean captured v1.5 overlap evaluation; local Windows evidence is not promoted into either claim.");
  }

  // 7. Close
  {
    const slide = presentation.slides.add();
    slide.background.fill = C.navy;
    rect(slide, { left: 64, top: 64, width: 10, height: 560 }, C.teal, "none", "rect", "close-rule");
    text(slide, "THE TAKEAWAY", { left: 112, top: 98, width: 300, height: 24 }, { fontSize: 14, color: C.teal, bold: true });
    text(slide, "A voice agent becomes\nuseful when it can act\nand show its work.", { left: 112, top: 154, width: 700, height: 160 }, { fontSize: 46, color: C.ink, bold: true });
    text(slide, "ME3 combines full-duplex audio, Qwen inference, typed tools, and verification into a deployable Raspberry Pi workflow.", { left: 116, top: 358, width: 700, height: 66 }, { fontSize: 22, color: C.muted });
    rect(slide, { left: 900, top: 160, width: 250, height: 250 }, C.panel, C.blue, "roundRect", "close-card");
    text(slide, "VOICE\n→\nTOOLS\n→\nEVIDENCE", { left: 934, top: 195, width: 190, height: 170 }, { fontSize: 25, color: C.blue, bold: true });
    text(slide, "Next: Pi audio + clean benchmark", { left: 116, top: 540, width: 600, height: 24 }, { fontSize: 19, color: C.green, bold: true });
    text(slide, "Documentation: ELI5/deep manual • ACM two-column paper • source + receipts", { left: 116, top: 590, width: 800, height: 24 }, { fontSize: 16, color: C.muted });
    footer(slide, 7);
    notes(slide, "Close by returning to the user-visible promise: speak a request, let the model call bounded tools, and receive an answer backed by what the tools actually returned. The remaining work is hardware audio and benchmark evidence, not model training.");
  }

  for (const [index, slide] of presentation.slides.items.entries()) {
    const stem = `draft-slide-${String(index + 1).padStart(2, "0")}`;
    await writeBlob(path.join(tmpDir, `${stem}.png`), await presentation.export({ slide, format: "png", scale: 1 }));
  }
  const draftPath = path.join(tmpDir, "candidate.pptx");
  await (await PresentationFile.exportPptx(presentation)).save(draftPath);

  const { finalizePresentation } = await import(pathToFileURL(path.join(skillDir, "container_tools/artifact_tool_utils.mjs")).href);
  const stagingDir = path.join(workspaceDir, "docs/.presentation-finalizer");
  await fs.mkdir(stagingDir, { recursive: true });
  const result = await finalizePresentation({
    explicitTotalSlideCount: 7,
    requiredNativeTableOwnerSlides: [],
    requiredNativeChartOwnerSlides: [],
    workspaceDir,
    candidatePath: draftPath,
    finalPath,
    pythonExecutable: "C:/Users/danda/.cache/codex-runtimes/codex-primary-runtime/dependencies/python/python.exe",
    integrityValidatorPath: path.join(skillDir, "container_tools/inspect_presentation_package_integrity.py"),
    layoutValidatorPath: path.join(skillDir, "container_tools/inspect_presentation_layout_geometry.py"),
    layoutArgs: ["--expected-slide-size-emu", "12192000,6858000", "--validate-bullet-geometry", "--validate-heading-fit"],
    fontPolicy: { basis: "design", families: [FONT] },
    verifyArtifactToolImport: true,
    receiptPath: path.join(stagingDir, "ME3_Voice_Duplex_Presentation_v6.validation.json"),
  });
  await fs.writeFile(path.join(stagingDir, "finalizer-result.json"), JSON.stringify(result, null, 2));
}

build().catch((error) => {
  console.error(error);
  process.exitCode = 1;
});

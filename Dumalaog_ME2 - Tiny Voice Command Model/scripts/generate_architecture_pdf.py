"""
Generates publication-quality architecture diagram PDF and PNG for TinyDSCNN-48.
AI 231 Machine Exercise 2 - Tiny Voice Command Model.
"""
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.backends.backend_pdf import PdfPages

OUT_DIR = Path(r"C:\Users\danda\Desktop\MEng AI Notebooks\AI 222 231\AI 231\Dumalaog_ME2 - Tiny Voice Command Model\docs")
OUT_DIR.mkdir(parents=True, exist_ok=True)
PDF_PATH = OUT_DIR / "TinyDSCNN48_Architecture.pdf"
PNG_PATH = OUT_DIR / "tinydscnn48_architecture.png"

def draw_page_1(fig):
    ax = fig.add_subplot(111)
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 140)
    ax.axis("off")

    # Header Banner
    header_box = patches.FancyBboxPatch((2, 126.5), 96, 11.5, boxstyle="round,pad=0.6,rounding_size=1.2",
                                        edgecolor="#1e293b", facecolor="#0f172a", lw=1.2)
    ax.add_patch(header_box)
    ax.text(50, 134.0, "TinyDSCNN-48 : Model Architecture & Data Flow",
            ha="center", va="center", color="#ffffff", fontsize=14, fontweight="bold", family="sans-serif")
    ax.text(50, 129.2, "AI 231 (Deep Learning) Machine Exercise 2 — From-Scratch Tiny Voice Command Model",
            ha="center", va="center", color="#94a3b8", fontsize=9.0, family="sans-serif")

    # Badges / Spec row (4 neat cards)
    badges = [
        ("14,282 PARAMS", "26-class baseline (~14.5k for 32-cl)", "#2563eb"),
        ("44.6 KB ONNX", "Static INT8 quantized footprint", "#059669"),
        ("1.5 s WINDOW", "16 kHz Mono (24,000 samples)", "#7c3aed"),
        ("0.50 ms CPU", "Single-thread x86 (< 10 ms RPi)", "#d97706")
    ]
    for i, (top_txt, sub_txt, col) in enumerate(badges):
        x = 2 + i * 24.3
        badge = patches.FancyBboxPatch((x, 119.5), 23.2, 5.5, boxstyle="round,pad=0.3,rounding_size=0.8",
                                      edgecolor=col, facecolor="#f8fafc", lw=1.1)
        ax.add_patch(badge)
        ax.text(x + 11.6, 123.4, top_txt, ha="center", va="center", color=col, fontsize=7.2, fontweight="bold")
        ax.text(x + 11.6, 121.0, sub_txt, ha="center", va="center", color="#475569", fontsize=5.8)

    # -------------------------------------------------------------
    # SECTION 1: END-TO-END AUDIO PIPELINE (TOP HORIZONTAL)
    # -------------------------------------------------------------
    ax.text(2, 116.5, "1. End-to-End Processing Pipeline", fontsize=10.5, fontweight="bold", color="#0f172a")

    pipeline_steps = [
        ("Raw Audio\nMicrophone", "16 kHz Mono\n1.5 s (24,000 pts)", "#e0f2fe", "#0284c7"),
        ("Log-Mel Frontend\nSTFT + Mel-Filter", "40 mel bands, 25ms win\nShape: (1, 40, 151)", "#e0e7ff", "#4338ca"),
        ("TinyDSCNN-48\nFeature Extractor", "Stem + 4 DS-Blocks\nShape: (48, 20, 38)", "#ede9fe", "#6d28d9"),
        ("Adaptive Pooling\n& Dropout", "Global AvgPool + 15%\nShape: (48,)", "#fef3c7", "#d97706"),
        ("Linear Classifier\n& Softmax", "FC Layer (48 -> Classes)\nLogits / Intent Class", "#dcfce7", "#15803d")
    ]

    for i, (title, desc, bg, border) in enumerate(pipeline_steps):
        px = 2 + i * 19.5
        box = patches.FancyBboxPatch((px, 105), 17.5, 9.5, boxstyle="round,pad=0.4,rounding_size=0.8",
                                     edgecolor=border, facecolor=bg, lw=1.2)
        ax.add_patch(box)
        ax.text(px + 8.75, 111.0, title, ha="center", va="center", color="#0f172a", fontsize=7.2, fontweight="bold")
        ax.text(px + 8.75, 107.0, desc, ha="center", va="center", color="#475569", fontsize=6.2)

        if i < len(pipeline_steps) - 1:
            ax.annotate("", xy=(px + 17.5 + 1.8, 109.8), xytext=(px + 17.5 + 0.1, 109.8),
                        arrowprops=dict(arrowstyle="-|>", lw=1.4, color="#64748b", mutation_scale=10))

    # -------------------------------------------------------------
    # SECTION 2: NEURAL NETWORK STAGES (LEFT SIDE)
    # -------------------------------------------------------------
    ax.text(2, 101.5, "2. TinyDSCNN-48 Layer Flow", fontsize=10.5, fontweight="bold", color="#0f172a")

    stages = [
        ("Input Spectrogram", "1 channel × 40 mel bands × 151 time frames", "(1, 40, 151)", "#f8fafc", "#64748b", "Input Data"),
        ("Stem Convolution", "Conv2d(1->48, k=5×5, stride=2, padding=2) + BN + ReLU", "(48, 20, 76)", "#eff6ff", "#2563eb", "1,296 params"),
        ("DS-Block 1", "DW 3×3 (s=1, g=48) + BN + ReLU → PW 1×1 (48ch) + BN + ReLU", "(48, 20, 76)", "#f5f3ff", "#7c3aed", "2,928 params"),
        ("DS-Block 2 [Time Downsample]", "DW 3×3 (s=(1,2), g=48) + BN + ReLU → PW 1×1 + BN + ReLU", "(48, 20, 38)", "#fae8ff", "#c026d3", "2,928 params"),
        ("DS-Block 3", "DW 3×3 (s=1, g=48) + BN + ReLU → PW 1×1 (48ch) + BN + ReLU", "(48, 20, 38)", "#f5f3ff", "#7c3aed", "2,928 params"),
        ("DS-Block 4", "DW 3×3 (s=1, g=48) + BN + ReLU → PW 1×1 (48ch) + BN + ReLU", "(48, 20, 38)", "#f5f3ff", "#7c3aed", "2,928 params"),
        ("Global Average Pooling", "nn.AdaptiveAvgPool2d(1) collapses time & frequency → Flatten", "(48,)", "#fefce8", "#ca8a04", "0 params"),
        ("Dropout Regularization", "nn.Dropout(p=0.15) — Active during training, identity in eval", "(48,)", "#fff7ed", "#ea580c", "0 params"),
        ("Linear Classifier Head", "nn.Linear(in_features=48, out_features=classes, bias=True)", "(classes,)", "#f0fdf4", "#16a34a", "1,274 / 1,568 params")
    ]

    sy = 95.0
    sh = 6.2
    gap = 1.6
    for idx, (s_name, s_op, s_shape, s_bg, s_bc, s_param) in enumerate(stages):
        cur_y = sy - idx * (sh + gap)
        box = patches.FancyBboxPatch((2, cur_y), 58, sh, boxstyle="round,pad=0.3,rounding_size=0.8",
                                     edgecolor=s_bc, facecolor=s_bg, lw=1.1)
        ax.add_patch(box)

        # Left Column: Layer name (top) and Operation details (bottom)
        ax.text(3.5, cur_y + 4.3, s_name, ha="left", va="center", color=s_bc, fontsize=7.2, fontweight="bold")
        ax.text(3.5, cur_y + 1.8, s_op, ha="left", va="center", color="#475569", fontsize=5.7)

        # Right Column: Output Shape (top) and Params (bottom), neatly right-aligned
        ax.text(58.5, cur_y + 4.3, s_shape, ha="right", va="center", color="#0f172a", fontsize=6.8, fontweight="bold", family="monospace")
        ax.text(58.5, cur_y + 1.8, s_param, ha="right", va="center", color="#64748b", fontsize=5.8)

        if idx < len(stages) - 1:
            ax.annotate("", xy=(31, cur_y - gap + 0.1), xytext=(31, cur_y),
                        arrowprops=dict(arrowstyle="-|>", lw=1.2, color="#94a3b8", mutation_scale=8))

    # -------------------------------------------------------------
    # SECTION 3: DEPTHWISE SEPARABLE CALLOUT (RIGHT SIDE)
    # -------------------------------------------------------------
    ax.text(63, 101.5, "3. Depthwise Separable Block Detail", fontsize=10.5, fontweight="bold", color="#0f172a")

    callout_box = patches.FancyBboxPatch((63, 25), 35, 76, boxstyle="round,pad=0.6,rounding_size=1.0",
                                         edgecolor="#94a3b8", facecolor="#ffffff", lw=1.1)
    ax.add_patch(callout_box)

    ax.text(80.5, 97.5, "Inside Each DS-Block", ha="center", va="center", color="#0f172a", fontsize=8.8, fontweight="bold")
    ax.text(80.5, 94.8, "Factorizes standard conv into two operations:", ha="center", va="center", color="#64748b", fontsize=6.4)

    # Sub-block 1: Depthwise
    dw_box = patches.FancyBboxPatch((65, 75), 31, 17, boxstyle="round,pad=0.3,rounding_size=0.8",
                                    edgecolor="#0284c7", facecolor="#f0f9ff", lw=1.1)
    ax.add_patch(dw_box)
    ax.text(80.5, 89.0, "Stage 1: Depthwise Conv (Spatial)", ha="center", va="center", color="#0369a1", fontsize=7.2, fontweight="bold")
    dw_desc = "• Kernel: 3×3, groups = 48\n• Each channel filtered independently\n• Stride: 1×1 (1×2 at Block 2)\n• Output: BatchNorm2d + ReLU"
    ax.text(67.0, 81.5, dw_desc, ha="left", va="center", color="#334155", fontsize=6.2, linespacing=1.3)

    ax.annotate("", xy=(80.5, 71.5), xytext=(80.5, 75), arrowprops=dict(arrowstyle="-|>", lw=1.2, color="#0284c7", mutation_scale=8))

    # Sub-block 2: Pointwise
    pw_box = patches.FancyBboxPatch((65, 52), 31, 17, boxstyle="round,pad=0.3,rounding_size=0.8",
                                    edgecolor="#7c3aed", facecolor="#f5f3ff", lw=1.1)
    ax.add_patch(pw_box)
    ax.text(80.5, 66.0, "Stage 2: Pointwise Conv (Channel)", ha="center", va="center", color="#6d28d9", fontsize=7.2, fontweight="bold")
    pw_desc = "• Kernel: 1×1, groups = 1\n• Linear projection across 48 channels\n• Learns cross-channel correlations\n• Output: BatchNorm2d + ReLU"
    ax.text(67.0, 58.5, pw_desc, ha="left", va="center", color="#334155", fontsize=6.2, linespacing=1.3)

    ax.annotate("", xy=(80.5, 48.5), xytext=(80.5, 52), arrowprops=dict(arrowstyle="-|>", lw=1.2, color="#7c3aed", mutation_scale=8))

    # Efficiency math card
    eff_box = patches.FancyBboxPatch((65, 28), 31, 18, boxstyle="round,pad=0.3,rounding_size=0.8",
                                     edgecolor="#059669", facecolor="#ecfdf5", lw=1.1)
    ax.add_patch(eff_box)
    ax.text(80.5, 42.5, "Computational Efficiency Gain", ha="center", va="center", color="#047857", fontsize=7.2, fontweight="bold")
    eff_desc = "Standard Conv FLOPs: Dk × Dk × M × N\nDS-Conv FLOPs: Dk × Dk × M + M × N\n\nComputation Ratio:\n  (1 / N) + (1 / Dk²)\n≈ 8.5× Fewer FLOPs vs Standard Conv"
    ax.text(67.0, 34.5, eff_desc, ha="left", va="center", color="#065f46", fontsize=5.8, linespacing=1.25)

    # Footer note
    ax.text(50, 19, "Designed for AI 231 ME2  |  From-scratch initialization (Zero pretrained weights)  |  Exportable to static INT8 ONNX",
            ha="center", va="center", color="#64748b", fontsize=6.8, style="italic")

def draw_page_2(fig):
    ax = fig.add_subplot(111)
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 140)
    ax.axis("off")

    # Header Banner
    header_box = patches.FancyBboxPatch((2, 126.5), 96, 11.5, boxstyle="round,pad=0.6,rounding_size=1.2",
                                        edgecolor="#1e293b", facecolor="#0f172a", lw=1.2)
    ax.add_patch(header_box)
    ax.text(50, 134.0, "TinyDSCNN-48 : Technical Specifications & Provenance",
            ha="center", va="center", color="#ffffff", fontsize=14, fontweight="bold", family="sans-serif")
    ax.text(50, 129.2, "Layer Specifications, Computational Complexity, Edge Footprint, and Academic References",
            ha="center", va="center", color="#94a3b8", fontsize=9.0, family="sans-serif")

    # -------------------------------------------------------------
    # SECTION 1: LAYER TABLE
    # -------------------------------------------------------------
    ax.text(2, 122.0, "1. Layer-by-Layer Dimensions & Parameter Counts", fontsize=10.5, fontweight="bold", color="#0f172a")

    hdr_box = patches.Rectangle((2, 115.5), 96, 5.0, facecolor="#1e293b", edgecolor="none")
    ax.add_patch(hdr_box)
    ax.text(3.5, 118.0, "Layer / Stage", color="#ffffff", fontsize=7.2, fontweight="bold")
    ax.text(21.5, 118.0, "Layer Operation / Class", color="#ffffff", fontsize=7.2, fontweight="bold")
    ax.text(49.5, 118.0, "Kernel / Stride / Config", color="#ffffff", fontsize=7.2, fontweight="bold")
    ax.text(73.5, 118.0, "Output Shape", color="#ffffff", fontsize=7.2, fontweight="bold")
    ax.text(96.0, 118.0, "Params", color="#ffffff", fontsize=7.2, fontweight="bold", ha="right")

    table_rows = [
        ("Input Audio", "Log-Mel Spectrogram", "16 kHz, 25ms win, 10ms hop", "(1, 40, 151)", "0"),
        ("Stem Conv", "nn.Conv2d + BN + ReLU", "5×5, s=(2,2), p=2, bias=F", "(48, 20, 76)", "1,296"),
        ("DS-Block 1 (DW)", "nn.Conv2d (groups=48) + BN + ReLU", "3×3, s=(1,1), p=1, bias=F", "(48, 20, 76)", "528"),
        ("DS-Block 1 (PW)", "nn.Conv2d + BN + ReLU", "1×1, s=(1,1), p=0, bias=F", "(48, 20, 76)", "2,400"),
        ("DS-Block 2 (DW)", "nn.Conv2d (groups=48) + BN + ReLU", "3×3, s=(1,2), p=1, bias=F [s_t=2]", "(48, 20, 38)", "528"),
        ("DS-Block 2 (PW)", "nn.Conv2d + BN + ReLU", "1×1, s=(1,1), p=0, bias=F", "(48, 20, 38)", "2,400"),
        ("DS-Block 3 (DW)", "nn.Conv2d (groups=48) + BN + ReLU", "3×3, s=(1,1), p=1, bias=F", "(48, 20, 38)", "528"),
        ("DS-Block 3 (PW)", "nn.Conv2d + BN + ReLU", "1×1, s=(1,1), p=0, bias=F", "(48, 20, 38)", "2,400"),
        ("DS-Block 4 (DW)", "nn.Conv2d (groups=48) + BN + ReLU", "3×3, s=(1,1), p=1, bias=F", "(48, 20, 38)", "528"),
        ("DS-Block 4 (PW)", "nn.Conv2d + BN + ReLU", "1×1, s=(1,1), p=0, bias=F", "(48, 20, 38)", "2,400"),
        ("Global Pooling", "AdaptiveAvgPool2d(1) + Flatten", "Global Average Pooling", "(48,)", "0"),
        ("Regularization", "nn.Dropout(p=0.15)", "Active during training", "(48,)", "0"),
        ("Classifier Head", "nn.Linear(48 -> 26 classes)", "Dense + Bias", "(26,)", "1,274"),
        ("Total (26 classes)", "Baseline Model (Synthetic Run)", "100 CUDA epochs", "Logits (26)", "14,282"),
        ("Total (32 classes)", "Option B Dataset Adaptation", "32 sub-classes / 19 intents", "Logits (32)", "14,576")
    ]

    ty = 111.0
    for idx, (stg, op, cfg, shp, prm) in enumerate(table_rows):
        is_total = "Total" in stg
        bg = "#e2e8f0" if is_total else ("#f8fafc" if idx % 2 == 0 else "#ffffff")
        rect = patches.Rectangle((2, ty), 96, 4.4, facecolor=bg, edgecolor="#cbd5e1", lw=0.5)
        ax.add_patch(rect)

        fw = "bold" if is_total else "normal"
        tc = "#0f172a" if is_total else "#334155"

        ax.text(3.5, ty + 2.2, stg, color=tc, fontsize=6.2, fontweight=fw, va="center")
        ax.text(21.5, ty + 2.2, op, color=tc, fontsize=6.0, fontweight=fw, va="center")
        ax.text(49.5, ty + 2.2, cfg, color=tc, fontsize=6.0, fontweight=fw, va="center")
        ax.text(73.5, ty + 2.2, shp, color=tc, fontsize=6.2, fontweight="bold" if is_total else "normal", family="monospace", va="center")
        ax.text(96.0, ty + 2.2, prm, color=tc, fontsize=6.2, fontweight="bold", ha="right", va="center")
        ty -= 4.4

    # -------------------------------------------------------------
    # SECTION 2: EDGE DEPLOYMENT SPECIFICATIONS
    # -------------------------------------------------------------
    ax.text(2, 42.0, "2. Edge Deployment & Hardware Footprint", fontsize=10.5, fontweight="bold", color="#0f172a")

    specs = [
        ("FP32 Uncompressed Model", "55.8 KB", "PyTorch .pt state_dict / FP32 ONNX weights"),
        ("INT8 Quantized ONNX Model", "44.6 KB", "Nine Conv2D layers and one Linear layer quantized to INT8"),
        ("Runtime RAM (RSS)", "73.45 MB", "Complete inference process including ONNX Runtime engine"),
        ("Single-Core CPU Latency", "0.501 ms", "P50 latency on standard x86 CPU thread; P95 is 0.679 ms"),
        ("Frontend + Inference Latency", "1.150 ms", "STFT + Mel filterbank (0.65 ms) + Neural forward pass (0.50 ms)"),
        ("Raspberry Pi Execution", "< 10 ms", "Easily satisfies real-time voice streaming constraint on ARM Cortex-A72/A76")
    ]

    for idx, (title, val, desc) in enumerate(specs):
        col = idx % 2
        row = idx // 2
        bx = 2 + col * 49
        by = 34 - row * 7.5
        box = patches.FancyBboxPatch((bx, by), 47, 6.6, boxstyle="round,pad=0.3,rounding_size=0.8",
                                     edgecolor="#cbd5e1", facecolor="#f8fafc", lw=1.0)
        ax.add_patch(box)
        ax.text(bx + 2, by + 4.6, title, color="#0f172a", fontsize=7.2, fontweight="bold")
        ax.text(bx + 45, by + 4.6, val, color="#2563eb", fontsize=7.2, fontweight="bold", ha="right")
        ax.text(bx + 2, by + 1.8, desc, color="#64748b", fontsize=5.8)

    # -------------------------------------------------------------
    # SECTION 3: ACADEMIC REFERENCES & PROVENANCE
    # -------------------------------------------------------------
    ax.text(2, 9.5, "3. Academic References & Design Attribution", fontsize=10.0, fontweight="bold", color="#0f172a")

    ref_box = patches.FancyBboxPatch((2, 1.5), 96, 6.8, boxstyle="round,pad=0.3,rounding_size=0.8",
                                     edgecolor="#94a3b8", facecolor="#ffffff", lw=1.0)
    ax.add_patch(ref_box)
    ax.text(3.5, 5.8, "[1] Zhang, Y., Suda, N., Lai, L., & Chandra, V. (2017). \"Hello Edge: Keyword Spotting on Microcontrollers.\" arXiv:1711.07128. ARM Applied ML Research.",
            color="#1e293b", fontsize=6.0, fontweight="bold")
    ax.text(3.5, 4.0, "[2] Howard, A. G., et al. (2017). \"MobileNets: Efficient Convolutional Neural Networks for Mobile Vision Applications.\" arXiv:1704.04861. Google Inc.",
            color="#475569", fontsize=6.0)
    ax.text(3.5, 2.3, "[3] AI 231 ME2 Adaptation: Scaled to 48 channels, time-strided Block 2, and trained purely from-scratch with zero pretrained weights or foundation models.",
            color="#64748b", fontsize=5.8, style="italic")

def generate():
    print("Regenerating TinyDSCNN-48 PDF and PNG...")
    with PdfPages(PDF_PATH) as pdf:
        # Page 1: Architecture & Block Diagram
        fig1 = plt.figure(figsize=(8.5, 11), dpi=300)
        draw_page_1(fig1)
        pdf.savefig(fig1, bbox_inches="tight")
        fig1.savefig(PNG_PATH, dpi=300, bbox_inches="tight")
        plt.close(fig1)

        # Page 2: Specs & References Table
        fig2 = plt.figure(figsize=(8.5, 11), dpi=300)
        draw_page_2(fig2)
        pdf.savefig(fig2, bbox_inches="tight")
        plt.close(fig2)

    print(f"Generated PDF: {PDF_PATH}")
    print(f"Generated PNG: {PNG_PATH}")

if __name__ == "__main__":
    generate()

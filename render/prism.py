"""
render/prism.py
Matplotlib visualisation dashboard for Neuryx sessions.

Called after inference completes; receives raw data and draws
an 8-panel dark-theme chart grid, then saves a PNG.
"""

from __future__ import annotations


def render_dashboard(
    train_docs:     list[str],
    pred_docs:      list[str],
    loss_chronicle: list[float],
    generated:      list[str],
    vocab_registry: list[str],
    output_path:    str = "neuryx_dashboard.png",
) -> None:
    """
    Draw a full Neuryx session dashboard.

    Parameters
    ----------
    train_docs      : raw training document strings
    pred_docs       : raw prediction-seed document strings
    loss_chronicle  : per-step loss values from Forge.anneal()
    generated       : list of generated / predicted output strings
    vocab_registry  : the Cipher's registry (vocabulary list)
    output_path     : where to save the PNG
    """
    try:
        import matplotlib
        import matplotlib.pyplot   as plt
        import matplotlib.gridspec as gridspec
        from   collections import Counter
    except ImportError:
        print("[render/prism] matplotlib not installed — skipping dashboard.")
        print("  Install with:  pip install matplotlib")
        return

    # ── Palette ───────────────────────────────────────────────────────────────
    BG   = "#0b0b18"
    PAN  = "#11112a"
    G1   = "#00e5c8"    # teal
    G2   = "#7b2fff"    # violet
    G3   = "#ff6b6b"    # coral
    G4   = "#ffd166"    # amber
    FG   = "#e4e4f0"
    DIM  = "#3a3a5c"
    GOOD = "#06d6a0"
    WARN = "#ef476f"

    plt.rcParams.update({
        "figure.facecolor":  BG,
        "axes.facecolor":    PAN,
        "axes.edgecolor":    DIM,
        "axes.labelcolor":   FG,
        "axes.titlecolor":   FG,
        "xtick.color":       FG,
        "ytick.color":       FG,
        "text.color":        FG,
        "grid.color":        DIM,
        "legend.facecolor":  "#1a1a38",
        "legend.edgecolor":  DIM,
        "font.family":       "monospace",
        "font.size":         9,
    })

    fig = plt.figure(figsize=(22, 16))
    fig.suptitle(
        "  NEURYX  ·  Neural Sequence Engine  ·  Session Dashboard",
        fontsize=14, fontweight="bold", color=G1, y=0.998, x=0.01, ha="left",
    )
    gs = gridspec.GridSpec(3, 3, figure=fig, hspace=0.55, wspace=0.38)

    def style(ax, title: str) -> None:
        ax.set_title(f"  {title}", fontweight="bold", fontsize=9.5, loc="left", pad=6)
        ax.grid(True, alpha=0.15)
        for sp in ax.spines.values():
            sp.set_edgecolor(DIM)

    # ── 1. Training Loss ──────────────────────────────────────────────────────
    ax1 = fig.add_subplot(gs[0, 0])
    steps = list(range(1, len(loss_chronicle) + 1))
    ax1.plot(steps, loss_chronicle, color=G2, lw=0.7, alpha=0.4, label="raw")
    win = max(1, len(steps) // 15)
    smooth = [
        sum(loss_chronicle[max(0, i - win): i + 1]) /
        len(loss_chronicle[max(0, i - win): i + 1])
        for i in range(len(loss_chronicle))
    ]
    ax1.plot(steps, smooth, color=G1, lw=2, label="smoothed")
    ax1.fill_between(steps, smooth, alpha=0.08, color=G1)
    ax1.set_xlabel("Step"); ax1.set_ylabel("Cross-Entropy Loss")
    ax1.legend(fontsize=8)
    style(ax1, "① Training Loss Curve")

    # ── 2. Document Length Distribution (train) ───────────────────────────────
    ax2 = fig.add_subplot(gs[0, 1])
    lens = [len(d) for d in train_docs]
    ax2.hist(lens, bins=min(40, len(set(lens))), color=G1, alpha=0.75, edgecolor=BG)
    ax2.axvline(sum(lens) / max(len(lens), 1), color=G4, lw=1.5,
                linestyle="--", label=f"mean={sum(lens)/max(len(lens),1):.0f}")
    ax2.set_xlabel("Document Length (chars)"); ax2.set_ylabel("Count")
    ax2.legend(fontsize=8)
    style(ax2, "② Training Document Lengths")

    # ── 3. Top-20 Vocabulary Symbols ─────────────────────────────────────────
    ax3 = fig.add_subplot(gs[0, 2])
    all_chars = "".join(train_docs)
    counts    = Counter(all_chars).most_common(20)
    if counts:
        labels, vals = zip(*counts)
        labels = [repr(c)[1:-1] for c in labels]  # escape special chars
        ax3.barh(range(len(vals)), list(vals), color=G2, alpha=0.80, edgecolor=BG)
        ax3.set_yticks(range(len(labels)))
        ax3.set_yticklabels(labels, fontsize=8)
        ax3.set_xlabel("Frequency")
    style(ax3, "③ Top Symbol Frequencies (Train)")

    # ── 4. Loss Moving Average — smoothed only ────────────────────────────────
    ax4 = fig.add_subplot(gs[1, 0])
    # Show first-half vs second-half loss to visualise convergence
    half = len(smooth) // 2
    ax4.plot(steps[:half], smooth[:half], color=G3, lw=1.5, label="first half")
    ax4.plot(steps[half:], smooth[half:], color=GOOD,  lw=1.5, label="second half")
    ax4.fill_between(steps[:half], smooth[:half], alpha=0.10, color=G3)
    ax4.fill_between(steps[half:], smooth[half:], alpha=0.10, color=GOOD)
    ax4.set_xlabel("Step"); ax4.set_ylabel("Smoothed Loss")
    ax4.legend(fontsize=8)
    style(ax4, "④ Convergence (first vs second half)")

    # ── 5. Generated Output Length Distribution ───────────────────────────────
    ax5 = fig.add_subplot(gs[1, 1])
    gen_lens = [len(g) for g in generated] if generated else [0]
    ax5.hist(gen_lens, bins=min(20, max(len(set(gen_lens)), 2)),
             color=G4, alpha=0.80, edgecolor=BG)
    ax5.set_xlabel("Output Length (chars)"); ax5.set_ylabel("Count")
    style(ax5, "⑤ Generated Output Lengths")

    # ── 6. Vocabulary coverage ────────────────────────────────────────────────
    ax6 = fig.add_subplot(gs[1, 2])
    n_vocab   = len(vocab_registry)
    n_used    = len(set("".join(train_docs)) & set(vocab_registry))
    n_unseen  = n_vocab - n_used
    ax6.bar(["In training", "Unseen"], [n_used, n_unseen],
            color=[GOOD, WARN], alpha=0.85, edgecolor=BG, width=0.5)
    ax6.set_ylabel("Token count")
    for i, v in enumerate([n_used, n_unseen]):
        ax6.text(i, v + 0.5, str(v), ha="center", fontsize=10, fontweight="bold")
    style(ax6, "⑥ Vocabulary Coverage")

    # ── 7. Per-step loss variance ─────────────────────────────────────────────
    ax7 = fig.add_subplot(gs[2, 0])
    seg = max(1, len(loss_chronicle) // 20)
    seg_std = [
        (
            sum(loss_chronicle[i: i + seg]) / seg,
            (sum((x - sum(loss_chronicle[i: i + seg]) / seg) ** 2
                 for x in loss_chronicle[i: i + seg]) / max(seg, 1)) ** 0.5
        )
        for i in range(0, len(loss_chronicle) - seg + 1, seg)
    ]
    xs  = list(range(len(seg_std)))
    mns = [s[0] for s in seg_std]
    std = [s[1] for s in seg_std]
    ax7.plot(xs, mns, color=G1, lw=2)
    ax7.fill_between(
        xs,
        [m - s for m, s in zip(mns, std)],
        [m + s for m, s in zip(mns, std)],
        alpha=0.25, color=G1, label="±1 std",
    )
    ax7.set_xlabel("Segment"); ax7.set_ylabel("Loss")
    ax7.legend(fontsize=8)
    style(ax7, "⑦ Loss Mean ± Variance")

    # ── 8. Unique vs total generated symbols ──────────────────────────────────
    ax8 = fig.add_subplot(gs[2, 1:])
    if generated:
        cum_total  = []
        cum_unique = []
        seen_set: set[str] = set()
        total = 0
        for g in generated:
            for ch in g:
                total += 1
                seen_set.add(ch)
            cum_total.append(total)
            cum_unique.append(len(seen_set))
        xi = range(len(generated))
        ax8.plot(xi, cum_total,  color=G1, lw=2, label="Total chars generated")
        ax8.plot(xi, cum_unique, color=G4, lw=2, label="Unique symbols used")
        ax8.fill_between(xi, cum_unique, alpha=0.12, color=G4)
        ax8.set_xlabel("Sample index"); ax8.set_ylabel("Cumulative count")
        ax8.legend(fontsize=9)
    style(ax8, "⑧ Generated Output — Cumulative Symbol Diversity")

    plt.savefig(output_path, dpi=150, bbox_inches="tight", facecolor=BG)
    print(f"  Dashboard saved → {output_path}")
    try:
        plt.show()
    except Exception:
        pass

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_SECONDARY = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
BASELINE = "#c3c2b7"

# Rampa ordinal azul (pasos 250-650), validada con validate_palette --ordinal.
TEMPERATURE_RAMP = ["#86b6ef", "#5598e7", "#2a78d6", "#1c5cab", "#104281"]
# Categórica, slots 1-3 (validados all-pairs) + gris recesivo para "ninguno".
SERIES_BLUE, SERIES_ORANGE, SERIES_AQUA = "#2a78d6", "#eb6834", "#1baf7a"
NEUTRAL_MARK = "#c3c2b7"

DPI = 200


def _style(ax) -> None:
    ax.set_facecolor(SURFACE)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(BASELINE)
        ax.spines[side].set_linewidth(0.8)
    ax.tick_params(colors=INK_SECONDARY, labelsize=8, length=3)
    ax.yaxis.grid(True, color=GRID, linewidth=0.6)
    ax.set_axisbelow(True)


def _new_figure(width: float, height: float, **kwargs):
    fig, axes = plt.subplots(figsize=(width, height), facecolor=SURFACE, **kwargs)
    for ax in np.atleast_1d(axes).ravel():
        _style(ax)
    return fig, axes


def visible_token(token: str) -> str:
    """Hace visibles los espacios y saltos de línea de los tokens de GPT-2."""
    return token.replace("\n", "\\n").replace(" ", "␣", 1) if token.startswith(" ") else token.replace("\n", "\\n")


def plot_temperature(prefix: str, entries: list[dict], path) -> None:
    """Top 15 tokens (mismo orden para toda temperatura) con una barra por temperatura."""
    fig, ax = _new_figure(11, 5.2)
    tokens = [visible_token(t) for t in entries[0]["top_tokens"]]
    n = len(entries)
    width = 0.84 / n
    x = np.arange(len(tokens))
    for i, entry in enumerate(entries):
        label = (
            f"T = {entry['temperature']:g}:  p(1.º) = {entry['top_probs'][0]:.3g}  ·  "
            f"{entry['entropy_bits']:.2f} bits  ·  núcleo {entry['nucleus_size_p90']}"
        )
        ax.bar(
            x + (i - (n - 1) / 2) * width, entry["top_probs"], width, label=label,
            color=TEMPERATURE_RAMP[i], edgecolor=SURFACE, linewidth=0.7,
        )

    ax.set_xticks(x)
    ax.set_xticklabels(tokens, rotation=40, ha="right")
    ax.set_ylabel("Probabilidad del siguiente token", color=INK_SECONDARY, fontsize=9)
    ax.set_xlabel("Top 15 tokens (orden de probabilidad)", color=INK_SECONDARY, fontsize=9)
    ax.set_title(f"GPT-2: probabilidad de los 15 tokens más probables por temperatura\nPrefijo: «{prefix}»",
                 loc="left", fontsize=11, color=INK)
    legend = ax.legend(title="Temperatura: probabilidad del 1.er token, entropía y tamaño del núcleo top-p 0.9", frameon=False, fontsize=8, title_fontsize=8)
    legend.get_title().set_color(INK_SECONDARY)
    for text in legend.get_texts():
        text.set_color(INK_SECONDARY)
    fig.tight_layout()
    fig.savefig(path, dpi=DPI, facecolor=SURFACE)
    plt.close(fig)


MEMBERSHIP = {
    "both": ("Top-k y top-p", SERIES_BLUE),
    "top_k_only": ("Solo top-k", SERIES_ORANGE),
    "top_p_only": ("Solo top-p", SERIES_AQUA),
    "neither": ("Ninguno", NEUTRAL_MARK),
}


def plot_topk_vs_topp(panels: list[dict], k: int, p: float, path) -> None:
    """Un panel por prefijo: los tokens más probables, coloreados según qué corte los conserva."""
    fig, axes = _new_figure(11, 3.4 * len(panels), nrows=len(panels))
    axes = np.atleast_1d(axes)
    # Escala log: con un token dominante (p ~ 0.99) el resto sería invisible y los cortes no se distinguirían.
    lowest = min(min(panel["probs"]) for panel in panels)
    bottom = 10 ** np.floor(np.log10(lowest))
    for ax, panel in zip(axes, panels):
        ax.set_yscale("log")
        ax.set_ylim(bottom, 1.6)
        tokens = [visible_token(t) for t in panel["tokens"]]
        colors = [MEMBERSHIP[m][1] for m in panel["membership"]]
        x = np.arange(len(tokens))
        ax.bar(x, panel["probs"], 0.62, color=colors, edgecolor=SURFACE, linewidth=0.7)
        ax.set_xticks(x)
        ax.set_xticklabels(tokens, rotation=40, ha="right")
        ax.set_ylabel("Probabilidad (T = 1, escala log)", color=INK_SECONDARY, fontsize=9)
        ax.set_title(
            f"Prefijo «{panel['prefix']}»: top-k={k} conserva {panel['top_k_size']} tokens, "
            f"top-p={p:g} conserva {panel['top_p_size']}",
            loc="left", fontsize=10, color=INK,
        )
    handles = [plt.Rectangle((0, 0), 1, 1, color=color) for _, color in MEMBERSHIP.values()]
    fig.legend(handles, [name for name, _ in MEMBERSHIP.values()], loc="upper right", ncol=4, frameon=False,
               fontsize=8, labelcolor=INK_SECONDARY)
    fig.suptitle("GPT-2: qué tokens sobreviven al corte top-k y al corte top-p", x=0.01, ha="left", fontsize=11, color=INK)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(path, dpi=DPI, facecolor=SURFACE)
    plt.close(fig)


def plot_entropy(series: dict[str, list[tuple[float, float]]], path) -> None:
    """Entropía frente a temperatura, una línea por prefijo, etiquetada al final de la línea."""
    fig, ax = _new_figure(7.5, 4.4)
    colors = [SERIES_BLUE, SERIES_ORANGE]
    offsets = [(8, 7), (8, -7)]  # etiquetas finales separadas para que no se superpongan
    for (label, points), color, offset in zip(series.items(), colors, offsets):
        temps, values = zip(*points)
        ax.plot(temps, values, color=color, linewidth=2, marker="o", markersize=7, markeredgecolor=SURFACE,
                markeredgewidth=1.5, label=label)
        ax.annotate(f"{values[-1]:.1f}", (temps[-1], values[-1]), xytext=offset, textcoords="offset points",
                    va="center", fontsize=8, color=INK)
    ax.set_xlabel("Temperatura", color=INK_SECONDARY, fontsize=9)
    ax.set_ylabel("Entropía (bits)", color=INK_SECONDARY, fontsize=9)
    ax.set_xlim(0, 2.25)
    ax.set_ylim(bottom=0)
    ax.set_title("GPT-2: entropía de la distribución del siguiente token según la temperatura", loc="left",
                 fontsize=11, color=INK)
    legend = ax.legend(frameon=False, fontsize=8)
    for text in legend.get_texts():
        text.set_color(INK_SECONDARY)
    fig.tight_layout()
    fig.savefig(path, dpi=DPI, facecolor=SURFACE)
    plt.close(fig)

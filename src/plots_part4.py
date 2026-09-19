from src.plots_part0 import INK, INK_SECONDARY, SERIES_BLUE, SERIES_ORANGE, SURFACE, DPI, _new_figure, plt


def group_points(points: list[tuple[float, float, str]], places: int = 9) -> list[tuple[float, float, str]]:
    """Une los puntos con las mismas coordenadas en una sola marca con una sola etiqueta ('none, low, medium')."""
    grouped: dict[tuple[float, float], list[str]] = {}
    for x, y, label in points:
        grouped.setdefault((round(x, places), round(y, places)), []).append(label)
    return [(x, y, ", ".join(labels)) for (x, y), labels in grouped.items()]


def _draw_series(ax, points, color, x_scale: float = 1.0) -> None:
    grouped = group_points([(x * x_scale, y, label) for x, y, label in points])
    xs, ys = [g[0] for g in grouped], [g[1] for g in grouped]
    ax.plot(xs, ys, color=color, linestyle="none", marker="o", markersize=9, markeredgecolor=SURFACE, markeredgewidth=1.5)
    # Etiquetas escalonadas para que las de puntos vecinos no se pisen.
    for i, (x, y, label) in enumerate(grouped):
        ax.annotate(label, (x, y), xytext=(-2 if i == 0 else 0, -16 - 15 * (i % 2)), textcoords="offset points",
                    ha="left" if i == 0 else "center", fontsize=8, color=INK)


def describe_range(low: float, high: float) -> str:
    return f"Exactitud {low:.2f} en todos los niveles" if abs(high - low) < 1e-9 else f"Exactitud entre {low:.2f} y {high:.2f}"


def _finish(ax, xlabel: str, title: str) -> None:
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Exactitud", color=INK_SECONDARY, fontsize=9)
    ax.set_xlabel(xlabel, color=INK_SECONDARY, fontsize=9)
    ax.set_title(title, loc="left", fontsize=10, color=INK)
    ax.margins(x=0.12)


def plot_accuracy_vs_reasoning(tickets: list[dict], control: list[dict], path) -> None:
    fig, axes = _new_figure(11, 3.8, ncols=2)
    for ax, table, color, name in ((axes[0], tickets, SERIES_BLUE, "Clasificación de tickets (10 casos × 3 corridas)"),
                                   (axes[1], control, SERIES_ORANGE, "Control: acertijo del bate y la pelota (3 corridas)")):
        points = [(t["reasoning_tokens_mean"], t["accuracy"] if "accuracy" in t else t["answer_correct_rate"], t["effort"])
                  for t in table if t["reasoning_tokens_mean"] is not None]
        _draw_series(ax, points, color)
        low, high = min(p[1] for p in points), max(p[1] for p in points)
        _finish(ax, "Tokens de razonamiento por llamada (media)", f"{name}\n{describe_range(low, high)}")
    fig.suptitle("gpt-5.6-luna: exactitud frente a tokens de razonamiento, por nivel de esfuerzo", x=0.01, ha="left", fontsize=11, color=INK)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(path, dpi=DPI, facecolor=SURFACE)
    plt.close(fig)


def plot_cost_vs_accuracy(tickets: list[dict], path) -> None:
    fig, ax = _new_figure(7.5, 3.8)
    scale = 1e5
    points = [(t["cost_usd_per_call"], t["accuracy"], t["effort"]) for t in tickets if t["accuracy"] is not None]
    _draw_series(ax, points, SERIES_BLUE, x_scale=scale)
    low, high = min(p[1] for p in points), max(p[1] for p in points)
    cheapest, dearest = min(p[0] for p in points), max(p[0] for p in points)
    _finish(ax, "Costo por llamada (×10⁻⁵ USD)",
            f"gpt-5.6-luna, clasificación de tickets: costo frente a exactitud\n"
            f"{describe_range(low, high)}; costo +{100 * (dearest / cheapest - 1):.0f} % del nivel más barato al más caro")
    fig.tight_layout()
    fig.savefig(path, dpi=DPI, facecolor=SURFACE)
    plt.close(fig)

"""
Helper-и для збереження двомовних рисунків.

Реалізує: розділ 4.3 протоколу R1
(див. docs/Протокол_R1.docx).
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

import matplotlib.pyplot as plt


def save_dual_lang(
    plot_func: Callable[..., plt.Figure],
    output_path_template: str,
    **kwargs,
) -> list[Path]:
    """
    Викликає plot_func двічі (uk/en) і зберігає два PNG.

    output_path_template має містити `{lang}`, наприклад:
        results/figures/article3/R1_strategy_comparison_{lang}.png
    """
    outputs: list[Path] = []
    for lang in ["uk", "en"]:
        fig = plot_func(lang=lang, **kwargs)
        out = Path(output_path_template.format(lang=lang))
        out.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out, dpi=300, bbox_inches="tight")
        plt.close(fig)
        outputs.append(out)
    return outputs

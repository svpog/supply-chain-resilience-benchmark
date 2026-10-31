"""
Тести i18n-інфраструктури для двомовних рисунків.

Реалізує: розділ 4.4 протоколу R1
(див. docs/Протокол_R1.docx).
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt

from src.visualization.i18n.labels import label, load_labels
from src.visualization.i18n.plot_helpers import save_dual_lang


def _assert_lang_pairs(obj) -> None:
    if isinstance(obj, dict):
        if set(obj.keys()) == {"uk", "en"}:
            assert obj["uk"]
            assert obj["en"]
        else:
            for value in obj.values():
                _assert_lang_pairs(value)


def test_labels_json_loads() -> None:
    labels = load_labels()
    assert "common" in labels
    assert "R1" in labels


def test_all_terminal_labels_have_uk_and_en() -> None:
    _assert_lang_pairs(load_labels())


def test_label_gets_nested_key() -> None:
    assert label("common.topology", "en") == "Topology"
    assert label("common.topology", "uk") == "Топологія"


def test_save_dual_lang_creates_two_files(tmp_path: Path) -> None:
    def plot_func(lang: str):
        fig, ax = plt.subplots()
        ax.set_title(label("common.topology", lang))
        return fig

    outputs = save_dual_lang(plot_func, str(tmp_path / "figure_{lang}.png"))

    assert len(outputs) == 2
    assert (tmp_path / "figure_uk.png").exists()
    assert (tmp_path / "figure_en.png").exists()

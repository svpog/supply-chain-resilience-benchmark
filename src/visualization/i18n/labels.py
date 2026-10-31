"""
Завантаження двомовних підписів для рисунків.

Реалізує: розділ 4.2 протоколу R1
(див. docs/Протокол_R1.docx).

Приклад використання:
    >>> from src.visualization.i18n.labels import label
    >>> label("common.topology", "en")
    'Topology'
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


_LABELS_CACHE: dict[str, Any] | None = None


def load_labels() -> dict[str, Any]:
    """Завантажує config/i18n/labels.json з кешуванням."""
    global _LABELS_CACHE
    if _LABELS_CACHE is None:
        path = Path(__file__).resolve().parents[3] / "config" / "i18n" / "labels.json"
        with path.open(encoding="utf-8") as f:
            _LABELS_CACHE = json.load(f)
    return _LABELS_CACHE


def label(key_path: str, lang: str = "uk") -> str:
    """
    Повертає підпис за шляхом виду 'R1.strategy_comparison.title'.

    Args:
        key_path: dot-separated шлях у labels.json.
        lang: 'uk' або 'en'.
    """
    obj: Any = load_labels()
    for part in key_path.split("."):
        obj = obj[part]
    return str(obj[lang])

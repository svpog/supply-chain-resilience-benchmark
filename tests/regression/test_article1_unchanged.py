"""
Regression-тести зворотної сумісності статті 1.

Реалізує: розділ 2.1 протоколу R1 для статті 3
(див. docs/Протокол_R1.docx).

Тест фіксує поточний прийнятий стан статті 1 після schema evolution S3.
Він не порівнює з історичним тегом v1.0-canonical-pilot, бо поточна схема
raw_results.csv вже містить додаткові S3-поля; натомість гарантує, що
подальші зміни recovery-архітектури не змінять поточний S1 output.
"""

from __future__ import annotations

import hashlib
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
EXPECTED = ROOT / "tests" / "regression" / "s1_canonical.sha256"


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def _expected_hashes() -> dict[str, str]:
    hashes: dict[str, str] = {}
    for line in EXPECTED.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        digest, rel_path = line.split(maxsplit=1)
        hashes[rel_path.strip()] = digest
    return hashes


def test_S1_canonical_unchanged(tmp_path: Path) -> None:
    """Канонічний S1 має лишатися побайтово ідентичним поточному baseline."""
    out_dir = tmp_path / "test_s1"
    subprocess.run(
        [
            sys.executable,
            "main.py",
            "--series",
            "S1",
            "--runs",
            "100",
            "--output-dir",
            str(out_dir),
        ],
        cwd=ROOT,
        check=True,
    )

    for rel_path, expected in _expected_hashes().items():
        actual_path = out_dir / rel_path
        assert actual_path.exists(), f"Очікуваний файл не створено: {actual_path}"
        actual = _sha256(actual_path)
        assert actual == expected, (
            f"S1 канонічний пілот змінився для {rel_path}: "
            f"expected {expected}, actual {actual}"
        )

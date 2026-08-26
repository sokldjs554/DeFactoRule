"""OCR engine boundary. Tesseract is the baseline, not a hard-coded product dependency."""
from __future__ import annotations

import shutil
import subprocess
from dataclasses import dataclass
from statistics import mean


class OcrEngineUnavailable(RuntimeError):
    """Raised when the configured OCR executable/language is not available."""


@dataclass(frozen=True)
class OcrOutput:
    text: str
    engine: str
    mean_confidence: float | None = None
    low_confidence_fraction: float | None = None


class TesseractOCR:
    """Small subprocess adapter around Tesseract 5.x with word confidence signals."""

    def __init__(self, language: str = "kor", psm: int = 6, executable: str = "tesseract"):
        self.language = language
        self.psm = psm
        self.executable = executable

    def version(self) -> str:
        exe = shutil.which(self.executable)
        if not exe:
            raise OcrEngineUnavailable(f"OCR executable not found: {self.executable}")
        proc = subprocess.run(
            [exe, "--version"], capture_output=True, text=True, check=False
        )
        first = (proc.stdout or proc.stderr).splitlines()
        return first[0].strip() if first else "tesseract"

    @staticmethod
    def _parse_tsv(payload: str) -> tuple[str, float | None, float | None]:
        lines: dict[tuple[int, int, int, int], list[str]] = {}
        confidences: list[float] = []
        for raw in payload.splitlines()[1:]:
            parts = raw.split("\t", 11)
            if len(parts) != 12:
                continue
            try:
                level = int(parts[0])
                page = int(parts[1])
                block = int(parts[2])
                paragraph = int(parts[3])
                line = int(parts[4])
                confidence = float(parts[10])
            except ValueError:
                continue
            text = parts[11].strip()
            if level != 5 or not text:
                continue
            key = (page, block, paragraph, line)
            lines.setdefault(key, []).append(text)
            if confidence >= 0:
                confidences.append(confidence)

        text = "\n".join(" ".join(words) for _, words in sorted(lines.items())).strip()
        if not confidences:
            return text, None, None
        low = sum(value < 60.0 for value in confidences)
        return text, mean(confidences), low / len(confidences)

    def recognize(self, image_bytes: bytes) -> OcrOutput:
        exe = shutil.which(self.executable)
        if not exe:
            raise OcrEngineUnavailable(f"OCR executable not found: {self.executable}")
        proc = subprocess.run(
            [
                exe,
                "stdin",
                "stdout",
                "-l",
                self.language,
                "--psm",
                str(self.psm),
                "tsv",
            ],
            input=image_bytes,
            capture_output=True,
            check=False,
        )
        if proc.returncode != 0:
            detail = proc.stderr.decode("utf-8", errors="replace").strip()
            raise OcrEngineUnavailable(
                f"tesseract failed (lang={self.language}, psm={self.psm}): {detail[:500]}"
            )
        text, mean_confidence, low_confidence_fraction = self._parse_tsv(
            proc.stdout.decode("utf-8", errors="replace")
        )
        return OcrOutput(
            text=text,
            engine=f"{self.version()} lang={self.language} psm={self.psm}",
            mean_confidence=mean_confidence,
            low_confidence_fraction=low_confidence_fraction,
        )

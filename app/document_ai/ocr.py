"""OCR engine boundary. Tesseract is the baseline, not a hard-coded product dependency."""
from __future__ import annotations

import shutil
import subprocess
from dataclasses import dataclass


class OcrEngineUnavailable(RuntimeError):
    """Raised when the configured OCR executable/language is not available."""


@dataclass(frozen=True)
class OcrOutput:
    text: str
    engine: str


class TesseractOCR:
    """Small subprocess adapter around Tesseract 5.x."""

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
            ],
            input=image_bytes,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False,
        )
        if proc.returncode != 0:
            detail = proc.stderr.decode("utf-8", errors="replace").strip()
            raise OcrEngineUnavailable(
                f"tesseract failed (lang={self.language}, psm={self.psm}): {detail[:500]}"
            )
        return OcrOutput(
            text=proc.stdout.decode("utf-8", errors="replace").strip(),
            engine=f"{self.version()} lang={self.language} psm={self.psm}",
        )

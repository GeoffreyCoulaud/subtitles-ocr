from fractions import Fraction
from pathlib import Path

import pytest

from subtitles_ocr.config import PipelineGlobals


@pytest.fixture
def tmp_workdir(tmp_path: Path) -> Path:
    for d in (
        "01_conform",
        "02_alignment",
        "04_mask",
        "06_ocr",
        "07_group",
        "08_animation",
        "09_color",
        "10_event_cleanup",
        "11_doc_cleanup",
    ):
        (tmp_path / d).mkdir(parents=True)
    return tmp_path


@pytest.fixture
def mock_globals(tmp_workdir: Path) -> PipelineGlobals:
    return PipelineGlobals(
        workdir=tmp_workdir,
        hardsub_path=Path("/dev/null/fake_hardsub.avi"),
        raw_path=Path("/dev/null/fake_raw.mkv"),
        out_path=tmp_workdir / "out.ass",
        fps=Fraction(24, 1),
        fansub_width=1920,
        fansub_height=1080,
        fansub_total_frames=1000,
        debug_images=False,
    )

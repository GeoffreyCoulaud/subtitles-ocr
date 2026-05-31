import json
import unicodedata
from fractions import Fraction
from pathlib import Path

from subtitles_ocr.config import NormalizeConfig, PipelineGlobals
from subtitles_ocr.pipeline.normalize import (
    NormalizedEvent,
    NormalizeResult,
    NormalizeStage,
    is_noise,
    normalize_text,
)


# All non-ASCII codepoints are constructed via chr() to avoid any visual
# ambiguity in this source file (decomposed vs composed, NBSP vs SPACE,
# invisible chars, etc.).

NBSP = chr(0x00A0)
ZWSP = chr(0x200B)
ZWNJ = chr(0x200C)
ZWJ = chr(0x200D)
BOM = chr(0xFEFF)
WORD_JOINER = chr(0x2060)
LIG_FI = chr(0xFB01)  # LATIN SMALL LIGATURE FI
FULLWIDTH_A = chr(0xFF21)  # FULLWIDTH LATIN CAPITAL LETTER A
E_ACUTE = chr(0x00E9)  # composed "é"
COMBINING_ACUTE = chr(0x0301)


# -------------------- NFKC --------------------


def test_normalize_text_nfkc_composes_decomposed_e_acute() -> None:
    composed = "caf" + E_ACUTE
    decomposed = unicodedata.normalize("NFD", composed)
    assert decomposed != composed  # sanity: NFD truly differs from NFC here
    assert len(decomposed) == 5 and len(composed) == 4
    assert normalize_text(decomposed) == composed


def test_normalize_text_nfkc_folds_ligature_fi() -> None:
    inp = "ef" + LIG_FI + "ciency"
    assert LIG_FI in inp and len(LIG_FI) == 1
    assert normalize_text(inp) == "efficiency"


def test_normalize_text_nfkc_normalizes_nbsp_to_space() -> None:
    inp = "a" + NBSP + "b"
    assert ord(inp[1]) == 0xA0  # sanity
    assert normalize_text(inp) == "a b"


def test_normalize_text_nfkc_folds_fullwidth_a() -> None:
    inp = FULLWIDTH_A + "BC"
    assert ord(inp[0]) == 0xFF21  # sanity
    assert normalize_text(inp) == "ABC"


# -------------------- invisible char stripping --------------------


def test_normalize_text_strips_zwsp() -> None:
    assert normalize_text("a" + ZWSP + "b") == "ab"


def test_normalize_text_strips_zwnj() -> None:
    assert normalize_text("a" + ZWNJ + "b") == "ab"


def test_normalize_text_strips_bom() -> None:
    assert normalize_text(BOM + "hello") == "hello"


def test_normalize_text_strips_word_joiner() -> None:
    assert normalize_text("a" + WORD_JOINER + "b") == "ab"


def test_normalize_text_keeps_zwj() -> None:
    inp = "a" + ZWJ + "b"
    assert normalize_text(inp) == inp


# -------------------- whitespace --------------------


def test_normalize_text_collapses_multiple_spaces() -> None:
    assert normalize_text("a   b    c") == "a b c"


def test_normalize_text_collapses_tabs() -> None:
    assert normalize_text("a\t\tb") == "a b"


def test_normalize_text_preserves_internal_newline() -> None:
    assert normalize_text("a\nb") == "a\nb"


def test_normalize_text_trims_spaces_around_newline() -> None:
    assert normalize_text("a  \n  b") == "a\nb"


def test_normalize_text_strips_leading_trailing_whitespace() -> None:
    assert normalize_text("  hello world  ") == "hello world"


def test_normalize_text_strips_leading_trailing_newlines() -> None:
    assert normalize_text("\n\nhello\n\n") == "hello"


# -------------------- combined --------------------


def test_normalize_text_idempotent_on_clean_input() -> None:
    assert normalize_text("hello world") == "hello world"


def test_normalize_text_empty_string() -> None:
    assert normalize_text("") == ""


# -------------------- noise detection --------------------


def test_is_noise_single_digit() -> None:
    assert is_noise("1") is True


def test_is_noise_single_punct() -> None:
    assert is_noise(":") is True


def test_is_noise_dash() -> None:
    assert is_noise("-") is True


def test_is_noise_all_digits() -> None:
    assert is_noise("123") is True  # len >= 2 but no alpha


def test_is_noise_punct_only() -> None:
    assert is_noise("...") is True


def test_is_noise_two_letters() -> None:
    assert is_noise("Ok") is False


def test_is_noise_letter_plus_digit() -> None:
    # Single-letter "word" (e.g. "a1", "1-E") is treated as noise because
    # fansubs render such labels with custom positioning/rotation our OCR
    # cannot reproduce; the ≥ 2-alpha floor keeps them out of the export.
    assert is_noise("a1") is True
    assert is_noise("ab1") is False


def test_is_noise_accented_letter_only() -> None:
    # Single composed "é" is alphabetic but len < 2.
    assert is_noise(E_ACUTE) is True


def test_is_noise_two_accented_letters() -> None:
    assert is_noise(E_ACUTE + E_ACUTE) is False


def test_is_noise_empty() -> None:
    assert is_noise("") is True


# -------------------- model construction --------------------


def test_normalized_event_round_trip() -> None:
    ev = NormalizedEvent(event_id=42, cleaned_text="hello")
    dumped = ev.model_dump_json()
    parsed = NormalizedEvent.model_validate_json(dumped)
    assert parsed == ev


def test_normalize_result_round_trip() -> None:
    res = NormalizeResult(
        events=[
            NormalizedEvent(event_id=0, cleaned_text="a"),
            NormalizedEvent(event_id=2, cleaned_text="b"),
        ]
    )
    dumped = res.model_dump_json()
    parsed = NormalizeResult.model_validate_json(dumped)
    assert parsed == res
    # Non-contiguous event_ids are explicitly allowed.
    assert [e.event_id for e in parsed.events] == [0, 2]


# -------------------- stage orchestration --------------------


def _write_event_cleanup_jsonl(workdir: Path, events: list[tuple[int, str]]) -> None:
    d = workdir / "10_event_cleanup"
    d.mkdir(parents=True, exist_ok=True)
    lines = [
        json.dumps(
            {"event_id": eid, "cleaned_text": text, "skipped_llm": False},
            ensure_ascii=False,
        )
        for (eid, text) in events
    ]
    (d / "cleaned.jsonl").write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")


def _make_globals(workdir: Path) -> PipelineGlobals:
    return PipelineGlobals(
        workdir=workdir,
        hardsub_path=Path("/dev/null/h"),
        raw_path=Path("/dev/null/r"),
        out_path=workdir / "out.ass",
        fps=Fraction(24, 1),
        fansub_width=1920,
        fansub_height=1080,
        fansub_total_frames=1000,
        debug_images=False,
    )


def test_stage_normalizes_and_keeps_clean_events(tmp_path: Path) -> None:
    _write_event_cleanup_jsonl(
        tmp_path,
        [
            (0, "Hello"),
            (1, "World"),
        ],
    )
    stage = NormalizeStage()
    result = stage.run(_make_globals(tmp_path), NormalizeConfig())
    assert [(e.event_id, e.cleaned_text) for e in result.events] == [
        (0, "Hello"),
        (1, "World"),
    ]


def test_stage_drops_noise_events(tmp_path: Path) -> None:
    _write_event_cleanup_jsonl(
        tmp_path,
        [
            (0, "1"),
            (1, "Hello"),
            (2, ":"),
            (3, "World"),
            (4, "-"),
        ],
    )
    stage = NormalizeStage()
    result = stage.run(_make_globals(tmp_path), NormalizeConfig())
    assert [(e.event_id, e.cleaned_text) for e in result.events] == [
        (1, "Hello"),
        (3, "World"),
    ]


def test_stage_applies_normalize_text(tmp_path: Path) -> None:
    _write_event_cleanup_jsonl(
        tmp_path,
        [
            (0, "  caf" + E_ACUTE + "   "),
            (1, "a" + ZWSP + "b"),
        ],
    )
    stage = NormalizeStage()
    result = stage.run(_make_globals(tmp_path), NormalizeConfig())
    assert [(e.event_id, e.cleaned_text) for e in result.events] == [
        (0, "caf" + E_ACUTE),
        (1, "ab"),
    ]


def test_stage_writes_output_file_atomic(tmp_path: Path) -> None:
    _write_event_cleanup_jsonl(tmp_path, [(0, "Hello")])
    stage = NormalizeStage()
    stage.run(_make_globals(tmp_path), NormalizeConfig())
    out = tmp_path / "11_normalize" / "normalized.json"
    meta = tmp_path / "11_normalize" / "normalized.meta.json"
    assert out.exists()
    assert meta.exists()
    # No .tmp leftover
    assert not (tmp_path / "11_normalize" / "normalized.json.tmp").exists()


def test_stage_cache_hit_skips_recomputation(tmp_path: Path) -> None:
    _write_event_cleanup_jsonl(tmp_path, [(0, "Hello")])
    stage = NormalizeStage()
    globals_ = _make_globals(tmp_path)
    config = NormalizeConfig()
    first = stage.run(globals_, config)
    out_path = tmp_path / "11_normalize" / "normalized.json"
    mtime1 = out_path.stat().st_mtime_ns
    # Second run should not rewrite the file (cache hit).
    second = stage.run(globals_, config)
    mtime2 = out_path.stat().st_mtime_ns
    assert first == second
    assert mtime1 == mtime2


def test_stage_cache_miss_when_input_changes(tmp_path: Path) -> None:
    _write_event_cleanup_jsonl(tmp_path, [(0, "Hello")])
    stage = NormalizeStage()
    globals_ = _make_globals(tmp_path)
    config = NormalizeConfig()
    stage.run(globals_, config)
    # Change the input -> next run must recompute.
    _write_event_cleanup_jsonl(tmp_path, [(0, "Goodbye")])
    second = stage.run(globals_, config)
    assert second.events[0].cleaned_text == "Goodbye"


def test_stage_handles_empty_input(tmp_path: Path) -> None:
    _write_event_cleanup_jsonl(tmp_path, [])
    stage = NormalizeStage()
    result = stage.run(_make_globals(tmp_path), NormalizeConfig())
    assert result.events == []


def test_stage_drops_event_that_becomes_noise_after_normalization(tmp_path: Path) -> None:
    # "   " becomes "" after normalize -> noise (len < 2).
    _write_event_cleanup_jsonl(
        tmp_path,
        [
            (0, "   "),
            (1, "Hello"),
        ],
    )
    stage = NormalizeStage()
    result = stage.run(_make_globals(tmp_path), NormalizeConfig())
    assert [(e.event_id, e.cleaned_text) for e in result.events] == [(1, "Hello")]

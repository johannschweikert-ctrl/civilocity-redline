"""Tests for the file classifier (Build Step 2).

Validates classifier output against:
  - The 15 development files in samples\\development\\, sourced
    canonically from RC_Benchmark_Manifest_v1.json
    (corpus_assignment == "development").
  - Synthetic fixtures (BFX, JPEG, blank PDF, malformed PDF) created
    in tmp_path for outcomes not represented in the dev set.

Manifest is authoritative. Disagreement = test FAIL, per S02
clarification (6). Skip-with-reason occurs only when samples or the
manifest are not present locally (e.g., a fresh checkout); never
when a comparison would otherwise have run.

The samples\\holdout\\ corpus is not referenced anywhere in this
module. test_classifier_source_does_not_reference_holdout guards
the classifier source file against the same.
"""

from __future__ import annotations

import json
from dataclasses import FrozenInstanceError
from pathlib import Path

import fitz
import pytest

from redline_checker import classifier
from redline_checker.classifier import (
    MSG_UNSUPPORTED_TYPE,
    NON_PDF_CLASSIFICATION,
    ClassificationResult,
    classify_file,
)

# --- Path resolution -----------------------------------------------
# tests/test_classifier.py  -> parents[0] = tests
#                              parents[1] = src
#                              parents[2] = civilocity_reviewer\
PROJECT_ROOT = Path(__file__).resolve().parents[2]
SAMPLES_DEV = PROJECT_ROOT / "samples" / "development"
MANIFEST_PATH = (
    PROJECT_ROOT / "audit" / "manifest" / "RC_Benchmark_Manifest_v1.json"
)


# --- Manifest-derived expected dev-set table -----------------------
# Sourced from RC_Benchmark_Manifest_v1.json. DEV-006 is the canonical
# pdf_shx_annotation_only example: pdfplot17.hdi flattens SHX glyphs to
# vector paths in the page content stream rather than producing PDF
# /Annot objects. Step 2 (which inspects only /Annot via page.annots())
# cannot detect this; SHX detection is the Step 6 CV pipeline's
# responsibility. Manifest stores ground truth; the per-file test below
# xfails DEV-006 with strict=True so the test will surface XPASSED
# when Step 6 begins correctly classifying it. See D-032 (initial
# decision) and D-033 (amendment / Option Y), both 2026-05-01.
EXPECTED_DEV_SET: dict[str, str] = {
    "0635-033 Chelan PUD HDD Redline.pdf": "pdf_text_only",
    "0906-51 Set 12-12-24_60percent w BV Comments.pdf": "pdf_annotated",
    "0906-51 Set 2-25-25 BV Red.pdf": "pdf_text_only",
    "0906-51 Set 2025-01-17 PRV Vault.pdf": "pdf_text_only",
    "0906_OLD_MILL_PARK_COMBINED_SET.pdf": "pdf_annotated",
    "17-0054 Morcos Civil 03-31-26_REV2.pdf": "pdf_shx_annotation_only",
    "17-0054 Morcos Civil 04-01-26_Util Plan.pdf": "pdf_annotated",
    "17-0054 Morcos Civil BV Markup.pdf": "pdf_mixed",
    "17-0054 Morcos Civil_REV2 BV Red.pdf": "pdf_text_only",
    "3.2.26 - 2605 - 26055 - 33137 Seneca Dr in Solon.pdf": "pdf_annotated",
    "Grading_Permit_Redlines (1).pdf": "pdf_mixed",
    "png_markup_1.png": "N/A (non-PDF)",
    "png_markup_2.png": "N/A (non-PDF)",
    "png_markup_3.png": "N/A (non-PDF)",
    "png_markup_4.png": "N/A (non-PDF)",
}

# File IDs whose Step 2 classifier output is known to diverge from the
# manifest's ground-truth pdf_classification. Each entry's value is the
# xfail reason. strict=True elsewhere ensures XPASSED becomes a failure
# when the underlying capability gap is closed (e.g. Step 6 lands and
# its output replaces or augments Step 2 for these files).
XFAIL_FILE_IDS: dict[str, str] = {
    "DEV-006": (
        "Step 2 inspects only PDF /Annot objects and cannot detect "
        "SHX-flattened-paths in the page content stream; ground-truth "
        "pdf_shx_annotation_only is reached by the Step 6 CV pipeline. "
        "See D-033 (2026-05-01)."
    ),
}


def _load_dev_files_from_manifest() -> list[dict]:
    if not MANIFEST_PATH.exists():
        return []
    text = MANIFEST_PATH.read_text(encoding="utf-8-sig")
    data = json.loads(text)
    return [f for f in data["files"] if f["corpus_assignment"] == "development"]


# Loaded at collection time so parametrize can use it.
DEV_ENTRIES: list[dict] = _load_dev_files_from_manifest()


def _build_dev_params() -> list:
    """Build pytest.param list, applying xfail marker to entries
    listed in XFAIL_FILE_IDS."""
    params = []
    for entry in DEV_ENTRIES:
        fid = entry["file_id"]
        if fid in XFAIL_FILE_IDS:
            params.append(
                pytest.param(
                    entry,
                    id=fid,
                    marks=pytest.mark.xfail(
                        reason=XFAIL_FILE_IDS[fid],
                        strict=True,
                    ),
                )
            )
        else:
            params.append(pytest.param(entry, id=fid))
    return params


# --- Manifest integrity --------------------------------------------


@pytest.mark.skipif(
    not MANIFEST_PATH.exists(),
    reason=f"benchmark manifest not present at {MANIFEST_PATH}",
)
def test_manifest_dev_set_matches_expected_table() -> None:
    """The manifest must contain exactly 15 development files matching
    the expected table. Drift here is caught before the parametrized
    classifier run, with a clearer error than per-file disagreement."""
    text = MANIFEST_PATH.read_text(encoding="utf-8-sig")
    data = json.loads(text)
    dev = [f for f in data["files"] if f["corpus_assignment"] == "development"]
    assert len(dev) == 15, f"expected 15 development files, got {len(dev)}"
    actual = {f["filename"]: f["pdf_classification"] for f in dev}
    assert actual == EXPECTED_DEV_SET, (
        "Manifest dev-set table drifted from expected. "
        "Update EXPECTED_DEV_SET in this test file or revert the manifest."
    )


# --- Per-file classifier validation against manifest ---------------


@pytest.mark.skipif(
    not SAMPLES_DEV.exists(),
    reason=f"samples\\development\\ not present at {SAMPLES_DEV}",
)
@pytest.mark.skipif(
    not DEV_ENTRIES,
    reason="benchmark manifest empty or missing — cannot parametrize",
)
@pytest.mark.parametrize("entry", _build_dev_params())
def test_dev_file_classification_matches_manifest(entry: dict) -> None:
    """Each dev file must classify exactly as the manifest says.

    Disagreement is a FAIL, not a skip — manifest is authoritative
    per S02 clarification (6) and D-033 (manifest = ground truth).
    Files in XFAIL_FILE_IDS are expected to diverge in Step 2 because
    of capability limitations recorded in the Decisions Log; the
    xfail markers carry strict=True so resolution surfaces as XPASSED.
    """
    path = SAMPLES_DEV / entry["filename"]
    assert path.exists(), f"sample missing on disk: {path}"

    result = classify_file(path)
    expected_class = entry["pdf_classification"]

    assert result.status == "classified", (
        f"{entry['file_id']} {entry['filename']}: expected classified, "
        f"got status={result.status!r}, error_message={result.error_message!r}"
    )
    assert result.error_message is None
    assert result.rejection_reason == ""

    if expected_class == NON_PDF_CLASSIFICATION:
        assert result.pdf_classification == NON_PDF_CLASSIFICATION
        ext = path.suffix.lower()
        if ext == ".png":
            assert result.file_type == "png"
        elif ext in (".jpg", ".jpeg"):
            assert result.file_type == "jpeg"
        else:
            pytest.fail(
                f"{entry['file_id']}: manifest says non-PDF but "
                f"extension is {ext!r}"
            )
    else:
        assert result.file_type == "pdf"
        assert result.pdf_classification == expected_class, (
            f"Manifest disagreement for {entry['file_id']} "
            f"({entry['filename']}): manifest={expected_class!r}, "
            f"classifier={result.pdf_classification!r}. Per S02 rules, "
            f"the manifest is authoritative; if this is the strict "
            f"pdf_mixed reading on DEV-008/DEV-011, STOP and propose "
            f"a new D-XXX entry — do not retune in place."
        )


# --- Synthetic fixtures: outcomes not represented in dev set --------


def test_jpeg_classification_synthetic(tmp_path: Path) -> None:
    """JPEG path is not exercised by the dev set; verify via fixture."""
    jpg = tmp_path / "tiny.jpg"
    jpg.write_bytes(b"\xff\xd8\xff\xd9")  # SOI + EOI
    result = classify_file(jpg)
    assert result.status == "classified"
    assert result.file_type == "jpeg"
    assert result.pdf_classification == NON_PDF_CLASSIFICATION
    assert result.error_message is None
    assert result.rejection_reason == ""


@pytest.mark.parametrize("ext", [".jpg", ".jpeg", ".JPG", ".JPEG"])
def test_jpeg_extension_variants(tmp_path: Path, ext: str) -> None:
    p = tmp_path / f"file{ext}"
    p.write_bytes(b"\xff\xd8\xff\xd9")
    assert classify_file(p).file_type == "jpeg"


def test_pdf_flattened_image_like_synthetic(tmp_path: Path) -> None:
    """A PDF page with no annotations and no extractable text classifies
    as pdf_flattened_image_like. The dev set contains no such file in
    its corpus_assignment == 'development' subset."""
    pdf = tmp_path / "blank_page.pdf"
    doc = fitz.open()
    doc.new_page(width=612, height=792)  # blank: no text, no annots
    doc.save(pdf)
    doc.close()

    result = classify_file(pdf)
    assert result.status == "classified"
    assert result.file_type == "pdf"
    assert result.pdf_classification == "pdf_flattened_image_like"


# --- Rejection paths -----------------------------------------------


def test_bfx_rejection(tmp_path: Path) -> None:
    bfx = tmp_path / "drawing.bfx"
    bfx.write_bytes(b"\x00\x00\x00\x00")
    result = classify_file(bfx)
    assert result.status == "rejected"
    assert result.file_type == ""
    assert result.pdf_classification is None
    assert result.rejection_reason == "bfx"
    assert result.error_message == MSG_UNSUPPORTED_TYPE


@pytest.mark.parametrize(
    "filename",
    [
        "drawing.dwg",
        "plot.dwf",
        "scan.tif",
        "doc.docx",
        "no_extension",
        "archive.zip",
    ],
)
def test_unsupported_extension_rejection(tmp_path: Path, filename: str) -> None:
    p = tmp_path / filename
    p.write_bytes(b"\x00\x00\x00\x00")
    result = classify_file(p)
    assert result.status == "rejected"
    assert result.file_type == ""
    assert result.pdf_classification is None
    assert result.rejection_reason == "unsupported_extension"
    assert result.error_message == MSG_UNSUPPORTED_TYPE


def test_unsupported_message_verbatim() -> None:
    """The mandated Section 2.3 string must remain exact. Edits to
    this constant require updating the Reference Document first."""
    expected = (
        "File type not supported. Supported formats: PDF, JPEG, PNG. "
        "Export to PDF and re-upload."
    )
    assert MSG_UNSUPPORTED_TYPE == expected


# --- Extension case-insensitivity ----------------------------------


def test_extension_case_insensitive_pdf(tmp_path: Path) -> None:
    pdf = tmp_path / "doc.PDF"
    doc = fitz.open()
    doc.new_page()
    doc.save(pdf)
    doc.close()
    result = classify_file(pdf)
    assert result.file_type == "pdf"
    assert result.status == "classified"


def test_extension_case_insensitive_bfx(tmp_path: Path) -> None:
    bfx = tmp_path / "drawing.BFX"
    bfx.write_bytes(b"\x00")
    assert classify_file(bfx).rejection_reason == "bfx"


def test_extension_case_insensitive_png(tmp_path: Path) -> None:
    png = tmp_path / "image.PNG"
    png.write_bytes(b"\x89PNG\r\n\x1a\n")
    assert classify_file(png).file_type == "png"


# --- Corruption propagation (S02 clarification 5) -------------------


def test_pdf_corruption_raises_not_silent(tmp_path: Path) -> None:
    """A non-PDF-magic file with .pdf extension must NOT be silently
    classified as one of the in-scope outcomes. The PyMuPDF error
    must propagate to the caller, who is responsible for handling
    the corruption case (deferred to Build Step 2b).

    Construction per S02 clarification (5): 4 bytes, .pdf extension,
    non-PDF magic bytes."""
    bogus = tmp_path / "garbage.pdf"
    bogus.write_bytes(b"\x00\x00\x00\x00")
    with pytest.raises(fitz.FileDataError):
        classify_file(bogus)


# --- Holdout protection --------------------------------------------


def test_classifier_source_does_not_reference_holdout() -> None:
    """Belt-and-suspenders against accidental future edits: the
    classifier module source file must not contain the string
    'holdout' in any case. The holdout corpus is sealed."""
    classifier_src = (
        Path(__file__).resolve().parent.parent
        / "redline_checker"
        / "classifier.py"
    )
    text = classifier_src.read_text(encoding="utf-8")
    assert "holdout" not in text.lower(), (
        "classifier.py contains a reference to 'holdout' — "
        "samples\\holdout\\ must not be referenced in source"
    )


# --- Dataclass shape -----------------------------------------------


def test_classification_result_is_frozen() -> None:
    """Result must be immutable so callers can't mutate it in place."""
    result = ClassificationResult(
        file_path=Path("x.pdf"),
        file_name="x.pdf",
        status="classified",
        file_type="pdf",
        pdf_classification="pdf_text_only",
        rejection_reason="",
        error_message=None,
    )
    with pytest.raises(FrozenInstanceError):
        result.file_type = "png"  # type: ignore[misc]


def test_module_exposes_expected_public_names() -> None:
    """Sanity check that the public surface is what tests and future
    modules import. Catches accidental rename/removal."""
    assert hasattr(classifier, "classify_file")
    assert hasattr(classifier, "ClassificationResult")
    assert hasattr(classifier, "MSG_UNSUPPORTED_TYPE")
    assert hasattr(classifier, "NON_PDF_CLASSIFICATION")

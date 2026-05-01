"""File classifier for Redline Checker — Build Step 2.

Implements the file-classifier per RC_Reference_Document_v1_FINAL.docx:
  - Section 2.1  (natively supported formats)
  - Section 2.3  (mandated user-facing error message strings)
  - Section 4.3  (pdf_classification field vocabulary)

S02 outcomes produced by this module:
  classified
    - pdf_annotated
    - pdf_text_only
    - pdf_flattened_image_like
    - pdf_mixed
    - jpeg  (pdf_classification = "N/A (non-PDF)")
    - png   (pdf_classification = "N/A (non-PDF)")
  rejected
    - bfx
    - unsupported_extension

pdf_shx_annotation_only is preserved as a controlled-vocabulary constant
for downstream consumers but is NOT produced by this classifier, per
D-032 Option 1: the SHX phenomenon in the dev set (DEV-006) is
SHX-glyphs-flattened-to-vector-paths in the page content stream, not
Square-subtype annotation noise. Detecting it requires render-time
visual analysis, which is the Step 6 CV pipeline's responsibility.
The Step 2 classifier is structural only.

The encrypted, corrupted, and zero-readable-content paths from
Section 2.3 are deferred to Build Step 2b. PyMuPDF exceptions on
malformed PDFs propagate from this module rather than being silently
classified — see test_pdf_corruption_raises_not_silent.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import fitz  # PyMuPDF

__all__ = [
    "ClassificationResult",
    "MSG_UNSUPPORTED_TYPE",
    "NON_PDF_CLASSIFICATION",
    "PDF_ANNOTATED",
    "PDF_FLATTENED_IMAGE_LIKE",
    "PDF_MIXED",
    "PDF_SHX_ANNOTATION_ONLY",
    "PDF_TEXT_ONLY",
    "classify_file",
]


# --- Mandated user-facing error message strings (Section 2.3) -----
# Verbatim. Do not edit without updating RC_Reference_Document_v1_FINAL.docx
# Section 2.3 first. test_unsupported_message_verbatim guards this string.
MSG_UNSUPPORTED_TYPE = (
    "File type not supported. Supported formats: PDF, JPEG, PNG. "
    "Export to PDF and re-upload."
)

# --- pdf_classification vocabulary (Section 4.3) ------------------
PDF_ANNOTATED = "pdf_annotated"
PDF_TEXT_ONLY = "pdf_text_only"
PDF_FLATTENED_IMAGE_LIKE = "pdf_flattened_image_like"
PDF_MIXED = "pdf_mixed"
# Vocabulary value preserved for downstream Step 6 use; not produced
# by this Step 2 classifier per D-032 Option 1.
PDF_SHX_ANNOTATION_ONLY = "pdf_shx_annotation_only"
NON_PDF_CLASSIFICATION = "N/A (non-PDF)"


@dataclass(frozen=True)
class ClassificationResult:
    """Result of classifying a single file.

    Field-value vocabulary verbatim from Reference Doc Section 4.3.
    """

    file_path: Path
    file_name: str
    status: Literal["classified", "rejected"]
    file_type: Literal["pdf", "jpeg", "png", ""]
    pdf_classification: str | None
    rejection_reason: Literal["bfx", "unsupported_extension", ""]
    error_message: str | None


def classify_file(path: Path) -> ClassificationResult:
    """Classify a single file by extension and (for PDFs) content.

    Parameters
    ----------
    path : Path
        Absolute or relative path to a single file.

    Returns
    -------
    ClassificationResult
        Frozen dataclass with the classification or rejection details.

    Raises
    ------
    Exception
        For PDFs that PyMuPDF cannot open (e.g. malformed magic bytes),
        the underlying fitz exception propagates. This is intentional:
        silent classification of broken PDFs as one of the in-scope
        outcomes would mask corruption that the encrypted / corrupted /
        zero-content paths (deferred to Build Step 2b) will handle
        explicitly.
    """
    path = Path(path)
    ext = path.suffix.lower()

    if ext == ".pdf":
        return _classify_pdf(path)
    if ext in (".jpg", ".jpeg"):
        return _ok(path, file_type="jpeg")
    if ext == ".png":
        return _ok(path, file_type="png")
    if ext == ".bfx":
        return _reject(path, reason="bfx")
    return _reject(path, reason="unsupported_extension")


# --- Helpers -------------------------------------------------------


def _ok(path: Path, *, file_type: Literal["jpeg", "png"]) -> ClassificationResult:
    """Build a classified result for a non-PDF supported file."""
    return ClassificationResult(
        file_path=path,
        file_name=path.name,
        status="classified",
        file_type=file_type,
        pdf_classification=NON_PDF_CLASSIFICATION,
        rejection_reason="",
        error_message=None,
    )


def _reject(
    path: Path,
    *,
    reason: Literal["bfx", "unsupported_extension"],
) -> ClassificationResult:
    """Build a rejected result with the mandated Section 2.3 message."""
    return ClassificationResult(
        file_path=path,
        file_name=path.name,
        status="rejected",
        file_type="",
        pdf_classification=None,
        rejection_reason=reason,
        error_message=MSG_UNSUPPORTED_TYPE,
    )


def _classify_pdf(path: Path) -> ClassificationResult:
    """Classify a PDF file by inspecting per-page annotations and text."""
    pdf_class = _classify_pdf_pages(path)
    return ClassificationResult(
        file_path=path,
        file_name=path.name,
        status="classified",
        file_type="pdf",
        pdf_classification=pdf_class,
        rejection_reason="",
        error_message=None,
    )


def _classify_pdf_pages(path: Path) -> str:
    """Open the PDF and return the file-level pdf_classification value.

    Per-page state:
      reviewer  — page has at least one annotation object.
      flattened — page has zero annotation objects.

    File-level decision (strict pdf_mixed per S02 clarification 3):
      - any reviewer + any flattened   -> pdf_mixed
      - any reviewer (no flattened)    -> pdf_annotated
      - all pages flattened:
          - any text on any page       -> pdf_text_only
          - else                       -> pdf_flattened_image_like
    """
    page_states: list[str] = []
    any_text = False

    with fitz.open(path) as doc:
        for page in doc:
            state = _classify_page(page)
            page_states.append(state)
            if state == "flattened" and page.get_text().strip():
                any_text = True

    has_reviewer = "reviewer" in page_states
    has_flattened = "flattened" in page_states

    if has_reviewer and has_flattened:
        return PDF_MIXED
    if has_reviewer:
        return PDF_ANNOTATED
    if any_text:
        return PDF_TEXT_ONLY
    return PDF_FLATTENED_IMAGE_LIKE


def _classify_page(page: fitz.Page) -> str:
    """Classify a single PDF page as 'reviewer' or 'flattened'.

    A page is 'reviewer' if it carries any annotation objects, regardless
    of subtype. SHX-pattern detection (previously a Square-subtype
    heuristic with a count threshold) was retired in D-032 — SHX
    artifacts in the dev set live in the page content stream, not in
    the annotation layer.
    """
    annots = list(page.annots() or [])
    return "reviewer" if annots else "flattened"

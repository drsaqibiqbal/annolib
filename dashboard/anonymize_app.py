"""prostate-deid — interactive anonymization dashboard.

Upload a prostate MRI DICOM, run the default de-identification pipeline, and
inspect exactly what changed: before/after header, before/after image, live
metrics, the full audit trail, and a capability comparison against related
tools. Download the de-identified .dcm at the end.

Run:
    pip install prostate-deid[dashboard]
    streamlit run dashboard/anonymize_app.py
"""

from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import streamlit as st

from prostate_deid import default_pipeline, summarize_audit
from prostate_deid.comparison import as_rows
from prostate_deid.dicom_io import load_sample, save_sample

st.set_page_config(
    page_title="prostate-deid — DICOM anonymizer",
    page_icon="🧬",
    layout="wide",
)

# ---------------------------------------------------------------------------
# Styling
# ---------------------------------------------------------------------------
st.markdown(
    """
    <style>
      .stApp { background: linear-gradient(180deg, #0b1020 0%, #0e1428 100%); }
      h1, h2, h3, h4 { color: #e8ecff; font-family: 'Segoe UI', system-ui, sans-serif; }
      .pd-hero {
        background: linear-gradient(120deg, #1b2a6b 0%, #2a1a5e 100%);
        border: 1px solid #33407e; border-radius: 18px;
        padding: 28px 32px; margin-bottom: 8px;
      }
      .pd-hero h1 { margin: 0; font-size: 2.0rem; letter-spacing: -0.5px; }
      .pd-hero p { color: #aab4e8; margin: 6px 0 0 0; font-size: 1.02rem; }
      .pd-metric {
        background: #131a33; border: 1px solid #263056; border-radius: 14px;
        padding: 16px 18px; text-align: center;
      }
      .pd-metric .v { font-size: 1.9rem; font-weight: 700; color: #6ee7b7; }
      .pd-metric .l { font-size: .78rem; color: #9aa4c8; text-transform: uppercase; letter-spacing: .06em; }
      .pd-flag .v { color: #fbbf24; }
      .pd-pill { display:inline-block; padding:2px 10px; border-radius:999px;
        font-size:.75rem; font-weight:600; }
      .pd-removed { background:#3b1220; color:#fca5a5; }
      .pd-kept { background:#0f2f22; color:#6ee7b7; }
      .pd-changed { background:#2a2410; color:#fcd34d; }
      .pd-note { color:#8b93b8; font-size:.85rem; }
      table td, table th { color:#dbe1ff !important; }
      .full { color:#6ee7b7; font-weight:700; }
      .partial { color:#fcd34d; font-weight:700; }
      .none { color:#6b7280; }
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    """
    <div class="pd-hero">
      <h1>🧬 prostate-deid</h1>
      <p>De-identify a prostate MRI DICOM in one pass — header scrubbing,
      consistent pseudonymization, UID regeneration, and best-effort burned-in
      text redaction — with a complete, auditable record of every change.</p>
    </div>
    """,
    unsafe_allow_html=True,
)

# ---------------------------------------------------------------------------
# Compliance banner (non-negotiable for a tool of this kind)
# ---------------------------------------------------------------------------
st.warning(
    "Running this pipeline does **not** by itself constitute HIPAA or GDPR "
    "compliance, and the pseudonymization mode retains a reversible mapping. "
    "Review the audit trail, and follow your institution's own determination "
    "process before releasing any data.",
    icon="⚠️",
)

# ---------------------------------------------------------------------------
# Sidebar controls
# ---------------------------------------------------------------------------
with st.sidebar:
    st.header("Pipeline options")
    profile = st.selectbox(
        "Tag-removal profile",
        ["hipaa_safe_harbor", "dicom_ps3.15_basic"],
        index=0,
    )
    do_ocr = st.checkbox(
        "Attempt burned-in text redaction (needs tesseract)",
        value=False,
        help="Scans pixel data with OCR for burned-in PHI. If tesseract isn't "
        "installed, the step is skipped and flagged for manual review rather "
        "than failing.",
    )
    st.caption(
        "Pseudonymization uses PROSTATE_DEID_KEY from the environment; set a "
        "real secret before production use."
    )

_PHI_KEYWORDS = {
    "PatientName", "PatientID", "PatientBirthDate", "PatientBirthTime",
    "ReferringPhysicianName", "PerformingPhysicianName", "OperatorsName",
    "InstitutionName", "InstitutionAddress", "StationName", "StudyID",
    "AccessionNumber", "OtherPatientIDs", "OtherPatientNames",
}


def _window(frame: np.ndarray) -> np.ndarray:
    frame = np.squeeze(frame).astype(np.float32)
    if frame.ndim > 2:
        frame = frame[frame.shape[0] // 2]
    lo, hi = np.percentile(frame, [1, 99]) if frame.size else (0, 1)
    if hi <= lo:
        hi = lo + 1
    return np.clip((frame - lo) / (hi - lo) * 255, 0, 255).astype(np.uint8)


uploaded = st.file_uploader("Upload a DICOM (.dcm) file", type=["dcm", "dicom"])

if uploaded is None:
    st.info("Upload a prostate MRI DICOM to begin. No file leaves this machine.")
    with st.expander("How prostate-deid compares to related tools", expanded=True):
        st.markdown("#### Capability comparison")
        st.caption(
            "Feature-support comparison based on each tool's public docs — "
            "NOT a quantitative accuracy benchmark. Re-verify before citing."
        )
        st.table(as_rows())
    st.stop()

# ---------------------------------------------------------------------------
# Process
# ---------------------------------------------------------------------------
tmp_dir = Path(st.session_state.setdefault("_tmp", "._pd_tmp"))
tmp_dir.mkdir(exist_ok=True)
in_path = tmp_dir / "input.dcm"
in_path.write_bytes(uploaded.getvalue())

sample = load_sample(in_path)
before_meta = dict(sample.dicom_meta or {})
before_img = _window(sample.image)

pipeline = default_pipeline(
    mapping_store=tmp_dir / "mapping.json",
    profile=profile,
    redact_burned_in_text=do_ocr,
)

t0 = time.perf_counter()
result = pipeline(sample)
elapsed_ms = (time.perf_counter() - t0) * 1000

out_path = tmp_dir / "deidentified.dcm"
save_sample(result, out_path)

after_meta = dict(result.dicom_meta or {})
after_img = _window(result.image)
summary = summarize_audit(result.audit)

# ---------------------------------------------------------------------------
# Metrics row
# ---------------------------------------------------------------------------
st.subheader("Result")
cols = st.columns(6)
metric_specs = [
    ("Tags removed", summary["tags_removed"] + summary["private_tags_removed"], ""),
    ("IDs pseudonymized", summary["ids_pseudonymized"], ""),
    ("Dates shifted", summary["dates_shifted"], ""),
    ("UIDs regenerated", summary["uids_regenerated"], ""),
    ("Pixel regions redacted", summary["pixel_regions_redacted"], ""),
    ("Flagged for review", summary["flagged_for_review"], "pd-flag"),
]
for col, (label, value, extra) in zip(cols, metric_specs):
    col.markdown(
        f'<div class="pd-metric {extra}"><div class="v">{value}</div>'
        f'<div class="l">{label}</div></div>',
        unsafe_allow_html=True,
    )
st.caption(f"Pipeline completed in {elapsed_ms:.0f} ms on this machine.")

tab_header, tab_image, tab_audit, tab_compare = st.tabs(
    ["Header before / after", "Image before / after", "Audit trail", "Comparison"]
)

with tab_header:
    all_keys = sorted(set(before_meta) | set(after_meta))
    rows = []
    for k in all_keys:
        b = before_meta.get(k, "—")
        a = after_meta.get(k, "—")
        if k not in after_meta:
            status = '<span class="pd-pill pd-removed">removed</span>'
        elif b != a:
            status = '<span class="pd-pill pd-changed">changed</span>'
        else:
            status = '<span class="pd-pill pd-kept">kept</span>'
        phi = "🔴" if k in _PHI_KEYWORDS or k.startswith("(") else ""
        rows.append(
            f"<tr><td>{phi} {k}</td><td>{b}</td><td>{a}</td><td>{status}</td></tr>"
        )
    st.markdown(
        "<table style='width:100%'><tr><th>Field</th><th>Before</th>"
        "<th>After</th><th>Status</th></tr>" + "".join(rows) + "</table>",
        unsafe_allow_html=True,
    )
    st.caption("🔴 marks direct-identifier fields and private tags.")

with tab_image:
    c1, c2 = st.columns(2)
    c1.markdown("**Before**")
    c1.image(before_img, use_container_width=True, clamp=True)
    c2.markdown("**After**")
    c2.image(after_img, use_container_width=True, clamp=True)
    if summary["pixel_regions_redacted"] == 0:
        st.caption(
            "No pixel regions were redacted "
            + ("(OCR disabled)." if not do_ocr else "(no burned-in text detected).")
        )

with tab_audit:
    audit_rows = [
        {
            "step": e.step,
            "action": e.action,
            "detail": e.detail,
            "confidence": "" if e.confidence is None else f"{e.confidence:.2f}",
            "review": "⚑" if e.flagged_for_review else "",
        }
        for e in result.audit
    ]
    st.dataframe(audit_rows, use_container_width=True, hide_index=True)

with tab_compare:
    st.caption(
        "Feature-support comparison from public docs — not a performance "
        "benchmark. Quantitative recall/precision requires a shared "
        "ground-truthed evaluation set (e.g. TCIA Pseudo-PHI / MIDI-B)."
    )
    st.table(as_rows())

# ---------------------------------------------------------------------------
# Download
# ---------------------------------------------------------------------------
st.download_button(
    "⬇️ Download de-identified DICOM",
    data=out_path.read_bytes(),
    file_name="deidentified.dcm",
    mime="application/dicom",
    type="primary",
)

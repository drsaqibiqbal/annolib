"""Reviewer QA dashboard.

Loads an audit bundle written by prostate_deid.export.export_review_bundle
and surfaces every flagged_for_review entry as a cropped thumbnail for quick
human triage. This exists because an audit trail nobody reviews creates
false confidence -- worse than not flagging at all.

Run with:
    streamlit run dashboard/review_app.py -- --bundle path/to/exported_bundle
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import streamlit as st


def _parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--bundle", type=str, default=None)
    args, _ = parser.parse_known_args()
    return args


def load_bundle(bundle_dir: Path):
    audit_path = bundle_dir / "audit.json"
    rows = json.loads(audit_path.read_text())
    return rows


def main():
    st.set_page_config(page_title="prostate-deid reviewer QA", layout="wide")
    st.title("prostate-deid — Reviewer QA")
    st.caption(
        "Triage every audit entry flagged_for_review before trusting a "
        "pipeline run's output. A flag with no reviewer is a dead end."
    )

    args = _parse_args()
    bundle_input = st.sidebar.text_input("Bundle directory", value=args.bundle or "")

    if not bundle_input:
        st.info("Enter the path to an exported review bundle in the sidebar.")
        return

    bundle_dir = Path(bundle_input)
    if not bundle_dir.exists():
        st.error(f"Path does not exist: {bundle_dir}")
        return

    rows = load_bundle(bundle_dir)
    flagged = [r for r in rows if r["flagged_for_review"]]
    total = len(rows)

    st.sidebar.metric("Total audit entries", total)
    st.sidebar.metric("Flagged for review", len(flagged))

    if not flagged:
        st.success("No entries flagged for review in this bundle.")
        st.subheader("Full audit trail")
        st.dataframe(rows, use_container_width=True)
        return

    st.subheader(f"{len(flagged)} entries need review")

    for row in flagged:
        with st.container(border=True):
            cols = st.columns([1, 2])
            with cols[0]:
                if row["thumbnail"]:
                    thumb_path = bundle_dir / row["thumbnail"]
                    if thumb_path.exists():
                        crop = np.load(thumb_path)
                        display = crop if crop.ndim == 2 else crop[0]
                        st.image(
                            np.clip(display, 0, 255).astype("uint8"),
                            caption=f"entry #{row['index']}",
                        )
                else:
                    st.write("(no cropped region available for this entry)")
            with cols[1]:
                st.write(f"**Step:** {row['step']}")
                st.write(f"**Action:** {row['action']}")
                st.write(f"**Detail:** {row['detail']}")
                if row["confidence"] is not None:
                    st.write(f"**Confidence:** {row['confidence']:.2f}")
                st.radio(
                    "Disposition",
                    ["Not yet reviewed", "Confirmed correct", "False positive / over-redacted"],
                    key=f"disposition_{row['index']}",
                    horizontal=True,
                )

    st.divider()
    st.subheader("Full audit trail")
    st.dataframe(rows, use_container_width=True)


if __name__ == "__main__":
    main()

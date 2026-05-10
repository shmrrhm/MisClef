"""app.py – Streamlit web front-end for MisClef.

Run with:
    streamlit run app.py
"""

import os
import tempfile

import streamlit as st

from MisClef import annotate_pdf

st.set_page_config(page_title="MisClef", page_icon="🎹", layout="centered")
st.title("🎹 MisClef – Piano Sheet Annotator")
st.write("Drag and drop a piano sheet music PDF to annotate it with note names.")

if "result_bytes" not in st.session_state:
    st.session_state.result_bytes = None
if "out_name" not in st.session_state:
    st.session_state.out_name = None

key_sig = st.number_input(
    "Key signature (sharps = positive, flats = negative)",
    value=0, min_value=-7, max_value=7, step=1,
)

uploaded = st.file_uploader("Upload PDF", type="pdf")

if uploaded:
    if st.button("Annotate"):
        st.session_state.result_bytes = None
        st.session_state.out_name = None
        with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp_in:
            tmp_in.write(uploaded.read())
            in_path = tmp_in.name
        out_path = in_path.replace(".pdf", "_annotated.pdf")

        progress_bar = st.progress(0, text="Starting…")
        status_text = st.empty()

        _STATUS_LABELS = {
            "rendering":        "Rendering page",
            "detecting staves": "Detecting staves",
            "running oemer":    "Running note-head detection (oemer)",
            "done":             "Annotated page",
        }

        def _on_progress(page_num, total_pages, status):
            # Progress advances to the *end* of each page only on 'done';
            # intermediate steps keep the bar at the start of that page.
            step_fraction = {"rendering": 0.0, "detecting staves": 0.25,
                             "running oemer": 0.5, "done": 1.0}.get(status, 0.0)
            value = min((page_num + step_fraction) / total_pages, 1.0)
            label = _STATUS_LABELS.get(status, status)
            progress_bar.progress(value, text=f"Page {page_num + 1} / {total_pages} — {label}…")
            status_text.markdown(
                f"**Page {page_num + 1} / {total_pages}** &nbsp;·&nbsp; {label}"
            )

        try:
            annotate_pdf(in_path, out_path, key_sig=int(key_sig),
                         progress_callback=_on_progress)
            with open(out_path, "rb") as f:
                st.session_state.result_bytes = f.read()
            st.session_state.out_name = uploaded.name.removesuffix(".pdf") + " - Annotated.pdf"
        finally:
            os.unlink(in_path)
            if os.path.exists(out_path):
                os.unlink(out_path)

        progress_bar.progress(1.0, text="Done!")
        status_text.empty()

if st.session_state.result_bytes is not None:
    st.success("Done!")
    st.download_button(
        label="⬇️ Download Annotated PDF",
        data=st.session_state.result_bytes,
        file_name=st.session_state.out_name,
        mime="application/pdf",
    )

if __name__ == "__main__":
    import subprocess
    import sys
    from streamlit.runtime.scriptrunner import get_script_run_ctx
    if get_script_run_ctx() is None:
        subprocess.run([sys.executable, "-m", "streamlit", "run", __file__])

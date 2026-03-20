# frontend/components/alignment_viewer.py
import streamlit as st


def render_alignment(ref_gapped: str, qry_gapped: str, width: int = 80):
    """Render base-pair alignment in monospace with color-coded mismatches."""
    if not ref_gapped or not qry_gapped:
        st.info("无比对数据")
        return

    lines = []
    idx = 0
    while idx < len(ref_gapped):
        ref_chunk = ref_gapped[idx:idx + width]
        qry_chunk = qry_gapped[idx:idx + width]

        mid = []
        for a, b in zip(ref_chunk, qry_chunk):
            if a == "-" or b == "-":
                mid.append(" ")
            elif a == b:
                mid.append("|")
            else:
                mid.append("*")
        mid_str = "".join(mid)

        lines.append(f"REF  {ref_chunk}")
        lines.append(f"     {mid_str}")
        lines.append(f"QRY  {qry_chunk}")
        lines.append("")
        idx += width

    st.code("\n".join(lines), language=None)

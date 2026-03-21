# frontend/components/alignment_viewer.py
import streamlit as st


def render_alignment(ref_gapped: str, qry_gapped: str, width: int = 80):
    """Render base-pair alignment with color-coded mismatches using HTML."""
    if not ref_gapped or not qry_gapped:
        st.info("无比对数据")
        return

    html_blocks = []
    idx = 0
    block_num = 0

    while idx < len(ref_gapped):
        ref_chunk = ref_gapped[idx:idx + width]
        qry_chunk = qry_gapped[idx:idx + width]
        block_num += 1

        # Build color-coded HTML for each position
        ref_html = []
        mid_html = []
        qry_html = []

        for i, (a, b) in enumerate(zip(ref_chunk, qry_chunk)):
            if a == "-":
                ref_html.append(f'<span class="aln-gap">{a}</span>')
                mid_html.append(' ')
                qry_html.append(f'<span class="aln-gap">{b}</span>')
            elif b == "-":
                ref_html.append(f'<span class="aln-gap">{a}</span>')
                mid_html.append(' ')
                qry_html.append(f'<span class="aln-gap">{b}</span>')
            elif a == b:
                ref_html.append(f'<span class="aln-match">{a}</span>')
                mid_html.append('<span class="aln-match">|</span>')
                qry_html.append(f'<span class="aln-match">{b}</span>')
            else:
                ref_html.append(f'<span class="aln-mismatch">{a}</span>')
                mid_html.append('<span class="aln-mismatch">*</span>')
                qry_html.append(f'<span class="aln-mismatch">{b}</span>')

        pos_start = idx + 1
        pos_end = min(idx + width, len(ref_gapped))
        pos_label = f'<span class="aln-pos">{pos_start:>5}-{pos_end:<5}</span> '

        html_blocks.append(
            f'{pos_label}<span class="aln-label">REF</span>  {"".join(ref_html)}\n'
            f'{"":>13}{"".join(mid_html)}\n'
            f'{" ":>12}<span class="aln-label">QRY</span>  {"".join(qry_html)}\n'
        )
        idx += width

    full_html = (
        '<div class="alignment-block">'
        + "\n".join(html_blocks)
        + '</div>'
    )

    st.markdown(full_html, unsafe_allow_html=True)

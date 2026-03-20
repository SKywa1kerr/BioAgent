# frontend/pages/1_analysis.py
import streamlit as st
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from backend.core.alignment import analyze_dataset
from backend.core.rules import judge_batch, load_thresholds
from backend.db.database import init_db, get_session_factory
from backend.db.models import Analysis, Sample, new_id

import json
from datetime import datetime, timezone

st.set_page_config(page_title="新建分析", page_icon="🔬", layout="wide")
st.title("新建分析")

init_db()

tab_scan, tab_upload = st.tabs(["扫描目录", "上传文件"])

with tab_scan:
    gb_dir = st.text_input("GenBank 目录路径", placeholder="例: D:/data/gb")
    ab1_dir = st.text_input("AB1 文件目录路径", placeholder="例: D:/data/ab1_files")
    analysis_name = st.text_input("分析名称（可选）", value="")

    if st.button("开始分析", type="primary", disabled=not (gb_dir and ab1_dir)):
        gb_path = Path(gb_dir)
        ab1_path = Path(ab1_dir)
        if not gb_path.exists():
            st.error(f"GenBank 目录不存在: {gb_dir}")
        elif not ab1_path.exists():
            st.error(f"AB1 目录不存在: {ab1_dir}")
        else:
            with st.spinner("正在分析..."):
                samples = analyze_dataset(gb_path, ab1_path)
                if not samples:
                    st.error("未发现可分析的样本")
                else:
                    thresholds = load_thresholds()
                    judgments = judge_batch(samples, thresholds)

                    # Save to database
                    Session = get_session_factory()
                    session = Session()
                    ok = sum(1 for j in judgments if j["status"] == "ok")
                    wrong = sum(1 for j in judgments if j["status"] == "wrong")
                    uncertain = sum(1 for j in judgments if j["status"] == "uncertain")

                    analysis = Analysis(
                        id=new_id(),
                        name=analysis_name or f"分析 {datetime.now().strftime('%m-%d %H:%M')}",
                        source_type="scan", source_path=ab1_dir,
                        status="done", total=len(samples),
                        ok_count=ok, wrong_count=wrong, uncertain_count=uncertain,
                        config_snapshot=json.dumps(thresholds),
                        finished_at=datetime.now(timezone.utc),
                    )
                    session.add(analysis)

                    for sd, jd in zip(samples, judgments):
                        session.add(Sample(
                            id=new_id(), analysis_id=analysis.id,
                            sid=sd["sid"], clone=sd.get("clone", ""),
                            status=jd["status"], reason=jd.get("reason", ""),
                            rule_id=jd.get("rule"),
                            identity=sd["identity"], cds_coverage=sd["cds_coverage"],
                            frameshift=sd["frameshift"],
                            aa_changes=json.dumps(sd.get("aa_changes", [])),
                            aa_changes_n=sd.get("aa_changes_n", 0),
                            raw_aa_changes_n=sd.get("raw_aa_changes_n", 0),
                            orientation=sd.get("orientation", ""),
                            seq_length=sd.get("seq_length", 0),
                            ref_length=sd.get("ref_length", 0),
                            avg_quality=sd.get("avg_qry_quality"),
                            sub_count=sd.get("sub", 0), ins_count=sd.get("ins", 0),
                            del_count=sd.get("del", 0),
                            ref_gapped=sd.get("ref_gapped", ""),
                            qry_gapped=sd.get("qry_gapped", ""),
                            quality_scores=json.dumps(sd.get("quality_scores", []) or []),
                            raw_data=json.dumps(sd, default=str),
                        ))
                    session.commit()
                    session.close()

                    st.success(f"分析完成: {len(samples)} 个样本 ({ok} ok / {wrong} wrong / {uncertain} uncertain)")
                    st.session_state["last_analysis_id"] = analysis.id

with tab_upload:
    uploaded = st.file_uploader(
        "上传 AB1 和 GenBank 文件",
        type=["ab1", "gb", "gbk"],
        accept_multiple_files=True,
    )
    if uploaded:
        st.info(f"已选择 {len(uploaded)} 个文件。上传功能将在扫描目录模式验证通过后完善。")

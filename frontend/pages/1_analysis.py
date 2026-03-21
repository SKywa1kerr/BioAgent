# frontend/pages/1_analysis.py
import streamlit as st
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from backend.core.alignment import analyze_dataset
from backend.core.rules import judge_batch, load_thresholds
from backend.db.database import init_db, get_session_factory
from backend.db.models import Analysis, Sample, new_id
from frontend.components.styles import inject_global_css, render_header, render_metric_cards

import json
from datetime import datetime, timezone

st.set_page_config(page_title="新建分析", page_icon="🔬", layout="wide")

inject_global_css()
render_header("🔬 新建分析", "扫描目录或上传文件，启动 QC 分析流程")

init_db()

tab_scan, tab_upload = st.tabs(["📂 扫描目录", "📤 上传文件"])

with tab_scan:
    st.markdown("")
    with st.container(border=True):
        st.markdown("##### 输入分析路径")
        col1, col2 = st.columns(2)
        gb_dir = col1.text_input(
            "GenBank 目录路径",
            placeholder="例: D:/data/gb",
            help="包含参考序列 .gb 文件的目录",
        )
        ab1_dir = col2.text_input(
            "AB1 文件目录路径",
            placeholder="例: D:/data/ab1_files",
            help="包含测序 .ab1 文件的目录",
        )
        analysis_name = st.text_input(
            "分析名称（可选）",
            value="",
            placeholder="留空将自动生成名称",
        )

        st.markdown("")
        if st.button("🚀 开始分析", type="primary", disabled=not (gb_dir and ab1_dir), use_container_width=True):
            gb_path = Path(gb_dir)
            ab1_path = Path(ab1_dir)
            if not gb_path.exists():
                st.error(f"GenBank 目录不存在: {gb_dir}")
            elif not ab1_path.exists():
                st.error(f"AB1 目录不存在: {ab1_dir}")
            else:
                with st.spinner("正在分析，请稍候..."):
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

                        st.session_state["last_analysis_id"] = analysis.id

                        st.markdown("")
                        st.success(f"分析完成！共 {len(samples)} 个样本")
                        render_metric_cards(len(samples), ok, wrong, uncertain)
                        st.markdown("")
                        st.info("💡 前往「分析结果」页面查看详细报告")

with tab_upload:
    st.markdown("")
    with st.container(border=True):
        st.markdown("##### 上传测序文件")
        st.caption("同时上传 .ab1 和 .gb/.gbk 文件，系统会自动分类并分析")
        uploaded = st.file_uploader(
            "选择 AB1 和 GenBank 文件",
            type=["ab1", "gb", "gbk"],
            accept_multiple_files=True,
        )

        upload_name = st.text_input("分析名称（可选）", value="", placeholder="留空将自动生成", key="upload_name")

        if uploaded:
            ab1_files = [f for f in uploaded if f.name.lower().endswith(".ab1")]
            gb_files = [f for f in uploaded if f.name.lower().endswith((".gb", ".gbk"))]
            st.markdown(f"已选择 **{len(ab1_files)}** 个 AB1 文件，**{len(gb_files)}** 个 GenBank 文件")

            if not ab1_files:
                st.warning("请至少上传一个 .ab1 文件")
            elif not gb_files:
                st.warning("请至少上传一个 .gb/.gbk 文件")
            elif st.button("🚀 上传并分析", type="primary", use_container_width=True):
              try:
                # Save files to data/uploads/<timestamp>/
                import time
                upload_id = time.strftime("%Y%m%d_%H%M%S")
                upload_base = Path(__file__).parent.parent.parent / "data" / "uploads" / upload_id
                upload_gb = upload_base / "gb"
                upload_ab1 = upload_base / "ab1"
                upload_gb.mkdir(parents=True, exist_ok=True)
                upload_ab1.mkdir(parents=True, exist_ok=True)

                for f in gb_files:
                    (upload_gb / f.name).write_bytes(f.read())
                for f in ab1_files:
                    (upload_ab1 / f.name).write_bytes(f.read())

                st.info(f"文件已保存到 data/uploads/{upload_id}/")

                with st.spinner("正在分析，请稍候..."):
                    samples = analyze_dataset(upload_gb, upload_ab1)
                    if not samples:
                        st.error("未发现可分析的样本，请检查文件是否匹配")
                    else:
                        thresholds = load_thresholds()
                        judgments = judge_batch(samples, thresholds)

                        Session = get_session_factory()
                        session = Session()
                        ok = sum(1 for j in judgments if j["status"] == "ok")
                        wrong = sum(1 for j in judgments if j["status"] == "wrong")
                        uncertain = sum(1 for j in judgments if j["status"] == "uncertain")

                        analysis = Analysis(
                            id=new_id(),
                            name=upload_name or f"上传分析 {datetime.now().strftime('%m-%d %H:%M')}",
                            source_type="upload", source_path=str(upload_base),
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

                        st.session_state["last_analysis_id"] = analysis.id
                        st.markdown("")
                        st.success(f"分析完成！共 {len(samples)} 个样本")
                        render_metric_cards(len(samples), ok, wrong, uncertain)
                        st.markdown("")
                        st.info("💡 前往「分析结果」页面查看详细报告")
              except Exception as e:
                st.error(f"文件上传或分析过程出错: {e}\n\n请检查上传的文件格式是否正确，或联系管理员。")
        else:
            st.markdown(
                "<div style='text-align:center; padding:2rem; opacity:0.5;'>"
                "将文件拖拽到此处，或点击上方按钮选择文件"
                "</div>",
                unsafe_allow_html=True,
            )

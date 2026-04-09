import sys
from pathlib import Path


sys.path.insert(0, str(Path(__file__).parent.parent / "src-python"))

from bioagent.llm_client import normalize_llm_base_url


def test_normalize_llm_base_url_strips_chat_completions_suffix():
    assert (
        normalize_llm_base_url("https://models.sjtu.edu.cn/api/v1/chat/completions")
        == "https://models.sjtu.edu.cn/api/v1"
    )


def test_normalize_llm_base_url_keeps_api_prefix():
    assert normalize_llm_base_url("https://models.sjtu.edu.cn/api/v1") == "https://models.sjtu.edu.cn/api/v1"

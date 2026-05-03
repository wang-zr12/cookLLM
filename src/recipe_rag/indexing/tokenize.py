from __future__ import annotations

import re

try:
    import jieba
except Exception:  # pragma: no cover
    jieba = None


def tokenize_zh(text: str) -> list[str]:
    text = text.lower().strip()
    if jieba is not None:
        tokens = [x.strip() for x in jieba.lcut(text) if x.strip()]
    else:
        tokens = re.findall(r"[\w]+|[\u4e00-\u9fff]", text)
    return [x for x in tokens if not x.isspace()]

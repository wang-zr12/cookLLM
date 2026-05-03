from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Iterable

from recipe_rag.schemas import Recipe
from recipe_rag.utils.io import read_jsonl


class BaseRecipeCrawler(ABC):
    """采集器接口。

    真实 App/网站采集建议在子类中处理鉴权、限流、重试和 robots/服务条款合规。
    输出统一收敛到 Recipe，后续链路不依赖来源。
    """

    @abstractmethod
    def crawl(self) -> Iterable[Recipe]:
        raise NotImplementedError


class JsonlRecipeCrawler(BaseRecipeCrawler):
    """从已落盘 JSONL 读取，便于把爬虫和 RAG 管道解耦。"""

    def __init__(self, path: str | Path):
        self.path = Path(path)

    def crawl(self) -> Iterable[Recipe]:
        for row in read_jsonl(self.path):
            yield Recipe.from_dict(row)

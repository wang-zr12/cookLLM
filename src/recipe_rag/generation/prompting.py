from __future__ import annotations

from recipe_rag.schemas import QueryUnderstanding, SearchResult


SYSTEM_PROMPT = """你是专业、严谨的中文菜谱助手。回答必须基于给定 context。
如果 context 不足以回答，要说明缺少的信息，并给出安全的替代建议。
回答优先包含：适用场景、食材/用量、步骤、技巧和注意事项。"""


def build_context(results: list[SearchResult]) -> str:
    blocks = []
    for item in results:
        blocks.append(
            f"[chunk_id={item.chunk.chunk_id} recipe_id={item.chunk.recipe_id} type={item.chunk.chunk_type.value} score={item.score:.4f}]\n"
            f"{item.chunk.text}"
        )
    return "\n\n".join(blocks)


def build_llama3_chat_prompt(query: QueryUnderstanding, results: list[SearchResult]) -> str:
    context = build_context(results)
    user = f"""用户问题：{query.raw_query}
识别意图：{query.intent.value}
识别食材：{'、'.join(query.ingredients) if query.ingredients else '无'}
约束条件：{'、'.join(query.constraints) if query.constraints else '无'}

context:
{context}

请用中文给出准确、可执行的回答。"""
    return (
        "<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n"
        + SYSTEM_PROMPT
        + "<|eot_id|><|start_header_id|>user<|end_header_id|>\n\n"
        + user
        + "<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"
    )

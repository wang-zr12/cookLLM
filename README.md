# Recipe RAG Assistant

**Project Time:** April 2024 – June 2024

这是一个菜谱领域的完整代码骨架，明确区分两条训练线：

- **LLaMA 微调**：发生在 RAG 之前，用菜谱问答/指令数据把生成模型从通用回复能力适配到菜谱助手表达。
- **BCE Reranker 微调**：发生在 RAG 训练优化阶段，用 query-passage 相关性数据把 BCE Reranker 从通用相关性判断适配到菜谱领域精准排序。

## 架构

```text
src/recipe_rag/
  schemas.py                  # Recipe / Chunk / QueryUnderstanding 等核心数据结构
  ingest/                     # 数据采集接口与 JSONL 采集器
  preprocess/                 # 清洗、食材归一化、语义分块
  indexing/                   # BCE embedding 适配、FAISS/NumPy 向量库、BM25
  retrieval/                  # Query 理解、双路召回、RRF、硬过滤、rerank
  generation/                 # LLaMA chat prompt 组装与 vLLM 客户端
  training/
    reranker/                 # BCE Reranker 数据构造、负样本挖掘、训练、评测
    llama_sft/                # LLaMA SFT 数据构造与 LoRA 训练
  cli/                        # 建库、查询、数据构造、评测命令
```

## 快速开始

安装轻量依赖：

```bash
pip install -e .
```

安装完整检索与训练依赖：

```bash
pip install -e ".[retrieval,train,serve,dev]"
```

准备 JSONL 菜谱数据，每行一个对象：

```json
{"recipe_id":"r1","title":"番茄炒蛋","ingredients":[{"name":"西红柿","amount":"2个"},{"name":"鸡蛋","amount":"3个"}],"steps":["鸡蛋打散。","番茄切块。","先炒蛋，再炒番茄，合炒调味。"],"tags":["家常","快手"],"tips":["番茄出汁后再放鸡蛋。"]}
```

离线建库：

```bash
recipe-build-index --input data/raw/recipes.jsonl --output artifacts/index --alias data/aliases.json
```

在线查询：

```bash
recipe-query --index artifacts/index --query "鸡胸肉低脂快手做法"
```

## 训练线 1：LLaMA SFT

SFT 目标是让生成模型更像菜谱助手，学习回答结构、语气、注意事项和步骤表达。它不负责检索排序。

```bash
python -m recipe_rag.training.llama_sft.prepare_sft_data \
  --recipes data/clean/recipes.jsonl \
  --output data/train/llama_sft.jsonl

python -m recipe_rag.training.llama_sft.train_lora \
  --model meta-llama/Meta-Llama-3-8B-Instruct \
  --train data/train/llama_sft.jsonl \
  --output artifacts/llama-recipe-lora
```

## 训练线 2：BCE Reranker 微调

Reranker 目标是判断 `query` 和 `passage/chunk` 在菜谱领域是否精准相关。

```bash
recipe-make-reranker-data \
  --queries data/train/generated_queries.jsonl \
  --chunks artifacts/index/chunks.jsonl \
  --output data/train/reranker_pairs.jsonl

python -m recipe_rag.training.reranker.train_bce_reranker \
  --model maidalun1020/bce-reranker-base_v1 \
  --train data/train/reranker_pairs.jsonl \
  --output artifacts/bce-reranker-recipe
```

评测：

```bash
recipe-evaluate --qrels data/eval/qrels.jsonl --run data/eval/run.jsonl --k 3
```

## 说明

代码默认提供轻量 fallback，方便在没有 FAISS、Transformers 或 GPU 的机器上验证流程。生产部署时建议安装 `retrieval` 和 `train` extras，并在配置中指定真实 BCE embedding、BCE reranker 与 vLLM 服务。

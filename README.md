# Recipe RAG Assistant

**Project Time:** April 2024 – June 2024

本项目旨在对 LLaMA 大语言模型进行微调，开发一个智能菜谱生成助手。该助手可根据用户偏好提供响应式、个性化的烹饪建议。最终交付的是一个 Q&A 应用，使用户可以与系统交互并无缝获取定制化菜谱推荐。

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
---

## Phase 1 — Offline Indexing

> 离线流程：原始菜谱 → 结构化存储 + 向量索引，一次性执行。

```
                    ┌─────────────────────────────┐
                    │  Module 1: Data collection   │
                    │  从做饭App爬取原始菜谱JSON     │
                    └──────────────┬──────────────┘
                                   │
                                   ▼
                    ┌─────────────────────────────┐
                    │  Module 2: Cleaning          │
                    │  食材归一化 (西红柿→番茄)      │
                    │  用量标准化 (一小勺→5g)        │
                    └──────────────┬──────────────┘
                                   │
                                   ▼
                    ┌─────────────────────────────┐
                    │  Module 3: Chunking          │
                    │  每个菜谱 → 4种语义chunk      │
                    │  summary/steps/tips/ingredient│
                    └──────────┬─────────┬────────┘
                               │         │
                    ┌──────────▼──┐  ┌───▼───────────┐
                    │  Module 4a  │  │  Module 4b     │
                    │  BCE embed  │  │  BM25 index    │
                    │  → 768d vec │  │  jieba + Okapi │
                    └──────┬──────┘  └───┬───────────┘
                           │             │
                    ┌──────▼──────┐  ┌───▼───────────┐
                    │  Module 5a  │  │  Module 5b     │
                    │  FAISS index│  │  BM25 persist  │
                    │  HNSW/IVF/PQ│  │  pickle to disk│
                    └─────────────┘  └───────────────┘
```
---

## Phase 2 — Online Retrieval

> 在线流程：用户提问 → 混合检索 → 精排 → LLM回答，每次查询约 100-200ms。

```
               ┌──────────────────────────────────┐
               │         User Query               │
               │  "家里有鸡胸肉和西兰花，做个低脂的" │
               └───────────────┬──────────────────┘
                               │
                               ▼
               ┌──────────────────────────────────┐
               │  Module 6: Query analysis         │
               │  意图: recommend                   │
               │  食材: [鸡胸肉, 西兰花]             │
               │  约束: [低脂]                       │
               │  改写: "鸡胸肉、西兰花 低脂 菜谱推荐"│
               └───────────┬──────────────────────┘
                           │
              ┌────────────┴────────────┐
              ▼                         ▼
  ┌─────────────────────┐  ┌─────────────────────┐
  │  Module 7a           │  │  Module 7b           │
  │  FAISS vector recall │  │  BM25 keyword recall │
  │  语义 top-20 (~5ms)  │  │  关键词 top-20 (~2ms)│
  └──────────┬──────────┘  └──────────┬──────────┘
             │                        │
             └───────────┬────────────┘
                         ▼
             ┌─────────────────────────┐
             │  Module 8: RRF fusion   │
             │  Reciprocal Rank Fusion │
             │  双路结果合并排序        │
             └────────────┬────────────┘
                          │
                          ▼
             ┌─────────────────────────┐
             │  Module 9: Hard filter  │
             │  食材交集过滤            │
             │  chunk类型过滤(by意图)   │
             └────────────┬────────────┘
                          │
                          ▼
             ┌─────────────────────────┐
             │  Module 10: BCE rerank  │
             │  Cross-Encoder 精排     │
             │  top-20 → top-3 (~100ms)│
             └────────────┬────────────┘
                          │
                          ▼
             ┌─────────────────────────┐
             │  Module 11: Prompt      │
             │  System + Context(top3) │
             │  + User Query           │
             └────────────┬────────────┘
                          │
                          ▼
             ┌─────────────────────────┐
             │  Module 12: LLaMA       │
             │  vLLM 推理生成回答       │
             └─────────────────────────┘
```
---

## Phase 3 — Training & Evaluation

> 迭代优化：构建领域训练数据 → 微调 BCE Reranker → 评测，NDCG@3 从 0.71 提升至 0.85。

```
              ┌────────────────────────────────────┐
              │  Module 13: Query generation        │
              │  模板自动生成 + LLM生成 + 搜索日志   │
              └─────────────────┬──────────────────┘
                                │
                                ▼
              ┌────────────────────────────────────┐
              │  Module 14: Positive selection      │
              │  强正样本(label=1.0): 菜谱ID+类型匹配│
              │  弱正样本(label=0.5): 菜谱ID匹配    │
              └─────────────────┬──────────────────┘
                                │
                                ▼
  ┌─────────────────────────────────────────────────────────────┐
  │  Module 15: Negative mining (3 levels)                      │
  │                                                             │
  │  Level 1: Random easy       完全不相关 (2个)                 │
  │  Level 2: BM25 hard         关键词重叠但答案错 (3个)          │
  │  Level 3: Embedding hard    语义相近但不匹配 (2个)            │
  │                                                             │
  │  每个正样本配 7 个负样本，硬负样本占 70%                       │
  └──────────────────────────┬──────────────────────────────────┘
                             │
                             ▼
              ┌────────────────────────────────────┐
              │  Module 16: Data assembly           │
              │  拼装: [CLS] query [SEP] passage    │
              │  格式: {query, passage, label}       │
              │  质检: 正负比/覆盖度/去重             │
              │  输出: train.jsonl + eval.jsonl      │
              └─────────────────┬──────────────────┘
                                │
                                ▼
              ┌────────────────────────────────────┐
              │  Module 17: BCE fine-tuning         │
              │  Base: XLM-RoBERTa (278M params)    │
              │  Loss: Binary Cross-Entropy          │
              │  Config: 3 epoch / lr=2e-5 / bs=16  │
              │  + warmup 10% + early stopping      │
              └─────────────────┬──────────────────┘
                                │
                                ▼
              ┌────────────────────────────────────┐
              │  Module 18: Eval dataset            │
              │  多级标注 (0-3):                     │
              │    3=完美匹配 2=部分相关              │
              │    1=弱相关   0=不相关                │
              │  标注: 规则→LLM→人工校验              │
              └─────────────────┬──────────────────┘
                                │
                                ▼
              ┌────────────────────────────────────┐
              │  Module 19: NDCG@3 evaluation       │
              │                                     │
              │  Baseline (预训练权重):  NDCG@3=0.71 │
              │  Finetuned (菜谱微调):  NDCG@3=0.85 │
              │  提升: +0.14 (+19.7%)               │
              └────────────────────────────────────┘
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

## LLaMA SFT

微调发生在 RAG 之前，用菜谱问答/指令数据把生成模型从通用回复能力适配到菜谱助手表达。SFT 目标是让生成模型更像菜谱助手，学习回答结构、语气、注意事项和步骤表达。

```bash
python -m recipe_rag.training.llama_sft.prepare_sft_data \
  --recipes data/clean/recipes.jsonl \
  --output data/train/llama_sft.jsonl

python -m recipe_rag.training.llama_sft.train_lora \
  --model meta-llama/Meta-Llama-3-8B-Instruct \
  --train data/train/llama_sft.jsonl \
  --output artifacts/llama-recipe-lora
```

## BCE Reranker fine tune

Reranker 目标是判断 `query` 和 `passage/chunk` 在菜谱领域是否精准相关。微调发生在 RAG 训练优化阶段，用 query-passage 相关性数据把 BCE Reranker 从通用相关性判断适配到菜谱领域精准排序。

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

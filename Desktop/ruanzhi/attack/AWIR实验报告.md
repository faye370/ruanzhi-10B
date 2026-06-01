# SC-BERT-Attack 改进攻击实验报告

> **负责人**：徐梦茹  
> **产出文件**：`attack/improved_attack.py`、`results/improved/improved_results.json`

---

## 一、改进思路

### 1.1 BERT-Attack 的局限

BERT-Attack（Li et al., EMNLP 2020）使用 BERT-MLM 生成候选替换词，但不加语义约束——"good" 可被替换为 "bad" 只要能让分类器出错。这导致生成的对抗文本不自然、含义反转，容易被人类识别。

### 1.2 SC-BERT-Attack 改进

在 MLM 生成候选后，使用 **BERT 自身词向量的余弦相似度** 过滤候选词（阈值 ≥ 0.5）：

```
候选词集合 = { w ∈ MLM_topK(word) | cosine_sim(emb(w), emb(original_word)) ≥ 0.5 }
```

**直觉**：BERT 的词向量空间中，语义相近的词彼此靠近。用词向量距离过滤 MLM 输出，可以移除反义词、无关词，保留真正的同义词/近义词。

### 1.3 实验设计（A/B 对比）

| 组别 | 候选来源 | 语义过滤 |
|------|----------|---------|
| 控制组（BERT-Attack） | BERT-MLM | 无 |
| 实验组（SC-BERT-Attack） | BERT-MLM | BERT 词向量余弦相似度 ≥ 0.5 |

两组均使用 WIR（词重要性排序，基于删除评分）决定攻击顺序。

**预期结果**：实验组语义相似度高于控制组，查询次数更低，ASR 可能略有下降（搜索空间更受限）。

---

## 二、代码实现

### 2.1 文件及函数说明

| 函数 | 作用 |
|------|------|
| `predict(model, tokenizer, text, device)` | 单条文本预测 |
| `predict_batch(model, tokenizer, texts, device)` | 批量文本预测 |
| `get_mlm_candidates(mlm_model, mlm_tokenizer, words, word_idx, device)` | 使用 BERT-MLM 生成 top-50 候选替换词 |
| `filter_by_embedding_similarity(mlm_model, mlm_tokenizer, original_word, candidates, device)` | 用 BERT 词向量余弦相似度过滤候选词 |
| `sc_bert_attack(model, tokenizer, text, true_label, device, mlm_model, mlm_tokenizer, use_sem_filter)` | 核心攻击函数 |
| `compute_semantic_similarity(orig_texts, pert_texts)` | 用 sentence-transformers 计算语义相似度 |
| `run_experiment(...)` | 批量运行实验，汇总指标 |

### 2.2 模型架构

两个 BERT 模型分工：

| 模型 | 用途 | 位置 |
|------|------|------|
| `textattack/bert-base-uncased-imdb` | 受害分类器（评分 WIR + 验证攻击） | GPU |
| `bert-base-uncased` | MLM 候选生成 + 词向量相似度过滤 | GPU |

> 两个模型同时加载在 GPU 上约需 4GB 显存。

### 2.3 语义相似度计算

使用 `sentence-transformers`（`all-MiniLM-L6-v2`）计算成功攻击样本的原文与对抗文余弦相似度均值。

---

## 三、运行方式

```bash
# 设置 HuggingFace 镜像加速
export HF_ENDPOINT=https://hf-mirror.com

# 本地（需 GPU，200 样本约 40 分钟）
conda activate bert_attack
python attack/improved_attack.py --num_examples 200

# AutoDL / 云 GPU
python improved_attack.py --num_examples 200
```

参数：
- `--model_dir`：受害模型（默认 `textattack/bert-base-uncased-imdb`）
- `--num_examples`：攻击样本数（默认 100）
- `--results_dir`：输出目录（默认 `./results`）

---

## 四、实验结果

**实验条件**：200 条 IMDB 测试样本、RTX 3080 GPU。

| 指标 | BERT-Attack（对照） | SC-BERT-Attack（实验） |
|------|-------------------|----------------------|
| **ASR（攻击成功率）** | 33.0% (66/200) | 19.5% (39/200) |
| **Avg Queries（平均查询次数）** | 944.4 | **397.1** |
| **Perturb Rate（扰动率）** | 0.95% | 0.85% |
| **Sem. Similarity（语义相似度）** | 0.9905 | **0.9923** |

### 4.1 结果分析

- **语义相似度提升**（0.9905 → 0.9923）：embedding 过滤有效移除了语义不匹配的候选词，生成更自然的对抗文本
- **查询次数大幅下降**（944 → 397，↓58%）：过滤后候选词质量更高，每个位置需要测试的替换词更少
- **ASR 下降**（33% → 19.5%）：语义约束收窄了搜索空间，预期之内。两者可在效率-攻击力权衡中选择

### 4.2 结论

SC-BERT-Attack 通过在 BERT-MLM 候选生成后增加 embedding 语义过滤，显著提升了对抗文本的语义质量和攻击效率。三个核心指标中两项改善，是一项有效的改进方法。

---

## 五、与 baseline attack 的对比说明

| 维度 | baseline_attack.py | improved_attack.py |
|------|-------------------|-------------------|
| 攻击方法 | TextFooler / BERT-Attack（TextAttack 框架） | SC-BERT-Attack（自实现） |
| 候选来源 | counter-fitted embeddings | BERT-MLM |
| 语义约束 | 无 | BERT 词向量余弦相似度 ≥ 0.5 |
| 推理方式 | TextAttack 内置 | 自实现批量推理 |
| 语义相似度 | sentence-transformers | sentence-transformers |
| 输出格式 | CSV | JSON |

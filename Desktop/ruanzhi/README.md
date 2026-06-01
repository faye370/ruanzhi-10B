# 基于 BERT 的对抗攻击与鲁棒性分析

**课程项目 · 5人分工 · Python + PyTorch**


## 目录结构

```
ruanzhi-10B/
├── configs/config.py                    # 全局参数（模型名、路径、超参数）
├── data/download_data.py                # 预下载数据集与 NLTK 资源
├── train/finetune_bert.py               # BERT 微调流程（参考用，攻击不依赖此步）
├── attack/
│   ├── baseline_attack.py               # TextFooler / BERT-Attack（可单独运行）
│   └── improved_attack.py               # 改进A：SC-BERT-Attack 语义约束攻击
├── defense/adversarial_training.py      # 改进B：对抗训练防御
├── evaluate/
│   ├── evaluate.py                      # 汇总所有结果，输出对比表
│   └── visualize.py                     # 生成图表
├── results/                             # 实验输出（自动创建）
├── checkpoints/                         # 模型权重（自动创建）
├── install.py                           # 一键安装脚本（自动判断 GPU/CPU）
├── requirements.txt
└── main.py                              # 环境验证
```


## 一、环境搭建（每位成员都要在自己电脑上做，只做一次）

### 前提条件

- Python 3.8 ~ 3.10（推荐 3.9）

### 步骤

**第1步：克隆代码**

```bash
git clone https://github.com/faye370/ruanzhi-10B.git
cd ruanzhi-10B
```

**第2步：安装依赖**

```bash
python install.py
```

> `install.py` 会自动检测有无 NVIDIA GPU：
> - **有 GPU** → 安装 CUDA 版 PyTorch
> - **无 GPU** → 安装 CPU 版 PyTorch
>
> ⚠️ **不要**直接 `pip install -r requirements.txt`，否则会把 CUDA 版 PyTorch 替换成 CPU 版。

**第3步：下载数据集和 NLTK 资源**

```bash
python data/download_data.py
```

**第4步：验证环境**

```bash
python main.py
```

输出 "所有环境依赖已全部就位" 即表示成功。

---

## 二、受害模型说明（无需自行训练）

本项目直接使用 **TextAttack 官方已微调好的模型**，所有成员无需运行训练脚本：

| 参数 | 值 |
|------|-----|
| 模型 | `textattack/bert-base-uncased-imdb` |
| 基础架构 | BERT-base-uncased |
| 微调数据集 | IMDB 情感分类（正面/负面二分类） |
| 干净准确率 | **93.7%** |

**首次运行攻击脚本时，模型自动下载到本地缓存（~418 MB）。**

---

## 三、五人分工

> 每人独立运行自己的脚本，结果文件互不覆盖。
> 成员3需要等成员1、2、4跑完后再执行汇总（成员5跑完后再次运行可追加防御对比）；成员5需要等成员1、2、4跑完后再执行对抗训练。

### 成员1 — TextFooler 复现（黑盒攻击）

**文件**：`attack/baseline_attack.py`

```bash
python attack/baseline_attack.py --attack textfooler
```

**产出**：`results/baseline/textfooler_results.csv`

**需要记录并与论文对比的指标**（对应论文 Table，BERT victim on IMDB）：

| 指标 | 论文报告值 | 你的复现值（填入） |
|------|-----------|-----------------|
| ASR（攻击成功率） | ~87% | |
| Avg Words Changed | ~6 词 | |
| Avg Queries | — | |
| Semantic Similarity | — | |

> Semantic Similarity 用 sentence-transformers（all-MiniLM-L6-v2）计算成功样本的余弦相似度均值。原论文使用 USE，模型不同数值会有轻微差异，需在报告中说明。

> 复现值与论文值存在差异是正常的（论文用的 BERT 版本、样本数可能不同），需在报告中说明原因。

**对应论文**：*Is BERT Really Robust? A Strong Baseline for Natural Language Attack on Text Classification and Entailment* Jin et al., AAAI 2020

---

### 成员2 — BERT-Attack 复现（黑盒攻击）

**文件**：`attack/baseline_attack.py`

```bash
python attack/baseline_attack.py --attack bertattack
```

**产出**：`results/baseline/bertattack_results.csv`

**需要记录并与论文对比的指标**（对应论文 Table 2，BERT on IMDB）：

| 指标 | 论文报告值 | 你的复现值（填入） |
|------|-----------|-----------------|
| ASR（攻击成功率） | ~90%+ | |
| Avg Words Changed | — | |
| Avg Queries | — | |
| Semantic Similarity | — | |

> Semantic Similarity 用 sentence-transformers（all-MiniLM-L6-v2）计算成功样本的余弦相似度均值。

**对应论文**：*BERT-ATTACK: Adversarial Attack Against BERT Using BERT* Li et al., EMNLP 2020

---

### 成员3 — 汇总评估与可视化

**文件**：`evaluate/evaluate.py`、`evaluate/visualize.py`

**步骤（等成员1、2、4跑完后执行）**：

```bash
# 第1步：汇总所有攻击实验结果
python evaluate/evaluate.py
# 产出：results/final_comparison.csv

# 第2步：生成可视化图表
python evaluate/visualize.py
# 产出：results/figures/*.png
```

**产出说明**：

| 文件 | 内容 |
|------|------|
| `results/final_comparison.csv` | 所有方法指标汇总对比表（ASR、查询次数、扰动率） |
| `results/defense_comparison.csv` | 防御对比表：干净模型 vs 鲁棒模型（含成员4改进方法） |
| `results/figures/asr_comparison.png` | 各方法 ASR 柱状图 |
| `results/figures/queries_comparison.png` | 各方法平均查询次数对比图 |
| `results/figures/wir_vs_awir.png` | WIR vs 改进方法对比图（成员4改进效果） |
| `results/figures/defense_comparison.png` | 防御效果分组柱状图：干净 vs 鲁棒模型 × 各攻击方法 |

> 等成员5跑完防御实验后，再次运行 `evaluate.py` 和 `visualize.py` 可自动追加防御对比结果与图表。

---

### 成员4 — 改进A：SC-BERT-Attack 语义约束攻击

**文件**：`attack/improved_attack.py`

```bash
export HF_ENDPOINT=https://hf-mirror.com
python attack/improved_attack.py --num_examples 100
```

**产出**：`results/improved/improved_results.json`（含 BERT-Attack 对照组和 SC-BERT-Attack 实验组的 ASR、查询次数、扰动率、语义相似度）

**需要额外下载** `bert-base-uncased` MLM 模型（~440MB），与分类器模型同时加载在 GPU 上约需 4GB 显存。

详见下方"改进A详解"。

---

### 成员5 — 鲁棒性分析：对抗训练防御

**文件**：`defense/adversarial_training.py`

**步骤（等成员1、2、4跑完后执行）**：

```bash
# 第1步：对抗训练，生成鲁棒模型
python defense/adversarial_training.py
# 产出：checkpoints/bert-imdb-adv/

# 第2步：用 TextFooler 攻击鲁棒模型，测量防御效果
python attack/baseline_attack.py \
    --attack textfooler \
    --model_dir checkpoints/bert-imdb-adv \
    --results_dir results/defense
# 产出：results/defense/baseline/textfooler_results.csv

# 第3步：用 BERT-Attack 攻击鲁棒模型
python attack/baseline_attack.py \
    --attack bertattack \
    --model_dir checkpoints/bert-imdb-adv \
    --results_dir results/defense
# 产出：results/defense/baseline/bertattack_results.csv

# 第4步：用改进攻击方法攻击鲁棒模型，验证防御是否对改进攻击也有效
python attack/improved_attack.py \
    --model_dir checkpoints/bert-imdb-adv \
    --results_dir results/defense
# 产出：results/defense/improved/improved_results.json
```

**需要记录的指标**（与原始模型对比，体现防御效果）：

| 攻击方法 | 原始模型 ASR | 鲁棒模型 ASR（填入） | 下降幅度 |
|----------|------------|-------------------|---------|
| TextFooler | （成员1的复现值） | | |
| BERT-Attack | （成员2的复现值） | | |
| 改进攻击方法 | （成员4的复现值） | | |

> 鲁棒准确率 = 1 − ASR。预期：对抗训练后 ASR 下降 20~40 个百分点，代价是干净准确率略降 3~5%。

详见下方"改进B详解"。

---

## 四、改进A详解：SC-BERT-Attack（语义约束 BERT 攻击）

### 标准 BERT-Attack 的局限性

BERT-Attack（Li et al., EMNLP 2020）使用 BERT-MLM 生成候选替换词，但不加语义约束。一个词如 "good" 可能被替换为 "bad" 只要能让分类器出错，导致对抗文本不自然、含义反转。

### SC-BERT-Attack 改进

在 MLM 生成候选后，使用 BERT 自身词向量的余弦相似度过滤候选词：

```
候选词 = { w ∈ MLM_top50(original_word) | cosine_sim(emb(w), emb(original_word)) ≥ 0.5 }
```

**直觉**：BERT 词向量空间中语义相近的词彼此靠近。用词向量距离过滤 MLM 输出，可移除反义词和无关词，保留真正的近义替换。

### 实验设计（A/B 对比）

两组使用相同的 WIR（删除评分）排序和 BERT-MLM 候选生成，唯一区别是语义过滤：

| 组别 | 候选来源 | 语义过滤 |
|------|----------|---------|
| 控制组（BERT-Attack） | BERT-MLM | 无 |
| 实验组（SC-BERT-Attack） | BERT-MLM | BERT 词向量余弦相似度 ≥ 0.5 |

**预期结果**：实验组语义相似度高于控制组，查询次数更低；ASR 可能略降（搜索空间受限）。

**实验结果（200 样本）**：

| 指标 | BERT-Attack | SC-BERT-Attack |
|------|------------|---------------|
| ASR | 33.0% | 19.5% |
| Avg Queries | 944.4 | **397.1**（↓58%） |
| Sem. Similarity | 0.9905 | **0.9923** |

### 局限性

1. 需要额外加载 BERT-MLM 模型（~440MB），对显存要求更高（约 4GB）
2. ASR 低于无约束的 BERT-Attack，适合追求对抗文本自然性的场景
3. 余弦阈值 0.5 为经验值，未对不同阈值做系统消融实验

---

## 五、鲁棒性分析：对抗训练防御

> **注意**：对抗训练是一种**防御方法**，不是对现有攻击工具的算法改进。改进A（成员4方法）在候选词生成阶段增加了语义约束过滤，属于方法层面的改进；对抗训练是从模型训练侧提升鲁棒性，对应题目"鲁棒性分析"部分。两者研究方向不同，互为补充。

### 核心思想

在训练时将对抗样本混入训练集，让模型见过被攻击的文本，从而提高鲁棒性。这是对抗鲁棒性研究的标准基线方法，直接对应题目中"鲁棒性分析"部分。

### 流程

```
① 用 TextFooler 攻击训练集，生成对抗样本
② 将对抗样本与干净样本 1:1 混合，构建混合训练集
③ 在混合数据集上重新微调 BERT（2 epoch）
④ 用 TextFooler 再次攻击，对比干净模型 vs 鲁棒模型的 ASR 变化
```

### 预期结果

| 模型 | 干净准确率 | 鲁棒准确率（被 TextFooler 攻击后） |
|------|-----------|----------------------------------|
| 原始模型 | | |
| 对抗训练后 ||  |


---

## 六、评估指标说明

| 指标 | 含义 | 越高代表 |
|------|------|---------|
| ASR（攻击成功率） | 成功让模型分类出错的比例 | 模型越脆弱 / 攻击越强 |
| Avg Queries | 平均每次攻击调用模型的次数 | 查询越少 = 攻击越高效 |
| Perturbation Rate | 平均修改词的比例 | 修改越少 = 攻击越隐蔽 |
| Semantic Similarity | 成功攻击样本中原文与对抗文本的余弦相似度均值（sentence-transformers） | 越高 = 对抗文本与原文越接近、攻击越隐蔽 |
| Robust Accuracy | 被攻击后仍能正确分类的比例 | 模型越鲁棒 |

---

## 七、常见问题

**Q：需要先运行训练脚本吗？**
> 不需要。所有攻击和防御脚本默认使用现成预训练模型，与 `train/finetune_bert.py` 完全独立。

---

## 八、环境要求

- Python 3.8~3.10
- PyTorch 2.0+（由 `install.py` 自动选择 GPU/CPU 版）
- transformers 4.38.0
- textattack 0.3.x
- datasets 2.18.0
- CUDA 11.8

详见 `requirements.txt`。

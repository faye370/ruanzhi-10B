# CLAUDE.md — BERT 对抗攻击与鲁棒性分析

5人课程项目，Python 3.8–3.10 + PyTorch + TextAttack + HuggingFace。

## 目录结构

```
ruanzhi/
├── configs/config.py          # 全局参数（模型、路径、超参）
├── data/download_data.py      # 下载数据集和 NLTK 资源
├── train/finetune_bert.py     # BERT 微调（参考用，攻击不依赖此步）
├── attack/
│   ├── baseline_attack.py     # TextFooler / BERT-Attack（TextAttack 框架）
│   └── improved_attack.py     # SC-BERT-Attack（BERT-MLM + embedding 语义过滤）
├── defense/adversarial_training.py  # 对抗训练防御
├── evaluate/
│   ├── evaluate.py            # 汇总所有结果 → CSV
│   └── visualize.py           # 生成对比图表
├── install.py                 # 一键安装（自动 GPU/CPU）
├── main.py                    # 环境验证
└── requirements.txt           # 依赖清单（不要直接 pip install -r！）
```

## 关键约束

- **禁止直接 `pip install -r requirements.txt`**：会把 CUDA 版 PyTorch 替换成 CPU 版。用 `python install.py`，它会自动检测 GPU。
- **Python 版本严格限制 3.8~3.10**，不能使用 3.11+（TextAttack 兼容性问题）。
- **无需本地训练**：所有脚本默认使用 `textattack/bert-base-uncased-imdb`（预训练好的 IMDB 分类模型，准确率 93.7%），首次运行自动下载 ~418MB。
- **结果目录自动创建**：`results/` 和 `checkpoints/` 由脚本自动生成，不需要手动创建。

## 模块间依赖关系

```
baseline_attack.py ──┐
improved_attack.py ──┼──→ evaluate.py → visualize.py
adversarial_training.py ──┘
```

- `evaluate.py` 需要等 baseline、improved、defense 的结果文件都存在后才跑得全。
- `adversarial_training.py` 用 TextFooler 生成对抗样本训练，不依赖 baseline_attack.py 的输出，但需要等 `train/finetune_bert.py`（或直接用预训练模型）。
- `visualize.py` 依赖 `evaluate.py` 的输出 CSV。

## 常用命令

```bash
# 环境搭建
python install.py
python data/download_data.py
python main.py                          # 验证环境

# 攻击实验
python attack/baseline_attack.py --attack textfooler
python attack/baseline_attack.py --attack bertattack
python attack/improved_attack.py --num_examples 100

# 防御实验（等攻击实验完成后）
python defense/adversarial_training.py
python attack/baseline_attack.py --attack textfooler --model_dir checkpoints/bert-imdb-adv --results_dir results/defense
python attack/baseline_attack.py --attack bertattack --model_dir checkpoints/bert-imdb-adv --results_dir results/defense
python attack/improved_attack.py --model_dir checkpoints/bert-imdb-adv --results_dir results/defense

# 汇总评估
python evaluate/evaluate.py
python evaluate/visualize.py
```

## 关键设计决策

- **baseline_attack.py 的输出是 CSV**（TextAttack 原生格式），**improved_attack.py 的输出是 JSON**（自实现格式，含 WIR 对照组 + AWIR 实验组）。evaluate.py 同时解析两种格式。
- **语义相似度**统一用 `sentence-transformers`（`all-MiniLM-L6-v2`），而非论文原用的 USE。数值可比但略有差异。
- **SC-BERT-Attack 实验结论**：在 BERT-Attack 的 MLM 候选生成基础上增加 BERT word embedding 余弦相似度过滤（阈值 0.5），可使语义相似度从 0.9905 提升至 0.9923（200 样本），查询次数降低 58%（944→397），代价是 ASR 从 33% 降至 19.5%。改进有效，攻击文本更自然、更高效。
- **improved_attack.py 需要额外下载 `bert-base-uncased` MLM 模型**（~440MB），设置 `HF_ENDPOINT=https://hf-mirror.com` 可加速。两个 BERT 模型（分类器 + MLM）同时加载在 GPU 上，需约 4GB 显存。
- **baseline_attack.py 用 TextAttack 框架内置的 counter-fitted embeddings**，ASR 可达 87%+；improved_attack.py 用自实现 BERT-MLM + embedding 过滤，ASR 较低但语义保持更好。两者因攻击策略不同，不可直接比较 ASR。

## 深入文档

| 文档 | 内容 |
|------|------|
| `ruanzhi/README.md` | 完整项目说明、五人分工、改进详解 |
| `ruanzhi/环境搭建README.md` | Conda 环境详细搭建指南（含 VS Code 配置） |
| `ruanzhi/attack/AWIR实验报告.md` | AWIR 实验完整报告（方法、实现、结果、分析） |

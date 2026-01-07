# NLP: ACL-2025-Fraud-R1
NLP大作业：对抗性数据改写与应用-Adversarial Data Rewriting & Application

## 项目概述

本项目是基于ACL 2025 Fraud-R1基准实现的欺诈对话检测系统，专注于对抗性数据改写与应用研究。该系统能够识别和检测5大类核心欺诈类型（虚假服务、冒充、钓鱼诈骗、虚假招聘、网络关系欺诈），并通过对抗性数据改写技术评估模型的鲁棒性。
论文链接：https://aclanthology.org/2025.findings-acl.226.pdf

系统基于BERT中文模型实现，支持多轮对话处理，提供数据增强、对抗样本生成、模型训练与评估等功能，并配有直观的Web应用界面。

## 项目特点

- **高精度欺诈检测**：基于BERT模型实现，具有较高的欺诈对话检测准确率
- **对抗性数据改写**：实现了多种对抗攻击技术，用于评估模型的鲁棒性
- **丰富的数据增强**：提供多种欺诈对话数据增强策略，提高模型泛化能力
- **可视化分析**：生成多种图表和报告，直观展示实验结果
- **Web应用界面**：提供用户友好的Web界面，便于使用和展示

### 数据集文件
- `data/train.csv`：原始训练数据集
- `data/test.csv`：测试数据集
- `data/augmented_train.csv`：增强后的训练数据集

## 安装指南

### 环境要求
- Python 3.10
- PyTorch 2.0+
- CUDA 11.7+ (NVIDIA GeForce RTX 2050[4GB])

### 安装依赖
```bash
pip install -r requirements.txt
```
注意：安装过程中可能需要根据系统配置调整PyTorch版本，确保与CUDA兼容。

## 主要功能模块

### 1. 实验主框架 (`compare_datasets.py`)
- 实验流程控制
- 数据加载和预处理
- 模型评估
- 实验结果记录

### 2. 欺诈对话增强器 (`fraud_dialogue_augmenter.py`)
- 基于Fraud-R1策略的欺诈对话增强
- 可信度建立策略
- 紧迫感制造策略
- 情感操纵策略

### 3. Web应用服务 (`app.py`)
- 应用程序入口
- 用户交互界面
- 演示功能

## 使用方法

### 1. 生成对抗性样本
```bash
python fraud_dialogue_augmenter.py
```

### 2. 运行实验
```bash
python compare_datasets.py
```

### 3. 启动应用程序
```bash
python app.py
```

## 评估指标

- **防御成功率（DSR）**：模型正确识别欺诈对话的比例
- **多轮防御成功率（DSR@k）**：模型在第k轮及之前正确识别欺诈对话的比例
- **平均检测轮数（AVG (k)）**：模型正确识别欺诈对话所需的平均轮数
- **攻击成功率**：对抗攻击成功降低模型判别准确率的比例

## 项目结构

```
├── data/                        # 数据集目录
│   ├── augmented_train.csv      # 增强后的训练数据
│   ├── test.csv                 # 测试数据
│   └── train.csv                # 原始训练数据
├── templates/                   # HTML模板
│   └── index.html               # Web应用界面模板
├── app.py                       # 应用程序入口
├── fraud_dialogue_augmenter.py  # 欺诈对话增强器
└── compare_datasets.py          # 对比不同数据集的实验
```

## 技术实现细节

### 对话改写/对抗生成流程
1. 数据预处理：清洗和预处理Fraud-R1基准数据集
2. 策略选择：根据三步骤诱导策略选择改写策略
3. 对抗样本生成：基于选定策略生成对抗性欺诈对话
4. 模型评估：使用生成的对抗样本评估目标模型
5. 结果分析：分析模型在对抗样本上的表现

### 三步骤诱导策略
1. **建立可信度**：通过虚假官方身份或认证信息获取信任
2. **制造紧迫感**：设置时间限制等方式制造紧迫感
3. **情感操纵**：通过情感诉求降低警惕性

## 应用场景

1. **电信欺诈检测**：实时检测电话中的欺诈对话
2. **金融安全**：识别金融领域的欺诈电话
3. **研究分析**：用于欺诈对话检测算法的研究和比较
4. **安全教育**：作为欺诈识别的教育工具使用

## 实验结果说明

### 模型文件
- `models/bert_fraud_original.pt`：使用原始数据集训练的BERT模型
- `models/bert_fraud_augmented.pt`：使用增强数据集训练的BERT模型

### 训练历史图像
- `results/training_history_original.png`：原始数据集的训练损失和准确率曲线
- `results/training_history_augmented.png`：增强数据集的训练损失和准确率曲线

## 项目相关文档

- `README.md`：项目说明文档，包含项目结构、环境配置和使用方法

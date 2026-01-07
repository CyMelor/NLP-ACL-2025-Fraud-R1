import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader, Dataset
from transformers import BertTokenizer, BertForSequenceClassification, AdamW, get_linear_schedule_with_warmup
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix
from sklearn.model_selection import train_test_split
import os
import time

# 设置GPU使用
if torch.cuda.is_available():
    device = torch.device("cuda:0")  # 指定具体的GPU设备索引
    torch.cuda.set_per_process_memory_fraction(0.5, device=0)  # 限制GPU内存使用约4G
    torch.backends.cudnn.benchmark = False  # 禁用基准测试以减少内存使用
    torch.backends.cudnn.deterministic = True  # 使用确定性算法
    print(f"使用GPU: {torch.cuda.get_device_name(0)}")
    # 检查是否支持混合精度
    use_amp = True
else:
    device = torch.device("cpu")
    print("使用CPU")
    use_amp = False

# 配置参数
class Config:
    def __init__(self):
        self.model_name = "bert-base-chinese"
        self.learning_rate = 2e-5
        self.epochs = 3
        self.batch_size = 4
        self.max_length = 64
        self.gradient_accumulation_steps = 4
        self.early_stopping_patience = 2
        self.save_dir = "models"
        self.result_dir = "results"
        
        # 创建必要的目录
        os.makedirs(self.save_dir, exist_ok=True)
        os.makedirs(self.result_dir, exist_ok=True)

config = Config()

# 自定义数据集类
class FraudDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_length):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = str(self.texts[idx])
        label = int(self.labels[idx])
        
        encoding = self.tokenizer(
            text,
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors='pt'
        )
        
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(label, dtype=torch.long)
        }

# 加载和预处理数据
def load_data(train_path, test_path):
    print("\n加载数据...")
    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)
    
    # 处理is_fraud列的类型转换
    print("处理数据类型...")
    
    # 检查并处理NaN值
    train_df = train_df.dropna(subset=['is_fraud', 'specific_dialogue_content'])
    test_df = test_df.dropna(subset=['is_fraud', 'specific_dialogue_content'])
    
    # 确保is_fraud是布尔类型
    train_df['is_fraud'] = train_df['is_fraud'].astype(bool)
    test_df['is_fraud'] = test_df['is_fraud'].astype(bool)
    
    # 输出数据规模
    print(f"训练集规模: {len(train_df)} 样本")
    print(f"测试集规模: {len(test_df)} 样本")
    print(f"训练集欺诈比例: {train_df['is_fraud'].mean():.2%}")
    print(f"测试集欺诈比例: {test_df['is_fraud'].mean():.2%}")
    
    # 准备数据
    X_train = train_df['specific_dialogue_content'].values
    y_train = train_df['is_fraud'].astype(int).values
    X_test = test_df['specific_dialogue_content'].values
    y_test = test_df['is_fraud'].astype(int).values
    
    return X_train, y_train, X_test, y_test, train_df, test_df

# 训练模型
def train_model(X_train, y_train, X_val, y_val, tokenizer, config, dataset_name):
    print(f"\n开始训练 {dataset_name}...")
    
    # 创建数据集和数据加载器
    train_dataset = FraudDataset(X_train, y_train, tokenizer, config.max_length)
    val_dataset = FraudDataset(X_val, y_val, tokenizer, config.max_length)
    
    train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=config.batch_size, shuffle=False)
    
    # 加载模型
    model = BertForSequenceClassification.from_pretrained(
        config.model_name,
        num_labels=2,
        output_attentions=False,
        output_hidden_states=False
    )
    model.to(device)
    
    # 优化器和调度器
    optimizer = AdamW(model.parameters(), lr=config.learning_rate, eps=1e-8)
    total_steps = len(train_loader) * config.epochs
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=0, num_training_steps=total_steps
    )
    
    # 混合精度训练设置
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)
    
    # 训练历史
    train_losses = []
    val_losses = []
    val_accuracies = []
    
    # 早停机制
    best_val_accuracy = 0
    patience_counter = 0
    
    for epoch in range(config.epochs):
        print(f"\nEpoch {epoch+1}/{config.epochs}")
        print("=" * 30)
        
        # 训练阶段
        model.train()
        total_train_loss = 0
        
        # 初始化时间记录
        t0 = time.time()
        
        # 梯度累积相关初始化
        optimizer.zero_grad()
        
        for step, batch in enumerate(train_loader):
            if step % 100 == 0 and not step == 0:
                elapsed = format_time(time.time() - t0)
                print(f"  Batch {step}/{len(train_loader)}  Elapsed: {elapsed}")
                # 重置时间记录
                t0 = time.time()
            
            b_input_ids = batch['input_ids'].to(device)
            b_attention_mask = batch['attention_mask'].to(device)
            b_labels = batch['labels'].to(device)
            
            # 前向传播 - 使用混合精度
            with torch.cuda.amp.autocast(enabled=use_amp):
                outputs = model(
                    b_input_ids,
                    attention_mask=b_attention_mask,
                    labels=b_labels
                )
                loss = outputs.loss
            
            total_train_loss += loss.item()
            
            # 梯度累积和缩放
            loss = loss / config.gradient_accumulation_steps
            scaler.scale(loss).backward()
            
            # 只有在累积了足够的梯度后才更新参数
            if (step + 1) % config.gradient_accumulation_steps == 0 or step == len(train_loader) - 1:
                # 梯度裁剪
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                
                # 更新参数
                scaler.step(optimizer)
                scheduler.step()
                scaler.update()
                optimizer.zero_grad()
        
        avg_train_loss = total_train_loss / len(train_loader)
        train_losses.append(avg_train_loss)
        print(f"\n  训练损失: {avg_train_loss:.4f}")
        
        # 验证阶段
        model.eval()
        total_val_loss = 0
        predictions, true_labels = [], []
        
        with torch.no_grad():
            for batch in val_loader:
                b_input_ids = batch['input_ids'].to(device)
                b_attention_mask = batch['attention_mask'].to(device)
                b_labels = batch['labels'].to(device)
                
                outputs = model(
                    b_input_ids,
                    attention_mask=b_attention_mask,
                    labels=b_labels
                )
                
                loss = outputs.loss
                total_val_loss += loss.item()
                
                logits = outputs.logits
                logits = logits.detach().cpu().numpy()
                label_ids = b_labels.to('cpu').numpy()
                
                predictions.append(logits)
                true_labels.append(label_ids)
        
        avg_val_loss = total_val_loss / len(val_loader)
        val_losses.append(avg_val_loss)
        
        # 计算准确率
        predictions = np.concatenate(predictions, axis=0)
        true_labels = np.concatenate(true_labels, axis=0)
        preds_flat = np.argmax(predictions, axis=1).flatten()
        labels_flat = true_labels.flatten()
        
        accuracy = accuracy_score(labels_flat, preds_flat)
        val_accuracies.append(accuracy)
        
        print(f"  验证损失: {avg_val_loss:.4f}")
        print(f"  验证准确率: {accuracy:.4f}")
        
        # 早停检查
        if accuracy > best_val_accuracy:
            best_val_accuracy = accuracy
            patience_counter = 0
            # 保存最佳模型
            model_save_path = os.path.join(config.save_dir, f"bert_fraud_{dataset_name}.pt")
            torch.save(model.state_dict(), model_save_path)
            print(f"  保存最佳模型到 {model_save_path}")
        else:
            patience_counter += 1
            print(f"  早停计数器: {patience_counter}/{config.early_stopping_patience}")
            if patience_counter >= config.early_stopping_patience:
                print(f"  早停触发，停止训练")
                break
    
    return model, train_losses, val_losses, val_accuracies

# 评估模型
def evaluate_model(model, X_test, y_test, tokenizer, config):
    print("\n评估模型...")
    
    test_dataset = FraudDataset(X_test, y_test, tokenizer, config.max_length)
    test_loader = DataLoader(test_dataset, batch_size=config.batch_size, shuffle=False)
    
    model.eval()
    predictions, true_labels = [], []
    
    with torch.no_grad():
        for batch in test_loader:
            b_input_ids = batch['input_ids'].to(device)
            b_attention_mask = batch['attention_mask'].to(device)
            b_labels = batch['labels'].to(device)
            
            outputs = model(
                b_input_ids,
                attention_mask=b_attention_mask
            )
            
            logits = outputs.logits
            logits = logits.detach().cpu().numpy()
            label_ids = b_labels.to('cpu').numpy()
            
            predictions.append(logits)
            true_labels.append(label_ids)
    
    predictions = np.concatenate(predictions, axis=0)
    true_labels = np.concatenate(true_labels, axis=0)
    preds_flat = np.argmax(predictions, axis=1).flatten()
    labels_flat = true_labels.flatten()
    
    # 计算指标
    accuracy = accuracy_score(labels_flat, preds_flat)
    precision, recall, f1, _ = precision_recall_fscore_support(labels_flat, preds_flat, average='binary')
    cm = confusion_matrix(labels_flat, preds_flat)
    
    print(f"\n测试结果:")
    print(f"准确率: {accuracy:.4f}")
    print(f"精确率: {precision:.4f}")
    print(f"召回率: {recall:.4f}")
    print(f"F1分数: {f1:.4f}")
    print(f"混淆矩阵:")
    print(cm)
    
    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'confusion_matrix': cm,
        'predictions': preds_flat,
        'true_labels': labels_flat
    }

# 生成训练图像
def plot_training_history(train_losses, val_losses, val_accuracies, dataset_name, config):
    plt.figure(figsize=(12, 5))
    
    # 损失曲线
    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label='训练损失')
    plt.plot(val_losses, label='验证损失')
    plt.title(f'{dataset_name} 训练与验证损失')
    plt.xlabel('轮次')
    plt.ylabel('损失')
    plt.legend()
    plt.grid(True)
    
    # 准确率曲线
    plt.subplot(1, 2, 2)
    plt.plot(val_accuracies, label='验证准确率')
    plt.title(f'{dataset_name} 验证准确率')
    plt.xlabel('轮次')
    plt.ylabel('准确率')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    image_path = os.path.join(config.result_dir, f"training_history_{dataset_name}.png")
    plt.savefig(image_path)
    plt.close()
    print(f"\n训练图像已保存到 {image_path}")

# 将结果保存到markdown文件
def save_results_to_markdown(results, dataset_name, config):
    markdown_path = os.path.join(config.result_dir, f"{dataset_name}_results.md")
    
    with open(markdown_path, 'w', encoding='utf-8') as f:
        f.write(f"# {dataset_name} 数据集训练结果\n\n")
        f.write("## 模型评估指标\n\n")
        f.write(f"- **准确率**: {results['accuracy']:.4f}\n")
        f.write(f"- **精确率**: {results['precision']:.4f}\n")
        f.write(f"- **召回率**: {results['recall']:.4f}\n")
        f.write(f"- **F1分数**: {results['f1']:.4f}\n\n")
        
        f.write("## 混淆矩阵\n\n")
        f.write("```\n")
        f.write(f"{results['confusion_matrix']}\n")
        f.write("```\n\n")
    
    print(f"\n{dataset_name} 训练结果已保存到 {markdown_path}")

# 比较两个模型的结果
def compare_models(results1, results2, name1, name2, config):
    print(f"\n比较 {name1} 和 {name2} 的结果:")
    print(f"准确率差异: {results2['accuracy'] - results1['accuracy']:.4f}")
    print(f"精确率差异: {results2['precision'] - results1['precision']:.4f}")
    print(f"召回率差异: {results2['recall'] - results1['recall']:.4f}")
    print(f"F1分数差异: {results2['f1'] - results1['f1']:.4f}")
    
    # 绘制对比图
    metrics = ['accuracy', 'precision', 'recall', 'f1']
    values1 = [results1[m] for m in metrics]
    values2 = [results2[m] for m in metrics]
    
    x = np.arange(len(metrics))
    width = 0.35
    
    plt.figure(figsize=(10, 6))
    plt.bar(x - width/2, values1, width, label=name1)
    plt.bar(x + width/2, values2, width, label=name2)
    plt.title('两个模型的性能对比')
    plt.ylabel('分数')
    plt.xticks(x, metrics)
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # 添加数值标签
    for i, v in enumerate(values1):
        plt.text(i - width/2, v + 0.01, f'{v:.4f}', ha='center')
    for i, v in enumerate(values2):
        plt.text(i + width/2, v + 0.01, f'{v:.4f}', ha='center')
    
    plt.tight_layout()
    image_path = os.path.join(config.result_dir, f"model_comparison.png")
    plt.savefig(image_path)
    plt.close()
    print(f"\n模型对比图已保存到 {image_path}")
    
    # 保存比较结果到markdown
    markdown_path = os.path.join(config.result_dir, "model_comparison.md")
    with open(markdown_path, 'w', encoding='utf-8') as f:
        f.write("# 模型性能比较\n\n")
        f.write(f"## {name1} vs {name2}\n\n")
        f.write(f"- **准确率差异**: {results2['accuracy'] - results1['accuracy']:.4f}\n")
        f.write(f"- **精确率差异**: {results2['precision'] - results1['precision']:.4f}\n")
        f.write(f"- **召回率差异**: {results2['recall'] - results1['recall']:.4f}\n")
        f.write(f"- **F1分数差异**: {results2['f1'] - results1['f1']:.4f}\n\n")
        
        f.write("## 详细指标\n\n")
        f.write("| 指标 | {name1} | {name2} |\n".format(name1=name1, name2=name2))
        f.write("|------|---------|---------|\n")
        f.write("| 准确率 | {:.4f} | {:.4f} |\n".format(results1['accuracy'], results2['accuracy']))
        f.write("| 精确率 | {:.4f} | {:.4f} |\n".format(results1['precision'], results2['precision']))
        f.write("| 召回率 | {:.4f} | {:.4f} |\n".format(results1['recall'], results2['recall']))
        f.write("| F1分数 | {:.4f} | {:.4f} |\n".format(results1['f1'], results2['f1']))
    
    print(f"\n模型比较结果已保存到 {markdown_path}")

# 主函数
def main():
    print("BERT中文模型欺诈检测 - 数据集比较")
    print("=" * 50)
    
    # 加载tokenizer
    tokenizer = BertTokenizer.from_pretrained(config.model_name)
    
    # 路径设置
    original_train_path = "data/train.csv"
    augmented_train_path = "data/augmented_train.csv"
    test_path = "data/test.csv"
    
    # 1. 加载测试数据
    print("\n" + "=" * 50)
    print("1. 加载测试数据")
    print("=" * 50)
    # 直接加载测试数据，不训练原始数据集
    _, _, X_test, y_test, _, test_df = load_data(original_train_path, test_path)
    
    # 2. 训练增强数据集
    print("\n" + "=" * 50)
    print("2. 增强数据集训练")
    print("=" * 50)
    X_train_augmented, y_train_augmented, _, _, train_df_augmented, _ = load_data(augmented_train_path, test_path)
    
    # 划分验证集
    X_train_aug, X_val_aug, y_train_aug, y_val_aug = train_test_split(
        X_train_augmented, y_train_augmented, test_size=0.1, random_state=42, stratify=y_train_augmented
    )
    
    # 训练模型
    model_augmented, train_losses_aug, val_losses_aug, val_accuracies_aug = train_model(
        X_train_aug, y_train_aug, X_val_aug, y_val_aug, tokenizer, config, "augmented"
    )
    
    # 绘制训练历史
    plot_training_history(train_losses_aug, val_losses_aug, val_accuracies_aug, "augmented", config)
    
    # 评估模型
    results_augmented = evaluate_model(model_augmented, X_test, y_test, tokenizer, config)
    # 保存结果到markdown文件
    save_results_to_markdown(results_augmented, "augmented", config)
    
    # 3. 增强数据集结果总结
    print("\n" + "=" * 50)
    print("3. 增强数据集结果总结")
    print("=" * 50)
    print("增强数据集模型已完成训练和评估")
    print(f"最终测试准确率: {results_augmented['accuracy']:.4f}")
    print(f"最终F1分数: {results_augmented['f1']:.4f}")
    print(f"详细结果已保存到: {os.path.join(config.result_dir, 'augmented_results.md')}")
    
    print("\n" + "=" * 50)
    print("训练完成！")
    print(f"模型已保存到: {config.save_dir}")
    print(f"结果图像已保存到: {config.result_dir}")
    print("=" * 50)

# 辅助函数：格式化时间
def format_time(elapsed):
    elapsed_rounded = int(round((elapsed)))
    return time.strftime("%H:%M:%S", time.gmtime(elapsed_rounded))

if __name__ == "__main__":
    main()
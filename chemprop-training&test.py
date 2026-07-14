# ==============================================
# 药物神经发育毒性预测 Chemprop v2.2.3 训练脚本
# 任务：二分类（1=有毒，0=无毒）
# 训练策略：从单一文件按比例划分训练/测试集
# ==============================================

# -------------------------- 1. 环境依赖导入 --------------------------
import os
import json
import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    roc_auc_score, accuracy_score, precision_score, recall_score,
    f1_score, confusion_matrix, roc_curve, classification_report
)
from lightning import pytorch as pl
from chemprop import data, featurizers, models, nn

# 设置绘图风格
sns.set_style("whitegrid")
sns.set_palette("husl")
plt.rcParams['font.size'] = 10
plt.rcParams['axes.labelsize'] = 10
plt.rcParams['axes.titlesize'] = 11
plt.rcParams['legend.fontsize'] = 9

# -------------------------- 2. 全局配置 --------------------------
# ========== 修改这里：改为你的原始数据文件路径 ==========
DATA_PATH = "C:/Users/JOHN/Desktop/原始SMILES数据/hf 原始SMILES数据.csv "
# ====================================================

OUTPUT_DIR = Path.cwd() / "E:/chemprop"
OUTPUT_DIR.mkdir(exist_ok=True)
CHECKPOINT_DIR = OUTPUT_DIR / "checkpoints"
CHECKPOINT_DIR.mkdir(exist_ok=True)

SMILES_COLUMN = "smiles"  # 你的SMILES列名
TARGET_COLUMN = "label"  # 你的标签列名

TEST_SIZE = 0.2  # 测试集比例（20%）
RANDOM_SEED = 42
MAX_EPOCHS = 50
BATCH_SIZE = 32
NUM_WORKERS = 0

EARLY_STOPPING_MONITOR = "val/roc"
EARLY_STOPPING_MODE = "max"
EARLY_STOPPING_PATIENCE = 15


# -------------------------- 3. 数据加载与划分 --------------------------
def load_and_split_data(file_path: str, test_size=0.2, random_state=42):
    """从单一文件加载数据，并划分为训练集和测试集"""

    # 明确指定第一行为列名
    df = pd.read_csv(file_path, header=0)

    # 打印列名确认
    print(f"列名: {df.columns.tolist()}")
    if len(df) > 0:
        print(f"第一行 SMILES 预览: {str(df.iloc[0][SMILES_COLUMN])[:80]}...")
    else:
        print("警告：文件无数据")

    # 检查必要的列是否存在
    if SMILES_COLUMN not in df.columns or TARGET_COLUMN not in df.columns:
        raise ValueError(f"数据集缺少必要列：{SMILES_COLUMN} 或 {TARGET_COLUMN}，\n可用列: {df.columns.tolist()}")

    # 过滤无效SMILES并创建 MoleculeDatapoint
    from rdkit import Chem
    smis = df[SMILES_COLUMN].values
    ys = df[[TARGET_COLUMN]].values

    valid_datapoints = []
    valid_indices = []

    for i, (smi, y) in enumerate(zip(smis, ys)):
        smi = str(smi).strip()
        if not smi or smi == 'nan':
            print(f"警告：跳过空SMILES - 行 {i}")
            continue
        try:
            dp = data.MoleculeDatapoint.from_smi(smi, y)
            if dp.mol is not None:
                valid_datapoints.append(dp)
                valid_indices.append(i)
            else:
                print(f"警告：无效SMILES被过滤 - {smi[:50]}...")
        except Exception as e:
            print(f"警告：处理SMILES出错 - {smi[:50]}..., 错误: {e}")

    # 保存过滤后的 DataFrame
    df_valid = df.iloc[valid_indices].reset_index(drop=True)

    # 按标签分层划分训练集和测试集
    from sklearn.model_selection import train_test_split
    labels = df_valid[TARGET_COLUMN].values
    train_idx, test_idx = train_test_split(
        range(len(valid_datapoints)),
        test_size=test_size,
        random_state=random_state,
        stratify=labels
    )

    train_dps = [valid_datapoints[i] for i in train_idx]
    test_dps = [valid_datapoints[i] for i in test_idx]

    train_df = df_valid.iloc[train_idx].reset_index(drop=True)
    test_df = df_valid.iloc[test_idx].reset_index(drop=True)

    print(f"原始数据有效分子数：{len(valid_datapoints)}")
    print(f"训练集样本数：{len(train_dps)} ({test_size * 100:.0f}% 用于测试)")
    print(f"测试集样本数：{len(test_dps)}")
    print(f"训练集类别分布：\n{train_df[TARGET_COLUMN].value_counts()}")
    print(f"测试集类别分布：\n{test_df[TARGET_COLUMN].value_counts()}")

    # 保存划分后的CSV文件
    from pathlib import Path
    output_dir = Path("E:/chemprop")
    output_dir.mkdir(exist_ok=True)
    train_df.to_csv(output_dir / f"{Path(file_path).stem}_train_split.csv", index=False)
    test_df.to_csv(output_dir / f"{Path(file_path).stem}_test_split.csv", index=False)
    print(f"✅ 训练集已保存：{output_dir / f'{Path(file_path).stem}_train_split.csv'}")
    print(f"✅ 测试集已保存：{output_dir / f'{Path(file_path).stem}_test_split.csv'}")

    return train_dps, test_dps, train_df, test_df


def build_dataloaders(train_dps, test_dps, batch_size=BATCH_SIZE):
    featurizer = featurizers.SimpleMoleculeMolGraphFeaturizer()
    train_dset = data.MoleculeDataset(train_dps, featurizer)
    test_dset = data.MoleculeDataset(test_dps, featurizer)

    train_loader = data.build_dataloader(
        train_dset, batch_size=batch_size, num_workers=NUM_WORKERS, shuffle=True
    )
    test_loader = data.build_dataloader(
        test_dset, batch_size=batch_size, num_workers=NUM_WORKERS, shuffle=False
    )
    return train_loader, test_loader


# -------------------------- 4. 模型构建 --------------------------
def build_default_model(n_tasks: int = 1):
    # 增加模型容量
    DEFAULT_DEPTH = 3                   # 从 3 改为 4
    DEFAULT_MESSAGE_HIDDEN_DIM = 300       # 从 300 改为 600
    DEFAULT_FFN_HIDDEN_DIM = 400          # 从 400 改为 600
    DEFAULT_FFN_NUM_LAYERS = 2
    DEFAULT_DROPOUT = 0.25
    DEFAULT_BATCH_NORM = True
    DEFAULT_ACTIVATION = "RELU"

    mp = nn.BondMessagePassing(
        d_h=DEFAULT_MESSAGE_HIDDEN_DIM,
        depth=DEFAULT_DEPTH,
        dropout=DEFAULT_DROPOUT,
        activation=DEFAULT_ACTIVATION
    )
    agg = nn.MeanAggregation()
    ffn = nn.BinaryClassificationFFN(
        n_tasks=n_tasks,
        input_dim=DEFAULT_MESSAGE_HIDDEN_DIM,
        hidden_dim=DEFAULT_FFN_HIDDEN_DIM,
        n_layers=DEFAULT_FFN_NUM_LAYERS,
        dropout=DEFAULT_DROPOUT,
        activation=DEFAULT_ACTIVATION
    )
    model = models.MPNN(
        message_passing=mp,
        agg=agg,
        predictor=ffn,
        batch_norm=DEFAULT_BATCH_NORM
    )
    return model


# -------------------------- 5. 自定义回调：记录每个epoch的完整指标 --------------------------
class FullMetricsCallback(pl.Callback):
    """在每个epoch结束时计算训练集和验证集的完整指标（Loss, ROC-AUC, Accuracy）"""

    def __init__(self, train_loader, val_loader):
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.history = {
            'epoch': [],
            'train_loss': [],
            'train_roc': [],
            'train_acc': [],
            'val_loss': [],
            'val_roc': [],
            'val_acc': []
        }

    def on_validation_epoch_end(self, trainer, pl_module):
        epoch = trainer.current_epoch
        # 获取验证集预测结果
        val_loss, val_roc, val_acc, _, _ = self._evaluate(pl_module, self.val_loader)
        # 训练集评估
        train_loss, train_roc, train_acc, _, _ = self._evaluate(pl_module, self.train_loader)

        self.history['epoch'].append(epoch)
        self.history['train_loss'].append(train_loss)
        self.history['train_roc'].append(train_roc)
        self.history['train_acc'].append(train_acc)
        self.history['val_loss'].append(val_loss)
        self.history['val_roc'].append(val_roc)
        self.history['val_acc'].append(val_acc)

        # 记录到日志
        if trainer.logger is not None:
            trainer.logger.log_metrics({
                'train/loss_epoch': train_loss,
                'train/roc_epoch': train_roc,
                'train/accuracy_epoch': train_acc,
                'val/loss_epoch': val_loss,
                'val/roc_epoch': val_roc,
                'val/accuracy_epoch': val_acc
            }, step=epoch)

        print(f"\nEpoch {epoch}: Train Loss={train_loss:.4f}, Train ROC={train_roc:.4f}, Train Acc={train_acc:.4f} | "
              f"Val Loss={val_loss:.4f}, Val ROC={val_roc:.4f}, Val Acc={val_acc:.4f}")

    def _evaluate(self, model, loader):
        """计算在给定loader上的平均损失、ROC-AUC、Accuracy"""
        model.eval()
        all_preds = []
        all_labels = []
        total_loss = 0.0
        loss_fn = torch.nn.BCELoss()
        with torch.no_grad():
            for batch in loader:
                bmg = batch.bmg
                y = batch.Y.float()  # 形状: (batch_size, 1)
                pred = model(bmg).view(-1)  # 形状: (batch_size,)
                y_flat = y.view(-1)  # 形状: (batch_size,)
                loss = loss_fn(pred, y_flat)
                total_loss += loss.item() * y.size(0)
                all_preds.extend(pred.cpu().numpy())
                all_labels.extend(y_flat.cpu().numpy())
        avg_loss = total_loss / len(loader.dataset)
        all_preds = np.array(all_preds)
        all_labels = np.array(all_labels)
        roc = roc_auc_score(all_labels, all_preds) if len(np.unique(all_labels)) > 1 else 0.5
        pred_class = (all_preds >= 0.5).astype(int)
        acc = accuracy_score(all_labels, pred_class)
        return avg_loss, roc, acc, all_labels, all_preds


# -------------------------- 6. 绘图函数 --------------------------
def plot_training_history(history: dict, save_path: Path, network_arch: str = "Chemprop MPNN"):
    """根据回调记录的history字典绘制训练曲线"""
    df = pd.DataFrame(history)
    best_idx = df['val_roc'].idxmax()
    best_epoch = int(df.loc[best_idx, 'epoch'])
    best_roc = df.loc[best_idx, 'val_roc']

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    # Loss曲线
    axes[0].plot(df['epoch'], df['train_loss'], 'b-o', label='Training Loss', markersize=4, linewidth=1.5)
    axes[0].plot(df['epoch'], df['val_loss'], 'r-s', label='Test Loss', markersize=4, linewidth=1.5)
    axes[0].axvline(x=best_epoch, color='gray', linestyle='--', alpha=0.7, label=f'Best Epoch: {best_epoch}')
    axes[0].plot(best_epoch, df.loc[best_idx, 'val_loss'], 'ro', markersize=8)
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].set_title('Model Loss')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    axes[0].set_ylim(bottom=0)

    # ROC-AUC曲线
    axes[1].plot(df['epoch'], df['train_roc'], 'b-o', label='Training AUC', markersize=4, linewidth=1.5)
    axes[1].plot(df['epoch'], df['val_roc'], 'r-s', label='Test AUC', markersize=4, linewidth=1.5)
    axes[1].axvline(x=best_epoch, color='gray', linestyle='--', alpha=0.7)
    axes[1].plot(best_epoch, df.loc[best_idx, 'val_roc'], 'ro', markersize=8, label=f'Best: {best_roc:.4f}')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('AUC')
    axes[1].set_title('Model AUC')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    axes[1].set_ylim(0, 1)

    # Accuracy曲线
    axes[2].plot(df['epoch'], df['train_acc'], 'b-o', label='Training Accuracy', markersize=4, linewidth=1.5)
    axes[2].plot(df['epoch'], df['val_acc'], 'r-s', label='Test Accuracy', markersize=4, linewidth=1.5)
    axes[2].axvline(x=best_epoch, color='gray', linestyle='--', alpha=0.7)
    axes[2].plot(best_epoch, df.loc[best_idx, 'val_acc'], 'ro', markersize=8)
    axes[2].set_xlabel('Epoch')
    axes[2].set_ylabel('Accuracy')
    axes[2].set_title('Model Accuracy')
    axes[2].legend()
    axes[2].grid(True, alpha=0.3)
    axes[2].set_ylim(0, 1)

    plt.suptitle(f"Training History - Network Architecture: {network_arch} | Best Epoch: {best_epoch}", fontsize=14)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✅ 训练历史图已保存：{save_path} (最佳epoch={best_epoch}, 验证集ROC-AUC={best_roc:.4f})")

    df.to_csv(save_path.with_suffix('.csv'), index=False)
    print(f"训练历史数据已保存：{save_path.with_suffix('.csv')}")


# -------------------------- 7. 评估函数 --------------------------
def evaluate_model(model, test_loader, test_df, output_dir: Path, model_name="Best_Model"):
    model.eval()
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for batch in test_loader:
            bmg = batch.bmg
            y = batch.Y
            pred = model(bmg)
            all_labels.extend(y.cpu().numpy().flatten())
            all_preds.extend(pred.cpu().numpy().flatten())

    y_true = np.array(all_labels)
    y_pred_prob = np.array(all_preds)
    y_pred_class = (y_pred_prob >= 0.5).astype(int)

    tn, fp, fn, tp = confusion_matrix(y_true, y_pred_class).ravel()
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0

    metrics = {
        'Accuracy': accuracy_score(y_true, y_pred_class),
        'Precision': precision_score(y_true, y_pred_class, zero_division=0),
        'Recall': recall_score(y_true, y_pred_class, zero_division=0),
        'F1-Score': f1_score(y_true, y_pred_class, zero_division=0),
        'ROC-AUC': roc_auc_score(y_true, y_pred_prob),
        'Specificity': specificity
    }

    with open(output_dir / f"{model_name}_metrics.json", 'w') as f:
        json.dump(metrics, f, indent=4)
    metrics_df = pd.DataFrame([metrics]).T
    metrics_df.columns = ['Value']
    metrics_df.to_csv(output_dir / f"{model_name}_metrics.csv")
    print(f"\n{model_name} 测试集评估指标：")
    for k, v in metrics.items():
        print(f"  {k}: {v:.4f}")

    # 混淆矩阵
    plt.figure(figsize=(5, 4))
    sns.heatmap(confusion_matrix(y_true, y_pred_class), annot=True, fmt='d', cmap='Blues',
                xticklabels=['Non-toxic (0)', 'Toxic (1)'],
                yticklabels=['Non-toxic (0)', 'Toxic (1)'])
    plt.title(f'{model_name} Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.savefig(output_dir / f"{model_name}_confusion_matrix.png", dpi=300)
    plt.close()

    # ROC曲线
    fpr, tpr, _ = roc_curve(y_true, y_pred_prob)
    plt.figure(figsize=(5, 4))
    plt.plot(fpr, tpr, label=f'ROC-AUC = {metrics["ROC-AUC"]:.4f}')
    plt.plot([0, 1], [0, 1], 'k--', label='Random')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'{model_name} ROC Curve')
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_dir / f"{model_name}_roc_curve.png", dpi=300)
    plt.close()

    # 预测结果CSV
    pred_df = test_df.copy()
    pred_df['true_label'] = y_true
    pred_df['predicted_probability'] = y_pred_prob
    pred_df['predicted_label'] = y_pred_class
    pred_df['correct'] = (y_true == y_pred_class).astype(int)
    pred_df.to_csv(output_dir / f"{model_name}_predictions.csv", index=False)
    print(f"预测结果已保存：{output_dir / f'{model_name}_predictions.csv'}")

    # 分类报告
    report = classification_report(y_true, y_pred_class, target_names=['Non-toxic', 'Toxic'])
    with open(output_dir / f"{model_name}_classification_report.txt", 'w') as f:
        f.write(report)
    print(f"分类报告已保存：{output_dir / f'{model_name}_classification_report.txt'}")

    return metrics


# -------------------------- 8. 主流程 --------------------------
def main():
    print("=" * 60)
    print("Chemprop 药物神经发育毒性预测模型训练")
    print(f"数据文件：{DATA_PATH}")
    print(f"训练/测试划分：训练集 = {int((1 - TEST_SIZE) * 100)}% / 测试集 = {int(TEST_SIZE * 100)}%")
    print("=" * 60)

    # 加载并划分数据
    print("\n[1/5] 加载并划分数据...")
    train_dps, test_dps, train_df, test_df = load_and_split_data(
        DATA_PATH,
        test_size=TEST_SIZE,
        random_state=RANDOM_SEED
    )

    # 构建DataLoader
    print("\n[2/5] 构建DataLoader...")
    train_loader, test_loader = build_dataloaders(train_dps, test_dps, BATCH_SIZE)

    # 构建模型
    print("\n[3/5] 构建Chemprop模型...")
    model = build_default_model()

    # 自定义回调
    metrics_callback = FullMetricsCallback(train_loader, test_loader)

    # 早停回调
    early_stop = pl.callbacks.EarlyStopping(
        monitor=EARLY_STOPPING_MONITOR,
        mode=EARLY_STOPPING_MODE,
        patience=EARLY_STOPPING_PATIENCE,
        verbose=True
    )
    # 模型保存回调
    checkpoint = pl.callbacks.ModelCheckpoint(
        dirpath=CHECKPOINT_DIR,
        filename="best_model-{epoch:02d}-{val/roc:.4f}",
        monitor=EARLY_STOPPING_MONITOR,
        mode=EARLY_STOPPING_MODE,
        save_top_k=1,
        save_weights_only=True,
        verbose=True
    )

    # 训练器
    trainer = pl.Trainer(
        logger=pl.loggers.CSVLogger(save_dir=OUTPUT_DIR, name="hf training_logs"),
        enable_checkpointing=True,
        enable_progress_bar=True,
        accelerator="cpu",
        devices=1,
        max_epochs=MAX_EPOCHS,
        callbacks=[early_stop, checkpoint, metrics_callback]
    )

    # 开始训练
    print("\n[4/5] 开始训练...")
    trainer.fit(model, train_loader, test_loader)

    # 绘制训练历史曲线
    print("\n[5/5] 生成训练历史曲线...")
    plot_training_history(metrics_callback.history, OUTPUT_DIR / "hf training_history.png", network_arch="Chemprop MPNN")

    # 加载最佳模型
    print("\n加载最佳模型并评估...")
    best_model_path = checkpoint.best_model_path
    if best_model_path:
        print(f"最佳模型路径：{best_model_path}")
        best_model = models.MPNN.load_from_checkpoint(best_model_path)
    else:
        print("未找到保存的最佳模型，使用训练结束时的模型")
        best_model = model

    # 评估最佳模型
    metrics = evaluate_model(best_model, test_loader, test_df, OUTPUT_DIR, model_name="hf Best_Model")

    # 保存训练配置信息
    config_info = {
        "data_path": DATA_PATH,
        "test_size": TEST_SIZE,
        "random_seed": RANDOM_SEED,
        "max_epochs": MAX_EPOCHS,
        "batch_size": BATCH_SIZE,
        "early_stopping_monitor": EARLY_STOPPING_MONITOR,
        "early_stopping_patience": EARLY_STOPPING_PATIENCE,
        "model_architecture": "Chemprop MPNN (default params)"
    }
    with open(OUTPUT_DIR / "hf training_config.json", "w") as f:
        json.dump(config_info, f, indent=4)

    # 最终报告
    report = f"""
    ===============================================
    药物神经发育毒性预测 Chemprop 模型训练完成报告
    ===============================================

    数据来源：{DATA_PATH}

    训练/测试划分：
    ----------
    训练集样本数：{len(train_dps)}
    测试集样本数：{len(test_dps)}
    测试集比例：{TEST_SIZE * 100}%

    训练配置：
    ----------
    最大训练轮数：{MAX_EPOCHS}
    早停轮数：{EARLY_STOPPING_PATIENCE}
    最佳轮次依据：测试集 {EARLY_STOPPING_MONITOR}

    测试集最终表现（最佳模型）：
    ----------
    Accuracy:  {metrics['Accuracy']:.4f}
    ROC-AUC:   {metrics['ROC-AUC']:.4f}
    F1-Score:  {metrics['F1-Score']:.4f}
    Precision: {metrics['Precision']:.4f}
    Recall:    {metrics['Recall']:.4f}
    Specificity: {metrics['Specificity']:.4f}

    输出文件：
    ----------
    - 训练集/测试集划分: train_split.csv / test_split.csv
    - 训练历史曲线: training_history.png / training_history.csv
    - 最佳模型: {best_model_path}
    - 测试集预测CSV: Best_Model_predictions.csv
    - 混淆矩阵图: Best_Model_confusion_matrix.png
    - ROC曲线图: Best_Model_roc_curve.png
    - 评估指标: Best_Model_metrics.json/csv
    - 分类报告: Best_Model_classification_report.txt

    所有结果保存在：{OUTPUT_DIR}
    ===============================================
    """
    with open(OUTPUT_DIR / "hf final_report.txt", "w") as f:
        f.write(report)
    print(report)


if __name__ == "__main__":
    if not os.path.exists(DATA_PATH):
        print(f"错误：数据文件不存在 {DATA_PATH}")
    else:
        main()
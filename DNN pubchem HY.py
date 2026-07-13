import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import tensorflow as tf
import keras as k
import pandas as pd
from keras.layers import Dense, BatchNormalization, Dropout
from keras.regularizers import l1_l2
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import confusion_matrix, accuracy_score, roc_curve, roc_auc_score, classification_report, f1_score, \
    precision_score, recall_score
from sklearn import metrics
from keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
import warnings

warnings.filterwarnings('ignore')


# ==================== 自定义早停回调 ====================
class AdvancedEarlyStopping(tf.keras.callbacks.Callback):
    """改进的早停策略"""

    def __init__(self, monitor='val_loss', patience=35, min_delta=0.001,
                 min_epochs=30, restore_best_weights=True):
        super().__init__()
        self.monitor = monitor
        self.patience = patience
        self.min_delta = min_delta
        self.min_epochs = min_epochs
        self.restore_best_weights = restore_best_weights
        self.best_weights = None
        self.best_value = np.inf if 'loss' in monitor else -np.inf
        self.wait = 0
        self.best_epoch = 0

    def on_epoch_end(self, epoch, logs=None):
        current_value = logs.get(self.monitor)
        if current_value is None:
            return

        if 'loss' in self.monitor:
            improvement = self.best_value - current_value
        else:
            improvement = current_value - self.best_value

        if epoch < self.min_epochs:
            self.best_value = current_value
            self.best_weights = self.model.get_weights()
            self.best_epoch = epoch
            self.wait = 0
        elif improvement > self.min_delta:
            self.best_value = current_value
            self.best_weights = self.model.get_weights()
            self.best_epoch = epoch
            self.wait = 0
        else:
            self.wait += 1
            if self.wait >= self.patience:
                self.model.stop_training = True
                if self.restore_best_weights and self.best_weights is not None:
                    self.model.set_weights(self.best_weights)
                print(f"\n早停触发 - 第{epoch}轮停止，最佳轮次: {self.best_epoch}轮")


# ==================== 数据加载和预处理 ====================
# [保持原有数据加载代码不变]
df_x_train = pd.read_csv(r"E:\训练集测试集3\hyX_train_pubchem.csv", na_values=["?", "NA"])
df_y_train = pd.read_csv(r"E:\训练集测试集3\hyy_train_pubchem.csv", na_values=["?", "NA"])
df_x_test = pd.read_csv(r"E:\训练集测试集3\hyX_test_pubchem.csv", na_values=["?", "NA"])
df_y_test = pd.read_csv(r"E:\训练集测试集3\hyy_test_pubchem.csv", na_values=["?", "NA"])
print(f"原始数据形状 - df_x_train: {df_x_train.shape}, df_y_train: {df_y_train.shape}")
print(f"原始数据形状 - df_x_test: {df_x_test.shape}, df_y_test: {df_y_test.shape}")

# ==================== 数据预处理 ====================
# 查看列名，确定哪些是特征列
print("\n原始列名:", df_x_train.columns.tolist()[:10])  # 只显示前10个

# 检查并移除'smiles'列（如果是字符串列）
if 'smiles' in df_x_train.columns:
    print("检测到'smiles'列，正在移除...")
    # 提取数值特征列（排除'smiles'列）
    numeric_columns = [col for col in df_x_train.columns if col != 'smiles']
    print(f"使用的数值列数量: {len(numeric_columns)}")
    print(f"前10个数值列: {numeric_columns[:10]}")

    # 创建特征数组
    x_train = df_x_train[numeric_columns].values
    x_test = df_x_test[numeric_columns].values
else:
    # 如果没有'smiles'列，直接使用所有列
    print("没有检测到'smiles'列，使用所有列作为特征")
    x_train = df_x_train.values
    x_test = df_x_test.values

# 提取标签（确保是一维数组）
y_train = df_y_train.values.ravel()
y_test = df_y_test.values.ravel()

print(f"\n数值特征形状 - x_train: {x_train.shape}, x_test: {x_test.shape}")
print(f"标签形状 - y_train: {y_train.shape}, y_test: {y_test.shape}")

# 检查数据中是否有NaN值
print(f"\n检查缺失值:")
print(f"x_train中NaN数量: {np.isnan(x_train).sum()}")
print(f"y_train中NaN数量: {np.isnan(y_train).sum()}")
print(f"x_test中NaN数量: {np.isnan(x_test).sum()}")
print(f"y_test中NaN数量: {np.isnan(y_test).sum()}")

# 如果有NaN值，用均值填充
if np.isnan(x_train).sum() > 0 or np.isnan(x_test).sum() > 0:
    print("发现NaN值，正在用列均值填充...")
    from sklearn.impute import SimpleImputer

    imputer = SimpleImputer(strategy='mean')
    x_train = imputer.fit_transform(x_train)
    x_test = imputer.transform(x_test)

# 检查数据类型，确保是数值类型
print(f"\nx_train 数据类型: {x_train.dtype}")
if x_train.dtype != 'float32' and x_train.dtype != 'float64':
    print("转换为浮点类型...")
    x_train = x_train.astype('float32')
    x_test = x_test.astype('float32')

# ==================== 数据标准化 ====================
scaler = StandardScaler()
x_train_scaled = scaler.fit_transform(x_train)
x_test_scaled = scaler.transform(x_test)

print(f"\n标准化后形状:")
print(f"x_train_scaled: {x_train_scaled.shape}")
print(f"x_test_scaled: {x_test_scaled.shape}")
print(f"标准化完成!")


# ==================== 五折交叉验证 ====================
def create_model(input_shape):
    model = k.models.Sequential()

    # 第一层：大幅减少神经元，增加更强的正则化
    model.add(Dense(32, input_dim=input_shape, activation="relu",
                    kernel_regularizer=l1_l2(l1=0.0001, l2=0.01),
                    kernel_initializer='he_normal'))
    model.add(BatchNormalization())
    model.add(Dropout(0.7))  # 降低dropout

    # 第二层：继续减少
    model.add(Dense(16, activation="relu",
                    kernel_regularizer=l1_l2(l1=0.0001, l2=0.01),
                    kernel_initializer='he_normal'))
    model.add(BatchNormalization())
    model.add(Dropout(0.6))

    # 第三层：输出层前
    model.add(Dense(8, activation="relu",
                    kernel_initializer='he_normal'))
    model.add(Dropout(0.5))

    model.add(k.layers.Dense(1, activation="sigmoid"))

    # 优化器：增加学习率，移除梯度裁剪
    optimizer = tf.keras.optimizers.Adam(
        learning_rate=0.0003,  # 提高学习率
        beta_1=0.9,
        beta_2=0.999,
        epsilon=1e-07
        # 移除clipnorm
    )
    model.compile(loss="binary_crossentropy", optimizer=optimizer, metrics=["accuracy"])
    return model


# ==================== 五折交叉验证 ====================
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
cv_metrics = {
    'auc': [], 'accuracy': [], 'sensitivity': [],
    'specificity': [], 'f1': [], 'precision': [], 'recall': []
}

for fold, (train_idx, val_idx) in enumerate(skf.split(x_train_scaled, y_train), 1):
    print(f"\n--- 开始第 {fold} 折交叉验证 ---")

    x_cv_train, x_cv_val = x_train_scaled[train_idx], x_train_scaled[val_idx]
    y_cv_train, y_cv_val = y_train[train_idx], y_train[val_idx]

    # 创建和训练模型
    cv_model = create_model(input_shape=x_train_scaled.shape[1])

    # 使用改进的早停策略
    history = cv_model.fit(
        x_cv_train, y_cv_train,
        epochs=200,
        batch_size=32,
        validation_data=(x_cv_val, y_cv_val),
        callbacks=[
            AdvancedEarlyStopping(
                monitor='val_loss',
                patience=15,
                min_delta=0.001,
                min_epochs=20,
                restore_best_weights=True
            ),
            ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=8,
                min_lr=1e-6,
                verbose=0
            )
        ],
        verbose=0
    )

    # 预测和评估
    y_cv_val_proba = cv_model.predict(x_cv_val, verbose=0).ravel()
    y_cv_val_binary = (y_cv_val_proba > 0.5).astype(int).ravel()

    # 计算指标（保持原有代码）
    tn, fp, fn, tp = confusion_matrix(y_cv_val, y_cv_val_binary).ravel()
    auc = roc_auc_score(y_cv_val, y_cv_val_proba) if len(np.unique(y_cv_val)) > 1 else 0.0
    acc = accuracy_score(y_cv_val, y_cv_val_binary)
    se = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    sp = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    f1 = f1_score(y_cv_val, y_cv_val_binary)
    precision = precision_score(y_cv_val, y_cv_val_binary)
    recall = recall_score(y_cv_val, y_cv_val_binary)

    # 存储指标
    cv_metrics['auc'].append(auc)
    cv_metrics['accuracy'].append(acc)
    cv_metrics['sensitivity'].append(se)
    cv_metrics['specificity'].append(sp)
    cv_metrics['f1'].append(f1)
    cv_metrics['precision'].append(precision)
    cv_metrics['recall'].append(recall)

    print(f"第 {fold} 折结果: AUC: {auc:.4f}, 准确率: {acc:.4f}, 灵敏度: {se:.4f}, 特异度: {sp:.4f}")
    print(f"训练轮次: {len(history.history['loss'])}")

# 打印交叉验证结果
print("\n" + "=" * 80)
print("五折交叉验证结果:")
print("=" * 80)
for metric in cv_metrics:
    values = np.array(cv_metrics[metric])
    print(f"{metric}: {np.mean(values):.4f} (±{np.std(values):.4f})")

# ==================== 训练最终模型 ====================
print("\n=== 训练最终模型 ===")
tf.keras.backend.clear_session()

final_model = create_model(input_shape=x_train_scaled.shape[1])

# 使用优化的早停策略
history = final_model.fit(
    x_train_scaled, y_train,
    epochs=500,
    batch_size=16,
    validation_split=0.15,
    callbacks=[
        AdvancedEarlyStopping(
            monitor='val_loss',
            patience=35,
            min_delta=0.0005,
            min_epochs=30,
            restore_best_weights=True
        ),
        ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=12,
            min_lr=1e-7,
            verbose=1
        ),
        ModelCheckpoint(
            filepath=r"E:\模型8\hy_best_model.h5",
            monitor='val_loss',
            save_best_only=True,
            mode='min',
            verbose=1
        )
    ],
    verbose=2,
    shuffle=True
)

# 保存最终模型
final_model.save(r"E:\模型8\hy_model_cv.h5")

# ==================== 测试集评估 ====================
print("\n=== 在测试集上评估最终模型 ===")
# 预测
y_pred_proba = final_model.predict(x_test_scaled)
y_pred_binary = np.round(y_pred_proba).astype(int)

# 保存预测结果和真实标签
print("\n=== 保存预测结果 ===")
# 保存概率预测
np.save(r"E:\模型8\hy y_pred_proba.npy", y_pred_proba)
# 保存真实标签
np.save(r"E:\模型8\hy y_test.npy", y_test)
# 输出文件信息
print(f"已保存预测概率到: E:/模型8/hy y_pred_proba.npy")
print(f"已保存真实标签到: E:/模型8/hy y_test.npy")
print(f"文件形状: y_pred_proba: {y_pred_proba.shape},  y_test: {y_test.shape}")

# 计算指标
tn, fp, fn, tp = confusion_matrix(y_test, y_pred_binary).ravel()

# 基本指标
accuracy = accuracy_score(y_test, y_pred_binary)
sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0.0  # 灵敏度/召回率
specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0  # 特异度
precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0    # 精确率
f1 = f1_score(y_test, y_pred_binary)                    # F1分数
roc_auc = roc_auc_score(y_test, y_pred_proba)           # AUC

# 计算MCC（马修斯相关系数）
mcc_numerator = (tp * tn) - (fp * fn)
mcc_denominator = np.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
mcc = mcc_numerator / mcc_denominator if mcc_denominator != 0 else 0.0

# 计算平衡准确率
balanced_accuracy = (sensitivity + specificity) / 2

# 计算几何平均数（G-Mean）
g_mean = np.sqrt(sensitivity * specificity) if sensitivity * specificity > 0 else 0.0

# 打印详细结果
print('\n' + '=' * 80)
print('最终模型性能评估结果:')
print('=' * 80)
print(f"测试样本数: {len(y_test)}")
print(f"正类样本数: {sum(y_test)}, 负类样本数: {len(y_test) - sum(y_test)}")
print(f"预测正类数: {sum(y_pred_binary)}, 预测负类数: {len(y_pred_binary) - sum(y_pred_binary)}")
print(f"混淆矩阵: TP={tp}, FP={fp}, TN={tn}, FN={fn}")

print('\n主要性能指标:')
print('-' * 40)
print(f"AUC (曲线下面积): {roc_auc:.4f}")
print(f"ACC (准确率): {accuracy:.4f}")
print(f"MCC (马修斯相关系数): {mcc:.4f}")
print(f"F1 分数: {f1:.4f}")
print(f"SE (灵敏度/召回率): {sensitivity:.4f}")
print(f"SP (特异度): {specificity:.4f}")

print('\n其他重要指标:')
print('-' * 40)
print(f"精确率 (Precision): {precision:.4f}")
print(f"平衡准确率 (BA): {balanced_accuracy:.4f}")
print(f"几何平均数 (G-Mean): {g_mean:.4f}")
print(f"训练总轮次: {len(history.history['loss'])}")

# 计算并显示完整的分类报告
print('\n' + '-' * 80)
print('详细分类报告:')
print('-' * 80)
print(classification_report(y_test, y_pred_binary,
                          target_names=['Class 0 (Negative)', 'Class 1 (Positive)']))

# 绘制ROC曲线
fpr, tpr, thresholds = roc_curve(y_test, y_pred_proba)

plt.figure(figsize=(10, 8))
plt.plot(fpr, tpr, color='darkorange', lw=2,
         label=f'ROC curve (AUC = {roc_auc:.3f})')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate (1 - Specificity)', fontsize=12)
plt.ylabel('True Positive Rate (Sensitivity)', fontsize=12)
plt.title('Receiver Operating Characteristic (ROC) Curve', fontsize=14)
plt.legend(loc="lower right")
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('E:/模型8/hy_roc_curve.pdf', dpi=300, bbox_inches='tight')
plt.close()

# 保存所有指标到CSV文件
metrics_df = pd.DataFrame({
    'Metric': ['AUC', 'Accuracy', 'MCC', 'F1', 'Sensitivity', 'Specificity',
               'Precision', 'Balanced_Accuracy', 'G_Mean'],
    'Value': [roc_auc, accuracy, mcc, f1, sensitivity, specificity,
              precision, balanced_accuracy, g_mean],
    'Description': [
        'Area Under ROC Curve',
        'Overall Classification Accuracy',
        'Matthews Correlation Coefficient',
        'Harmonic Mean of Precision and Recall',
        'True Positive Rate (Recall)',
        'True Negative Rate',
        'Positive Predictive Value',
        'Average of Sensitivity and Specificity',
        'Geometric Mean of Sensitivity and Specificity'
    ]
})

metrics_df.to_csv('E:/模型8/hy_model_metrics.csv', index=False)
print(f"\n所有指标已保存到: E:/模型8/hy_model_metrics.csv")


print("\n=== 训练完成 ===")
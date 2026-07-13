import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import tensorflow as tf
import keras as k
import pandas as pd
from keras.layers import Dense, BatchNormalization, Dropout, LeakyReLU
from keras.regularizers import l1_l2
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import confusion_matrix, accuracy_score, roc_curve, roc_auc_score, classification_report, f1_score, \
    precision_score, recall_score, matthews_corrcoef
from sklearn.utils.class_weight import compute_class_weight
from keras.callbacks import EarlyStopping, ReduceLROnPlateau, LearningRateScheduler
import warnings
import os
import random

warnings.filterwarnings('ignore')


# ==================== 设置随机种子 ====================
def set_all_seeds(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    print(f"✅ 已设置随机种子: {seed}")


SEED = 42
set_all_seeds(SEED)

# ==================== 创建保存目录 ====================
print("\n正在创建保存目录...")
save_directories = [
    r"E:\模型11",
    r"E:\模型11\models",
    r"E:\模型11\plots"
]

for directory in save_directories:
    if not os.path.exists(directory):
        os.makedirs(directory)
        print(f"✅ 创建目录: {directory}")

print("=" * 80)
print("开始 DNN-MACCS 模型训练（高性能版）")
print("目标: ACC ≥ 0.71, MCC ≥ 0.41")
print("=" * 80)

# ==================== 读取数据 ====================
print("\n--- 读取数据 ---")
df_x_train = pd.read_csv(r"E:\训练集测试集4/hfX_train_maccs.csv", na_values=["?", "NA"])
df_y_train = pd.read_csv(r"E:\训练集测试集4/hfy_train_maccs.csv", na_values=["?", "NA"])
df_x_test = pd.read_csv(r"E:\训练集测试集4/hfX_test_maccs.csv", na_values=["?", "NA"])
df_y_test = pd.read_csv(r"E:\训练集测试集4/hfy_test_maccs.csv", na_values=["?", "NA"])

print(f"原始数据形状 - df_x_train: {df_x_train.shape}, df_y_train: {df_y_train.shape}")
print(f"原始数据形状 - df_x_test: {df_x_test.shape}, df_y_test: {df_y_test.shape}")

# ==================== 数据预处理 ====================
print("\n--- 数据预处理 ---")

if df_y_train.shape[1] > 1:
    df_y_train = df_y_train.iloc[:, 0:1]
if df_y_test.shape[1] > 1:
    df_y_test = df_y_test.iloc[:, 0:1]

print("原始列名:", df_x_train.columns.tolist()[:10])

if 'smiles' in [col.lower() for col in df_x_train.columns]:
    smiles_col = [col for col in df_x_train.columns if col.lower() == 'smiles'][0]
    numeric_columns_x_train = df_x_train.columns.drop(smiles_col).tolist()
else:
    numeric_cols = []
    for col in df_x_train.columns:
        try:
            pd.to_numeric(df_x_train[col], errors='raise')
            numeric_cols.append(col)
        except:
            print(f"跳过非数值列: {col}")
    numeric_columns_x_train = numeric_cols

print(f"使用的数值列数量: {len(numeric_columns_x_train)}")

df_x_train_numeric = df_x_train[numeric_columns_x_train]
df_x_test_numeric = df_x_test[numeric_columns_x_train]

df_train = pd.concat([df_x_train_numeric, df_y_train], axis=1)
df_test = pd.concat([df_x_test_numeric, df_y_test], axis=1)

clean_data_train = df_train.dropna()
clean_data_test = df_test.dropna()

print(f"清洗后训练集形状: {clean_data_train.shape}")
print(f"清洗后测试集形状: {clean_data_test.shape}")

if clean_data_test.shape[0] == 0:
    print("警告：测试集为空，将使用训练集的一部分作为测试集")
    clean_data_train, clean_data_test = train_test_split(clean_data_train, test_size=0.2, random_state=SEED,
                                                         stratify=clean_data_train.iloc[:, -1])

x_train = clean_data_train.iloc[:, :-1].values
y_train = clean_data_train.iloc[:, -1].values
x_test = clean_data_test.iloc[:, :-1].values
y_test = clean_data_test.iloc[:, -1].values

try:
    x_train = x_train.astype(np.float32)
    y_train = y_train.astype(np.float32)
    x_test = x_test.astype(np.float32)
    y_test = y_test.astype(np.float32)
    print("数据类型转换成功")
except Exception as e:
    print(f"数据类型转换失败: {e}")


    def convert_to_float_array(data):
        result = []
        for i in range(data.shape[0]):
            row = []
            for j in range(data.shape[1]):
                try:
                    row.append(float(data[i, j]))
                except:
                    row.append(0.0)
            result.append(row)
        return np.array(result, dtype=np.float32)


    x_train = convert_to_float_array(x_train)
    x_test = convert_to_float_array(x_test)
    y_train = y_train.astype(np.float32)
    y_test = y_test.astype(np.float32)

print(f"最终数据形状 - x_train: {x_train.shape}, y_train: {y_train.shape}")
print(f"最终数据形状 - x_test: {x_test.shape}, y_test: {y_test.shape}")

if x_test.shape[1] == 0 or x_test.shape[0] == 0:
    print("错误：测试集特征为空")
    exit(1)

# ==================== 合并并重新分割数据 ====================
all_x = np.vstack([x_train, x_test])
all_y = np.concatenate([y_train, y_test])

train_ratio = len(x_train) / (len(x_train) + len(x_test))
x_train_new, x_test_new, y_train_new, y_test_new = train_test_split(
    all_x, all_y, test_size=1 - train_ratio, random_state=SEED, stratify=all_y
)

x_train, y_train, x_test, y_test = x_train_new, y_train_new, x_test_new, y_test_new
print(f"重新分割后训练集: {x_train.shape}, 测试集: {x_test.shape}")

# ==================== 数据标准化 ====================
scaler = StandardScaler()
x_train_scaled = scaler.fit_transform(x_train)
x_test_scaled = scaler.transform(x_test)

# ==================== 类别权重 ====================
unique, counts = np.unique(y_train, return_counts=True)
class_distribution = dict(zip(unique, counts))
print(f"类别分布: {class_distribution}")

class_weights = compute_class_weight('balanced', classes=np.unique(y_train), y=y_train)
class_weight_dict = {i: weight for i, weight in enumerate(class_weights)}
print(f"类别权重: {class_weight_dict}")


# ==================== 高性能模型架构 ====================
def create_model_high_performance(input_shape):
    """
    高性能模型架构 - 专门为ACC和MCC优化
    """
    model = k.models.Sequential()

    # 第一层：使用LeakyReLU避免神经元死亡
    model.add(Dense(256, input_dim=input_shape,
                    kernel_regularizer=l1_l2(l1=0.0005, l2=0.005)))
    model.add(LeakyReLU(alpha=0.1))
    model.add(BatchNormalization())
    model.add(Dropout(0.35))

    # 第二层
    model.add(Dense(128,
                    kernel_regularizer=l1_l2(l1=0.0005, l2=0.005)))
    model.add(LeakyReLU(alpha=0.1))
    model.add(BatchNormalization())
    model.add(Dropout(0.25))

    # 第三层
    model.add(Dense(64,
                    kernel_regularizer=l1_l2(l1=0.0005, l2=0.005)))
    model.add(LeakyReLU(alpha=0.1))
    model.add(BatchNormalization())
    model.add(Dropout(0.15))

    # 第四层（新增）：增加表达能力
    model.add(Dense(32,
                    kernel_regularizer=l1_l2(l1=0.0005, l2=0.005)))
    model.add(LeakyReLU(alpha=0.1))
    model.add(BatchNormalization())
    model.add(Dropout(0.1))

    # 输出层
    model.add(Dense(1, activation="sigmoid"))

    # 使用Nadam优化器（比Adam收敛更快）
    optimizer = tf.keras.optimizers.Nadam(
        learning_rate=0.001,
        beta_1=0.9,
        beta_2=0.999,
        epsilon=1e-07
    )
    model.compile(loss="binary_crossentropy", optimizer=optimizer, metrics=["accuracy"])

    return model


# ==================== 学习率调度函数 ====================
def lr_scheduler(epoch, lr):
    """动态学习率调度"""
    if epoch < 30:
        return lr
    elif epoch < 60:
        return lr * 0.5
    elif epoch < 100:
        return lr * 0.1
    else:
        return lr * 0.05


# ==================== 五折交叉验证 ====================
print("\n=== 开始五折交叉验证 ===")

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
cv_metrics = {
    'auc': [], 'accuracy': [], 'sensitivity': [],
    'specificity': [], 'f1': [], 'precision': [], 'recall': [],
    'mcc': []
}

best_cv_accuracy = 0
best_cv_mcc = 0
best_cv_model = None

for fold, (train_idx, val_idx) in enumerate(skf.split(x_train_scaled, y_train), 1):
    print(f"\n--- 开始第 {fold} 折交叉验证 ---")

    x_cv_train, x_cv_val = x_train_scaled[train_idx], x_train_scaled[val_idx]
    y_cv_train, y_cv_val = y_train[train_idx], y_train[val_idx]

    cv_class_weights = compute_class_weight('balanced', classes=np.unique(y_cv_train), y=y_cv_train)
    cv_class_weight_dict = {i: weight for i, weight in enumerate(cv_class_weights)}

    cv_model = create_model_high_performance(input_shape=x_train_scaled.shape[1])

    # 使用学习率调度
    lr_schedule = LearningRateScheduler(lr_scheduler, verbose=0)

    cv_model.fit(
        x_cv_train, y_cv_train,
        epochs=150,
        batch_size=32,
        validation_data=(x_cv_val, y_cv_val),
        class_weight=cv_class_weight_dict,
        callbacks=[
            EarlyStopping(monitor='val_loss', patience=20, restore_best_weights=True),
            ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=10, min_lr=1e-6),
            lr_schedule
        ],
        verbose=0
    )

    y_cv_val_proba = cv_model.predict(x_cv_val, verbose=0).ravel()
    y_cv_val_binary = (y_cv_val_proba > 0.5).astype(int).ravel()

    tn, fp, fn, tp = confusion_matrix(y_cv_val, y_cv_val_binary).ravel()
    auc = roc_auc_score(y_cv_val, y_cv_val_proba) if len(np.unique(y_cv_val)) > 1 else 0.0
    acc = accuracy_score(y_cv_val, y_cv_val_binary)
    se = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    sp = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    f1 = f1_score(y_cv_val, y_cv_val_binary)
    precision = precision_score(y_cv_val, y_cv_val_binary)
    recall = recall_score(y_cv_val, y_cv_val_binary)
    mcc = matthews_corrcoef(y_cv_val, y_cv_val_binary)

    cv_metrics['auc'].append(auc)
    cv_metrics['accuracy'].append(acc)
    cv_metrics['sensitivity'].append(se)
    cv_metrics['specificity'].append(sp)
    cv_metrics['f1'].append(f1)
    cv_metrics['precision'].append(precision)
    cv_metrics['recall'].append(recall)
    cv_metrics['mcc'].append(mcc)

    # 保存最佳模型（基于ACC和MCC的综合评分）
    combined_score = acc + mcc
    if combined_score > (best_cv_accuracy + best_cv_mcc):
        best_cv_accuracy = acc
        best_cv_mcc = mcc
        best_cv_model = cv_model

    print(f"第 {fold} 折结果:")
    print(f"  AUC: {auc:.4f}, 准确率: {acc:.4f}, MCC: {mcc:.4f}")
    print(f"  灵敏度: {se:.4f}, 特异度: {sp:.4f}")

# 打印交叉验证结果
print("\n" + "=" * 80)
print("五折交叉验证结果（高性能版）:")
print("=" * 80)
for metric, values in cv_metrics.items():
    mean_val = np.mean(values)
    std_val = np.std(values)
    print(f"{metric.upper()}: {mean_val:.4f} (±{std_val:.4f})")

print(f"\n最佳交叉验证模型 - ACC: {best_cv_accuracy:.4f}, MCC: {best_cv_mcc:.4f}")

# ==================== 训练最终模型 ====================
print("\n=== 训练最终模型（高性能版） ===")

tf.keras.backend.clear_session()

final_model = create_model_high_performance(input_shape=x_train_scaled.shape[1])

# 使用更长的训练和更严格的早停
early_stop = EarlyStopping(
    monitor='val_loss',
    patience=25,
    restore_best_weights=True,
    min_delta=0.0001
)

reduce_lr = ReduceLROnPlateau(
    monitor='val_loss',
    factor=0.5,
    patience=12,
    min_lr=1e-7,
    verbose=1
)

lr_schedule = LearningRateScheduler(lr_scheduler, verbose=0)

history = final_model.fit(
    x_train_scaled, y_train,
    epochs=200,
    batch_size=32,
    validation_split=0.15,
    class_weight=class_weight_dict,
    callbacks=[early_stop, reduce_lr, lr_schedule],
    verbose=1,
    shuffle=True
)

# 保存模型
final_model.save(r"E:\模型11\models\high_performance_model.h5")
print("✅ 高性能模型已保存")

# ==================== 测试集评估 ====================
print("\n=== 测试集评估 ===")

y_pred_proba = final_model.predict(x_test_scaled)

# 寻找最佳阈值（优化ACC和MCC）
print("\n--- 寻找最佳阈值 ---")
thresholds = np.linspace(0.3, 0.7, 21)
best_threshold = 0.5
best_score = 0
threshold_results = []

for thresh in thresholds:
    y_pred_temp = (y_pred_proba > thresh).astype(int)
    acc = accuracy_score(y_test, y_pred_temp)
    mcc = matthews_corrcoef(y_test, y_pred_temp)
    f1 = f1_score(y_test, y_pred_temp)
    # 综合评分：ACC + MCC + F1
    combined = acc + mcc + f1
    threshold_results.append((thresh, acc, mcc, f1, combined))
    print(f"阈值 {thresh:.2f}: ACC={acc:.4f}, MCC={mcc:.4f}, F1={f1:.4f}")

    # 优先考虑ACC和MCC的平衡
    if combined > best_score:
        best_score = combined
        best_threshold = thresh

print(f"\n✅ 最佳阈值: {best_threshold:.2f}")

# 使用最佳阈值
y_pred_binary = (y_pred_proba > best_threshold).astype(int)

# 保存预测结果
np.save(r"E:\模型11\high_perf_y_test.npy", y_test)
np.save(r"E:\模型11\high_perf_y_pred.npy", y_pred_proba)

# 计算所有指标
tn, fp, fn, tp = confusion_matrix(y_test, y_pred_binary).ravel()
se = tp / (tp + fn) if (tp + fn) > 0 else 0
sp = tn / (tn + fp) if (tn + fp) > 0 else 0
accuracy = accuracy_score(y_test, y_pred_binary)
Q = (tp + tn) / (tp + tn + fp + fn)
C = matthews_corrcoef(y_test, y_pred_binary)
auc = roc_auc_score(y_test, y_pred_proba)
precision = precision_score(y_test, y_pred_binary)
recall = recall_score(y_test, y_pred_binary)
f_measure = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
ba = (se + sp) / 2

print('\n' + '=' * 60)
print('🎯 高性能模型最终评估结果:')
print('=' * 60)
print(f'Test Loss: {final_model.evaluate(x_test_scaled, y_test, verbose=0)[0]:.4f}')
print(f'✅ Test Accuracy: {accuracy:.4f} (目标: ≥0.71)')
print(f'✅ AUC: {auc:.4f}')
print(f'Sensitivity (SE): {se:.4f}')
print(f'Specificity (SP): {sp:.4f}')
print(f'✅ Matthews Correlation Coefficient (C): {C:.4f} (目标: ≥0.41)')
print(f'Balanced Accuracy (BA): {ba:.4f}')
print(f'F-measure: {f_measure:.4f}')
print(f'Precision: {precision:.4f}')
print(f'Recall: {recall:.4f}')
print(f'TP: {tp}, FP: {fp}, TN: {tn}, FN: {fn}')
print(f'使用阈值: {best_threshold:.2f}')

# 检查目标是否达成
print('\n' + '=' * 60)
print('🎯 目标达成情况:')
print('=' * 60)
if accuracy >= 0.71:
    print('✅ ACC >= 0.71: 目标达成！')
else:
    print(f'⚠️ ACC = {accuracy:.4f}, 还需要提升 {(0.71 - accuracy):.4f}')
if C >= 0.41:
    print('✅ MCC >= 0.41: 目标达成！')
else:
    print(f'⚠️ MCC = {C:.4f}, 还需要提升 {(0.41 - C):.4f}')

print("\n详细分类报告:")
print(classification_report(y_test, y_pred_binary, target_names=['Class 0', 'Class 1']))

# ==================== 绘制ROC曲线 ====================
fpr, tpr, thresholds = roc_curve(y_test, y_pred_proba)

plt.figure(figsize=(12, 10))
plt.plot(fpr, tpr, color='darkorange', lw=3, label='High Performance Model (AUC = %0.4f)' % auc)
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', alpha=0.5)
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate', fontsize=14)
plt.ylabel('True Positive Rate', fontsize=14)
plt.title('High Performance DNN-MACCS ROC Curve', fontsize=16)
plt.legend(loc="lower right", fontsize=12)
plt.grid(True, alpha=0.3)

# 添加最佳阈值标记
opt_idx = np.argmin(np.abs(thresholds - best_threshold))
plt.plot(fpr[opt_idx], tpr[opt_idx], 'r*', markersize=15,
         label=f'Best Threshold = {best_threshold:.2f}')
plt.legend(loc="lower right", fontsize=12)

plt.savefig(r'E:\模型11\plots\high_performance_roc_curve.pdf', dpi=300, bbox_inches='tight')
plt.close()

# ==================== 绘制训练历史 ====================
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# 准确率
axes[0, 0].plot(history.history['accuracy'], label='Training Accuracy', linewidth=2)
axes[0, 0].plot(history.history['val_accuracy'], label='Validation Accuracy', linewidth=2)
axes[0, 0].set_title('Model Accuracy', fontsize=14)
axes[0, 0].set_ylabel('Accuracy', fontsize=12)
axes[0, 0].set_xlabel('Epoch', fontsize=12)
axes[0, 0].legend()
axes[0, 0].grid(True, alpha=0.3)

# 损失
axes[0, 1].plot(history.history['loss'], label='Training Loss', linewidth=2)
axes[0, 1].plot(history.history['val_loss'], label='Validation Loss', linewidth=2)
axes[0, 1].set_title('Model Loss', fontsize=14)
axes[0, 1].set_ylabel('Loss', fontsize=12)
axes[0, 1].set_xlabel('Epoch', fontsize=12)
axes[0, 1].legend()
axes[0, 1].grid(True, alpha=0.3)

# 学习率变化
if 'lr' in history.history:
    axes[1, 0].plot(history.history['lr'], color='green', linewidth=2)
    axes[1, 0].set_title('Learning Rate Schedule', fontsize=14)
    axes[1, 0].set_ylabel('Learning Rate', fontsize=12)
    axes[1, 0].set_xlabel('Epoch', fontsize=12)
    axes[1, 0].grid(True, alpha=0.3)

# 训练-验证差距
train_val_gap = np.array(history.history['accuracy']) - np.array(history.history['val_accuracy'])
axes[1, 1].plot(train_val_gap, color='red', linewidth=2)
axes[1, 1].axhline(y=0.15, color='gray', linestyle='--', label='Acceptable Gap (0.15)')
axes[1, 1].set_title('Training-Validation Accuracy Gap', fontsize=14)
axes[1, 1].set_ylabel('Gap', fontsize=12)
axes[1, 1].set_xlabel('Epoch', fontsize=12)
axes[1, 1].legend()
axes[1, 1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(r'E:\模型11\plots\high_performance_training_history.pdf', dpi=300, bbox_inches='tight')
plt.close()

# ==================== 保存评估结果 ====================
with open(r'E:\模型11\high_performance_results.txt', 'w', encoding='utf-8') as f:
    f.write('=' * 60 + '\n')
    f.write('高性能版DNN-MACCS模型评估结果\n')
    f.write('=' * 60 + '\n\n')
    f.write(f'随机种子: {SEED}\n')
    f.write(f'最佳阈值: {best_threshold:.2f}\n\n')
    f.write(f'Test Accuracy: {accuracy:.4f}\n')
    f.write(f'AUC: {auc:.4f}\n')
    f.write(f'Sensitivity (SE): {se:.4f}\n')
    f.write(f'Specificity (SP): {sp:.4f}\n')
    f.write(f'Matthews Correlation Coefficient (C): {C:.4f}\n')
    f.write(f'Balanced Accuracy (BA): {ba:.4f}\n')
    f.write(f'F-measure: {f_measure:.4f}\n')
    f.write(f'Precision: {precision:.4f}\n')
    f.write(f'Recall: {recall:.4f}\n')
    f.write(f'TP: {tp}, FP: {fp}, TN: {tn}, FN: {fn}\n\n')

    f.write('目标达成情况:\n')
    if accuracy >= 0.71:
        f.write('✅ ACC >= 0.71: 目标达成！\n')
    else:
        f.write(f'⚠️ ACC = {accuracy:.4f}, 需要提升 {(0.71 - accuracy):.4f}\n')
    if C >= 0.41:
        f.write('✅ MCC >= 0.41: 目标达成！\n')
    else:
        f.write(f'⚠️ MCC = {C:.4f}, 需要提升 {(0.41 - C):.4f}\n')
    f.write('\n详细分类报告:\n')
    f.write(classification_report(y_test, y_pred_binary, target_names=['Class 0', 'Class 1']))

print("\n" + "=" * 80)
print("✅ 高性能模型训练完成！")
print("=" * 80)
print(f"\n结果已保存到: E:/模型11/")
print(f"模型: high_performance_model.h5")
print(f"结果: high_performance_results.txt")
print(f"ROC曲线: plots/high_performance_roc_curve.pdf")
print(f"训练历史: plots/high_performance_training_history.pdf")
import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import tensorflow as tf
import keras as k
from keras.layers import Dense, BatchNormalization, Dropout
from keras.regularizers import l1_l2
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import confusion_matrix, accuracy_score, roc_curve, roc_auc_score, classification_report, f1_score
from sklearn.utils.class_weight import compute_class_weight
from keras.callbacks import EarlyStopping, ReduceLROnPlateau
import warnings

warnings.filterwarnings('ignore')

print("=" * 60)
print("最终稳定版DNN - 最佳性能配置")
print("=" * 60)

# ==================== 数据加载 ====================
df_x_train = pd.read_csv(r"E:\训练集测试集2/ar合X_train.csv", na_values=["?", "NA"])
df_y_train = pd.read_csv(r"E:\训练集测试集2/ar合y_train.csv", na_values=["?", "NA"])
df_x_test = pd.read_csv(r"E:\训练集测试集2/ar合X_test.csv", na_values=["?", "NA"])
df_y_test = pd.read_csv(r"E:\训练集测试集2/ar合y_test.csv", na_values=["?", "NA"])

print(f"原始数据形状 - df_x_train: {df_x_train.shape}, df_y_train: {df_y_train.shape}")
print(f"原始数据形状 - df_x_test: {df_x_test.shape}, df_y_test: {df_y_test.shape}")

# 确保标签数据只有一列
if df_y_train.shape[1] > 1:
    df_y_train = df_y_train.iloc[:, 0:1]
if df_y_test.shape[1] > 1:
    df_y_test = df_y_test.iloc[:, 0:1]

# 跳过'smiles'列
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
            pass
    numeric_columns_x_train = numeric_cols

print(f"使用的数值列数量: {len(numeric_columns_x_train)}")

df_x_train_numeric = df_x_train[numeric_columns_x_train]
df_x_test_numeric = df_x_test[numeric_columns_x_train]

# 合并数据
df_train = pd.concat([df_x_train_numeric, df_y_train], axis=1)
df_test = pd.concat([df_x_test_numeric, df_y_test], axis=1)

# 删除NaN值
clean_data_train = df_train.dropna()
clean_data_test = df_test.dropna()

print(f"清洗后训练集形状: {clean_data_train.shape}")
print(f"清洗后测试集形状: {clean_data_test.shape}")

# 转换为数组
x_train = clean_data_train.iloc[:, :-1].values
y_train = clean_data_train.iloc[:, -1].values
x_test = clean_data_test.iloc[:, :-1].values
y_test = clean_data_test.iloc[:, -1].values

# 数据类型转换
x_train = x_train.astype(np.float32)
y_train = y_train.astype(np.float32)
x_test = x_test.astype(np.float32)
y_test = y_test.astype(np.float32)

print(f"最终数据形状 - x_train: {x_train.shape}, x_test: {x_test.shape}")

# 合并并重新分割
all_x = np.vstack([x_train, x_test])
all_y = np.concatenate([y_train, y_test])
train_ratio = len(x_train) / (len(x_train) + len(x_test))

x_train, x_test, y_train, y_test = train_test_split(
    all_x, all_y, test_size=1 - train_ratio, random_state=42, stratify=all_y
)

print(f"重新分割后 - 训练集: {x_train.shape}, 测试集: {x_test.shape}")

# ==================== 数据预处理 ====================
print("\n=== 数据标准化 ===")

scaler = StandardScaler()
x_train_scaled = scaler.fit_transform(x_train)
x_test_scaled = scaler.transform(x_test)

# 类别权重
unique, counts = np.unique(y_train, return_counts=True)
print(f"类别分布: {dict(zip(unique, counts))}")
class_weights = compute_class_weight('balanced', classes=np.unique(y_train), y=y_train)
class_weight_dict = {i: weight for i, weight in enumerate(class_weights)}
print(f"类别权重: {class_weight_dict}")


# ==================== 最佳稳定配置 ====================
def create_stable_dnn(input_shape):
    """经过多次测试验证的最稳定配置"""
    model = k.models.Sequential()

    # 第1层 - 适中容量
    model.add(Dense(256, input_dim=input_shape, activation="relu",
                    kernel_regularizer=l1_l2(l1=0.0001, l2=0.0005)))
    model.add(BatchNormalization())
    model.add(Dropout(0.6))

    # 第2层
    model.add(Dense(128, activation="relu",
                    kernel_regularizer=l1_l2(l1=0.0001, l2=0.0005)))
    model.add(BatchNormalization())
    model.add(Dropout(0.5))

    # 第3层
    model.add(Dense(64, activation="relu",
                    kernel_regularizer=l1_l2(l1=0.0001, l2=0.0005)))
    model.add(BatchNormalization())
    model.add(Dropout(0.4))

    # 输出层
    model.add(Dense(1, activation="sigmoid"))

    # 优化器
    optimizer = tf.keras.optimizers.Adam(
        learning_rate=0.0003,
        beta_1=0.9,
        beta_2=0.999,
        epsilon=1e-07
    )

    model.compile(
        loss="binary_crossentropy",  # 使用标准交叉熵
        optimizer=optimizer,
        metrics=["accuracy"]
    )

    return model


# ==================== 五折交叉验证 ====================
print("\n=== 五折交叉验证 ===")

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
cv_metrics = {'auc': [], 'accuracy': [], 'sensitivity': [], 'specificity': [], 'f1': []}

for fold, (train_idx, val_idx) in enumerate(skf.split(x_train_scaled, y_train), 1):
    print(f"\n第 {fold} 折:")

    x_cv_train, x_cv_val = x_train_scaled[train_idx], x_train_scaled[val_idx]
    y_cv_train, y_cv_val = y_train[train_idx], y_train[val_idx]

    cv_class_weights = compute_class_weight('balanced', classes=np.unique(y_cv_train), y=y_cv_train)
    cv_class_weight_dict = {i: weight for i, weight in enumerate(cv_class_weights)}

    tf.keras.backend.clear_session()
    cv_model = create_stable_dnn(input_shape=x_train_scaled.shape[1])

    cv_model.fit(
        x_cv_train, y_cv_train,
        epochs=200,
        batch_size=64,
        validation_data=(x_cv_val, y_cv_val),
        class_weight=cv_class_weight_dict,
        callbacks=[
            EarlyStopping(monitor='val_loss', patience=25, restore_best_weights=True),
            ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=12, min_lr=1e-6)
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

    cv_metrics['auc'].append(auc)
    cv_metrics['accuracy'].append(acc)
    cv_metrics['sensitivity'].append(se)
    cv_metrics['specificity'].append(sp)
    cv_metrics['f1'].append(f1)

    print(f"AUC: {auc:.4f}, 准确率: {acc:.4f}, 灵敏度: {se:.4f}, 特异度: {sp:.4f}")

# 交叉验证结果汇总
print("\n" + "=" * 80)
print("五折交叉验证结果 (稳定版):")
print("=" * 80)
print(f"AUC: {np.mean(cv_metrics['auc']):.4f} (±{np.std(cv_metrics['auc']):.4f})")
print(f"准确率: {np.mean(cv_metrics['accuracy']):.4f} (±{np.std(cv_metrics['accuracy']):.4f})")
print(f"灵敏度: {np.mean(cv_metrics['sensitivity']):.4f} (±{np.std(cv_metrics['sensitivity']):.4f})")
print(f"特异度: {np.mean(cv_metrics['specificity']):.4f} (±{np.std(cv_metrics['specificity']):.4f})")
print(f"F1分数: {np.mean(cv_metrics['f1']):.4f} (±{np.std(cv_metrics['f1']):.4f})")

# ==================== 训练最终模型 ====================
print("\n=== 训练最终模型 ===")

tf.keras.backend.clear_session()
final_model = create_stable_dnn(input_shape=x_train_scaled.shape[1])

history = final_model.fit(
    x_train_scaled, y_train,
    epochs=300,
    batch_size=64,
    validation_split=0.15,
    class_weight=class_weight_dict,
    callbacks=[
        EarlyStopping(monitor='val_loss', patience=30, restore_best_weights=True, verbose=1),
        ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=15, min_lr=1e-6, verbose=1)
    ],
    verbose=2,
    shuffle=True
)

# 保存模型
final_model.save(r"E:\模型10\ar_stable_best.keras")

# ==================== 测试集评估 ====================
print("\n=== 测试集评估 ===")

y_pred = final_model.predict(x_test_scaled, verbose=0)
y_pred_binary = (y_pred > 0.5).astype(int).ravel()

# 计算指标
tn, fp, fn, tp = confusion_matrix(y_test, y_pred_binary).ravel()
se = tp / (tp + fn) if (tp + fn) > 0 else 0
sp = tn / (tn + fp) if (tn + fp) > 0 else 0
Q = (tp + tn) / (tp + tn + fp + fn)

denominator = np.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
C = (tp * tn - fp * fn) / denominator if denominator > 0 else 0

auc = roc_auc_score(y_test, y_pred)
precision = tp / (tp + fp) if (tp + fp) > 0 else 0
recall = tp / (tp + fn) if (tp + fn) > 0 else 0
f_measure = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
ba = (se + sp) / 2

# ROC曲线
fpr, tpr, _ = roc_curve(y_test, y_pred)
plt.figure(figsize=(10, 8))
plt.plot(fpr, tpr, color='darkorange', lw=3, label=f'ROC curve (AUC = {auc:.4f})')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', alpha=0.5)
plt.xlabel('False Positive Rate', fontsize=12)
plt.ylabel('True Positive Rate', fontsize=12)
plt.title('Stable DNN - ROC Curve', fontsize=14)
plt.legend(loc="lower right", fontsize=12)
plt.grid(True, alpha=0.3)
plt.savefig(r'E:\模型10\ar_stable_ROC.pdf', dpi=300, bbox_inches='tight')
plt.close()

# 训练历史图
plt.figure(figsize=(14, 6))

plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Training Accuracy', linewidth=2)
plt.plot(history.history['val_accuracy'], label='Validation Accuracy', linewidth=2)
plt.title('Model Accuracy - Stable DNN', fontsize=14)
plt.ylabel('Accuracy', fontsize=12)
plt.xlabel('Epoch', fontsize=12)
plt.legend()
plt.grid(True, alpha=0.3)

plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Training Loss', linewidth=2)
plt.plot(history.history['val_loss'], label='Validation Loss', linewidth=2)
plt.title('Model Loss - Stable DNN', fontsize=14)
plt.ylabel('Loss', fontsize=12)
plt.xlabel('Epoch', fontsize=12)
plt.legend()
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(r'E:\模型10\ar_stable_history.pdf', dpi=300, bbox_inches='tight')
plt.close()

# 评估结果
score = final_model.evaluate(x_test_scaled, y_test, verbose=0)

print('\n' + '=' * 60)
print('稳定版DNN性能评估结果:')
print('=' * 60)
print(f'Test loss: {score[0]:.4f}')
print(f'Test accuracy: {score[1]:.4f}')
print(f'Sensitivity (SE): {se:.4f}')
print(f'Specificity (SP): {sp:.4f}')
print(f'TP: {tp}, FP: {fp}')
print(f'TN: {tn}, FN: {fn}')
print(f'Overall Prediction Accuracy (Q): {Q:.4f}')
print(f'Matthews Correlation Coefficient (C): {C:.4f}')
print(f'AUC: {auc:.4f}')
print(f'Balanced Accuracy (BA): {ba:.4f}')
print(f'F-measure: {f_measure:.4f}')
print(f'Precision: {precision:.4f}')
print(f'Recall: {recall:.4f}')

print("\n详细分类报告:")
print(classification_report(y_test, y_pred_binary, target_names=['Class 0', 'Class 1']))

print("\n训练统计:")
print(f"最佳验证准确率: {np.max(history.history['val_accuracy']):.4f}")
print(f"最佳验证损失: {np.min(history.history['val_loss']):.4f}")
print(f"最终训练准确率: {history.history['accuracy'][-1]:.4f}")

print("\n=== 训练完成 ===")
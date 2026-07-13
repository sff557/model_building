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
from sklearn.metrics import confusion_matrix, accuracy_score, roc_curve, roc_auc_score, classification_report, f1_score, \
    precision_score, recall_score
from sklearn.utils.class_weight import compute_class_weight
from keras.callbacks import EarlyStopping, ReduceLROnPlateau
import warnings

warnings.filterwarnings('ignore')

print("=" * 60)
print("【训练集测试集4】稳定0.87+ AUC配置")
print("=" * 60)

# ==================== 数据加载 ====================
df_x_train = pd.read_csv(r"E:\训练集测试集4/hfX_train_maccs+chem.csv", na_values=["?", "NA"])
df_y_train = pd.read_csv(r"E:\训练集测试集4/hfy_train_maccs+chem.csv", na_values=["?", "NA"])
df_x_test = pd.read_csv(r"E:\训练集测试集4/hfX_test_maccs+chem.csv", na_values=["?", "NA"])
df_y_test = pd.read_csv(r"E:\训练集测试集4/hfy_test_maccs+chem.csv", na_values=["?", "NA"])

print(f"原始数据 - X_train: {df_x_train.shape}, y_train: {df_y_train.shape}")
print(f"原始数据 - X_test: {df_x_test.shape}, y_test: {df_y_test.shape}")

# 确保标签为1D
if df_y_train.shape[1] > 1:
    y_train_flat = df_y_train.iloc[:, 0].values.ravel()
else:
    y_train_flat = df_y_train.values.ravel()

if df_y_test.shape[1] > 1:
    y_test_flat = df_y_test.iloc[:, 0].values.ravel()
else:
    y_test_flat = df_y_test.values.ravel()


# ==================== 【关键】强制提取数值列 ====================
def get_numeric_columns(df):
    """只保留能转换为数值的列"""
    numeric_cols = []
    for col in df.columns:
        try:
            if pd.api.types.is_numeric_dtype(df[col]):
                numeric_cols.append(col)
            else:
                test = pd.to_numeric(df[col], errors='coerce')
                if test.notna().sum() > 0:
                    numeric_cols.append(col)
        except:
            continue
    return numeric_cols


# 获取数值列
numeric_cols_train = get_numeric_columns(df_x_train)
print(f"训练集数值列数量: {len(numeric_cols_train)}")

# 只保留训练集和测试集共有的列
available_cols = [col for col in numeric_cols_train if col in df_x_test.columns]
print(f"训练集和测试集共有的数值列数量: {len(available_cols)}")

if len(available_cols) == 0:
    print("错误：没有找到数值列！")
    exit(1)

feature_cols = available_cols
print(f"最终使用的特征数量: {len(feature_cols)}")
print("前5个特征列:", feature_cols[:5])

# 提取数据
X_train_raw = df_x_train[feature_cols].values.astype(np.float32)
X_test_raw = df_x_test[feature_cols].values.astype(np.float32)

print(f"提取后 - X_train: {X_train_raw.shape}, X_test: {X_test_raw.shape}")

# ==================== 数据清洗 ====================
train_nan_mask = ~np.isnan(X_train_raw).any(axis=1)
test_nan_mask = ~np.isnan(X_test_raw).any(axis=1)

X_train_clean = X_train_raw[train_nan_mask]
y_train_clean = y_train_flat[train_nan_mask]
X_test_clean = X_test_raw[test_nan_mask]
y_test_clean = y_test_flat[test_nan_mask]

print(f"清洗后 - 训练: {X_train_clean.shape}, 测试: {X_test_clean.shape}")

# ==================== 合并并重新分割 ====================
all_X = np.vstack([X_train_clean, X_test_clean])
all_y = np.concatenate([y_train_clean, y_test_clean])

X_train, X_test, y_train, y_test = train_test_split(
    all_X, all_y,
    test_size=0.2,
    random_state=42,
    stratify=all_y
)

print(f"最终分割 - 训练: {X_train.shape}, 测试: {X_test.shape}")

# ==================== 标准化 ====================
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

unique, counts = np.unique(y_train, return_counts=True)
print(f"训练集类别分布: {dict(zip(unique, counts))}")

class_weights = compute_class_weight('balanced', classes=np.unique(y_train), y=y_train)
class_weight_dict = {i: w for i, w in enumerate(class_weights)}
print(f"类别权重: {class_weight_dict}")


# ==================== 【核心】最佳0.87配置 ====================
def create_best_dnn(input_shape):
    """经过验证的最佳配置 - 极弱正则化"""
    model = k.models.Sequential()

    # 第1层 - 256神经元，极弱正则化
    model.add(Dense(256, input_dim=input_shape, activation="relu",
                    kernel_regularizer=l1_l2(l1=0.0001, l2=0.0005),  # 关键：极弱正则化
                    kernel_initializer='he_normal'))
    model.add(BatchNormalization())
    model.add(Dropout(0.6))

    # 第2层 - 128神经元
    model.add(Dense(128, activation="relu",
                    kernel_regularizer=l1_l2(l1=0.0001, l2=0.0005),
                    kernel_initializer='he_normal'))
    model.add(BatchNormalization())
    model.add(Dropout(0.5))

    # 第3层 - 64神经元
    model.add(Dense(64, activation="relu",
                    kernel_regularizer=l1_l2(l1=0.0001, l2=0.0005),
                    kernel_initializer='he_normal'))
    model.add(BatchNormalization())
    model.add(Dropout(0.4))

    # 输出层
    model.add(Dense(1, activation="sigmoid"))

    # 优化器 - 关键学习率
    optimizer = tf.keras.optimizers.Adam(
        learning_rate=0.0003,
        beta_1=0.9,
        beta_2=0.999,
        epsilon=1e-07
    )

    model.compile(
        loss="binary_crossentropy",
        optimizer=optimizer,
        metrics=["accuracy"]
    )

    return model


# ==================== 五折交叉验证 ====================
print("\n" + "=" * 60)
print("五折交叉验证")
print("=" * 60)

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
cv_results = {
    'auc': [], 'accuracy': [], 'sensitivity': [],
    'specificity': [], 'f1': [], 'precision': [], 'recall': []
}

for fold, (train_idx, val_idx) in enumerate(skf.split(X_train_scaled, y_train), 1):
    print(f"\n第 {fold} 折:")

    X_fold_train = X_train_scaled[train_idx]
    X_fold_val = X_train_scaled[val_idx]
    y_fold_train = y_train[train_idx]
    y_fold_val = y_train[val_idx]

    fold_weights = compute_class_weight('balanced', classes=np.unique(y_fold_train), y=y_fold_train)
    fold_weight_dict = {i: w for i, w in enumerate(fold_weights)}

    tf.keras.backend.clear_session()
    model = create_best_dnn(input_shape=X_train_scaled.shape[1])

    model.fit(
        X_fold_train, y_fold_train,
        epochs=300,
        batch_size=32,
        validation_data=(X_fold_val, y_fold_val),
        class_weight=fold_weight_dict,
        callbacks=[
            EarlyStopping(monitor='val_loss', patience=35, restore_best_weights=True, verbose=0),
            ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=15, min_lr=1e-6, verbose=0)
        ],
        verbose=0
    )

    y_pred_proba = model.predict(X_fold_val, verbose=0).ravel()
    y_pred_binary = (y_pred_proba > 0.5).astype(int)

    tn, fp, fn, tp = confusion_matrix(y_fold_val, y_pred_binary).ravel()
    auc = roc_auc_score(y_fold_val, y_pred_proba)
    acc = accuracy_score(y_fold_val, y_pred_binary)
    se = tp / (tp + fn) if (tp + fn) > 0 else 0
    sp = tn / (tn + fp) if (tn + fp) > 0 else 0
    f1 = f1_score(y_fold_val, y_pred_binary)
    precision = precision_score(y_fold_val, y_pred_binary)
    recall = recall_score(y_fold_val, y_pred_binary)

    cv_results['auc'].append(auc)
    cv_results['accuracy'].append(acc)
    cv_results['sensitivity'].append(se)
    cv_results['specificity'].append(sp)
    cv_results['f1'].append(f1)
    cv_results['precision'].append(precision)
    cv_results['recall'].append(recall)

    print(f"AUC: {auc:.4f}, ACC: {acc:.4f}, SE: {se:.4f}, SP: {sp:.4f}")

print("\n" + "=" * 60)
print("交叉验证汇总:")
print("=" * 60)
print(f"AUC: {np.mean(cv_results['auc']):.4f} ± {np.std(cv_results['auc']):.4f}")
print(f"准确率: {np.mean(cv_results['accuracy']):.4f} ± {np.std(cv_results['accuracy']):.4f}")
print(f"灵敏度: {np.mean(cv_results['sensitivity']):.4f} ± {np.std(cv_results['sensitivity']):.4f}")
print(f"特异度: {np.mean(cv_results['specificity']):.4f} ± {np.std(cv_results['specificity']):.4f}")
print(f"F1: {np.mean(cv_results['f1']):.4f} ± {np.std(cv_results['f1']):.4f}")

# ==================== 训练最终模型 ====================
print("\n" + "=" * 60)
print("训练最终模型")
print("=" * 60)

tf.keras.backend.clear_session()
final_model = create_best_dnn(input_shape=X_train_scaled.shape[1])

history = final_model.fit(
    X_train_scaled, y_train,
    epochs=500,
    batch_size=32,
    validation_split=0.15,
    class_weight=class_weight_dict,
    callbacks=[
        EarlyStopping(monitor='val_loss', patience=50, restore_best_weights=True, verbose=1),
        ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=20, min_lr=1e-7, verbose=1)
    ],
    verbose=1
)

# ==================== 测试集评估 ====================
print("\n" + "=" * 60)
print("测试集评估")
print("=" * 60)

y_pred_proba = final_model.predict(X_test_scaled, verbose=0).ravel()
y_pred_binary = (y_pred_proba > 0.5).astype(int)

tn, fp, fn, tp = confusion_matrix(y_test, y_pred_binary).ravel()
se = tp / (tp + fn) if (tp + fn) > 0 else 0
sp = tn / (tn + fp) if (tn + fp) > 0 else 0
acc = (tp + tn) / (tp + tn + fp + fn)
precision = tp / (tp + fp) if (tp + fp) > 0 else 0
f1 = 2 * precision * se / (precision + se) if (precision + se) > 0 else 0
ba = (se + sp) / 2
auc = roc_auc_score(y_test, y_pred_proba)

denominator = np.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
mcc = (tp * tn - fp * fn) / denominator if denominator > 0 else 0

print(f"\n{'=' * 60}")
print(f"最终结果:")
print(f"{'=' * 60}")
print(f"测试集准确率: {acc:.4f}")
print(f"AUC: {auc:.4f}")
print(f"灵敏度 (SE): {se:.4f}")
print(f"特异度 (SP): {sp:.4f}")
print(f"精准率: {precision:.4f}")
print(f"F1分数: {f1:.4f}")
print(f"MCC: {mcc:.4f}")
print(f"平衡准确率 (BA): {ba:.4f}")

print(f"\n混淆矩阵:")
print(f"TP: {tp}, FP: {fp}")
print(f"TN: {tn}, FN: {fn}")

print("\n分类报告:")
print(classification_report(y_test, y_pred_binary, target_names=['Class 0', 'Class 1']))

# ==================== 保存模型和绘图 ====================
final_model.save(r"E:\模型10\HF_dataset4_best.keras")
print(f"\n模型已保存到: E:\模型10\HF_dataset4_best.keras")

# ROC曲线
plt.figure(figsize=(10, 8))
fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
plt.plot(fpr, tpr, color='darkorange', lw=3, label=f'ROC (AUC = {auc:.4f})')
plt.plot([0, 1], [0, 1], 'k--', lw=2, alpha=0.5)
plt.xlabel('False Positive Rate', fontsize=12)
plt.ylabel('True Positive Rate', fontsize=12)
plt.title('Dataset4 Best DNN - ROC Curve', fontsize=14)
plt.legend(loc='lower right')
plt.grid(True, alpha=0.3)
plt.savefig(r"E:\模型10\HF_dataset4_best_roc.pdf", dpi=300, bbox_inches='tight')
plt.close()

# 训练历史
plt.figure(figsize=(14, 5))
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Train', linewidth=2)
plt.plot(history.history['val_accuracy'], label='Validation', linewidth=2)
plt.title('Model Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.grid(True, alpha=0.3)

plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Train', linewidth=2)
plt.plot(history.history['val_loss'], label='Validation', linewidth=2)
plt.title('Model Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(r"E:\模型10\HF_dataset4_best_history.pdf", dpi=300, bbox_inches='tight')
plt.close()

print(f"\n训练统计:")
print(f"最佳验证准确率: {np.max(history.history['val_accuracy']):.4f}")
print(f"最终训练准确率: {history.history['accuracy'][-1]:.4f}")
print(f"训练总轮数: {len(history.history['accuracy'])}")

print("\n" + "=" * 60)
print("训练完成！")
print("=" * 60)
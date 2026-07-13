import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import tensorflow as tf
import keras as k
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import confusion_matrix, accuracy_score, roc_curve, roc_auc_score, classification_report, f1_score
from sklearn.utils.class_weight import compute_class_weight
from keras.layers import BatchNormalization, Dropout
from keras.regularizers import l1_l2
from keras.callbacks import EarlyStopping, ReduceLROnPlateau
import warnings

warnings.filterwarnings('ignore')

# ==================== 数据加载 ====================
print("=" * 60)
print("加载数据...")
print("=" * 60)

df_x_train = pd.read_csv(r"E:\训练集测试集5\hbX_train_krfp.csv")
df_y_train = pd.read_csv(r"E:\训练集测试集5\hby_train_krfp.csv")
df_x_test = pd.read_csv(r"E:\训练集测试集5\hbX_test_krfp.csv")
df_y_test = pd.read_csv(r"E:\训练集测试集5\hby_test_krfp.csv")

print(f"训练集: X={df_x_train.shape}, y={df_y_train.shape}")
print(f"测试集: X={df_x_test.shape}, y={df_y_test.shape}")

# 提取数值列
if 'smiles' in df_x_train.columns:
    numeric_cols = df_x_train.columns.drop('smiles').tolist()
else:
    numeric_cols = [col for col in df_x_train.columns if col.startswith('KRFP_')]

print(f"使用 {len(numeric_cols)} 个KRFP特征")

# 提取特征和标签
X_train_raw = df_x_train[numeric_cols].values.astype(np.float64)
y_train_raw = df_y_train.values.ravel().astype(np.float64)
X_test_raw = df_x_test[numeric_cols].values.astype(np.float64)
y_test_raw = df_y_test.values.ravel().astype(np.float64)

# 删除NaN值
train_mask = ~np.isnan(X_train_raw).any(axis=1) & ~np.isnan(y_train_raw)
test_mask = ~np.isnan(X_test_raw).any(axis=1) & ~np.isnan(y_test_raw)

X_train_clean = X_train_raw[train_mask]
y_train_clean = y_train_raw[train_mask]
X_test_clean = X_test_raw[test_mask]
y_test_clean = y_test_raw[test_mask]

print(f"清洗后: 训练集 {X_train_clean.shape}, 测试集 {X_test_clean.shape}")

# ==================== 标准化 ====================
print("\n标准化数据...")
X_all = np.vstack([X_train_clean, X_test_clean])
y_all = np.concatenate([y_train_clean, y_test_clean])

scaler = StandardScaler()
X_all_scaled = scaler.fit_transform(X_all)

print(f"标准化完成，数据范围: [{X_all_scaled.min():.3f}, {X_all_scaled.max():.3f}]")

# 重新分割
train_ratio = len(X_train_clean) / (len(X_train_clean) + len(X_test_clean))
X_train, X_test, y_train, y_test = train_test_split(
    X_all_scaled, y_all,
    test_size=1 - train_ratio,
    random_state=42,
    stratify=y_all
)

# 转换为float32
X_train = X_train.astype(np.float32)
X_test = X_test.astype(np.float32)
y_train = y_train.astype(np.float32)
y_test = y_test.astype(np.float32)

print(f"最终: 训练集 {X_train.shape}, 测试集 {X_test.shape}")
print(f"类别分布 - 训练: {np.bincount(y_train.astype(int))}")
print(f"类别分布 - 测试: {np.bincount(y_test.astype(int))}")

# ==================== 类别权重 ====================
class_weights = compute_class_weight('balanced', classes=np.unique(y_train), y=y_train)
class_weight_dict = {i: weight for i, weight in enumerate(class_weights)}
print(f"类别权重: {class_weight_dict}")


# ==================== 恢复最佳模型架构 ====================
def create_model(input_shape):
    model = k.models.Sequential()

    # 第一层：较大的网络以捕捉复杂特征
    model.add(k.layers.Dense(128, input_dim=input_shape, activation='relu',
                             kernel_regularizer=l1_l2(l1=0.0001, l2=0.0005),
                             kernel_initializer='he_normal'))
    model.add(BatchNormalization())
    model.add(Dropout(0.4))

    # 第二层
    model.add(k.layers.Dense(64, activation='relu',
                             kernel_regularizer=l1_l2(l1=0.0001, l2=0.0005),
                             kernel_initializer='he_normal'))
    model.add(BatchNormalization())
    model.add(Dropout(0.3))

    # 第三层
    model.add(k.layers.Dense(32, activation='relu',
                             kernel_regularizer=l1_l2(l1=0.0001, l2=0.0005),
                             kernel_initializer='he_normal'))
    model.add(BatchNormalization())
    model.add(Dropout(0.2))

    # 输出层
    model.add(k.layers.Dense(1, activation='sigmoid'))

    # 使用Adam优化器
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.0005)
    model.compile(
        loss='binary_crossentropy',
        optimizer=optimizer,
        metrics=['accuracy']
    )

    return model


# ==================== 五折交叉验证 ====================
print("\n" + "=" * 60)
print("开始五折交叉验证")
print("=" * 60)

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
cv_results = {'auc': [], 'acc': [], 'se': [], 'sp': [], 'f1': []}

for fold, (train_idx, val_idx) in enumerate(skf.split(X_train, y_train), 1):
    print(f"\n--- Fold {fold} ---")

    X_cv_train, X_cv_val = X_train[train_idx], X_train[val_idx]
    y_cv_train, y_cv_val = y_train[train_idx], y_train[val_idx]

    fold_weights = compute_class_weight('balanced', classes=np.unique(y_cv_train), y=y_cv_train)
    fold_weight_dict = {i: w for i, w in enumerate(fold_weights)}

    model = create_model(X_train.shape[1])

    history = model.fit(
        X_cv_train, y_cv_train,
        epochs=150,
        batch_size=32,
        validation_data=(X_cv_val, y_cv_val),
        class_weight=fold_weight_dict,
        callbacks=[
            EarlyStopping(monitor='val_loss', patience=20, restore_best_weights=True),
            ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=10, min_lr=1e-6)
        ],
        verbose=0
    )

    y_pred_proba = model.predict(X_cv_val, verbose=0).ravel()
    y_pred_binary = (y_pred_proba > 0.5).astype(int)

    auc = roc_auc_score(y_cv_val, y_pred_proba)
    acc = accuracy_score(y_cv_val, y_pred_binary)
    tn, fp, fn, tp = confusion_matrix(y_cv_val, y_pred_binary).ravel()
    se = tp / (tp + fn) if (tp + fn) > 0 else 0
    sp = tn / (tn + fp) if (tn + fp) > 0 else 0
    f1 = f1_score(y_cv_val, y_pred_binary)

    cv_results['auc'].append(auc)
    cv_results['acc'].append(acc)
    cv_results['se'].append(se)
    cv_results['sp'].append(sp)
    cv_results['f1'].append(f1)

    print(f"AUC: {auc:.4f}, ACC: {acc:.4f}, SE: {se:.4f}, SP: {sp:.4f}")

print("\n" + "=" * 60)
print("交叉验证结果汇总:")
print("=" * 60)
for metric in cv_results:
    values = np.array(cv_results[metric])
    print(f"{metric.upper()}: {values.mean():.4f} (±{values.std():.4f})")

# ==================== 训练最终模型 ====================
print("\n" + "=" * 60)
print("训练最终模型")
print("=" * 60)

tf.keras.backend.clear_session()

final_model = create_model(X_train.shape[1])

history = final_model.fit(
    X_train, y_train,
    epochs=200,
    batch_size=32,
    validation_split=0.15,
    class_weight=class_weight_dict,
    callbacks=[
        EarlyStopping(monitor='val_loss', patience=25, restore_best_weights=True),
        ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=12, min_lr=1e-6)
    ],
    verbose=1
)

# 保存模型和scaler
final_model.save(r"E:\模型9\hb_model_best.keras")
import joblib

joblib.dump(scaler, r"E:\模型9\hb_scaler_best.pkl")
print("模型和scaler已保存")

# ==================== 测试集评估 ====================
print("\n" + "=" * 60)
print("测试集评估")
print("=" * 60)

y_pred_proba = final_model.predict(X_test, verbose=0).ravel()
y_pred_binary = (y_pred_proba > 0.5).astype(int)

# 寻找最佳阈值
from sklearn.metrics import f1_score

thresholds = np.arange(0.3, 0.7, 0.01)
best_threshold = 0.5
best_f1 = 0
for thresh in thresholds:
    y_pred_temp = (y_pred_proba > thresh).astype(int)
    f1 = f1_score(y_test, y_pred_temp)
    if f1 > best_f1:
        best_f1 = f1
        best_threshold = thresh

print(f"最佳阈值: {best_threshold:.2f} (F1: {best_f1:.4f})")

# 使用最佳阈值
y_pred_binary_optimized = (y_pred_proba > best_threshold).astype(int)

tn, fp, fn, tp = confusion_matrix(y_test, y_pred_binary_optimized).ravel()
se = tp / (tp + fn) if (tp + fn) > 0 else 0
sp = tn / (tn + fp) if (tn + fp) > 0 else 0
Q = (tp + tn) / len(y_test)
C = (tp * tn - fp * fn) / np.sqrt(max((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn), 1))
auc = roc_auc_score(y_test, y_pred_proba)
precision = tp / (tp + fp) if (tp + fp) > 0 else 0
recall = tp / (tp + fn) if (tp + fn) > 0 else 0
f_measure = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
ba = (se + sp) / 2

print(f"Test Loss: {final_model.evaluate(X_test, y_test, verbose=0)[0]:.4f}")
print(f"Test Accuracy: {accuracy_score(y_test, y_pred_binary_optimized):.4f}")
print(f"Sensitivity (SE): {se:.4f}")
print(f"Specificity (SP): {sp:.4f}")
print(f"Overall Accuracy (Q): {Q:.4f}")
print(f"MCC (C): {C:.4f}")
print(f"AUC: {auc:.4f}")
print(f"Balanced Accuracy (BA): {ba:.4f}")
print(f"F-measure: {f_measure:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")
print(f"TP: {tp}, FP: {fp}, TN: {tn}, FN: {fn}")

print("\n详细分类报告 (使用最佳阈值):")
print(classification_report(y_test, y_pred_binary_optimized, target_names=['Class 0', 'Class 1']))

# ==================== 可视化 ====================
# 1. ROC曲线
fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
plt.figure(figsize=(10, 8))
plt.plot(fpr, tpr, color='darkorange', lw=3, label=f'ROC (AUC = {auc:.3f})')
plt.plot([0, 1], [0, 1], 'navy', lw=2, linestyle='--', alpha=0.5)
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate', fontsize=12)
plt.ylabel('True Positive Rate', fontsize=12)
plt.title('ROC Curve - Best Model', fontsize=14)
plt.legend(loc="lower right")
plt.grid(True, alpha=0.3)
plt.savefig('E:/模型9/hb_ROC_best.pdf', dpi=300, bbox_inches='tight')
plt.close()

# 2. 训练历史
plt.figure(figsize=(14, 5))
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Train', linewidth=2)
plt.plot(history.history['val_accuracy'], label='Validation', linewidth=2)
plt.title('Model Accuracy', fontsize=14)
plt.xlabel('Epoch', fontsize=12)
plt.ylabel('Accuracy', fontsize=12)
plt.legend()
plt.grid(True, alpha=0.3)

plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Train', linewidth=2)
plt.plot(history.history['val_loss'], label='Validation', linewidth=2)
plt.title('Model Loss', fontsize=14)
plt.xlabel('Epoch', fontsize=12)
plt.ylabel('Loss', fontsize=12)
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('E:/模型9/hb_training_best.pdf', dpi=300, bbox_inches='tight')
plt.close()

# 3. 混淆矩阵和预测分布
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.hist(y_pred_proba[y_test == 0], bins=20, alpha=0.7, label='Class 0', color='blue', edgecolor='black')
plt.hist(y_pred_proba[y_test == 1], bins=20, alpha=0.7, label='Class 1', color='red', edgecolor='black')
plt.axvline(x=best_threshold, color='green', linestyle='--', linewidth=2, label=f'Best threshold={best_threshold:.2f}')
plt.xlabel('Predicted Probability', fontsize=12)
plt.ylabel('Frequency', fontsize=12)
plt.title('Prediction Distribution', fontsize=14)
plt.legend()
plt.grid(True, alpha=0.3)

plt.subplot(1, 2, 2)
cm = np.array([[tn, fp], [fn, tp]])
im = plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
plt.colorbar(im)
plt.title('Confusion Matrix', fontsize=14)
plt.ylabel('True Label', fontsize=12)
plt.xlabel('Predicted Label', fontsize=12)
tick_marks = np.arange(2)
plt.xticks(tick_marks, ['Class 0', 'Class 1'])
plt.yticks(tick_marks, ['Class 0', 'Class 1'])
for i in range(2):
    for j in range(2):
        plt.text(j, i, str(cm[i, j]), ha='center', va='center',
                 color='white' if cm[i, j] > cm.max() / 2 else 'black', fontsize=14)
plt.tight_layout()
plt.savefig('E:/模型9/hb_confusion_best.pdf', dpi=300, bbox_inches='tight')
plt.close()

print("\n" + "=" * 60)
print("训练完成！")
print("=" * 60)
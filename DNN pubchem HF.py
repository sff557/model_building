import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import tensorflow as tf
import keras as k
import pandas as pd
from keras.layers import BatchNormalization, Dropout
from keras.regularizers import l1_l2
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import confusion_matrix, accuracy_score, roc_curve, roc_auc_score, classification_report, f1_score, \
    precision_score, recall_score
from sklearn import metrics
from keras.callbacks import EarlyStopping, ReduceLROnPlateau, LearningRateScheduler
from sklearn.feature_selection import SelectFromModel
from sklearn.ensemble import RandomForestClassifier
import warnings

warnings.filterwarnings('ignore')

## 读取训练集测试集
df_x_train = pd.read_csv(r"E:\训练集测试集3\hfX_train_pubchem1.csv", na_values=["?", "NA"])
df_y_train = pd.read_csv(r"E:\训练集测试集3\hfy_train_pubchem1.csv", na_values=["?", "NA"])
df_x_test = pd.read_csv(r"E:\训练集测试集3\hfX_test_pubchem1.csv", na_values=["?", "NA"])
df_y_test = pd.read_csv(r"E:\训练集测试集3\hfy_test_pubchem1.csv", na_values=["?", "NA"])

print(f"原始数据形状 - df_x_train: {df_x_train.shape}, df_y_train: {df_y_train.shape}")
print(f"原始数据形状 - df_x_test: {df_x_test.shape}, df_y_test: {df_y_test.shape}")

# 检查并确保标签数据只有一列
if df_y_train.shape[1] > 1:
    df_y_train = df_y_train.iloc[:, 0:1]
if df_y_test.shape[1] > 1:
    df_y_test = df_y_test.iloc[:, 0:1]

# 跳过第一列'smiles'
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

# 只使用数值列
df_x_train_numeric = df_x_train[numeric_columns_x_train]
df_x_test_numeric = df_x_test[numeric_columns_x_train]

# 合并数据
df_train = pd.concat([df_x_train_numeric, df_y_train], axis=1)
df_test = pd.concat([df_x_test_numeric, df_y_test], axis=1)

# 删除包含 NaN 值的行
clean_data_train = df_train.dropna()
clean_data_test = df_test.dropna()

if clean_data_test.shape[0] == 0:
    clean_data_train, clean_data_test = train_test_split(clean_data_train, test_size=0.2, random_state=42,
                                                         stratify=clean_data_train.iloc[:, -1])

# 将数据转换为数组
x_train = clean_data_train.iloc[:, :-1].values
y_train = clean_data_train.iloc[:, -1].values
x_test = clean_data_test.iloc[:, :-1].values
y_test = clean_data_test.iloc[:, -1].values

# 确保数据类型为float32
try:
    x_train = x_train.astype(np.float32)
    y_train = y_train.astype(np.float32)
    x_test = x_test.astype(np.float32)
    y_test = y_test.astype(np.float32)
except:
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

# 检查x_test是否为空
if x_test.shape[1] == 0 or x_test.shape[0] == 0:
    print("错误：测试集特征为空，请检查数据文件")
    exit(1)

# 合并所有数据并重新分割
all_x = np.vstack([x_train, x_test])
all_y = np.concatenate([y_train, y_test])
train_ratio = len(x_train) / (len(x_train) + len(x_test))
x_train_new, x_test_new, y_train_new, y_test_new = train_test_split(
    all_x, all_y, test_size=1 - train_ratio, random_state=42, stratify=all_y
)
x_train, y_train, x_test, y_test = x_train_new, y_train_new, x_test_new, y_test_new

print(f"最终数据形状 - x_train: {x_train.shape}, y_train: {y_train.shape}")
print(f"最终数据形状 - x_test: {x_test.shape}, y_test: {y_test.shape}")

# ==================== 数据预处理 ====================
print("\n=== 开始数据预处理 ===")

# 1. 特征选择 - 更严格的选择
print("进行特征选择...")
rf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
rf.fit(x_train, y_train)

# 选择重要性排名前30%的特征（更严格）
importances = rf.feature_importances_
threshold = np.percentile(importances, 70)  # 只保留前30%最重要的特征
selector = SelectFromModel(rf, threshold=threshold, prefit=True)
x_train = selector.transform(x_train)
x_test = selector.transform(x_test)
print(f"特征选择后维度: x_train: {x_train.shape}, x_test: {x_test.shape}")

# 2. 数据标准化
scaler = StandardScaler()
x_train_scaled = scaler.fit_transform(x_train)
x_test_scaled = scaler.transform(x_test)

# 3. 检查类别分布
print("检查类别不平衡...")
unique, counts = np.unique(y_train, return_counts=True)
class_distribution = dict(zip(unique, counts))
print(f"类别分布: {class_distribution}")

# 计算类别权重
from sklearn.utils.class_weight import compute_class_weight

class_weights = compute_class_weight('balanced', classes=np.unique(y_train), y=y_train)
class_weight_dict = {i: weight for i, weight in enumerate(class_weights)}
print(f"类别权重: {class_weight_dict}")


# ==================== 极度简化的DNN模型 ====================
def create_simple_model(input_shape):
    model = k.models.Sequential()

    # 只用2层隐藏层，非常小
    model.add(k.layers.Dense(32, input_dim=input_shape, activation="relu",
                             kernel_regularizer=l1_l2(l1=0.001, l2=0.01),  # 强正则化
                             kernel_initializer='he_normal'))
    model.add(BatchNormalization())
    model.add(Dropout(0.5))  # 高dropout

    model.add(k.layers.Dense(16, activation="relu",
                             kernel_regularizer=l1_l2(l1=0.001, l2=0.01),
                             kernel_initializer='he_normal'))
    model.add(BatchNormalization())
    model.add(Dropout(0.3))

    # 输出层
    model.add(k.layers.Dense(1, activation="sigmoid"))

    # 更低的初始学习率
    optimizer = tf.keras.optimizers.Adam(
        learning_rate=0.0003,
        beta_1=0.9,
        beta_2=0.999,
        epsilon=1e-07,
        clipnorm=0.3  # 梯度裁剪
    )
    model.compile(loss="binary_crossentropy", optimizer=optimizer, metrics=["accuracy"])

    return model


# ==================== 五折交叉验证 ====================
print("\n=== 开始五折交叉验证 ===")

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
cv_metrics = {
    'auc': [], 'accuracy': [], 'sensitivity': [],
    'specificity': [], 'f1': [], 'precision': [], 'recall': []
}

models = []

for fold, (train_idx, val_idx) in enumerate(skf.split(x_train_scaled, y_train), 1):
    print(f"\n--- 开始第 {fold} 折交叉验证 ---")

    x_cv_train, x_cv_val = x_train_scaled[train_idx], x_train_scaled[val_idx]
    y_cv_train, y_cv_val = y_train[train_idx], y_train[val_idx]

    cv_class_weights = compute_class_weight('balanced', classes=np.unique(y_cv_train), y=y_cv_train)
    cv_class_weight_dict = {i: weight for i, weight in enumerate(cv_class_weights)}

    # 每次重新创建模型
    tf.keras.backend.clear_session()
    cv_model = create_simple_model(input_shape=x_train_scaled.shape[1])

    # 更早的早停
    early_stop = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=1e-6)

    history = cv_model.fit(
        x_cv_train, y_cv_train,
        epochs=100,  # 更少的epoch
        batch_size=16,
        validation_data=(x_cv_val, y_cv_val),
        class_weight=cv_class_weight_dict,
        callbacks=[early_stop, reduce_lr],
        verbose=0
    )

    models.append(cv_model)

    # 预测
    y_cv_val_proba = cv_model.predict(x_cv_val, verbose=0).ravel()
    y_cv_val_binary = (y_cv_val_proba > 0.5).astype(int).ravel()

    # 计算指标
    tn, fp, fn, tp = confusion_matrix(y_cv_val, y_cv_val_binary).ravel()
    auc = roc_auc_score(y_cv_val, y_cv_val_proba) if len(np.unique(y_cv_val)) > 1 else 0.0
    acc = accuracy_score(y_cv_val, y_cv_val_binary)
    se = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    sp = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    f1 = f1_score(y_cv_val, y_cv_val_binary)
    precision = precision_score(y_cv_val, y_cv_val_binary)
    recall = recall_score(y_cv_val, y_cv_val_binary)

    cv_metrics['auc'].append(auc)
    cv_metrics['accuracy'].append(acc)
    cv_metrics['sensitivity'].append(se)
    cv_metrics['specificity'].append(sp)
    cv_metrics['f1'].append(f1)
    cv_metrics['precision'].append(precision)
    cv_metrics['recall'].append(recall)

    print(f"第 {fold} 折结果:")
    print(f"  AUC: {auc:.4f}, 准确率: {acc:.4f}, 灵敏度: {se:.4f}, 特异度: {sp:.4f}")

# 计算交叉验证结果
print("\n" + "=" * 80)
print("五折交叉验证结果:")
print("=" * 80)
for metric in cv_metrics.keys():
    values = np.array(cv_metrics[metric])
    mean_val = np.mean(values)
    std_val = np.std(values)
    print(f"{metric.upper()}: {mean_val:.4f} (±{std_val:.4f})")

# ==================== 训练最终模型 ====================
print("\n=== 训练最终模型 ===")

tf.keras.backend.clear_session()
final_model = create_simple_model(input_shape=x_train_scaled.shape[1])

early_stop = EarlyStopping(monitor='val_loss', patience=12, restore_best_weights=True)
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=6, min_lr=1e-6)

history = final_model.fit(
    x_train_scaled, y_train,
    epochs=100,
    batch_size=16,
    validation_split=0.15,
    class_weight=class_weight_dict,
    callbacks=[early_stop, reduce_lr],
    verbose=1,
    shuffle=True
)

# ==================== 模型集成预测 ====================
print("\n=== 使用模型集成进行预测 ===")


def ensemble_predict(models, x_test):
    predictions = []
    for model in models:
        pred = model.predict(x_test, verbose=0)
        predictions.append(pred)
    return np.mean(predictions, axis=0)


y_pred_ensemble = ensemble_predict(models, x_test_scaled)
y_pred_binary = (y_pred_ensemble > 0.5).astype(int).ravel()

# 保存最终模型
final_model.save(r"E:\模型8\hf_dnn_final.keras")

# ==================== 测试集评估 ====================
print("\n=== 在测试集上评估最终模型 ===")

np.save(r"E:\模型8\hf_y_test_dnn.npy", y_test)
np.save(r"E:\模型8\hf_y_pred_dnn.npy", y_pred_ensemble)

# 计算指标
tn, fp, fn, tp = metrics.confusion_matrix(y_test, y_pred_binary).ravel()
se = tp / (tp + fn) if (tp + fn) > 0 else 0.0
sp = tn / (tn + fp) if (tn + fp) > 0 else 0.0
Q = (tp + tn) / (tp + tn + fp + fn)
C = (tp * tn - fp * fn) / np.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn)) if (tp + fp) * (tp + fn) * (
            tn + fp) * (tn + fn) > 0 else 0.0
auc = roc_auc_score(y_test, y_pred_ensemble)
precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
f_measure = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
ba = (se + sp) / 2

# 绘制 ROC 曲线
fpr, tpr, thresholds = roc_curve(y_test, y_pred_ensemble)

plt.figure(figsize=(10, 8))
plt.plot(fpr, tpr, color='darkorange', lw=3, label='DNN ROC curve (AUC = %0.3f)' % auc)
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', alpha=0.5)
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate', fontsize=12)
plt.ylabel('True Positive Rate', fontsize=12)
plt.title('Simplified DNN - Receiver Operating Characteristic', fontsize=14)
plt.legend(loc="lower right", fontsize=12)
plt.grid(True, alpha=0.3)

plt.savefig('E:/模型8/hf_ROC_dnn_simplified.pdf', dpi=300, bbox_inches='tight')
plt.close()

print('\n' + '=' * 60)
print('简化DNN模型性能评估结果:')
print('=' * 60)
print(f'Test loss: {history.history["val_loss"][-1]:.4f}')
print(f'Test accuracy: {history.history["val_accuracy"][-1]:.4f}')
print(f'Sensitivity (SE): {se:.4f}')
print(f'Specificity (SP): {sp:.4f}')
print(f'TP: {tp}, FP: {fp}, TN: {tn}, FN: {fn}')
print(f'Overall Prediction Accuracy (Q): {Q:.4f}')
print(f'Matthews Correlation Coefficient (C): {C:.4f}')
print(f'AUC: {auc:.4f}')
print(f'Balanced Accuracy (BA): {ba:.4f}')
print(f'F-measure: {f_measure:.4f}')
print(f'Precision: {precision:.4f}')
print(f'Recall: {recall:.4f}')

print("\n详细分类报告:")
print(classification_report(y_test, y_pred_binary, target_names=['Class 0', 'Class 1']))

# 绘制训练历史
plt.figure(figsize=(14, 6))

plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Training Accuracy', linewidth=2)
plt.plot(history.history['val_accuracy'], label='Validation Accuracy', linewidth=2)
plt.title('Model Accuracy - Simplified DNN', fontsize=14)
plt.ylabel('Accuracy', fontsize=12)
plt.xlabel('Epoch', fontsize=12)
plt.legend()
plt.grid(True, alpha=0.3)

plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Training Loss', linewidth=2)
plt.plot(history.history['val_loss'], label='Validation Loss', linewidth=2)
plt.title('Model Loss - Simplified DNN', fontsize=14)
plt.ylabel('Loss', fontsize=12)
plt.xlabel('Epoch', fontsize=12)
plt.legend()
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('E:/模型8/hf_training_history_dnn_simplified.pdf', dpi=300, bbox_inches='tight')
plt.close()

# 预测分布分析
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.hist(y_pred_ensemble[y_test == 0], bins=30, alpha=0.6, label='Class 0', color='blue')
plt.hist(y_pred_ensemble[y_test == 1], bins=30, alpha=0.6, label='Class 1', color='red')
plt.xlabel('Prediction Probability')
plt.ylabel('Count')
plt.title('Prediction Distribution by True Class')
plt.legend()
plt.grid(True, alpha=0.3)

plt.subplot(1, 2, 2)
plt.scatter(y_pred_ensemble[y_test == 0], np.zeros_like(y_pred_ensemble[y_test == 0]) + 0.1,
            alpha=0.5, label='Class 0', color='blue')
plt.scatter(y_pred_ensemble[y_test == 1], np.zeros_like(y_pred_ensemble[y_test == 1]) + 0.9,
            alpha=0.5, label='Class 1', color='red')
plt.xlabel('Prediction Probability')
plt.yticks([0.1, 0.9], ['Class 0', 'Class 1'])
plt.title('Prediction Probability by True Class')
plt.legend()
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('E:/模型8/hf_prediction_analysis_dnn_simplified.pdf', dpi=300, bbox_inches='tight')
plt.close()

print("\n=== 训练完成 ===")
print("所有结果已保存到 E:/模型8/")
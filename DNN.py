import matplotlib
matplotlib.use('Agg')  # 使用非交互式后端，避免PyCharm兼容性问题
import matplotlib.pyplot as plt
import tensorflow as tf
import keras as k
import pandas as pd
from keras.layers import Dense, BatchNormalization, Dropout  # 添加了 Dense
from keras.regularizers import l1_l2
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import confusion_matrix, accuracy_score, roc_curve, roc_auc_score, classification_report, f1_score, \
    precision_score, recall_score
from imblearn.over_sampling import SMOTE
from sklearn import metrics
from keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint, TerminateOnNaN
import warnings
warnings.filterwarnings('ignore')

roundCount = 100

## 读取训练集测试集（读之前需要手动去除表头）

# 读取 CSV 文件并将非数值类型的数据转换为 NaN
# 注意：确保文件路径正确
df_x_train = pd.read_csv(r"E:\训练集测试集4/hfX_train_chem.csv", na_values=["?", "NA"])
df_y_train = pd.read_csv(r"E:\训练集测试集4/hfy_train_chem.csv", na_values=["?", "NA"])
df_x_test = pd.read_csv(r"E:\训练集测试集4/hfX_test_chem.csv", na_values=["?", "NA"])
df_y_test = pd.read_csv(r"E:\训练集测试集4/hfy_test_chem.csv", na_values=["?", "NA"])

print(f"原始数据形状 - df_x_train: {df_x_train.shape}, df_y_train: {df_y_train.shape}")
print(f"原始数据形状 - df_x_test: {df_x_test.shape}, df_y_test: {df_y_test.shape}")

# 检查并确保标签数据只有一列
if df_y_train.shape[1] > 1:
    print(f"警告：y_train 有 {df_y_train.shape[1]} 列，将使用第一列作为标签")
    df_y_train = df_y_train.iloc[:, 0:1]

if df_y_test.shape[1] > 1:
    print(f"警告：y_test 有 {df_y_test.shape[1]} 列，将使用第一列作为标签")
    df_y_test = df_y_test.iloc[:, 0:1]

# 关键修复：跳过第一列'smiles'（字符串列），只选择数值列
print("原始列名:", df_x_train.columns.tolist()[:10])  # 只打印前10列

# 确保我们跳过SMILES列（第一列）
# 方法1：通过列名判断（不区分大小写）
if 'smiles' in [col.lower() for col in df_x_train.columns]:
    # 找到smiles列的位置
    smiles_col = [col for col in df_x_train.columns if col.lower() == 'smiles'][0]
    numeric_columns_x_train = df_x_train.columns.drop(smiles_col).tolist()
else:
    # 方法2：如果列名不包含'smiles'，则检查数据类型
    numeric_cols = []
    for col in df_x_train.columns:
        # 尝试转换为数值类型，如果不能转换则跳过
        try:
            pd.to_numeric(df_x_train[col], errors='raise')
            numeric_cols.append(col)
        except:
            print(f"跳过非数值列: {col}")
    numeric_columns_x_train = numeric_cols

print(f"使用的数值列数量: {len(numeric_columns_x_train)}")
print("前10个数值列:", numeric_columns_x_train[:10])

# 只使用数值列
df_x_train_numeric = df_x_train[numeric_columns_x_train]
df_x_test_numeric = df_x_test[numeric_columns_x_train]  # 使用相同的列名

print(f"数值特征形状 - x_train: {df_x_train_numeric.shape}, x_test: {df_x_test_numeric.shape}")

# 合并 x_train 和 y_train 为一个 DataFrame（使用数值列）
df_train = pd.concat([df_x_train_numeric, df_y_train], axis=1)
df_test = pd.concat([df_x_test_numeric, df_y_test], axis=1)

print(f"合并后形状 - 训练集: {df_train.shape}, 测试集: {df_test.shape}")

# 删除包含 NaN 值的行
clean_data_train = df_train.dropna()
clean_data_test = df_test.dropna()

print(f"清洗后训练集形状: {clean_data_train.shape}")
print(f"清洗后测试集形状: {clean_data_test.shape}")

# 检查测试集是否为空
if clean_data_test.shape[0] == 0:
    print("警告：测试集为空，将使用训练集的一部分作为测试集")
    # 如果测试集为空，从训练集分割一部分作为测试集
    clean_data_train, clean_data_test = train_test_split(clean_data_train, test_size=0.2, random_state=42,
                                                         stratify=clean_data_train.iloc[:, -1])
    print(f"重新分割后训练集形状: {clean_data_train.shape}")
    print(f"重新分割后测试集形状: {clean_data_test.shape}")

# 将数据转换为数组
x_train = clean_data_train.iloc[:, :-1].values  # 使用.values而不是.to_numpy()
y_train = clean_data_train.iloc[:, -1].values
x_test = clean_data_test.iloc[:, :-1].values
y_test = clean_data_test.iloc[:, -1].values

print(f"转换后形状 - x_train: {x_train.shape}, y_train: {y_train.shape}")
print(f"转换后形状 - x_test: {x_test.shape}, y_test: {y_test.shape}")

# 检查数据类型
print(f"x_train 数据类型: {x_train.dtype}")
print(f"x_train 中非数值数据的示例:")
# 检查是否有非数值数据
for i in range(min(5, len(x_train))):
    for j in range(min(5, x_train.shape[1])):
        if not isinstance(x_train[i, j], (int, float, np.integer, np.floating)):
            print(f"  位置 [{i}, {j}]: {x_train[i, j]} (类型: {type(x_train[i, j])})")

# 确保数据类型为float32
try:
    x_train = x_train.astype(np.float32)
    y_train = y_train.astype(np.float32)
    x_test = x_test.astype(np.float32)
    y_test = y_test.astype(np.float32)
    print("数据类型转换成功")
except Exception as e:
    print(f"数据类型转换失败: {e}")
    # 尝试手动转换非数值数据
    print("尝试手动处理非数值数据...")


    # 创建一个函数来处理数据转换
    def convert_to_float_array(data):
        result = []
        for i in range(data.shape[0]):
            row = []
            for j in range(data.shape[1]):
                try:
                    row.append(float(data[i, j]))
                except:
                    row.append(0.0)  # 将无法转换的值设为0
            result.append(row)
        return np.array(result, dtype=np.float32)


    x_train = convert_to_float_array(x_train)
    x_test = convert_to_float_array(x_test)
    y_train = y_train.astype(np.float32)
    y_test = y_test.astype(np.float32)

print(f"最终数据形状 - x_train: {x_train.shape}, y_train: {y_train.shape}")
print(f"最终数据形状 - x_test: {x_test.shape}, y_test: {y_test.shape}")

# 检查x_test是否为空
if x_test.shape[1] == 0 or x_test.shape[0] == 0:
    print("错误：测试集特征为空，请检查数据文件")
    exit(1)

# 合并所有数据
all_x = np.vstack([x_train, x_test])
all_y = np.concatenate([y_train, y_test])

print(f"合并后总数据形状: {all_x.shape}, {all_y.shape}")

# 重新分割数据，保持原始训练集和测试集的大小比例（大约80%训练，20%测试）
# 计算训练集所占比例
train_ratio = len(x_train) / (len(x_train) + len(x_test))

# 使用分层抽样重新分割
x_train_new, x_test_new, y_train_new, y_test_new = train_test_split(
    all_x, all_y, test_size=1 - train_ratio, random_state=42, stratify=all_y
)

# 替换原来的数据
x_train, y_train, x_test, y_test = x_train_new, y_train_new, x_test_new, y_test_new

print(f"重新分割后训练集形状: {x_train.shape}, {y_train.shape}")
print(f"重新分割后测试集形状: {x_test.shape}, {y_test.shape}")

# ==================== 数据预处理优化 ====================
print("\n=== 开始数据预处理优化 ===")

# 1. 数据标准化
scaler = StandardScaler()
x_train_scaled = scaler.fit_transform(x_train)
x_test_scaled = scaler.transform(x_test)

# 2. 检查并处理类别不平衡
print("检查类别不平衡...")
unique, counts = np.unique(y_train, return_counts=True)
class_distribution = dict(zip(unique, counts))
print(f"类别分布: {class_distribution}")

# ==================== 使用类别权重 ====================
# 计算类别权重
from sklearn.utils.class_weight import compute_class_weight

class_weights = compute_class_weight('balanced', classes=np.unique(y_train), y=y_train)
class_weight_dict = {i: weight for i, weight in enumerate(class_weights)}
print(f"类别权重: {class_weight_dict}")

# ==================== 五折交叉验证 ====================
print("\n=== 开始五折交叉验证 ===")


# 定义模型构建函数
def create_model(input_shape):
    # 梯度式调整正则化强度（顶层强，底层弱）
    model = k.models.Sequential()

    # 第一层 - 强正则化
    model.add(Dense(256, input_dim=input_shape, activation="relu",
                    kernel_regularizer=l1_l2(l1=0.001, l2=0.01)))  # L2增加到0.02
    model.add(BatchNormalization())
    model.add(Dropout(0.8))

    # 第二层 - 中等正则化
    model.add(Dense(128, activation="relu",
                    kernel_regularizer=l1_l2(l1=0.001, l2=0.01)))
    model.add(BatchNormalization())
    model.add(Dropout(0.7))  # 稍微降低到0.75

    # 第三层 - 弱正则化
    model.add(Dense(64, activation="relu",
                    kernel_regularizer=l1_l2(l1=0.001, l2=0.01)))  # L2降低到0.01
    model.add(BatchNormalization())
    model.add(Dropout(0.6))  # 降低到0.7

    # 输出层
    model.add(k.layers.Dense(1, activation="sigmoid"))
    # 输出层
    model.add(k.layers.Dense(1, activation="sigmoid"))

    # 调整优化器参数
    optimizer = tf.keras.optimizers.Adam(
        learning_rate=0.00015,  # 中等学习率
        beta_1=0.9,
        beta_2=0.999,
        epsilon=1e-07,
        clipnorm=1.0
    )
    model.compile(loss="binary_crossentropy", optimizer=optimizer, metrics=["accuracy"])

    return model


# 设置五折交叉验证
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
cv_metrics = {
    'auc': [], 'accuracy': [], 'sensitivity': [],
    'specificity': [], 'f1': [], 'precision': [], 'recall': []
}

for fold, (train_idx, val_idx) in enumerate(skf.split(x_train_scaled, y_train), 1):
    print(f"\n--- 开始第 {fold} 折交叉验证 ---")

    x_cv_train, x_cv_val = x_train_scaled[train_idx], x_train_scaled[val_idx]
    y_cv_train, y_cv_val = y_train[train_idx], y_train[val_idx]

    # 为当前fold计算类别权重
    cv_class_weights = compute_class_weight('balanced', classes=np.unique(y_cv_train), y=y_cv_train)
    cv_class_weight_dict = {i: weight for i, weight in enumerate(cv_class_weights)}

    # 创建和训练模型
    cv_model = create_model(input_shape=x_train_scaled.shape[1])

    cv_model.fit(
        x_cv_train, y_cv_train,
        epochs=300,
        batch_size=32,
        validation_data=(x_cv_val, y_cv_val),
        class_weight=cv_class_weight_dict,
        callbacks=[
            EarlyStopping(monitor='val_loss', patience=15, restore_best_weights=True),
            ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=10, min_lr=1e-6)
        ],
        verbose=0
    )

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

    # 存储指标
    cv_metrics['auc'].append(auc)
    cv_metrics['accuracy'].append(acc)
    cv_metrics['sensitivity'].append(se)
    cv_metrics['specificity'].append(sp)
    cv_metrics['f1'].append(f1)
    cv_metrics['precision'].append(precision)
    cv_metrics['recall'].append(recall)

    print(f"第 {fold} 折结果:")
    print(f"  AUC: {auc:.4f}, 准确率: {acc:.4f}, 灵敏度: {se:.4f}, 特异度: {sp:.4f}")

# 计算交叉验证结果的统计信息
cv_results = {}
for metric in cv_metrics.keys():
    values = np.array(cv_metrics[metric])
    cv_results[metric] = {
        'mean': np.mean(values),
        'std': np.std(values),
        'all_folds': values,
        'str': f"{np.mean(values):.4f} (±{np.std(values):.4f})"
    }

print("\n" + "=" * 80)
print("五折交叉验证结果:")
print("=" * 80)
print(f"AUC: {cv_results['auc']['str']}")
print(f"准确率: {cv_results['accuracy']['str']}")
print(f"灵敏度: {cv_results['sensitivity']['str']}")
print(f"特异度: {cv_results['specificity']['str']}")
print(f"F1分数: {cv_results['f1']['str']}")
print(f"精确率: {cv_results['precision']['str']}")
print(f"召回率: {cv_results['recall']['str']}")

# ==================== 训练最终模型 ====================
print("\n=== 训练最终模型 ===")

# 清除之前的模型
tf.keras.backend.clear_session()

final_model = create_model(input_shape=x_train_scaled.shape[1])

history = final_model.fit(
    x_train_scaled, y_train,
    epochs=500,
    batch_size=16,
    validation_split=0.15,
    class_weight=class_weight_dict,
    callbacks=[
        EarlyStopping(monitor='val_loss', patience=30, restore_best_weights=True),
        ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=15, min_lr=1e-6)
    ],
    verbose=2,
    shuffle=True
)

# 保存最终模型
final_model.save(r"E:\模型7\hf dnnchem model_cv.h5")

# ==================== 在测试集上评估 ====================
print("\n=== 在测试集上评估最终模型 ===")

# 预测
y_pred = final_model.predict(x_test_scaled)
y_pred_binary = np.round(y_pred).astype(int)

# 保存y_test和y_pred
np.save(r"E:\模型7\hf dnnchem_y_test.npy", y_test)
np.save(r"E:\模型7\hf dnnchem_y_pred.npy", y_pred)

# 计算 SE 和 SP 指标
tn, fp, fn, tp = metrics.confusion_matrix(y_test, y_pred_binary).ravel()
se = tp / (tp + fn)
sp = tn / (tn + fp)

# 计算整体预测准确度 Q
Q = (tp + tn) / (tp + tn + fp + fn)

# 计算马修斯相关系数 C
C = (tp * tn - fp * fn) / np.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))

# 计算 AUC 值
auc = roc_auc_score(y_test, y_pred)

# 计算精确率和召回率
precision = tp / (tp + fp)
recall = tp / (tp + fn)

# 计算 F-召回率
f_measure = 2 * (precision * recall) / (precision + recall)

# 计算平衡准确率
ba = (se + sp) / 2

# 绘制 ROC 曲线
fpr, tpr, thresholds = roc_curve(y_test, y_pred)

plt.figure(figsize=(10, 8))
plt.plot(fpr, tpr, color='darkorange', lw=3, label='ROC curve (area = %0.3f)' % auc)
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', alpha=0.5)
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate', fontsize=12)
plt.ylabel('True Positive Rate', fontsize=12)
plt.title('Final Model with 5-Fold CV - Receiver Operating Characteristic', fontsize=14)
plt.legend(loc="lower right", fontsize=12)
plt.grid(True, alpha=0.3)

# 保存 ROC 曲线图像
plt.savefig('E:\模型7\hf dnnchem_ROC with_cv.pdf', dpi=300, bbox_inches='tight')
plt.close()

# 评估模型
score = final_model.evaluate(x_test_scaled, y_test)
print('\n' + '=' * 60)
print('最终模型性能评估结果:')
print('=' * 60)
print('Test loss:', score[0])
print('Test accuracy:', score[1])
print('Sensitivity (SE):', se)
print('Specificity (SP):', sp)
print('TP:', tp)
print('FP:', fp)
print('TN:', tn)
print('FN:', fn)
print('Overall Prediction Accuracy (Q):', Q)
print('Matthews Correlation Coefficient (C):', C)
print('AUC:', auc)
print('Balanced Accuracy (BA):', ba)
print('F-measure:', f_measure)
print('Precision:', precision)
print('Recall:', recall)

# 添加详细分类报告
print("\n详细分类报告:")
print(classification_report(y_test, y_pred_binary, target_names=['Class 0', 'Class 1']))

# 绘制训练历史
plt.figure(figsize=(14, 6))

plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Training Accuracy', linewidth=2)
plt.plot(history.history['val_accuracy'], label='Validation Accuracy', linewidth=2)
plt.title('Model Accuracy - Final Model', fontsize=14)
plt.ylabel('Accuracy', fontsize=12)
plt.xlabel('Epoch', fontsize=12)
plt.legend()
plt.grid(True, alpha=0.3)

plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Training Loss', linewidth=2)
plt.plot(history.history['val_loss'], label='Validation Loss', linewidth=2)
plt.title('Model Loss - Final Model', fontsize=14)
plt.ylabel('Loss', fontsize=12)
plt.xlabel('Epoch', fontsize=12)
plt.legend()
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('E:\模型7\hf dnnchem training_history.pdf', dpi=300, bbox_inches='tight')
plt.close()

print("\n=== 训练完成 ===")
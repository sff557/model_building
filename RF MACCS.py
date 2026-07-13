import pandas as pd
import numpy as np
import matplotlib
import os

# 定义保存目录 - 修改为RF专用目录
SAVE_DIR = r"E:\RF_MACCS_Results"  # 修改这里为您想要的路径
os.makedirs(SAVE_DIR, exist_ok=True)  # 如果目录不存在则创建

matplotlib.rcParams['font.sans-serif'] = ['arial']
matplotlib.rcParams['axes.unicode_minus'] = False
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, roc_curve, roc_auc_score, f1_score, accuracy_score, matthews_corrcoef
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier
import joblib
from scipy.stats import randint
import warnings

warnings.filterwarnings('ignore')


# ---------------------- 1. 评估指标计算函数 ----------------------
def calculate_se(y_true, y_pred):
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    return tp / (tp + fn) if (tp + fn) != 0 else 0.0


def calculate_sp(y_true, y_pred):
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    return tn / (tn + fp) if (tn + fp) != 0 else 0.0


def calculate_acc(y_true, y_pred):
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    total = tp + tn + fp + fn
    return (tp + tn) / total if total != 0 else 0.0


# ---------------------- 2. 数据加载与预处理 --------------------
print("加载MACCS数据...")

# 读取数据
df_x_train = pd.read_csv(r"E:\训练集测试集\haX_train_maccs.csv", na_values=["?", "NA", " ", ""])
df_y_train = pd.read_csv(r"E:\训练集测试集\hay_train_maccs.csv", na_values=["?", "NA", " ", ""])
df_x_test = pd.read_csv(r"E:\训练集测试集\haX_test_maccs.csv", na_values=["?", "NA", " ", ""])
df_y_test = pd.read_csv(r"E:\训练集测试集\hay_test_maccs.csv", na_values=["?", "NA", " ", ""])

print(f"原始训练集形状: {df_x_train.shape}")
print(f"原始测试集形状: {df_x_test.shape}")

# 检查数据类型
print("\n检查数据类型...")
print("训练集数据类型分布:")
print(df_x_train.dtypes.value_counts())
print("\n测试集数据类型分布:")
print(df_x_test.dtypes.value_counts())

# 找出非数值列
non_numeric_cols_train = df_x_train.select_dtypes(exclude=['int64', 'float64']).columns.tolist()
non_numeric_cols_test = df_x_test.select_dtypes(exclude=['int64', 'float64']).columns.tolist()

print(f"\n发现 {len(non_numeric_cols_train)} 个非数值列在训练集中")
print(f"发现 {len(non_numeric_cols_test)} 个非数值列在测试集中")

if non_numeric_cols_train:
    print(f"训练集非数值列 '{non_numeric_cols_train[0]}' 示例值:", df_x_train[non_numeric_cols_train[0]].iloc[0])
if non_numeric_cols_test:
    print(f"测试集非数值列 '{non_numeric_cols_test[0]}' 示例值:", df_x_test[non_numeric_cols_test[0]].iloc[0])

# 删除非数值列（如SMILES字符串）
if non_numeric_cols_train or non_numeric_cols_test:
    cols_to_drop = list(set(non_numeric_cols_train + non_numeric_cols_test))
    df_x_train = df_x_train.drop(columns=cols_to_drop)
    df_x_test = df_x_test.drop(columns=cols_to_drop)
    print(f"删除非数值列: {cols_to_drop}")

print(f"\n删除后训练集形状: {df_x_train.shape}")
print(f"删除后测试集形状: {df_x_test.shape}")

# 检查转换后的数据类型
print("\n转换后数据类型分布:")
print(df_x_train.dtypes.value_counts())

# 合并数据
df_train = pd.concat([df_x_train, df_y_train], axis=1)
df_test = pd.concat([df_x_test, df_y_test], axis=1)

print(f"\n处理后训练集: {df_train.shape[0]} 个样本, {df_train.shape[1] - 1} 个特征")
print(f"处理后测试集: {df_test.shape[0]} 个样本, {df_test.shape[1] - 1} 个特征")

# 转换为numpy数组
x_train = df_train.iloc[:, :-1].to_numpy()
y_train = df_train.iloc[:, -1].to_numpy()
x_test = df_test.iloc[:, :-1].to_numpy()
y_test = df_test.iloc[:, -1].to_numpy()

# 检查缺失值
print(f"\n训练集缺失值比例: {np.isnan(x_train).sum() / x_train.size * 100:.2f}%")
print(f"测试集缺失值比例: {np.isnan(x_test).sum() / x_test.size * 100:.2f}%")

# 处理缺失值
print("\n处理缺失值...")
imputer = SimpleImputer(strategy='mean')
x_train = imputer.fit_transform(x_train)
x_test = imputer.transform(x_test)

# 保存数据预处理器
imputer_save_path = os.path.join(SAVE_DIR, "ha_rf_imputer.pkl")
joblib.dump(imputer, imputer_save_path)
print(f"✅ 缺失值处理器已保存: {imputer_save_path}")

# 检查类别分布
print("\n=== 训练集类别分布 ===")
print(f"Class 0: {np.sum(y_train == 0)}, Class 1: {np.sum(y_train == 1)}")
print(f"正负样本比例: {np.sum(y_train == 1) / len(y_train):.3f}")

# 计算类别权重
n_samples = len(y_train)
n_classes = 2
class_counts = np.bincount(y_train.astype(int))
class_weights = n_samples / (n_classes * class_counts)
class_weight_dict = {0: class_weights[0], 1: class_weights[1]}
print(f"类别权重: {class_weight_dict}")

# 直接使用原始数据（RF不需要标准化）
x_train_final = x_train.copy()
x_test_final = x_test.copy()

# ---------------------- 3. 随机森林模型训练和调优 ----------------------
print("\n训练随机森林模型...")

# 定义模型
model = RandomForestClassifier(
    random_state=42,
    class_weight=class_weight_dict,
    n_jobs=-1,
    oob_score=True  # 使用袋外分数评估
)

# 参数分布
param_dist = {
    'n_estimators': randint(100, 1000),  # 树的数量
    'max_depth': [None, 10, 20, 30, 40, 50],  # 树的最大深度
    'min_samples_split': randint(2, 20),  # 内部节点再划分所需最小样本数
    'min_samples_leaf': randint(1, 10),  # 叶子节点最少样本数
    'max_features': ['sqrt', 'log2', None, 0.5, 0.7, 0.9],  # 最大特征数
    'bootstrap': [True, False],  # 是否使用bootstrap采样
    'criterion': ['gini', 'entropy'],  # 划分标准
}

# 随机搜索
random_search = RandomizedSearchCV(
    model, param_dist, n_iter=50,
    scoring='roc_auc', cv=5, n_jobs=-1,
    random_state=42, verbose=1
)

random_search.fit(x_train_final, y_train)

print("\n=== 最佳参数 ===")
best_params = random_search.best_params_
print(best_params)

# 训练集整体性能评估
print("\n=== 训练集整体性能 ===")
train_pred_proba = random_search.predict_proba(x_train_final)[:, 1]
train_pred = random_search.predict(x_train_final)
train_auc = roc_auc_score(y_train, train_pred_proba)
train_acc = accuracy_score(y_train, train_pred)
train_se = calculate_se(y_train, train_pred)
train_sp = calculate_sp(y_train, train_pred)
train_f1 = f1_score(y_train, train_pred)
train_mcc = matthews_corrcoef(y_train, train_pred)

print(f"训练集整体性能 - AUC: {train_auc:.4f}, ACC: {train_acc:.4f}, "
      f"SE: {train_se:.4f}, SP: {train_sp:.4f}, F1: {train_f1:.4f}, MCC: {train_mcc:.4f}")

# ---------------------- 4. 交叉验证评估 ----------------------
print("\n=== 交叉验证性能 ===")
kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

fold_metrics = {
    'auc': [], 'acc': [], 'se': [], 'sp': [], 'f1': [], 'mcc': []
}

for fold, (train_idx, val_idx) in enumerate(kfold.split(x_train_final, y_train)):
    print(f"训练折叠 {fold + 1}/5...")

    # 训练该折叠的模型
    fold_model = RandomForestClassifier(**random_search.best_params_,
                                        random_state=42 + fold,
                                        class_weight=class_weight_dict,
                                        n_jobs=-1)
    fold_model.fit(x_train_final[train_idx], y_train[train_idx])

    # 预测
    y_val_pred_proba = fold_model.predict_proba(x_train_final[val_idx])[:, 1]
    y_val_pred = fold_model.predict(x_train_final[val_idx])

    # 计算指标
    fold_auc = roc_auc_score(y_train[val_idx], y_val_pred_proba)
    fold_acc = accuracy_score(y_train[val_idx], y_val_pred)
    fold_se = calculate_se(y_train[val_idx], y_val_pred)
    fold_sp = calculate_sp(y_train[val_idx], y_val_pred)
    fold_f1 = f1_score(y_train[val_idx], y_val_pred)
    fold_mcc = matthews_corrcoef(y_train[val_idx], y_val_pred)

    # 存储指标
    fold_metrics['auc'].append(fold_auc)
    fold_metrics['acc'].append(fold_acc)
    fold_metrics['se'].append(fold_se)
    fold_metrics['sp'].append(fold_sp)
    fold_metrics['f1'].append(fold_f1)
    fold_metrics['mcc'].append(fold_mcc)

    # 打印该折叠结果
    print(f"Fold {fold + 1} - AUC: {fold_auc:.4f}, ACC: {fold_acc:.4f}, "
          f"SE: {fold_se:.4f}, SP: {fold_sp:.4f}, F1: {fold_f1:.4f}, MCC: {fold_mcc:.4f}")

# 计算交叉验证平均性能
print("\n=== 交叉验证平均性能 ===")
for metric_name, values in fold_metrics.items():
    mean_value = np.mean(values)
    std_value = np.std(values)
    print(f"{metric_name.upper()}: {mean_value:.4f} ± {std_value:.4f}")

# ---------------------- 5. 测试集评估 ----------------------
print("\n" + "=" * 50)
print("测试集评估结果:")
print("=" * 50)

best_model = random_search.best_estimator_

# 在测试集上预测
y_pred_proba = best_model.predict_proba(x_test_final)[:, 1]
y_pred = best_model.predict(x_test_final)

# 计算测试集指标
auc = roc_auc_score(y_test, y_pred_proba)
acc = accuracy_score(y_test, y_pred)
se = calculate_se(y_test, y_pred)
sp = calculate_sp(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
mcc = matthews_corrcoef(y_test, y_pred)

# 混淆矩阵
tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()

# 输出结果
print(f'AUC: {auc:.4f}')
print(f'ACC: {acc:.4f}')
print(f'SE: {se:.4f}')
print(f'SP: {sp:.4f}')
print(f'F1: {f1:.4f}')
print(f'MCC: {mcc:.4f}')
print(f'TP: {tp}, FP: {fp}, TN: {tn}, FN: {fn}')

# ---------------------- 6. ROC曲线 ----------------------
fpr, tpr, thresholds = roc_curve(y_test, y_pred_proba)

plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'Random Forest (AUC = {auc:.2f})')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate', fontsize=12)
plt.ylabel('True Positive Rate', fontsize=12)
plt.title('Receiver Operating Characteristic (ROC) Curve', fontsize=14)
plt.legend(loc="lower right")
plt.grid(True, alpha=0.3)
plt.tight_layout()

# 保存ROC曲线
roc_save_path = os.path.join(SAVE_DIR, 'ha_rf_ROC.pdf')
plt.savefig(roc_save_path, dpi=300, bbox_inches='tight')
plt.close()

# ---------------------- 7. 保存结果和模型 ----------------------
print("\n保存结果...")

# 保存模型
model_save_path = os.path.join(SAVE_DIR, "ha_rf_model.pkl")
joblib.dump(best_model, model_save_path)

# 保存预测结果
y_test_save_path = os.path.join(SAVE_DIR, "y_test_model_ha_rf.npy")
y_pred_save_path = os.path.join(SAVE_DIR, "y_pred_model_ha_rf.npy")
np.save(y_test_save_path, y_test)
np.save(y_pred_save_path, y_pred_proba)

# 保存详细结果到CSV
results = pd.DataFrame({
    'Model': ['Random Forest'],
    'Best Parameters': [str(random_search.best_params_)],
    'Test AUC': [f"{auc:.4f}"],
    'Test Accuracy': [f"{acc:.4f}"],
    'Test Sensitivity (SE)': [f"{se:.4f}"],
    'Test Specificity (SP)': [f"{sp:.4f}"],
    'Test F1-score': [f"{f1:.4f}"],
    'Test MCC': [f"{mcc:.4f}"],
    'TP': [tp],
    'FP': [fp],
    'TN': [tn],
    'FN': [fn],
    'CV Mean AUC': [f"{np.mean(fold_metrics['auc']):.4f} ± {np.std(fold_metrics['auc']):.4f}"],
    'CV Mean Accuracy': [f"{np.mean(fold_metrics['acc']):.4f} ± {np.std(fold_metrics['acc']):.4f}"],
    'CV Mean SE': [f"{np.mean(fold_metrics['se']):.4f} ± {np.std(fold_metrics['se']):.4f}"],
    'CV Mean SP': [f"{np.mean(fold_metrics['sp']):.4f} ± {np.std(fold_metrics['sp']):.4f}"],
    'CV Mean F1': [f"{np.mean(fold_metrics['f1']):.4f} ± {np.std(fold_metrics['f1']):.4f}"],
    'CV Mean MCC': [f"{np.mean(fold_metrics['mcc']):.4f} ± {np.std(fold_metrics['mcc']):.4f}"],
    'Training AUC': [f"{train_auc:.4f}"],
    'Training Accuracy': [f"{train_acc:.4f}"],
    'Training SE': [f"{train_se:.4f}"],
    'Training SP': [f"{train_sp:.4f}"],
    'Training F1': [f"{train_f1:.4f}"],
    'Training MCC': [f"{train_mcc:.4f}"]
})

results_save_path = os.path.join(SAVE_DIR, 'ha_rf_results.csv')
results.to_csv(results_save_path, index=False)

# 保存每个折叠的详细结果
fold_results = pd.DataFrame({
    'Fold': range(1, 6),
    'AUC': fold_metrics['auc'],
    'Accuracy': fold_metrics['acc'],
    'Sensitivity': fold_metrics['se'],
    'Specificity': fold_metrics['sp'],
    'F1_Score': fold_metrics['f1'],
    'MCC': fold_metrics['mcc']
})

fold_results_save_path = os.path.join(SAVE_DIR, 'ha_rf_cv_fold_results.csv')
fold_results.to_csv(fold_results_save_path, index=False)

# ---------------------- 8. 打印总结 ----------------------
print(f"\n✅ 模型已保存到: {model_save_path}")
print(f"✅ 缺失值处理器已保存到: {imputer_save_path}")
print(f"✅ 测试集预测结果已保存到: {y_test_save_path} 和 {y_pred_save_path}")
print(f"✅ 详细结果已保存到: {results_save_path}")
print(f"✅ 交叉验证各折叠结果已保存到: {fold_results_save_path}")
print(f"✅ ROC曲线已保存到: {roc_save_path}")

# 显示所有保存的文件
print(f"\n📁 所有文件已保存到: {SAVE_DIR}")
print("📋 文件列表:")
for file in os.listdir(SAVE_DIR):
    file_path = os.path.join(SAVE_DIR, file)
    if os.path.isfile(file_path):
        print(f"  - {file}")

print("\n✅ 所有结果已保存!")
import pandas as pd
import numpy as np
import matplotlib
import os

# 定义保存目录
SAVE_DIR = r"E:\SVM_Results"
os.makedirs(SAVE_DIR, exist_ok=True)

matplotlib.rcParams['font.sans-serif'] = ['arial']
matplotlib.rcParams['axes.unicode_minus'] = False
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, roc_curve, roc_auc_score, f1_score, accuracy_score
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.svm import SVC
import joblib
from scipy.stats import loguniform


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


# ---------------------- 2. 数据预处理 ----------------------
# 读取数据
df_x_train = pd.read_csv(r"E:\训练集测试集\haX_train_maccs.csv", na_values=["?", "NA", " ", ""])
df_y_train = pd.read_csv(r"E:\训练集测试集\hay_train_maccs.csv", na_values=["?", "NA", " ", ""])
df_x_test = pd.read_csv(r"E:\训练集测试集\haX_test_maccs.csv", na_values=["?", "NA", " ", ""])
df_y_test = pd.read_csv(r"E:\训练集测试集\hay_test_maccs.csv", na_values=["?", "NA", " ", ""])

# 检查非数值并转换
for df in [df_x_train, df_x_test]:
    for col in df.columns:
        df[col] = pd.to_numeric(df[col], errors='coerce')

# 合并数据
df_train = pd.concat([df_x_train, df_y_train], axis=1)
df_test = pd.concat([df_x_test, df_y_test], axis=1)

x_train = df_train.iloc[:, :-1].to_numpy()
y_train = df_train.iloc[:, -1].to_numpy()
x_test = df_test.iloc[:, :-1].to_numpy()
y_test = df_test.iloc[:, -1].to_numpy()

# 处理缺失值
imputer = SimpleImputer(strategy='mean')
x_train = imputer.fit_transform(x_train)
x_test = imputer.transform(x_test)

# 标准化数据
scaler = StandardScaler()
x_train_scaled = scaler.fit_transform(x_train)
x_test_scaled = scaler.transform(x_test)

# 保存标准化器
scaler_save_path = os.path.join(SAVE_DIR, "mm_svm_scaler.pkl")
joblib.dump(scaler, scaler_save_path)
print(f"✅ 标准化器已保存到: {scaler_save_path}")

# 检查类别分布
print("\n=== 训练集类别分布 ===")
print(f"Class 0: {np.sum(y_train == 0)}, Class 1: {np.sum(y_train == 1)}")
print(f"正负样本比例: {np.sum(y_train == 1) / len(y_train):.3f}")

# ================== 修改点1: 调整类别权重以提升特异度 ==================
# 增加阴性样本权重，重点关注提升特异度
print("\n=== 使用优化权重提升特异度 ===")
print("原始权重: {0: 1.8, 1: 1}")
print("优化权重: {0: 3.0, 1: 1} (增加阴性样本权重)")
class_weights_dict = {0: 3.0, 1: 1}  # 修改这里：从1.8提升到3.0

# ---------------------- 3. 模型训练和调优 ----------------------
# 定义模型
model = SVC(
    kernel='rbf',
    probability=True,
    max_iter=-1,
    random_state=40,
    class_weight=class_weights_dict
)

# 参数分布
param_dist = {
    'C': loguniform(1e-3, 1e3),
    'gamma': loguniform(1e-4, 1e1),
    'tol': [1e-5, 1e-4, 1e-3],
    'shrinking': [True, False]
}

# 随机搜索
random_search = RandomizedSearchCV(
    model, param_dist, n_iter=50,
    scoring='roc_auc', cv=5, n_jobs=-1, random_state=40
)
random_search.fit(x_train_scaled, y_train)

print("\n=== 最佳参数 ===")
print(random_search.best_params_)

# ---------------------- 4. 交叉验证评估 ----------------------
print("\n=== 交叉验证性能 ===")
kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=40)

fold_metrics = {
    'auc': [], 'acc': [], 'se': [], 'sp': [], 'f1': []
}

for fold, (train_idx, val_idx) in enumerate(kfold.split(x_train_scaled, y_train)):
    y_val = y_train[val_idx]

    # 训练该折叠的模型
    fold_model = SVC(**random_search.best_params_,
                     kernel='rbf', probability=True,
                     class_weight=class_weights_dict,  # 使用优化权重
                     random_state=40)
    fold_model.fit(x_train_scaled[train_idx], y_train[train_idx])

    # 预测
    y_val_pred_proba = fold_model.predict_proba(x_train_scaled[val_idx])[:, 1]
    y_val_pred = fold_model.predict(x_train_scaled[val_idx])

    # 计算指标
    fold_auc = roc_auc_score(y_val, y_val_pred_proba)
    fold_acc = accuracy_score(y_val, y_val_pred)
    fold_se = calculate_se(y_val, y_val_pred)
    fold_sp = calculate_sp(y_val, y_val_pred)
    fold_f1 = f1_score(y_val, y_val_pred)

    # 存储指标
    fold_metrics['auc'].append(fold_auc)
    fold_metrics['acc'].append(fold_acc)
    fold_metrics['se'].append(fold_se)
    fold_metrics['sp'].append(fold_sp)
    fold_metrics['f1'].append(fold_f1)

    # 打印该折叠结果
    print(f"Fold {fold + 1} - AUC: {fold_auc:.4f}, ACC: {fold_acc:.4f}, "
          f"SE: {fold_se:.4f}, SP: {fold_sp:.4f}, F1: {fold_f1:.4f}")

# 计算交叉验证平均性能
print("\n=== 交叉验证平均性能 ===")
for metric_name, values in fold_metrics.items():
    mean_value = np.mean(values)
    std_value = np.std(values)
    print(f"{metric_name.upper()}: {mean_value:.4f} ± {std_value:.4f}")

# 使用最佳模型在整个训练集上的性能
cv_pred_proba = random_search.predict_proba(x_train_scaled)[:, 1]
cv_pred = random_search.predict(x_train_scaled)
cv_auc = roc_auc_score(y_train, cv_pred_proba)
cv_acc = accuracy_score(y_train, cv_pred)
cv_se = calculate_se(y_train, cv_pred)
cv_sp = calculate_sp(y_train, cv_pred)
cv_f1 = f1_score(y_train, cv_pred)

print(f"\n训练集整体性能 - AUC: {cv_auc:.4f}, ACC: {cv_acc:.4f}, "
      f"SE: {cv_se:.4f}, SP: {cv_sp:.4f}, F1: {cv_f1:.4f}")

# ---------------------- 5. 测试集评估 ----------------------
best_model = random_search.best_estimator_
y_pred_proba = best_model.predict_proba(x_test_scaled)[:, 1]

# ================== 修改点2: 优化阈值以平衡敏感度和特异度 ==================
# 寻找最优阈值（约登指数最大）
print("\n=== 阈值优化 ===")
fpr_train, tpr_train, thresholds_train = roc_curve(y_train, cv_pred_proba)
youden_idx = tpr_train - fpr_train
optimal_idx = np.argmax(youden_idx)
optimal_threshold = thresholds_train[optimal_idx]
print(f"默认阈值: 0.5")
print(f"最优阈值: {optimal_threshold:.4f} (基于约登指数)")

# 使用最优阈值进行预测
y_pred_optimized = (y_pred_proba >= optimal_threshold).astype(int)

# 计算优化后的测试集指标
auc = roc_auc_score(y_test, y_pred_proba)  # AUC不受阈值影响
acc = accuracy_score(y_test, y_pred_optimized)
se = calculate_se(y_test, y_pred_optimized)
sp = calculate_sp(y_test, y_pred_optimized)
f1 = f1_score(y_test, y_pred_optimized)

# 混淆矩阵和MCC
tn, fp, fn, tp = confusion_matrix(y_test, y_pred_optimized).ravel()
mcc = (tp * tn - fp * fn) / np.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn)) if (tp + fp) * (tp + fn) * (
            tn + fp) * (tn + fn) != 0 else 0.0

# 输出结果
print("\n" + "=" * 50)
print("测试集评估结果（优化后）:")
print("=" * 50)
print(f'使用阈值: {optimal_threshold:.4f}')
print(f'AUC: {auc:.4f}')
print(f'ACC: {acc:.4f}')
print(f'SE: {se:.4f}')
print(f'SP: {sp:.4f}')
print(f'F1: {f1:.4f}')
print(f'MCC: {mcc:.4f}')
print(f'TP: {tp}, FP: {fp}, TN: {tn}, FN: {fn}')

# ================== 修改点3: 与原始阈值(0.5)对比 ==================
y_pred_original = best_model.predict(x_test_scaled)
tn_orig, fp_orig, fn_orig, tp_orig = confusion_matrix(y_test, y_pred_original).ravel()
sp_orig = tn_orig / (tn_orig + fp_orig) if (tn_orig + fp_orig) > 0 else 0
se_orig = tp_orig / (tp_orig + fn_orig) if (tp_orig + fn_orig) > 0 else 0

print(f"\n=== 优化效果对比 ===")
print(f"原始阈值(0.5): SP={sp_orig:.4f}, SE={se_orig:.4f}")
print(f"优化阈值({optimal_threshold:.3f}): SP={sp:.4f}, SE={se:.4f}")
print(f"特异度提升: {abs(sp - sp_orig):.4f} (+{((sp - sp_orig)/sp_orig*100):.1f}%)")

# ---------------------- 6. ROC曲线 ----------------------
fpr, tpr, thresholds = roc_curve(y_test, y_pred_proba)
plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'SVM (AUC = {auc:.2f})')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')

# 标记最优阈值点
optimal_point_idx = np.argmin(np.abs(thresholds - optimal_threshold))
plt.plot(fpr[optimal_point_idx], tpr[optimal_point_idx], 'ro', markersize=8,
         label=f'Optimal Threshold ({optimal_threshold:.2f})')

plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate', fontsize=12)
plt.ylabel('True Positive Rate', fontsize=12)
plt.title(f'ROC Curve (SP={sp:.3f}, SE={se:.3f})', fontsize=14)
plt.legend(loc="lower right")
plt.grid(True, alpha=0.3)

# 保存ROC曲线
roc_save_path = os.path.join(SAVE_DIR, 'mm_svm_ROC_optimized.pdf')
plt.savefig(roc_save_path, dpi=300, bbox_inches='tight')
plt.close()

# ---------------------- 7. 保存结果 ----------------------
# 保存模型
model_save_path = os.path.join(SAVE_DIR, "mm_svm_model_optimized.pkl")
joblib.dump(best_model, model_save_path)

# 保存阈值信息
threshold_info = {'optimal_threshold': optimal_threshold}
threshold_save_path = os.path.join(SAVE_DIR, "optimal_threshold.pkl")
joblib.dump(threshold_info, threshold_save_path)

# 保存预测结果
y_test_save_path = os.path.join(SAVE_DIR, "y_test_optimized.npy")
y_pred_save_path = os.path.join(SAVE_DIR, "y_pred_optimized.npy")
y_pred_proba_save_path = os.path.join(SAVE_DIR, "y_pred_proba_optimized.npy")
np.save(y_test_save_path, y_test)
np.save(y_pred_save_path, y_pred_optimized)
np.save(y_pred_proba_save_path, y_pred_proba)

# 保存详细结果到CSV
results = pd.DataFrame({
    'Model': ['SVM (Optimized)'],
    'Best Parameters': [str(random_search.best_params_)],
    'Class Weights': [str(class_weights_dict)],
    'Optimal Threshold': [f"{optimal_threshold:.4f}"],
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
    'CV Mean F1': [f"{np.mean(fold_metrics['f1']):.4f} ± {np.std(fold_metrics['f1']):.4f}"]
})

results_save_path = os.path.join(SAVE_DIR, 'mm_svm_results_optimized.csv')
results.to_csv(results_save_path, index=False)

# 保存每个折叠的详细结果
fold_results = pd.DataFrame({
    'Fold': range(1, 6),
    'AUC': fold_metrics['auc'],
    'Accuracy': fold_metrics['acc'],
    'Sensitivity': fold_metrics['se'],
    'Specificity': fold_metrics['sp'],
    'F1_Score': fold_metrics['f1']
})

fold_results_save_path = os.path.join(SAVE_DIR, 'mm_svm_cv_fold_results_optimized.csv')
fold_results.to_csv(fold_results_save_path, index=False)

print(f"\n✅ 优化模型已保存到: {model_save_path}")
print(f"✅ 最优阈值已保存到: {threshold_save_path}")
print(f"✅ 详细结果已保存到: {results_save_path}")
print(f"✅ ROC曲线已保存到: {roc_save_path}")

print(f"\n📊 优化总结:")
print(f"   权重调整: {0: 1.8, 1: 1} → {class_weights_dict}")
print(f"   阈值优化: 0.5 → {optimal_threshold:.3f}")
print(f"   特异度提升: {sp_orig:.4f} → {sp:.4f} (+{((sp - sp_orig)/sp_orig*100):.1f}%)")
print(f"   敏感度变化: {se_orig:.4f} → {se:.4f}")

print(f"\n📁 所有文件已保存到: {SAVE_DIR}")
print("\n✅ 优化完成!")import pandas as pd
import numpy as np
import matplotlib
import os  # 新增导入os模块

# 定义保存目录
SAVE_DIR = r"E:\SVM_Results"  # 修改这里为您想要的E盘路径
os.makedirs(SAVE_DIR, exist_ok=True)  # 如果目录不存在则创建

matplotlib.rcParams['font.sans-serif'] = ['arial']
matplotlib.rcParams['axes.unicode_minus'] = False
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, roc_curve, roc_auc_score, f1_score, accuracy_score
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.svm import SVC
import joblib
from scipy.stats import loguniform


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


# ---------------------- 2. 数据预处理 ----------------------
# 读取数据
df_x_train = pd.read_csv(r"E:\训练集测试集\arX_train_maccs.csv", na_values=["?", "NA", " ", ""])
df_y_train = pd.read_csv(r"E:\训练集测试集\ary_train_maccs.csv", na_values=["?", "NA", " ", ""])
df_x_test = pd.read_csv(r"E:\训练集测试集\arX_test_maccs.csv", na_values=["?", "NA", " ", ""])
df_y_test = pd.read_csv(r"E:\训练集测试集\ary_test_maccs.csv", na_values=["?", "NA", " ", ""])

# 检查非数值并转换
for df in [df_x_train, df_x_test]:
    for col in df.columns:
        df[col] = pd.to_numeric(df[col], errors='coerce')

# 合并数据
df_train = pd.concat([df_x_train, df_y_train], axis=1)
df_test = pd.concat([df_x_test, df_y_test], axis=1)

x_train = df_train.iloc[:, :-1].to_numpy()
y_train = df_train.iloc[:, -1].to_numpy()
x_test = df_test.iloc[:, :-1].to_numpy()
y_test = df_test.iloc[:, -1].to_numpy()

# 处理缺失值
imputer = SimpleImputer(strategy='mean')
x_train = imputer.fit_transform(x_train)
x_test = imputer.transform(x_test)

# 标准化数据
scaler = StandardScaler()
x_train_scaled = scaler.fit_transform(x_train)
x_test_scaled = scaler.transform(x_test)

# 保存标准化器 - 修改为保存到E盘
scaler_save_path = os.path.join(SAVE_DIR, "mm_svm_scaler.pkl")
joblib.dump(scaler, scaler_save_path)
print(f"✅ 标准化器已保存到: {scaler_save_path}")

# 检查类别分布
print("\n=== 训练集类别分布 ===")
print(f"Class 0: {np.sum(y_train == 0)}, Class 1: {np.sum(y_train == 1)}")
print(f"正负样本比例: {np.sum(y_train == 1) / len(y_train):.3f}")

# 类别权重
class_weights_dict = {0: 1.8, 1: 1}

# ---------------------- 3. 模型训练和调优 ----------------------
# 定义模型
model = SVC(
    kernel='rbf',
    probability=True,
    max_iter=-1,
    random_state=40,
    class_weight=class_weights_dict
)

# 参数分布
param_dist = {
    'C': loguniform(1e-3, 1e3),
    'gamma': loguniform(1e-4, 1e1),
    'tol': [1e-5, 1e-4, 1e-3],
    'shrinking': [True, False]
}

# 随机搜索
random_search = RandomizedSearchCV(
    model, param_dist, n_iter=50,
    scoring='roc_auc', cv=5, n_jobs=-1, random_state=40
)
random_search.fit(x_train_scaled, y_train)

print("\n=== 最佳参数 ===")
print(random_search.best_params_)

# ---------------------- 4. 交叉验证评估 ----------------------
print("\n=== 交叉验证性能 ===")
kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=40)

fold_metrics = {
    'auc': [], 'acc': [], 'se': [], 'sp': [], 'f1': []
}

for fold, (train_idx, val_idx) in enumerate(kfold.split(x_train_scaled, y_train)):
    y_val = y_train[val_idx]

    # 训练该折叠的模型
    fold_model = SVC(**random_search.best_params_,
                     kernel='rbf', probability=True,
                     class_weight=class_weights_dict,
                     random_state=40)
    fold_model.fit(x_train_scaled[train_idx], y_train[train_idx])

    # 预测
    y_val_pred_proba = fold_model.predict_proba(x_train_scaled[val_idx])[:, 1]
    y_val_pred = fold_model.predict(x_train_scaled[val_idx])

    # 计算指标
    fold_auc = roc_auc_score(y_val, y_val_pred_proba)
    fold_acc = accuracy_score(y_val, y_val_pred)
    fold_se = calculate_se(y_val, y_val_pred)
    fold_sp = calculate_sp(y_val, y_val_pred)
    fold_f1 = f1_score(y_val, y_val_pred)

    # 存储指标
    fold_metrics['auc'].append(fold_auc)
    fold_metrics['acc'].append(fold_acc)
    fold_metrics['se'].append(fold_se)
    fold_metrics['sp'].append(fold_sp)
    fold_metrics['f1'].append(fold_f1)

    # 打印该折叠结果
    print(f"Fold {fold + 1} - AUC: {fold_auc:.4f}, ACC: {fold_acc:.4f}, "
          f"SE: {fold_se:.4f}, SP: {fold_sp:.4f}, F1: {fold_f1:.4f}")

# 计算交叉验证平均性能
print("\n=== 交叉验证平均性能 ===")
for metric_name, values in fold_metrics.items():
    mean_value = np.mean(values)
    std_value = np.std(values)
    print(f"{metric_name.upper()}: {mean_value:.4f} ± {std_value:.4f}")

# 使用最佳模型在整个训练集上的性能
cv_pred_proba = random_search.predict_proba(x_train_scaled)[:, 1]
cv_pred = random_search.predict(x_train_scaled)
cv_auc = roc_auc_score(y_train, cv_pred_proba)
cv_acc = accuracy_score(y_train, cv_pred)
cv_se = calculate_se(y_train, cv_pred)
cv_sp = calculate_sp(y_train, cv_pred)
cv_f1 = f1_score(y_train, cv_pred)

print(f"\n训练集整体性能 - AUC: {cv_auc:.4f}, ACC: {cv_acc:.4f}, "
      f"SE: {cv_se:.4f}, SP: {cv_sp:.4f}, F1: {cv_f1:.4f}")

# ---------------------- 5. 测试集评估 ----------------------
best_model = random_search.best_estimator_
y_pred_proba = best_model.predict_proba(x_test_scaled)[:, 1]
y_pred = best_model.predict(x_test_scaled)

# 计算测试集指标
auc = roc_auc_score(y_test, y_pred_proba)
acc = accuracy_score(y_test, y_pred)
se = calculate_se(y_test, y_pred)
sp = calculate_sp(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

# 混淆矩阵和MCC
tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
mcc = (tp * tn - fp * fn) / np.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn)) if (tp + fp) * (tp + fn) * (
            tn + fp) * (tn + fn) != 0 else 0.0

# 输出结果
print("\n" + "=" * 50)
print("测试集评估结果:")
print("=" * 50)
print(f'AUC: {auc:.4f}')
print(f'ACC: {acc:.4f}')
print(f'SE: {se:.4f}')
print(f'SP: {sp:.4f}')
print(f'F1: {f1:.4f}')
print(f'MCC: {mcc:.4f}')
print(f'TP: {tp}, FP: {fp}, TN: {tn}, FN: {fn}')

# ---------------------- 6. ROC曲线 ----------------------
fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'SVM (AUC = {auc:.2f})')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate', fontsize=12)
plt.ylabel('True Positive Rate', fontsize=12)
plt.title('Receiver Operating Characteristic (ROC) Curve', fontsize=14)
plt.legend(loc="lower right")
plt.grid(True, alpha=0.3)

# 修改ROC曲线保存路径
roc_save_path = os.path.join(SAVE_DIR, 'mm_svm_ROC.pdf')
plt.savefig(roc_save_path, dpi=300, bbox_inches='tight')
plt.close()

# ---------------------- 7. 保存结果 ----------------------
# 保存模型 - 修改为保存到E盘
model_save_path = os.path.join(SAVE_DIR, "mm_svm_model.pkl")
joblib.dump(best_model, model_save_path)

# 保存预测结果 - 修改为保存到E盘
y_test_save_path = os.path.join(SAVE_DIR, "y_test_model_mm_svm.npy")
y_pred_save_path = os.path.join(SAVE_DIR, "y_pred_model_mm_svm.npy")
np.save(y_test_save_path, y_test)
np.save(y_pred_save_path, y_pred_proba)

# 保存详细结果到CSV - 修改为保存到E盘
results = pd.DataFrame({
    'Model': ['SVM'],
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
    'Training AUC': [f"{cv_auc:.4f}"],
    'Training Accuracy': [f"{cv_acc:.4f}"],
    'Training SE': [f"{cv_se:.4f}"],
    'Training SP': [f"{cv_sp:.4f}"],
    'Training F1': [f"{cv_f1:.4f}"]
})

results_save_path = os.path.join(SAVE_DIR, 'mm_svm_results.csv')
results.to_csv(results_save_path, index=False)

# 保存每个折叠的详细结果 - 修改为保存到E盘
fold_results = pd.DataFrame({
    'Fold': range(1, 6),
    'AUC': fold_metrics['auc'],
    'Accuracy': fold_metrics['acc'],
    'Sensitivity': fold_metrics['se'],
    'Specificity': fold_metrics['sp'],
    'F1_Score': fold_metrics['f1']
})

fold_results_save_path = os.path.join(SAVE_DIR, 'mm_svm_cv_fold_results.csv')
fold_results.to_csv(fold_results_save_path, index=False)

print(f"\n✅ 模型已保存到: {model_save_path}")
print(f"✅ 标准化器已保存到: {scaler_save_path}")
print(f"✅ 测试集预测结果已保存到: {y_test_save_path} 和 {y_pred_save_path}")
print(f"✅ 详细结果已保存到: {results_save_path}")
print(f"✅ 交叉验证各折叠结果已保存到: {fold_results_save_path}")
print(f"✅ ROC曲线已保存到: {roc_save_path}")

# 显示所有保存的文件
print(f"\n📁 所有文件已保存到: {SAVE_DIR}")
print("📋 文件列表:")
for file in os.listdir(SAVE_DIR):
    file_path = os.path.join(SAVE_DIR, file)
    print(f"  - {file} ({os.path.getsize(file_path)} 字节)")

print("\n✅ 所有结果已保存!")
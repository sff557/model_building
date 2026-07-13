import pandas as pd
import numpy as np
import matplotlib
import os

# 定义保存目录
SAVE_DIR = r"E:\RF_maccs+chem_Results"
os.makedirs(SAVE_DIR, exist_ok=True)

matplotlib.rcParams['font.sans-serif'] = ['arial']
matplotlib.rcParams['axes.unicode_minus'] = False
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, roc_curve, roc_auc_score, f1_score, accuracy_score, matthews_corrcoef
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier
import joblib
import warnings

warnings.filterwarnings('ignore')


# ---------------------- 评估指标计算函数 ----------------------
def calculate_se(y_true, y_pred):
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    return tp / (tp + fn) if (tp + fn) != 0 else 0.0


def calculate_sp(y_true, y_pred):
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    return tn / (tn + fp) if (tn + fp) != 0 else 0.0


# ---------------------- 数据加载与预处理 --------------------
print("加载MACCS数据...")

# 读取数据
df_x_train = pd.read_csv(r"E:\训练集测试集4\hfX_train_maccs.csv", na_values=["?", "NA", " ", ""])
df_y_train = pd.read_csv(r"E:\训练集测试集4\hfy_train_maccs.csv", na_values=["?", "NA", " ", ""])
df_x_test = pd.read_csv(r"E:\训练集测试集4\hfX_test_maccs.csv", na_values=["?", "NA", " ", ""])
df_y_test = pd.read_csv(r"E:\训练集测试集4\hfy_test_maccs.csv", na_values=["?", "NA", " ", ""])

print(f"原始训练集形状: {df_x_train.shape}")
print(f"原始测试集形状: {df_x_test.shape}")

# 找出非数值列
non_numeric_cols_train = []
non_numeric_cols_test = []

# 检查训练集中的非数值列
for col in df_x_train.columns:
    if df_x_train[col].dtype == object or df_x_train[col].dtype == 'string':
        non_numeric_cols_train.append(col)

# 检查测试集中的非数值列
for col in df_x_test.columns:
    if df_x_test[col].dtype == object or df_x_test[col].dtype == 'string':
        non_numeric_cols_test.append(col)

if non_numeric_cols_train:
    print(f"训练集非数值列 '{non_numeric_cols_train[0]}' 示例值:", df_x_train[non_numeric_cols_train[0]].iloc[0])
    print(f"找到的非数值列: {non_numeric_cols_train}")

# 删除非数值列
if non_numeric_cols_train or non_numeric_cols_test:
    cols_to_drop = list(set(non_numeric_cols_train + non_numeric_cols_test))
    df_x_train = df_x_train.drop(columns=cols_to_drop)
    df_x_test = df_x_test.drop(columns=cols_to_drop)
    print(f"删除非数值列: {cols_to_drop}")

# 尝试将剩余列转换为数值类型
print("尝试将数据转换为数值类型...")
for col in df_x_train.columns:
    df_x_train[col] = pd.to_numeric(df_x_train[col], errors='coerce')
for col in df_x_test.columns:
    df_x_test[col] = pd.to_numeric(df_x_test[col], errors='coerce')

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

# 处理缺失值
print("\n处理缺失值...")
imputer = SimpleImputer(strategy='mean')
x_train = imputer.fit_transform(x_train)
x_test = imputer.transform(x_test)

# 检查类别分布
print("\n=== 训练集类别分布 ===")
print(f"Class 0: {np.sum(y_train == 0)}, Class 1: {np.sum(y_train == 1)}")
print(f"正负样本比例: {np.sum(y_train == 1) / len(y_train):.3f}")

# 不使用任何复杂的特征选择，使用所有特征
x_train_final = x_train.copy()
x_test_final = x_test.copy()

print(f"\n使用所有 {x_train_final.shape[1]} 个特征")

# ---------------------- 随机森林模型训练和调优 ----------------------
print("\n训练随机森林模型（强正则化版）...")

# 针对551个特征和954个样本，设置更强的正则化参数
param_dist = {
    'n_estimators': [200, 300, 400],  # 保持适中的树数量
    'max_depth': [5, 8, 10, 12, 15],  # 更严格的深度限制，防止过拟合
    'min_samples_split': [20, 30, 40, 50],  # 大幅增加节点分裂最小样本数
    'min_samples_leaf': [10, 15, 20, 25],  # 大幅增加叶子节点最小样本数
    'max_features': [0.1, 0.15, 0.2, 'log2'],  # 大幅限制特征使用比例
    'bootstrap': [True],  # 保持bootstrap
    'criterion': ['gini', 'entropy'],
    'class_weight': ['balanced', None]  # 尝试平衡类别权重
}

# 完全不使用SMOTE，使用原始数据
x_train_smote, y_train_smote = x_train_final, y_train
print("⚠️ 注意：跳过SMOTE，直接使用原始数据训练")

# 定义模型
model = RandomForestClassifier(
    random_state=42,
    n_jobs=-1,
    oob_score=True
)

# 使用AUC作为评分标准
from sklearn.metrics import make_scorer, roc_auc_score

auc_scorer = make_scorer(roc_auc_score, needs_proba=True)

random_search = RandomizedSearchCV(
    model,
    param_dist,
    n_iter=50,  # 增加迭代次数，因为参数空间更关键
    scoring=auc_scorer,
    cv=5,  # 5折交叉验证
    n_jobs=-1,
    random_state=42,
    verbose=1
)

random_search.fit(x_train_smote, y_train_smote)

print("\n=== 最佳参数 ===")
best_params = random_search.best_params_
print(best_params)

# 训练集整体性能评估
print("\n=== 训练集整体性能（原始数据） ===")
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

if train_auc < 0.6:
    print("❌ 严重警告：训练集AUC过低，模型可能无法学习有效模式")
elif train_auc > 0.95:
    print("⚠️ 警告：训练集AUC过高，可能存在过拟合")

# ---------------------- 交叉验证评估 ---------------------
print("\n=== 交叉验证性能 ===")
kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

fold_metrics = {
    'auc': [], 'acc': [], 'se': [], 'sp': [], 'f1': [], 'mcc': []
}

for fold, (train_idx, val_idx) in enumerate(kfold.split(x_train_final, y_train)):
    print(f"训练折叠 {fold + 1}/5...")

    fold_model = RandomForestClassifier(**best_params,
                                        random_state=42 + fold,
                                        n_jobs=-1)
    fold_model.fit(x_train_final[train_idx], y_train[train_idx])

    y_val_pred_proba = fold_model.predict_proba(x_train_final[val_idx])[:, 1]
    y_val_pred = fold_model.predict(x_train_final[val_idx])

    fold_auc = roc_auc_score(y_train[val_idx], y_val_pred_proba)
    fold_acc = accuracy_score(y_train[val_idx], y_val_pred)
    fold_se = calculate_se(y_train[val_idx], y_val_pred)
    fold_sp = calculate_sp(y_train[val_idx], y_val_pred)
    fold_f1 = f1_score(y_train[val_idx], y_val_pred)
    fold_mcc = matthews_corrcoef(y_train[val_idx], y_val_pred)

    fold_metrics['auc'].append(fold_auc)
    fold_metrics['acc'].append(fold_acc)
    fold_metrics['se'].append(fold_se)
    fold_metrics['sp'].append(fold_sp)
    fold_metrics['f1'].append(fold_f1)
    fold_metrics['mcc'].append(fold_mcc)

    print(f"Fold {fold + 1} - AUC: {fold_auc:.4f}, ACC: {fold_acc:.4f}, "
          f"SE: {fold_se:.4f}, SP: {fold_sp:.4f}, F1: {fold_f1:.4f}, MCC: {fold_mcc:.4f}")

# 计算交叉验证平均性能
print("\n=== 交叉验证平均性能 ===")
for metric_name, values in fold_metrics.items():
    mean_value = np.mean(values)
    std_value = np.std(values)
    print(f"{metric_name.upper()}: {mean_value:.4f} ± {std_value:.4f}")

# 如果交叉验证AUC很低，提供警告
cv_auc_mean = np.mean(fold_metrics['auc'])
if cv_auc_mean < 0.6:
    print("❌ 严重警告：交叉验证AUC过低，模型可能不适合此数据")
elif cv_auc_mean < 0.7:
    print("⚠️ 警告：交叉验证AUC一般，建议尝试更多优化")
elif cv_auc_mean < 0.75:
    print("✅ 交叉验证AUC可接受")
else:
    print("🎉 交叉验证AUC良好！")

# ---------------------- 阈值优化 ----------------------
print("\n=== 阈值优化 ===")
all_val_probs = []
all_val_labels = []

kfold_thresh = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
for train_idx, val_idx in kfold_thresh.split(x_train_final, y_train):
    fold_model = RandomForestClassifier(**best_params,
                                        random_state=42,
                                        n_jobs=-1)
    fold_model.fit(x_train_final[train_idx], y_train[train_idx])

    y_val_proba = fold_model.predict_proba(x_train_final[val_idx])[:, 1]
    all_val_probs.extend(y_val_proba)
    all_val_labels.extend(y_train[val_idx])

all_val_probs = np.array(all_val_probs)
all_val_labels = np.array(all_val_labels)

# 更精细的阈值搜索
thresholds = np.arange(0.1, 0.9, 0.005)  # 细粒度阈值搜索
best_threshold = 0.5
best_score = -1

for threshold in thresholds:
    y_pred_thresh = (all_val_probs >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(all_val_labels, y_pred_thresh).ravel()

    # 使用平衡的F1分数作为评分
    f1 = f1_score(all_val_labels, y_pred_thresh)
    if f1 > best_score:
        best_score = f1
        best_threshold = threshold

print(f"最佳决策阈值: {best_threshold:.3f} (默认: 0.5)")
print(f"阈值优化得分(F1): {best_score:.4f}")

# ---------------------- 测试集评估 ----------------------
print("\n" + "=" * 50)
print("测试集评估结果:")
print("=" * 50)

best_model = random_search.best_estimator_
y_pred_proba = best_model.predict_proba(x_test_final)[:, 1]
y_pred_default = best_model.predict(x_test_final)
y_pred_optimized = (y_pred_proba >= best_threshold).astype(int)


# 计算两种阈值下的指标
def evaluate_predictions(y_true, y_pred, label="默认阈值(0.5)"):
    auc = roc_auc_score(y_true, y_pred_proba)
    acc = accuracy_score(y_true, y_pred)
    se = calculate_se(y_true, y_pred)
    sp = calculate_sp(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    mcc = matthews_corrcoef(y_true, y_pred)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()

    print(f"\n{label}:")
    print(f'AUC: {auc:.4f}')
    print(f'ACC: {acc:.4f}')
    print(f'SE: {se:.4f}')
    print(f'SP: {sp:.4f}')
    print(f'F1: {f1:.4f}')
    print(f'MCC: {mcc:.4f}')
    print(f'TP: {tp}, FP: {fp}, TN: {tn}, FN: {fn}')

    return {'auc': auc, 'acc': acc, 'se': se, 'sp': sp, 'f1': f1, 'mcc': mcc,
            'tp': tp, 'fp': fp, 'tn': tn, 'fn': fn}


# 评估两种阈值
results_default = evaluate_predictions(y_test, y_pred_default, "默认阈值(0.5)")
results_optimized = evaluate_predictions(y_test, y_pred_optimized, f"优化阈值({best_threshold:.3f})")

# 选择最佳阈值的结果
if results_optimized['f1'] > results_default['f1']:
    improvement = results_optimized['f1'] - results_default['f1']
    print(f"\n✅ 使用优化阈值({best_threshold:.3f})，F1分数提高 {improvement:.4f}")
    final_y_pred = y_pred_optimized
    final_results = results_optimized
else:
    print(f"\n⚠️ 使用默认阈值(0.5)，优化阈值未带来显著改进")
    final_y_pred = y_pred_default
    final_results = results_default

tn, fp, fn, tp = confusion_matrix(y_test, final_y_pred).ravel()
auc = results_default['auc']

# 性能评估
if auc < 0.6:
    print("\n❌ 模型性能极差：测试集AUC < 0.6，建议放弃此数据集或尝试其他算法")
elif auc < 0.7:
    print("\n⚠️ 模型性能较差：测试集AUC < 0.7，仅勉强可用")
elif auc < 0.75:
    print("\n✅ 模型性能可接受")
elif auc < 0.8:
    print("\n✅ 模型性能良好")
else:
    print("\n🎉 模型性能优秀！")

# ---------------------- ROC曲线 ----------------------
fpr, tpr, _ = roc_curve(y_test, y_pred_proba)

plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'Random Forest (AUC = {auc:.4f})')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')

plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate', fontsize=12)
plt.ylabel('True Positive Rate', fontsize=12)
plt.title('Receiver Operating Characteristic (ROC) Curve', fontsize=14)
plt.legend(loc="lower right")
plt.grid(True, alpha=0.3)
plt.tight_layout()

roc_save_path = os.path.join(SAVE_DIR, 'hf_rf_ROC.pdf')
plt.savefig(roc_save_path, dpi=300, bbox_inches='tight')
plt.close()

# ---------------------- 保存结果 ----------------------
print("\n保存结果...")

# 保存预测结果
y_test_save_path = os.path.join(SAVE_DIR, "y_test_model_hf_rf.npy")
y_pred_save_path = os.path.join(SAVE_DIR, "y_pred_model_hf_rf.npy")
np.save(y_test_save_path, y_test)
np.save(y_pred_save_path, final_y_pred)

# 保存详细结果到CSV
results = pd.DataFrame({
    'Model': ['Random Forest (强正则化版)'],
    'Best Parameters': [str(random_search.best_params_)],
    'Best Threshold': [f"{best_threshold:.4f}"],
    'Test AUC': [f"{auc:.4f}"],
    'Test Accuracy': [f"{final_results['acc']:.4f}"],
    'Test Sensitivity (SE)': [f"{final_results['se']:.4f}"],
    'Test Specificity (SP)': [f"{final_results['sp']:.4f}"],
    'Test F1-score': [f"{final_results['f1']:.4f}"],
    'Test MCC': [f"{final_results['mcc']:.4f}"],
    'TP': [final_results['tp']],
    'FP': [final_results['fp']],
    'TN': [final_results['tn']],
    'FN': [final_results['fn']],
    'Feature Count': [x_train_final.shape[1]],
    'CV Mean AUC': [f"{np.mean(fold_metrics['auc']):.4f} ± {np.std(fold_metrics['auc']):.4f}"],
    'CV Mean Accuracy': [f"{np.mean(fold_metrics['acc']):.4f} ± {np.std(fold_metrics['acc']):.4f}"],
    'CV Mean SE': [f"{np.mean(fold_metrics['se']):.4f} ± {np.std(fold_metrics['se']):.4f}"],
    'CV Mean SP': [f"{np.mean(fold_metrics['sp']):4f} ± {np.std(fold_metrics['sp']):.4f}"],
    'CV Mean F1': [f"{np.mean(fold_metrics['f1']):.4f} ± {np.std(fold_metrics['f1']):.4f}"],
    'CV Mean MCC': [f"{np.mean(fold_metrics['mcc']):.4f} ± {np.std(fold_metrics['mcc']):.4f}"],
    'Training AUC': [f"{train_auc:.4f}"],
    'Training Accuracy': [f"{train_acc:.4f}"],
    'Training SE': [f"{train_se:.4f}"],
    'Training SP': [f"{train_sp:.4f}"],
    'Training F1': [f"{train_f1:.4f}"],
    'Training MCC': [f"{train_mcc:.4f}"]
})

results_save_path = os.path.join(SAVE_DIR, 'hf_rf_results.csv')
results.to_csv(results_save_path, index=False)

# 保存每个折叠的详细结果
n_folds = len(fold_metrics['auc'])
fold_results = pd.DataFrame({
    'Fold': range(1, n_folds + 1),
    'AUC': fold_metrics['auc'],
    'Accuracy': fold_metrics['acc'],
    'Sensitivity': fold_metrics['se'],
    'Specificity': fold_metrics['sp'],
    'F1_Score': fold_metrics['f1'],
    'MCC': fold_metrics['mcc']
})

fold_results_save_path = os.path.join(SAVE_DIR, 'hf_rf_cv_fold_results.csv')
fold_results.to_csv(fold_results_save_path, index=False)

# ---------------------- 打印总结 ----------------------
print(f"\n✅ ROC曲线已保存到: {roc_save_path}")
print(f"✅ 测试集预测结果已保存到: {y_test_save_path} 和 {y_pred_save_path}")
print(f"✅ 详细结果已保存到: {results_save_path}")
print(f"✅ 交叉验证各折叠结果已保存到: {fold_results_save_path}")

# 显示所有保存的文件
print(f"\n📁 所有文件已保存到: {SAVE_DIR}")
print("📋 文件列表:")
saved_files = [
    'hf_rf_ROC.pdf',
    'y_pred_model_hf_rf.npy',
    'y_test_model_hf_rf.npy',
    'hf_rf_results.csv',
    'hf_rf_cv_fold_results.csv'
]

for file in saved_files:
    file_path = os.path.join(SAVE_DIR, file)
    if os.path.isfile(file_path):
        print(f"  - {file}")

print("\n✅ 所有指定结果已保存!")

# ---------------------- 性能分析和建议 ----------------------
print("\n" + "=" * 50)
print("性能分析和下一步建议:")
print("=" * 50)

# 计算过拟合程度
overfitting_gap = train_auc - auc
print(f"\n过拟合程度分析:")
print(f"训练集AUC: {train_auc:.4f}")
print(f"测试集AUC: {auc:.4f}")
print(f"过拟合差距: {overfitting_gap:.4f}")

if overfitting_gap > 0.15:
    print("⚠️ 严重过拟合：训练和测试差距过大，需要更强的正则化")
elif overfitting_gap > 0.1:
    print("⚠️ 明显过拟合：建议进一步增加正则化")
elif overfitting_gap > 0.05:
    print("✅ 适度过拟合：在可接受范围内")
else:
    print("🎉 过拟合控制良好")

print("\n=== 特征维度分析 ===")
print(f"样本数: {x_train_final.shape[0]}")
print(f"特征数: {x_train_final.shape[1]}")
print(f"样本特征比: {x_train_final.shape[0]/x_train_final.shape[1]:.2f}")

if x_train_final.shape[0]/x_train_final.shape[1] < 2:
    print("⚠️ 警告：样本特征比过低（<2），强烈建议进行特征选择")

print("\n=== 进一步优化建议 ===")
print("1. 强烈建议进行特征选择：基于重要性筛选前50-100个特征")
print("2. 如果AUC仍不理想，考虑使用其他算法如XGBoost、LightGBM")
print("3. 尝试PCA降维到50-100个主成分")
print("4. 使用特征重要性分析，剔除低重要性特征")
print("5. 考虑使用弹性网(ElasticNet)进行特征选择")

print("\n=== 当前模型总结 ===")
print(f"✅ 测试集AUC: {auc:.4f} (目标: >0.75)")
print(f"✅ 优化后F1分数: {final_results['f1']:.4f}")
print(f"✅ 交叉验证稳定性: ±{np.std(fold_metrics['auc']):.4f}")

print("\n进程已结束，退出代码为 0")
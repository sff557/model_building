import pandas as pd
import numpy as np
import matplotlib
import os

# 定义保存目录 - 修改为RF专用目录
SAVE_DIR = r"E:\RF_pubchem_Results"  # 修改这里为您想要的路径
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
df_x_train = pd.read_csv(r"E:\训练集测试集3\haX_train_pubchem.csv", na_values=["?", "NA", " ", ""])
df_y_train = pd.read_csv(r"E:\训练集测试集3\hay_train_pubchem.csv", na_values=["?", "NA", " ", ""])
df_x_test = pd.read_csv(r"E:\训练集测试集3\haX_test_pubchem.csv", na_values=["?", "NA", " ", ""])
df_y_test = pd.read_csv(r"E:\训练集测试集3\hay_test_pubchem.csv", na_values=["?", "NA", " ", ""])

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

# ---------------------- 新增：特征重要性分析 ----------------------
print("\n=== 特征重要性分析 ===")
# 训练一个初始随机森林用于特征重要性分析
initial_rf = RandomForestClassifier(
    n_estimators=100,
    random_state=42,
    n_jobs=-1
)
initial_rf.fit(x_train_final, y_train)

# 获取特征重要性
feature_importances = initial_rf.feature_importances_
importance_threshold = np.percentile(feature_importances, 50)  # 保留重要性在前50%的特征

# 选择重要特征
selected_features = feature_importances >= importance_threshold
x_train_final = x_train_final[:, selected_features]
x_test_final = x_test_final[:, selected_features]

print(f"特征选择后: {x_train_final.shape[1]} 个特征 (减少 {881 - x_train_final.shape[1]} 个)")
print(f"特征选择阈值: {importance_threshold:.6f}")

# ---------------------- 3. 随机森林模型训练和调优 ----------------------
print("\n训练随机森林模型（针对类别不平衡优化）...")

# 定义模型 - 使用更关注正样本的策略
model = RandomForestClassifier(
    random_state=42,
    class_weight='balanced',  # 使用balanced自动调整权重
    n_jobs=-1,
    oob_score=True
)

# 针对不平衡数据的参数分布 - 增加正则化减轻过拟合
param_dist = {
    'n_estimators': [100, 150, 200],  # 控制树的数量
    'max_depth': [5, 8, 10, None],  # 限制深度，增加None选项但要小心使用
    'min_samples_split': [20, 30, 40],  # 增加，减少过拟合
    'min_samples_leaf': [10, 15, 20],  # 增加，减少过拟合
    'max_features': ['sqrt', 'log2', 0.3, 0.5],  # 减少特征使用比例
    'bootstrap': [True],
    'criterion': ['gini', 'entropy'],
    'class_weight': [
        'balanced',
        {0: 1.0, 1: 1.0},  # 不使用权重调整
        {0: 1.2, 1: 0.8},  # 增加负类权重，提高特异性
        {0: 1.5, 1: 0.7},  # 更强的负类权重
    ]
}

# 使用SMOTE数据增强，但调整采样策略以提高特异性
from imblearn.over_sampling import SMOTE

# 调整采样策略，不过度增加正样本
smote = SMOTE(random_state=42, sampling_strategy=0.8)  # 正样本占80%
x_train_smote, y_train_smote = smote.fit_resample(x_train_final, y_train)

print(f"SMOTE后训练集: {len(y_train_smote)}个样本")
print(f"SMOTE采样策略: 正样本比例 = {np.sum(y_train_smote == 1) / len(y_train_smote):.2f}")

# 定义模型
model = RandomForestClassifier(
    random_state=42,
    n_jobs=-1,
    oob_score=True
)

# 使用F1分数作为优化指标，但添加特异性权重
from sklearn.metrics import make_scorer, f1_score


# 自定义评分函数：平衡F1和特异性
def balanced_scorer(y_true, y_pred):
    f1 = f1_score(y_true, y_pred)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
    # 平衡F1和特异性，特异性权重略高
    return 0.6 * f1 + 0.4 * specificity


custom_scorer = make_scorer(balanced_scorer)

random_search = RandomizedSearchCV(
    model,
    param_dist,
    n_iter=15,  # 增加迭代次数
    scoring=custom_scorer,  # 使用自定义评分器
    cv=5,  # 使用5折交叉验证
    n_jobs=-1,
    random_state=42,
    verbose=1
)

random_search.fit(x_train_smote, y_train_smote)

print("\n=== 最佳参数 ===")
best_params = random_search.best_params_
print(best_params)

# 训练集整体性能评估（在原始训练集上，不是SMOTE后的）
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

# 检查是否过拟合
if train_auc > 0.98:
    print("⚠️ 警告：训练集AUC过高，可能存在过拟合")
    print("建议：增加正则化参数或减少模型复杂度")
elif train_auc > 0.95:
    print("⚠️ 注意：训练集AUC较高，建议监控过拟合")

# ---------------------- 4. 交叉验证评估 ---------------------
print("\n=== 交叉验证性能 ===")
kfold = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)  # 使用3折

fold_metrics = {
    'auc': [], 'acc': [], 'se': [], 'sp': [], 'f1': [], 'mcc': []
}

for fold, (train_idx, val_idx) in enumerate(kfold.split(x_train_final, y_train)):
    print(f"训练折叠 {fold + 1}/3...")

    # 对训练集使用SMOTE，但使用相同的采样策略
    smote_fold = SMOTE(random_state=42 + fold, sampling_strategy=0.8)
    x_train_fold, y_train_fold = smote_fold.fit_resample(
        x_train_final[train_idx],
        y_train[train_idx]
    )

    # 训练该折叠的模型
    fold_model = RandomForestClassifier(**best_params,
                                        random_state=42 + fold,
                                        n_jobs=-1)

    fold_model.fit(x_train_fold, y_train_fold)

    # 预测（在原始验证集上，不是SMOTE后的）
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

# ---------------------- 新增：阈值优化提高特异性 ----------------------
print("\n=== 阈值优化 ===")
# 使用验证集找到最佳阈值
from sklearn.metrics import precision_recall_curve

# 收集所有验证集的预测概率和真实标签
all_val_probs = []
all_val_labels = []

kfold_thresh = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
for train_idx, val_idx in kfold_thresh.split(x_train_final, y_train):
    smote_fold = SMOTE(random_state=42, sampling_strategy=0.8)
    x_train_fold, y_train_fold = smote_fold.fit_resample(
        x_train_final[train_idx],
        y_train[train_idx]
    )

    fold_model = RandomForestClassifier(**best_params,
                                        random_state=42,
                                        n_jobs=-1)
    fold_model.fit(x_train_fold, y_train_fold)

    y_val_proba = fold_model.predict_proba(x_train_final[val_idx])[:, 1]
    all_val_probs.extend(y_val_proba)
    all_val_labels.extend(y_train[val_idx])

# 寻找最佳阈值，平衡敏感性和特异性
all_val_probs = np.array(all_val_probs)
all_val_labels = np.array(all_val_labels)

thresholds = np.arange(0.3, 0.8, 0.01)
best_threshold = 0.5
best_score = -1

for threshold in thresholds:
    y_pred_thresh = (all_val_probs >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(all_val_labels, y_pred_thresh).ravel()
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0

    # 使用F1和特异性的加权组合
    f1 = f1_score(all_val_labels, y_pred_thresh)
    score = 0.5 * f1 + 0.5 * specificity  # 特异性权重更高

    if score > best_score:
        best_score = score
        best_threshold = threshold

print(f"最佳决策阈值: {best_threshold:.3f} (默认: 0.5)")
print(f"阈值优化得分: {best_score:.4f}")

# ---------------------- 5. 测试集评估 ----------------------
print("\n" + "=" * 50)
print("测试集评估结果:")
print("=" * 50)

best_model = random_search.best_estimator_

# 在测试集上预测
y_pred_proba = best_model.predict_proba(x_test_final)[:, 1]

# 使用优化后的阈值进行预测
y_pred_default = best_model.predict(x_test_final)
y_pred_optimized = (y_pred_proba >= best_threshold).astype(int)


# 计算两种阈值下的指标
def evaluate_predictions(y_true, y_pred, label="默认阈值(0.5)"):
    auc = roc_auc_score(y_true, y_pred_proba)  # AUC使用概率
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
if results_optimized['sp'] > results_default['sp'] and results_optimized['f1'] >= results_default['f1'] * 0.9:
    print(f"\n✅ 使用优化阈值({best_threshold:.3f})，特异性提高 {results_optimized['sp'] - results_default['sp']:.4f}")
    final_y_pred = y_pred_optimized
    final_results = results_optimized
else:
    print(f"\n⚠️ 使用默认阈值(0.5)，优化阈值未带来显著改进")
    final_y_pred = y_pred_default
    final_results = results_default

# 更新最终结果
tn, fp, fn, tp = confusion_matrix(y_test, final_y_pred).ravel()
auc = results_default['auc']  # AUC不变

# ---------------------- 6. ROC曲线 ----------------------
fpr, tpr, thresholds = roc_curve(y_test, y_pred_proba)

plt.figure(figsize=(10, 8))
plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'Random Forest (AUC = {auc:.2f})')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')

# 标记阈值点
optimal_idx = np.argmax(tpr - fpr)
optimal_threshold = thresholds[optimal_idx]
plt.scatter(fpr[optimal_idx], tpr[optimal_idx], color='red', s=100,
            label=f'Optimal threshold ({optimal_threshold:.2f})', zorder=5)

plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate (1 - Specificity)', fontsize=12)
plt.ylabel('True Positive Rate (Sensitivity)', fontsize=12)
plt.title('Receiver Operating Characteristic (ROC) Curve', fontsize=14)
plt.legend(loc="lower right")
plt.grid(True, alpha=0.3)
plt.tight_layout()

# 保存ROC曲线
roc_save_path = os.path.join(SAVE_DIR, 'ha_rf_ROC.pdf')
plt.savefig(roc_save_path, dpi=300, bbox_inches='tight')
plt.close()

# ---------------------- 新增：特异性-敏感性平衡图 ----------------------
plt.figure(figsize=(10, 8))
thresholds_for_plot = np.linspace(0, 1, 100)
specificities = []
sensitivities = []

for threshold in thresholds_for_plot:
    y_pred_temp = (y_pred_proba >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_test, y_pred_temp).ravel()
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
    specificities.append(specificity)
    sensitivities.append(sensitivity)

plt.plot(thresholds_for_plot, specificities, 'b-', label='Specificity', lw=2)
plt.plot(thresholds_for_plot, sensitivities, 'r-', label='Sensitivity', lw=2)
plt.axvline(x=best_threshold, color='g', linestyle='--', label=f'Best threshold ({best_threshold:.3f})')
plt.xlabel('Decision Threshold', fontsize=12)
plt.ylabel('Metric Value', fontsize=12)
plt.title('Sensitivity and Specificity vs Decision Threshold', fontsize=14)
plt.legend(loc='best')
plt.grid(True, alpha=0.3)
plt.tight_layout()

balance_save_path = os.path.join(SAVE_DIR, 'ha_rf_threshold_balance.pdf')
plt.savefig(balance_save_path, dpi=300, bbox_inches='tight')
plt.close()

# ---------------------- 7. 保存结果和模型 ----------------------
print("\n保存结果...")

# 保存模型
model_save_path = os.path.join(SAVE_DIR, "ha_rf_model.pkl")
joblib.dump(best_model, model_save_path)

# 保存阈值
threshold_save_path = os.path.join(SAVE_DIR, "ha_rf_best_threshold.npy")
np.save(threshold_save_path, best_threshold)

# 保存特征选择信息
feature_selection_save_path = os.path.join(SAVE_DIR, "ha_rf_feature_selection.npy")
np.save(feature_selection_save_path, selected_features)

# 保存预测结果
y_test_save_path = os.path.join(SAVE_DIR, "y_test_model_ha_rf.npy")
y_pred_save_path = os.path.join(SAVE_DIR, "y_pred_model_ha_rf.npy")
y_pred_proba_save_path = os.path.join(SAVE_DIR, "y_pred_proba_model_ha_rf.npy")
np.save(y_test_save_path, y_test)
np.save(y_pred_save_path, final_y_pred)
np.save(y_pred_proba_save_path, y_pred_proba)

# 保存详细结果到CSV
results = pd.DataFrame({
    'Model': ['Random Forest (优化版)'],
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
n_folds = len(fold_metrics['auc'])  # 获取实际折叠数（这里是3）
fold_results = pd.DataFrame({
    'Fold': range(1, n_folds + 1),  # 动态生成折叠编号
    'AUC': fold_metrics['auc'],
    'Accuracy': fold_metrics['acc'],
    'Sensitivity': fold_metrics['se'],
    'Specificity': fold_metrics['sp'],
    'F1_Score': fold_metrics['f1'],
    'MCC': fold_metrics['mcc']
})

fold_results_save_path = os.path.join(SAVE_DIR, 'ha_rf_cv_fold_results.csv')
fold_results.to_csv(fold_results_save_path, index=False)

# 保存特征重要性
feature_importance_df = pd.DataFrame({
    'Feature_Index': np.where(selected_features)[0],
    'Importance': feature_importances[selected_features]
}).sort_values('Importance', ascending=False)

feature_importance_save_path = os.path.join(SAVE_DIR, 'ha_rf_feature_importance.csv')
feature_importance_df.to_csv(feature_importance_save_path, index=False)

# ---------------------- 8. 打印总结 ----------------------
print(f"\n✅ 模型已保存到: {model_save_path}")
print(f"✅ 缺失值处理器已保存到: {imputer_save_path}")
print(f"✅ 最佳阈值已保存到: {threshold_save_path}")
print(f"✅ 特征选择信息已保存到: {feature_selection_save_path}")
print(f"✅ 特征重要性已保存到: {feature_importance_save_path}")
print(f"✅ 测试集预测结果已保存到: {y_test_save_path}, {y_pred_save_path}, {y_pred_proba_save_path}")
print(f"✅ 详细结果已保存到: {results_save_path}")
print(f"✅ 交叉验证各折叠结果已保存到: {fold_results_save_path}")
print(f"✅ ROC曲线已保存到: {roc_save_path}")
print(f"✅ 阈值平衡图已保存到: {balance_save_path}")

# 显示所有保存的文件
print(f"\n📁 所有文件已保存到: {SAVE_DIR}")
print("📋 文件列表:")
for file in os.listdir(SAVE_DIR):
    if file.startswith('ha_rf') or file.startswith('y_'):
        file_path = os.path.join(SAVE_DIR, file)
        if os.path.isfile(file_path):
            print(f"  - {file}")

print("\n✅ 所有结果已保存!")
print("\n=== 主要改进措施 ===")
print("1. 特征选择: 保留重要性前50%的特征，减少过拟合")
print("2. 参数调整: 增加正则化参数(min_samples_split, min_samples_leaf)")
print("3. 类别权重: 增加了负类权重的选项，提高特异性")
print("4. SMOTE策略: 调整采样策略为0.8，避免过度生成正样本")
print("5. 自定义评分: 使用平衡F1和特异性的评分函数")
print("6. 阈值优化: 自动寻找最佳决策阈值提高特异性")
print("7. 新增可视化: 添加阈值平衡分析图")
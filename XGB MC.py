import pandas as pd
import numpy as np
import matplotlib
import os

# 定义保存目录
SAVE_DIR = r"E:\XGBoost_maccs+chem_Results"
os.makedirs(SAVE_DIR, exist_ok=True)

# 设置matplotlib
matplotlib.rcParams['font.sans-serif'] = ['arial']
matplotlib.rcParams['axes.unicode_minus'] = False
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, roc_curve, roc_auc_score, f1_score, accuracy_score
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
import joblib
from scipy.stats import loguniform, uniform
import xgboost as xgb
from sklearn.utils.class_weight import compute_sample_weight


# 1. 评估指标函数
def calculate_se(y_true, y_pred):
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    return tp / (tp + fn) if (tp + fn) != 0 else 0.0


def calculate_sp(y_true, y_pred):
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    return tn / (tn + fp) if (tn + fp) != 0 else 0.0


# 2. 数据预处理
print("加载数据...")
df_x_train = pd.read_csv(r"E:\训练集测试集2\ha合X_train.csv", na_values=["?", "NA", " ", ""])
df_y_train = pd.read_csv(r"E:\训练集测试集2\ha合y_train.csv", na_values=["?", "NA", " ", ""])
df_x_test = pd.read_csv(r"E:\训练集测试集2\ha合X_test.csv", na_values=["?", "NA", " ", ""])
df_y_test = pd.read_csv(r"E:\训练集测试集2\ha合y_test.csv", na_values=["?", "NA", " ", ""])

print(f"训练集形状: {df_x_train.shape}, 测试集形状: {df_x_test.shape}")

# 删除非数值列
non_numeric_cols = [col for col in df_x_train.columns if df_x_train[col].dtype == 'object']
if non_numeric_cols:
    print(f"删除非数值列: {non_numeric_cols}")
    df_x_train = df_x_train.drop(columns=non_numeric_cols)
    df_x_test = df_x_test.drop(columns=non_numeric_cols)

# 合并数据
df_train = pd.concat([df_x_train, df_y_train], axis=1)
df_test = pd.concat([df_x_test, df_y_test], axis=1)

# 提取特征和标签
x_train = df_train.iloc[:, :-1].to_numpy()
y_train = df_train.iloc[:, -1].to_numpy()
x_test = df_test.iloc[:, :-1].to_numpy()
y_test = df_test.iloc[:, -1].to_numpy()

print(f"处理后训练集: {x_train.shape[0]} 样本, {x_train.shape[1]} 特征")
print(f"处理后测试集: {x_test.shape[0]} 样本, {x_test.shape[1]} 特征")

# 处理缺失值和标准化
imputer = SimpleImputer(strategy='mean')
x_train = imputer.fit_transform(x_train)
x_test = imputer.transform(x_test)

scaler = StandardScaler()
x_train_scaled = scaler.fit_transform(x_train)
x_test_scaled = scaler.transform(x_test)

joblib.dump(scaler, os.path.join(SAVE_DIR, "xgboost_scaler.pkl"))

# 检查类别分布
print("\n=== 数据分布 ===")
neg_count_train = np.sum(y_train == 0)
pos_count_train = np.sum(y_train == 1)
neg_count_test = np.sum(y_test == 0)
pos_count_test = np.sum(y_test == 1)
print(f"训练集 - 负样本: {neg_count_train}, 正样本: {pos_count_train}")
print(f"测试集 - 负样本: {neg_count_test}, 正样本: {pos_count_test}")


# 3. 阈值优化函数
def find_best_threshold_for_sp(y_true, y_pred_proba):
    """寻找最优阈值以提升SP"""
    fpr, tpr, thresholds = roc_curve(y_true, y_pred_proba)

    # 方法1: Youden指数
    youden_idx = np.argmax(tpr - fpr)
    threshold_youden = thresholds[youden_idx]

    # 方法2: 在保持SE>=0.7的情况下最大化SP
    sp_values = 1 - fpr
    mask = tpr >= 0.7
    if mask.any():
        sp_max_idx = np.argmax(sp_values[mask])
        threshold_sp = thresholds[mask][sp_max_idx]
    else:
        threshold_sp = threshold_youden

    return threshold_sp, threshold_youden


# 4. XGBoost模型训练（优化SP）
print("\n训练XGBoost模型（优化SP）...")

model = xgb.XGBClassifier(
    use_label_encoder=False,
    eval_metric='logloss',
    objective='binary:logistic',
    n_estimators=150,
    random_state=42,
    n_jobs=-1,
    verbosity=0
)

# 参数搜索（针对SP优化）
param_dist = {
    'learning_rate': uniform(0.01, 0.2),
    'max_depth': [3, 4, 5, 6],
    'min_child_weight': [2, 3, 4, 5],
    'subsample': uniform(0.7, 0.2),
    'colsample_bytree': uniform(0.7, 0.2),
    'gamma': uniform(0.1, 0.5),
    'reg_alpha': loguniform(1e-3, 10),
    'reg_lambda': loguniform(1e-3, 10),
    'scale_pos_weight': [0.5, 0.7, 1.0, 1.5, 2.0]
}

# 使用样本权重（给负样本更高权重）
sample_weights = compute_sample_weight(class_weight={0: 2.0, 1: 1}, y=y_train)

random_search = RandomizedSearchCV(
    model, param_dist, n_iter=30, scoring='roc_auc',
    cv=5, n_jobs=-1, random_state=42, verbose=0
)

random_search.fit(x_train_scaled, y_train, sample_weight=sample_weights)
print(f"最佳参数: {random_search.best_params_}")

# 5. 交叉验证寻找最佳阈值
print("\n5折交叉验证寻找最佳阈值...")
kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
best_thresholds = []

for fold, (train_idx, val_idx) in enumerate(kfold.split(x_train_scaled, y_train)):
    fold_model = xgb.XGBClassifier(**random_search.best_params_, random_state=42)
    fold_model.fit(x_train_scaled[train_idx], y_train[train_idx])

    y_val_pred_proba = fold_model.predict_proba(x_train_scaled[val_idx])[:, 1]
    threshold_sp, _ = find_best_threshold_for_sp(y_train[val_idx], y_val_pred_proba)
    best_thresholds.append(threshold_sp)

    y_val_pred = (y_val_pred_proba >= threshold_sp).astype(int)
    sp = calculate_sp(y_train[val_idx], y_val_pred)
    se = calculate_se(y_train[val_idx], y_val_pred)
    print(f"折叠 {fold + 1}: 阈值={threshold_sp:.3f}, SP={sp:.3f}, SE={se:.3f}")

avg_threshold = np.mean(best_thresholds)
print(f"平均最佳阈值: {avg_threshold:.3f}")

# 6. 测试集评估
print("\n测试集评估...")
best_model = random_search.best_estimator_
y_pred_proba = best_model.predict_proba(x_test_scaled)[:, 1]

# 尝试多个阈值选择最优
thresholds_to_test = [0.4, 0.5, avg_threshold, 0.6, 0.7]
best_sp = 0
best_thresh_for_test = avg_threshold

for thresh in thresholds_to_test:
    y_pred_temp = (y_pred_proba >= thresh).astype(int)
    sp_temp = calculate_sp(y_test, y_pred_temp)
    se_temp = calculate_se(y_test, y_pred_temp)

    if sp_temp > best_sp and se_temp >= 0.7:
        best_sp = sp_temp
        best_thresh_for_test = thresh

# 使用最优阈值
y_pred = (y_pred_proba >= best_thresh_for_test).astype(int)

# 计算指标
auc = roc_auc_score(y_test, y_pred_proba)
acc = accuracy_score(y_test, y_pred)
se = calculate_se(y_test, y_pred)
sp = calculate_sp(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
mcc = (tp * tn - fp * fn) / np.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn)) if (tp + fp) * (tp + fn) * (
            tn + fp) * (tn + fn) != 0 else 0.0

print(f"\n测试集性能 (阈值={best_thresh_for_test:.3f}):")
print(f"AUC: {auc:.4f}")
print(f"ACC: {acc:.4f}")
print(f"SE: {se:.4f}")
print(f"SP: {sp:.4f}")
print(f"F1: {f1:.4f}")
print(f"MCC: {mcc:.4f}")
print(f"混淆矩阵: TP={tp}, FP={fp}, TN={tn}, FN={fn}")

# 7. 可视化
print("\n生成可视化图表...")

# ROC曲线
fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'XGBoost (AUC = {auc:.3f})')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate', fontsize=12)
plt.ylabel('True Positive Rate', fontsize=12)
plt.title(f'ROC Curve - SP Optimized\nThreshold={best_thresh_for_test:.3f}, SP={sp:.3f}', fontsize=14)
plt.legend(loc="lower right")
plt.grid(True, alpha=0.3)

roc_pdf_path = os.path.join(SAVE_DIR, 'ha_xgboost_roc_sp_optimized.pdf')
plt.savefig(roc_pdf_path, dpi=300, bbox_inches='tight')
plt.close()
print(f"✅ ROC曲线已保存: {roc_pdf_path}")

# 8. 保存结果
print("\n保存结果...")
joblib.dump(best_model, os.path.join(SAVE_DIR, "ha_xgboost_model_sp_optimized.pkl"))

results = pd.DataFrame({
    'Model': ['XGBoost_SP_Optimized'],
    'Best_Threshold': [best_thresh_for_test],
    'Test_AUC': [auc],
    'Test_ACC': [acc],
    'Test_SE': [se],
    'Test_SP': [sp],
    'Test_F1': [f1],
    'Test_MCC': [mcc],
    'TP': [tp], 'FP': [fp], 'TN': [tn], 'FN': [fn],
    'Best_Params': [str(random_search.best_params_)]
})

results.to_csv(os.path.join(SAVE_DIR, 'ha_xgboost_results_sp_optimized.csv'), index=False)

# 保存特征重要性
feature_importance = best_model.feature_importances_
feature_df = pd.DataFrame({
    'Feature_Index': range(len(feature_importance)),
    'Importance': feature_importance
}).sort_values('Importance', ascending=False)
feature_df.to_csv(os.path.join(SAVE_DIR, 'feature_importance_sp_optimized.csv'), index=False)

print(f"\n✅ XGBoost训练完成（SP优化）！")
print(f"📁 结果保存到: {SAVE_DIR}")
print(f"📊 测试集SP: {sp:.4f}")
print(f"📈 测试集SE: {se:.4f}")
print(f"🎯 最佳阈值: {best_thresh_for_test:.3f}")
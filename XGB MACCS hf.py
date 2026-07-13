import pandas as pd
import numpy as np
import matplotlib
import os

# 定义保存目录
SAVE_DIR = r"E:\XGBoost_MACCS_Results"
os.makedirs(SAVE_DIR, exist_ok=True)

# 设置matplotlib中文字体
matplotlib.rcParams['font.sans-serif'] = ['arial']
matplotlib.rcParams['axes.unicode_minus'] = False
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, roc_curve, roc_auc_score, f1_score, accuracy_score, matthews_corrcoef
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
import joblib
from scipy.stats import loguniform, uniform, randint
import xgboost as xgb
from sklearn.feature_selection import SelectFromModel, VarianceThreshold
import warnings

warnings.filterwarnings('ignore', category=UserWarning)


# 1. 评估指标函数
def calculate_se(y_true, y_pred):
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    return tp / (tp + fn) if (tp + fn) != 0 else 0.0


def calculate_sp(y_true, y_pred):
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    return tn / (tn + fp) if (tn + fp) != 0 else 0.0


# 2. 数据预处理
print("加载MACCS数据...")

try:
    df_x_train = pd.read_csv(r"E:\训练集测试集4\hfX_train_maccs.csv", na_values=["?", "NA", " ", ""])
    df_y_train = pd.read_csv(r"E:\训练集测试集4\hfy_train_maccs.csv", na_values=["?", "NA", " ", ""])
    df_x_test = pd.read_csv(r"E:\训练集测试集4\hfX_test_maccs.csv", na_values=["?", "NA", " ", ""])
    df_y_test = pd.read_csv(r"E:\训练集测试集4\hfy_test_maccs.csv", na_values=["?", "NA", " ", ""])

    print(f"原始训练集形状: {df_x_train.shape}")
    print(f"原始测试集形状: {df_x_test.shape}")

except Exception as e:
    print(f"数据加载失败: {e}")
    exit()

# 处理非数值列（SMILES）
print("处理SMILES列...")
if 'SMILES' in df_x_train.columns:
    df_x_train = df_x_train.drop(columns=['SMILES'])
    df_x_test = df_x_test.drop(columns=['SMILES'])
    print(f"删除SMILES列后形状: 训练集={df_x_train.shape}, 测试集={df_x_test.shape}")

# 合并数据
df_train = pd.concat([df_x_train, df_y_train], axis=1)
df_test = pd.concat([df_x_test, df_y_test], axis=1)

# 提取特征和标签
x_train = df_train.iloc[:, :-1].to_numpy()
y_train = df_train.iloc[:, -1].to_numpy()
x_test = df_test.iloc[:, :-1].to_numpy()
y_test = df_test.iloc[:, -1].to_numpy()

print(f"\n处理后训练集: {x_train.shape[0]} 个样本, {x_train.shape[1]} 个特征")
print(f"处理后测试集: {x_test.shape[0]} 个样本, {x_test.shape[1]} 个特征")

# 检查数据分布
print("\n=== 数据分布 ===")
print(f"训练集 - Class 0 (负样本): {np.sum(y_train == 0)}")
print(f"训练集 - Class 1 (正样本): {np.sum(y_train == 1)}")
print(f"测试集 - Class 0 (负样本): {np.sum(y_test == 0)}")
print(f"测试集 - Class 1 (正样本): {np.sum(y_test == 1)}")

# MACCS是二进制指纹，不需要标准化，直接使用
x_train_scaled = x_train.astype(float)
x_test_scaled = x_test.astype(float)

# 特征选择：移除低方差特征
print("\n特征选择...")
selector = VarianceThreshold(threshold=0.01)  # 移除方差低于0.01的特征
x_train_selected = selector.fit_transform(x_train_scaled)
x_test_selected = selector.transform(x_test_scaled)
print(f"特征选择后: 训练集={x_train_selected.shape}, 测试集={x_test_selected.shape}")

# 3. 使用贝叶斯优化进行参数调优
print("\n使用贝叶斯优化进行参数调优...")
!pip
install
scikit - optimize - q

from skopt import BayesSearchCV
from skopt.space import Real, Categorical, Integer
from skopt.callbacks import DeadlineStopper, DeltaYStopper

# 计算类别不平衡比例
pos_weight = np.sum(y_train == 0) / np.sum(y_train == 1)

# 定义贝叶斯搜索空间
search_spaces = {
    'learning_rate': Real(0.01, 0.3, prior='log-uniform'),
    'max_depth': Integer(3, 10),
    'min_child_weight': Integer(1, 10),
    'subsample': Real(0.6, 1.0),
    'colsample_bytree': Real(0.6, 1.0),
    'gamma': Real(0, 1),
    'reg_alpha': Real(1e-6, 10, prior='log-uniform'),
    'reg_lambda': Real(1e-6, 10, prior='log-uniform'),
    'scale_pos_weight': Real(0.5, 5),  # 自动调整
    'n_estimators': Integer(100, 500)
}

# 创建基础模型
base_model = xgb.XGBClassifier(
    use_label_encoder=False,
    eval_metric='logloss',
    objective='binary:logistic',
    random_state=42,
    n_jobs=-1
)

# 贝叶斯搜索
bayes_search = BayesSearchCV(
    estimator=base_model,
    search_spaces=search_spaces,
    n_iter=50,  # 贝叶斯优化50次
    cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=42),
    scoring='roc_auc',
    n_jobs=-1,
    random_state=42,
    verbose=2
)

print("开始贝叶斯优化搜索...")
bayes_search.fit(x_train_selected, y_train)

print(f"\n贝叶斯优化最佳参数: {bayes_search.best_params_}")
print(f"贝叶斯优化最佳分数: {bayes_search.best_score_:.4f}")

# 4. 使用最佳参数进行更详细的验证
best_params = bayes_search.best_params_

# 5折交叉验证
print("\n5折交叉验证...")
kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

cv_metrics = {
    'auc': [], 'acc': [], 'se': [], 'sp': [], 'f1': [], 'mcc': []
}

for fold, (train_idx, val_idx) in enumerate(kfold.split(x_train_selected, y_train)):
    print(f"训练折叠 {fold + 1}/5...")

    # 创建模型
    fold_model = xgb.XGBClassifier(
        **best_params,
        use_label_encoder=False,
        eval_metric='logloss',
        random_state=42 + fold,
        n_jobs=-1
    )

    # 训练
    fold_model.fit(
        x_train_selected[train_idx],
        y_train[train_idx],
        eval_set=[(x_train_selected[val_idx], y_train[val_idx])],
        verbose=False
    )

    # 预测
    y_val_pred = fold_model.predict(x_train_selected[val_idx])
    y_val_pred_proba = fold_model.predict_proba(x_train_selected[val_idx])[:, 1]

    # 计算指标
    cv_metrics['auc'].append(roc_auc_score(y_train[val_idx], y_val_pred_proba))
    cv_metrics['acc'].append(accuracy_score(y_train[val_idx], y_val_pred))
    cv_metrics['se'].append(calculate_se(y_train[val_idx], y_val_pred))
    cv_metrics['sp'].append(calculate_sp(y_train[val_idx], y_val_pred))
    cv_metrics['f1'].append(f1_score(y_train[val_idx], y_val_pred))
    cv_metrics['mcc'].append(matthews_corrcoef(y_train[val_idx], y_val_pred))

print("\n交叉验证平均性能:")
for metric, values in cv_metrics.items():
    print(f"{metric}: {np.mean(values):.4f} ± {np.std(values):.4f}")

# 5. 使用早停训练最终模型
print("\n使用早停训练最终模型...")
best_model = xgb.XGBClassifier(
    **best_params,
    use_label_encoder=False,
    eval_metric='logloss',
    random_state=42,
    n_jobs=-1
)

# 使用早停
best_model.fit(
    x_train_selected, y_train,
    eval_set=[(x_test_selected, y_test)],
    early_stopping_rounds=50,
    verbose=100
)

# 6. 测试集评估
print("\n测试集评估...")
y_pred = best_model.predict(x_test_selected)
y_pred_proba = best_model.predict_proba(x_test_selected)[:, 1]

# 计算指标
metrics = {
    'auc': roc_auc_score(y_test, y_pred_proba),
    'acc': accuracy_score(y_test, y_pred),
    'se': calculate_se(y_test, y_pred),
    'sp': calculate_sp(y_test, y_pred),
    'f1': f1_score(y_test, y_pred),
    'mcc': matthews_corrcoef(y_test, y_pred)
}

tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()

print("\n测试集性能:")
for metric, value in metrics.items():
    print(f"{metric.upper()}: {value:.4f}")
print(f"混淆矩阵: TP={tp}, FP={fp}, TN={tn}, FN={fn}")

# 7. 尝试集成方法
print("\n尝试集成方法...")
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import VotingClassifier

# 创建集成模型
estimators = [
    ('xgb', xgb.XGBClassifier(**best_params, use_label_encoder=False, eval_metric='logloss', random_state=42)),
    ('rf', RandomForestClassifier(n_estimators=200, max_depth=10, random_state=42, n_jobs=-1)),
    ('lr', LogisticRegression(C=1.0, random_state=42, max_iter=1000))
]

voting_clf = VotingClassifier(
    estimators=estimators,
    voting='soft',
    n_jobs=-1
)

# 训练集成模型
voting_clf.fit(x_train_selected, y_train)

# 评估集成模型
y_pred_ensemble = voting_clf.predict(x_test_selected)
y_pred_proba_ensemble = voting_clf.predict_proba(x_test_selected)[:, 1]

metrics_ensemble = {
    'auc': roc_auc_score(y_test, y_pred_proba_ensemble),
    'acc': accuracy_score(y_test, y_pred_ensemble),
    'se': calculate_se(y_test, y_pred_ensemble),
    'sp': calculate_sp(y_test, y_pred_ensemble),
    'f1': f1_score(y_test, y_pred_ensemble),
    'mcc': matthews_corrcoef(y_test, y_pred_ensemble)
}

print("\n集成模型测试集性能:")
for metric, value in metrics_ensemble.items():
    print(f"{metric.upper()}: {value:.4f}")

# 选择最佳模型
if metrics_ensemble['auc'] > metrics['auc']:
    print("\n✅ 集成模型表现更好，使用集成模型")
    final_model = voting_clf
    final_y_pred = y_pred_ensemble
    final_y_pred_proba = y_pred_proba_ensemble
    final_metrics = metrics_ensemble
else:
    print("\n✅ XGBoost单模型表现更好，使用XGBoost")
    final_model = best_model
    final_y_pred = y_pred
    final_y_pred_proba = y_pred_proba
    final_metrics = metrics

# 8. ROC曲线
print("\n生成ROC曲线...")
fpr, tpr, _ = roc_curve(y_test, final_y_pred_proba)
plt.figure(figsize=(10, 8))
plt.plot(fpr, tpr, color='darkorange', lw=2,
         label=f'ROC curve (AUC = {final_metrics["auc"]:.3f})')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate', fontsize=14)
plt.ylabel('True Positive Rate', fontsize=14)
plt.title(f'ROC Curve - {type(final_model).__name__} on MACCS Fingerprints', fontsize=16)
plt.legend(loc="lower right", fontsize=12)
plt.grid(True, alpha=0.3)
plt.tight_layout()

roc_pdf_path = os.path.join(SAVE_DIR, 'optimized_roc_curve.pdf')
plt.savefig(roc_pdf_path, dpi=300, bbox_inches='tight')
plt.close()
print(f"✅ ROC曲线已保存为PDF: {roc_pdf_path}")

# 9. 特征重要性分析（如果是XGBoost）
if hasattr(final_model, 'feature_importances_'):
    print("\n生成特征重要性图...")
    feature_importance = final_model.feature_importances_

    # 获取特征名称（如果可用）
    feature_names = df_x_train.columns.tolist() if hasattr(df_x_train, 'columns') else [f'Feature_{i}' for i in
                                                                                        range(len(feature_importance))]

    # 选择最重要的20个特征
    top_n = min(20, len(feature_importance))
    sorted_idx = np.argsort(feature_importance)[-top_n:]

    plt.figure(figsize=(12, 8))
    plt.barh(range(top_n), feature_importance[sorted_idx])
    plt.yticks(range(top_n), [feature_names[i] for i in sorted_idx])
    plt.xlabel('Feature Importance', fontsize=14)
    plt.ylabel('Feature', fontsize=14)
    plt.title(f'Top {top_n} Feature Importance', fontsize=16)
    plt.tight_layout()

    importance_pdf_path = os.path.join(SAVE_DIR, 'optimized_feature_importance.pdf')
    plt.savefig(importance_pdf_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✅ 特征重要性图已保存为PDF: {importance_pdf_path}")

# 10. 学习曲线
print("\n生成学习曲线...")
from sklearn.model_selection import learning_curve

train_sizes, train_scores, test_scores = learning_curve(
    final_model, x_train_selected, y_train,
    cv=5, scoring='roc_auc', n_jobs=-1,
    train_sizes=np.linspace(0.1, 1.0, 10)
)

train_scores_mean = np.mean(train_scores, axis=1)
train_scores_std = np.std(train_scores, axis=1)
test_scores_mean = np.mean(test_scores, axis=1)
test_scores_std = np.std(test_scores, axis=1)

plt.figure(figsize=(10, 6))
plt.fill_between(train_sizes, train_scores_mean - train_scores_std,
                 train_scores_mean + train_scores_std, alpha=0.1, color="r")
plt.fill_between(train_sizes, test_scores_mean - test_scores_std,
                 test_scores_mean + test_scores_std, alpha=0.1, color="g")
plt.plot(train_sizes, train_scores_mean, 'o-', color="r", label="Training score")
plt.plot(train_sizes, test_scores_mean, 'o-', color="g", label="Cross-validation score")
plt.xlabel("Training examples", fontsize=14)
plt.ylabel("AUC Score", fontsize=14)
plt.title("Learning Curve", fontsize=16)
plt.legend(loc="best", fontsize=12)
plt.grid(True, alpha=0.3)
plt.tight_layout()

learning_curve_path = os.path.join(SAVE_DIR, 'learning_curve.pdf')
plt.savefig(learning_curve_path, dpi=300, bbox_inches='tight')
plt.close()
print(f"✅ 学习曲线已保存为PDF: {learning_curve_path}")

# 11. 保存结果
print("\n保存结果...")
joblib.dump(final_model, os.path.join(SAVE_DIR, "optimized_model.pkl"))
joblib.dump(selector, os.path.join(SAVE_DIR, "feature_selector.pkl"))

# 保存预测结果
np.save(os.path.join(SAVE_DIR, "y_test.npy"), y_test)
np.save(os.path.join(SAVE_DIR, "y_pred.npy"), final_y_pred)
np.save(os.path.join(SAVE_DIR, "y_pred_proba.npy"), final_y_pred_proba)

# 保存详细结果
results_summary = pd.DataFrame({
    'Model': [type(final_model).__name__],
    'Best_Params': [str(best_params)],
    'Test_AUC': [final_metrics['auc']],
    'Test_ACC': [final_metrics['acc']],
    'Test_SE': [final_metrics['se']],
    'Test_SP': [final_metrics['sp']],
    'Test_F1': [final_metrics['f1']],
    'Test_MCC': [final_metrics['mcc']],
    'TP': [tp],
    'FP': [fp],
    'TN': [tn],
    'FN': [fn],
    'CV_AUC_mean': [np.mean(cv_metrics['auc'])],
    'CV_AUC_std': [np.std(cv_metrics['auc'])],
    'Feature_Count': [x_train_selected.shape[1]],
    'Data_Shape': [f"训练: {x_train.shape}, 测试: {x_test.shape}"]
})

results_summary.to_csv(os.path.join(SAVE_DIR, 'optimized_results_summary.csv'), index=False)

# 保存所有交叉验证结果
cv_results_df = pd.DataFrame(cv_metrics)
cv_results_df.to_csv(os.path.join(SAVE_DIR, 'cv_results.csv'), index=False)

print(f"\n✅ 模型训练完成！")
print(f"📁 结果保存到: {SAVE_DIR}")
print(f"📊 测试集AUC: {final_metrics['auc']:.4f}")
print(f"📈 ROC曲线PDF: {roc_pdf_path}")
print(f"📊 学习曲线PDF: {learning_curve_path}")
if hasattr(final_model, 'feature_importances_'):
    print(f"🔍 特征重要性PDF: {importance_pdf_path}")
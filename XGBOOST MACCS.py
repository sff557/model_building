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
print("加载MACCS数据...")

# 读取数据
try:
    df_x_train = pd.read_csv(r"E:\训练集测试集\haX_train_maccs.csv", na_values=["?", "NA", " ", ""])
    df_y_train = pd.read_csv(r"E:\训练集测试集\hay_train_maccs.csv", na_values=["?", "NA", " ", ""])
    df_x_test = pd.read_csv(r"E:\训练集测试集\haX_test_maccs.csv", na_values=["?", "NA", " ", ""])
    df_y_test = pd.read_csv(r"E:\训练集测试集\hay_test_maccs.csv", na_values=["?", "NA", " ", ""])

    print(f"原始训练集形状: {df_x_train.shape}")
    print(f"原始测试集形状: {df_x_test.shape}")

except Exception as e:
    print(f"数据加载失败: {e}")
    print("请检查文件路径和格式是否正确")
    exit()

# 检查数据类型
print("\n检查数据类型...")
print("训练集数据类型分布:")
print(df_x_train.dtypes.value_counts())
print("\n测试集数据类型分布:")
print(df_x_test.dtypes.value_counts())

# 查看非数值列
non_numeric_cols_train = []
non_numeric_cols_test = []

for col in df_x_train.columns:
    if df_x_train[col].dtype == 'object':
        non_numeric_cols_train.append(col)
        print(f"训练集非数值列 '{col}' 示例值: {df_x_train[col].iloc[0]}")

for col in df_x_test.columns:
    if df_x_test[col].dtype == 'object':
        non_numeric_cols_test.append(col)
        print(f"测试集非数值列 '{col}' 示例值: {df_x_test[col].iloc[0]}")

# 处理非数值列 - 根据您的数据情况选择适当的方法
print(f"\n发现 {len(non_numeric_cols_train)} 个非数值列在训练集中")
print(f"发现 {len(non_numeric_cols_test)} 个非数值列在测试集中")

# 方法1: 如果非数值列是SMILES字符串（分子结构），可能需要删除或编码
# 假设第一列是SMILES字符串，我们将其删除
if len(non_numeric_cols_train) > 0:
    print(f"删除非数值列: {non_numeric_cols_train}")
    df_x_train = df_x_train.drop(columns=non_numeric_cols_train)
    df_x_test = df_x_test.drop(columns=non_numeric_cols_test)

    print(f"删除后训练集形状: {df_x_train.shape}")
    print(f"删除后测试集形状: {df_x_test.shape}")

# 方法2: 如果非数值列是二进制特征（0/1但是字符串类型），可以转换为数值
# 这个示例中我们假设已删除非数值列

# 方法3: 将所有列尝试转换为数值类型
# 这可以处理包含数字的字符串列
for col in df_x_train.columns:
    try:
        # 尝试转换为数值，转换失败的设为NaN
        df_x_train[col] = pd.to_numeric(df_x_train[col], errors='coerce')
    except Exception as e:
        print(f"列 '{col}' 转换为数值失败: {e}")

for col in df_x_test.columns:
    try:
        df_x_test[col] = pd.to_numeric(df_x_test[col], errors='coerce')
    except Exception as e:
        print(f"列 '{col}' 转换为数值失败: {e}")

print("\n转换后数据类型分布:")
print(df_x_train.dtypes.value_counts())

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

# 检查缺失值比例
nan_percentage_train = np.isnan(x_train).sum() / (x_train.shape[0] * x_train.shape[1]) * 100
nan_percentage_test = np.isnan(x_test).sum() / (x_test.shape[0] * x_test.shape[1]) * 100
print(f"\n训练集缺失值比例: {nan_percentage_train:.2f}%")
print(f"测试集缺失值比例: {nan_percentage_test:.2f}%")

# 处理缺失值和标准化
print("\n处理缺失值...")
imputer = SimpleImputer(strategy='mean')
x_train = imputer.fit_transform(x_train)
x_test = imputer.transform(x_test)

scaler = StandardScaler()
x_train_scaled = scaler.fit_transform(x_train)
x_test_scaled = scaler.transform(x_test)

# 保存标准化器
joblib.dump(scaler, os.path.join(SAVE_DIR, "xgboost_scaler.pkl"))
print("✅ 标准化器已保存")

# 检查类别分布
print("\n=== 数据分布 ===")
print(f"训练集 - Class 0 (负样本): {np.sum(y_train == 0)}")
print(f"训练集 - Class 1 (正样本): {np.sum(y_train == 1)}")
print(f"测试集 - Class 0 (负样本): {np.sum(y_test == 0)}")
print(f"测试集 - Class 1 (正样本): {np.sum(y_test == 1)}")

# 3. XGBoost模型训练
print("\n训练XGBoost模型...")
model = xgb.XGBClassifier(
    objective='binary:logistic',
    n_estimators=100,
    random_state=42,
    n_jobs=-1,
    use_label_encoder=False,
    eval_metric='logloss'
)

# 参数搜索
param_dist = {
    'learning_rate': uniform(0.01, 0.2),
    'max_depth': [3, 4, 5, 6, 7, 8, 9],
    'min_child_weight': [1, 2, 3, 4, 5],
    'subsample': uniform(0.6, 0.3),
    'colsample_bytree': uniform(0.6, 0.3),
    'gamma': uniform(0, 2),
    'reg_alpha': loguniform(1e-5, 10),
    'reg_lambda': loguniform(1e-5, 10),
}

random_search = RandomizedSearchCV(
    model, param_dist, n_iter=30, scoring='roc_auc',
    cv=3, n_jobs=-1, random_state=42, verbose=1
)

sample_weights = compute_sample_weight(class_weight={0: 1.8, 1: 1}, y=y_train)
random_search.fit(x_train_scaled, y_train, sample_weight=sample_weights)

print(f"\n最佳参数: {random_search.best_params_}")

# 4. 交叉验证
print("\n5折交叉验证...")
kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

fold_metrics = {'auc': [], 'acc': [], 'se': [], 'sp': [], 'f1': []}

for fold, (train_idx, val_idx) in enumerate(kfold.split(x_train_scaled, y_train)):
    print(f"训练折叠 {fold + 1}/5...")
    fold_model = xgb.XGBClassifier(**random_search.best_params_, random_state=42)
    fold_model.fit(x_train_scaled[train_idx], y_train[train_idx])

    y_val_pred = fold_model.predict(x_train_scaled[val_idx])
    y_val_pred_proba = fold_model.predict_proba(x_train_scaled[val_idx])[:, 1]

    fold_metrics['auc'].append(roc_auc_score(y_train[val_idx], y_val_pred_proba))
    fold_metrics['acc'].append(accuracy_score(y_train[val_idx], y_val_pred))
    fold_metrics['se'].append(calculate_se(y_train[val_idx], y_val_pred))
    fold_metrics['sp'].append(calculate_sp(y_train[val_idx], y_val_pred))
    fold_metrics['f1'].append(f1_score(y_train[val_idx], y_val_pred))

print("\n交叉验证平均性能:")
for metric, values in fold_metrics.items():
    print(f"{metric}: {np.mean(values):.4f} ± {np.std(values):.4f}")

# 5. 测试集评估
print("\n测试集评估...")
best_model = random_search.best_estimator_
y_pred = best_model.predict(x_test_scaled)
y_pred_proba = best_model.predict_proba(x_test_scaled)[:, 1]

# 计算指标
auc = roc_auc_score(y_test, y_pred_proba)
acc = accuracy_score(y_test, y_pred)
se = calculate_se(y_test, y_pred)
sp = calculate_sp(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
mcc = (tp * tn - fp * fn) / np.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn)) if (tp + fp) * (tp + fn) * (
            tn + fp) * (tn + fn) != 0 else 0.0

print(f"\n测试集性能:")
print(f"AUC: {auc:.4f}")
print(f"ACC: {acc:.4f}")
print(f"SE: {se:.4f}")
print(f"SP: {sp:.4f}")
print(f"F1: {f1:.4f}")
print(f"MCC: {mcc:.4f}")
print(f"混淆矩阵: TP={tp}, FP={fp}, TN={tn}, FN={fn}")

# 6. ROC曲线（PDF格式）
print("\n生成ROC曲线...")
fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'XGBoost (AUC = {auc:.2f})')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate', fontsize=12)
plt.ylabel('True Positive Rate', fontsize=12)
plt.title('ROC Curve - XGBoost on MACCS Fingerprints', fontsize=14)
plt.legend(loc="lower right")
plt.grid(True, alpha=0.3)

roc_pdf_path = os.path.join(SAVE_DIR, 'xgboost_roc_curve.pdf')
plt.savefig(roc_pdf_path, dpi=300, bbox_inches='tight')
plt.close()
print(f"✅ ROC曲线已保存为PDF: {roc_pdf_path}")

# 7. 特征重要性图（PDF格式）
print("生成特征重要性图...")
feature_importance = best_model.feature_importances_
top_n = min(20, len(feature_importance))
sorted_idx = np.argsort(feature_importance)[-top_n:]

plt.figure(figsize=(10, 6))
plt.barh(range(top_n), feature_importance[sorted_idx])
plt.yticks(range(top_n), [f'Feature {i}' for i in sorted_idx])
plt.xlabel('Feature Importance', fontsize=12)
plt.ylabel('Feature Index', fontsize=12)
plt.title(f'Top {top_n} Feature Importance - XGBoost on MACCS', fontsize=14)
plt.tight_layout()

importance_pdf_path = os.path.join(SAVE_DIR, 'xgboost_feature_importance.pdf')
plt.savefig(importance_pdf_path, dpi=300, bbox_inches='tight')
plt.close()
print(f"✅ 特征重要性图已保存为PDF: {importance_pdf_path}")

# 8. 保存模型和结果
print("\n保存结果...")
joblib.dump(best_model, os.path.join(SAVE_DIR, "xgboost_model.pkl"))

# 保存预测结果
np.save(os.path.join(SAVE_DIR, "y_test.npy"), y_test)
np.save(os.path.join(SAVE_DIR, "y_pred_proba.npy"), y_pred_proba)

# 保存详细结果到CSV
results = pd.DataFrame({
    'Model': ['XGBoost_MACCS'],
    'Best_Params': [str(random_search.best_params_)],
    'Test_AUC': [auc],
    'Test_ACC': [acc],
    'Test_SE': [se],
    'Test_SP': [sp],
    'Test_F1': [f1],
    'Test_MCC': [mcc],
    'TP': [tp],
    'FP': [fp],
    'TN': [tn],
    'FN': [fn],
    'CV_AUC_mean': [np.mean(fold_metrics['auc'])],
    'CV_AUC_std': [np.std(fold_metrics['auc'])],
    'CV_ACC_mean': [np.mean(fold_metrics['acc'])],
    'CV_ACC_std': [np.std(fold_metrics['acc'])],
    'Data_Shape': [f"训练: {x_train.shape}, 测试: {x_test.shape}"],
    'Non_Numeric_Columns_Removed': [f"训练: {len(non_numeric_cols_train)}, 测试: {len(non_numeric_cols_test)}"]
})

results.to_csv(os.path.join(SAVE_DIR, 'xgboost_results.csv'), index=False)

# 保存特征重要性数据
feature_df = pd.DataFrame({
    'Feature_Index': range(len(feature_importance)),
    'Importance': feature_importance
}).sort_values('Importance', ascending=False)
feature_df.to_csv(os.path.join(SAVE_DIR, 'feature_importance.csv'), index=False)

print(f"\n✅ XGBoost训练完成！")
print(f"📁 结果保存到: {SAVE_DIR}")
print(f"📊 测试集AUC: {auc:.4f}")
print(f"📈 ROC曲线PDF: {roc_pdf_path}")
print(f"🔍 特征重要性PDF: {importance_pdf_path}")
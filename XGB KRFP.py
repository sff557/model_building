import pandas as pd
import numpy as np
import matplotlib
import os
import warnings

warnings.filterwarnings('ignore')

# 定义保存目录
SAVE_DIR = r"E:\XGBoost_krfp_Results 123"
os.makedirs(SAVE_DIR, exist_ok=True)

matplotlib.rcParams['font.sans-serif'] = ['arial']
matplotlib.rcParams['axes.unicode_minus'] = False
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, roc_curve, roc_auc_score, f1_score, accuracy_score, matthews_corrcoef
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold, train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
import joblib
from scipy.stats import loguniform, uniform, randint
import xgboost as xgb
from sklearn.feature_selection import VarianceThreshold, SelectFromModel, SelectKBest, f_classif
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer


# 评估指标函数
def evaluate_model(y_true, y_pred, y_pred_proba):
    """计算所有评估指标"""
    auc = roc_auc_score(y_true, y_pred_proba)
    acc = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    mcc = matthews_corrcoef(y_true, y_pred)

    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    se = tp / (tp + fn) if (tp + fn) != 0 else 0.0
    sp = tn / (tn + fp) if (tn + fp) != 0 else 0.0

    return {
        'auc': auc, 'acc': acc, 'f1': f1, 'mcc': mcc,
        'se': se, 'sp': sp, 'tp': tp, 'fp': fp, 'tn': tn, 'fn': fn
    }


# 特征选择函数
def feature_selection_pipeline(x_train, y_train, x_test, method='rf', n_features=150):
    """特征选择管道"""
    if method == 'rf':
        rf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
        rf.fit(x_train, y_train)
        selector = SelectFromModel(rf, threshold=-np.inf, max_features=n_features, prefit=True)
    elif method == 'anova':
        selector = SelectKBest(f_classif, k=n_features)
        selector.fit(x_train, y_train)
    elif method == 'variance':
        selector = VarianceThreshold(threshold=0.01)
        selector.fit(x_train)

    x_train_selected = selector.transform(x_train)
    x_test_selected = selector.transform(x_test)

    return x_train_selected, x_test_selected, selector


# 自定义交叉验证函数
def custom_cross_val_score(model, X, y, cv=5):
    """自定义交叉验证"""
    cv_scores = []
    skf = StratifiedKFold(n_splits=cv, shuffle=True, random_state=42)

    model_params = model.get_params().copy()
    params_to_remove = ['n_estimators', 'random_state', 'n_jobs', 'verbosity']
    for param in params_to_remove:
        if param in model_params:
            del model_params[param]

    n_estimators = model.get_params().get('n_estimators', 500)

    for train_idx, val_idx in skf.split(X, y):
        X_train_fold, X_val_fold = X[train_idx], X[val_idx]
        y_train_fold, y_val_fold = y[train_idx], y[val_idx]

        model_clone = xgb.XGBClassifier(
            **model_params,
            n_estimators=n_estimators,
            random_state=42,
            n_jobs=-1,
            verbosity=0
        )
        model_clone.fit(X_train_fold, y_train_fold)

        y_pred_proba = model_clone.predict_proba(X_val_fold)[:, 1]
        auc = roc_auc_score(y_val_fold, y_pred_proba)
        cv_scores.append(auc)

    return np.array(cv_scores)


def main():
    print("=== 加载和处理数据 ===")

    # 读取数据
    df_x_train = pd.read_csv(r"E:\训练集测试集5\haX_train_krfp.csv", na_values=["?", "NA", " ", ""])
    df_y_train = pd.read_csv(r"E:\训练集测试集5\hay_train_krfp.csv", na_values=["?", "NA", " ", ""])
    df_x_test = pd.read_csv(r"E:\训练集测试集5\haX_test_krfp.csv", na_values=["?", "NA", " ", ""])
    df_y_test = pd.read_csv(r"E:\训练集测试集5\hay_test_krfp.csv", na_values=["?", "NA", " ", ""])

    print(f"原始形状: 训练集={df_x_train.shape}, 测试集={df_x_test.shape}")

    df_x_train = df_x_train.select_dtypes(include=[np.number])
    df_x_test = df_x_test.select_dtypes(include=[np.number])

    x_train = df_x_train.to_numpy()
    y_train = df_y_train.iloc[:, -1].to_numpy()
    x_test = df_x_test.to_numpy()
    y_test = df_y_test.iloc[:, -1].to_numpy()

    print(f"\n处理后: 训练集={x_train.shape}, 测试集={x_test.shape}")
    print(f"类别分布 - 训练集: 0={np.sum(y_train == 0)}, 1={np.sum(y_train == 1)}")
    print(f"类别分布 - 测试集: 0={np.sum(y_test == 0)}, 1={np.sum(y_test == 1)}")

    # === 数据预处理 ===
    print("\n=== 数据预处理 ===")
    print("处理缺失值...")
    imputer = SimpleImputer(strategy='median')
    x_train = imputer.fit_transform(x_train)
    x_test = imputer.transform(x_test)

    print("标准化数据...")
    scaler = StandardScaler()
    x_train_scaled = scaler.fit_transform(x_train)
    x_test_scaled = scaler.transform(x_test)

    print("处理异常值...")

    def winsorize_data(data, limits=(0.01, 0.99)):
        data_winsorized = data.copy()
        for i in range(data.shape[1]):
            lower = np.percentile(data[:, i], limits[0] * 100)
            upper = np.percentile(data[:, i], limits[1] * 100)
            data_winsorized[:, i] = np.clip(data[:, i], lower, upper)
        return data_winsorized

    x_train_scaled = winsorize_data(x_train_scaled)
    x_test_scaled = winsorize_data(x_test_scaled)

    # === 特征选择 ===
    print("\n=== 特征选择 ===")
    print("使用随机森林进行特征选择...")

    # 增加特征数量以保留更多信息
    n_features = min(500, x_train_scaled.shape[1])
    x_train_selected, x_test_selected, selector = feature_selection_pipeline(
        x_train_scaled, y_train, x_test_scaled,
        method='rf', n_features=n_features
    )
    print(f"特征选择后维度: {x_train_selected.shape[1]}")

    # === 模型训练 ===
    print("\n=== 模型训练 ===")

    x_train_final, x_val, y_train_final, y_val = train_test_split(
        x_train_selected, y_train, test_size=0.2,
        stratify=y_train, random_state=42
    )

    from sklearn.utils.class_weight import compute_class_weight
    classes = np.unique(y_train_final)
    class_weights = compute_class_weight('balanced', classes=classes, y=y_train_final)
    class_weight_dict = dict(zip(classes, class_weights))
    scale_pos_weight = class_weight_dict[1] / class_weight_dict[
        0] if 0 in class_weight_dict and 1 in class_weight_dict else 1

    print(f"scale_pos_weight: {scale_pos_weight:.3f}")

    # === 改进：更平衡的超参数搜索 ===
    print("\n=== 超参数优化（平衡正则化） ===")

    # 放宽正则化限制，增加模型复杂度
    param_dist = {
        'learning_rate': loguniform(0.02, 0.3),  # 提高学习率上限
        'max_depth': randint(3, 8),  # 适当增加深度
        'min_child_weight': randint(1, 6),  # 降低min_child_weight
        'subsample': uniform(0.7, 0.3),
        'colsample_bytree': uniform(0.6, 0.4),
        'colsample_bylevel': uniform(0.6, 0.4),
        'gamma': uniform(0, 3),  # 降低gamma
        'reg_alpha': uniform(0, 3),  # 降低正则化
        'reg_lambda': uniform(0.5, 5),  # 降低正则化
        'scale_pos_weight': [1, scale_pos_weight, 1.5, 2, 2.5],
    }

    base_model = xgb.XGBClassifier(
        eval_metric=['logloss', 'auc'],
        n_estimators=1000,
        random_state=42,
        n_jobs=-1,
        verbosity=0
    )

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    random_search = RandomizedSearchCV(
        base_model, param_dist, n_iter=100,  # 增加搜索次数
        scoring='roc_auc', cv=cv, n_jobs=-1,
        random_state=42, verbose=1
    )

    print("执行超参数搜索...")
    random_search.fit(x_train_final, y_train_final)

    print(f"\n最佳参数: {random_search.best_params_}")
    print(f"最佳交叉验证分数: {random_search.best_score_:.4f}")

    # === 使用早停训练最终模型 ===
    print("\n使用早停训练最终模型...")

    best_params = random_search.best_params_.copy()

    final_model = xgb.XGBClassifier(
        **best_params,
        n_estimators=2000,
        early_stopping_rounds=50,
        eval_metric=['logloss', 'auc'],
        random_state=42,
        n_jobs=-1,
        verbosity=0
    )

    final_model.fit(
        x_train_final, y_train_final,
        eval_set=[(x_val, y_val)],
        verbose=False
    )

    best_iteration = final_model.best_iteration if final_model.best_iteration is not None else 800
    print(f"早停在 {best_iteration} 轮停止")
    print(f"最佳验证分数: {final_model.best_score:.4f}")

    # 训练用于交叉验证的模型
    print("\n训练用于交叉验证的模型...")
    final_model_cv = xgb.XGBClassifier(
        **best_params,
        n_estimators=best_iteration,
        random_state=42,
        n_jobs=-1,
        verbosity=0
    )
    final_model_cv.fit(x_train_selected, y_train)
    print("交叉验证模型训练完成")

    # === 模型评估 ===
    print("\n=== 模型评估 ===")

    y_val_pred = final_model.predict(x_val)
    y_val_pred_proba = final_model.predict_proba(x_val)[:, 1]
    val_metrics = evaluate_model(y_val, y_val_pred, y_val_pred_proba)

    print("验证集性能:")
    for k, v in val_metrics.items():
        if k not in ['tp', 'fp', 'tn', 'fn']:
            print(f"{k.upper()}: {v:.4f}")

    y_test_pred = final_model.predict(x_test_selected)
    y_test_pred_proba = final_model.predict_proba(x_test_selected)[:, 1]
    test_metrics = evaluate_model(y_test, y_test_pred, y_test_pred_proba)

    print("\n测试集性能:")
    for k, v in test_metrics.items():
        if k not in ['tp', 'fp', 'tn', 'fn']:
            print(f"{k.upper()}: {v:.4f}")

    print(
        f"混淆矩阵: TP={test_metrics['tp']}, FP={test_metrics['fp']}, TN={test_metrics['tn']}, FN={test_metrics['fn']}")

    # === 交叉验证评估 ===
    print("\n=== 交叉验证评估 ===")

    cv_scores = custom_cross_val_score(
        final_model_cv, x_train_selected, y_train, cv=5
    )
    print(f"交叉验证AUC得分: {cv_scores.mean():.4f} (±{cv_scores.std():.4f})")
    print(f"各折AUC: {cv_scores}")

    # === 特征重要性分析 ===
    print("\n=== 特征重要性分析 ===")
    feature_importances = final_model.feature_importances_
    top_n = min(20, len(feature_importances))
    top_indices = np.argsort(feature_importances)[-top_n:][::-1]

    print(f"Top {top_n} 重要特征:")
    for i, idx in enumerate(top_indices):
        print(f"{i + 1}. 特征 {idx}: {feature_importances[idx]:.4f}")

    # === 保存结果 ===
    print("\n=== 保存结果 ===")

    fpr, tpr, _ = roc_curve(y_test, y_test_pred_proba)
    plt.figure(figsize=(10, 8))
    plt.plot(fpr, tpr, color='darkorange', lw=3, label=f'XGBoost (AUC = {test_metrics["auc"]:.3f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate', fontsize=14)
    plt.ylabel('True Positive Rate', fontsize=14)
    plt.title('ROC Curve - Optimized XGBoost', fontsize=16)
    plt.legend(loc="lower right", fontsize=12)
    plt.grid(True, alpha=0.3)

    plt.text(0.6, 0.2, f'AUC = {test_metrics["auc"]:.3f}\n'
                       f'ACC = {test_metrics["acc"]:.3f}\n'
                       f'F1 = {test_metrics["f1"]:.3f}\n'
                       f'MCC = {test_metrics["mcc"]:.3f}',
             transform=plt.gca().transAxes, fontsize=12,
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

    roc_path = os.path.join(SAVE_DIR, 'optimized_xgboost_roc.pdf')
    plt.savefig(roc_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✅ ROC曲线保存到: {roc_path}")

    # 保存模型
    model_path = os.path.join(SAVE_DIR, 'optimized_xgboost_model.pkl')
    joblib.dump({
        'model': final_model,
        'model_cv': final_model_cv,
        'imputer': imputer,
        'scaler': scaler,
        'feature_selector': selector,
        'best_params': best_params,
        'best_iteration': best_iteration
    }, model_path)
    print(f"✅ 模型保存到: {model_path}")

    # 保存结果
    results_df = pd.DataFrame({
        'Metric': list(test_metrics.keys()),
        'Value': list(test_metrics.values()),
        'Description': [
            'Area Under ROC Curve', 'Accuracy', 'F1 Score',
            "Matthews Correlation Coefficient", 'Sensitivity/Recall',
            'Specificity', 'True Positives', 'False Positives',
            'True Negatives', 'False Negatives'
        ]
    })
    results_path = os.path.join(SAVE_DIR, 'optimized_results.csv')
    results_df.to_csv(results_path, index=False)
    print(f"✅ 结果保存到: {results_path}")

    importance_df = pd.DataFrame({
        'Feature_Index': range(len(feature_importances)),
        'Importance': feature_importances
    }).sort_values('Importance', ascending=False)
    importance_path = os.path.join(SAVE_DIR, 'feature_importance.csv')
    importance_df.to_csv(importance_path, index=False)
    print(f"✅ 特征重要性保存到: {importance_path}")

    print(f"\n🎉 优化后的模型训练完成！")
    print(f"📊 测试集AUC: {test_metrics['auc']:.4f}")
    print(f"📊 测试集ACC: {test_metrics['acc']:.4f}")
    print(f"📊 测试集MCC: {test_metrics['mcc']:.4f}")
    print(f"📊 测试集F1: {test_metrics['f1']:.4f}")
    print(f"📊 交叉验证AUC: {cv_scores.mean():.4f} (±{cv_scores.std():.4f})")


if __name__ == "__main__":
    main()
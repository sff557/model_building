import pandas as pd
import numpy as np
import matplotlib
import os
from imblearn.over_sampling import SMOTE
from sklearn.metrics import confusion_matrix, roc_curve, roc_auc_score, f1_score, accuracy_score, matthews_corrcoef
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.svm import SVC
import joblib
from scipy.stats import loguniform, uniform
import matplotlib.pyplot as plt
import warnings
from sklearn.metrics import make_scorer, recall_score

warnings.filterwarnings('ignore')

# ---------------------- 1. 配置 ----------------------
SAVE_DIR = r"E:\SVM_Results_MACCS"
os.makedirs(SAVE_DIR, exist_ok=True)


# ---------------------- 2. 数据加载和预处理 ----------------------
def load_and_preprocess_data():
    # 读取数据
    df_x_train = pd.read_csv(r"E:\训练集测试集4\hfX_train_maccs.csv", na_values=["?", "NA", " ", ""])
    df_y_train = pd.read_csv(r"E:\训练集测试集4\hfy_train_maccs.csv", na_values=["?", "NA", " ", ""])
    df_x_test = pd.read_csv(r"E:\训练集测试集4\hfX_test_maccs.csv", na_values=["?", "NA", " ", ""])
    df_y_test = pd.read_csv(r"E:\训练集测试集4\hfy_test_maccs.csv", na_values=["?", "NA", " ", ""])

    # 转换为数值
    for df in [df_x_train, df_x_test]:
        for col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')

    # 合并
    x_train = df_x_train.values
    y_train = df_y_train.iloc[:, -1].values.ravel()
    x_test = df_x_test.values
    y_test = df_y_test.iloc[:, -1].values.ravel()

    # 处理缺失值
    imputer = SimpleImputer(strategy='median')  # 使用median而不是mean
    x_train = imputer.fit_transform(x_train)
    x_test = imputer.transform(x_test)

    # 标准化
    scaler = StandardScaler()
    x_train_scaled = scaler.fit_transform(x_train)
    x_test_scaled = scaler.transform(x_test)

    return x_train_scaled, y_train, x_test_scaled, y_test, scaler


# ---------------------- 3. 自定义评分函数 ----------------------
def specificity_score(y_true, y_pred):
    """
    计算特异度 (Specificity)
    """
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    return tn / (tn + fp) if (tn + fp) > 0 else 0


def sensitivity_score(y_true, y_pred):
    """
    计算灵敏度 (Sensitivity)
    """
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    return tp / (tp + fn) if (tp + fn) > 0 else 0


# 创建自定义评分器
specificity_scorer = make_scorer(specificity_score)
sensitivity_scorer = make_scorer(sensitivity_score)


# ---------------------- 4. 改进的SVM模型 ----------------------
def train_improved_svm(x_train, y_train, x_test, y_test):
    # 计算类别权重
    n_pos = np.sum(y_train == 1)
    n_neg = np.sum(y_train == 0)
    pos_weight = n_neg / n_pos if n_pos > 0 else 1
    print(f"训练集样本分布: 正类={n_pos}, 负类={n_neg}, 比例={n_pos / len(y_train):.3f}")
    print(f"建议类别权重: 0:1.0, 1:{pos_weight:.2f}")

    # 定义模型 - 增加正则化
    svm_model = SVC(
        probability=True,
        random_state=42,
        max_iter=10000,  # 减少迭代次数，防止过拟合
        cache_size=500,
        tol=1e-3,  # 增加容忍度，加速收敛
        verbose=0
    )

    # 修改后的参数空间（重点防止过拟合）
    param_dist = {
        'C': loguniform(0.01, 1.0),  # 减小C的范围，增加正则化
        'gamma': ['scale', 'auto', 0.001, 0.005, 0.01],  # 减小gamma范围
        'kernel': ['rbf', 'linear'],
        'class_weight': [
            'balanced',
            {0: 1.0, 1: pos_weight * 0.8},  # 降低正类权重
            {0: 1.0, 1: pos_weight * 0.6},
            {0: 1.0, 1: pos_weight * 0.4},  # 尝试更低权重
            {0: 1.0, 1: 1.0}
        ],
        'tol': [1e-3, 1e-4],
        'shrinking': [True],
        'coef0': uniform(0.0, 1.0)
    }

    # 如果特征很多，考虑增加线性核的权重
    if x_train.shape[1] > 100:  # 特征维度大于100
        param_dist['kernel'] = ['linear', 'rbf']  # 优先线性核
        param_dist['gamma'] = ['scale', 'auto']  # 简化gamma选择

    # 使用分层交叉验证
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    # 定义多个评分指标
    scoring = {
        'roc_auc': 'roc_auc',
        'accuracy': 'accuracy',
        'f1': 'f1',
        'precision': 'precision',
        'recall': 'recall',
        'specificity': specificity_scorer,
        'sensitivity': sensitivity_scorer
    }

    # 随机搜索
    random_search = RandomizedSearchCV(
        svm_model,
        param_distributions=param_dist,
        n_iter=80,  # 减少迭代次数
        cv=cv,
        scoring=scoring,
        refit='roc_auc',  # 使用AUC作为主要优化指标
        n_jobs=-1,
        random_state=42,
        verbose=1,
        return_train_score=True
    )

    print("开始参数搜索...")
    random_search.fit(x_train, y_train)

    print(f"\n最佳参数: {random_search.best_params_}")
    print(f"最佳交叉验证AUC: {random_search.best_score_:.4f}")

    # 检查交叉验证的性能
    cv_results = random_search.cv_results_

    # 获取最佳参数对应的索引
    best_idx = random_search.best_index_

    # 安全地获取各个指标
    print("\n交叉验证性能:")
    print(f"AUC: {cv_results['mean_test_roc_auc'][best_idx]:.4f}")
    print(f"准确率: {cv_results['mean_test_accuracy'][best_idx]:.4f}")
    print(f"灵敏度: {cv_results['mean_test_sensitivity'][best_idx]:.4f}")
    print(f"特异度: {cv_results['mean_test_specificity'][best_idx]:.4f}")
    print(f"F1分数: {cv_results['mean_test_f1'][best_idx]:.4f}")
    print(f"精确率: {cv_results['mean_test_precision'][best_idx]:.4f}")
    print(f"召回率: {cv_results['mean_test_recall'][best_idx]:.4f}")

    return random_search.best_estimator_, random_search.best_params_


# ---------------------- 5. 评估函数 ----------------------
def evaluate_model(model, x_train, y_train, x_test, y_test, scaler):
    # 在训练集上评估
    y_train_pred = model.predict(x_train)
    y_train_proba = model.predict_proba(x_train)[:, 1]

    # 在测试集上评估
    y_test_pred = model.predict(x_test)
    y_test_proba = model.predict_proba(x_test)[:, 1]

    # 计算指标
    def calculate_metrics(y_true, y_pred, y_proba):
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()

        metrics = {
            'AUC': roc_auc_score(y_true, y_proba),
            'Accuracy': accuracy_score(y_true, y_pred),
            'Sensitivity': tp / (tp + fn) if (tp + fn) > 0 else 0,
            'Specificity': tn / (tn + fp) if (tn + fp) > 0 else 0,
            'F1': f1_score(y_true, y_pred),
            'MCC': matthews_corrcoef(y_true, y_pred),
            'TP': tp, 'FP': fp, 'TN': tn, 'FN': fn,
            'Precision': tp / (tp + fp) if (tp + fp) > 0 else 0,
            'NPV': tn / (tn + fn) if (tn + fn) > 0 else 0
        }
        return metrics

    train_metrics = calculate_metrics(y_train, y_train_pred, y_train_proba)
    test_metrics = calculate_metrics(y_test, y_test_pred, y_test_proba)

    # 打印结果
    print("\n" + "=" * 60)
    print("训练集性能:")
    print("=" * 60)
    for key, value in train_metrics.items():
        if key not in ['TP', 'FP', 'TN', 'FN']:
            print(f"{key}: {value:.4f}")

    print("\n" + "=" * 60)
    print("测试集性能:")
    print("=" * 60)
    for key, value in test_metrics.items():
        if key not in ['TP', 'FP', 'TN', 'FN']:
            print(f"{key}: {value:.4f}")

    print(f"\n混淆矩阵:")
    print(f"TP: {test_metrics['TP']}, FP: {test_metrics['FP']}")
    print(f"FN: {test_metrics['FN']}, TN: {test_metrics['TN']}")

    return test_metrics, y_test_proba


# ---------------------- 6. 绘制ROC曲线（简化版） ----------------------
def plot_roc_curve_simple(y_true, y_proba, save_path):
    fpr, tpr, thresholds = roc_curve(y_true, y_proba)
    auc_score = roc_auc_score(y_true, y_proba)

    plt.figure(figsize=(10, 8))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {auc_score:.3f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate', fontsize=12)
    plt.ylabel('True Positive Rate', fontsize=12)
    plt.title('Receiver Operating Characteristic (ROC) Curve', fontsize=14)
    plt.legend(loc="lower right")
    plt.grid(True, alpha=0.3)

    # 注意：移除了最佳阈值的红点和标签

    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

    # 仍然计算最佳阈值，但不显示在图上
    youden_idx = np.argmax(tpr - fpr)
    best_threshold = thresholds[youden_idx]

    return best_threshold


# ---------------------- 7. 性能提升建议函数 ----------------------
def performance_enhancement_suggestions(test_metrics):
    """
    根据当前性能提供提升建议
    """
    print("\n" + "=" * 60)
    print("性能提升建议:")
    print("=" * 60)

    if test_metrics['AUC'] < 0.8:
        print(f"1. AUC ({test_metrics['AUC']:.4f}) 低于0.8，建议:")
        print("   - 使用特征选择减少过拟合")
        print("   - 尝试不同的核函数或线性核")
        print("   - 调整正则化参数C值")

    if test_metrics['MCC'] < 0.4:
        print(f"2. MCC ({test_metrics['MCC']:.4f}) 低于0.4，建议:")
        print("   - 类别不平衡可能是问题，调整class_weight")
        print("   - 尝试SMOTE过采样")
        print("   - 检查特征质量，可能需要特征工程")

    if test_metrics['Sensitivity'] < 0.6 or test_metrics['Specificity'] < 0.7:
        print(f"3. 灵敏度({test_metrics['Sensitivity']:.4f})或特异度({test_metrics['Specificity']:.4f})不理想，建议:")
        print("   - 调整决策阈值")
        print("   - 优化class_weight参数")
        print("   - 考虑代价敏感学习")

    # 检查是否过拟合
    print("\n4. 通用建议:")
    print("   - 增加训练数据量")
    print("   - 使用交叉验证调整超参数")
    print("   - 考虑特征降维（PCA等）")
    print("=" * 60)


# ---------------------- 8. 主函数 ----------------------
def main():
    print("开始改进的SVM模型训练...")

    # 1. 加载和预处理数据
    x_train, y_train, x_test, y_test, scaler = load_and_preprocess_data()

    # 保存标准化器
    scaler_path = os.path.join(SAVE_DIR, "hf_svm_scaler.pkl")
    joblib.dump(scaler, scaler_path)
    print(f"标准化器已保存到: {scaler_path}")

    # 2. 训练模型
    best_model, best_params = train_improved_svm(x_train, y_train, x_test, y_test)

    # 3. 评估模型
    test_metrics, y_test_proba = evaluate_model(best_model, x_train, y_train,
                                                x_test, y_test, scaler)

    # 4. 绘制简化版ROC曲线（无红点和阈值标记）
    roc_path = os.path.join(SAVE_DIR, "hf_svm_roc_simple.pdf")
    best_threshold = plot_roc_curve_simple(y_test, y_test_proba, roc_path)
    print(f"\n最佳阈值: {best_threshold:.4f}")

    # 使用最佳阈值重新预测
    y_test_pred_optimal = (y_test_proba >= best_threshold).astype(int)
    optimal_f1 = f1_score(y_test, y_test_pred_optimal)
    optimal_se = confusion_matrix(y_test, y_test_pred_optimal).ravel()[3] / np.sum(y_test == 1)
    optimal_sp = confusion_matrix(y_test, y_test_pred_optimal).ravel()[0] / np.sum(y_test == 0)

    print(f"使用最佳阈值的性能:")
    print(f"灵敏度: {optimal_se:.4f}, 特异度: {optimal_sp:.4f}, F1分数: {optimal_f1:.4f}")

    # 5. 保存模型和结果
    model_path = os.path.join(SAVE_DIR, "hf_svm_model.pkl")
    joblib.dump(best_model, model_path)
    print(f"\n模型已保存到: {model_path}")

    # 保存预测结果
    np.save(os.path.join(SAVE_DIR, "hf_y_test.npy"), y_test)
    np.save(os.path.join(SAVE_DIR, "hf_y_pred_proba.npy"), y_test_proba)
    np.save(os.path.join(SAVE_DIR, "hf_y_pred_optimal.npy"), y_test_pred_optimal)

    # 保存结果到CSV
    results_df = pd.DataFrame({
        '参数': [str(best_params)],
        '测试集AUC': [test_metrics['AUC']],
        '准确率': [test_metrics['Accuracy']],
        '灵敏度': [test_metrics['Sensitivity']],
        '特异度': [test_metrics['Specificity']],
        'F1分数': [test_metrics['F1']],
        'MCC': [test_metrics['MCC']],
        '精确率': [test_metrics['Precision']],
        'NPV': [test_metrics['NPV']],
        '最佳阈值': [best_threshold],
        '最佳阈值F1': [optimal_f1],
        '最佳阈值灵敏度': [optimal_se],
        '最佳阈值特异度': [optimal_sp]
    })

    results_path = os.path.join(SAVE_DIR, "hf_svm_results.csv")
    results_df.to_csv(results_path, index=False)
    print(f"结果已保存到: {results_path}")

    # 6. 提供性能提升建议
    performance_enhancement_suggestions(test_metrics)

    print("\n" + "=" * 60)
    print("改进完成！")
    print("=" * 60)


# ---------------------- 9. 运行 ----------------------
if __name__ == "__main__":
    main()
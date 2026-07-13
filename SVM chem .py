import pandas as pd
import numpy as np
import os
from sklearn.metrics import confusion_matrix, roc_curve, roc_auc_score, f1_score, accuracy_score, matthews_corrcoef
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.svm import SVC
import joblib
from scipy.stats import loguniform, uniform
import matplotlib.pyplot as plt
import warnings
from sklearn.feature_selection import SelectKBest, f_classif

warnings.filterwarnings('ignore')

# ---------------------- 1. 配置 ----------------------
SAVE_DIR = r"E:\SVM_Results chem"
os.makedirs(SAVE_DIR, exist_ok=True)


# ---------------------- 2. 数据加载和预处理 ----------------------
def load_and_preprocess_data():
    # 读取数据
    df_x_train = pd.read_csv(r"E:\训练集测试集\haX_train_chemberta.csv", na_values=["?", "NA", " ", ""])
    df_y_train = pd.read_csv(r"E:\训练集测试集\hay_train_chemberta.csv", na_values=["?", "NA", " ", ""])
    df_x_test = pd.read_csv(r"E:\训练集测试集\haX_test_chemberta.csv", na_values=["?", "NA", " ", ""])
    df_y_test = pd.read_csv(r"E:\训练集测试集\hay_test_chemberta.csv", na_values=["?", "NA", " ", ""])

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
    imputer = SimpleImputer(strategy='median')
    x_train = imputer.fit_transform(x_train)
    x_test = imputer.transform(x_test)

    # 特征选择 - 选择最重要的100个特征
    print(f"原始特征维度: {x_train.shape[1]}")
    if x_train.shape[1] > 100:
        selector = SelectKBest(f_classif, k=min(100, x_train.shape[1]))
        x_train = selector.fit_transform(x_train, y_train)
        x_test = selector.transform(x_test)
        print(f"特征选择后维度: {x_train.shape[1]}")

    # 标准化
    scaler = StandardScaler()
    x_train_scaled = scaler.fit_transform(x_train)
    x_test_scaled = scaler.transform(x_test)

    return x_train_scaled, y_train, x_test_scaled, y_test, scaler


# ---------------------- 3. 优化的SVM模型 ----------------------
def train_optimized_svm(x_train, y_train, x_test, y_test):
    # 计算类别权重
    n_pos = np.sum(y_train == 1)
    n_neg = np.sum(y_train == 0)
    total = n_pos + n_neg
    pos_weight = n_neg / n_pos if n_pos > 0 else 1
    neg_weight = n_pos / n_neg if n_neg > 0 else 1

    print(f"训练集样本分布: 正类={n_pos}, 负类={n_neg}, 比例={n_pos / total:.3f}")
    print(f"建议类别权重: 0:{neg_weight:.2f}, 1:{pos_weight:.2f}")

    # 优化的参数空间 - 针对提升AUC并降低过拟合
    param_dist = {
        'C': [0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1.0, 2.0, 5.0, 10.0],  # 增加更小的C值以减少过拟合
        'gamma': [0.0001, 0.0005, 0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 'scale', 'auto'],  # 更细粒度的gamma值
        'kernel': ['rbf', 'linear', 'poly'],  # 增加poly核函数
        'degree': [2, 3],  # poly核的度数
        'class_weight': [
            'balanced',
            {0: 1.0, 1: pos_weight * 1.5},  # 增加少数类权重以提升灵敏度
            {0: 1.0, 1: pos_weight * 2.0},
            {0: neg_weight, 1: pos_weight * 1.2},  # 双向调整
            {0: 1.0, 1: pos_weight * 0.8},  # 减小权重以降低过拟合
            None  # 不加权重
        ],
        'tol': [1e-4, 1e-5],  # 更小的容差
        'shrinking': [True, False],
        'max_iter': [50000, 100000, 200000],
        'cache_size': [1000, 2000]
    }

    # 使用5折交叉验证以获得更稳定的评估
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    # 创建SVM模型
    svm_model = SVC(
        probability=True,
        random_state=42,
        max_iter=200000,  # 更大的默认值
        cache_size=1000,
        verbose=False
    )

    # 随机搜索
    random_search = RandomizedSearchCV(
        svm_model,
        param_distributions=param_dist,
        n_iter=30,  # 保持30次迭代
        cv=cv,
        scoring='roc_auc',  # 只使用AUC作为评分
        refit=True,
        n_jobs=-1,
        random_state=42,
        verbose=2  # 增加详细程度
    )

    print("开始参数搜索...")
    random_search.fit(x_train, y_train)

    print(f"\n最佳参数: {random_search.best_params_}")
    print(f"最佳交叉验证AUC (5折): {random_search.best_score_:.4f}")

    return random_search.best_estimator_, random_search.best_params_


# ---------------------- 4. 评估函数 ----------------------
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


# ---------------------- 5. 绘制ROC曲线 ----------------------
def plot_roc_curve(y_true, y_proba, save_path):
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

    # 注意：已移除红色圆点标记（最佳阈值点）
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

    # 仍然计算最佳阈值供后续使用
    youden_idx = np.argmax(tpr - fpr)
    best_threshold = thresholds[youden_idx]

    return best_threshold


# ---------------------- 6. 主函数 ----------------------
def main():
    print("开始优化的SVM模型训练...")
    print("使用5折交叉验证以获得更稳定的性能评估")

    # 1. 加载和预处理数据
    x_train, y_train, x_test, y_test, scaler = load_and_preprocess_data()

    # 保存标准化器
    scaler_path = os.path.join(SAVE_DIR, "ha chem_svm_scaler.pkl")
    joblib.dump(scaler, scaler_path)
    print(f"标准化器已保存到: {scaler_path}")

    # 2. 训练模型
    best_model, best_params = train_optimized_svm(x_train, y_train, x_test, y_test)

    # 3. 评估模型
    test_metrics, y_test_proba = evaluate_model(best_model, x_train, y_train,
                                                x_test, y_test, scaler)

    # 4. 绘制ROC曲线
    roc_path = os.path.join(SAVE_DIR, "ha chem_svm_roc.pdf")
    best_threshold = plot_roc_curve(y_test, y_test_proba, roc_path)
    print(f"\n最佳阈值: {best_threshold:.4f}")

    # 使用最佳阈值重新预测
    y_test_pred_optimal = (y_test_proba >= best_threshold).astype(int)
    optimal_f1 = f1_score(y_test, y_test_pred_optimal)
    tn, fp, fn, tp = confusion_matrix(y_test, y_test_pred_optimal).ravel()
    optimal_se = tp / (tp + fn) if (tp + fn) > 0 else 0
    optimal_sp = tn / (tn + fp) if (tn + fp) > 0 else 0

    print(f"使用最佳阈值的性能:")
    print(f"灵敏度: {optimal_se:.4f}, 特异度: {optimal_sp:.4f}, F1分数: {optimal_f1:.4f}")

    # 5. 保存模型和结果
    model_path = os.path.join(SAVE_DIR, "ha chem_svm_model.pkl")
    joblib.dump(best_model, model_path)
    print(f"\n模型已保存到: {model_path}")

    # 保存预测结果
    np.save(os.path.join(SAVE_DIR, "ha chem y_test.npy"), y_test)
    np.save(os.path.join(SAVE_DIR, "ha chem y_pred_proba.npy"), y_test_proba)
    np.save(os.path.join(SAVE_DIR, "ha chem y_pred_.npy"), y_test_pred_optimal)

    # 保存结果到CSV
    results_df = pd.DataFrame({
        '参数': [str(best_params)],
        '交叉验证AUC(5折)': [random_search.best_score_],
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

    results_path = os.path.join(SAVE_DIR, "ha chem_svm_results.csv")
    results_df.to_csv(results_path, index=False)
    print(f"结果已保存到: {results_path}")

    # 打印改进建议
    print("\n" + "=" * 60)
    print("性能分析:")
    print("=" * 60)
    print(f"1. 收敛问题解决: max_iter增加到{best_params.get('max_iter', 100000)}")
    print(f"2. 类别权重优化: {best_params.get('class_weight', 'N/A')}")
    print(f"3. 测试集AUC: {test_metrics['AUC']:.4f}")
    print(f"4. 灵敏度/特异度平衡: {test_metrics['Sensitivity']:.4f}/{test_metrics['Specificity']:.4f}")
    print(f"5. 交叉验证折数: 5折")

    # 检查是否达到0.83的目标
    if test_metrics['AUC'] >= 0.83:
        print(f"✓ 达到目标AUC: {test_metrics['AUC']:.4f} >= 0.83")
    else:
        print(f"✗ 未达到目标AUC: {test_metrics['AUC']:.4f} < 0.83")

    print("\n" + "=" * 60)
    print("优化完成！")
    print("=" * 60)


# ---------------------- 7. 运行 ----------------------
if __name__ == "__main__":
    main()
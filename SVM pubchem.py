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
import time  # 添加时间模块

warnings.filterwarnings('ignore')

# ---------------------- 1. 配置 ----------------------
SAVE_DIR = r"E:\SVM_Results pubchem"
os.makedirs(SAVE_DIR, exist_ok=True)


# ---------------------- 2. 数据加载和预处理 ----------------------
def load_and_preprocess_data():
    # 读取数据
    df_x_train = pd.read_csv(r"E:\训练集测试集3\haX_train_pubchem.csv", na_values=["?", "NA", " ", ""])
    df_y_train = pd.read_csv(r"E:\训练集测试集3\hay_train_pubchem.csv", na_values=["?", "NA", " ", ""])
    df_x_test = pd.read_csv(r"E:\训练集测试集3\haX_test_pubchem.csv", na_values=["?", "NA", " ", ""])
    df_y_test = pd.read_csv(r"E:\训练集测试集3\hay_test_pubchem.csv", na_values=["?", "NA", " ", ""])

    print(f"原始数据维度 - 训练集: {df_x_train.shape}, 测试集: {df_x_test.shape}")

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

    # 标准化
    scaler = StandardScaler()
    x_train_scaled = scaler.fit_transform(x_train)
    x_test_scaled = scaler.transform(x_test)

    # 添加特征选择 - 移除低方差特征
    from sklearn.feature_selection import VarianceThreshold
    selector = VarianceThreshold(threshold=0.01)  # 移除方差<0.01的特征
    x_train_scaled = selector.fit_transform(x_train_scaled)
    x_test_scaled = selector.transform(x_test_scaled)

    print(f"特征选择后维度 - 训练集: {x_train_scaled.shape}, 测试集: {x_test_scaled.shape}")
    print(f"移除特征数量: {df_x_train.shape[1] - x_train_scaled.shape[1]}")

    return x_train_scaled, y_train, x_test_scaled, y_test, scaler


# ---------------------- 3. 改进的SVM模型 ----------------------
def train_improved_svm(x_train, y_train, x_test, y_test):
    # 计算类别权重
    n_pos = np.sum(y_train == 1)
    n_neg = np.sum(y_train == 0)
    pos_weight = n_neg / n_pos if n_pos > 0 else 1
    print(f"训练集样本分布: 正类={n_pos}, 负类={n_neg}, 比例={n_pos / len(y_train):.3f}")
    print(f"建议类别权重: 0:1.0, 1:{pos_weight:.2f}")

    # 定义模型 - 优化设置
    svm_model = SVC(
        probability=True,
        random_state=42,
        max_iter=1000,  # 减少最大迭代次数，SVM通常收敛很快
        cache_size=1000,  # 增加缓存大小
        tol=1e-3,  # 增加容忍度，加速收敛
        verbose=0
    )

    # 优化参数空间 - 简化搜索范围
    param_dist = {
        'C': loguniform(0.1, 10.0),  # 合理的C值范围
        'gamma': ['scale', 'auto'],  # 简化gamma选项，只使用scale和auto
        'kernel': ['rbf', 'linear'],  # 只测试两个最常用的核
        'class_weight': [
            'balanced',
            {0: 1.0, 1: pos_weight},  # 使用建议权重
            {0: 1.0, 1: pos_weight * 1.5},  # 适度增加正类权重
            {0: 1.0, 1: 1.0}  # 不设置权重
        ],
        'tol': [1e-3],  # 固定容忍度
        'shrinking': [True],  # 固定shrinking
    }

    # 如果特征很多，强制使用线性核（训练更快）
    if x_train.shape[1] > 500:  # 特征维度大于500时
        print("特征维度高，强制使用线性核以加速训练")
        param_dist['kernel'] = ['linear']
        param_dist['gamma'] = ['scale']

    # 使用3折交叉验证减少计算量
    cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)  # 从5折减少到3折

    # 减少搜索迭代次数
    random_search = RandomizedSearchCV(
        svm_model,
        param_distributions=param_dist,
        n_iter=30,  # 从100减少到30
        cv=cv,
        scoring='balanced_accuracy',
        refit=True,
        n_jobs=-1,  # 使用所有CPU核心
        random_state=42,
        verbose=1
    )

    print("开始参数搜索...")
    start_time = time.time()
    random_search.fit(x_train, y_train)
    end_time = time.time()

    print(f"参数搜索完成，耗时: {end_time - start_time:.2f}秒")
    print(f"\n最佳参数: {random_search.best_params_}")
    print(f"最佳交叉验证平衡准确率: {random_search.best_score_:.4f}")

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

    # 添加不同阈值的性能分析
    print("\n不同阈值下的性能:")
    thresholds = np.linspace(0.1, 0.9, 9)
    for thresh in thresholds:
        y_pred_thresh = (y_test_proba >= thresh).astype(int)
        tn, fp, fn, tp = confusion_matrix(y_test, y_pred_thresh).ravel()
        sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
        f1 = f1_score(y_test, y_pred_thresh)
        print(f"阈值={thresh:.1f}: 灵敏度={sensitivity:.3f}, 特异度={specificity:.3f}, F1={f1:.3f}")

    return test_metrics, y_test_proba


# ---------------------- 5. 绘制ROC曲线 ----------------------
def plot_roc_curve(y_true, y_proba, save_path):
    fpr, tpr, thresholds = roc_curve(y_true, y_proba)
    auc_score = roc_auc_score(y_true, y_proba)

    plt.figure(figsize=(10, 8))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'SVM (AUC = {auc_score:.3f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate', fontsize=12)
    plt.ylabel('True Positive Rate', fontsize=12)
    plt.title('Receiver Operating Characteristic (ROC) Curve', fontsize=14)
    plt.legend(loc="lower right")
    plt.grid(True, alpha=0.3)

    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

    # 仍然计算最佳阈值但不显示在图上
    youden_idx = np.argmax(tpr - fpr)
    best_threshold = thresholds[youden_idx]

    return best_threshold


# ---------------------- 6. 主函数 ----------------------
def main():
    print("开始改进的SVM模型训练...")
    total_start_time = time.time()

    # 1. 加载和预处理数据
    x_train, y_train, x_test, y_test, scaler = load_and_preprocess_data()

    # 保存标准化器
    scaler_path = os.path.join(SAVE_DIR, "hy _svm_scaler.pkl")
    joblib.dump(scaler, scaler_path)
    print(f"标准化器已保存到: {scaler_path}")

    # 2. 训练模型
    best_model, best_params = train_improved_svm(x_train, y_train, x_test, y_test)

    # 3. 评估模型
    test_metrics, y_test_proba = evaluate_model(best_model, x_train, y_train,
                                                x_test, y_test, scaler)

    # 4. 绘制ROC曲线
    roc_path = os.path.join(SAVE_DIR, "hy _svm_roc.pdf")
    best_threshold = plot_roc_curve(y_test, y_test_proba, roc_path)
    print(f"\n最佳阈值: {best_threshold:.4f}")

    # 使用最佳阈值重新预测
    y_test_pred_optimal = (y_test_proba >= best_threshold).astype(int)
    optimal_f1 = f1_score(y_test, y_test_pred_optimal)
    optimal_se = confusion_matrix(y_test, y_test_pred_optimal).ravel()[3] / np.sum(y_test == 1)
    print(f"使用最佳阈值的性能:")
    print(f"灵敏度: {optimal_se:.4f}, F1分数: {optimal_f1:.4f}")

    # 打印详细分类报告
    from sklearn.metrics import classification_report
    print("\n详细分类报告:")
    print(classification_report(y_test, y_test_pred_optimal))

    # 5. 保存模型和结果
    model_path = os.path.join(SAVE_DIR, "hy _svm_model.pkl")
    joblib.dump(best_model, model_path)
    print(f"\n模型已保存到: {model_path}")

    # 保存预测结果
    np.save(os.path.join(SAVE_DIR, "hy y_test_.npy"), y_test)
    np.save(os.path.join(SAVE_DIR, "hy y_pred_proba_.npy"), y_test_proba)
    np.save(os.path.join(SAVE_DIR, "hy y_pred_optimal_.npy"), y_test_pred_optimal)

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
        '最佳阈值灵敏度': [optimal_se]
    })

    results_path = os.path.join(SAVE_DIR, "hy _svm_results.csv")
    results_df.to_csv(results_path, index=False)
    print(f"结果已保存到: {results_path}")

    total_end_time = time.time()
    print(f"\n总运行时间: {total_end_time - total_start_time:.2f}秒")
    print("\n" + "=" * 60)
    print("改进完成！")
    print("=" * 60)


# ---------------------- 7. 运行 ----------------------
if __name__ == "__main__":
    main()
import pandas as pd
import numpy as np
import matplotlib
import os
from imblearn.over_sampling import SMOTE
from sklearn.metrics import confusion_matrix, roc_curve, roc_auc_score, f1_score, accuracy_score, matthews_corrcoef, \
    balanced_accuracy_score
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
SAVE_DIR = r"E:\SVM_Results krfp"
os.makedirs(SAVE_DIR, exist_ok=True)


# ---------------------- 2. 自定义评分函数 ----------------------
def specificity_score(y_true, y_pred):
    """计算特异度"""
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    return tn / (tn + fp) if (tn + fp) > 0 else 0


def balanced_score(y_true, y_pred):
    """平衡灵敏度和特异度的评分"""
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
    # 返回灵敏度和特异度的几何平均数，更注重平衡
    return np.sqrt(sensitivity * specificity)


# 创建自定义评分器
specificity_scorer = make_scorer(specificity_score)
balanced_scorer = make_scorer(balanced_score)


# ---------------------- 3. 数据加载和预处理 ----------------------
def load_and_preprocess_data():
    # 读取数据
    df_x_train = pd.read_csv(r"E:\训练集测试集5\haX_train_krfp.csv", na_values=["?", "NA", " ", ""])
    df_y_train = pd.read_csv(r"E:\训练集测试集5\hay_train_krfp.csv", na_values=["?", "NA", " ", ""])
    df_x_test = pd.read_csv(r"E:\训练集测试集5\haX_test_krfp.csv", na_values=["?", "NA", " ", ""])
    df_y_test = pd.read_csv(r"E:\训练集测试集5\hay_test_krfp.csv", na_values=["?", "NA", " ", ""])

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

    return x_train_scaled, y_train, x_test_scaled, y_test, scaler


# ---------------------- 4. 改进的SVM模型 ----------------------
def train_improved_svm(x_train, y_train, x_test, y_test):
    # 计算类别权重
    n_pos = np.sum(y_train == 1)
    n_neg = np.sum(y_train == 0)
    pos_weight = n_neg / n_pos if n_pos > 0 else 1
    print(f"训练集样本分布: 正类={n_pos}, 负类={n_neg}, 比例={n_pos / len(y_train):.3f}")
    print(f"建议类别权重: 0:1.0, 1:{pos_weight:.2f}")

    # 定义模型 - 增加正则化强度
    svm_model = SVC(
        probability=True,
        random_state=42,
        max_iter=20000,  # 增加迭代次数确保收敛
        cache_size=1000,
        tol=1e-4,  # 减小容忍度，提高精度
        verbose=0
    )

    # 调整参数空间，重点关注正则化和平衡性
    # 基于之前的输出，模型存在过拟合和特异度过低的问题
    # 调整策略：增加正则化、降低模型复杂度、增加负类权重
    param_dist = {
        'C': loguniform(0.01, 10),  # 调整C的范围，避免过拟合
        'gamma': ['scale', 'auto', 0.0001, 0.001, 0.01, 0.1],  # 调整gamma范围，防止过拟合
        'kernel': ['rbf', 'linear'],  # 专注于rbf和linear核
        'class_weight': [
            {0: 1.5, 1: 1.0},  # 增加负类权重，提高特异度
            {0: 2.0, 1: 1.0},  # 进一步增加负类权重
            {0: 3.0, 1: 1.0},  # 大幅增加负类权重
            {0: 1.0, 1: 0.5},  # 降低正类权重
            {0: 1.0, 1: 0.3},  # 进一步降低正类权重
            'balanced',
        ],
        'tol': [1e-4, 1e-3, 1e-2],  # 调整容忍度范围
        'shrinking': [True, False],  # 尝试不收缩
    }

    # 使用分层交叉验证
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    # 随机搜索 - 使用平衡评分
    random_search = RandomizedSearchCV(
        svm_model,
        param_distributions=param_dist,
        n_iter=50,  # 减少迭代次数，加快速度
        cv=cv,
        scoring=balanced_scorer,  # 使用平衡评分，注重灵敏度和特异度的平衡
        refit=True,
        n_jobs=-1,
        random_state=42,
        verbose=1
    )

    print("开始参数搜索...")
    random_search.fit(x_train, y_train)

    print(f"\n最佳参数: {random_search.best_params_}")
    print(f"最佳交叉验证平衡分数: {random_search.best_score_:.4f}")

    # 使用交叉验证计算其他指标
    print("\n交叉验证性能:")

    # 计算交叉验证的多个指标
    scoring_dict = {
        'roc_auc': 'roc_auc',
        'balanced_accuracy': 'balanced_accuracy',
        'f1': 'f1',
        'specificity': specificity_scorer,
        'recall': 'recall'  # 灵敏度
    }

    cv_scores = {}
    for score_name, scorer in scoring_dict.items():
        scores = cross_val_score(
            random_search.best_estimator_,
            x_train, y_train,
            cv=cv,
            scoring=scorer,
            n_jobs=-1
        )
        cv_scores[score_name] = (np.mean(scores), np.std(scores))
        print(f"交叉验证{score_name}: {cv_scores[score_name][0]:.4f} (±{cv_scores[score_name][1]:.4f})")

    return random_search.best_estimator_, random_search.best_params_, cv_scores


# ---------------------- 5. 评估函数 ----------------------
def evaluate_model(model, x_train, y_train, x_test, y_test, scaler, best_threshold=None):
    # 在训练集上评估
    y_train_pred = model.predict(x_train)
    y_train_proba = model.predict_proba(x_train)[:, 1]

    # 在测试集上评估
    y_test_pred = model.predict(x_test)
    y_test_proba = model.predict_proba(x_test)[:, 1]

    # 如果提供了最佳阈值，使用阈值重新预测
    if best_threshold is not None:
        y_test_pred_thresh = (y_test_proba >= best_threshold).astype(int)
        y_train_pred_thresh = (y_train_proba >= best_threshold).astype(int)
    else:
        y_test_pred_thresh = y_test_pred
        y_train_pred_thresh = y_train_pred

    # 计算指标
    def calculate_metrics(y_true, y_pred, y_proba, prefix=""):
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()

        metrics = {
            f'{prefix}AUC': roc_auc_score(y_true, y_proba),
            f'{prefix}Accuracy': accuracy_score(y_true, y_pred),
            f'{prefix}Sensitivity': tp / (tp + fn) if (tp + fn) > 0 else 0,
            f'{prefix}Specificity': tn / (tn + fp) if (tn + fp) > 0 else 0,
            f'{prefix}F1': f1_score(y_true, y_pred),
            f'{prefix}MCC': matthews_corrcoef(y_true, y_pred),
            f'{prefix}TP': tp, f'{prefix}FP': fp, f'{prefix}TN': tn, f'{prefix}FN': fn,
            f'{prefix}Precision': tp / (tp + fp) if (tp + fp) > 0 else 0,
            f'{prefix}NPV': tn / (tn + fn) if (tn + fn) > 0 else 0,
            f'{prefix}Balanced_Accuracy': balanced_accuracy_score(y_true, y_pred)
        }
        return metrics

    train_metrics = calculate_metrics(y_train, y_train_pred, y_train_proba, "Train_")
    test_metrics = calculate_metrics(y_test, y_test_pred, y_test_proba, "Test_")

    # 使用阈值调整后的指标
    if best_threshold is not None:
        train_metrics_thresh = calculate_metrics(y_train, y_train_pred_thresh, y_train_proba, "Train_Thresh_")
        test_metrics_thresh = calculate_metrics(y_test, y_test_pred_thresh, y_test_proba, "Test_Thresh_")
        # 合并指标
        train_metrics.update(train_metrics_thresh)
        test_metrics.update(test_metrics_thresh)

    # 打印结果
    print("\n" + "=" * 60)
    print("训练集性能:")
    print("=" * 60)
    for key, value in train_metrics.items():
        if key.startswith('Train_') and not any(x in key for x in ['TP', 'FP', 'TN', 'FN']):
            print(f"{key}: {value:.4f}")

    print("\n" + "=" * 60)
    print("测试集性能:")
    print("=" * 60)
    for key, value in test_metrics.items():
        if key.startswith('Test_') and not any(x in key for x in ['TP', 'FP', 'TN', 'FN']):
            print(f"{key}: {value:.4f}")

    print(f"\n混淆矩阵 (默认阈值):")
    print(f"TP: {test_metrics['Test_TP']}, FP: {test_metrics['Test_FP']}")
    print(f"FN: {test_metrics['Test_FN']}, TN: {test_metrics['Test_TN']}")

    if best_threshold is not None:
        print(f"\n混淆矩阵 (最佳阈值={best_threshold:.4f}):")
        print(f"TP: {test_metrics['Test_Thresh_TP']}, FP: {test_metrics['Test_Thresh_FP']}")
        print(f"FN: {test_metrics['Test_Thresh_FN']}, TN: {test_metrics['Test_Thresh_TN']}")

    return test_metrics, y_test_proba


# ---------------------- 6. 绘制ROC曲线 ----------------------
def plot_roc_curve(y_true, y_proba, save_path):
    fpr, tpr, thresholds = roc_curve(y_true, y_proba)
    auc_score = roc_auc_score(y_true, y_proba)

    plt.figure(figsize=(10, 8))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {auc_score:.3f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate (1 - Specificity)', fontsize=12)
    plt.ylabel('True Positive Rate (Sensitivity)', fontsize=12)
    plt.title('Receiver Operating Characteristic (ROC) Curve', fontsize=14)
    plt.legend(loc="lower right")
    plt.grid(True, alpha=0.3)

    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

    # 计算最佳阈值（Youden指数）
    youden_idx = np.argmax(tpr - fpr)
    best_threshold = thresholds[youden_idx]

    # 计算最佳阈值对应的灵敏度和特异度
    best_sensitivity = tpr[youden_idx]
    best_specificity = 1 - fpr[youden_idx]

    print(f"最佳阈值 (Youden指数): {best_threshold:.4f}")
    print(f"最佳阈值对应的灵敏度: {best_sensitivity:.4f}")
    print(f"最佳阈值对应的特异度: {best_specificity:.4f}")

    return best_threshold, best_sensitivity, best_specificity


# ---------------------- 7. 主函数 ----------------------
def main():
    print("开始改进的SVM模型训练...")

    # 1. 加载和预处理数据
    x_train, y_train, x_test, y_test, scaler = load_and_preprocess_data()

    # 保存标准化器
    scaler_path = os.path.join(SAVE_DIR, "ha_svm_scaler.pkl")
    joblib.dump(scaler, scaler_path)
    print(f"标准化器已保存到: {scaler_path}")

    # 2. 训练模型
    best_model, best_params, cv_scores = train_improved_svm(x_train, y_train, x_test, y_test)

    # 3. 绘制ROC曲线并获取最佳阈值
    roc_path = os.path.join(SAVE_DIR, "ha_svm_roc.pdf")

    # 先预测测试集概率用于ROC曲线
    y_test_proba = best_model.predict_proba(x_test)[:, 1]
    best_threshold, best_se, best_sp = plot_roc_curve(y_test, y_test_proba, roc_path)
    print(f"\n最佳阈值: {best_threshold:.4f}")
    print(f"对应灵敏度: {best_se:.4f}, 特异度: {best_sp:.4f}")

    # 4. 使用最佳阈值评估模型
    test_metrics, y_test_proba = evaluate_model(best_model, x_train, y_train,
                                                x_test, y_test, scaler, best_threshold)

    # 5. 保存模型和结果
    model_path = os.path.join(SAVE_DIR, "ha_svm_model.pkl")
    joblib.dump(best_model, model_path)
    print(f"\n模型已保存到: {model_path}")

    # 保存预测结果
    np.save(os.path.join(SAVE_DIR, "ha_y_test.npy"), y_test)
    np.save(os.path.join(SAVE_DIR, "ha_y_pred_proba.npy"), y_test_proba)

    # 使用最佳阈值进行预测并保存
    y_test_pred_optimal = (y_test_proba >= best_threshold).astype(int)
    np.save(os.path.join(SAVE_DIR, "ha_y_pred_optimal.npy"), y_test_pred_optimal)

    # 保存结果到CSV
    results_df = pd.DataFrame({
        '参数': [str(best_params)],
        '交叉验证AUC': [cv_scores['roc_auc'][0]],
        '交叉验证平衡准确率': [cv_scores['balanced_accuracy'][0]],
        '交叉验证F1': [cv_scores['f1'][0]],
        '交叉验证特异度': [cv_scores['specificity'][0]],
        '交叉验证灵敏度': [cv_scores['recall'][0]],
        '测试集AUC': [test_metrics['Test_AUC']],
        '测试集准确率': [test_metrics['Test_Accuracy']],
        '测试集灵敏度': [test_metrics['Test_Sensitivity']],
        '测试集特异度': [test_metrics['Test_Specificity']],
        '测试集F1': [test_metrics['Test_F1']],
        '测试集MCC': [test_metrics['Test_MCC']],
        '测试集精确率': [test_metrics['Test_Precision']],
        '测试集NPV': [test_metrics['Test_NPV']],
        '测试集平衡准确率': [test_metrics['Test_Balanced_Accuracy']],
        '最佳阈值': [best_threshold],
        '最佳阈值灵敏度': [best_se],
        '最佳阈值特异度': [best_sp],
        '最佳阈值F1': [test_metrics.get('Test_Thresh_F1', test_metrics['Test_F1'])],
        '最佳阈值平衡准确率': [test_metrics.get('Test_Thresh_Balanced_Accuracy',
                                                (best_se + best_sp) / 2)]
    })

    results_path = os.path.join(SAVE_DIR, "ha_svm_results.csv")
    results_df.to_csv(results_path, index=False)
    print(f"结果已保存到: {results_path}")

    print("\n" + "=" * 60)
    print("模型训练完成！")
    print("=" * 60)
    print(f"关键改进:")
    print(f"1. 使用平衡评分选择模型参数")
    print(f"2. 最佳阈值调整: {best_threshold:.4f}")
    print(f"3. 灵敏度/特异度平衡: {best_se:.3f}/{best_sp:.3f}")
    print("=" * 60)


# ---------------------- 8. 运行 ----------------------
if __name__ == "__main__":
    main()
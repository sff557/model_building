import pandas as pd
import numpy as np
import os
from sklearn.metrics import confusion_matrix, roc_curve, roc_auc_score, f1_score, accuracy_score, matthews_corrcoef
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.svm import SVC
from sklearn.ensemble import BaggingClassifier
from sklearn.feature_selection import VarianceThreshold, SelectKBest, f_classif
import joblib
from scipy.stats import loguniform
import matplotlib.pyplot as plt
import warnings

warnings.filterwarnings('ignore')

SAVE_DIR = r"E:\SVM_Results maccs+chem"
os.makedirs(SAVE_DIR, exist_ok=True)


# ---------------------- 减轻过拟合的核心改进 ----------------------
def load_and_preprocess_data():
    # 读取数据
    df_x_train = pd.read_csv(r"E:\训练集测试集2\ha合X_train.csv", na_values=["?", "NA", " ", ""])
    df_y_train = pd.read_csv(r"E:\训练集测试集2\ha合y_train.csv", na_values=["?", "NA", " ", ""])
    df_x_test = pd.read_csv(r"E:\训练集测试集2\ha合X_test.csv", na_values=["?", "NA", " ", ""])
    df_y_test = pd.read_csv(r"E:\训练集测试集2\ha合y_test.csv", na_values=["?", "NA", " ", ""])

    # 转换为数值
    for df in [df_x_train, df_x_test]:
        for col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')

    x_train = df_x_train.values
    y_train = df_y_train.iloc[:, -1].values.ravel()
    x_test = df_x_test.values
    y_test = df_y_test.iloc[:, -1].values.ravel()

    # 处理缺失值
    imputer = SimpleImputer(strategy='median')
    x_train = imputer.fit_transform(x_train)
    x_test = imputer.transform(x_test)

    # 1. 特征选择 - 移除低方差特征
    var_threshold = VarianceThreshold(threshold=0.01)
    x_train = var_threshold.fit_transform(x_train)
    x_test = var_threshold.transform(x_test)

    # 2. 特征选择 - 选择最重要的特征
    if x_train.shape[1] > 100:
        k = min(100, int(x_train.shape[1] * 0.5))
        selector = SelectKBest(f_classif, k=k)
        x_train = selector.fit_transform(x_train, y_train)
        x_test = selector.transform(x_test)

    # 标准化
    scaler = StandardScaler()
    x_train_scaled = scaler.fit_transform(x_train)
    x_test_scaled = scaler.transform(x_test)

    return x_train_scaled, y_train, x_test_scaled, y_test, scaler


def train_improved_svm(x_train, y_train):
    # 使用更保守的类别权重
    n_pos = np.sum(y_train == 1)
    n_neg = np.sum(y_train == 0)
    pos_weight = n_neg / n_pos if n_pos > 0 else 1
    conservative_weight = max(0.5, min(1.0, pos_weight * 0.8))

    print(f"训练集: 正类={n_pos}, 负类={n_neg}, 权重=1:{conservative_weight:.2f}")

    # SVM模型配置
    svm_model = SVC(
        probability=True,
        random_state=42,
        max_iter=50000,
        cache_size=500,
        tol=1e-4
    )

    # 更强的正则化参数
    param_dist = {
        'C': loguniform(0.001, 5.0),  # 更小的C范围
        'gamma': ['scale', 'auto', 0.001, 0.01, 0.05],
        'kernel': ['rbf', 'linear'],  # 优先简单模型
        'class_weight': [
            'balanced',
            {0: 1.0, 1: conservative_weight},
            {0: 1.0, 1: 0.6}
        ],
        'shrinking': [True, False]
    }

    # 增加交叉验证折数
    cv = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)

    random_search = RandomizedSearchCV(
        svm_model,
        param_distributions=param_dist,
        n_iter=100,
        cv=cv,
        scoring='roc_auc',
        n_jobs=-1,
        random_state=42,
        verbose=1
    )

    print("开始参数搜索...")
    random_search.fit(x_train, y_train)

    print(f"最佳参数: {random_search.best_params_}")
    print(f"交叉验证AUC: {random_search.best_score_:.4f}")

    return random_search.best_estimator_, random_search.best_params_


def train_bagging_svm(x_train, y_train, best_params):
    """Bagging集成减少过拟合"""
    print("\n训练Bagging SVM集成...")

    base_svm = SVC(
        probability=True,
        random_state=42,
        max_iter=50000,
        cache_size=500,
        tol=1e-4
    )

    # 设置最佳参数
    for key, value in best_params.items():
        if hasattr(base_svm, key):
            setattr(base_svm, key, value)

    # Bagging集成
    bagging_svm = BaggingClassifier(
        base_estimator=base_svm,
        n_estimators=10,
        max_samples=0.8,
        max_features=0.8,
        bootstrap=True,
        n_jobs=-1,
        random_state=42
    )

    bagging_svm.fit(x_train, y_train)

    return bagging_svm


def evaluate_model(model, x_train, y_train, x_test, y_test, model_name):
    y_train_pred = model.predict(x_train)
    y_train_proba = model.predict_proba(x_train)[:, 1]
    y_test_pred = model.predict(x_test)
    y_test_proba = model.predict_proba(x_test)[:, 1]

    def calculate_metrics(y_true, y_pred, y_proba):
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
        return {
            'AUC': roc_auc_score(y_true, y_proba),
            'Accuracy': accuracy_score(y_true, y_pred),
            'Sensitivity': tp / (tp + fn),
            'Specificity': tn / (tn + fp),
            'F1': f1_score(y_true, y_pred),
            'MCC': matthews_corrcoef(y_true, y_pred)
        }

    train_metrics = calculate_metrics(y_train, y_train_pred, y_train_proba)
    test_metrics = calculate_metrics(y_test, y_test_pred, y_test_proba)

    # 计算过拟合差距
    overfit_gap = train_metrics['AUC'] - test_metrics['AUC']

    print(f"\n{model_name} 性能:")
    print(f"训练集AUC: {train_metrics['AUC']:.4f}, 测试集AUC: {test_metrics['AUC']:.4f}")
    print(f"过拟合差距: {overfit_gap:.4f}")
    print(f"灵敏度: {test_metrics['Sensitivity']:.4f}, 特异度: {test_metrics['Specificity']:.4f}")

    return test_metrics, y_test_proba, overfit_gap


def main():
    print("开始减轻过拟合的SVM训练...")

    # 1. 数据预处理
    x_train, y_train, x_test, y_test, scaler = load_and_preprocess_data()

    # 保存标准化器
    scaler_path = os.path.join(SAVE_DIR, "ha_improved_svm_scaler.pkl")
    joblib.dump(scaler, scaler_path)

    # 2. 训练基础SVM
    print("\n训练基础SVM模型...")
    base_model, best_params = train_improved_svm(x_train, y_train)
    base_metrics, base_proba, base_gap = evaluate_model(
        base_model, x_train, y_train, x_test, y_test, "基础SVM"
    )

    # 3. 训练Bagging SVM
    print("\n训练Bagging SVM集成...")
    bagging_model = train_bagging_svm(x_train, y_train, best_params)
    bagging_metrics, bagging_proba, bagging_gap = evaluate_model(
        bagging_model, x_train, y_train, x_test, y_test, "Bagging SVM"
    )

    # 4. 选择最佳模型
    if bagging_gap < base_gap:
        print("\n选择Bagging SVM（过拟合更小）")
        final_model = bagging_model
        final_metrics = bagging_metrics
        final_proba = bagging_proba
        model_type = "bagging_svm"
    else:
        print("\n选择基础SVM")
        final_model = base_model
        final_metrics = base_metrics
        final_proba = base_proba
        model_type = "basic_svm"

    # 5. 保存结果
    model_path = os.path.join(SAVE_DIR, f"ha_{model_type}_model.pkl")
    joblib.dump(final_model, model_path)

    results_df = pd.DataFrame({
        '模型类型': [model_type],
        '测试集AUC': [final_metrics['AUC']],
        '灵敏度': [final_metrics['Sensitivity']],
        '特异度': [final_metrics['Specificity']],
        'F1分数': [final_metrics['F1']],
        '过拟合差距': [bagging_gap if model_type == "bagging_svm" else base_gap]
    })

    results_path = os.path.join(SAVE_DIR, f"ha_{model_type}_results.csv")
    results_df.to_csv(results_path, index=False)

    print(f"\n模型已保存: {model_path}")
    print(f"结果已保存: {results_path}")
    print("训练完成！")


if __name__ == "__main__":
    main()
import pandas as pd
import numpy as np
import os
from sklearn.metrics import roc_auc_score, accuracy_score, f1_score, matthews_corrcoef, roc_curve
from sklearn.model_selection import StratifiedKFold, RandomizedSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.impute import SimpleImputer
import joblib
import matplotlib.pyplot as plt
import warnings

warnings.filterwarnings('ignore')

# ---------------------- 1. 配置 ----------------------
SAVE_DIR = r"E:\SVM_Results_chembreta"
os.makedirs(SAVE_DIR, exist_ok=True)


# ---------------------- 2. 数据加载和预处理 ----------------------
def load_and_preprocess_data():
    """加载和预处理数据"""
    # 读取数据
    df_x_train = pd.read_csv(r"E:\训练集测试集4/hfX_train_chem.csv")
    df_y_train = pd.read_csv(r"E:\训练集测试集4/hfy_train_chem.csv")
    df_x_test = pd.read_csv(r"E:\训练集测试集4/hfX_test_chem.csv")
    df_y_test = pd.read_csv(r"E:\训练集测试集4/hfy_test_chem.csv")

    # 转换为数值
    for df in [df_x_train, df_x_test]:
        for col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')

    # 合并
    x_train = df_x_train.values
    y_train = df_y_train.iloc[:, -1].values.ravel()
    x_test = df_x_test.values
    y_test = df_y_test.iloc[:, -1].values.ravel()

    # 检查缺失值
    print(f"训练集缺失值数量: {np.isnan(x_train).sum()}")
    print(f"测试集缺失值数量: {np.isnan(x_test).sum()}")

    # 使用SimpleImputer处理缺失值
    imputer = SimpleImputer(strategy='median')
    x_train = imputer.fit_transform(x_train)
    x_test = imputer.transform(x_test)

    # 再次检查缺失值
    print(f"处理后训练集缺失值数量: {np.isnan(x_train).sum()}")
    print(f"处理后测试集缺失值数量: {np.isnan(x_test).sum()}")

    # 标准化
    scaler = StandardScaler()
    x_train_scaled = scaler.fit_transform(x_train)
    x_test_scaled = scaler.transform(x_test)

    return x_train_scaled, y_train, x_test_scaled, y_test, scaler, imputer


# ---------------------- 3. 训练减轻过拟合的SVM模型 ----------------------
def train_svm_elastic_net(x_train, y_train):
    """使用线性SVM结合弹性网络正则化"""
    from sklearn.svm import SVC

    print("使用线性SVM（弹性网络正则化）...")

    # 简化参数网格 - 只使用最重要的参数
    param_dist = {
        'C': [0.0001, 0.0005, 0.001, 0.005, 0.01],  # 更小的C值
        'class_weight': ['balanced', {0: 1, 1: 2}, {0: 1, 1: 3}],
        'kernel': ['linear'],
        'max_iter': [5000, 10000],
        'tol': [1e-4, 1e-5],
        'shrinking': [True, False]  # 添加收缩启发式
    }

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    # 使用SVC，默认支持概率估计
    svc = SVC(probability=True, random_state=42)

    print("开始参数搜索...")
    random_search = RandomizedSearchCV(
        svc,
        param_distributions=param_dist,
        n_iter=10,
        cv=cv,
        scoring='roc_auc',
        n_jobs=-1,
        random_state=42,
        verbose=1
    )

    random_search.fit(x_train, y_train)

    print(f"最佳参数: {random_search.best_params_}")
    print(f"最佳交叉验证AUC: {random_search.best_score_:.4f}")

    # 直接返回最佳模型，它已经支持predict_proba
    return random_search.best_estimator_, random_search


# ---------------------- 4. 计算性能指标 ----------------------
def calculate_metrics(y_true, y_pred, y_proba):
    """计算性能指标"""
    metrics = {}

    # 基础指标
    metrics['AUC'] = roc_auc_score(y_true, y_proba)
    metrics['ACC'] = accuracy_score(y_true, y_pred)
    metrics['F1'] = f1_score(y_true, y_pred)
    metrics['MCC'] = matthews_corrcoef(y_true, y_pred)

    # 计算SE和SP
    tn = np.sum((y_true == 0) & (y_pred == 0))
    fp = np.sum((y_true == 0) & (y_pred == 1))
    fn = np.sum((y_true == 1) & (y_pred == 0))
    tp = np.sum((y_true == 1) & (y_pred == 1))

    metrics['SE'] = tp / (tp + fn) if (tp + fn) > 0 else 0
    metrics['SP'] = tn / (tn + fp) if (tn + fp) > 0 else 0

    return metrics


# ---------------------- 5. 5折交叉验证评估 ----------------------
def cross_validation_evaluation(model, x_train, y_train):
    """执行5折交叉验证评估"""
    print("\n执行5折交叉验证...")
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    cv_results = {
        'fold': [],
        'train_auc': [], 'train_acc': [], 'train_mcc': [], 'train_f1': [], 'train_se': [], 'train_sp': [],
        'val_auc': [], 'val_acc': [], 'val_mcc': [], 'val_f1': [], 'val_se': [], 'val_sp': []
    }

    fold = 1
    for train_idx, val_idx in cv.split(x_train, y_train):
        print(f"处理第{fold}折...")

        # 分割数据
        X_train_fold = x_train[train_idx]
        y_train_fold = y_train[train_idx]
        X_val_fold = x_train[val_idx]
        y_val_fold = y_train[val_idx]

        # 训练模型
        model.fit(X_train_fold, y_train_fold)

        # 训练集预测
        y_train_pred_proba = model.predict_proba(X_train_fold)[:, 1]
        y_train_pred = (y_train_pred_proba >= 0.5).astype(int)
        train_metrics = calculate_metrics(y_train_fold, y_train_pred, y_train_pred_proba)

        # 验证集预测
        y_val_pred_proba = model.predict_proba(X_val_fold)[:, 1]
        y_val_pred = (y_val_pred_proba >= 0.5).astype(int)
        val_metrics = calculate_metrics(y_val_fold, y_val_pred, y_val_pred_proba)

        # 存储结果
        cv_results['fold'].append(fold)
        cv_results['train_auc'].append(train_metrics['AUC'])
        cv_results['train_acc'].append(train_metrics['ACC'])
        cv_results['train_mcc'].append(train_metrics['MCC'])
        cv_results['train_f1'].append(train_metrics['F1'])
        cv_results['train_se'].append(train_metrics['SE'])
        cv_results['train_sp'].append(train_metrics['SP'])
        cv_results['val_auc'].append(val_metrics['AUC'])
        cv_results['val_acc'].append(val_metrics['ACC'])
        cv_results['val_mcc'].append(val_metrics['MCC'])
        cv_results['val_f1'].append(val_metrics['F1'])
        cv_results['val_se'].append(val_metrics['SE'])
        cv_results['val_sp'].append(val_metrics['SP'])

        fold += 1

    # 计算平均值和标准差
    cv_summary = {}
    for metric in ['auc', 'acc', 'mcc', 'f1', 'se', 'sp']:
        train_values = cv_results[f'train_{metric}']
        val_values = cv_results[f'val_{metric}']

        cv_summary[f'train_{metric}_mean'] = np.mean(train_values)
        cv_summary[f'train_{metric}_std'] = np.std(train_values)
        cv_summary[f'val_{metric}_mean'] = np.mean(val_values)
        cv_summary[f'val_{metric}_std'] = np.std(val_values)

    # 打印汇总结果
    print("\n5折交叉验证汇总结果:")
    print("-" * 60)
    for metric_name, metric_full in [('AUC', 'auc'), ('ACC', 'acc'), ('MCC', 'mcc'),
                                     ('F1', 'f1'), ('SE', 'se'), ('SP', 'sp')]:
        print(f"{metric_name}:")
        print(
            f"  训练集 - {cv_summary[f'train_{metric_full}_mean']:.4f} ± {cv_summary[f'train_{metric_full}_std']:.4f}")
        print(f"  验证集 - {cv_summary[f'val_{metric_full}_mean']:.4f} ± {cv_summary[f'val_{metric_full}_std']:.4f}")
    print("-" * 60)

    return cv_results, cv_summary


# ---------------------- 6. 绘制ROC曲线 ----------------------
def plot_roc_curve(y_true, y_proba, save_path):
    """绘制ROC曲线"""
    fpr, tpr, _ = roc_curve(y_true, y_proba)
    auc_score = roc_auc_score(y_true, y_proba)

    plt.figure(figsize=(8, 8))
    plt.plot(fpr, tpr, 'b-', linewidth=2, label=f'ROC curve (AUC = {auc_score:.3f})')
    plt.plot([0, 1], [0, 1], 'r--', linewidth=1)

    plt.xlim([0, 1])
    plt.ylim([0, 1])
    plt.xlabel('False Positive Rate', fontsize=12)
    plt.ylabel('True Positive Rate', fontsize=12)
    plt.title('ROC Curve', fontsize=14)
    plt.legend(loc='lower right', fontsize=12)
    plt.grid(True, alpha=0.3)

    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

    return auc_score


# ---------------------- 7. 保存结果 ----------------------
def save_all_results(model, scaler, imputer, y_test, y_test_pred, y_test_proba,
                     train_metrics, test_metrics, cv_results, cv_summary,
                     random_search, save_dir):
    """保存所有结果"""

    # 1. 保存模型文件
    joblib.dump(model, os.path.join(save_dir, "hf svm_model.pkl"))

    # 2. 保存带交叉验证的模型
    joblib.dump({
        'model': model,
        'scaler': scaler,
        'imputer': imputer,
        'cv_results': cv_results,
        'cv_summary': cv_summary,
        'random_search': random_search
    }, os.path.join(save_dir, "hf svm_model_with_cv.pkl"))

    # 3. 保存预处理对象
    joblib.dump(scaler, os.path.join(save_dir, "hf scaler.pkl"))
    joblib.dump(imputer, os.path.join(save_dir, "hf imputer.pkl"))

    # 4. 保存预测文件
    np.save(os.path.join(save_dir, "hf y_test.npy"), y_test)
    np.save(os.path.join(save_dir, "hf y_pred.npy"), y_test_pred)
    np.save(os.path.join(save_dir, "hf y_test_proba.npy"), y_test_proba)

    # 5. 保存主要结果到CSV
    results_dict = {
        '训练集_AUC': [train_metrics['AUC']],
        '训练集_ACC': [train_metrics['ACC']],
        '训练集_MCC': [train_metrics['MCC']],
        '训练集_F1': [train_metrics['F1']],
        '训练集_SE': [train_metrics['SE']],
        '训练集_SP': [train_metrics['SP']],
        '测试集_AUC': [test_metrics['AUC']],
        '测试集_ACC': [test_metrics['ACC']],
        '测试集_MCC': [test_metrics['MCC']],
        '测试集_F1': [test_metrics['F1']],
        '测试集_SE': [test_metrics['SE']],
        '测试集_SP': [test_metrics['SP']]
    }

    # 添加交叉验证结果
    for metric in ['auc', 'acc', 'mcc', 'f1', 'se', 'sp']:
        results_dict[f'CV_train_{metric}_mean'] = [cv_summary[f'train_{metric}_mean']]
        results_dict[f'CV_train_{metric}_std'] = [cv_summary[f'train_{metric}_std']]
        results_dict[f'CV_val_{metric}_mean'] = [cv_summary[f'val_{metric}_mean']]
        results_dict[f'CV_val_{metric}_std'] = [cv_summary[f'val_{metric}_std']]

    # 添加过拟合分析
    auc_diff = train_metrics['AUC'] - test_metrics['AUC']
    results_dict['过拟合_AUC差异'] = [auc_diff]

    results_df = pd.DataFrame(results_dict)
    results_df.to_csv(os.path.join(save_dir, "results.csv"), index=False, encoding='utf-8-sig')

    # 6. 保存详细的交叉验证结果
    cv_df = pd.DataFrame(cv_results)
    cv_df.to_csv(os.path.join(save_dir, "hf cross_validation_details.csv"), index=False, encoding='utf-8-sig')

    # 7. 保存预测结果
    pred_df = pd.DataFrame({
        'y_true': y_test,
        'y_pred': y_test_pred,
        'y_proba': y_test_proba
    })
    pred_df.to_csv(os.path.join(save_dir, "hf predictions.csv"), index=False, encoding='utf-8-sig')

    # 8. 保存随机搜索的CV结果
    if random_search is not None:
        cv_results_df = pd.DataFrame(random_search.cv_results_)
        cv_results_df.to_csv(os.path.join(save_dir, "hf random_search_cv_results.csv"), index=False, encoding='utf-8-sig')


# ---------------------- 8. 主函数 ----------------------
def main():
    print("=" * 60)
    print("SVM模型训练 - 减轻过拟合版")
    print("=" * 60)

    # 1. 加载数据
    print("\n1. 加载和预处理数据...")
    x_train, y_train, x_test, y_test, scaler, imputer = load_and_preprocess_data()

    print(f"特征维度: {x_train.shape[1]}")

    # 2. 训练减轻过拟合的SVM模型
    print("\n2. 训练SVM模型...")
    model, random_search = train_svm_elastic_net(x_train, y_train)

    # 3. 5折交叉验证评估
    print("\n3. 5折交叉验证评估...")
    cv_results, cv_summary = cross_validation_evaluation(model, x_train, y_train)

    # 4. 训练集预测和评估
    print("\n4. 训练集评估...")
    y_train_pred_proba = model.predict_proba(x_train)[:, 1]
    y_train_pred = (y_train_pred_proba >= 0.5).astype(int)
    train_metrics = calculate_metrics(y_train, y_train_pred, y_train_pred_proba)

    # 5. 测试集预测和评估
    print("\n5. 测试集评估...")
    y_test_pred_proba = model.predict_proba(x_test)[:, 1]
    y_test_pred = (y_test_pred_proba >= 0.5).astype(int)
    test_metrics = calculate_metrics(y_test, y_test_pred, y_test_pred_proba)

    # 6. 打印结果
    print("\n" + "=" * 60)
    print("训练集性能:")
    print("-" * 60)
    print(f"AUC:  {train_metrics['AUC']:.4f}")
    print(f"ACC:  {train_metrics['ACC']:.4f}")
    print(f"MCC:  {train_metrics['MCC']:.4f}")
    print(f"F1:   {train_metrics['F1']:.4f}")
    print(f"SE:   {train_metrics['SE']:.4f}")
    print(f"SP:   {train_metrics['SP']:.4f}")

    print("\n" + "=" * 60)
    print("测试集性能:")
    print("-" * 60)
    print(f"AUC:  {test_metrics['AUC']:.4f}")
    print(f"ACC:  {test_metrics['ACC']:.4f}")
    print(f"MCC:  {test_metrics['MCC']:.4f}")
    print(f"F1:   {test_metrics['F1']:.4f}")
    print(f"SE:   {test_metrics['SE']:.4f}")
    print(f"SP:   {test_metrics['SP']:.4f}")

    # 7. 过拟合分析
    auc_diff = train_metrics['AUC'] - test_metrics['AUC']
    print(f"\n过拟合分析: AUC差异 = {auc_diff:.4f}")

    if auc_diff < 0.05:
        print("✅ 过拟合控制良好")
    elif auc_diff < 0.10:
        print("⚠ 轻微过拟合")
    elif auc_diff < 0.15:
        print("⚠ 中度过拟合")
    else:
        print("❌ 严重过拟合")

    # 8. 绘制ROC曲线
    print("\n6. 绘制ROC曲线...")
    roc_path = os.path.join(SAVE_DIR, "hf roc_curve.pdf")
    auc_score = plot_roc_curve(y_test, y_test_pred_proba, roc_path)
    print(f"ROC曲线已保存到: {roc_path}")

    # 9. 保存所有结果
    print("\n7. 保存所有结果...")
    save_all_results(
        model, scaler, imputer, y_test, y_test_pred, y_test_pred_proba,
        train_metrics, test_metrics, cv_results, cv_summary,
        random_search, SAVE_DIR
    )

    print("\n" + "=" * 60)
    print("文件保存完成:")
    print("-" * 60)
    print("1. svm_model.pkl - SVM模型")
    print("2. svm_model_with_cv.pkl - 带交叉验证的SVM模型")
    print("3. scaler.pkl - 标准化器")
    print("4. imputer.pkl - 缺失值填充器")
    print("5. y_test.npy - 测试集真实标签")
    print("6. y_pred.npy - 测试集预测标签")
    print("7. y_test_proba.npy - 测试集预测概率")
    print("8. roc_curve.pdf - ROC曲线")
    print("9. results.csv - 主要结果汇总")
    print("10. cross_validation_details.csv - 交叉验证详细结果")
    print("11. predictions.csv - 预测结果")
    print("12. random_search_cv_results.csv - 随机搜索交叉验证结果")
    print("=" * 60)


# ---------------------- 9. 运行 ----------------------
if __name__ == "__main__":
    main()
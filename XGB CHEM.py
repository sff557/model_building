import pandas as pd
import numpy as np
import os
from sklearn.metrics import roc_auc_score, accuracy_score, f1_score, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV
import joblib
import warnings

warnings.filterwarnings('ignore')


def minimal_svm_solution():
    """
    最小过拟合风险的SVM解决方案
    """
    print("最小过拟合SVM解决方案")
    print("=" * 60)

    # 1. 加载数据
    df_x_train = pd.read_csv(r"E:\训练集测试集\haX_train_chemberta.csv", na_values=["?", "NA", " ", ""])
    df_y_train = pd.read_csv(r"E:\训练集测试集\hay_train_chemberta.csv", na_values=["?", "NA", " ", ""])
    df_x_test = pd.read_csv(r"E:\训练集测试集\haX_test_chemberta.csv", na_values=["?", "NA", " ", ""])
    df_y_test = pd.read_csv(r"E:\训练集测试集\hay_test_chemberta.csv", na_values=["?", "NA", " ", ""])

    # 转换为数值
    for df in [df_x_train, df_x_test]:
        for col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')

    x_train = df_x_train.values
    y_train = df_y_train.iloc[:, -1].values.ravel()
    x_test = df_x_test.values
    y_test = df_y_test.iloc[:, -1].values.ravel()

    print(f"原始数据: 训练集={x_train.shape}, 测试集={x_test.shape}")

    # 2. 简单预处理
    imputer = SimpleImputer(strategy='median')
    x_train = imputer.fit_transform(x_train)
    x_test = imputer.transform(x_test)

    scaler = StandardScaler()
    x_train = scaler.fit_transform(x_train)
    x_test = scaler.transform(x_test)

    # 3. 使用非常简单的线性SVM（强制正则化）
    print("\n训练非常正则化的线性SVM...")

    # 尝试不同的C值（正则化强度）
    best_model = None
    best_score = 0
    best_c = 0.1

    # 在训练集上划分验证集来选择合适的C
    x_train_sub, x_val, y_train_sub, y_val = train_test_split(
        x_train, y_train, test_size=0.2, random_state=42, stratify=y_train
    )

    for C in [0.001, 0.01, 0.1, 0.5, 1.0, 2.0]:
        model = LinearSVC(
            C=C,
            dual=False,
            max_iter=100000,
            tol=1e-4,
            random_state=42,
            class_weight='balanced'  # 自动平衡类别
        )

        model.fit(x_train_sub, y_train_sub)

        # 校准以获得概率
        calibrated_model = CalibratedClassifierCV(model, cv='prefit')
        calibrated_model.fit(x_train_sub, y_train_sub)

        y_val_proba = calibrated_model.predict_proba(x_val)[:, 1]
        val_auc = roc_auc_score(y_val, y_val_proba)

        print(f"C={C}: 验证集AUC = {val_auc:.4f}")

        if val_auc > best_score:
            best_score = val_auc
            best_c = C
            best_model = calibrated_model

    print(f"\n选择C={best_c}, 验证集AUC={best_score:.4f}")

    # 4. 使用最佳C在整个训练集上训练
    final_model = LinearSVC(
        C=best_c,
        dual=False,
        max_iter=100000,
        tol=1e-4,
        random_state=42,
        class_weight='balanced'
    )

    final_model.fit(x_train, y_train)

    # 校准
    final_calibrated = CalibratedClassifierCV(final_model, cv='prefit')
    final_calibrated.fit(x_train, y_train)

    # 5. 评估
    y_train_proba = final_calibrated.predict_proba(x_train)[:, 1]
    y_test_proba = final_calibrated.predict_proba(x_test)[:, 1]

    train_auc = roc_auc_score(y_train, y_train_proba)
    test_auc = roc_auc_score(y_test, y_test_proba)

    print(f"\n最终性能:")
    print(f"训练集AUC: {train_auc:.4f}")
    print(f"测试集AUC: {test_auc:.4f}")
    print(f"AUC差距: {train_auc - test_auc:.4f}")

    if train_auc - test_auc > 0.15:
        print("警告: 仍然存在过拟合，建议进一步减小C值")

    # 6. 保存
    SAVE_DIR = r"E:\SVM_Minimal_Overfit"
    os.makedirs(SAVE_DIR, exist_ok=True)

    joblib.dump(final_calibrated, os.path.join(SAVE_DIR, "minimal_svm_model.pkl"))
    joblib.dump(scaler, os.path.join(SAVE_DIR, "minimal_svm_scaler.pkl"))

    # 保存详细结果
    results = {
        'C_value': best_c,
        'train_auc': train_auc,
        'test_auc': test_auc,
        'auc_gap': train_auc - test_auc,
        'train_accuracy': accuracy_score(y_train, final_calibrated.predict(x_train)),
        'test_accuracy': accuracy_score(y_test, final_calibrated.predict(x_test)),
    }

    pd.DataFrame([results]).to_csv(os.path.join(SAVE_DIR, "minimal_svm_results.csv"), index=False)

    print(f"\n模型已保存到: {SAVE_DIR}")

    return final_calibrated, scaler, results


if __name__ == "__main__":
    minimal_svm_solution()
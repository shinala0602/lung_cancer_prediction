"""
predict.py
----------
학습된 모델로 테스트셋 예측 및 결과 저장
실행: python src/predict.py
"""

import os
import pickle
import numpy as np
import pandas as pd

from preprocess import load_expression, load_survival

# ─── 경로 설정 ────────────────────────────────────────────────
DATA_DIR  = "data/test"
MODEL_DIR = "models"
RESULT_DIR = "results"
os.makedirs(RESULT_DIR, exist_ok=True)


def load_model(task_name: str) -> dict:
    path = f"{MODEL_DIR}/{task_name}.pkl"
    with open(path, "rb") as f:
        return pickle.load(f)


def predict_from_bundle(bundle: dict, X_test: pd.DataFrame) -> tuple:
    """
    저장된 bundle(model + imputer + scaler + valid_mask + selector)로 예측
    """
    features   = bundle["features"]
    valid_mask = bundle.get("valid_mask")  # 없으면 None (구버전 호환)

    # 공통 feature만 선택 (테스트에 없는 gene은 NaN으로 채움)
    X = X_test.reindex(columns=features)

    X_imp = bundle["imputer"].transform(X)
    X_scl = bundle["scaler"].transform(X_imp)

    # 상수 feature 제거 (train과 동일하게 적용)
    if valid_mask is not None:
        X_scl = X_scl[:, valid_mask]

    X_sel  = bundle["selector"].transform(X_scl)
    preds  = bundle["model"].predict(X_sel)
    probas = bundle["model"].predict_proba(X_sel)[:, 1]
    return preds, probas


# ─── 테스트 데이터 로드 ───────────────────────────────────────
print("=== 테스트 데이터 로드 중 ===")

luad_prot_tumor_test = load_expression(f"{DATA_DIR}/LUAD_testset_protein_expression_tumor.tsv")
luad_prot_nat_test   = load_expression(f"{DATA_DIR}/LUAD_testset_protein_expression_nat.tsv")
lscc_prot_tumor_test = load_expression(f"{DATA_DIR}/LSCC_testset_protein_expression_tumor.tsv")
lscc_prot_nat_test   = load_expression(f"{DATA_DIR}/LSCC_testset_protein_expression_nat.tsv")

luad_rna_tumor_test  = load_expression(f"{DATA_DIR}/LUAD_testset_rna_expression_tumor.tsv")
lscc_rna_tumor_test  = load_expression(f"{DATA_DIR}/LSCC_testset_rna_expression_tumor.tsv")

# ═══════════════════════════════════════════════════════
# Task 1: Tumor vs Normal
# ═══════════════════════════════════════════════════════
print("\n[Task 1] Tumor vs Normal 예측")
bundle1 = load_model("task1_tumor_vs_normal")

tumor_test = pd.concat([luad_prot_tumor_test, lscc_prot_tumor_test])
nat_test   = pd.concat([luad_prot_nat_test,   lscc_prot_nat_test])
X_test1    = pd.concat([tumor_test, nat_test])
true_label = [1]*len(tumor_test) + [0]*len(nat_test)

preds1, probas1 = predict_from_bundle(bundle1, X_test1)

result1 = pd.DataFrame({
    "case_id":         X_test1.index,
    "true_label":      true_label,
    "predicted_label": preds1,
    "prob_tumor":      probas1,
    "predicted_class": ["Tumor" if p == 1 else "Normal" for p in preds1]
})
result1.to_csv(f"{RESULT_DIR}/task1_predictions.csv", index=False)
print(f"  저장 완료: results/task1_predictions.csv")
print(f"  예측 분포: {result1['predicted_class'].value_counts().to_dict()}")

# ═══════════════════════════════════════════════════════
# Task 2: LUAD vs LSCC
# ═══════════════════════════════════════════════════════
print("\n[Task 2] LUAD vs LSCC 예측")
bundle2 = load_model("task2_luad_vs_lscc")

luad_t2_test = pd.concat([luad_prot_tumor_test, luad_rna_tumor_test], axis=1)
lscc_t2_test = pd.concat([lscc_prot_tumor_test, lscc_rna_tumor_test], axis=1)
X_test2 = pd.concat([luad_t2_test, lscc_t2_test])
true2   = [0]*len(luad_t2_test) + [1]*len(lscc_t2_test)

preds2, probas2 = predict_from_bundle(bundle2, X_test2)

result2 = pd.DataFrame({
    "case_id":         X_test2.index,
    "true_label":      true2,
    "predicted_label": preds2,
    "prob_lscc":       probas2,
    "predicted_class": ["LSCC" if p == 1 else "LUAD" for p in preds2]
})
result2.to_csv(f"{RESULT_DIR}/task2_predictions.csv", index=False)
print(f"  저장 완료: results/task2_predictions.csv")
print(f"  예측 분포: {result2['predicted_class'].value_counts().to_dict()}")

# ═══════════════════════════════════════════════════════
# Task 3: Survival
# ═══════════════════════════════════════════════════════
print("\n[Task 3] Survival 예측")
bundle3 = load_model("task3_survival")

X_test3 = pd.concat([luad_prot_tumor_test, lscc_prot_tumor_test])
preds3, probas3 = predict_from_bundle(bundle3, X_test3)

result3 = pd.DataFrame({
    "case_id":         X_test3.index,
    "predicted_label": preds3,
    "prob_death":      probas3,
    "predicted_class": ["Death" if p == 1 else "Survival" for p in preds3]
})
result3.to_csv(f"{RESULT_DIR}/task3_predictions.csv", index=False)
print(f"  저장 완료: results/task3_predictions.csv")
print(f"  예측 분포: {result3['predicted_class'].value_counts().to_dict()}")

print("\n✅ 모든 예측 완료! results/ 폴더를 확인하세요.")
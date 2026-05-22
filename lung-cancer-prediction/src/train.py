"""
train.py
--------
세 가지 Task에 대한 모델 학습 및 저장
실행: python src/train.py
"""

import os
import pickle
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import RepeatedStratifiedKFold, cross_val_score
from sklearn.metrics import roc_auc_score
from sklearn.feature_selection import SelectKBest, f_classif

from preprocess import (
    load_expression, load_survival,
    build_task1_data, build_task2_data, build_task3_data,
    impute_and_scale, get_common_genes
)

# ─── 경로 설정 ────────────────────────────────────────────────
DATA_DIR = "data/train"
MODEL_DIR = "models"
os.makedirs(MODEL_DIR, exist_ok=True)

# ─── 데이터 로드 ──────────────────────────────────────────────
print("=== 데이터 로드 중 ===")

luad_prot_tumor = load_expression(f"{DATA_DIR}/LUAD_trainingset_protein_expression_tumor.tsv")
luad_prot_nat   = load_expression(f"{DATA_DIR}/LUAD_trainingset_protein_expression_nat.tsv")
lscc_prot_tumor = load_expression(f"{DATA_DIR}/LSCC_trainingset_protein_expression_tumor.tsv")
lscc_prot_nat   = load_expression(f"{DATA_DIR}/LSCC_trainingset_protein_expression_nat.tsv")

luad_rna_tumor  = load_expression(f"{DATA_DIR}/LUAD_trainingset_rna_expression_tumor.tsv")
luad_rna_nat    = load_expression(f"{DATA_DIR}/LUAD_trainingset_rna_expression_nat.tsv")
lscc_rna_tumor  = load_expression(f"{DATA_DIR}/LSCC_trainingset_rna_expression_tumor.tsv")
lscc_rna_nat    = load_expression(f"{DATA_DIR}/LSCC_trainingset_rna_expression_nat.tsv")

luad_survival   = load_survival(f"{DATA_DIR}/LUAD_trainingset_overall_survival.tsv")
lscc_survival   = load_survival(f"{DATA_DIR}/LSCC_trainingset_overall_survival.tsv")

print(f"LUAD tumor: {luad_prot_tumor.shape}, LSCC tumor: {lscc_prot_tumor.shape}")

# ─── 공통 gene 선택 ──────────────────────────────────────────
prot_genes = get_common_genes([luad_prot_tumor, luad_prot_nat,
                                lscc_prot_tumor, lscc_prot_nat])
rna_genes  = get_common_genes([luad_rna_tumor,  luad_rna_nat,
                                lscc_rna_tumor,  lscc_rna_nat])
print(f"공통 Protein genes: {len(prot_genes)}, RNA genes: {len(rna_genes)}")


def train_and_evaluate(X, y, task_name, model_name="rf", n_features=200):
    """
    학습 + Repeated 5-fold CV (10회 반복) 평가 + 전체 데이터로 최종 모델 저장
    - 샘플 수가 적기 때문에 train/val 분리 없이 전체 데이터로 학습
    - RepeatedStratifiedKFold로 CV 안정성 향상 (5-fold × 10회 = 50회 평균)
    """
    print(f"\n{'='*50}")
    print(f"[{task_name}] 학습 시작 | samples={len(y)}, pos_rate={y.mean():.2f}")

    # Impute & Scale
    X_arr, imputer, scaler = impute_and_scale(X)

    # 분산 0인 컬럼 제거 (constant feature → SelectKBest 경고 방지)
    std = X_arr.std(axis=0)
    valid_mask = std > 0
    X_arr = X_arr[:, valid_mask]
    print(f"  상수 feature 제거: {(~valid_mask).sum()}개 제거 → {valid_mask.sum()}개 사용")

    # Feature selection
    k = min(n_features, X_arr.shape[1])
    selector = SelectKBest(f_classif, k=k)
    X_sel = selector.fit_transform(X_arr, y)

    # 모델 선택
    if model_name == "rf":
        model = RandomForestClassifier(
            n_estimators=300,
            max_depth=None,
            class_weight="balanced",
            random_state=42,
            n_jobs=-1
        )
    else:
        model = LogisticRegression(
            C=0.1,
            class_weight="balanced",
            max_iter=1000,
            random_state=42
        )

    # Repeated 5-fold CV (5-fold × 10회 반복 = 50회 평균)
    # 샘플 수가 적을 때 단순 5-fold보다 안정적인 성능 추정 가능
    cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=10, random_state=42)
    aucs = cross_val_score(model, X_sel, y, cv=cv, scoring="roc_auc")
    accs = cross_val_score(model, X_sel, y, cv=cv, scoring="accuracy")
    print(f"  CV AUC:      {aucs.mean():.4f} ± {aucs.std():.4f}  (5-fold × 10회)")
    print(f"  CV Accuracy: {accs.mean():.4f} ± {accs.std():.4f}  (5-fold × 10회)")

    # 전체 데이터로 최종 학습
    model.fit(X_sel, y)

    # 저장 (valid_mask도 함께 저장 → predict 시 동일하게 적용)
    save_path = f"{MODEL_DIR}/{task_name}.pkl"
    with open(save_path, "wb") as f:
        pickle.dump({
            "model":      model,
            "imputer":    imputer,
            "scaler":     scaler,
            "valid_mask": valid_mask,   # 상수 feature 제거 마스크
            "selector":   selector,
            "features":   list(X.columns),
            "task":       task_name
        }, f)
    print(f"  모델 저장: {save_path}")
    return model


# ═══════════════════════════════════════════════════════
# Task 1: Tumor vs Normal (Protein)
# ═══════════════════════════════════════════════════════
X1, y1 = build_task1_data(
    luad_prot_tumor[prot_genes], luad_prot_nat[prot_genes],
    lscc_prot_tumor[prot_genes], lscc_prot_nat[prot_genes],
    genes=prot_genes
)
train_and_evaluate(X1, y1, task_name="task1_tumor_vs_normal", model_name="rf")

# ═══════════════════════════════════════════════════════
# Task 2: LUAD vs LSCC (Protein + RNA 병합)
# ═══════════════════════════════════════════════════════
luad_t2 = pd.concat([luad_prot_tumor[prot_genes], luad_rna_tumor[rna_genes]], axis=1)
lscc_t2 = pd.concat([lscc_prot_tumor[prot_genes], lscc_rna_tumor[rna_genes]], axis=1)

luad_t2 = luad_t2.dropna(how="all")
lscc_t2 = lscc_t2.dropna(how="all")

X2, y2 = build_task2_data(luad_t2, lscc_t2)
train_and_evaluate(X2, y2, task_name="task2_luad_vs_lscc", model_name="rf")

# ═══════════════════════════════════════════════════════
# Task 3: Survival Prediction (Protein)
# ═══════════════════════════════════════════════════════
X3, y3 = build_task3_data(
    luad_prot_tumor[prot_genes], lscc_prot_tumor[prot_genes],
    luad_survival, lscc_survival,
    genes=prot_genes
)
train_and_evaluate(X3, y3, task_name="task3_survival", model_name="rf", n_features=100)

print("\n✅ 모든 Task 학습 완료!")
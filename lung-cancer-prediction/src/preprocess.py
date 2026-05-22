"""
preprocess.py
-------------
데이터 로드 및 전처리 유틸리티
- TSV 파일 로드 (행=gene, 열=patient → 전치 필요)
- 결측치 처리 (median imputation)
- 정규화 (StandardScaler)
- 공통 gene feature만 선택
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer


def load_expression(filepath: str) -> pd.DataFrame:
    """
    TSV 발현 파일 로드 후 전치 (행=patient, 열=gene)
    """
    df = pd.read_csv(filepath, sep="\t", index_col=0)
    df = df.T  # 전치: 행=patient, 열=gene
    df.index.name = "case_id"
    return df


def load_survival(filepath: str) -> pd.DataFrame:
    """
    Survival TSV 로드
    """
    df = pd.read_csv(filepath, sep="\t")
    df = df.set_index("case_id")
    return df


def merge_protein_rna(protein_df: pd.DataFrame,
                       rna_df: pd.DataFrame,
                       suffix: tuple = ("_prot", "_rna")) -> pd.DataFrame:
    """
    Protein + RNA 발현 데이터 병합 (공통 patient만)
    """
    merged = protein_df.join(rna_df, how="inner", lsuffix=suffix[0], rsuffix=suffix[1])
    return merged


def get_common_genes(dfs: list[pd.DataFrame]) -> list[str]:
    """
    여러 DataFrame의 공통 컬럼(gene) 반환
    """
    common = set(dfs[0].columns)
    for df in dfs[1:]:
        common &= set(df.columns)
    return sorted(list(common))


def impute_and_scale(train_df: pd.DataFrame,
                     test_df: pd.DataFrame = None,
                     strategy: str = "median"):
    """
    Median imputation + StandardScaler
    train 기준으로 fit, test에 transform
    """
    imputer = SimpleImputer(strategy=strategy)
    scaler = StandardScaler()

    X_train = imputer.fit_transform(train_df)
    X_train = scaler.fit_transform(X_train)

    if test_df is not None:
        X_test = imputer.transform(test_df)
        X_test = scaler.transform(X_test)
        return X_train, X_test, imputer, scaler

    return X_train, imputer, scaler


def build_task1_data(luad_tumor, luad_nat, lscc_tumor, lscc_nat, genes=None):
    """
    Task 1: Tumor vs Normal
    - Tumor label = 1, NAT(Normal) label = 0
    """
    tumor = pd.concat([luad_tumor, lscc_tumor])
    normal = pd.concat([luad_nat, lscc_nat])

    if genes:
        tumor = tumor[genes]
        normal = normal[genes]

    X = pd.concat([tumor, normal])
    y = pd.Series(
        [1] * len(tumor) + [0] * len(normal),
        index=X.index,
        name="label"
    )
    return X, y


def build_task2_data(luad_tumor, lscc_tumor, genes=None):
    """
    Task 2: LUAD vs LSCC
    - LUAD = 0, LSCC = 1
    """
    if genes:
        luad_tumor = luad_tumor[genes]
        lscc_tumor = lscc_tumor[genes]

    X = pd.concat([luad_tumor, lscc_tumor])
    y = pd.Series(
        [0] * len(luad_tumor) + [1] * len(lscc_tumor),
        index=X.index,
        name="label"
    )
    return X, y


def build_task3_data(luad_tumor, lscc_tumor,
                     luad_survival, lscc_survival, genes=None):
    """
    Task 3: Survival Prediction (OS_event: 1=Death, 0=Censored)
    """
    tumor = pd.concat([luad_tumor, lscc_tumor])
    survival = pd.concat([luad_survival, lscc_survival])

    # 공통 patient만
    common_idx = tumor.index.intersection(survival.index)
    tumor = tumor.loc[common_idx]
    survival = survival.loc[common_idx]

    if genes:
        tumor = tumor[genes]

    X = tumor
    y = survival["OS_event"].astype(int)
    return X, y

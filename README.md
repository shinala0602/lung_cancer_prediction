# Lung Cancer Multi-Task Prediction

---

## 프로젝트 구조

```
lung-cancer-prediction/
├── README.md
├── requirements.txt
├── data/
│   ├── train/        # 학습 데이터 (TSV)
│   └── test/         # 테스트 데이터 (TSV) ← 여기에 test 데이터를 넣어주세요
├── src/
│   ├── preprocess.py # 데이터 로드 및 전처리
│   ├── train.py      # 모델 학습 (참고용)
│   └── predict.py    # 테스트셋 예측 ← 실행 파일
├── models/           # 학습된 모델 (.pkl)
└── results/          # 예측 결과 CSV (자동 생성)
```

---

## 테스트 데이터 준비

`data/test/` 폴더에 아래 파일을 넣어주세요.

```
data/test/
├── LUAD_testset_protein_expression_tumor.tsv
├── LUAD_testset_protein_expression_nat.tsv
├── LUAD_testset_rna_expression_tumor.tsv
├── LSCC_testset_protein_expression_tumor.tsv
├── LSCC_testset_protein_expression_nat.tsv
└── LSCC_testset_rna_expression_tumor.tsv
```

---

## 예측 실행

```bash
python src/predict.py
```

예측이 완료되면 `results/` 폴더에 아래 파일이 생성됩니다.

```
results/
├── task1_predictions.csv   # Tumor / Normal
├── task2_predictions.csv   # LUAD / LSCC
└── task3_predictions.csv   # Death / Survival
```

---

## 결과 파일 형식

**task1_predictions.csv**
| 컬럼 | 설명 |
|------|------|
| case_id | 환자 ID |
| predicted_label | 0 = Normal, 1 = Tumor |
| prob_tumor | Tumor 확률 |
| predicted_class | Tumor / Normal |

**task2_predictions.csv**
| 컬럼 | 설명 |
|------|------|
| case_id | 환자 ID |
| predicted_label | 0 = LUAD, 1 = LSCC |
| prob_lscc | LSCC 확률 |
| predicted_class | LUAD / LSCC |

**task3_predictions.csv**
| 컬럼 | 설명 |
|------|------|
| case_id | 환자 ID |
| predicted_label | 0 = Survival, 1 = Death |
| prob_death | Death 확률 |
| predicted_class | Death / Survival |

---

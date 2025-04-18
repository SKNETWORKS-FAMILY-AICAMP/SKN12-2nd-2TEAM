# 🏷️ SKN12-2nd-2TEAM
## 👨‍💻 프로젝트 : e-commerce 고객 이탈자 예측 

![image](https://github.com/user-attachments/assets/d9136909-8301-4bb8-bdab-88c39b7afe91)




---
## 👀 팀 소개
### 🐰 팀 명 : 당근 그리고 채찍
- e-commerce 시장에서 고객 유입이 당근이라면 고객 이탈은 채찍이라고 생각했습니다. 예비 이탈자를 이미 알고 체계적인 고객 대응으로 채찍을 줄이자는 목표를 가지고 있는 팀입니다.
### 👥 팀 멤버


| ![](https://github.com/user-attachments/assets/cdc6b8cd-155c-46de-99a5-d31673585b36) | ![](https://i.pinimg.com/236x/04/9d/5b/049d5b422b254da9edb8bebe2b2a79c4.jpg) | ![](https://i.pinimg.com/236x/04/9d/5b/049d5b422b254da9edb8bebe2b2a79c4.jpg) | ![](https://i.pinimg.com/236x/04/9d/5b/049d5b422b254da9edb8bebe2b2a79c4.jpg) |
|:--:|:--:|:--:|:--:|
| **김원우** | **김재현** | **남의헌** | **황차해** |

---

## 🛠 기술 스택 개요

### 🧠 AI & 데이터 처리
<img src="https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white">
<img src="https://img.shields.io/badge/Pandas-150458?style=for-the-badge&logo=pandas&logoColor=white">
<img src="https://img.shields.io/badge/Numpy-013243?style=for-the-badge&logo=numpy&logoColor=white">
<img src="https://img.shields.io/badge/Scikit--Learn-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white">

### 🧪 실험 및 개발 환경
<img src="https://img.shields.io/badge/Google%20Colab-F9AB00?style=for-the-badge&logo=googlecolab&logoColor=white">
<img src="https://img.shields.io/badge/Jupyter-F37626?style=for-the-badge&logo=jupyter&logoColor=white">

### 🖼 대시보드 & 프론트엔드
<img src="https://img.shields.io/badge/Streamlit-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white">

### 💾 데이터베이스
<img src="https://img.shields.io/badge/MySQL-4479A1?style=for-the-badge&logo=mysql&logoColor=white">

### 📦 머신러닝 모델
<img src="https://img.shields.io/badge/LightGBM-3C3C3C?style=for-the-badge&logo=lightgbm&logoColor=white">
<img src="https://img.shields.io/badge/XGBoost-EC0000?style=for-the-badge&logo=xgboost&logoColor=white">
<img src="https://img.shields.io/badge/RandomForest-00B050?style=for-the-badge">
<img src="https://img.shields.io/badge/SGDClassifier-006699?style=for-the-badge">

---

## 📊 모델 성능 비교

| 🆔 | 모델명         | Accuracy | F1 Score | ROC AUC | 🔧 하이퍼파라미터 |
|:--:|----------------|:--------:|:--------:|:-------:|------------------|
| 0  | **LGBM**       | 0.9159   | 0.8479   | 0.9807  | {'n_estimators': 500, 'num_leaves': 109, 'min_child_samples': 2, 'learning_rate': 0.0432, 'log_max_bin': 9, 'colsample_bytree': 0.5127, 'reg_alpha': 0.0090, 'reg_lambda': 0.0377} |
| 1  | **RandomForest** | 0.9154 | 0.8456   | 0.9803  | {'n_estimators': 40, 'max_features': 0.8184, 'max_leaves': 2236, 'criterion': 'entropy'} |
| 2  | **XGBoost**     | 0.9073  | 0.8405   | 0.9794  | {'n_estimators': 107, 'max_leaves': 1767, 'min_child_weight': 2.1123, 'learning_rate': 0.1778, 'subsample': 0.9360, 'colsample_bylevel': 0.7357, 'colsample_bytree': 0.8312, 'reg_alpha': 0.0813, 'reg_lambda': 0.1893} |
| 3  | **ExtraTree**   | 0.9146  | 0.8378   | 0.9791  | {'n_estimators': 9, 'max_features': 1.0, 'max_leaves': 4632, 'criterion': 'entropy'} |
| 4  | **SGD**         | 0.8333  | 0.4392   | 0.8466  | {'penalty': 'None', 'alpha': 0.0015, 'l1_ratio': 0.2418, 'epsilon': 0.1, 'learning_rate': 'constant', 'eta0': 0.0053, 'power_t': 0.4250, 'average': False, 'loss': 'modified_huber'} |

---

## 📂 데이터 전처리

### 📦 데이터셋 정보  
- 사용 데이터: `category_tree.csv`, `events.csv`, `item_properties_part1.csv`, `item_properties_part2.csv`  
- 출처: [Kaggle - Retail Rocket Dataset](https://www.kaggle.com/datasets/retailrocket/ecommerce-dataset)

---

### 📄 원본 컬럼 설명

| 컬럼명       | 설명                         | 타입    |
|--------------|------------------------------|---------|
| timestamp    | 이벤트 발생 시각             | int     |
| visitorid    | 사용자 ID                    | int     |
| event        | 이벤트 유형                  | object  |
| itemid       | 상품 ID                      | int     |
| transactionid| 거래 ID (구매시에만 존재)    | float   |
| property     | 속성명                       | object  |
| value        | 속성 값                      | object  |
| categoryid   | 카테고리 ID                  | int     |
| parentid     | 상위 카테고리 ID             | float   |

---

### 🧬 파생 컬럼 설명

| 컬럼명                | 설명                             | 타입    |
|-----------------------|----------------------------------|---------|
| visitorid             | 사용자 ID (유지)                 | int     |
| sessionid             | 세션 수                          | int     |
| item_n                | 구매 상품 수                     | float   |
| cat_n                 | 카테고리 수                      | float   |
| int_n                 | 상호작용 수                      | float   |
| spend                 | 총 지출액                        | float   |
| length_min            | 평균 머문 시간(분)               | float   |
| recency               | 마지막 방문으로부터의 일수       | int     |
| user_age              | 가입 이후 경과 일수              | int     |
| target_class          | 이탈 여부 (1 = 이탈)             | int     |
| session_gap_trend     | 세션 간 간격 변화(표준편차)      | float   |
| activiti_decay_ratio  | 최근 활동 감소율                 | float   |
| engagement_volatility | 참여도 변동성                    | float   |
| session_interval_std  | 세션 간격 표준편차               | float   |
| min_recency_ratio     | 최소 리센시 비율                 | float   |
| repeat_category_ratio | 반복 클릭한 카테고리 비율        | float   |

---

### 🔁 데이터 전처리 흐름도

<p align="center">
  <img src="https://cdn.discordapp.com/attachments/1361611376614707345/1362653548663406682/image.png?ex=68032d82&is=6801dc02&hm=b50626e7f6b16c731831baaf68f5c88e43250a8c3be96cbdaa0f11900512440c&" width="400" height="400">
</p>

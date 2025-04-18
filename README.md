# SKN12-2nd-2TEAM
## 👥 팀 멤버
| ![](https://i.pinimg.com/236x/04/9d/5b/049d5b422b254da9edb8bebe2b2a79c4.jpg) | ![](https://i.pinimg.com/236x/04/9d/5b/049d5b422b254da9edb8bebe2b2a79c4.jpg) | ![](https://i.pinimg.com/236x/04/9d/5b/049d5b422b254da9edb8bebe2b2a79c4.jpg) | ![](https://i.pinimg.com/236x/04/9d/5b/049d5b422b254da9edb8bebe2b2a79c4.jpg) |
|:--:|:--:|:--:|:--:|
| **김원우** | **김재현** | **남의헌** | **황차해** |
----------
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

----------

| Index | Model      | Accuracy   | F1 Score   | ROC AUC   | Best Parameters |
|:-----:|:-----------|:-----------|:-----------|:----------|-----------------|
| 0     | **LGBM**    | 0.9159     | 0.8479     | 0.9807    | {'n_estimators': 500, 'num_leaves': 109, 'min_child_samples': 2, 'learning_rate': 0.0432, 'log_max_bin': 9, 'colsample_bytree': 0.5127, 'reg_alpha': 0.0090, 'reg_lambda': 0.0377} |
| 1     | **RF**      | 0.9154     | 0.8456     | 0.9803    | {'n_estimators': 40, 'max_features': 0.8184, 'max_leaves': 2236, 'criterion': 'entropy'} |
| 2     | **XGBoost** | 0.9073     | 0.8405     | 0.9794    | {'n_estimators': 107, 'max_leaves': 1767, 'min_child_weight': 2.1123, 'learning_rate': 0.1778, 'subsample': 0.9360, 'colsample_bylevel': 0.7357, 'colsample_bytree': 0.8312, 'reg_alpha': 0.0813, 'reg_lambda': 0.1893} |
| 3     | **ExtraTree**| 0.9146    | 0.8378     | 0.9791    | {'n_estimators': 9, 'max_features': 1.0, 'max_leaves': 4632, 'criterion': 'entropy'} |
| 4     | **SGD**     | 0.8333     | 0.4392     | 0.8466    | {'penalty': 'None', 'alpha': 0.0015, 'l1_ratio': 0.2418, 'epsilon': 0.1, 'learning_rate': 'constant', 'eta0': 0.0053, 'power_t': 0.4250, 'average': False, 'loss': 'modified_huber'} |

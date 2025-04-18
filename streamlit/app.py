import streamlit as st
import pandas as pd
import plotly.express as px
import matplotlib.font_manager as fm
import joblib
import numpy as np
import random
import plotly.graph_objects as go
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import xgboost as xgb

# ✅ 한글 폰트 설정
font_dirs = ['/usr/share/fonts/truetype/nanum']
font_files = fm.findSystemFonts(fontpaths=font_dirs)
for f in font_files:
    fm.fontManager.addfont(f)

st.set_page_config(layout="wide")
st.title("🛍️ E-Commerce 고객 이탈 분석 및 예측 대시보드")

# 📦 데이터 로딩
@st.cache_data
def load_data():
    df_overview = pd.read_csv("streamlit_ready_data.csv")
    df_pred = pd.read_csv("predicted_data.csv")
    if '방문자ID' not in df_pred.columns:
        df_pred['방문자ID'] = [f"USER_{i}" for i in range(len(df_pred))]
    model = joblib.load("saved_model.pkl")
    scaler = joblib.load("scaler.pkl")
    return df_overview, df_pred, model, scaler

@st.cache_resource
def load_booster():
    booster = xgb.Booster()
    booster.load_model("xgb_booster_only.json")
    return booster

df, df_pred, model, scaler = load_data()
booster = load_booster()

# 탭 구성
tab1, tab2, tab3 = st.tabs(["📊 이탈 고객 현황", "🔮 예측 결과 분석", "🧪 실시간 예측 시뮬레이션"])

# ========================================================
# 📊 탭 1: 이탈 고객 현황 분석
# ========================================================
with tab1:
    st.header("📊 이탈 고객 현황 분석")

    col1, col2, col3 = st.columns(3)
    col1.metric("👥 전체 고객 수", f"{len(df):,}")
    col2.metric("❌ 이탈 고객 수", f"{df['이탈여부'].sum():,}")
    col3.metric("📉 이탈률", f"{(df['이탈여부'].mean() * 100):.2f}%")

    st.divider()
    st.subheader("📈 이탈 여부 관련 주요 인사이트")

    col_a, col_b, col_c = st.columns(3)

    with col_a:
        st.markdown("**🔋 전체 고객의 활성/비활성 비율**")
        active_ratio = df['활성여부'].value_counts(normalize=True).reset_index()
        active_ratio.columns = ['활성여부', '비율']
        fig = px.pie(active_ratio, names='활성여부', values='비율', color_discrete_sequence=px.colors.qualitative.Set3)
        st.plotly_chart(fig, use_container_width=True, key="churn_gauge_chart231")

    with col_b:
        st.markdown("**📊 이탈 여부별 총지출액 분포**")
        fig = px.box(df, x='이탈여부', y='총지출액', points="outliers", color='이탈여부')
        st.plotly_chart(fig, use_container_width=True, key="churn_gauge_chart4421")

    with col_c:
        st.markdown("**📊 이탈 여부별 세션간격표준편차**")
        fig = px.box(df, x='이탈여부', y='세션간격표준편차', points="outliers", color='이탈여부')
        st.plotly_chart(fig, use_container_width=True, key="churn_gauge_chart1")

# ========================================================
# 🔮 탭 2: 예측 결과 분석
# ========================================================
with tab2:
    st.header("🔮 고객 이탈 예측 결과")

    col1, col2, col3 = st.columns(3)
    col1.metric("예측 대상 수", f"{len(df_pred):,}")
    col2.metric("위험등급 3 고객 수", f"{df_pred[df_pred['이탈위험등급'] == 2].shape[0]:,}")
    col3.metric("평균 이탈 확률", f"{df_pred['예측이탈확률'].mean() * 100:.2f}%")

    st.divider()

    col_a, col_b, col_c = st.columns(3)

    with col_a:
        st.subheader("🎯 Feature Importance")
        importance = booster.get_score(importance_type='weight')
        imp_df = pd.DataFrame(importance.items(), columns=['Feature', 'Importance'])
        imp_df = imp_df.sort_values('Importance', ascending=True)
        fig = px.bar(imp_df, x='Importance', y='Feature', orientation='h')
        st.plotly_chart(fig, use_container_width=True)

    with col_b:
        st.subheader("🚨 위험고객 Top 10 (이탈위험등급=2)")
        top_risk = df_pred[df_pred['이탈위험등급'] == 2].sort_values('예측이탈확률', ascending=False).head(10)
        st.dataframe(top_risk[['예측이탈확률', '총지출액', '세션길이(분)', '참여도변동성']], use_container_width=True)

    with col_c:
        st.subheader("📊 위험등급별 주요 수치 비교")
        compare = df_pred.groupby('이탈위험등급')[['총지출액', '세션길이(분)', '참여도변동성', '세션간격표준편차']].mean().reset_index()
        fig = px.bar(compare.melt(id_vars='이탈위험등급'), x='이탈위험등급', y='value', color='variable', barmode='group')
        st.plotly_chart(fig, use_container_width=True, key="churn_gauge_chart2")

    st.divider()
    col_m1, col_m2, col_m3 = st.columns(3)

    with col_m1:
        st.subheader("🤖 모델 성능 비교")
        model_compare = pd.DataFrame({
            'Model': ['lgbm', 'rf', 'xgboost', 'extra_tree', 'sgd'],
            'ROC AUC': [0.9159, 0.9154, 0.9073, 0.9146, 0.8333],
            'F1 (Class 0)': [0.8479, 0.8456, 0.8405, 0.8378, 0.4392],
            'F1 (Class 1)': [0.9807, 0.9803, 0.9794, 0.9791, 0.8466],
        })
        st.dataframe(model_compare.style.format({
            'ROC AUC': '{:.4f}',
            'F1 (Class 0)': '{:.4f}',
            'F1 (Class 1)': '{:.4f}'
        }), use_container_width=True)

    with col_m2:
        st.subheader("📋 XGBOOST | 최적 모델 선정 과정")
        preds_df = pd.read_csv("preds_for_confusion.csv")
        y_true = preds_df['이탈여부']
        y_pred = preds_df['예측이탈여부']
        report_dict = classification_report(y_true, y_pred, output_dict=True)
        report_df = pd.DataFrame(report_dict).T.round(2)
        st.dataframe(report_df.loc[['0', '1', 'accuracy']], use_container_width=True)

    with col_m3:
        st.subheader("📉 XGBOOST | 혼동행렬")
        cm = confusion_matrix(y_true, y_pred)
        fig, ax = plt.subplots()
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                    xticklabels=["pred_0", "pred_1"],
                    yticklabels=["label_0", "label_1"])
        ax.set_title("Confusion Matrix")
        st.pyplot(fig)

# ========================================================
# 🧪 탭 3: 실시간 예측 시뮬레이션
# ========================================================
with tab3:
    st.header("🧪 실시간 고객 이탈 예측 시뮬레이션")

    top_features = ['참여도변동성', '세션간격표준편차', '방문횟수', '마지막방문일차이(일)']

    input_data = {}
    for col in top_features:
        input_data[col] = float(df[col].mean())

    for col in scaler.feature_names_in_:
        if col not in input_data:
            input_data[col] = float(df[col].mean())

    input_df = pd.DataFrame([input_data])
    input_df.columns = scaler.feature_names_in_

    input_scaled = scaler.transform(input_df)
    pred = model.predict(input_scaled)
    proba = model.predict_proba(input_scaled)[:, 1]

    st.markdown("### 📈 예측 결과 및 입력")
    col1, col2 = st.columns([1, 1])

    with col1:
        churn_probability = float(proba[0])
        if churn_probability < 0.25:
            risk_level = "🟢 매우 낮음"
            risk_color = "green"
        elif churn_probability < 0.5:
            risk_level = "🟡 낮음"
            risk_color = "yellow"
        elif churn_probability < 0.75:
            risk_level = "🟠 중간"
            risk_color = "orange"
        else:
            risk_level = "🔴 높음"
            risk_color = "red"

        fig = go.Figure(go.Indicator(
            mode="gauge+number",
            value=churn_probability * 100,
            domain={'x': [0, 1], 'y': [0, 1]},
            title={'text': "이탈 확률 (%)"},
            gauge={
                'axis': {'range': [0, 100]},
                'bar': {'color': risk_color},
                'steps': [
                    {'range': [0, 25], 'color': "lightgreen"},
                    {'range': [25, 50], 'color': "lightyellow"},
                    {'range': [50, 75], 'color': "orange"},
                    {'range': [75, 100], 'color': "red"}
                ]
            }
        ))
        st.plotly_chart(fig, use_container_width=True, key="churn_gauge_chart3")

    with col2:
        st.markdown(f"**예측된 이탈 확률:** `{churn_probability * 100:.1f}%`")
        st.markdown(f"**🚦 이탈 위험 등급:** <span style='color:{risk_color}; font-size:20px'>{risk_level}</span>", unsafe_allow_html=True)

        with st.expander("✏️ 입력값 입력 (중요 변수 기준)", expanded=True):
            for col in top_features:
                input_data[col] = st.number_input(
                    f"{col} 입력",
                    float(df[col].min()),
                    float(df[col].max()),
                    float(df[col].mean()),
                    key=col
                )

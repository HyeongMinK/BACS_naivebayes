# app.py ───────────────────────────────────────────────
import streamlit as st
import pandas as pd
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import LabelEncoder
from sklearn.impute import SimpleImputer

# ───── 페이지 기본 설정 ─────
st.set_page_config(
    page_title="Hotel Cancellation Dashboard",
    page_icon="🏨",
    layout="wide",
)

# ───── 헤더 / 로고 ─────
st.markdown(
    """
    <style>
    thead tr th {background:#003049; color:#fff;}
    div.stButton > button {background:#669bbc; color:white;}
    .big-font {font-size:36px !important; font-weight:700;}
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    '<p class="big-font">🚀 호텔 예약 취소 확률 & 개인화 요금 계산기 (Naive Bayes)</p>',
    unsafe_allow_html=True,
)

# ────────────────────────────────
# 1) 모델 로딩 (한 번만 캐싱)
# ────────────────────────────────
@st.cache_data(show_spinner=False)
def load_nb(train_csv: str):
    df = pd.read_csv(train_csv)

    # 날짜 파생
    df['reservation_status_date'] = pd.to_datetime(df['reservation_status_date'])
    df['res_year']  = df['reservation_status_date'].dt.year
    df['res_month'] = df['reservation_status_date'].dt.month
    df['res_day']   = df['reservation_status_date'].dt.day
    df = df.drop('reservation_status_date', axis=1)

    # 범주형 인코딩
    le = LabelEncoder()
    for c in df.select_dtypes('object'):
        df[c] = le.fit_transform(df[c])

    X, y = df.drop('is_canceled', axis=1), df['is_canceled']
    imp  = SimpleImputer(strategy="mean").fit(X)
    X_imp = imp.transform(X)

    model = GaussianNB().fit(X_imp, y)
    return model, imp, X.columns.tolist()

model, imputer, feat_cols = load_nb("hotel_bookings.csv")

# ────────────────────────────────
# 2) 사이드바 입력
# ────────────────────────────────
with st.sidebar:
    st.header("⚙️ 설정")
    up_file   = st.file_uploader("고객 데이터 업로드 (CSV)", type="csv")
    risk_pct  = st.number_input(
        "리스크 프리미엄 (% of ADR)",
        min_value=0.0,
        step=1.0,
        value=50.0,
        help="ADR의 몇 %를 리스크 금액으로 볼지 설정"
    )
    st.markdown("---")
    st.write("데이터 셀을 수정하면 예측 결과가 즉시 반영됩니다.")

# ────────────────────────────────
# 3) 업로드 → 편집 → 예측
# ────────────────────────────────
if up_file:
    # ① 업로드 파일 로드 + feature 맞추기
    df = pd.read_csv(up_file)
    for c in feat_cols:
        if c not in df.columns:
            df[c] = 0
    df = df[feat_cols]

    # ② 탭 구성
    tab_edit, tab_pred = st.tabs(["📝 원본·편집", "📊 예측 결과"])

    # ▸ 편집 탭
    with tab_edit:
        edited_df = st.data_editor(
            df,
            height=400,
            num_rows="dynamic",
            use_container_width=True,
        )

    # ▸ 예측 탭
    with tab_pred:
        X_imp = imputer.transform(edited_df)
        prob  = model.predict_proba(X_imp)[:, 1]

        res = edited_df.copy()
        res["cancellation_probability"] = prob

        # ── ★ 요금 계산 규칙 ★ ───────────────
        # 비취소 요금 = ADR (행별)
        # 리스크 프리미엄 금액 = ADR × (% / 100)
        # 환불 가능 요금 = ADR + 취소확률 × 리스크 프리미엄 금액
        # ───────────────────────────────────
        res["non_refundable_rate"] = edited_df["adr"]
        risk_amount                = edited_df["adr"] * (risk_pct / 100)
        res["refundable_rate"]     = edited_df["adr"] + prob * risk_amount

        st.metric(
            "평균 취소 확률",
            f"{prob.mean()*100:.1f}%",
            help="현재 데이터(편집 포함) 전체의 평균 값"
        )

        st.dataframe(
            res[
                ["adr", "cancellation_probability",
                 "non_refundable_rate", "refundable_rate"]
            ],
            use_container_width=True,
        )

        st.download_button(
            "⬇️ 결과 CSV 다운로드",
            data=res.to_csv(index=False).encode("utf-8-sig"),
            file_name="cancellation_predictions.csv",
            mime="text/csv",
        )
else:
    st.info("사이드바에서 고객 CSV 파일을 업로드하세요.")

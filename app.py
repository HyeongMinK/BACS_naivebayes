# app.py  ─────────────────────────────────────────────
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
    /* 테이블 헤더 톤 */
    thead tr th {background:#003049; color:#fff;}
    /* 버튼 컬러 */
    div.stButton > button {background:#669bbc; color:white;}
    /* 카드 느낌 헤더 */
    .big-font {font-size:36px !important; font-weight:700;}
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown('<p class="big-font">🚀 호텔 예약 취소 확률 & 요금 계산기(Naive Bayes 기반)</p>', unsafe_allow_html=True)

# ───────────────────────────────────────────
# 1) 학습용 모델 한 번만 로드
# ───────────────────────────────────────────
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

# ───────────────────────────────────────────
# 2) 사이드바 – 입력
# ───────────────────────────────────────────
with st.sidebar:
    st.header("⚙️ 설정")
    up_file = st.file_uploader("고객 데이터 업로드", type="csv")
    base_rate  = st.number_input("비취소 기준 요금 (€)", 0, step=10, value=100)
    premium    = st.number_input("취소 프리미엄 (€)", 0, step=5,  value=50)
    st.markdown("---")
    st.write("수정하면\n예측 결과가 **즉시** 업데이트됩니다.")

if up_file:
    # ── ① 원본 로드 및 feature 보정
    df = pd.read_csv(up_file)
    for c in feat_cols:
        if c not in df.columns:
            df[c] = 0
    df = df[feat_cols]   # 순서 맞추기

    # ── ② 탭 구성
    tab_edit, tab_pred = st.tabs(["📝 원본 편집", "📊 예측 결과"])

    # ▸ ① 편집 탭
    with tab_edit:
        edited_df = st.data_editor(
            df,
            height=400,
            num_rows="dynamic",
            use_container_width=True,
        )

    # ▸ ② 예측 탭 (편집된 데이터 사용)
    with tab_pred:
        X_imp = imputer.transform(edited_df)
        prob  = model.predict_proba(X_imp)[:, 1]

        res = edited_df.copy()
        res["cancellation_probability"] = prob
        res["non_refundable_rate"] = base_rate
        res["refundable_rate"]     = base_rate + prob * premium

        st.metric(
            "평균 취소 확률",
            f"{prob.mean()*100:.1f}%",
            help="업로드·편집된 모든 행의 평균"
        )

        st.dataframe(
            res[["cancellation_probability", "non_refundable_rate", "refundable_rate"]],
            use_container_width=True,
        )

        st.download_button(
            "⬇️ 결과 다운로드",
            data=res.to_csv(index=False).encode("utf-8-sig"),
            file_name="cancellation_predictions.csv",
            mime="text/csv",
        )
else:
    st.info("좌측 사이드바에서 DB(CSV) 파일을 업로드하세요.")

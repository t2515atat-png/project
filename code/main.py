
import streamlit as st
import pandas as pd
import plotly.express as px

# ---------------------------------
# 1. 페이지 설정
# ---------------------------------
st.set_page_config(
    page_title="공공자전거 대여량 데이터 분석",
    page_icon="🚲",
    layout="wide"
)

st.title("🚲 공공자전거 대여량 데이터 분석")
st.write(
    "기온과 시간대 정보를 활용하여 공공자전거 대여량의 "
    "특성과 변수 간 관계를 분석합니다."
)

# ---------------------------------
# 2. 데이터 불러오기
# ---------------------------------
DATA_URL = (
    "https://raw.githubusercontent.com/"
    "t2515atat-png/project/main/data/SeoulBikeData.csv"
)

@st.cache_data
def load_data():
    data = pd.read_csv(DATA_URL, encoding="unicode_escape")
    data.columns = data.columns.str.strip()

    # 중복된 열 이름이 있는 경우 첫 번째 열만 사용
    data = data.loc[:, ~data.columns.duplicated()]

    return data

try:
    df = load_data()
except Exception as e:
    st.error("데이터를 불러오지 못했습니다.")
    st.write("인터넷 연결과 CSV 주소를 확인해 주세요.")
    st.stop()

# ---------------------------------
# 3. 분석에 필요한 변수 확인
# ---------------------------------
temperature_col = "Temperature(°C)"
hour_col = "Hour"
rental_col = "Rented Bike Count"

required_cols = [
    temperature_col,
    hour_col,
    rental_col
]

missing_cols = [
    col for col in required_cols if col not in df.columns
]

if missing_cols:
    st.error(f"필요한 열을 찾을 수 없습니다: {missing_cols}")
    st.write("현재 데이터의 열 이름:", list(df.columns))
    st.stop()

# 분석에 필요한 변수는 숫자형으로 변환
for col in required_cols:
    df[col] = pd.to_numeric(df[col], errors="coerce")

# ---------------------------------
# 4. 메뉴 구성
# ---------------------------------
menu = st.sidebar.radio(
    "분석 메뉴",
    [
        "데이터 소개",
        "기초통계량",
        "데이터 분포 확인",
        "속성 간 관계 분석",
        "이상치 확인"
    ]
)

# ---------------------------------
# 5. 데이터 소개
# ---------------------------------
if menu == "데이터 소개":

    st.header("📋 데이터 소개")

    st.write(
        "서울 공공자전거 대여량 데이터를 이용하여 "
        "기온과 시간대에 따른 대여량의 변화를 분석합니다."
    )

    col1, col2 = st.columns(2)

    with col1:
        st.metric("전체 데이터 수", f"{len(df):,}개")

    with col2:
        st.metric("전체 속성 수", f"{len(df.columns)}개")

    st.subheader("데이터 미리보기")
    st.dataframe(df.head(10), use_container_width=True)

    st.subheader("데이터 속성")
    st.write(df.dtypes.astype(str).to_frame("데이터 유형"))

    st.subheader("결측값 확인")
    st.dataframe(
        df.isnull().sum().to_frame("결측값 개수"),
        use_container_width=True
    )

# ---------------------------------
# 6. 기초통계량
# ---------------------------------
elif menu == "기초통계량":

    st.header("📊 기초통계량")

    selected = st.multiselect(
        "분석할 데이터 속성을 선택하세요.",
        required_cols,
        default=required_cols
    )

    if selected:
        stats = df[selected].describe().T

        stats["중앙값"] = df[selected].median()
        stats = stats.rename(columns={
            "count": "데이터 개수",
            "mean": "평균",
            "std": "표준편차",
            "min": "최솟값",
            "25%": "1사분위수",
            "50%": "중앙값(50%)",
            "75%": "3사분위수",
            "max": "최댓값"
        })

        # 중복된 중앙값 열 제거
        if "중앙값(50%)" in stats.columns:
            stats = stats.drop(columns=["중앙값(50%)"])

        st.dataframe(
            stats.round(2),
            use_container_width=True
        )

# ---------------------------------
# 7. 데이터 분포 확인
# ---------------------------------
elif menu == "데이터 분포 확인":

    st.header("📈 데이터 분포 확인")

    selected_col = st.selectbox(
        "분포를 확인할 속성을 선택하세요.",
        required_cols
    )

    plot_data = df[selected_col].dropna()

    st.subheader(f"{selected_col} 히스토그램")

    fig = px.histogram(
        df,
        x=selected_col,
        nbins=30,
        title=f"{selected_col} 데이터 분포",
        labels={selected_col: selected_col},
        color_discrete_sequence=["#4C78A8"]
    )

    st.plotly_chart(fig, use_container_width=True)

    st.subheader(f"{selected_col} 상자그림")

    box_fig = px.box(
        df,
        y=selected_col,
        title=f"{selected_col} 상자그림",
        color_discrete_sequence=["#F58518"]
    )

    st.plotly_chart(box_fig, use_container_width=True)

    st.write("**분포 요약**")
    st.write(f"- 데이터 개수: {len(plot_data):,}개")
    st.write(f"- 평균: {plot_data.mean():.2f}")
    st.write(f"- 최솟값: {plot_data.min():.2f}")
    st.write(f"- 최댓값: {plot_data.max():.2f}")

# ---------------------------------
# 8. 데이터 속성 간 관계 분석
# ---------------------------------
elif menu == "속성 간 관계 분석":

    st.header("🔍 데이터 속성 간 관계 분석")

    st.write(
        "기온과 시간대가 공공자전거 대여량과 "
        "어떤 관계를 가지는지 확인합니다."
    )

    # 분석에 필요한 세 변수의 결측값만 제외
    analysis_df = df[required_cols].dropna().copy()

    # ---------------------------------
    # ① 기온과 대여량의 관계
    # ---------------------------------
    st.subheader("① 기온과 자전거 대여량의 관계")

    st.write(
        "기온이 변화할 때 자전거 대여량이 "
        "어떻게 달라지는지 산점도로 확인합니다."
    )

    fig1 = px.scatter(
        analysis_df,
        x=temperature_col,
        y=rental_col,
        opacity=0.45,
        title="기온과 자전거 대여량의 관계",
        labels={
            temperature_col: "기온(℃)",
            rental_col: "자전거 대여량(대)"
        }
    )

    st.plotly_chart(fig1, use_container_width=True)

    # 피어슨 상관계수
    corr = analysis_df[temperature_col].corr(
        analysis_df[rental_col]
    )

    st.metric(
        "기온과 대여량의 피어슨 상관계수",
        f"{corr:.3f}"
    )

    st.write(
        "상관계수는 두 변수 사이의 선형적인 관계를 나타냅니다. "
        "1에 가까우면 양의 상관관계, -1에 가까우면 음의 상관관계, "
        "0에 가까우면 선형적인 상관관계가 약하다는 뜻입니다."
    )

    st.divider()

    # ---------------------------------
    # ② 시간대와 대여량의 관계
    # ---------------------------------
    st.subheader("② 시간대와 자전거 대여량의 관계")

    st.write(
        "각 시간대의 평균 대여량을 계산하여 "
        "어느 시간대에 자전거 이용이 많은지 확인합니다."
    )

    hourly_avg = (
        analysis_df.groupby(hour_col)[rental_col]
        .mean()
        .reset_index()
        .sort_values(hour_col)
    )

    fig2 = px.bar(
        hourly_avg,
        x=hour_col,
        y=rental_col,
        title="시간대별 평균 자전거 대여량",
        labels={
            hour_col: "시간대(시)",
            rental_col: "평균 대여량(대)"
        }
    )

    st.plotly_chart(fig2, use_container_width=True)

    # 대여량이 가장 많은 시간대와 적은 시간대
    max_row = hourly_avg.loc[hourly_avg[rental_col].idxmax()]
    min_row = hourly_avg.loc[hourly_avg[rental_col].idxmin()]

    col1, col2 = st.columns(2)

    with col1:
        st.metric(
            "평균 대여량이 가장 많은 시간대",
            f"{int(max_row[hour_col])}시",
            f"{max_row[rental_col]:.1f}대"
        )

    with col2:
        st.metric(
            "평균 대여량이 가장 적은 시간대",
            f"{int(min_row[hour_col])}시",
            f"{min_row[rental_col]:.1f}대"
        )

    st.subheader("시간대별 평균 대여량 데이터")
    st.dataframe(
        hourly_avg.rename(columns={
            hour_col: "시간대",
            rental_col: "평균 대여량"
        }).round(2),
        use_container_width=True
    )

    st.divider()

    st.subheader("분석 결과 해석")

    st.write(
        "위의 산점도에서는 기온과 대여량의 관계를, "
        "막대그래프에서는 시간대에 따른 평균 대여량의 차이를 "
        "확인할 수 있습니다."
    )

    st.info(
        "그래프에서 확인한 실제 패턴을 바탕으로 "
        "기온과 시간대가 자전거 대여량 예측에 "
        "활용될 수 있는지 판단해 보세요."
    )

# ---------------------------------
# 9. 이상치 확인
# ---------------------------------
elif menu == "이상치 확인":

    st.header("🔎 이상치 확인")

    st.write(
        "IQR(사분위 범위)을 이용하여 "
        "대여량에서 통계적 이상치로 분류되는 값을 확인합니다."
    )

    Q1 = df[rental_col].quantile(0.25)
    Q3 = df[rental_col].quantile(0.75)
    IQR = Q3 - Q1

    lower = Q1 - 1.5 * IQR
    upper = Q3 + 1.5 * IQR

    outliers = df[
        (df[rental_col] < lower) |
        (df[rental_col] > upper)
    ].copy()

    st.write(f"- 1사분위수(Q1): {Q1:.2f}")
    st.write(f"- 3사분위수(Q3): {Q3:.2f}")
    st.write(f"- IQR: {IQR:.2f}")
    st.write(f"- 이상치 판단 기준 상한: {upper:.2f}")
    st.write(f"- 이상치 개수: {len(outliers):,}개")

    st.subheader("이상치로 분류된 데이터")

    check_cols = [
        col for col in [
            "Date",
            "Hour",
            "Rented Bike Count",
            "Temperature(°C)",
            "Functioning Day"
        ]
        if col in outliers.columns
    ]

    st.dataframe(
        outliers[check_cols],
        use_container_width=True
    )

    st.warning(
        "통계적 이상치가 반드시 잘못된 데이터라는 뜻은 아닙니다. "
        "실제 수요 변화일 가능성도 있으므로 오류라는 근거가 부족하면 "
        "임의로 삭제하지 않고 유지합니다."
    )

st.sidebar.divider()
st.sidebar.caption("공공자전거 대여량 데이터 분석 프로젝트")

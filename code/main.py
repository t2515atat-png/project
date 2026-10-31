
import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import io

# -----------------------------------------
# 1. 기본 설정
# -----------------------------------------
st.set_page_config(
    page_title="공공자전거 기초통계분석",
    page_icon="🚲",
    layout="wide"
)

DATA_URL = (
    "https://raw.githubusercontent.com/"
    "t2515atat-png/project/main/data/SeoulBikeData.csv"
)

st.title("🚲 공공자전거 데이터 기초통계분석")
st.write(
    "서울 공공자전거 대여량과 날씨 데이터를 이용하여 "
    "데이터의 특징과 변수 사이의 관계를 분석합니다."
)

# -----------------------------------------
# 2. 데이터 불러오기
# -----------------------------------------
@st.cache_data
def load_online_data():
    df = pd.read_csv(
        DATA_URL,
        sep=";",
        encoding="unicode_escape"
    )

    if df.shape[1] == 1:
        df = pd.read_csv(
            DATA_URL,
            encoding="utf-8-sig"
        )

    return df


def load_uploaded_data(uploaded_file):
    raw = uploaded_file.getvalue()

    for encoding in ["utf-8-sig", "cp949", "unicode_escape"]:
        try:
            data = pd.read_csv(
                io.BytesIO(raw),
                sep=None,
                engine="python",
                encoding=encoding
            )

            if data.shape[1] == 1:
                data = pd.read_csv(
                    io.BytesIO(raw),
                    sep=";",
                    encoding=encoding
                )

            return data

        except Exception:
            continue

    raise ValueError("CSV 파일을 읽을 수 없습니다.")


with st.sidebar:
    st.header("📂 데이터 설정")

    uploaded_file = st.file_uploader(
        "CSV 파일 업로드",
        type=["csv"]
    )

if uploaded_file is not None:
    try:
        df = load_uploaded_data(uploaded_file)
        st.sidebar.success("업로드한 데이터를 불러왔습니다.")
    except Exception as e:
        st.error(f"데이터 불러오기 오류: {e}")
        st.stop()
else:
    try:
        df = load_online_data()
        st.sidebar.success("GitHub 데이터를 불러왔습니다.")
    except Exception as e:
        st.error(f"데이터를 불러오지 못했습니다: {e}")
        st.stop()

df.columns = df.columns.astype(str).str.strip()

# -----------------------------------------
# 3. 데이터 전처리
# -----------------------------------------
for col in df.columns:
    if col.strip().lower() != "date":
        converted = pd.to_numeric(df[col], errors="coerce")

        non_missing = df[col].notna().sum()

        if non_missing > 0:
            rate = converted.notna().sum() / non_missing

            if rate >= 0.8:
                df[col] = converted

numeric_cols = df.select_dtypes(include="number").columns.tolist()


def find_column(possible_names):
    for col in df.columns:
        normalized = (
            str(col).lower()
            .replace(" ", "")
            .replace("_", "")
        )

        for name in possible_names:
            if name in normalized:
                return col

    return None


rental_col = find_column([
    "rentedbikecount",
    "rentalcount",
    "대여량",
    "대여건수"
])

hour_col = find_column([
    "hour",
    "시간대",
    "시간"
])

temp_col = find_column([
    "temperature",
    "기온",
    "온도"
])

# -----------------------------------------
# 4. 데이터 기본 정보
# -----------------------------------------
st.header("1. 데이터 기본 정보")

col1, col2, col3, col4 = st.columns(4)

col1.metric("전체 행 수", f"{len(df):,}개")
col2.metric("전체 열 수", f"{len(df.columns):,}개")
col3.metric("수치형 변수 수", f"{len(numeric_cols):,}개")
col4.metric("전체 결측값", f"{int(df.isna().sum().sum()):,}개")

with st.expander("데이터 열 이름 확인"):
    st.write(df.columns.tolist())

st.subheader("데이터 미리보기")
st.dataframe(df.head(10), use_container_width=True)

st.subheader("열별 데이터 유형")

type_df = pd.DataFrame({
    "열 이름": df.columns,
    "데이터 유형": [str(df[col].dtype) for col in df.columns],
    "결측값 수": [int(df[col].isna().sum()) for col in df.columns],
    "고유값 수": [int(df[col].nunique()) for col in df.columns]
})

st.dataframe(type_df, use_container_width=True)

# -----------------------------------------
# 5. 기초통계량 분석
# -----------------------------------------
st.header("2. 기초통계량 분석")

if numeric_cols:

    selected_stats_cols = st.multiselect(
        "통계량을 확인할 변수 선택",
        numeric_cols,
        default=numeric_cols
    )

    if selected_stats_cols:

        stats = df[selected_stats_cols].describe().T

        stats["중앙값"] = df[selected_stats_cols].median()
        stats["결측값 수"] = df[selected_stats_cols].isna().sum()

        stats = stats.rename(columns={
            "count": "데이터 수",
            "mean": "평균",
            "std": "표준편차",
            "min": "최솟값",
            "25%": "제1사분위수",
            "50%": "중앙값",
            "75%": "제3사분위수",
            "max": "최댓값"
        })

        st.dataframe(
            stats.round(2),
            use_container_width=True
        )

        st.download_button(
            "📥 기초통계량 CSV 다운로드",
            data=stats.to_csv(
                encoding="utf-8-sig"
            ).encode("utf-8-sig"),
            file_name="기초통계량.csv",
            mime="text/csv"
        )

    else:
        st.info("분석할 변수를 선택해 주세요.")

# -----------------------------------------
# 6. 결측값 분석
# -----------------------------------------
st.header("3. 결측값 분석")

missing_df = pd.DataFrame({
    "변수": df.columns,
    "결측값 수": df.isna().sum().values,
    "결측 비율(%)": (
        df.isna().mean().values * 100
    ).round(2)
})

st.dataframe(missing_df, use_container_width=True)

if df.isna().sum().sum() == 0:
    st.success("결측값이 없습니다.")
else:
    fig = px.bar(
        missing_df,
        x="변수",
        y="결측값 수",
        title="변수별 결측값 현황",
        text="결측값 수"
    )

    fig.update_layout(
        xaxis_title="변수",
        yaxis_title="결측값 수"
    )

    st.plotly_chart(fig, use_container_width=True)

# -----------------------------------------
# 7. 변수 분포 분석
# -----------------------------------------
st.header("4. 변수 분포 분석")

if numeric_cols:

    distribution_col = st.selectbox(
        "분포를 확인할 변수",
        numeric_cols,
        index=(
            numeric_cols.index(rental_col)
            if rental_col in numeric_cols
            else 0
        )
    )

    plot_data = df[distribution_col].dropna()

    col1, col2 = st.columns(2)

    with col1:
        fig = px.histogram(
            plot_data,
            x=distribution_col,
            nbins=30,
            title=f"{distribution_col} 히스토그램"
        )

        fig.update_layout(
            xaxis_title=distribution_col,
            yaxis_title="빈도"
        )

        st.plotly_chart(fig, use_container_width=True)

    with col2:
        fig = px.box(
            plot_data,
            y=distribution_col,
            title=f"{distribution_col} 상자그림"
        )

        fig.update_layout(
            yaxis_title=distribution_col
        )

        st.plotly_chart(fig, use_container_width=True)

# -----------------------------------------
# 8. 시간대별 공공자전거 대여량
# -----------------------------------------
st.header("5. 시간대별 공공자전거 대여량")

if hour_col and rental_col:

    hourly = (
        df.groupby(hour_col)[rental_col]
        .agg(["mean", "median", "count"])
        .reset_index()
        .sort_values(hour_col)
    )

    hourly = hourly.rename(columns={
        hour_col: "시간대",
        "mean": "평균 대여량",
        "median": "대여량 중앙값",
        "count": "데이터 수"
    })

    fig = px.line(
        hourly,
        x="시간대",
        y="평균 대여량",
        markers=True,
        title="시간대별 평균 공공자전거 대여량"
    )

    fig.update_layout(
        xaxis_title="시간대",
        yaxis_title="평균 대여량"
    )

    st.plotly_chart(fig, use_container_width=True)

    st.subheader("시간대별 통계")
    st.dataframe(
        hourly.round(2),
        use_container_width=True
    )

    max_hour = hourly.loc[
        hourly["평균 대여량"].idxmax()
    ]

    min_hour = hourly.loc[
        hourly["평균 대여량"].idxmin()
    ]

    col1, col2 = st.columns(2)

    col1.metric(
        "대여량이 가장 많은 시간대",
        f"{max_hour['시간대']}시",
        f"평균 {max_hour['평균 대여량']:.1f}대"
    )

    col2.metric(
        "대여량이 가장 적은 시간대",
        f"{min_hour['시간대']}시",
        f"평균 {min_hour['평균 대여량']:.1f}대"
    )

else:
    st.warning("시간대 또는 대여량 변수를 찾을 수 없습니다.")

# -----------------------------------------
# 9. 기온과 대여량의 관계
# -----------------------------------------
st.header("6. 기온과 대여량의 관계")

if temp_col and rental_col:

    relation_df = df[
        [temp_col, rental_col]
    ].dropna()

    if len(relation_df) > 1:

        corr = relation_df[temp_col].corr(
            relation_df[rental_col]
        )

        st.metric(
            "기온과 대여량의 피어슨 상관계수",
            f"{corr:.3f}"
        )

        fig = px.scatter(
            relation_df,
            x=temp_col,
            y=rental_col,
            title="기온과 공공자전거 대여량의 관계",
            opacity=0.5
        )

        fig.update_layout(
            xaxis_title="기온(℃)",
            yaxis_title="대여량"
        )

        st.plotly_chart(fig, use_container_width=True)

        if corr > 0:
            st.write(
                "양의 상관관계가 나타납니다. "
                "기온이 높아질수록 대여량이 증가하는 경향이 있습니다."
            )
        elif corr < 0:
            st.write(
                "음의 상관관계가 나타납니다. "
                "기온이 높아질수록 대여량이 감소하는 경향이 있습니다."
            )
        else:
            st.write("선형 상관관계가 거의 나타나지 않습니다.")

        st.caption(
            "상관관계만으로 기온이 대여량 변화의 원인이라고 "
            "단정할 수는 없습니다."
        )

else:
    st.warning("기온 또는 대여량 변수를 찾을 수 없습니다.")

# -----------------------------------------
# 10. 수치형 변수 사이의 상관관계
# -----------------------------------------
st.header("7. 수치형 변수 사이의 상관관계")

if len(numeric_cols) >= 2:

    corr_matrix = df[numeric_cols].corr()

    fig = px.imshow(
        corr_matrix,
        text_auto=".2f",
        aspect="auto",
        color_continuous_scale="RdBu_r",
        zmin=-1,
        zmax=1,
        title="수치형 변수 상관관계 히트맵"
    )

    st.plotly_chart(fig, use_container_width=True)

    st.subheader("상관관계 표")
    st.dataframe(
        corr_matrix.round(2),
        use_container_width=True
    )

    st.subheader("두 변수의 관계 직접 확인하기")

    col1, col2 = st.columns(2)

    with col1:
        x_col = st.selectbox(
            "X축 변수",
            numeric_cols,
            index=(
                numeric_cols.index(temp_col)
                if temp_col in numeric_cols
                else 0
            )
        )

    with col2:
        y_col = st.selectbox(
            "Y축 변수",
            numeric_cols,
            index=(
                numeric_cols.index(rental_col)
                if rental_col in numeric_cols
                else 1
            )
        )

    if x_col != y_col:

        pair_df = df[
            [x_col, y_col]
        ].dropna()

        pair_corr = pair_df[x_col].corr(
            pair_df[y_col]
        )

        st.metric(
            "선택한 두 변수의 상관계수",
            f"{pair_corr:.3f}"
        )

        fig = px.scatter(
            pair_df,
            x=x_col,
            y=y_col,
            title=f"{x_col}와 {y_col}의 관계",
            opacity=0.5
        )

        st.plotly_chart(fig, use_container_width=True)

    else:
        st.info("서로 다른 두 변수를 선택해 주세요.")

# -----------------------------------------
# 11. 데이터 다운로드
# -----------------------------------------
st.header("8. 데이터 다운로드")

st.download_button(
    "📥 현재 분석 데이터 CSV 다운로드",
    data=df.to_csv(
        index=False,
        encoding="utf-8-sig"
    ).encode("utf-8-sig"),
    file_name="공공자전거_분석데이터.csv",
    mime="text/csv"
)

st.divider()

st.caption(
    "공공자전거 기초통계분석 | "
    "기온과 시간대에 따른 대여량 분석"
)

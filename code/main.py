
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
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
    "서울 공공자전거 대여량과 날씨 데이터를 이용해 "
    "데이터의 특징을 살펴보고 기온, 시간대와 대여량의 관계를 분석합니다."
)

# -----------------------------------------
# 2. 데이터 불러오기
# -----------------------------------------
@st.cache_data
def load_online_data():
    # 구분자를 자동으로 판별해 CSV 읽기
    data = pd.read_csv(
        DATA_URL,
        sep=None,
        engine="python",
        encoding="utf-8-sig"
    )

    # 구분자 판별이 잘되지 않아 열이 하나만 생겼을 때 재시도
    if data.shape[1] == 1:
        data = pd.read_csv(
            DATA_URL,
            sep=";",
            encoding="utf-8-sig"
        )

    return data


def load_uploaded_data(uploaded_file):
    raw = uploaded_file.getvalue()

    # 인코딩과 구분자를 바꾸어 읽기 시도
    for encoding in ["utf-8-sig", "cp949", "latin1"]:
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
        "CSV 파일 직접 업로드",
        type=["csv"]
    )

if uploaded_file is not None:
    try:
        df = load_uploaded_data(uploaded_file)
        st.sidebar.success("업로드한 데이터를 불러왔습니다.")
    except Exception as e:
        st.error(f"파일을 읽는 중 오류가 발생했습니다: {e}")
        st.stop()
else:
    try:
        df = load_online_data()
        st.sidebar.success("GitHub 데이터를 불러왔습니다.")
    except Exception as e:
        st.error(
            "GitHub에서 데이터를 불러오지 못했습니다. "
            "CSV 파일을 다운로드한 뒤 왼쪽 메뉴에서 업로드해 주세요."
        )
        st.caption(f"오류 내용: {e}")
        st.stop()

# 열 이름의 앞뒤 공백 제거
df.columns = df.columns.astype(str).str.strip()

# 숫자처럼 저장된 열을 수치형으로 변환
# 날짜 열은 숫자로 변환하지 않음
for col in df.columns:
    if col.strip().lower() != "date":
        converted = pd.to_numeric(df[col], errors="coerce")

        # 기존 값 중 80% 이상이 숫자로 변환되는 열만 변환
        non_missing = df[col].notna().sum()
        if non_missing > 0:
            conversion_rate = converted.notna().sum() / non_missing
            if conversion_rate >= 0.8:
                df[col] = converted

# -----------------------------------------
# 3. 주요 변수 자동 찾기
# -----------------------------------------
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

numeric_cols = df.select_dtypes(include="number").columns.tolist()

# -----------------------------------------
# 4. 데이터 기본 정보
# -----------------------------------------
st.header("1. 데이터 기본 정보")

col1, col2, col3, col4 = st.columns(4)

col1.metric("전체 행 수", f"{len(df):,}개")
col2.metric("전체 열 수", f"{len(df.columns):,}개")
col3.metric("수치형 변수 수", f"{len(numeric_cols):,}개")
col4.metric("전체 결측값", f"{int(df.isna().sum().sum()):,}개")

with st.expander("데이터 열 이름 확인하기"):
    st.write(df.columns.tolist())

st.subheader("데이터 미리보기")
st.dataframe(df.head(10), use_container_width=True)

st.subheader("열별 데이터 유형")
type_df = pd.DataFrame({
    "열 이름": df.columns,
    "데이터 유형": [str(df[col].dtype) for col in df.columns],
    "결측값 수": [int(df[col].isna().sum()) for col in df.columns],
    "고유값 수": [int(df[col].nunique(dropna=True)) for col in df.columns]
})
st.dataframe(type_df, use_container_width=True)

# -----------------------------------------
# 5. 기초통계량
# -----------------------------------------
st.header("2. 기초통계량 분석")

if numeric_cols:
    selected_stats_cols = st.multiselect(
        "통계량을 확인할 변수 선택",
        options=numeric_cols,
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
            "50%": "중앙값(50%)",
            "75%": "제3사분위수",
            "max": "최댓값"
        })

        st.dataframe(
            stats.round(2),
            use_container_width=True
        )

        st.download_button(
            "📥 기초통계량 CSV 다운로드",
            data=stats.to_csv(encoding="utf-8-sig").encode("utf-8-sig"),
            file_name="기초통계량.csv",
            mime="text/csv"
        )
    else:
        st.info("통계량을 확인할 변수를 하나 이상 선택해 주세요.")
else:
    st.warning("분석할 수 있는 수치형 변수가 없습니다.")

# -----------------------------------------
# 6. 결측값 분석
# -----------------------------------------
st.header("3. 결측값 분석")

missing_df = pd.DataFrame({
    "열 이름": df.columns,
    "결측값 수": df.isna().sum().values,
    "결측 비율(%)": (
        df.isna().mean().values * 100
    ).round(2)
})

missing_df = missing_df.sort_values(
    "결측값 수",
    ascending=False
)

st.dataframe(missing_df, use_container_width=True)

if df.isna().sum().sum() == 0:
    st.success("이 데이터에는 결측값이 없습니다.")
else:
    st.warning(
        "일부 열에 결측값이 있습니다. "
        "분석할 때 결측값의 처리 방법을 고려해야 합니다."
    )

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

    if len(plot_data) > 0:
        fig, ax = plt.subplots(figsize=(9, 4))
        ax.hist(
            plot_data,
            bins=30,
            edgecolor="black"
        )
        ax.set_title(f"{distribution_col} 분포")
        ax.set_xlabel(distribution_col)
        ax.set_ylabel("빈도")
        ax.grid(axis="y", alpha=0.3)

        st.pyplot(fig)
        plt.close(fig)

        st.caption(
            "히스토그램은 각 값의 구간에 데이터가 "
            "얼마나 많이 분포하는지 보여 줍니다."
        )
else:
    st.info("수치형 변수가 없어 분포 그래프를 만들 수 없습니다.")

# -----------------------------------------
# 8. 시간대별 대여량
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

    st.subheader("시간대별 평균 대여량")
    st.line_chart(
        hourly,
        x="시간대",
        y="평균 대여량"
    )

    st.subheader("시간대별 통계")
    st.dataframe(hourly.round(2), use_container_width=True)

    max_hour = hourly.loc[
        hourly["평균 대여량"].idxmax()
    ]
    min_hour = hourly.loc[
        hourly["평균 대여량"].idxmin()
    ]

    c1, c2 = st.columns(2)
    c1.metric(
        "평균 대여량이 가장 많은 시간대",
        f"{max_hour['시간대']}시",
        f"평균 {max_hour['평균 대여량']:.1f}대"
    )
    c2.metric(
        "평균 대여량이 가장 적은 시간대",
        f"{min_hour['시간대']}시",
        f"평균 {min_hour['평균 대여량']:.1f}대"
    )
else:
    st.info(
        "시간대 또는 대여량 열을 자동으로 찾지 못했습니다. "
        "아래 데이터 열 이름을 확인해 주세요."
    )

# -----------------------------------------
# 9. 기온과 대여량의 관계
# -----------------------------------------
st.header("6. 기온과 대여량의 관계")

if temp_col and rental_col:
    relation_df = df[[temp_col, rental_col]].dropna()

    if len(relation_df) > 1:
        corr = relation_df[temp_col].corr(
            relation_df[rental_col]
        )

        st.metric(
            "기온과 대여량의 피어슨 상관계수",
            f"{corr:.3f}"
        )

        st.scatter_chart(
            relation_df,
            x=temp_col,
            y=rental_col
        )

        if corr > 0:
            st.write(
                "해석: 이 데이터에서는 기온이 높을수록 "
                "대여량도 증가하는 양의 상관관계가 나타납니다."
            )
        elif corr < 0:
            st.write(
                "해석: 이 데이터에서는 기온이 높을수록 "
                "대여량이 감소하는 음의 상관관계가 나타납니다."
            )
        else:
            st.write(
                "해석: 기온과 대여량 사이의 선형 상관관계가 "
                "거의 나타나지 않습니다."
            )

        st.caption(
            "상관관계는 두 변수의 함께 변하는 정도를 나타냅니다. "
            "상관관계만으로 기온이 대여량 변화의 원인이라고 "
            "단정할 수는 없습니다."
        )
else:
    st.info(
        "기온 또는 대여량 열을 자동으로 찾지 못했습니다. "
        "CSV의 실제 열 이름을 확인해 주세요."
    )

# -----------------------------------------
# 10. 원하는 변수 사이의 상관관계
# -----------------------------------------
st.header("7. 수치형 변수 사이의 상관관계")

if len(numeric_cols) >= 2:
    corr_matrix = df[numeric_cols].corr()

    st.write(
        "상관계수는 -1에서 1 사이의 값입니다. "
        "1에 가까우면 양의 선형 상관관계, "
        "-1에 가까우면 음의 선형 상관관계를 의미합니다."
    )

    st.dataframe(
        corr_matrix.round(2),
        use_container_width=True
    )

    st.subheader("두 변수의 관계 직접 확인하기")

    left, right = st.columns(2)

    with left:
        x_col = st.selectbox(
            "X축 변수",
            numeric_cols,
            index=(
                numeric_cols.index(temp_col)
                if temp_col in numeric_cols
                else 0
            ),
            key="x_variable"
        )

    with right:
        y_col = st.selectbox(
            "Y축 변수",
            numeric_cols,
            index=(
                numeric_cols.index(rental_col)
                if rental_col in numeric_cols
                else min(1, len(numeric_cols) - 1)
            ),
            key="y_variable"
        )

    pair_df = df[[x_col, y_col]].dropna()

    if x_col != y_col and len(pair_df) > 1:
        pair_corr = pair_df[x_col].corr(pair_df[y_col])

        st.metric(
            f"{x_col}와 {y_col}의 상관계수",
            f"{pair_corr:.3f}"
        )

        st.scatter_chart(
            pair_df,
            x=x_col,
            y=y_col
        )
    elif x_col == y_col:
        st.info("서로 다른 두 변수를 선택해 주세요.")
else:
    st.info("상관관계를 계산하려면 수치형 변수가 2개 이상 필요합니다.")

# -----------------------------------------
# 11. 원본 데이터 다운로드
# -----------------------------------------
st.header("8. 데이터 다운로드")

st.download_button(
    "📥 현재 분석 데이터 CSV 다운로드",
    data=df.to_csv(index=False).encode("utf-8-sig"),
    file_name="공공자전거_분석데이터.csv",
    mime="text/csv"
)

st.divider()
st.caption(
    "분석 주제: 기온과 시간대에 따른 공공자전거 대여량 예측 | "
    "기초통계분석 도구"
)


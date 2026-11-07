
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px

from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score


# =====================================
# 1. 페이지 설정
# =====================================

st.set_page_config(
    page_title="공공자전거 대여량 분석",
    page_icon="🚲",
    layout="wide"
)

st.title("🚲 공공자전거 대여량 데이터 분석")
st.write(
    "기온과 날씨, 시간대에 따른 공공자전거 대여량을 분석하고 "
    "회귀 모델을 활용하여 대여량을 예측합니다."
)


# =====================================
# 2. 데이터 불러오기
# =====================================

DATA_URL = (
    "https://raw.githubusercontent.com/"
    "t2515atat-png/project/main/data/SeoulBikeData.csv"
)


@st.cache_data
def load_data():
    data = pd.read_csv(
        DATA_URL,
        encoding="unicode_escape"
    )
    data.columns = data.columns.str.strip()
    return data


try:
    df = load_data()
except Exception as e:
    st.error("데이터를 불러오지 못했습니다.")
    st.exception(e)
    st.stop()


# =====================================
# 3. 변수 설정
# =====================================

temperature_col = "Temperature(°C)"
hour_col = "Hour"
rental_col = "Rented Bike Count"

required_cols = [
    temperature_col,
    hour_col,
    rental_col
]

missing_cols = [
    col for col in required_cols
    if col not in df.columns
]

if missing_cols:
    st.error(f"필요한 열이 없습니다: {missing_cols}")
    st.write("현재 데이터 열:", df.columns.tolist())
    st.stop()


# 날씨 관련 수치형 변수
weather_candidates = [
    "Temperature(°C)",
    "Humidity(%)",
    "Wind speed (m/s)",
    "Visibility (10m)",
    "Dew point temperature(°C)",
    "Solar Radiation (MJ/m2)",
    "Rainfall(mm)",
    "Snowfall (cm)"
]

weather_cols = [
    col for col in weather_candidates
    if col in df.columns
]

# 범주형 날씨 및 환경 변수
category_candidates = [
    "Seasons",
    "Holiday",
    "Functioning Day"
]

category_cols = [
    col for col in category_candidates
    if col in df.columns
]

# 숫자형 변환
numeric_cols = list(dict.fromkeys(
    required_cols + weather_cols
))

for col in numeric_cols:
    df[col] = pd.to_numeric(
        df[col],
        errors="coerce"
    )


# =====================================
# 4. 분석 데이터 준비
# =====================================

analysis_cols = list(dict.fromkeys(
    required_cols + weather_cols
))

analysis_df = df[analysis_cols].copy()

missing_rows = analysis_df.isnull().any(axis=1).sum()
missing_count = analysis_df.isnull().sum().sum()

clean_df = analysis_df.dropna().copy()

# 날짜 순서 정렬
if "Date" in df.columns:
    dates = pd.to_datetime(
        df["Date"],
        dayfirst=True,
        errors="coerce"
    )

    clean_df["Date"] = dates.loc[clean_df.index]

    clean_df = clean_df.sort_values(
        ["Date", hour_col]
    )

    clean_df = clean_df.drop(columns=["Date"])
else:
    clean_df = clean_df.sort_index()

clean_df = clean_df.reset_index(drop=True)


# 범주형 분석용 데이터
category_df = df[category_cols].copy()

category_df[rental_col] = df[rental_col]
category_df[temperature_col] = df[temperature_col]


# =====================================
# 5. 사이드바 메뉴
# =====================================

menu = st.sidebar.radio(
    "분석 메뉴",
    [
        "1. 데이터 소개",
        "2. 기초통계분석",
        "3. 데이터 탐색 및 시각화",
        "4. 데이터 속성 간 관계 분석",
        "5. 데이터 전처리",
        "6. 모델 학습 및 성능 비교",
        "7. 최종 결과"
    ]
)


# =====================================
# 6. 모델 학습 함수
# =====================================

def train_models(data):

    X = data[[temperature_col, hour_col]]
    y = data[rental_col]

    split_index = int(len(data) * 0.8)

    X_train = X.iloc[:split_index]
    X_test = X.iloc[split_index:]

    y_train = y.iloc[:split_index]
    y_test = y.iloc[split_index:]

    # 베이스라인
    baseline_prediction = np.full(
        len(y_test),
        y_train.mean()
    )

    models = {
        "선형 회귀": LinearRegression(),

        "의사결정나무 회귀": DecisionTreeRegressor(
            max_depth=8,
            min_samples_leaf=5,
            random_state=42
        ),

        "랜덤 포레스트 회귀": RandomForestRegressor(
            n_estimators=100,
            max_depth=12,
            min_samples_leaf=3,
            random_state=42,
            n_jobs=-1
        )
    }

    results = []
    predictions = {}

    results.append({
        "모델": "베이스라인",
        "MAE": mean_absolute_error(
            y_test,
            baseline_prediction
        ),
        "R²": r2_score(
            y_test,
            baseline_prediction
        )
    })

    predictions["베이스라인"] = baseline_prediction

    for name, model in models.items():

        model.fit(X_train, y_train)

        pred = model.predict(X_test)

        results.append({
            "모델": name,
            "MAE": mean_absolute_error(y_test, pred),
            "R²": r2_score(y_test, pred)
        })

        predictions[name] = pred

    return (
        pd.DataFrame(results),
        predictions,
        y_test.reset_index(drop=True)
    )


# =====================================
# 7. 데이터 소개
# =====================================

if menu == "1. 데이터 소개":

    st.header("📋 데이터 소개")

    st.write(
        "서울 공공자전거 대여량 데이터를 활용하여 "
        "기온, 날씨, 시간대와 대여량 사이의 관계를 분석합니다."
    )

    col1, col2, col3 = st.columns(3)

    col1.metric("전체 데이터 수", f"{len(df):,}개")
    col2.metric("전체 속성 수", f"{len(df.columns)}개")
    col3.metric("날씨 수치형 변수 수", f"{len(weather_cols)}개")

    st.subheader("데이터 미리보기")
    st.dataframe(df.head(10), use_container_width=True)

    st.subheader("주요 분석 변수")

    st.write("입력 변수: 기온, 시간대")
    st.write("예측 대상: 자전거 대여량")

    st.write("날씨 분석 변수:")
    st.write(", ".join(weather_cols))

    st.subheader("데이터 열 이름")
    st.write(df.columns.tolist())


# =====================================
# 8. 기초통계분석
# =====================================

elif menu == "2. 기초통계분석":

    st.header("📈 기초통계분석")

    st.write(
        "기온, 습도, 풍속, 강수량 등 날씨 변수와 "
        "자전거 대여량의 기초통계량을 확인합니다."
    )

    st.subheader("① 전체 변수의 기초통계량")

    stats = analysis_df.describe().T

    stats["중앙값"] = analysis_df.median()

    stats = stats.rename(columns={
        "count": "데이터 개수",
        "mean": "평균",
        "std": "표준편차",
        "min": "최솟값",
        "25%": "1사분위수",
        "75%": "3사분위수",
        "max": "최댓값"
    })

    stats = stats.drop(
        columns=["50%"],
        errors="ignore"
    )

    st.dataframe(
        stats.round(2),
        use_container_width=True
    )

    st.divider()

    st.subheader("② 변수별 통계량")

    selected_col = st.selectbox(
        "확인할 변수를 선택하세요.",
        analysis_cols
    )

    values = analysis_df[selected_col]

    c1, c2, c3 = st.columns(3)

    c1.metric("평균", f"{values.mean():,.2f}")
    c2.metric("중앙값", f"{values.median():,.2f}")
    c3.metric("표준편차", f"{values.std():,.2f}")

    c4, c5, c6 = st.columns(3)

    c4.metric("최솟값", f"{values.min():,.2f}")
    c5.metric("최댓값", f"{values.max():,.2f}")
    c6.metric("데이터 개수", f"{values.count():,}개")

    st.divider()

    st.subheader("③ 기온 구간별 평균 대여량")

    temp_bins = [
        -np.inf, 0, 10, 20, 30, np.inf
    ]

    temp_labels = [
        "0℃ 미만",
        "0~10℃",
        "10~20℃",
        "20~30℃",
        "30℃ 이상"
    ]

    temp_data = clean_df.copy()

    temp_data["기온 구간"] = pd.cut(
        temp_data[temperature_col],
        bins=temp_bins,
        labels=temp_labels,
        right=False
    )

    temp_stats = (
        temp_data.groupby(
            "기온 구간",
            observed=False
        )[rental_col]
        .agg(["count", "mean", "median", "std"])
        .reset_index()
    )

    temp_stats.columns = [
        "기온 구간",
        "데이터 개수",
        "평균 대여량",
        "중앙값",
        "표준편차"
    ]

    st.dataframe(
        temp_stats.round(2),
        use_container_width=True,
        hide_index=True
    )

    st.write(
        "기온 구간별 평균 대여량을 비교하여 "
        "어떤 기온에서 자전거 이용이 많은지 확인할 수 있습니다."
    )

    st.divider()

    st.subheader("④ 날씨 관련 범주별 평균 대여량")

    available_categories = [
        col for col in category_cols
        if col in df.columns
    ]

    if available_categories:

        selected_category = st.selectbox(
            "비교할 범주형 변수를 선택하세요.",
            available_categories
        )

        category_data = pd.DataFrame({
            "분류": df[selected_category],
            "대여량": df[rental_col]
        })

        category_data["대여량"] = pd.to_numeric(
            category_data["대여량"],
            errors="coerce"
        )

        category_result = (
            category_data.dropna()
            .groupby("분류")["대여량"]
            .agg(["count", "mean", "median"])
            .reset_index()
        )

        category_result.columns = [
            "분류",
            "데이터 개수",
            "평균 대여량",
            "중앙값"
        ]

        st.dataframe(
            category_result.round(2),
            use_container_width=True,
            hide_index=True
        )

    else:
        st.info("분석 가능한 범주형 변수가 없습니다.")


# =====================================
# 9. 데이터 탐색 및 시각화
# =====================================

elif menu == "3. 데이터 탐색 및 시각화":

    st.header("📊 데이터 탐색 및 시각화")

    st.write(
        "날씨 변수의 분포와 기온에 따른 대여량 변화를 그래프로 확인합니다."
    )

    st.subheader("① 날씨 변수별 히스토그램")

    selected_weather = st.selectbox(
        "분포를 확인할 날씨 변수",
        weather_cols
    )

    fig_hist = px.histogram(
        clean_df,
        x=selected_weather,
        nbins=30,
        title=f"{selected_weather} 분포"
    )

    st.plotly_chart(
        fig_hist,
        use_container_width=True
    )

    st.divider()

    st.subheader("② 날씨 변수별 상자그림")

    fig_box = px.box(
        clean_df,
        y=selected_weather,
        title=f"{selected_weather} 상자그림"
    )

    st.plotly_chart(
        fig_box,
        use_container_width=True
    )

    st.write(
        "상자그림을 통해 중앙값, 데이터의 퍼짐, "
        "통계적 이상치로 표시되는 값을 확인할 수 있습니다."
    )

    st.divider()

    st.subheader("③ 기온에 따른 대여량 산점도")

    fig_temp = px.scatter(
        clean_df,
        x=temperature_col,
        y=rental_col,
        opacity=0.4,
        title="기온과 자전거 대여량의 관계"
    )

    st.plotly_chart(
        fig_temp,
        use_container_width=True
    )

    st.divider()

    st.subheader("④ 기온 구간별 평균 대여량 그래프")

    temp_data = clean_df.copy()

    temp_data["기온 구간"] = pd.cut(
        temp_data[temperature_col],
        bins=[
            -np.inf, 0, 10, 20, 30, np.inf
        ],
        labels=[
            "0℃ 미만",
            "0~10℃",
            "10~20℃",
            "20~30℃",
            "30℃ 이상"
        ],
        right=False
    )

    temp_avg = (
        temp_data.groupby(
            "기온 구간",
            observed=False
        )[rental_col]
        .mean()
        .reset_index()
    )

    fig_temp_avg = px.bar(
        temp_avg,
        x="기온 구간",
        y=rental_col,
        title="기온 구간별 평균 자전거 대여량",
        text_auto=".1f"
    )

    st.plotly_chart(
        fig_temp_avg,
        use_container_width=True
    )

    st.divider()

    st.subheader("⑤ 날씨 변수별 대여량 비교")

    selected_weather_relation = st.selectbox(
        "대여량과 비교할 날씨 변수",
        [
            col for col in weather_cols
            if col != temperature_col
        ] or weather_cols,
        key="weather_relation"
    )

    fig_weather = px.scatter(
        clean_df,
        x=selected_weather_relation,
        y=rental_col,
        opacity=0.4,
        title=f"{selected_weather_relation}과 대여량의 관계"
    )

    st.plotly_chart(
        fig_weather,
        use_container_width=True
    )


# =====================================
# 10. 데이터 속성 간 관계 분석
# =====================================

elif menu == "4. 데이터 속성 간 관계 분석":

    st.header("🔍 데이터 속성 간 관계 분석")

    st.subheader("① 날씨 변수와 대여량의 상관관계")

    corr_cols = list(dict.fromkeys(
        weather_cols + [rental_col, hour_col]
    ))

    corr_matrix = clean_df[
        corr_cols
    ].corr()

    fig_corr = px.imshow(
        corr_matrix,
        text_auto=".2f",
        aspect="auto",
        color_continuous_scale="RdBu_r",
        zmin=-1,
        zmax=1,
        title="날씨 변수와 대여량의 상관관계"
    )

    st.plotly_chart(
        fig_corr,
        use_container_width=True
    )

    st.write(
        "상관계수는 두 변수 사이의 선형적인 관계를 나타냅니다. "
        "상관관계가 높다고 해서 한 변수가 다른 변수의 원인이라고 "
        "단정할 수는 없습니다."
    )

    st.divider()

    st.subheader("② 기온과 대여량의 관계")

    temp_corr = clean_df[
        temperature_col
    ].corr(clean_df[rental_col])

    st.metric(
        "기온과 대여량의 상관계수",
        f"{temp_corr:.3f}"
    )

    fig_scatter = px.scatter(
        clean_df,
        x=temperature_col,
        y=rental_col,
        opacity=0.4,
        trendline="ols",
        title="기온과 대여량의 관계"
    )

    st.plotly_chart(
        fig_scatter,
        use_container_width=True
    )

    st.divider()

    st.subheader("③ 시간대별 평균 대여량")

    hourly_avg = (
        clean_df.groupby(hour_col)[rental_col]
        .mean()
        .reset_index()
    )

    fig_hour = px.bar(
        hourly_avg,
        x=hour_col,
        y=rental_col,
        title="시간대별 평균 자전거 대여량"
    )

    st.plotly_chart(
        fig_hour,
        use_container_width=True
    )

    st.divider()

    st.subheader("④ 기온과 시간대에 따른 평균 대여량")

    heat_data = clean_df.copy()

    heat_data["기온 구간"] = pd.cut(
        heat_data[temperature_col],
        bins=[
            -np.inf, 0, 10, 20, 30, np.inf
        ],
        labels=[
            "0℃ 미만",
            "0~10℃",
            "10~20℃",
            "20~30℃",
            "30℃ 이상"
        ],
        right=False
    )

    heat_table = heat_data.pivot_table(
        index="기온 구간",
        columns=hour_col,
        values=rental_col,
        aggfunc="mean",
        observed=False
    )

    fig_heat = px.imshow(
        heat_table,
        text_auto=".0f",
        aspect="auto",
        color_continuous_scale="YlOrRd",
        title="기온 구간과 시간대별 평균 대여량"
    )

    st.plotly_chart(
        fig_heat,
        use_container_width=True
    )


# =====================================
# 11. 데이터 전처리
# =====================================

elif menu == "5. 데이터 전처리":

    st.header("🧹 데이터 전처리")

    st.subheader("① 분석에 사용할 변수")

    st.write("입력 변수(X): 기온, 시간대")
    st.write("예측 대상(y): 자전거 대여량")

    st.write(
        "날씨 분석에서는 습도, 풍속, 가시거리, "
        "강수량 등 데이터에 존재하는 변수를 추가로 활용합니다."
    )

    st.divider()

    st.subheader("② 결측값 확인")

    col1, col2, col3 = st.columns(3)

    col1.metric(
        "전처리 전 데이터",
        f"{len(analysis_df):,}개"
    )

    col2.metric(
        "결측값 포함 행",
        f"{missing_rows:,}개"
    )

    col3.metric(
        "전처리 후 데이터",
        f"{len(clean_df):,}개"
    )

    st.write(
        f"분석 변수에서 확인된 결측값은 총 {missing_count:,}개입니다."
    )

    st.write(
        "기온, 시간대, 대여량 등 분석에 필요한 값이 없는 행을 제외했습니다."
    )

    st.divider()

    st.subheader("③ 이상치 확인")

    selected_outlier_col = st.selectbox(
        "상자그림으로 확인할 변수",
        analysis_cols,
        key="outlier_col"
    )

    fig_outlier = px.box(
        analysis_df,
        y=selected_outlier_col,
        title=f"{selected_outlier_col} 이상치 확인"
    )

    st.plotly_chart(
        fig_outlier,
        use_container_width=True
    )

    st.write(
        "상자그림에서 이상치로 표시된 값이 반드시 잘못된 데이터는 아닙니다. "
        "실제 이용량이나 날씨 상황으로 발생했을 가능성도 있으므로 "
        "오류라는 근거가 부족한 값은 임의로 삭제하지 않고 유지했습니다."
    )

    st.divider()

    st.subheader("④ 학습용 및 테스트용 데이터 분리")

    split_index = int(len(clean_df) * 0.8)

    col1, col2 = st.columns(2)

    col1.metric(
        "학습용 데이터 (80%)",
        f"{split_index:,}개"
    )

    col2.metric(
        "테스트용 데이터 (20%)",
        f"{len(clean_df) - split_index:,}개"
    )

    st.write(
        "시간 순서를 유지하여 앞의 80%는 학습용, "
        "뒤의 20%는 테스트용으로 분리합니다."
    )

    st.subheader("전처리된 데이터")
    st.dataframe(
        clean_df.head(10),
        use_container_width=True
    )


# =====================================
# 12. 모델 학습 및 성능 비교
# =====================================

elif menu == "6. 모델 학습 및 성능 비교":

    st.header("🤖 모델 학습 및 성능 비교")

    st.subheader("① 회귀 분석을 선택한 이유")

    st.write(
        "공공자전거 대여량은 숫자로 표현되는 값이므로 "
        "수치형 값을 예측하는 회귀 분석이 적합합니다."
    )

    st.write("입력 변수(X): 기온, 시간대")
    st.write("예측 대상(y): 자전거 대여량")

    st.divider()

    st.subheader("② 모델 설명")

    model_info = pd.DataFrame({
        "모델": [
            "베이스라인",
            "선형 회귀",
            "의사결정나무 회귀",
            "랜덤 포레스트 회귀"
        ],
        "설명": [
            "학습 데이터의 평균 대여량으로 예측",
            "직선 형태의 관계를 학습",
            "조건에 따라 데이터를 나누어 예측",
            "여러 의사결정나무의 결과를 종합"
        ]
    })

    st.dataframe(
        model_info,
        use_container_width=True,
        hide_index=True
    )

    st.divider()

    st.subheader("③ 모델 학습")

    if st.button("네 가지 모델 학습하기", type="primary"):

        if len(clean_df) < 10:
            st.error("학습할 데이터가 부족합니다.")

        else:
            with st.spinner("모델을 학습하고 있습니다..."):

                result_df, predictions, y_test = train_models(
                    clean_df
                )

                st.session_state["result_df"] = result_df
                st.session_state["predictions"] = predictions
                st.session_state["y_test"] = y_test

            st.success("모델 학습과 평가가 완료되었습니다.")

    if "result_df" in st.session_state:

        result_df = st.session_state["result_df"]
        predictions = st.session_state["predictions"]
        y_test = st.session_state["y_test"]

        st.divider()

        st.subheader("④ 모델 성능 비교")

        st.write(
            "MAE는 실제값과 예측값의 평균적인 차이입니다. "
            "MAE는 낮을수록 좋습니다."
        )

        st.write(
            "R²는 모델이 실제 데이터의 변화를 얼마나 설명하는지 나타냅니다. "
            "일반적으로 높을수록 좋습니다."
        )

        st.dataframe(
            result_df.round(4),
            use_container_width=True,
            hide_index=True
        )

        candidate_df = result_df[
            result_df["모델"] != "베이스라인"
        ]

        best_idx = candidate_df["MAE"].idxmin()

        best_model = candidate_df.loc[
            best_idx, "모델"
        ]

        best_mae = candidate_df.loc[
            best_idx, "MAE"
        ]

        baseline_mae = result_df.loc[
            result_df["모델"] == "베이스라인",
            "MAE"
        ].iloc[0]

        st.success(
            f"MAE가 가장 낮은 후보 모델: {best_model}"
        )

        if best_mae < baseline_mae:
            reduction = (
                (baseline_mae - best_mae)
                / baseline_mae * 100
            )

            st.write(
                f"베이스라인보다 MAE가 {reduction:.2f}% 낮습니다."
            )
        else:
            st.warning(
                "후보 모델 중 베이스라인보다 MAE가 낮은 모델은 없습니다."
            )

        st.divider()

        st.subheader("⑤ MAE 비교 그래프")

        fig_mae = px.bar(
            result_df,
            x="모델",
            y="MAE",
            title="모델별 MAE 비교",
            text_auto=".2f"
        )

        st.plotly_chart(
            fig_mae,
            use_container_width=True
        )

        st.subheader("⑥ R² 비교 그래프")

        fig_r2 = px.bar(
            result_df,
            x="모델",
            y="R²",
            title="모델별 R² 비교",
            text_auto=".3f"
        )

        st.plotly_chart(
            fig_r2,
            use_container_width=True
        )

        st.divider()

        st.subheader("⑦ 실제값과 예측값 비교")

        selected_model = st.selectbox(
            "비교할 모델",
            result_df["모델"].tolist()
        )

        compare_df = pd.DataFrame({
            "실제 대여량": y_test,
            "예측 대여량": predictions[selected_model]
        })

        fig_compare = px.line(
            compare_df.head(100),
            y=["실제 대여량", "예측 대여량"],
            title=f"{selected_model} 실제값과 예측값"
        )

        st.plotly_chart(
            fig_compare,
            use_container_width=True
        )

        st.dataframe(
            compare_df.head(20).round(1),
            use_container_width=True
        )


# =====================================
# 13. 최종 결과
# =====================================

elif menu == "7. 최종 결과":

    st.header("📝 최종 결과")

    if "result_df" not in st.session_state:

        st.warning(
            "먼저 모델 학습 및 성능 비교 메뉴에서 "
            "모델 학습을 실행해 주세요."
        )

    else:

        result_df = st.session_state["result_df"]

        st.subheader("① 모델 성능 비교")

        st.dataframe(
            result_df.round(4),
            use_container_width=True,
            hide_index=True
        )

        candidate_df = result_df[
            result_df["모델"] != "베이스라인"
        ]

        best_idx = candidate_df["MAE"].idxmin()

        best_model = candidate_df.loc[
            best_idx, "모델"
        ]

        best_mae = candidate_df.loc[
            best_idx, "MAE"
        ]

        best_r2 = candidate_df.loc[
            best_idx, "R²"
        ]

        baseline_mae = result_df.loc[
            result_df["모델"] == "베이스라인",
            "MAE"
        ].iloc[0]

        st.divider()

        st.subheader("② 가장 낮은 MAE를 기록한 모델")

        st.metric("모델", best_model)
        st.metric("MAE", f"{best_mae:.2f}")
        st.metric("R²", f"{best_r2:.4f}")

        if best_mae < baseline_mae:

            reduction = (
                (baseline_mae - best_mae)
                / baseline_mae * 100
            )

            st.success(
                f"베이스라인보다 MAE가 {reduction:.2f}% 낮습니다."
            )

        else:

            st.warning(
                "후보 모델 중 베이스라인보다 MAE가 낮은 모델은 없습니다."
            )

        st.divider()

        st.subheader("③ 탐구 결과")

        st.write(
            "이번 탐구에서는 기온과 시간대 정보를 활용하여 "
            "공공자전거 대여량을 분석하고 예측했습니다. "
            "먼저 기초통계량과 그래프를 통해 날씨와 대여량의 분포 및 관계를 "
            "살펴본 뒤, 베이스라인과 세 가지 회귀 모델의 성능을 비교했습니다."
        )

        st.write(
            "MAE와 R²를 기준으로 모델의 예측 성능을 확인하고 "
            "기준 모델보다 예측이 개선되었는지 살펴보았습니다."
        )

        st.divider()

        st.subheader("④ 탐구의 한계와 보완 방법")

        st.write(
            "이번 예측에서는 기온과 시간대 두 변수만 사용했기 때문에 "
            "습도, 풍속, 강수량, 요일 등의 영향을 예측 모델에 반영하지 못했습니다."
        )

        st.write(
            "앞으로 날씨와 시간 관련 변수를 추가하고 "
            "다양한 모델을 비교하면 예측 성능을 개선할 수 있는지 "
            "추가로 탐구할 수 있습니다."
        )


# =====================================
# 14. 사이드바 하단
# =====================================

st.sidebar.divider()
st.sidebar.caption("공공자전거 대여량 데이터 분석 프로젝트")


import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


# ==================================================
# 1. 기본 설정
# ==================================================

st.set_page_config(
    page_title="공공자전거 대여량 예측",
    page_icon="🚲",
    layout="wide"
)

DATA_URL = (
    "https://raw.githubusercontent.com/"
    "t2515atat-png/project/main/data/SeoulBikeData.csv"
)

st.title("🚲 공공자전거 대여량 예측 프로젝트")

st.markdown("""
### 탐구 주제
**기온과 시간대에 따른 공공자전거 대여량 예측**

### 탐구 질문
**기온과 시간대 정보를 활용하면 공공자전거 대여량을 얼마나 정확하게 예측할 수 있을까?**
""")

st.divider()


# ==================================================
# 2. 데이터 불러오기
# ==================================================

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
    st.error(str(e))
    st.stop()


# ==================================================
# 3. 변수 설정 및 데이터 정리
# ==================================================

rental_col = "Rented Bike Count"
temperature_col = "Temperature(°C)"
hour_col = "Hour"

required_cols = [
    rental_col,
    temperature_col,
    hour_col
]

missing_cols = [
    col for col in required_cols
    if col not in df.columns
]

if missing_cols:
    st.error(f"필요한 열이 없습니다: {missing_cols}")
    st.write("현재 데이터 열:", df.columns.tolist())
    st.stop()


numeric_candidates = [
    rental_col,
    temperature_col,
    hour_col,
    "Humidity(%)",
    "Wind speed (m/s)",
    "Visibility (10m)",
    "Dew point temperature(°C)",
    "Solar Radiation (MJ/m2)",
    "Rainfall(mm)",
    "Snowfall (cm)"
]

for col in numeric_candidates:
    if col in df.columns:
        df[col] = pd.to_numeric(
            df[col],
            errors="coerce"
        )


weather_candidates = [
    temperature_col,
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

analysis_df = df.dropna(
    subset=[
        rental_col,
        temperature_col,
        hour_col
    ]
).copy()


# ==================================================
# 4. 사이드바 메뉴
# ==================================================

menu = st.sidebar.radio(
    "📌 분석 메뉴",
    [
        "1. 데이터 소개",
        "2. 기초통계분석",
        "3. 기온과 시간대별 대여량 분석",
        "4. 데이터 속성 간 관계 분석",
        "5. 데이터 전처리",
        "6. 모델 학습 및 성능 비교",
        "7. 최종 결과"
    ]
)


# ==================================================
# 1. 데이터 소개
# ==================================================

if menu == "1. 데이터 소개":

    st.header("📂 데이터 소개")

    st.write("""
    본 탐구에서는 서울 공공자전거 대여량 데이터를 활용합니다.

    데이터에 포함된 기온과 시간대 정보를 분석하여
    공공자전거 대여량이 어떻게 달라지는지 알아보고,
    두 가지 정보를 활용해 대여량을 예측하고자 합니다.
    """)

    col1, col2, col3 = st.columns(3)

    col1.metric("전체 데이터 수", f"{len(df):,}개")
    col2.metric("전체 변수 수", f"{len(df.columns)}개")
    col3.metric("분석에 사용하는 주요 변수", "2개")

    st.subheader("데이터 미리보기")
    st.dataframe(
        df.head(10),
        use_container_width=True
    )

    st.subheader("주요 변수 설명")

    st.markdown("""
    | 변수 | 의미 |
    |---|---|
    | Temperature(°C) | 기온 |
    | Hour | 시간대 |
    | Rented Bike Count | 공공자전거 대여량 |
    """)

    st.subheader("데이터 열 정보")

    info_df = pd.DataFrame({
        "열 이름": df.columns,
        "데이터 유형": df.dtypes.astype(str).values,
        "결측치 수": df.isnull().sum().values
    })

    st.dataframe(
        info_df,
        use_container_width=True
    )


# ==================================================
# 2. 기초통계분석
# ==================================================

elif menu == "2. 기초통계분석":

    st.header("📊 기초통계분석")

    st.write("""
    데이터의 평균, 중앙값, 표준편차, 최솟값, 최댓값 등을
    확인하여 공공자전거 대여량과 기온 및 시간대의
    전반적인 특징을 알아봅니다.
    """)

    st.subheader("① 전체 수치형 데이터의 기초통계량")

    numeric_df = df.select_dtypes(
        include=np.number
    )

    st.dataframe(
        numeric_df.describe().T.round(2),
        use_container_width=True
    )

    st.divider()

    st.subheader("② 주요 변수의 기초통계량")

    main_cols = [
        temperature_col,
        hour_col,
        rental_col
    ]

    st.dataframe(
        df[main_cols].describe().T.round(2),
        use_container_width=True
    )

    st.divider()

    st.subheader("③ 날씨 변수의 기초통계량")

    if weather_cols:
        st.dataframe(
            df[weather_cols].describe().T.round(2),
            use_container_width=True
        )


# ==================================================
# 3. 기온과 시간대별 대여량 분석
# ==================================================

elif menu == "3. 기온과 시간대별 대여량 분석":

    st.header("🌡️ 기온과 시간대별 대여량 분석")

    st.write("""
    이번 탐구에서 가장 중요하게 살펴보는 부분입니다.

    어떤 기온에서 평균 대여량이 가장 많은지,
    어떤 시간대에 평균 대여량이 가장 많은지 확인합니다.
    """)

    # ----------------------------------------------
    # ① 기온 구간별 평균 대여량
    # ----------------------------------------------

    st.subheader("① 어떤 기온에서 대여량이 가장 많을까?")

    temp_data = analysis_df[
        [temperature_col, rental_col]
    ].copy()

    temp_bins = [
        -np.inf, 0, 5, 10, 15, 20, 25, 30, np.inf
    ]

    temp_labels = [
        "0℃ 미만",
        "0~5℃",
        "5~10℃",
        "10~15℃",
        "15~20℃",
        "20~25℃",
        "25~30℃",
        "30℃ 이상"
    ]

    temp_data["기온 구간"] = pd.cut(
        temp_data[temperature_col],
        bins=temp_bins,
        labels=temp_labels,
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

    temp_avg = temp_avg.dropna(
        subset=[rental_col]
    )

    if not temp_avg.empty:

        best_temp = temp_avg.loc[
            temp_avg[rental_col].idxmax()
        ]

        st.metric(
            "평균 대여량이 가장 많은 기온 구간",
            str(best_temp["기온 구간"]),
            f"평균 {best_temp[rental_col]:,.1f}대"
        )

        fig = px.bar(
            temp_avg,
            x="기온 구간",
            y=rental_col,
            text_auto=".0f",
            title="기온 구간별 평균 공공자전거 대여량",
            labels={
                "기온 구간": "기온 구간",
                rental_col: "평균 대여량"
            }
        )

        fig.update_layout(
            xaxis_title="기온",
            yaxis_title="평균 대여량"
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )

        st.dataframe(
            temp_avg.rename(
                columns={rental_col: "평균 대여량"}
            ).round(2),
            use_container_width=True
        )

    st.info("""
    이 결과는 각 기온 구간에 속하는 데이터의 평균 대여량을
    비교한 것입니다. 따라서 특정 기온 한 지점의 대여량이 아니라
    해당 기온 범위에서 나타난 평균적인 이용량을 의미합니다.
    """)

    st.divider()

    # ----------------------------------------------
    # ② 시간대별 평균 대여량
    # ----------------------------------------------

    st.subheader("② 어떤 시간대에 대여량이 가장 많을까?")

    hour_data = analysis_df[
        [hour_col, rental_col]
    ].copy()

    hour_avg = (
        hour_data.groupby(hour_col)[rental_col]
        .mean()
        .reset_index()
        .sort_values(hour_col)
    )

    if not hour_avg.empty:

        best_hour = hour_avg.loc[
            hour_avg[rental_col].idxmax()
        ]

        st.metric(
            "평균 대여량이 가장 많은 시간대",
            f"{int(best_hour[hour_col])}시",
            f"평균 {best_hour[rental_col]:,.1f}대"
        )

        fig = px.bar(
            hour_avg,
            x=hour_col,
            y=rental_col,
            text_auto=".0f",
            title="시간대별 평균 공공자전거 대여량",
            labels={
                hour_col: "시간대",
                rental_col: "평균 대여량"
            }
        )

        fig.update_layout(
            xaxis=dict(
                tickmode="linear",
                dtick=1
            ),
            xaxis_title="시간",
            yaxis_title="평균 대여량"
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )

        st.dataframe(
            hour_avg.rename(
                columns={rental_col: "평균 대여량"}
            ).round(2),
            use_container_width=True
        )

    st.divider()

    # ----------------------------------------------
    # ③ 기온과 시간대를 함께 분석
    # ----------------------------------------------

    st.subheader("③ 기온과 시간대에 따른 대여량")

    heat_data = analysis_df[
        [temperature_col, hour_col, rental_col]
    ].copy()

    heat_data["기온 구간"] = pd.cut(
        heat_data[temperature_col],
        bins=temp_bins,
        labels=temp_labels,
        right=False
    )

    heat_table = heat_data.pivot_table(
        index="기온 구간",
        columns=hour_col,
        values=rental_col,
        aggfunc="mean",
        observed=False
    )

    fig = px.imshow(
        heat_table,
        text_auto=".0f",
        aspect="auto",
        color_continuous_scale="YlOrRd",
        title="기온 구간과 시간대별 평균 대여량"
    )

    fig.update_layout(
        xaxis_title="시간대",
        yaxis_title="기온 구간"
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )

    st.write("""
    히트맵에서 색이 진할수록 평균 대여량이 많은 것을 의미합니다.
    이를 통해 기온과 시간대가 함께 달라질 때 대여량이
    어떻게 달라지는지 확인할 수 있습니다.
    """)


# ==================================================
# 4. 데이터 속성 간 관계 분석
# ==================================================

elif menu == "4. 데이터 속성 간 관계 분석":

    st.header("🔍 데이터 속성 간 관계 분석")

    st.write("""
    기온, 습도, 풍속 등의 날씨 변수와 시간대가
    공공자전거 대여량과 어떤 관계를 가지는지 살펴봅니다.
    """)

    st.subheader("① 변수 간 상관관계")

    corr_cols = list(dict.fromkeys(
        weather_cols + [hour_col, rental_col]
    ))

    corr_data = df[corr_cols].apply(
        pd.to_numeric,
        errors="coerce"
    )

    corr_matrix = corr_data.corr()

    fig = px.imshow(
        corr_matrix,
        text_auto=".2f",
        color_continuous_scale="RdBu",
        zmin=-1,
        zmax=1,
        title="변수 간 상관관계"
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )

    st.write("""
    상관계수는 두 변수 사이의 선형적인 관계를 나타냅니다.
    1에 가까우면 양의 상관관계, -1에 가까우면 음의 상관관계,
    0에 가까우면 선형적인 관계가 약하다는 의미입니다.
    상관관계가 있다고 해서 인과관계가 성립하는 것은 아닙니다.
    """)

    st.divider()

    st.subheader("② 선택한 날씨 변수와 대여량의 관계")

    selected_weather = st.selectbox(
        "날씨 변수를 선택하세요.",
        weather_cols
    )

    weather_data = df[
        [selected_weather, rental_col]
    ].dropna()

    fig = px.scatter(
        weather_data,
        x=selected_weather,
        y=rental_col,
        opacity=0.4,
        title=f"{selected_weather}와 대여량의 관계"
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )

    weather_corr = weather_data[
        selected_weather
    ].corr(weather_data[rental_col])

    st.metric(
        "선택한 변수와 대여량의 상관계수",
        f"{weather_corr:.3f}"
    )

    st.divider()

    st.subheader("③ 기온과 시간대의 관계")

    fig = px.scatter(
        analysis_df,
        x=temperature_col,
        y=rental_col,
        color=hour_col,
        opacity=0.5,
        title="기온과 대여량의 관계 (시간대별 구분)"
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )


# ==================================================
# 5. 데이터 전처리
# ==================================================

elif menu == "5. 데이터 전처리":

    st.header("🧹 데이터 전처리")

    st.write("""
    예측 모델에 사용할 기온과 시간대 데이터를 확인하고,
    결측치와 이상치를 살펴봅니다.
    """)

    st.subheader("① 결측치 확인")

    missing_df = pd.DataFrame({
        "변수": df.columns,
        "결측치 수": df.isnull().sum().values
    })

    st.dataframe(
        missing_df,
        use_container_width=True
    )

    st.divider()

    st.subheader("② 예측에 사용할 변수")

    st.write("""
    - 입력 변수: 기온, 시간대
    - 예측 대상: 공공자전거 대여량
    """)

    model_data = df[
        [temperature_col, hour_col, rental_col]
    ].copy()

    model_data = model_data.replace(
        [np.inf, -np.inf],
        np.nan
    )

    st.write(
        "전처리 전 데이터 수:",
        f"{len(model_data):,}개"
    )

    model_data = model_data.dropna()

    st.write(
        "결측치 제거 후 데이터 수:",
        f"{len(model_data):,}개"
    )

    st.divider()

    st.subheader("③ 대여량 이상치 확인")

    fig = px.box(
        model_data,
        y=rental_col,
        title="공공자전거 대여량 상자그림"
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )

    st.write("""
    상자그림에서 일반적인 범위를 벗어난 값이 나타날 수 있습니다.
    하지만 실제 이용량이 많았던 시간대일 가능성도 있으므로
    이상치라는 이유만으로 무조건 제거하지 않습니다.
    """)

    st.divider()

    st.subheader("④ 최종 전처리 데이터")

    st.dataframe(
        model_data.head(10),
        use_container_width=True
    )


# ==================================================
# 6. 모델 학습 및 성능 비교
# ==================================================

elif menu == "6. 모델 학습 및 성능 비교":

    st.header("🤖 모델 학습 및 성능 비교")

    st.write("""
    기온과 시간대 정보를 입력 변수로 사용하여
    공공자전거 대여량을 예측합니다.

    베이스라인과 세 가지 회귀 모델의 성능을 비교합니다.
    """)

    model_data = df[
        [temperature_col, hour_col, rental_col]
    ].copy()

    model_data = model_data.replace(
        [np.inf, -np.inf],
        np.nan
    ).dropna()

    if len(model_data) < 10:
        st.error("모델 학습에 필요한 데이터가 부족합니다.")
        st.stop()

    X = model_data[
        [temperature_col, hour_col]
    ]

    y = model_data[rental_col]

    # 학습 데이터와 테스트 데이터 분리
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42
    )

    st.subheader("① 학습 데이터와 테스트 데이터")

    col1, col2 = st.columns(2)

    col1.metric(
        "학습 데이터",
        f"{len(X_train):,}개"
    )

    col2.metric(
        "테스트 데이터",
        f"{len(X_test):,}개"
    )

    st.write("""
    전체 데이터를 학습 데이터 80%, 테스트 데이터 20%로 나누었습니다.
    학습 데이터로 모델을 학습하고, 테스트 데이터로
    예측 성능을 확인합니다.
    """)

    st.divider()

    # ----------------------------------------------
    # 베이스라인
    # ----------------------------------------------

    st.subheader("② 베이스라인 모델")

    baseline_value = y_train.mean()

    baseline_pred = np.full(
        len(y_test),
        baseline_value
    )

    st.write(
        f"베이스라인은 학습 데이터의 평균 대여량인 "
        f"**{baseline_value:.1f}대**를 모든 테스트 데이터의 "
        "예측값으로 사용합니다."
    )

    st.write("""
    베이스라인은 복잡한 모델을 사용하지 않았을 때의
    기본적인 예측 성능을 확인하기 위한 기준입니다.
    다른 모델의 성능이 이 기준보다 좋아지는지 비교합니다.
    """)

    st.divider()

    # ----------------------------------------------
    # 모델 학습
    # ----------------------------------------------

    st.subheader("③ 회귀 모델 학습")

    models = {
        "선형 회귀": Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("model", LinearRegression())
        ]),

        "의사결정나무 회귀": Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("model", DecisionTreeRegressor(
                max_depth=8,
                random_state=42
            ))
        ]),

        "랜덤 포레스트 회귀": Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("model", RandomForestRegressor(
                n_estimators=100,
                max_depth=12,
                random_state=42,
                n_jobs=-1
            ))
        ])
    }

    results = []
    predictions = {}

    # 베이스라인 평가
    results.append({
        "모델": "베이스라인",
        "MAE": mean_absolute_error(y_test, baseline_pred),
        "RMSE": np.sqrt(
            mean_squared_error(y_test, baseline_pred)
        ),
        "결정계수(R²)": r2_score(y_test, baseline_pred)
    })

    predictions["베이스라인"] = baseline_pred

    progress = st.progress(0)

    for i, (name, model) in enumerate(models.items()):

        model.fit(X_train, y_train)

        pred = model.predict(X_test)

        mae = mean_absolute_error(
            y_test,
            pred
        )

        rmse = np.sqrt(
            mean_squared_error(y_test, pred)
        )

        r2 = r2_score(
            y_test,
            pred
        )

        results.append({
            "모델": name,
            "MAE": mae,
            "RMSE": rmse,
            "결정계수(R²)": r2
        })

        predictions[name] = pred

        progress.progress((i + 1) / len(models))

    result_df = pd.DataFrame(results)

    st.success("모델 학습이 완료되었습니다.")

    st.divider()

    # ----------------------------------------------
    # 성능 비교
    # ----------------------------------------------

    st.subheader("④ 모델별 성능 비교")

    st.dataframe(
        result_df.round(3),
        use_container_width=True
    )

    st.write("""
    **평가지표 설명**

    - **MAE:** 실제 대여량과 예측 대여량의 평균적인 차이입니다.
      낮을수록 좋습니다.
    - **RMSE:** 예측 오차가 큰 경우 더 큰 영향을 주는 지표입니다.
      낮을수록 좋습니다.
    - **결정계수(R²):** 모델이 실제 대여량의 변화를 얼마나 설명하는지
      나타냅니다. 일반적으로 높을수록 좋습니다.
    """)

    st.divider()

    st.subheader("⑤ MAE 비교")

    fig = px.bar(
        result_df,
        x="모델",
        y="MAE",
        text_auto=".2f",
        title="모델별 평균 절대 오차 비교"
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )

    st.divider()

    st.subheader("⑥ 결정계수(R²) 비교")

    fig = px.bar(
        result_df,
        x="모델",
        y="결정계수(R²)",
        text_auto=".3f",
        title="모델별 결정계수 비교"
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )

    st.divider()

    st.subheader("⑦ 실제 대여량과 예측 대여량 비교")

    selected_model = st.selectbox(
        "비교할 모델을 선택하세요.",
        list(predictions.keys())
    )

    comparison_df = pd.DataFrame({
        "실제 대여량": y_test.values,
        "예측 대여량": predictions[selected_model]
    })

    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            y=comparison_df["실제 대여량"],
            mode="lines",
            name="실제 대여량"
        )
    )

    fig.add_trace(
        go.Scatter(
            y=comparison_df["예측 대여량"],
            mode="lines",
            name="예측 대여량"
        )
    )

    fig.update_layout(
        title=f"{selected_model} 실제값과 예측값 비교",
        xaxis_title="테스트 데이터 순서",
        yaxis_title="공공자전거 대여량"
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )

    st.dataframe(
        comparison_df.head(20).round(2),
        use_container_width=True
    )

    st.divider()

    # ----------------------------------------------
    # 최종 모델 선택
    # ----------------------------------------------

    st.subheader("⑧ 가장 좋은 성능의 모델")

    best_idx = result_df["MAE"].idxmin()
    best_result = result_df.loc[best_idx]

    st.metric(
        "MAE 기준 가장 낮은 오차를 보인 모델",
        best_result["모델"]
    )

    st.write(
        f"MAE: {best_result['MAE']:.2f}"
    )

    st.write(
        f"RMSE: {best_result['RMSE']:.2f}"
    )

    st.write(
        f"결정계수(R²): {best_result['결정계수(R²)']:.3f}"
    )

    st.info("""
    MAE가 가장 낮은 모델을 평균적인 예측 오차가 가장 작은 모델로
    판단했습니다. 결정계수와 RMSE도 함께 확인하여
    모델의 성능을 종합적으로 비교할 수 있습니다.
    """)


# ==================================================
# 7. 최종 결과
# ==================================================

elif menu == "7. 최종 결과":

    st.header("📝 최종 결과")

    st.subheader("탐구 주제")

    st.markdown("""
    **기온과 시간대에 따른 공공자전거 대여량 예측**
    """)

    st.subheader("탐구 질문")

    st.markdown("""
    **기온과 시간대 정보를 활용하면 공공자전거 대여량을
    얼마나 정확하게 예측할 수 있을까?**
    """)

    st.divider()

    st.subheader("탐구 과정")

    st.markdown("""
    1. 공공자전거 대여량 데이터를 수집하였다.
    2. 기초통계분석으로 데이터의 특징을 확인하였다.
    3. 기온별 평균 대여량과 시간대별 평균 대여량을 비교하였다.
    4. 기온과 시간대에 따른 대여량의 관계를 시각화하였다.
    5. 기온과 시간대 데이터를 이용하여 회귀 모델을 학습하였다.
    6. 베이스라인과 세 가지 회귀 모델의 예측 성능을 비교하였다.
    """)

    st.divider()

    st.subheader("탐구를 통해 확인할 내용")

    st.markdown("""
    - 평균 대여량이 가장 높은 기온 구간은 어디인가?
    - 평균 대여량이 가장 높은 시간대는 언제인가?
    - 기온과 시간대에 따라 대여량은 어떻게 달라지는가?
    - 어떤 회귀 모델이 가장 작은 예측 오차를 보이는가?
    """)

    st.divider()

    st.subheader("탐구의 한계")

    st.write("""
    이번 예측에서는 기온과 시간대 두 가지 변수만 사용하였습니다.
    따라서 습도, 강수량, 계절, 휴일 등 다른 요인은
    예측에 반영되지 않았습니다.

    또한 데이터에 나타난 관계가 모든 상황에서 동일하게
    나타난다고 단정할 수는 없습니다.
    """)

    st.info("""
    실제 탐구 결론은 3번 메뉴에서 확인한 기온별·시간대별 대여량과
    6번 메뉴에서 확인한 모델별 성능 결과를 바탕으로 작성합니다.
    """)

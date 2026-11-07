
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


# --------------------------------------------------
# 1. 기본 설정
# --------------------------------------------------

st.set_page_config(
    page_title="공공자전거 대여량 예측",
    page_icon="🚲",
    layout="wide"
)

DATA_URL = (
    "https://raw.githubusercontent.com/"
    "t2515atat-png/project/main/data/SeoulBikeData.csv"
)

st.title("🚲 공공자전거 대여량 데이터 분석")
st.caption("기초통계분석부터 대여량 예측까지")

st.markdown("""
### 탐구 주제
**기온과 시간대에 따른 공공자전거 대여량 예측**

### 탐구 질문
**기온과 시간대 정보를 활용하면 공공자전거 대여량을 얼마나 정확하게 예측할 수 있을까?**
""")

st.divider()


# --------------------------------------------------
# 2. 데이터 불러오기
# --------------------------------------------------

@st.cache_data
def load_data():
    data = pd.read_csv(DATA_URL, encoding="unicode_escape")

    # 열 이름 앞뒤 공백 제거
    data.columns = data.columns.str.strip()

    return data


try:
    df = load_data()
except Exception as e:
    st.error("데이터를 불러오지 못했습니다.")
    st.error(f"오류 내용: {e}")
    st.stop()


# --------------------------------------------------
# 3. 데이터 열 설정
# --------------------------------------------------

rental_col = "Rented Bike Count"
hour_col = "Hour"
temperature_col = "Temperature(°C)"

required_cols = [
    rental_col,
    hour_col,
    temperature_col
]

missing_cols = [
    col for col in required_cols
    if col not in df.columns
]

if missing_cols:
    st.error(f"필요한 데이터 열이 없습니다: {missing_cols}")
    st.write("현재 데이터의 열 이름:", df.columns.tolist())
    st.stop()


# 숫자형으로 변환
numeric_candidates = [
    rental_col,
    hour_col,
    temperature_col,
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
        df[col] = pd.to_numeric(df[col], errors="coerce")


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

analysis_cols = list(dict.fromkeys(
    weather_cols + [hour_col, rental_col]
))

analysis_df = df[analysis_cols].copy()
analysis_df = analysis_df.dropna(
    subset=[rental_col, hour_col, temperature_col]
)


# --------------------------------------------------
# 4. 메뉴
# --------------------------------------------------

menu = st.sidebar.radio(
    "📌 분석 메뉴",
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


# --------------------------------------------------
# 1. 데이터 소개
# --------------------------------------------------

if menu == "1. 데이터 소개":

    st.header("📂 데이터 소개")

    st.write("""
    본 탐구에서는 서울 공공자전거 대여량 데이터를 활용합니다.

    기온과 시간대에 따라 공공자전거 대여량이 어떻게 달라지는지
    분석하고, 두 가지 정보를 이용하여 대여량을 예측하는 것이
    탐구의 목적입니다.
    """)

    col1, col2, col3 = st.columns(3)

    col1.metric("전체 데이터 수", f"{len(df):,}개")
    col2.metric("전체 변수 수", f"{len(df.columns)}개")
    col3.metric("분석에 사용하는 날씨 변수", f"{len(weather_cols)}개")

    st.subheader("데이터 미리보기")
    st.dataframe(df.head(10), use_container_width=True)

    st.subheader("데이터 열 정보")
    column_info = pd.DataFrame({
        "열 이름": df.columns,
        "데이터 유형": df.dtypes.astype(str).values,
        "결측치 수": df.isnull().sum().values
    })

    st.dataframe(column_info, use_container_width=True)

    st.subheader("탐구에 사용하는 주요 변수")

    st.markdown("""
    - **기온:** 공기 온도
    - **시간대:** 자전거를 대여한 시간
    - **대여량:** 해당 시간에 대여된 자전거 수

    예측 모델에서는 기온과 시간대를 입력하고
    공공자전거 대여량을 예측합니다.
    """)


# --------------------------------------------------
# 2. 기초통계분석
# --------------------------------------------------

elif menu == "2. 기초통계분석":

    st.header("📊 기초통계분석")

    st.write("""
    데이터의 평균, 중앙값, 표준편차, 최솟값, 최댓값 등을 확인하여
    공공자전거 대여량과 날씨 데이터의 전반적인 특징을 알아봅니다.
    """)

    st.subheader("① 전체 수치형 데이터의 기초통계량")

    numeric_df = df.select_dtypes(include=np.number)

    st.dataframe(
        numeric_df.describe().T,
        use_container_width=True
    )

    st.divider()

    st.subheader("② 주요 변수의 기초통계량")

    selected_cols = [
        col for col in
        [temperature_col, hour_col, rental_col]
        if col in df.columns
    ]

    st.dataframe(
        df[selected_cols].describe().T,
        use_container_width=True
    )

    st.divider()

    st.subheader("③ 날씨 변수별 기초통계량")

    if weather_cols:
        st.dataframe(
            df[weather_cols].describe().T,
            use_container_width=True
        )

    st.divider()

    st.subheader("④ 기온 구간별 평균 대여량")

    temp_df = analysis_df[
        [temperature_col, rental_col]
    ].dropna().copy()

    temp_df["기온 구간"] = pd.cut(
        temp_df[temperature_col],
        bins=[-np.inf, 0, 10, 20, 30, np.inf],
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
        temp_df.groupby(
            "기온 구간",
            observed=False
        )[rental_col]
        .mean()
        .reset_index()
    )

    fig = px.bar(
        temp_avg,
        x="기온 구간",
        y=rental_col,
        title="기온 구간별 평균 공공자전거 대여량",
        labels={rental_col: "평균 대여량"}
    )

    st.plotly_chart(fig, use_container_width=True)


# --------------------------------------------------
# 3. 데이터 탐색 및 시각화
# --------------------------------------------------

elif menu == "3. 데이터 탐색 및 시각화":

    st.header("📈 데이터 탐색 및 시각화")

    st.write("""
    그래프를 통해 데이터가 어떤 형태로 분포하는지 살펴보고,
    기온과 시간대에 따른 대여량의 특징을 확인합니다.
    """)

    st.subheader("① 공공자전거 대여량 분포")

    fig = px.histogram(
        df,
        x=rental_col,
        nbins=40,
        title="공공자전거 대여량 분포"
    )

    st.plotly_chart(fig, use_container_width=True)

    st.divider()

    st.subheader("② 기온 분포")

    fig = px.histogram(
        df,
        x=temperature_col,
        nbins=30,
        title="기온 데이터 분포"
    )

    st.plotly_chart(fig, use_container_width=True)

    st.divider()

    st.subheader("③ 날씨 변수별 분포")

    selected_weather = st.selectbox(
        "분석할 날씨 변수를 선택하세요.",
        weather_cols
    )

    fig = px.histogram(
        df,
        x=selected_weather,
        nbins=30,
        title=f"{selected_weather} 분포"
    )

    st.plotly_chart(fig, use_container_width=True)

    st.divider()

    st.subheader("④ 기온과 대여량의 관계")

    fig = px.scatter(
        df,
        x=temperature_col,
        y=rental_col,
        opacity=0.4,
        title="기온에 따른 공공자전거 대여량"
    )

    st.plotly_chart(fig, use_container_width=True)

    st.divider()

    st.subheader("⑤ 시간대별 평균 대여량")

    hourly_avg = (
        df.groupby(hour_col)[rental_col]
        .mean()
        .reset_index()
        .sort_values(hour_col)
    )

    fig = px.line(
        hourly_avg,
        x=hour_col,
        y=rental_col,
        markers=True,
        title="시간대별 평균 공공자전거 대여량"
    )

    st.plotly_chart(fig, use_container_width=True)

    st.divider()

    st.subheader("⑥ 날씨 변수와 대여량 비교")

    selected_weather2 = st.selectbox(
        "비교할 날씨 변수를 선택하세요.",
        weather_cols,
        key="weather_select_2"
    )

    fig = px.scatter(
        df,
        x=selected_weather2,
        y=rental_col,
        opacity=0.4,
        title=f"{selected_weather2}와 대여량의 관계"
    )

    st.plotly_chart(fig, use_container_width=True)


# --------------------------------------------------
# 4. 데이터 속성 간 관계 분석
# --------------------------------------------------

elif menu == "4. 데이터 속성 간 관계 분석":

    st.header("🔍 데이터 속성 간 관계 분석")

    st.write("""
    기온, 날씨, 시간대 등의 속성이 공공자전거 대여량과
    어떤 관계를 가지는지 분석합니다.
    """)

    st.subheader("① 날씨 변수와 대여량의 상관관계")

    corr_cols = list(dict.fromkeys(
        weather_cols + [hour_col, rental_col]
    ))

    corr_data = df[corr_cols].copy()

    for col in corr_cols:
        corr_data[col] = pd.to_numeric(
            corr_data[col],
            errors="coerce"
        )

    corr_matrix = corr_data.corr()

    fig = px.imshow(
        corr_matrix,
        text_auto=".2f",
        color_continuous_scale="RdBu",
        zmin=-1,
        zmax=1,
        title="날씨 변수와 대여량의 상관관계"
    )

    st.plotly_chart(fig, use_container_width=True)

    st.write("""
    상관계수는 두 변수 사이의 선형적인 관계를 나타냅니다.
    1에 가까우면 양의 관계, -1에 가까우면 음의 관계,
    0에 가까우면 선형 관계가 약하다는 의미입니다.
    상관관계만으로 인과관계를 판단할 수는 없습니다.
    """)

    st.divider()

    st.subheader("② 기온과 대여량의 관계")

    temp_data = df[
        [temperature_col, rental_col]
    ].dropna()

    temp_corr = temp_data[
        temperature_col
    ].corr(temp_data[rental_col])

    st.metric(
        "기온과 대여량의 상관계수",
        f"{temp_corr:.3f}"
    )

    fig = px.scatter(
        temp_data,
        x=temperature_col,
        y=rental_col,
        opacity=0.4,
        title="기온과 대여량의 관계"
    )

    st.plotly_chart(fig, use_container_width=True)

    st.divider()

    st.subheader("③ 시간대별 평균 대여량")

    hourly_avg = (
        df.groupby(hour_col)[rental_col]
        .mean()
        .reset_index()
        .sort_values(hour_col)
    )

    fig = px.bar(
        hourly_avg,
        x=hour_col,
        y=rental_col,
        title="시간대별 평균 대여량"
    )

    st.plotly_chart(fig, use_container_width=True)

    st.divider()

    st.subheader("④ 기온과 시간대에 따른 평균 대여량")

    heat_data = df[
        [temperature_col, hour_col, rental_col]
    ].dropna().copy()

    heat_data["기온 구간"] = pd.cut(
        heat_data[temperature_col],
        bins=[-np.inf, 0, 10, 20, 30, np.inf],
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

    fig = px.imshow(
        heat_table,
        text_auto=".0f",
        aspect="auto",
        color_continuous_scale="YlOrRd",
        title="기온 구간과 시간대별 평균 대여량"
    )

    st.plotly_chart(fig, use_container_width=True)

    st.write("""
    이 그래프를 통해 같은 시간대라도 기온 구간에 따라
    평균 대여량이 달라지는지 비교할 수 있습니다.
    """)


# --------------------------------------------------
# 5. 데이터 전처리
# --------------------------------------------------

elif menu == "5. 데이터 전처리":

    st.header("🧹 데이터 전처리")

    st.write("""
    모델 학습에 앞서 데이터의 결측치와 변수의 형태를 확인하고,
    예측에 필요한 기온과 시간대 정보를 준비합니다.
    """)

    st.subheader("① 결측치 확인")

    missing = df.isnull().sum()

    missing_df = pd.DataFrame({
        "열 이름": missing.index,
        "결측치 수": missing.values
    })

    st.dataframe(
        missing_df,
        use_container_width=True
    )

    st.divider()

    st.subheader("② 예측에 사용하는 변수")

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

    st.write("전처리 전 데이터 수:", len(model_data))

    model_data = model_data.dropna(
        subset=[temperature_col, hour_col, rental_col]
    )

    st.write("필수 변수의 결측치를 제거한 데이터 수:", len(model_data))

    st.divider()

    st.subheader("③ 이상치 확인")

    fig = px.box(
        model_data,
        y=rental_col,
        title="공공자전거 대여량의 이상치 확인"
    )

    st.plotly_chart(fig, use_container_width=True)

    st.write("""
    상자그림에서 일반적인 범위를 벗어난 값이 나타날 수 있습니다.
    하지만 실제로 대여량이 많았던 시간대일 가능성도 있으므로
    이상치라는 이유만으로 무조건 제거하지 않습니다.
    """)

    st.divider()

    st.subheader("④ 최종 전처리 데이터")

    st.dataframe(
        model_data.head(10),
        use_container_width=True
    )

    st.success(
        "기온과 시간대 정보를 이용한 대여량 예측에 필요한 "
        "데이터를 준비했습니다."
    )


# --------------------------------------------------
# 6. 모델 학습 및 성능 비교
# --------------------------------------------------

elif menu == "6. 모델 학습 및 성능 비교":

    st.header("🤖 모델 학습 및 성능 비교")

    st.write("""
    기온과 시간대 정보를 입력하여 공공자전거 대여량을 예측합니다.
    먼저 베이스라인을 기준으로 삼고, 세 가지 회귀 모델의
    예측 성능을 비교합니다.
    """)

    model_data = df[
        [temperature_col, hour_col, rental_col]
    ].copy()

    for col in [temperature_col, hour_col, rental_col]:
        model_data[col] = pd.to_numeric(
            model_data[col],
            errors="coerce"
        )

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

    # 날짜가 있다면 시간 순서대로 정렬
    if "Date" in df.columns:
        date_values = pd.to_datetime(
            df.loc[model_data.index, "Date"],
            dayfirst=True,
            errors="coerce"
        )

        model_data = model_data.copy()
        model_data["_date"] = date_values

        model_data = model_data.sort_values(
            "_date",
            kind="stable"
        )

        X = model_data[
            [temperature_col, hour_col]
        ]

        y = model_data[rental_col]

    # 시간 순서 기준 80:20 분할
    split_index = int(len(X) * 0.8)

    X_train = X.iloc[:split_index]
    X_test = X.iloc[split_index:]

    y_train = y.iloc[:split_index]
    y_test = y.iloc[split_index:]

    st.subheader("① 학습 데이터와 테스트 데이터")

    col1, col2 = st.columns(2)

    col1.metric("학습 데이터", f"{len(X_train):,}개")
    col2.metric("테스트 데이터", f"{len(X_test):,}개")

    st.write("""
    전체 데이터를 학습 데이터와 테스트 데이터로 나누었습니다.
    학습 데이터로 모델을 학습하고, 테스트 데이터로
    새로운 데이터에 대한 예측 성능을 평가합니다.
    """)

    st.divider()

    st.subheader("② 베이스라인 모델")

    baseline_value = y_train.mean()

    baseline_predictions = np.full(
        len(y_test),
        baseline_value
    )

    baseline_mae = mean_absolute_error(
        y_test,
        baseline_predictions
    )

    baseline_rmse = np.sqrt(
        mean_squared_error(
            y_test,
            baseline_predictions
        )
    )

    baseline_r2 = r2_score(
        y_test,
        baseline_predictions
    )

    st.write(
        f"베이스라인은 학습 데이터의 평균 대여량인 "
        f"**{baseline_value:.1f}대**를 모든 테스트 데이터의 "
        "예측값으로 사용합니다."
    )

    st.write("""
    베이스라인은 복잡한 예측 모델을 사용하지 않았을 때의
    기본 성능을 확인하기 위한 기준입니다.
    다른 모델이 이 기준보다 좋은 성능을 내는지 비교합니다.
    """)

    st.divider()

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

    results = [{
        "모델": "베이스라인",
        "MAE": baseline_mae,
        "RMSE": baseline_rmse,
        "R²": baseline_r2
    }]

    predictions = {
        "베이스라인": baseline_predictions
    }

    progress = st.progress(0)

    for i, (name, model) in enumerate(models.items()):

        model.fit(X_train, y_train)

        pred = model.predict(X_test)

        mae = mean_absolute_error(y_test, pred)

        rmse = np.sqrt(
            mean_squared_error(y_test, pred)
        )

        r2 = r2_score(y_test, pred)

        results.append({
            "모델": name,
            "MAE": mae,
            "RMSE": rmse,
            "R²": r2
        })

        predictions[name] = pred

        progress.progress((i + 1) / len(models))

    result_df = pd.DataFrame(results)

    st.success("모델 학습과 성능 비교가 완료되었습니다.")

    st.divider()

    st.subheader("④ 모델별 성능 비교")

    st.dataframe(
        result_df.round(3),
        use_container_width=True
    )

    st.write("""
    - MAE: 실제 대여량과 예측 대여량의 평균적인 차이입니다.
      낮을수록 예측 오차가 작습니다.
    - RMSE: 예측 오차를 제곱하여 계산한 뒤 제곱근을 구한 값입니다.
      낮을수록 좋습니다.
    - R²: 모델이 실제 대여량의 변화를 얼마나 설명하는지 나타냅니다.
      일반적으로 높을수록 좋습니다.
    """)

    st.divider()

    st.subheader("⑤ MAE 비교 그래프")

    fig = px.bar(
        result_df,
        x="모델",
        y="MAE",
        title="모델별 평균 절대 오차(MAE) 비교",
        text_auto=".2f"
    )

    st.plotly_chart(fig, use_container_width=True)

    st.divider()

    st.subheader("⑥ 실제 대여량과 예측 대여량 비교")

    selected_model = st.selectbox(
        "비교할 모델을 선택하세요.",
        list(predictions.keys())
    )

    comparison_df = pd.DataFrame({
        "실제 대여량": y_test.values,
        "예측 대여량": predictions[selected_model]
    })

    comparison_df = comparison_df.reset_index(drop=True)

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
        title=f"{selected_model}: 실제값과 예측값 비교",
        xaxis_title="테스트 데이터 순서",
        yaxis_title="자전거 대여량"
    )

    st.plotly_chart(fig, use_container_width=True)

    st.dataframe(
        comparison_df.head(20).round(2),
        use_container_width=True
    )

    st.divider()

    best_model = result_df.loc[
        result_df["MAE"].idxmin()
    ]

    st.subheader("⑦ 가장 낮은 예측 오차를 보인 모델")

    st.write(
        f"MAE 기준으로 가장 낮은 오차를 보인 모델은 "
        f"**{best_model['모델']}**입니다."
    )

    st.metric(
        "가장 낮은 MAE",
        f"{best_model['MAE']:.2f}"
    )


# --------------------------------------------------
# 7. 최종 결과
# --------------------------------------------------

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
    1. 서울 공공자전거 대여량 데이터를 수집하였다.
    2. 기초통계분석을 통해 데이터의 전반적인 특징을 확인하였다.
    3. 기온, 시간대, 날씨 변수와 대여량 사이의 관계를 그래프로 분석하였다.
    4. 예측에 필요한 기온과 시간대 데이터를 전처리하였다.
    5. 베이스라인과 세 가지 회귀 모델을 학습하였다.
    6. MAE, RMSE, R²를 이용하여 모델의 예측 성능을 비교하였다.
    """)

    st.divider()

    st.subheader("예측 모델의 의미")

    st.write("""
    이번 탐구에서는 기온과 시간대라는 두 가지 정보를 활용하여
    공공자전거 대여량을 예측하였습니다.

    베이스라인을 기준으로 선형 회귀, 의사결정나무 회귀,
    랜덤 포레스트 회귀의 성능을 비교하여 어떤 모델이
    상대적으로 더 정확하게 대여량을 예측하는지 확인할 수 있습니다.
    """)

    st.info("""
    실제 탐구 결론을 작성할 때는
    '모델 학습 및 성능 비교' 메뉴에 표시된 MAE와 R² 결과를
    확인하여 가장 성능이 좋은 모델과 그 이유를 작성해야 합니다.
    """)

    st.divider()

    st.subheader("탐구의 한계")

    st.write("""
    이번 예측에서는 기온과 시간대만을 입력 변수로 사용하였습니다.
    따라서 습도, 강수량, 계절, 휴일 등 다른 요인은
    예측에 반영되지 않았습니다.

    이후에는 더 다양한 변수를 활용하여 예측 성능이
    어떻게 달라지는지 비교해 볼 수 있습니다.
    """)

    st.caption("공공자전거 대여량 데이터 분석 프로젝트")

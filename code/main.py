
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px

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
    page_title="공공자전거 대여량 분석",
    page_icon="🚲",
    layout="wide"
)

DATA_URL = (
    "https://raw.githubusercontent.com/"
    "t2515atat-png/project/main/data/SeoulBikeData.csv"
)

TARGET = "Rented Bike Count"
TEMP = "Temperature(°C)"
HOUR = "Hour"

st.title("🚲 기온과 시간대에 따른 공공자전거 대여량 예측")
st.caption("기온과 시간대 정보를 활용한 데이터 분석 및 회귀 모델 비교")


# --------------------------------------------------
# 2. 데이터 불러오기 및 정리
# --------------------------------------------------

@st.cache_data
def load_data():
    encodings = ["utf-8-sig", "latin1"]
    separators = [",", ";"]

    for encoding in encodings:
        for separator in separators:
            try:
                data = pd.read_csv(
                    DATA_URL,
                    encoding=encoding,
                    sep=separator
                )

                if len(data.columns) > 2:
                    break
            except Exception:
                continue
        else:
            continue

        if len(data.columns) > 2:
            break
    else:
        raise ValueError("데이터를 불러오지 못했습니다.")

    data.columns = data.columns.str.strip()

    required = [TARGET, TEMP, HOUR]
    missing = [col for col in required if col not in data.columns]

    if missing:
        raise ValueError(f"필요한 열이 없습니다: {missing}")

    for col in [TARGET, TEMP, HOUR]:
        data[col] = pd.to_numeric(data[col], errors="coerce")

    # 날짜와 시간 순서대로 정렬
    if "Date" in data.columns:
        data["_date_sort"] = pd.to_datetime(
            data["Date"],
            dayfirst=True,
            errors="coerce"
        )

        data = data.sort_values(
            ["_date_sort", HOUR],
            kind="stable",
            na_position="last"
        )

    data = data.dropna(subset=[TARGET, TEMP, HOUR])
    data = data.reset_index(drop=True)

    return data


try:
    df = load_data()
except Exception as e:
    st.error(f"데이터를 불러오는 중 오류가 발생했습니다: {e}")
    st.stop()


# --------------------------------------------------
# 3. 공통 분석 함수
# --------------------------------------------------

def make_features(data, improved=False):
    """기존 방식 또는 시간대 주기 변환 방식으로 입력 변수 생성"""

    temperature = data[TEMP].reset_index(drop=True)
    hour = data[HOUR].reset_index(drop=True)

    if improved:
        # 24시간을 주기로 하는 사인·코사인 변환
        return pd.DataFrame({
            TEMP: temperature,
            "Hour_sin": np.sin(2 * np.pi * hour / 24),
            "Hour_cos": np.cos(2 * np.pi * hour / 24)
        })

    return pd.DataFrame({
        TEMP: temperature,
        HOUR: hour
    })


def make_models():
    """비교할 회귀 모델 생성"""

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

    return models


@st.cache_data(show_spinner=False)
def evaluate_models(data, improved=False):
    """시간 순서대로 80:20 분할 후 모델 평가"""

    X = make_features(data, improved)
    y = data[TARGET].reset_index(drop=True)

    split_index = int(len(data) * 0.8)

    X_train = X.iloc[:split_index]
    X_test = X.iloc[split_index:]

    y_train = y.iloc[:split_index]
    y_test = y.iloc[split_index:]

    if len(X_train) == 0 or len(X_test) == 0:
        raise ValueError("학습 데이터 또는 테스트 데이터가 부족합니다.")

    results = []
    predictions = {}

    # 베이스라인: 학습 데이터의 평균값으로 예측
    baseline_pred = np.full(
        len(y_test),
        y_train.mean()
    )

    predictions["베이스라인"] = baseline_pred

    results.append({
        "모델": "베이스라인",
        "MAE": mean_absolute_error(y_test, baseline_pred),
        "RMSE": np.sqrt(mean_squared_error(y_test, baseline_pred)),
        "R²": r2_score(y_test, baseline_pred)
    })

    # 세 가지 회귀 모델 학습
    for name, model in make_models().items():
        model.fit(X_train, y_train)
        pred = model.predict(X_test)

        predictions[name] = pred

        results.append({
            "모델": name,
            "MAE": mean_absolute_error(y_test, pred),
            "RMSE": np.sqrt(mean_squared_error(y_test, pred)),
            "R²": r2_score(y_test, pred)
        })

    result_df = pd.DataFrame(results)

    return result_df, predictions, y_test.reset_index(drop=True)


def show_result_table(result_df):
    st.dataframe(
        result_df.style.format({
            "MAE": "{:.2f}",
            "RMSE": "{:.2f}",
            "R²": "{:.4f}"
        }),
        use_container_width=True,
        hide_index=True
    )


# 기존 방식과 개선 방식을 각각 평가
try:
    original_results, original_predictions, y_test = evaluate_models(
        df, improved=False
    )

    improved_results, improved_predictions, improved_y_test = evaluate_models(
        df, improved=True
    )
except Exception as e:
    st.error(f"모델을 학습하는 중 오류가 발생했습니다: {e}")
    st.stop()


# --------------------------------------------------
# 4. 메뉴
# --------------------------------------------------

menu = st.sidebar.radio(
    "📂 탐구 메뉴",
    [
        "데이터 소개",
        "기초통계분석",
        "기온과 시간대별 대여량 분석",
        "데이터 속성 간 관계 분석",
        "데이터 전처리",
        "모델 학습 및 성능 비교",
        "성능 개선",
        "최종 결과"
    ]
)


# --------------------------------------------------
# 5. 데이터 소개
# --------------------------------------------------

if menu == "데이터 소개":

    st.header("📊 데이터 소개")

    st.write(
        "서울 공공자전거 대여 데이터를 활용하여 "
        "기온과 시간대에 따른 대여량의 특징을 분석하고 "
        "대여량을 예측하였다."
    )

    col1, col2, col3 = st.columns(3)

    col1.metric("전체 데이터 수", f"{len(df):,}개")
    col2.metric("전체 변수 수", f"{len(df.columns)}개")
    col3.metric("예측 대상", "대여량")

    st.subheader("데이터 미리보기")
    st.dataframe(df.drop(columns=["_date_sort"], errors="ignore").head(10),
                 use_container_width=True)

    st.subheader("주요 변수")

    variable_info = pd.DataFrame({
        "변수": [TEMP, HOUR, TARGET],
        "설명": [
            "공공자전거 대여 시점의 기온",
            "공공자전거 대여 시점의 시간",
            "공공자전거 대여량"
        ],
        "역할": ["입력 변수", "입력 변수", "예측 대상"]
    })

    st.table(variable_info)

    st.info(
        "이번 예측 모델은 기온과 시간대 두 가지 정보를 입력받아 "
        "공공자전거 대여량을 예측한다."
    )

    st.caption(f"데이터 출처: {DATA_URL}")


# --------------------------------------------------
# 6. 기초통계분석
# --------------------------------------------------

elif menu == "기초통계분석":

    st.header("📈 기초통계분석")

    st.write(
        "데이터의 전체적인 특징을 파악하기 위해 "
        "평균, 표준편차, 최솟값, 최댓값 등을 확인하였다."
    )

    numeric_df = df.select_dtypes(include=np.number)

    st.subheader("기초통계량")
    st.dataframe(
        numeric_df.describe().T.style.format("{:.2f}"),
        use_container_width=True
    )

    col1, col2, col3 = st.columns(3)

    col1.metric("평균 대여량", f"{df[TARGET].mean():,.1f}대")
    col2.metric("최대 대여량", f"{df[TARGET].max():,.0f}대")
    col3.metric("최소 대여량", f"{df[TARGET].min():,.0f}대")

    st.subheader("대여량 분포")

    fig = px.histogram(
        df,
        x=TARGET,
        nbins=40,
        title="공공자전거 대여량 분포"
    )

    st.plotly_chart(fig, use_container_width=True)

    st.subheader("기온 분포")

    fig = px.histogram(
        df,
        x=TEMP,
        nbins=30,
        title="기온 분포"
    )

    st.plotly_chart(fig, use_container_width=True)


# --------------------------------------------------
# 7. 기온과 시간대별 대여량 분석
# --------------------------------------------------

elif menu == "기온과 시간대별 대여량 분석":

    st.header("🌡️ 기온과 시간대별 대여량 분석")

    st.write(
        "기온 구간과 시간대에 따라 공공자전거 평균 대여량이 "
        "어떻게 달라지는지 분석하였다."
    )

    # 기온 구간 설정
    bins = [-np.inf, 0, 5, 10, 15, 20, 25, 30, np.inf]
    labels = [
        "0℃ 미만",
        "0~5℃",
        "5~10℃",
        "10~15℃",
        "15~20℃",
        "20~25℃",
        "25~30℃",
        "30℃ 이상"
    ]

    temp_data = df.copy()

    temp_data["기온 구간"] = pd.cut(
        temp_data[TEMP],
        bins=bins,
        labels=labels,
        right=False
    )

    temp_avg = (
        temp_data.groupby("기온 구간", observed=False)[TARGET]
        .mean()
        .reset_index()
    )

    st.subheader("① 기온 구간별 평균 대여량")

    fig = px.bar(
        temp_avg,
        x="기온 구간",
        y=TARGET,
        text_auto=".0f",
        title="기온 구간별 평균 공공자전거 대여량"
    )

    st.plotly_chart(fig, use_container_width=True)

    best_temp = temp_avg.loc[temp_avg[TARGET].idxmax()]

    st.metric(
        "평균 대여량이 가장 많은 기온 구간",
        str(best_temp["기온 구간"]),
        f"{best_temp[TARGET]:,.1f}대"
    )

    st.subheader("② 시간대별 평균 대여량")

    hour_avg = (
        df.groupby(HOUR)[TARGET]
        .mean()
        .reset_index()
        .sort_values(HOUR)
    )

    fig = px.line(
        hour_avg,
        x=HOUR,
        y=TARGET,
        markers=True,
        title="시간대별 평균 공공자전거 대여량"
    )

    st.plotly_chart(fig, use_container_width=True)

    best_hour = hour_avg.loc[hour_avg[TARGET].idxmax()]

    st.metric(
        "평균 대여량이 가장 많은 시간대",
        f"{int(best_hour[HOUR])}시",
        f"{best_hour[TARGET]:,.1f}대"
    )

    st.subheader("③ 기온과 시간대에 따른 평균 대여량")

    pivot = temp_data.pivot_table(
        index="기온 구간",
        columns=HOUR,
        values=TARGET,
        aggfunc="mean",
        observed=False
    )

    fig = px.imshow(
        pivot,
        aspect="auto",
        color_continuous_scale="YlOrRd",
        labels={
            "x": "시간대",
            "y": "기온 구간",
            "color": "평균 대여량"
        },
        title="기온과 시간대에 따른 평균 대여량 히트맵"
    )

    st.plotly_chart(fig, use_container_width=True)


# --------------------------------------------------
# 8. 데이터 속성 간 관계 분석
# --------------------------------------------------

elif menu == "데이터 속성 간 관계 분석":

    st.header("🔗 데이터 속성 간 관계 분석")

    st.write(
        "수치형 변수 사이의 상관관계를 확인하여 "
        "각 변수들이 서로 어떤 관계를 가지는지 분석하였다."
    )

    numeric_df = df.select_dtypes(include=np.number)
    corr = numeric_df.corr()

    st.subheader("상관관계 히트맵")

    fig = px.imshow(
        corr,
        text_auto=".2f",
        aspect="auto",
        color_continuous_scale="RdBu_r",
        zmin=-1,
        zmax=1,
        title="수치형 변수 간 상관관계"
    )

    st.plotly_chart(fig, use_container_width=True)

    st.subheader("기온과 대여량의 관계")

    fig = px.scatter(
        df,
        x=TEMP,
        y=TARGET,
        opacity=0.3,
        title="기온과 공공자전거 대여량의 관계"
    )

    st.plotly_chart(fig, use_container_width=True)

    st.metric(
        "기온과 대여량의 상관계수",
        f"{df[TEMP].corr(df[TARGET]):.3f}"
    )

    st.subheader("시간대와 대여량의 관계")

    fig = px.scatter(
        df,
        x=HOUR,
        y=TARGET,
        opacity=0.3,
        title="시간대와 공공자전거 대여량의 관계"
    )

    st.plotly_chart(fig, use_container_width=True)

    st.metric(
        "시간대와 대여량의 상관계수",
        f"{df[HOUR].corr(df[TARGET]):.3f}"
    )

    st.caption(
        "상관관계는 두 변수의 관련성을 나타내며, "
        "한 변수가 다른 변수의 원인이라는 뜻은 아니다."
    )


# --------------------------------------------------
# 9. 데이터 전처리
# --------------------------------------------------

elif menu == "데이터 전처리":

    st.header("🧹 데이터 전처리")

    st.write(
        "모델 학습에 앞서 결측치, 중복 데이터, "
        "이상치 및 불필요한 변수를 확인하였다."
    )

    st.subheader("① 결측치 확인")

    missing = df.isnull().sum()

    missing_df = pd.DataFrame({
        "변수": missing.index,
        "결측치 수": missing.values
    })

    st.dataframe(missing_df, use_container_width=True)

    st.subheader("② 중복 데이터 확인")

    duplicate_count = df.duplicated().sum()

    st.metric("중복 데이터 수", f"{duplicate_count:,}개")

    st.subheader("③ 이상치 확인")

    outlier_rows = []

    for col in [TEMP, TARGET]:
        q1 = df[col].quantile(0.25)
        q3 = df[col].quantile(0.75)
        iqr = q3 - q1

        lower = q1 - 1.5 * iqr
        upper = q3 + 1.5 * iqr

        count = ((df[col] < lower) | (df[col] > upper)).sum()

        outlier_rows.append({
            "변수": col,
            "하한": lower,
            "상한": upper,
            "IQR 기준 이상치 수": count
        })

    st.dataframe(
        pd.DataFrame(outlier_rows).style.format({
            "하한": "{:.2f}",
            "상한": "{:.2f}"
        }),
        use_container_width=True
    )

    st.info(
        "IQR 기준으로 이상치가 확인되더라도 실제 기온이나 "
        "이용량의 특수한 상황일 수 있으므로 무조건 삭제하지 않았다. "
        "측정 오류라고 판단할 근거가 부족한 값은 유지하였다."
    )

    st.subheader("④ 불필요한 변수 확인")

    st.write(
        "Functioning Day 변수는 모든 값이 Yes로 동일하므로 "
        "이번 분석에서 구분 정보를 제공하지 않아 모델 입력에서 제외하였다."
    )

    st.write(
        "Date는 데이터의 시간 순서를 정렬하는 데 사용하고, "
        "모델의 입력 변수에는 포함하지 않았다."
    )

    st.success(
        "최종 모델 입력 변수: 기온과 시간대 / 예측 대상: 대여량"
    )


# --------------------------------------------------
# 10. 모델 학습 및 성능 비교
# --------------------------------------------------

elif menu == "모델 학습 및 성능 비교":

    st.header("🤖 모델 학습 및 성능 비교")

    st.write(
        "기온과 시간대를 입력 변수로 사용하여 "
        "공공자전거 대여량을 예측하는 회귀 모델을 학습하였다."
    )

    st.subheader("① 학습 방법")

    st.write(
        f"전체 {len(df):,}개 데이터를 시간 순서대로 정렬한 뒤 "
        "앞의 80%는 학습 데이터, 뒤의 20%는 테스트 데이터로 사용하였다."
    )

    st.write("입력 변수: 기온, 시간대")
    st.write("예측 대상: 공공자전거 대여량")

    st.subheader("② 모델별 성능 비교")

    show_result_table(original_results)

    st.caption(
        "MAE와 RMSE는 낮을수록 좋고, R²는 높을수록 좋다."
    )

    st.subheader("③ MAE 비교")

    fig = px.bar(
        original_results,
        x="모델",
        y="MAE",
        text_auto=".2f",
        title="기존 모델의 MAE 비교"
    )

    st.plotly_chart(fig, use_container_width=True)

    st.subheader("④ R² 비교")

    fig = px.bar(
        original_results,
        x="모델",
        y="R²",
        text_auto=".3f",
        title="기존 모델의 R² 비교"
    )

    st.plotly_chart(fig, use_container_width=True)

    best_original = original_results.loc[
        original_results["MAE"].idxmin()
    ]

    st.subheader("⑤ 기존 모델의 최종 결과")

    st.metric(
        "MAE 기준 가장 좋은 모델",
        best_original["모델"],
        f"MAE {best_original['MAE']:.2f}"
    )

    st.write(
        "MAE가 가장 낮은 모델을 테스트 데이터에서 "
        "평균적인 예측 오차가 가장 작은 모델로 판단하였다."
    )

    st.subheader("⑥ 실제값과 예측값 비교")

    selected_model = st.selectbox(
        "비교할 모델 선택",
        list(original_predictions.keys())
    )

    comparison = pd.DataFrame({
        "실제 대여량": y_test,
        "예측 대여량": original_predictions[selected_model]
    })

    fig = px.scatter(
        comparison,
        x="실제 대여량",
        y="예측 대여량",
        title=f"{selected_model}: 실제값과 예측값 비교"
    )

    st.plotly_chart(fig, use_container_width=True)


# --------------------------------------------------
# 11. 성능 개선
# --------------------------------------------------

elif menu == "성능 개선":

    st.header("🚀 성능 개선")

    st.write(
        "기존 모델은 시간대를 0~23의 숫자로 사용하였다. "
        "하지만 시간은 24시간을 주기로 반복되므로 "
        "23시와 0시가 서로 가까운 시간이라는 특성을 반영할 필요가 있다."
    )

    st.subheader("① 개선 방법: 시간대의 주기적 변환")

    st.write("기존 입력 변수")
    st.code("Temperature(°C), Hour")

    st.write("개선된 입력 변수")
    st.code(
        "Temperature(°C), "
        "Hour_sin = sin(2π × Hour / 24), "
        "Hour_cos = cos(2π × Hour / 24)"
    )

    st.info(
        "시간대를 사인과 코사인으로 변환하여 "
        "하루 24시간이 반복되는 특성을 모델에 반영하였다."
    )

    st.subheader("② 개선 전후 모델 성능")

    st.write("기존 모델 성능")
    show_result_table(original_results)

    st.write("개선 모델 성능")
    show_result_table(improved_results)

    st.subheader("③ MAE 개선 전후 비교")

    mae_comparison = pd.DataFrame({
        "모델": original_results["모델"],
        "기존 MAE": original_results["MAE"],
        "개선 후 MAE": improved_results["MAE"]
    })

    mae_chart = mae_comparison.melt(
        id_vars="모델",
        var_name="구분",
        value_name="MAE"
    )

    fig = px.bar(
        mae_chart,
        x="모델",
        y="MAE",
        color="구분",
        barmode="group",
        title="모델별 개선 전후 MAE 비교",
        text_auto=".2f"
    )

    st.plotly_chart(fig, use_container_width=True)

    st.subheader("④ 개선 결과 비교표")

    comparison_rows = []

    for _, old_row in original_results.iterrows():
        new_row = improved_results[
            improved_results["모델"] == old_row["모델"]
        ].iloc[0]

        mae_change = (
            (old_row["MAE"] - new_row["MAE"])
            / old_row["MAE"] * 100
            if old_row["MAE"] != 0 else 0
        )

        comparison_rows.append({
            "모델": old_row["모델"],
            "기존 MAE": old_row["MAE"],
            "개선 후 MAE": new_row["MAE"],
            "MAE 개선율(%)": mae_change,
            "기존 R²": old_row["R²"],
            "개선 후 R²": new_row["R²"]
        })

    comparison_df = pd.DataFrame(comparison_rows)

    st.dataframe(
        comparison_df.style.format({
            "기존 MAE": "{:.2f}",
            "개선 후 MAE": "{:.2f}",
            "MAE 개선율(%)": "{:.2f}",
            "기존 R²": "{:.4f}",
            "개선 후 R²": "{:.4f}"
        }),
        use_container_width=True,
        hide_index=True
    )

    st.caption(
        "MAE 개선율이 양수이면 기존보다 오차가 감소한 것이고, "
        "음수이면 오차가 증가한 것이다."
    )

    st.subheader("⑤ 성능 개선 결과 해석")

    best_improved = improved_results.loc[
        improved_results["MAE"].idxmin()
    ]

    st.metric(
        "개선 후 MAE 기준 가장 좋은 모델",
        best_improved["모델"],
        f"MAE {best_improved['MAE']:.2f}"
    )

    best_old_mae = original_results.set_index("모델")["MAE"]
    best_new_mae = improved_results.set_index("모델")["MAE"]

    changed_models = []

    for model_name in best_old_mae.index:
        old_value = best_old_mae[model_name]
        new_value = best_new_mae[model_name]

        if new_value < old_value:
            changed_models.append(f"{model_name}: 오차 감소")
        elif new_value > old_value:
            changed_models.append(f"{model_name}: 오차 증가")
        else:
            changed_models.append(f"{model_name}: 변화 없음")

    for item in changed_models:
        st.write("• " + item)

    st.warning(
        "시간대 변환이 모든 모델의 성능을 향상시키는 것은 아니다. "
        "실제 MAE, RMSE, R² 결과를 바탕으로 개선 여부를 판단해야 한다."
    )


# --------------------------------------------------
# 12. 최종 결과
# --------------------------------------------------

elif menu == "최종 결과":

    st.header("🏁 최종 결과")

    st.subheader("① 기온과 시간대에 따른 대여량")

    bins = [-np.inf, 0, 5, 10, 15, 20, 25, 30, np.inf]
    labels = [
        "0℃ 미만", "0~5℃", "5~10℃", "10~15℃",
        "15~20℃", "20~25℃", "25~30℃", "30℃ 이상"
    ]

    temp_data = df.copy()

    temp_data["기온 구간"] = pd.cut(
        temp_data[TEMP],
        bins=bins,
        labels=labels,
        right=False
    )

    temp_avg = (
        temp_data.groupby("기온 구간", observed=False)[TARGET]
        .mean()
        .reset_index()
    )

    hour_avg = (
        df.groupby(HOUR)[TARGET]
        .mean()
        .reset_index()
    )

    best_temp = temp_avg.loc[temp_avg[TARGET].idxmax()]
    best_hour = hour_avg.loc[hour_avg[TARGET].idxmax()]

    col1, col2 = st.columns(2)

    col1.metric(
        "대여량이 가장 많은 기온 구간",
        str(best_temp["기온 구간"])
    )

    col2.metric(
        "대여량이 가장 많은 시간대",
        f"{int(best_hour[HOUR])}시"
    )

    st.subheader("② 기존 모델과 개선 모델 비교")

    show_result_table(original_results)

    st.write("시간대 주기 변환을 적용한 모델")
    show_result_table(improved_results)

    best_final = improved_results.loc[
        improved_results["MAE"].idxmin()
    ]

    st.subheader("③ 최종 예측 모델")

    st.metric(
        "개선 후 MAE 기준 최종 모델",
        best_final["모델"],
        f"MAE {best_final['MAE']:.2f}"
    )

    st.write(
        f"개선된 입력 변수를 사용했을 때 MAE가 가장 낮은 모델은 "
        f"{best_final['모델']}이다. "
        "다만 성능 개선 여부는 기존 모델의 결과와 함께 비교하여 "
        "판단해야 한다."
    )

    st.subheader("④ 느낀점")

    st.write(
        "기온과 시간대에 따라 공공자전거 대여량이 달라지는 것을 확인하고, "
        "여러 회귀 모델의 성능을 비교하면서 데이터 분석에서 "
        "모델의 선택과 평가 조건이 중요하다는 것을 느꼈다."
    )

    st.subheader("⑤ 탐구의 한계")

    st.write(
        "이번 탐구에서는 기온과 시간대만을 이용하여 대여량을 예측했기 때문에 "
        "강수량, 습도, 계절 등 다른 요인의 영향을 충분히 반영하지 못했다. "
        "또한 시간대 변환이 모든 모델의 성능을 향상시키는 것은 아니므로 "
        "다른 입력 변수와 전처리 방법을 추가로 비교할 필요가 있다."
    )

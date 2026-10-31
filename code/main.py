
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px

from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score

# ==========================================
# 1. 페이지 설정
# ==========================================

st.set_page_config(
    page_title="공공자전거 대여량 예측",
    page_icon="🚲",
    layout="wide"
)

st.title("🚲 공공자전거 대여량 예측 프로젝트")
st.write(
    "기온과 시간대 정보를 활용하여 "
    "공공자전거 대여량을 예측하고 여러 회귀 모델의 성능을 비교합니다."
)

# ==========================================
# 2. 데이터 불러오기
# ==========================================

DATA_URL = (
    "https://raw.githubusercontent.com/"
    "t2515atat-png/project/main/data/SeoulBikeData.csv"
)

@st.cache_data
def load_data():
    data = pd.read_csv(
        DATA_URL,
        sep=None,
        engine="python",
        encoding="unicode_escape"
    )

    data.columns = data.columns.str.strip()
    data = data.loc[:, ~data.columns.duplicated()]

    return data


try:
    df = load_data()
except Exception as e:
    st.error("데이터를 불러오지 못했습니다.")
    st.write(e)
    st.stop()


# ==========================================
# 3. 분석에 필요한 변수
# ==========================================

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
    st.error(f"필요한 열을 찾을 수 없습니다: {missing_cols}")
    st.write("현재 데이터 열 이름:", list(df.columns))
    st.stop()

# 숫자형으로 변환
for col in required_cols:
    df[col] = pd.to_numeric(df[col], errors="coerce")


# ==========================================
# 4. 메뉴 구성
# ==========================================

menu = st.sidebar.radio(
    "분석 메뉴",
    [
        "1. 데이터 소개",
        "2. 데이터 탐색 및 시각화",
        "3. 데이터 전처리",
        "4. 모델 학습 및 성능 비교",
        "5. 최종 결과"
    ]
)


# ==========================================
# 5. 데이터 분석용 데이터 준비
# ==========================================

# 분석에 필요한 세 변수만 선택
analysis_df = df[required_cols].copy()

# 결측값이 있는 행 확인
missing_count = analysis_df.isnull().sum().sum()

# 필요한 변수의 결측값이 있는 행만 제외
clean_df = analysis_df.dropna().copy()

# 시간 순서대로 데이터 정렬
# Date 열이 있으면 날짜와 시간 순서로 정렬
if "Date" in df.columns:
    date_info = pd.to_datetime(
        df["Date"],
        dayfirst=True,
        errors="coerce"
    )

    clean_df["Date"] = date_info.loc[clean_df.index]

    clean_df = clean_df.sort_values(
        by=["Date", hour_col]
    )

    clean_df = clean_df.drop(columns=["Date"])

else:
    clean_df = clean_df.sort_index()

clean_df = clean_df.reset_index(drop=True)


# ==========================================
# 6. 모델 학습 및 평가 함수
# ==========================================

@st.cache_resource
def train_models(data):

    X = data[[temperature_col, hour_col]]
    y = data[rental_col]

    # 시간 순서에 따라 앞의 80%는 학습,
    # 뒤의 20%는 테스트에 사용
    split_index = int(len(data) * 0.8)

    X_train = X.iloc[:split_index]
    X_test = X.iloc[split_index:]

    y_train = y.iloc[:split_index]
    y_test = y.iloc[split_index:]

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

    for name, model in models.items():

        # 모델 학습
        model.fit(X_train, y_train)

        # 테스트 데이터 예측
        pred = model.predict(X_test)

        # 성능 평가
        mae = mean_absolute_error(y_test, pred)
        r2 = r2_score(y_test, pred)

        results.append({
            "모델": name,
            "MAE": mae,
            "R²": r2
        })

        predictions[name] = pred

    result_df = pd.DataFrame(results)

    return (
        result_df,
        predictions,
        y_test.reset_index(drop=True),
        X_train,
        X_test,
        models
    )


# ==========================================
# 7. 1. 데이터 소개
# ==========================================

if menu == "1. 데이터 소개":

    st.header("📋 데이터 소개")

    st.write(
        "서울 공공자전거 대여량 데이터를 활용하여 "
        "기온과 시간대에 따른 대여량의 특징을 분석합니다."
    )

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric("전체 데이터 수", f"{len(df):,}개")

    with col2:
        st.metric("전체 속성 수", f"{len(df.columns)}개")

    with col3:
        st.metric(
            "분석에 사용할 속성 수",
            f"{len(required_cols)}개"
        )

    st.subheader("데이터 미리보기")
    st.dataframe(df.head(10), use_container_width=True)

    st.subheader("분석에 사용할 주요 데이터")

    variable_info = pd.DataFrame({
        "속성": [
            "Temperature(°C)",
            "Hour",
            "Rented Bike Count"
        ],
        "데이터 유형": [
            "수치형(연속형)",
            "수치형(이산형)",
            "수치형(이산형)"
        ],
        "역할": [
            "입력 변수",
            "입력 변수",
            "예측 대상"
        ]
    })

    st.dataframe(variable_info, use_container_width=True)

    st.subheader("결측값 확인")
    st.dataframe(
        df.isnull().sum().to_frame("결측값 개수"),
        use_container_width=True
    )


# ==========================================
# 8. 2. 데이터 탐색 및 시각화
# ==========================================

elif menu == "2. 데이터 탐색 및 시각화":

    st.header("📊 데이터 탐색 및 시각화")

    st.subheader("① 기초통계량")

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

    stats = stats.drop(columns=["50%"], errors="ignore")

    st.dataframe(
        stats.round(2),
        use_container_width=True
    )

    st.divider()

    st.subheader("② 데이터 분포 확인")

    selected_col = st.selectbox(
        "분포를 확인할 속성을 선택하세요.",
        required_cols
    )

    fig_hist = px.histogram(
        df,
        x=selected_col,
        nbins=30,
        title=f"{selected_col} 분포"
    )

    st.plotly_chart(fig_hist, use_container_width=True)

    fig_box = px.box(
        df,
        y=selected_col,
        title=f"{selected_col} 상자그림"
    )

    st.plotly_chart(fig_box, use_container_width=True)

    st.divider()

    st.subheader("③ 기온과 자전거 대여량의 관계")

    fig_scatter = px.scatter(
        clean_df,
        x=temperature_col,
        y=rental_col,
        opacity=0.45,
        title="기온과 자전거 대여량의 관계"
    )

    st.plotly_chart(
        fig_scatter,
        use_container_width=True
    )

    corr = clean_df[temperature_col].corr(
        clean_df[rental_col]
    )

    st.metric(
        "기온과 대여량의 피어슨 상관계수",
        f"{corr:.3f}"
    )

    st.divider()

    st.subheader("④ 시간대별 평균 자전거 대여량")

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


# ==========================================
# 9. 3. 데이터 전처리
# ==========================================

elif menu == "3. 데이터 전처리":

    st.header("🧹 데이터 전처리")

    st.write(
        "모델 학습에 필요한 변수를 선택하고 "
        "결측값을 확인한 뒤 학습용과 테스트용 데이터로 나눕니다."
    )

    st.subheader("① 분석 변수 선택")

    st.dataframe(
        pd.DataFrame({
            "변수": required_cols,
            "역할": [
                "입력 변수(X)",
                "입력 변수(X)",
                "예측 대상(y)"
            ]
        }),
        use_container_width=True
    )

    st.write(
        "기온과 시간대를 입력 변수로 사용하고 "
        "자전거 대여량을 예측 대상으로 설정했습니다."
    )

    st.divider()

    st.subheader("② 결측값 확인")

    col1, col2 = st.columns(2)

    with col1:
        st.metric(
            "전처리 전 데이터 수",
            f"{len(analysis_df):,}개"
        )

    with col2:
        st.metric(
            "결측값이 포함된 행 수",
            f"{analysis_df.isnull().any(axis=1).sum():,}개"
        )

    st.write(
        f"필요한 세 변수에 포함된 결측값은 총 {missing_count:,}개입니다."
    )

    st.write(
        f"결측값이 있는 행을 제외한 데이터는 {len(clean_df):,}개입니다."
    )

    st.divider()

    st.subheader("③ 이상치 처리")

    st.write(
        "앞선 데이터 탐색에서 대여량에 일부 통계적 이상치가 "
        "확인되었지만, 실제 대여 수요에 따른 값일 가능성도 "
        "있으므로 임의로 삭제하지 않고 유지했습니다."
    )

    st.write(
        "따라서 모델 학습에도 이상치를 포함한 데이터를 사용합니다."
    )

    st.divider()

    st.subheader("④ 학습용·테스트용 데이터 분리")

    split_index = int(len(clean_df) * 0.8)

    col1, col2 = st.columns(2)

    with col1:
        st.metric(
            "학습용 데이터",
            f"{split_index:,}개"
        )

    with col2:
        st.metric(
            "테스트용 데이터",
            f"{len(clean_df) - split_index:,}개"
        )

    st.write(
        "시간 순서를 유지하여 앞의 80%는 학습용, "
        "뒤의 20%는 테스트용 데이터로 사용합니다."
    )

    st.subheader("전처리된 데이터 미리보기")

    st.dataframe(
        clean_df.head(10),
        use_container_width=True
    )


# ==========================================
# 10. 4. 모델 학습 및 성능 비교
# ==========================================

elif menu == "4. 모델 학습 및 성능 비교":

    st.header("🤖 모델 학습 및 성능 비교")

    st.write(
        "기온과 시간대를 입력하여 자전거 대여량을 예측하는 "
        "세 가지 회귀 모델을 비교합니다."
    )

    st.subheader("① 회귀 모델을 선택한 이유")

    st.write(
        "자전거 대여량은 숫자로 표현되는 수치형 데이터이므로 "
        "값을 예측하는 회귀 분석이 적합합니다."
    )

    st.write("**입력 변수(X):** 기온, 시간대")
    st.write("**예측 대상(y):** 자전거 대여량")

    st.divider()

    st.subheader("② 선정한 모델 3가지")

    model_info = pd.DataFrame({
        "모델": [
            "선형 회귀",
            "의사결정나무 회귀",
            "랜덤 포레스트 회귀"
        ],
        "작동 방식": [
            "데이터의 직선 관계를 학습하여 예측",
            "조건에 따라 데이터를 여러 구간으로 나누어 예측",
            "여러 의사결정나무의 예측 결과를 평균하여 예측"
        ]
    })

    st.dataframe(
        model_info,
        use_container_width=True,
        hide_index=True
    )

    st.divider()

    st.subheader("③ 모델 학습 실행")

    if len(clean_df) < 10:
        st.error("모델 학습을 진행하기에 데이터가 부족합니다.")

    else:

        if st.button("세 가지 모델 학습하기", type="primary"):

            with st.spinner("모델을 학습하고 있습니다..."):

                result_df, predictions, y_test, X_train, X_test, models = (
                    train_models(clean_df)
                )

                st.session_state["result_df"] = result_df
                st.session_state["predictions"] = predictions
                st.session_state["y_test"] = y_test
                st.session_state["X_test"] = X_test
                st.session_state["models"] = models

            st.success("세 가지 모델의 학습과 평가가 완료되었습니다.")

        if "result_df" in st.session_state:

            result_df = st.session_state["result_df"]
            predictions = st.session_state["predictions"]
            y_test = st.session_state["y_test"]

            st.subheader("④ 모델 성능 비교")

            st.write(
                "MAE는 실제값과 예측값의 평균적인 차이이며, "
                "R²는 모델이 데이터의 변화를 얼마나 설명하는지 나타냅니다."
            )

            st.dataframe(
                result_df.round(4),
                use_container_width=True,
                hide_index=True
            )

            best_idx = result_df["MAE"].idxmin()
            best_model = result_df.loc[best_idx, "모델"]

            st.success(
                f"MAE 기준으로 가장 좋은 성능을 보인 모델은 "
                f"'{best_model}'입니다."
            )

            st.divider()

            st.subheader("⑤ 모델별 성능 비교 그래프")

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

            st.subheader("⑥ 실제 대여량과 예측 대여량 비교")

            selected_model = st.selectbox(
                "비교할 모델을 선택하세요.",
                result_df["모델"].tolist()
            )

            pred = predictions[selected_model]

            compare_df = pd.DataFrame({
                "실제 대여량": y_test,
                "예측 대여량": pred
            })

            fig_compare = px.line(
                compare_df.head(100),
                y=["실제 대여량", "예측 대여량"],
                title=f"{selected_model}: 실제값과 예측값 비교",
                labels={
                    "index": "테스트 데이터 순서",
                    "value": "자전거 대여량(대)",
                    "variable": "구분"
                }
            )

            st.plotly_chart(
                fig_compare,
                use_container_width=True
            )

            st.dataframe(
                compare_df.head(20).round(1),
                use_container_width=True
            )

            st.caption(
                "실제값과 예측값을 비교하기 위해 테스트 데이터 중 "
                "앞의 100개를 그래프로 표시했습니다."
            )

    st.divider()

    st.subheader("모델 성능 평가 기준")

    st.write("""
    - MAE: 실제 대여량과 예측 대여량의 평균적인 차이입니다.
      값이 작을수록 예측이 정확합니다.
    - R²: 실제 데이터의 변화를 모델이 얼마나 설명하는지 나타냅니다.
      일반적으로 1에 가까울수록 좋습니다.
    """)


# ==========================================
# 11. 5. 최종 결과
# ==========================================

elif menu == "5. 최종 결과":

    st.header("📝 최종 결과")

    st.write(
        "세 가지 회귀 모델의 성능을 비교하여 "
        "공공자전거 대여량 예측에 적합한 모델을 확인합니다."
    )

    if "result_df" not in st.session_state:

        st.warning(
            "아직 모델 학습 결과가 없습니다. "
            "'모델 학습 및 성능 비교' 메뉴에서 먼저 모델을 학습해 주세요."
        )

    else:

        result_df = st.session_state["result_df"]

        best_mae_idx = result_df["MAE"].idxmin()
        best_r2_idx = result_df["R²"].idxmax()

        best_mae_model = result_df.loc[
            best_mae_idx, "모델"
        ]

        best_r2_model = result_df.loc[
            best_r2_idx, "모델"
        ]

        st.subheader("① 최종 모델 성능 비교")

        st.dataframe(
            result_df.round(4),
            use_container_width=True,
            hide_index=True
        )

        col1, col2 = st.columns(2)

        with col1:
            st.metric(
                "MAE가 가장 낮은 모델",
                best_mae_model
            )

        with col2:
            st.metric(
                "R²가 가장 높은 모델",
                best_r2_model
            )

        st.divider()

        st.subheader("② 탐구 결과")

        st.write(
            f"세 가지 회귀 모델을 비교한 결과, "
            f"MAE가 가장 낮은 모델은 {best_mae_model}이며, "
            f"R²가 가장 높은 모델은 {best_r2_model}입니다."
        )

        st.write(
            "이를 통해 기온과 시간대가 공공자전거 대여량 예측에 "
            "활용될 수 있음을 확인하고, 모델별 예측 성능의 차이를 "
            "비교할 수 있었습니다."
        )

        st.info(
            "MAE와 R²는 서로 다른 평가 기준이므로 "
            "두 지표의 결과를 함께 살펴보고 모델의 성능을 판단해야 합니다."
        )

        st.subheader("③ 탐구의 한계")

        st.write(
            "이번 탐구에서는 기온과 시간대 두 가지 변수만 사용했습니다. "
            "따라서 날씨, 강수량, 계절, 요일 등 다른 요인을 고려하지 못했습니다. "
            "또한 사용한 데이터와 학습 조건에 따라 모델의 성능이 달라질 수 있습니다."
        )

        st.subheader("④ 보완 방법")

        st.write(
            "추후 습도, 강수량, 계절, 요일 등의 변수를 추가하고 "
            "다양한 모델의 예측 결과를 비교한다면 "
            "대여량을 더 정확하게 예측할 수 있는지 확인할 수 있습니다."
        )


# ==========================================
# 12. 사이드바 하단
# ==========================================

st.sidebar.divider()
st.sidebar.caption("공공자전거 대여량 예측 프로젝트")

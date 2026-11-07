
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
    "기온과 시간대 정보를 활용하여 공공자전거 대여량을 분석하고 "
    "회귀 모델을 이용해 대여량을 예측합니다."
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
        encoding="latin1"
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
# 3. 분석 변수 설정
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
    st.error(f"필요한 열이 없습니다: {missing_cols}")
    st.write("현재 데이터 열 이름:", list(df.columns))
    st.stop()

for col in required_cols:
    df[col] = pd.to_numeric(
        df[col],
        errors="coerce"
    )


# ==========================================
# 4. 분석용 데이터 및 전처리
# ==========================================

analysis_df = df[required_cols].copy()

missing_rows = analysis_df.isnull().any(axis=1).sum()
missing_values = analysis_df.isnull().sum().sum()

clean_df = analysis_df.dropna().copy()

# 날짜가 있다면 날짜와 시간 순서로 정렬
if "Date" in df.columns:

    date_info = pd.to_datetime(
        df["Date"],
        dayfirst=True,
        errors="coerce"
    )

    clean_df["Date"] = date_info.loc[clean_df.index]

    clean_df = clean_df.sort_values(
        ["Date", hour_col]
    )

    clean_df = clean_df.drop(columns=["Date"])

else:
    clean_df = clean_df.sort_index()

clean_df = clean_df.reset_index(drop=True)


# ==========================================
# 5. 메뉴 구성
# ==========================================

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


# ==========================================
# 6. 모델 학습 함수
# ==========================================

def train_models(data):

    X = data[[temperature_col, hour_col]]
    y = data[rental_col]

    # 시간 순서를 유지하여 80:20으로 분리
    split_index = int(len(data) * 0.8)

    X_train = X.iloc[:split_index]
    X_test = X.iloc[split_index:]

    y_train = y.iloc[:split_index]
    y_test = y.iloc[split_index:]

    # 베이스라인: 학습 데이터 평균으로 예측
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

    # 베이스라인 평가
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

    # 회귀 모델 3개 학습
    for name, model in models.items():

        model.fit(X_train, y_train)

        prediction = model.predict(X_test)

        results.append({
            "모델": name,
            "MAE": mean_absolute_error(
                y_test,
                prediction
            ),
            "R²": r2_score(
                y_test,
                prediction
            )
        })

        predictions[name] = prediction

    result_df = pd.DataFrame(results)

    return (
        result_df,
        predictions,
        y_test.reset_index(drop=True)
    )


# ==========================================
# 7. 데이터 소개
# ==========================================

if menu == "1. 데이터 소개":

    st.header("📋 데이터 소개")

    st.write(
        "서울 공공자전거 대여량 데이터를 이용하여 "
        "기온과 시간대에 따른 대여량을 분석합니다."
    )

    col1, col2, col3 = st.columns(3)

    col1.metric("전체 데이터 수", f"{len(df):,}개")
    col2.metric("전체 속성 수", f"{len(df.columns)}개")
    col3.metric("분석 변수 수", "3개")

    st.subheader("데이터 미리보기")

    st.dataframe(
        df.head(10),
        use_container_width=True
    )

    st.subheader("분석에 사용할 주요 변수")

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
            "입력 변수(X)",
            "입력 변수(X)",
            "예측 대상(y)"
        ]
    })

    st.dataframe(
        variable_info,
        use_container_width=True,
        hide_index=True
    )

    st.subheader("전체 데이터의 결측값")

    st.dataframe(
        df.isnull().sum().to_frame("결측값 개수"),
        use_container_width=True
    )


# ==========================================
# 8. 기초통계분석
# ==========================================

elif menu == "2. 기초통계분석":

    st.header("📈 기초통계분석")

    st.write(
        "기온, 시간대, 자전거 대여량의 기본적인 통계량을 확인합니다."
    )

    st.subheader("① 주요 기초통계량")

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

    st.subheader("② 변수별 통계량 확인")

    selected_col = st.selectbox(
        "분석할 변수를 선택하세요.",
        required_cols
    )

    col1, col2, col3 = st.columns(3)

    col1.metric(
        "평균",
        f"{analysis_df[selected_col].mean():,.2f}"
    )

    col2.metric(
        "중앙값",
        f"{analysis_df[selected_col].median():,.2f}"
    )

    col3.metric(
        "표준편차",
        f"{analysis_df[selected_col].std():,.2f}"
    )

    col4, col5, col6 = st.columns(3)

    col4.metric(
        "최솟값",
        f"{analysis_df[selected_col].min():,.2f}"
    )

    col5.metric(
        "최댓값",
        f"{analysis_df[selected_col].max():,.2f}"
    )

    col6.metric(
        "데이터 개수",
        f"{analysis_df[selected_col].count():,}개"
    )

    st.divider()

    st.subheader("③ 기초통계량의 의미")

    st.write("""
    - 평균: 전체 데이터의 합을 데이터 개수로 나눈 값
    - 중앙값: 데이터를 크기순으로 나열했을 때 가운데에 위치하는 값
    - 표준편차: 데이터가 평균을 중심으로 얼마나 퍼져 있는지 나타내는 값
    - 최솟값과 최댓값: 데이터에서 가장 작은 값과 가장 큰 값
    """)


# ==========================================
# 9. 데이터 탐색 및 시각화
# ==========================================

elif menu == "3. 데이터 탐색 및 시각화":

    st.header("📊 데이터 탐색 및 시각화")

    st.subheader("① 히스토그램")

    selected_col = st.selectbox(
        "분포를 확인할 변수",
        required_cols
    )

    fig_hist = px.histogram(
        analysis_df,
        x=selected_col,
        nbins=30,
        title=f"{selected_col} 분포"
    )

    st.plotly_chart(
        fig_hist,
        use_container_width=True
    )

    st.divider()

    st.subheader("② 상자그림")

    fig_box = px.box(
        analysis_df,
        y=selected_col,
        title=f"{selected_col} 상자그림"
    )

    st.plotly_chart(
        fig_box,
        use_container_width=True
    )

    st.divider()

    st.subheader("③ 데이터 분포 해석")

    st.write(
        "히스토그램을 통해 데이터가 어느 구간에 많이 분포하는지 "
        "확인할 수 있으며, 상자그림을 통해 중앙값과 데이터의 퍼짐, "
        "통계적 이상치를 확인할 수 있습니다."
    )


# ==========================================
# 10. 데이터 속성 간 관계 분석
# ==========================================

elif menu == "4. 데이터 속성 간 관계 분석":

    st.header("🔍 데이터 속성 간 관계 분석")

    st.write(
        "기온과 시간대가 자전거 대여량과 어떤 관계를 가지는지 확인합니다."
    )

    st.subheader("① 기온과 대여량의 관계")

    fig_temp = px.scatter(
        clean_df,
        x=temperature_col,
        y=rental_col,
        opacity=0.45,
        title="기온과 자전거 대여량의 관계"
    )

    st.plotly_chart(
        fig_temp,
        use_container_width=True
    )

    corr = clean_df[temperature_col].corr(
        clean_df[rental_col]
    )

    st.metric(
        "기온과 대여량의 피어슨 상관계수",
        f"{corr:.3f}"
    )

    st.write(
        "상관계수는 두 변수 사이의 선형적인 관계를 나타냅니다. "
        "0에 가까우면 선형 관계가 약하고, 1 또는 -1에 가까우면 "
        "선형 관계가 강합니다."
    )

    st.divider()

    st.subheader("② 시간대별 평균 대여량")

    hourly_avg = (
        clean_df.groupby(hour_col)[rental_col]
        .mean()
        .reset_index()
    )

    fig_hour = px.bar(
        hourly_avg,
        x=hour_col,
        y=rental_col,
        title="시간대별 평균 자전거 대여량",
        labels={
            hour_col: "시간",
            rental_col: "평균 대여량"
        }
    )

    st.plotly_chart(
        fig_hour,
        use_container_width=True
    )

    st.write(
        "시간대별 평균 대여량을 비교하여 자전거 이용이 많은 시간과 "
        "적은 시간을 확인할 수 있습니다."
    )


# ==========================================
# 11. 데이터 전처리
# ==========================================

elif menu == "5. 데이터 전처리":

    st.header("🧹 데이터 전처리")

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
        use_container_width=True,
        hide_index=True
    )

    st.write(
        "기온과 시간대를 입력 변수로 사용하고 "
        "자전거 대여량을 예측 대상으로 설정했습니다."
    )

    st.divider()

    st.subheader("② 결측값 처리")

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
        f"분석에 필요한 세 변수에서 확인된 결측값은 "
        f"총 {missing_values:,}개입니다."
    )

    st.write(
        "필요한 변수에 결측값이 포함된 행만 제외했습니다."
    )

    st.divider()

    st.subheader("③ 이상치 처리")

    st.write(
        "대여량에서 통계적 이상치가 확인되었지만, "
        "이상치가 반드시 잘못된 데이터라는 의미는 아닙니다. "
        "실제 수요 증가로 발생했을 가능성도 있으므로 "
        "임의로 제거하지 않고 유지했습니다."
    )

    st.divider()

    st.subheader("④ 학습용 및 테스트용 데이터 분리")

    split_index = int(len(clean_df) * 0.8)

    col1, col2 = st.columns(2)

    col1.metric(
        "학습용 데이터",
        f"{split_index:,}개"
    )

    col2.metric(
        "테스트용 데이터",
        f"{len(clean_df) - split_index:,}개"
    )

    st.write(
        "시간 순서를 유지하여 앞의 80%는 학습용, "
        "뒤의 20%는 테스트용으로 사용합니다."
    )

    st.subheader("전처리된 데이터 미리보기")

    st.dataframe(
        clean_df.head(10),
        use_container_width=True
    )


# ==========================================
# 12. 모델 학습 및 성능 비교
# ==========================================

elif menu == "6. 모델 학습 및 성능 비교":

    st.header("🤖 모델 학습 및 성능 비교")

    st.write(
        "베이스라인과 세 가지 회귀 모델을 동일한 조건에서 비교합니다."
    )

    st.subheader("① 회귀 분석을 선택한 이유")

    st.write(
        "자전거 대여량은 숫자로 표현되는 수치형 데이터이므로 "
        "값을 예측하는 회귀 분석이 적합합니다."
    )

    st.write("입력 변수(X): 기온, 시간대")
    st.write("예측 대상(y): 자전거 대여량")

    st.divider()

    st.subheader("② 모델 후보와 작동 방식")

    model_info = pd.DataFrame({
        "모델": [
            "베이스라인",
            "선형 회귀",
            "의사결정나무 회귀",
            "랜덤 포레스트 회귀"
        ],
        "작동 방식": [
            "학습 데이터의 평균 대여량으로 예측",
            "직선 관계를 학습하여 예측",
            "조건에 따라 데이터를 구간으로 나누어 예측",
            "여러 의사결정나무의 결과를 평균하여 예측"
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

        st.error("모델 학습에 필요한 데이터가 부족합니다.")

    else:

        if st.button("네 가지 모델 학습하기", type="primary"):

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

            st.subheader("④ 모델 성능 비교표")

            st.write(
                "MAE는 실제값과 예측값의 평균적인 차이이며, "
                "R²는 모델이 실제 데이터의 변화를 얼마나 설명하는지 나타냅니다."
            )

            st.dataframe(
                result_df.round(4),
                use_container_width=True,
                hide_index=True
            )

            baseline_mae = result_df.loc[
                result_df["모델"] == "베이스라인",
                "MAE"
            ].iloc[0]

            candidate_df = result_df[
                result_df["모델"] != "베이스라인"
            ]

            best_idx = candidate_df["MAE"].idxmin()
            best_model = candidate_df.loc[best_idx, "모델"]
            best_mae = candidate_df.loc[best_idx, "MAE"]

            st.success(
                f"MAE가 가장 낮은 후보 모델은 {best_model}입니다."
            )

            if best_mae < baseline_mae:

                reduction = (
                    (baseline_mae - best_mae)
                    / baseline_mae * 100
                )

                st.write(
                    f"베이스라인보다 MAE가 "
                    f"{reduction:.2f}% 낮습니다."
                )

            else:

                st.warning(
                    "현재 후보 모델 중 베이스라인보다 "
                    "MAE가 낮은 모델은 없습니다."
                )

            st.divider()

            st.subheader("⑤ MAE 비교")

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

            st.caption(
                "MAE는 낮을수록 실제 대여량과 예측 대여량의 차이가 작습니다."
            )

            st.subheader("⑥ R² 비교")

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

            st.caption(
                "R²는 일반적으로 높을수록 모델이 데이터의 변화를 잘 설명합니다."
            )

            st.divider()

            st.subheader("⑦ 실제 대여량과 예측 대여량 비교")

            selected_model = st.selectbox(
                "비교할 모델을 선택하세요.",
                result_df["모델"].tolist()
            )

            compare_df = pd.DataFrame({
                "실제 대여량": y_test,
                "예측 대여량": predictions[selected_model]
            })

            fig_compare = px.line(
                compare_df.head(100),
                y=["실제 대여량", "예측 대여량"],
                title=f"{selected_model} 실제값과 예측값 비교"
            )

            st.plotly_chart(
                fig_compare,
                use_container_width=True
            )

            st.dataframe(
                compare_df.head(20).round(1),
                use_container_width=True
            )


# ==========================================
# 13. 최종 결과
# ==========================================

elif menu == "7. 최종 결과":

    st.header("📝 최종 결과")

    if "result_df" not in st.session_state:

        st.warning(
            "먼저 모델 학습 및 성능 비교 메뉴에서 "
            "네 가지 모델을 학습해 주세요."
        )

    else:

        result_df = st.session_state["result_df"]

        st.subheader("① 모델 성능 비교")

        st.dataframe(
            result_df.round(4),
            use_container_width=True,
            hide_index=True
        )

        baseline_mae = result_df.loc[
            result_df["모델"] == "베이스라인",
            "MAE"
        ].iloc[0]

        candidate_df = result_df[
            result_df["모델"] != "베이스라인"
        ]

        best_idx = candidate_df["MAE"].idxmin()

        best_model = candidate_df.loc[
            best_idx,
            "모델"
        ]

        best_mae = candidate_df.loc[
            best_idx,
            "MAE"
        ]

        st.divider()

        st.subheader("② 가장 좋은 모델")

        st.metric(
            "MAE 기준 최우수 모델",
            best_model
        )

        st.metric(
            "해당 모델의 MAE",
            f"{best_mae:.2f}"
        )

        st.metric(
            "해당 모델의 R²",
            f"{candidate_df.loc[best_idx, 'R²']:.4f}"
        )

        if best_mae < baseline_mae:

            reduction = (
                (baseline_mae - best_mae)
                / baseline_mae * 100
            )

            st.success(
                f"{best_model}은 베이스라인보다 "
                f"MAE가 {reduction:.2f}% 낮습니다."
            )

        else:

            st.warning(
                "세 후보 모델 중 MAE가 가장 낮은 모델도 "
                "베이스라인보다 성능이 좋지 않습니다."
            )

        st.divider()

        st.subheader("③ 탐구 결과")

        st.write(
            "이번 탐구에서는 기온과 시간대를 입력 변수로 사용하여 "
            "공공자전거 대여량을 예측했습니다. "
            "베이스라인과 세 가지 회귀 모델을 같은 조건에서 비교하여 "
            "모델별 예측 성능을 확인했습니다."
        )

        st.write(
            "MAE가 낮을수록 실제 대여량과 예측 대여량의 차이가 작고, "
            "R²가 높을수록 모델이 데이터의 변화를 잘 설명한다고 판단할 수 있습니다."
        )

        st.divider()

        st.subheader("④ 탐구의 한계와 보완 방법")

        st.write(
            "이번 탐구에서는 기온과 시간대 두 변수만 사용했기 때문에 "
            "요일, 계절, 강수량, 습도 등의 다른 요인을 충분히 고려하지 못했습니다."
        )

        st.write(
            "앞으로 다양한 변수를 추가하고 모델의 설정을 조정하여 "
            "예측 성능이 개선되는지 확인할 수 있습니다."
        )


# ==========================================
# 14. 사이드바 하단
# ==========================================

st.sidebar.divider()
st.sidebar.caption("공공자전거 대여량 예측 프로젝트")

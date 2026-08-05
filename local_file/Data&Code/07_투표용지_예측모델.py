"""
투표용지 수요 예측모델 (수업용 단순모형)

입력
----
05_지방선거_읍면동_통합_6to8회.csv

출력
----
07_제8회_주모델_백테스트.csv
08_제8회_주모델_요약.csv
09_제9회_가중모델_예측.csv
10_제9회_가중모델_요약.csv
11_모델_파라미터.csv
12_생성_QA_요약.csv

모형 개요
---------
1) 제8회 주모형
   - 제6회→제7회 읍면동별 당일투표율 변화의 전국 평균과 표준편차를 계산한다.
   - 제8회 기본 예측률 = 제7회 지역 당일투표율 + 제6→7회 전국 평균 변화
   - 95/99/99.9% 준비율 = 기본 예측률 + z × 제6→7회 변화 표준편차
   - 실제 제8회 선거일투표수와 비교하여 부족량과 잔여량을 계산한다.

2) 제9회 가중모형
   - 전국가중모형: 제6→7회 전국 평균 변화 30% + 제7→8회 전국 평균 변화 70%
   - 지역가중모형: 각 지역의 제6→7회 변화 30% + 제7→8회 변화 70%
     (세 선거가 정확히 연결되지 않는 지역은 전국가중모형으로 대체)
   - 제8회 백테스트의 예측오차 평균과 표준편차로 95/99/99.9% 준비율을 계산한다.
   - 제9회 선거인수는 현재 입력자료에 없으므로 제8회 선거인수를 임시 기준으로 사용한다.

주의
----
- 여기서 99%는 과거 지역별 변동이 정규분포와 유사하고 같은 변동성이 반복된다는
  가정 아래 계산한 '지역별 단측 예측상한'이다.
- 전국 모든 지역에서 동시에 부족이 없을 확률이 99%라는 뜻은 아니다.
- 행정구역이 정확히 연결되지 않는 지역은 직전 선거의 구시군/시도/전국 평균을 사용한다.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Dict, Iterable, Tuple

import numpy as np
import pandas as pd


# -----------------------------------------------------------------------------
# 사용자가 조정할 수 있는 설정
# -----------------------------------------------------------------------------
WEIGHT_67 = 0.30  # 제9회 예측 시 제6→7회 변화 가중치
WEIGHT_78 = 0.70  # 제9회 예측 시 제7→8회 변화 가중치

Z_SCORES: Dict[str, float] = {
    "95": 1.645,
    "99": 2.326,
    "99.9": 3.090,
}

KEYS = ["시도명", "구시군명", "읍면동명"]
REQUIRED_COLUMNS = [
    "선거ID",
    "선거명",
    "시도명",
    "구시군명",
    "읍면동명",
    "선거인수_총",
    "선거일투표수",
    "당일투표율",
]


# -----------------------------------------------------------------------------
# 공통 함수
# -----------------------------------------------------------------------------
def find_input_csv(data_dir: Path) -> Path:
    """한글 파일명이 정상인 경우와 압축과정에서 이름이 깨진 경우를 모두 찾는다."""
    preferred = data_dir / "05_지방선거_읍면동_통합_6to8회.csv"
    if preferred.exists():
        return preferred

    candidates = sorted(data_dir.glob("05_*6to8*.csv"))
    if len(candidates) == 1:
        return candidates[0]
    if not candidates:
        raise FileNotFoundError("Data&Code 폴더에서 05번 통합 CSV를 찾지 못했습니다.")
    raise FileNotFoundError(
        "05번 통합 CSV 후보가 여러 개라 자동 선택할 수 없습니다:\n- "
        + "\n- ".join(str(p.name) for p in candidates)
    )


def read_source(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, encoding="utf-8-sig")

    missing = [col for col in REQUIRED_COLUMNS if col not in df.columns]
    if missing:
        raise ValueError("필수 열이 없습니다: " + ", ".join(missing))

    for col in ["선거인수_총", "선거일투표수", "당일투표율"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df["선거회차"] = pd.to_numeric(
        df["선거명"].astype(str).str.extract(r"제(\d+)회", expand=False),
        errors="coerce",
    )

    df = df[
        df["선거회차"].isin([6, 7, 8])
        & df["선거인수_총"].gt(0)
        & df["선거일투표수"].notna()
        & df["당일투표율"].between(0, 1, inclusive="both")
    ].copy()

    duplicate_mask = df.duplicated(KEYS + ["선거회차"], keep=False)
    if duplicate_mask.any():
        duplicated = df.loc[duplicate_mask, KEYS + ["선거회차"]].head(10)
        raise ValueError(
            "동일 지역·선거회차 중복행이 있습니다. 먼저 원자료를 점검하세요.\n"
            + duplicated.to_string(index=False)
        )

    return df


def aggregate_rate(
    frame: pd.DataFrame,
    group_cols: Iterable[str],
    rate_name: str,
) -> pd.DataFrame:
    """단순 평균이 아니라 투표수 합계/선거인수 합계로 집계율을 계산한다."""
    result = (
        frame.groupby(list(group_cols), dropna=False)
        .agg(
            집계선거인수=("선거인수_총", "sum"),
            집계선거일투표수=("선거일투표수", "sum"),
        )
        .reset_index()
    )
    result[rate_name] = (
        result["집계선거일투표수"] / result["집계선거인수"]
    )
    return result[list(group_cols) + [rate_name]]


def add_previous_rate_with_fallback(
    target: pd.DataFrame,
    previous: pd.DataFrame,
    previous_round: int,
) -> pd.DataFrame:
    """직전 선거 지역값을 연결하고, 미연결 지역은 구시군→시도→전국 평균으로 대체한다."""
    result = target.copy()
    prev_label = f"제{previous_round}회"

    exact = previous[KEYS + ["당일투표율"]].rename(
        columns={"당일투표율": "이전선거_당일투표율_정확매칭"}
    )
    result = result.merge(exact, on=KEYS, how="left", validate="one_to_one")

    gu_rate = aggregate_rate(previous, ["시도명", "구시군명"], "구시군_이전투표율")
    sido_rate = aggregate_rate(previous, ["시도명"], "시도_이전투표율")
    national_rate = previous["선거일투표수"].sum() / previous["선거인수_총"].sum()

    result = result.merge(gu_rate, on=["시도명", "구시군명"], how="left")
    result = result.merge(sido_rate, on=["시도명"], how="left")

    result["이전선거_당일투표율_사용"] = result["이전선거_당일투표율_정확매칭"]
    result["이전선거값_출처"] = np.where(
        result["이전선거_당일투표율_정확매칭"].notna(),
        f"{prev_label} 동일 읍면동",
        "",
    )

    masks_and_values = [
        (
            result["이전선거_당일투표율_사용"].isna()
            & result["구시군_이전투표율"].notna(),
            "구시군_이전투표율",
            f"{prev_label} 구시군 평균",
        ),
        (
            result["이전선거_당일투표율_사용"].isna()
            & result["시도_이전투표율"].notna(),
            "시도_이전투표율",
            f"{prev_label} 시도 평균",
        ),
    ]

    for mask, value_col, source_label in masks_and_values:
        result.loc[mask, "이전선거_당일투표율_사용"] = result.loc[mask, value_col]
        result.loc[mask, "이전선거값_출처"] = source_label

    remaining = result["이전선거_당일투표율_사용"].isna()
    result.loc[remaining, "이전선거_당일투표율_사용"] = national_rate
    result.loc[remaining, "이전선거값_출처"] = f"{prev_label} 전국 평균"
    result["직전선거_정확매칭"] = result["이전선거_당일투표율_정확매칭"].notna()

    return result.drop(columns=["구시군_이전투표율", "시도_이전투표율"])


def clip_rate(series: pd.Series) -> pd.Series:
    return series.clip(lower=0.0, upper=1.0)


def ceil_ballots(registered: pd.Series, rate: pd.Series) -> pd.Series:
    values = np.ceil(registered.astype(float) * rate.astype(float))
    return pd.Series(values, index=registered.index, dtype="int64")


def add_preparation_columns(
    frame: pd.DataFrame,
    predicted_rate_col: str,
    registered_col: str,
    prefix: str,
    error_mean: float = 0.0,
    error_sd: float | None = None,
) -> pd.DataFrame:
    """예측률과 안전수준별 준비율·수량을 넓은 표 형식으로 추가한다."""
    result = frame.copy()
    result[f"{prefix}_예측당일투표율"] = clip_rate(result[predicted_rate_col])
    result[f"{prefix}_예측투표용지수"] = ceil_ballots(
        result[registered_col], result[f"{prefix}_예측당일투표율"]
    )

    if error_sd is None:
        return result

    result[f"{prefix}_오차보정기준율"] = clip_rate(
        result[f"{prefix}_예측당일투표율"] + error_mean
    )
    result[f"{prefix}_오차보정기준수량"] = ceil_ballots(
        result[registered_col], result[f"{prefix}_오차보정기준율"]
    )

    for level, z in Z_SCORES.items():
        suffix = level.replace(".", "")
        result[f"{prefix}_안전여유_{suffix}"] = error_mean + z * error_sd
        result[f"{prefix}_준비율_{suffix}"] = clip_rate(
            result[f"{prefix}_예측당일투표율"]
            + result[f"{prefix}_안전여유_{suffix}"]
        )
        result[f"{prefix}_준비수량_{suffix}"] = ceil_ballots(
            result[registered_col], result[f"{prefix}_준비율_{suffix}"]
        )

    return result


def evaluate_preparation(
    frame: pd.DataFrame,
    actual_votes_col: str,
    prepared_cols: Dict[str, str],
    scope_name: str,
) -> pd.DataFrame:
    rows = []
    actual_total = int(frame[actual_votes_col].sum())
    registered_total = int(frame["선거인수_총"].sum())

    for level, prepared_col in prepared_cols.items():
        prepared = frame[prepared_col].astype(int)
        actual = frame[actual_votes_col].astype(int)
        shortage = (actual - prepared).clip(lower=0)
        unused = (prepared - actual).clip(lower=0)

        rows.append(
            {
                "평가범위": scope_name,
                "안전수준": level,
                "지역수": int(len(frame)),
                "전체선거인수": registered_total,
                "실제선거일투표수": actual_total,
                "총준비수량": int(prepared.sum()),
                "전체선거인수대비_준비율": prepared.sum() / registered_total,
                "부족지역수": int((shortage > 0).sum()),
                "충족지역수": int((shortage == 0).sum()),
                "실제충족률": float((shortage == 0).mean()),
                "총부족량": int(shortage.sum()),
                "최대지역부족량": int(shortage.max()),
                "총잔여량": int(unused.sum()),
            }
        )

    return pd.DataFrame(rows)


def summarize_forecast_totals(
    frame: pd.DataFrame,
    model_prefix: str,
    model_name: str,
) -> pd.DataFrame:
    rows = []
    registered_total = int(frame["제9회_가정선거인수"].sum())

    levels = {
        "기본예측": f"{model_prefix}_예측투표용지수",
        "오차평균보정": f"{model_prefix}_오차보정기준수량",
        "95%": f"{model_prefix}_준비수량_95",
        "99%": f"{model_prefix}_준비수량_99",
        "99.9%": f"{model_prefix}_준비수량_999",
    }

    rate_cols = {
        "기본예측": f"{model_prefix}_예측당일투표율",
        "오차평균보정": f"{model_prefix}_오차보정기준율",
        "95%": f"{model_prefix}_준비율_95",
        "99%": f"{model_prefix}_준비율_99",
        "99.9%": f"{model_prefix}_준비율_999",
    }

    for level, count_col in levels.items():
        rows.append(
            {
                "모델명": model_name,
                "안전수준": level,
                "지역수": int(len(frame)),
                "가정선거인수_합계": registered_total,
                "준비수량_합계": int(frame[count_col].sum()),
                "가정선거인수대비_준비율": frame[count_col].sum() / registered_total,
                "지역평균_준비율": float(frame[rate_cols[level]].mean()),
            }
        )

    return pd.DataFrame(rows)


def write_csv(frame: pd.DataFrame, path: Path) -> None:
    frame.to_csv(path, index=False, encoding="utf-8-sig")
    print(f"저장 완료: {path.name} ({len(frame):,}행 × {len(frame.columns):,}열)")


# -----------------------------------------------------------------------------
# 분석 실행
# -----------------------------------------------------------------------------
def main() -> None:
    script_dir = Path(__file__).resolve().parent
    input_path = find_input_csv(script_dir)
    df = read_source(input_path)

    election6 = df[df["선거회차"] == 6].copy()
    election7 = df[df["선거회차"] == 7].copy()
    election8 = df[df["선거회차"] == 8].copy()

    # 세 선거를 지역키로 넓게 펼쳐 변화량을 계산한다.
    rate_pivot = df.pivot(index=KEYS, columns="선거회차", values="당일투표율")

    stable67 = rate_pivot.dropna(subset=[6, 7]).copy()
    stable67["변화_6to7"] = stable67[7] - stable67[6]
    mean_delta67 = float(stable67["변화_6to7"].mean())
    sd_delta67 = float(stable67["변화_6to7"].std(ddof=1))

    stable78 = rate_pivot.dropna(subset=[7, 8]).copy()
    stable78["변화_7to8"] = stable78[8] - stable78[7]
    mean_delta78 = float(stable78["변화_7to8"].mean())
    sd_delta78 = float(stable78["변화_7to8"].std(ddof=1))

    stable678 = rate_pivot.dropna(subset=[6, 7, 8]).copy()
    stable678["변화_6to7"] = stable678[7] - stable678[6]
    stable678["변화_7to8"] = stable678[8] - stable678[7]
    stable678["지역가중_예상변화_8to9"] = (
        WEIGHT_67 * stable678["변화_6to7"]
        + WEIGHT_78 * stable678["변화_7to8"]
    )

    # -------------------------------------------------------------------------
    # 제8회 주모형 백테스트
    # -------------------------------------------------------------------------
    backtest8 = add_previous_rate_with_fallback(election8, election7, previous_round=7)
    backtest8["전국평균변화_6to7"] = mean_delta67
    backtest8["전국변화표준편차_6to7"] = sd_delta67
    backtest8["주모형_예측당일투표율_원시값"] = (
        backtest8["이전선거_당일투표율_사용"] + mean_delta67
    )
    backtest8["주모형_예측당일투표율"] = clip_rate(
        backtest8["주모형_예측당일투표율_원시값"]
    )
    backtest8["주모형_예측투표용지수"] = ceil_ballots(
        backtest8["선거인수_총"], backtest8["주모형_예측당일투표율"]
    )

    for level, z in Z_SCORES.items():
        suffix = level.replace(".", "")
        backtest8[f"주모형_안전여유_{suffix}"] = z * sd_delta67
        backtest8[f"주모형_준비율_{suffix}"] = clip_rate(
            backtest8["주모형_예측당일투표율"]
            + backtest8[f"주모형_안전여유_{suffix}"]
        )
        backtest8[f"주모형_준비수량_{suffix}"] = ceil_ballots(
            backtest8["선거인수_총"], backtest8[f"주모형_준비율_{suffix}"]
        )

    # 실제 제8회 결과와 비교
    backtest8["주모형_예측오차_실제minus예측"] = (
        backtest8["당일투표율"] - backtest8["주모형_예측당일투표율"]
    )

    prep_columns8 = {
        "기본예측": "주모형_예측투표용지수",
        "95%": "주모형_준비수량_95",
        "99%": "주모형_준비수량_99",
        "99.9%": "주모형_준비수량_999",
    }

    for label, prepared_col in prep_columns8.items():
        suffix = {
            "기본예측": "기본",
            "95%": "95",
            "99%": "99",
            "99.9%": "999",
        }[label]
        backtest8[f"부족수량_{suffix}"] = (
            backtest8["선거일투표수"] - backtest8[prepared_col]
        ).clip(lower=0).astype(int)
        backtest8[f"잔여수량_{suffix}"] = (
            backtest8[prepared_col] - backtest8["선거일투표수"]
        ).clip(lower=0).astype(int)
        backtest8[f"충족여부_{suffix}"] = backtest8[f"부족수량_{suffix}"].eq(0)

    # 제9회 안전여유에 사용할 제8회 백테스트 오차 통계는 정확매칭 지역만 사용한다.
    exact8 = backtest8[backtest8["직전선거_정확매칭"]].copy()
    error8 = exact8["주모형_예측오차_실제minus예측"].dropna()
    error8_mean = float(error8.mean())
    error8_sd = float(error8.std(ddof=1))

    summary8_all = evaluate_preparation(
        backtest8,
        actual_votes_col="선거일투표수",
        prepared_cols=prep_columns8,
        scope_name="전체 제8회 지역(미연결지역 대체값 포함)",
    )
    summary8_exact = evaluate_preparation(
        exact8,
        actual_votes_col="선거일투표수",
        prepared_cols=prep_columns8,
        scope_name="제7→8회 정확매칭 지역",
    )
    summary8 = pd.concat([summary8_all, summary8_exact], ignore_index=True)

    # 기본 예측 정확도는 안전수준과 무관하므로 요약표 전체 행에 동일하게 표시한다.
    for scope_name, scope_df in [
        ("전체 제8회 지역(미연결지역 대체값 포함)", backtest8),
        ("제7→8회 정확매칭 지역", exact8),
    ]:
        mask = summary8["평가범위"] == scope_name
        absolute_error = (
            scope_df["당일투표율"] - scope_df["주모형_예측당일투표율"]
        ).abs()
        summary8.loc[mask, "기본예측_MAE_pp"] = float(absolute_error.mean() * 100)
        summary8.loc[mask, "기본예측_평균오차_pp"] = float(
            scope_df["주모형_예측오차_실제minus예측"].mean() * 100
        )
        summary8.loc[mask, "기본예측_오차표준편차_pp"] = float(
            scope_df["주모형_예측오차_실제minus예측"].std(ddof=1) * 100
        )

    # -------------------------------------------------------------------------
    # 제9회 가중 예측
    # -------------------------------------------------------------------------
    forecast9 = election8.copy()
    forecast9 = forecast9.rename(
        columns={
            "선거ID": "기준선거ID",
            "선거명": "기준선거명",
            "선거인수_총": "제8회_선거인수",
            "선거일투표수": "제8회_선거일투표수",
            "당일투표율": "제8회_당일투표율",
        }
    )
    forecast9["예측대상선거ID"] = 20260603
    forecast9["예측대상선거명"] = "제9회 전국동시지방선거"
    forecast9["제9회_가정선거인수"] = forecast9["제8회_선거인수"].astype(int)
    forecast9["제9회_선거인수_가정설명"] = "제9회 읍면동별 선거인수 미확보로 제8회 선거인수를 임시 사용"

    national_weighted_change = WEIGHT_67 * mean_delta67 + WEIGHT_78 * mean_delta78
    forecast9["가중치_6to7"] = WEIGHT_67
    forecast9["가중치_7to8"] = WEIGHT_78
    forecast9["전국평균변화_6to7"] = mean_delta67
    forecast9["전국평균변화_7to8"] = mean_delta78
    forecast9["전국가중_예상변화_8to9"] = national_weighted_change
    forecast9["전국가중_예측률_원시값"] = (
        forecast9["제8회_당일투표율"] + national_weighted_change
    )

    # 지역별 변화자료 연결
    local_change = stable678.reset_index()[
        KEYS + ["변화_6to7", "변화_7to8", "지역가중_예상변화_8to9"]
    ]
    forecast9 = forecast9.merge(local_change, on=KEYS, how="left", validate="one_to_one")
    forecast9["지역가중_자료상태"] = np.where(
        forecast9["지역가중_예상변화_8to9"].notna(),
        "제6·7·8회 동일 읍면동 변화 사용",
        "이력 미연결: 전국가중 변화로 대체",
    )
    forecast9["지역가중_예상변화_8to9_사용"] = forecast9[
        "지역가중_예상변화_8to9"
    ].fillna(national_weighted_change)
    forecast9["지역가중_예측률_원시값"] = (
        forecast9["제8회_당일투표율"]
        + forecast9["지역가중_예상변화_8to9_사용"]
    )

    # 제8회 백테스트의 평균오차와 표준편차로 제9회 안전수준 계산
    forecast9["제8회백테스트_평균오차"] = error8_mean
    forecast9["제8회백테스트_오차표준편차"] = error8_sd

    forecast9 = add_preparation_columns(
        forecast9,
        predicted_rate_col="전국가중_예측률_원시값",
        registered_col="제9회_가정선거인수",
        prefix="전국가중모형",
        error_mean=error8_mean,
        error_sd=error8_sd,
    )
    forecast9 = add_preparation_columns(
        forecast9,
        predicted_rate_col="지역가중_예측률_원시값",
        registered_col="제9회_가정선거인수",
        prefix="지역가중모형",
        error_mean=error8_mean,
        error_sd=error8_sd,
    )

    summary9 = pd.concat(
        [
            summarize_forecast_totals(forecast9, "전국가중모형", "전국 평균변화 가중모형"),
            summarize_forecast_totals(forecast9, "지역가중모형", "지역별 변화 가중모형"),
        ],
        ignore_index=True,
    )
    summary9["가중치_6to7"] = WEIGHT_67
    summary9["가중치_7to8"] = WEIGHT_78
    summary9["수량해석주의"] = "제9회 수량은 제8회 선거인수를 사용한 임시 추정치"

    # -------------------------------------------------------------------------
    # 파라미터 및 QA
    # -------------------------------------------------------------------------
    parameters = pd.DataFrame(
        [
            {
                "파라미터": "제6→7회 변화 평균",
                "값": mean_delta67,
                "단위": "비율(1=100%)",
                "설명": "제6·7회 동일 읍면동의 당일투표율 변화 산술평균",
            },
            {
                "파라미터": "제6→7회 변화 표준편차",
                "값": sd_delta67,
                "단위": "비율(1=100%)",
                "설명": "제8회 주모형 안전여유 산정에 사용",
            },
            {
                "파라미터": "제7→8회 변화 평균",
                "값": mean_delta78,
                "단위": "비율(1=100%)",
                "설명": "제9회 가중추세 계산에 사용",
            },
            {
                "파라미터": "제7→8회 변화 표준편차",
                "값": sd_delta78,
                "단위": "비율(1=100%)",
                "설명": "참고용 변동성",
            },
            {
                "파라미터": "제9회 전국가중 예상변화",
                "값": national_weighted_change,
                "단위": "비율(1=100%)",
                "설명": f"제6→7회 {WEIGHT_67:.0%} + 제7→8회 {WEIGHT_78:.0%}",
            },
            {
                "파라미터": "제8회 백테스트 평균오차",
                "값": error8_mean,
                "단위": "비율(1=100%)",
                "설명": "실제 제8회 당일투표율 - 제8회 주모형 예측률",
            },
            {
                "파라미터": "제8회 백테스트 오차 표준편차",
                "값": error8_sd,
                "단위": "비율(1=100%)",
                "설명": "제9회 95/99/99.9% 안전여유 산정에 사용",
            },
            *[
                {
                    "파라미터": f"단측 z값 {level}%",
                    "값": z,
                    "단위": "계수",
                    "설명": f"{level}% 지역별 단측 예측상한",
                }
                for level, z in Z_SCORES.items()
            ],
        ]
    )

    qa = pd.DataFrame(
        [
            {"점검항목": "입력 전체 행수", "값": len(df)},
            {"점검항목": "제6회 지역수", "값": len(election6)},
            {"점검항목": "제7회 지역수", "값": len(election7)},
            {"점검항목": "제8회 지역수", "값": len(election8)},
            {"점검항목": "제6→7회 정확매칭 지역수", "값": len(stable67)},
            {"점검항목": "제7→8회 정확매칭 지역수", "값": len(stable78)},
            {"점검항목": "제6·7·8회 모두 정확매칭 지역수", "값": len(stable678)},
            {
                "점검항목": "제8회 직전선거 대체값 사용 지역수",
                "값": int((~backtest8["직전선거_정확매칭"]).sum()),
            },
            {
                "점검항목": "제9회 지역가중모형 전국값 대체 지역수",
                "값": int(forecast9["지역가중_예상변화_8to9"].isna().sum()),
            },
        ]
    )

    # 보기 쉬운 열 순서: 기존 지역·선거 열을 앞에 두고 새 분석열을 뒤에 둔다.
    backtest_front = [
        "선거ID",
        "선거명",
        "시도명",
        "구시군명",
        "읍면동명",
        "선거인수_총",
        "선거일투표수",
        "당일투표율",
        "이전선거_당일투표율_사용",
        "이전선거값_출처",
        "직전선거_정확매칭",
        "전국평균변화_6to7",
        "전국변화표준편차_6to7",
        "주모형_예측당일투표율",
        "주모형_예측투표용지수",
        "주모형_준비율_95",
        "주모형_준비수량_95",
        "주모형_준비율_99",
        "주모형_준비수량_99",
        "주모형_준비율_999",
        "주모형_준비수량_999",
        "주모형_예측오차_실제minus예측",
        "부족수량_기본",
        "부족수량_95",
        "부족수량_99",
        "부족수량_999",
        "잔여수량_기본",
        "잔여수량_95",
        "잔여수량_99",
        "잔여수량_999",
        "충족여부_기본",
        "충족여부_95",
        "충족여부_99",
        "충족여부_999",
    ]
    backtest_other = [col for col in backtest8.columns if col not in backtest_front]
    backtest8 = backtest8[backtest_front + backtest_other]

    forecast_front = [
        "예측대상선거ID",
        "예측대상선거명",
        "시도명",
        "구시군명",
        "읍면동명",
        "제9회_가정선거인수",
        "제9회_선거인수_가정설명",
        "제8회_당일투표율",
        "가중치_6to7",
        "가중치_7to8",
        "전국평균변화_6to7",
        "전국평균변화_7to8",
        "전국가중_예상변화_8to9",
        "전국가중모형_예측당일투표율",
        "전국가중모형_예측투표용지수",
        "전국가중모형_준비율_95",
        "전국가중모형_준비수량_95",
        "전국가중모형_준비율_99",
        "전국가중모형_준비수량_99",
        "전국가중모형_준비율_999",
        "전국가중모형_준비수량_999",
        "변화_6to7",
        "변화_7to8",
        "지역가중_예상변화_8to9_사용",
        "지역가중_자료상태",
        "지역가중모형_예측당일투표율",
        "지역가중모형_예측투표용지수",
        "지역가중모형_준비율_95",
        "지역가중모형_준비수량_95",
        "지역가중모형_준비율_99",
        "지역가중모형_준비수량_99",
        "지역가중모형_준비율_999",
        "지역가중모형_준비수량_999",
        "제8회백테스트_평균오차",
        "제8회백테스트_오차표준편차",
    ]
    forecast_other = [col for col in forecast9.columns if col not in forecast_front]
    forecast9 = forecast9[forecast_front + forecast_other]

    # 파일 저장
    write_csv(backtest8, script_dir / "07_제8회_주모델_백테스트.csv")
    write_csv(summary8, script_dir / "08_제8회_주모델_요약.csv")
    write_csv(forecast9, script_dir / "09_제9회_가중모델_예측.csv")
    write_csv(summary9, script_dir / "10_제9회_가중모델_요약.csv")
    write_csv(parameters, script_dir / "11_모델_파라미터.csv")
    write_csv(qa, script_dir / "12_생성_QA_요약.csv")

    print("\n핵심 파라미터")
    print(f"- 제6→7회 평균 변화: {mean_delta67 * 100:.3f}%p")
    print(f"- 제6→7회 변화 표준편차: {sd_delta67 * 100:.3f}%p")
    print(f"- 제7→8회 평균 변화: {mean_delta78 * 100:.3f}%p")
    print(f"- 제9회 전국가중 예상 변화: {national_weighted_change * 100:.3f}%p")
    print(f"- 제8회 백테스트 평균오차: {error8_mean * 100:.3f}%p")
    print(f"- 제8회 백테스트 오차 표준편차: {error8_sd * 100:.3f}%p")


if __name__ == "__main__":
    main()

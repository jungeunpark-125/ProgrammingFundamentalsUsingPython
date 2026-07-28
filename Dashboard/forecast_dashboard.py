"""
Ballot Demand Forecast Dashboard

One page, two languages: the ENG/KOR control top-right (common.language_toggle)
switches every label on this page via the STR dict below. Registered as a page
from app.py (the st.navigation router).

Required files (relative to the project root):
    Data&Code/07_제8회_주모델_백테스트.csv
    Data&Code/08_제8회_주모델_요약.csv
    Data&Code/09_제9회_가중모델_예측.csv
    Data&Code/10_제9회_가중모델_요약.csv
    GeoData/skorea-municipalities-2018-topo-simple.json
"""

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

from common import (
    GEOJSON_PATH,
    SAFETY_LEVELS,
    SAFETY_SUFFIX,
    SIDO_EN,
    build_name_maps,
    choropleth_map as shared_choropleth_map,
    format_int,
    get_backtest_columns,
    get_forecast9_columns,
    language_toggle,
    load_geojson,
    resolve_data_file,
)

# The exact filenames below are known-Korean strings; resolving by numeric
# prefix (like the risk dashboard's main dataset) avoids the NFC/NFD Unicode
# filename mismatch that an exact-string path can hit on some filesystems
# (notably macOS), and gives a clear error message if the file simply hasn't
# been added to Data&Code yet.
BACKTEST_PATH = resolve_data_file("07_")
BACKTEST_SUMMARY_PATH = resolve_data_file("08_")
FORECAST9_PATH = resolve_data_file("09_")
FORECAST9_SUMMARY_PATH = resolve_data_file("10_")

STR = {
    "en": dict(
        main_title="Ballot Demand Forecast Dashboard",
        tab_labels=["8th Election Backtest", "9th Election Forecast"],
        sidebar_backtest_header="Backtest",
        safety_label="Safety level",
        map_metric_label="Map metric",
        map_metric_options=["Shortage ballots", "Surplus ballots", "Forecast error (pp)", "Prepared ballots"],
        region_header="Region",
        sido_label="Sido",
        gu_label="Gu / Gun / Si",
        dong_label="Dong",
        kpi_total_prepared="Total prepared",
        kpi_shortage_areas="Shortage areas",
        kpi_shortage_ballots="Shortage ballots",
        kpi_surplus_areas="Surplus areas",
        kpi_surplus_ballots="Surplus ballots",
        kpi_coverage_rate="Coverage rate",
        map_subheader="Municipality map · {safety}",
        map_not_found="Map file not found.",
        comparison_subheader="Safety-level comparison",
        safety_col_label="Safety level",
        prepared_col_label="Prepared ballots",
        detail_subheader="{dong} · detail",
        registered_voters_label="Registered voters",
        actual_demand_label="Actual demand",
        base_forecast_label="Base forecast",
        prepared_template="Prepared · {safety}",
        shortage_label="Shortage",
        surplus_label="Surplus",
        area_list_label="Area list",
        table_mode_options=["Shortage areas", "Largest forecast underestimation", "Largest surplus"],
        forecast_error_col_label="Forecast error (pp)",
        forecast_subheader="9th Election Forecast",
        model_label="Forecast model",
        model_options=["National weighted", "Local weighted"],
        kpi_projected_prepared="Projected prepared",
        kpi_additional_safety="Additional safety ballots",
        kpi_preparation_rate="Preparation rate",
        denominator_caption="* Ballot counts use 8th-election registered voters as the provisional 9th-election denominator.",
        map9_metric_options=["Preparation rate", "Prepared ballots"],
        comparison9_y_label="Projected prepared ballots",
        z_assumed_voters="Assumed registered voters",
        z_forecast_demand="Forecast demand",
        color_labels={
            "shortage": "Shortage ballots",
            "surplus": "Surplus ballots",
            "forecast_error_pp": "Forecast error (pp)",
            "prepared": "Prepared ballots",
            "preparation_rate": "Preparation rate",
            "base_forecast": "Base forecast",
        },
    ),
    "ko": dict(
        main_title="투표용지 수요 예측 대시보드",
        tab_labels=["8회 백테스트", "9회 예측"],
        sidebar_backtest_header="백테스트",
        safety_label="안전 수준",
        map_metric_label="지도 지표",
        map_metric_options=["부족 용지", "잔여 용지", "예측 오차(%p)", "준비 용지"],
        region_header="지역",
        sido_label="시도",
        gu_label="구/군/시",
        dong_label="읍면동",
        kpi_total_prepared="총 준비 수량",
        kpi_shortage_areas="부족 지역 수",
        kpi_shortage_ballots="부족 용지 수",
        kpi_surplus_areas="잔여 지역 수",
        kpi_surplus_ballots="잔여 용지 수",
        kpi_coverage_rate="충족률",
        map_subheader="시군구 지도 · {safety}",
        map_not_found="지도 파일을 찾을 수 없습니다.",
        comparison_subheader="안전 수준별 비교",
        safety_col_label="안전 수준",
        prepared_col_label="준비 수량",
        detail_subheader="{dong} 상세",
        registered_voters_label="선거인수",
        actual_demand_label="실제 수요",
        base_forecast_label="기본 예측",
        prepared_template="준비 수량 · {safety}",
        shortage_label="부족",
        surplus_label="잔여",
        area_list_label="지역 목록",
        table_mode_options=["부족 지역", "예측 과소평가 상위", "잔여 상위"],
        forecast_error_col_label="예측 오차(%p)",
        forecast_subheader="9회 예측",
        model_label="예측 모델",
        model_options=["전국 가중모형", "지역 가중모형"],
        kpi_projected_prepared="예상 준비 수량",
        kpi_additional_safety="추가 안전 용지",
        kpi_preparation_rate="준비율",
        denominator_caption="* 용지 수량은 9회 선거인수가 확정되기 전까지 8회 선거인수를 잠정 분모로 사용합니다.",
        map9_metric_options=["준비율", "준비 수량"],
        comparison9_y_label="예상 준비 수량",
        z_assumed_voters="가정 선거인수",
        z_forecast_demand="예측 수요",
        color_labels={
            "shortage": "부족 용지",
            "surplus": "잔여 용지",
            "forecast_error_pp": "예측 오차(%p)",
            "prepared": "준비 용지",
            "preparation_rate": "준비율",
            "base_forecast": "기본 예측",
        },
    ),
}


@st.cache_data
def load_csv(path):
    return pd.read_csv(path, encoding="utf-8-sig")


def choropleth_map(data_frame, geojson, color_col, hover_data, title=None, labels=None):
    return shared_choropleth_map(
        data_frame,
        geojson,
        color_col,
        hover_data,
        color_continuous_scale="Reds",
        title=title,
        labels=labels,
    )


BACKTEST = load_csv(BACKTEST_PATH)
BACKTEST_SUMMARY = load_csv(BACKTEST_SUMMARY_PATH)
FORECAST9 = load_csv(FORECAST9_PATH)
FORECAST9_SUMMARY = load_csv(FORECAST9_SUMMARY_PATH)
GEOJSON = load_geojson(GEOJSON_PATH)
GU_EN, DONG_EN = build_name_maps(BACKTEST)

lang = language_toggle()
t = STR[lang]

sido_fmt = (lambda x: SIDO_EN.get(x, x)) if lang == "en" else (lambda x: x)
gu_fmt = (lambda x: GU_EN.get(x, x)) if lang == "en" else (lambda x: x)
dong_fmt = (lambda x: DONG_EN.get(x, x)) if lang == "en" else (lambda x: x)

st.title(t["main_title"])

tab8, tab9 = st.tabs(t["tab_labels"])

# =============================================================================
# 8th Election Backtest
# =============================================================================
with tab8:
    sidebar8 = st.sidebar.container()
    with sidebar8:
        st.header(t["sidebar_backtest_header"])
        safety8 = st.select_slider(t["safety_label"], options=SAFETY_LEVELS, value="99%", key="safety8")
        map_metric8 = st.selectbox(t["map_metric_label"], t["map_metric_options"], key="map_metric8")
        st.divider()
        st.subheader(t["region_header"])
        sido_list = sorted(BACKTEST["시도명"].dropna().unique())
        default_sido_idx = sido_list.index("서울특별시") if "서울특별시" in sido_list else 0
        sido8 = st.selectbox(t["sido_label"], sido_list, index=default_sido_idx, format_func=sido_fmt, key="sido8")
        gu_list = sorted(BACKTEST.loc[BACKTEST["시도명"] == sido8, "구시군명"].dropna().unique())
        gu8 = st.selectbox(t["gu_label"], gu_list, format_func=gu_fmt, key="gu8")
        dong_list = sorted(BACKTEST.loc[(BACKTEST["시도명"] == sido8) & (BACKTEST["구시군명"] == gu8), "읍면동명"].dropna().unique())
        dong8 = st.selectbox(t["dong_label"], dong_list, format_func=dong_fmt, key="dong8")

    c8 = get_backtest_columns(safety8)
    d8 = BACKTEST.copy()

    total_areas = len(d8)
    total_prepared = d8[c8["prepared"]].sum()
    shortage_areas = int((d8[c8["shortage"]] > 0).sum())
    shortage_ballots = d8[c8["shortage"]].sum()
    surplus_areas = int((d8[c8["surplus"]] > 0).sum())
    surplus_ballots = d8[c8["surplus"]].sum()
    covered_areas = int(d8[c8["covered"]].fillna(False).astype(bool).sum())
    coverage_rate = covered_areas / total_areas if total_areas else np.nan

    k1, k2, k3, k4, k5, k6 = st.columns(6)
    k1.metric(t["kpi_total_prepared"], format_int(total_prepared))
    k2.metric(t["kpi_shortage_areas"], format_int(shortage_areas))
    k3.metric(t["kpi_shortage_ballots"], format_int(shortage_ballots))
    k4.metric(t["kpi_surplus_areas"], format_int(surplus_areas))
    k5.metric(t["kpi_surplus_ballots"], format_int(surplus_ballots))
    k6.metric(t["kpi_coverage_rate"], f"{coverage_rate * 100:.2f}%")

    st.divider()

    left8, right8 = st.columns([1.25, 1], gap="large")

    with left8:
        gu_summary = (
            d8.groupby(["시도명", "구시군명"], as_index=False)
            .agg(
                registered=("선거인수_총", "sum"),
                actual=("선거일투표수", "sum"),
                predicted=("주모형_예측투표용지수", "sum"),
                prepared=(c8["prepared"], "sum"),
                shortage=(c8["shortage"], "sum"),
                surplus=(c8["surplus"], "sum"),
            )
        )
        gu_summary["forecast_error_pp"] = (
            (gu_summary["actual"] / gu_summary["registered"])
            - (gu_summary["predicted"] / gu_summary["registered"])
        ) * 100
        gu_summary["join_key"] = gu_summary["시도명"] + "_" + gu_summary["구시군명"]

        if lang == "en":
            gu_summary["Sido"] = gu_summary["시도명"].map(SIDO_EN)
            gu_summary["Municipality"] = gu_summary["구시군명"].map(GU_EN)
            region_hover = {"Sido": True, "Municipality": True}
        else:
            region_hover = {"시도명": True, "구시군명": True}

        metric_map8 = dict(zip(t["map_metric_options"], ["shortage", "surplus", "forecast_error_pp", "prepared"]))
        color_col8 = metric_map8[map_metric8]
        fig_map8 = choropleth_map(
            gu_summary,
            GEOJSON,
            color_col8,
            hover_data={
                **region_hover,
                "shortage": ":,.0f",
                "surplus": ":,.0f",
                "forecast_error_pp": ":.2f",
                "prepared": ":,.0f",
                "join_key": False,
            },
            labels=t["color_labels"],
        )
        st.subheader(t["map_subheader"].format(safety=safety8))
        if fig_map8 is not None:
            st.plotly_chart(fig_map8, use_container_width=True)
        else:
            st.info(t["map_not_found"])

    with right8:
        comp = BACKTEST_SUMMARY[
            (BACKTEST_SUMMARY["평가범위"] == "전체 제8회 지역(미연결지역 대체값 포함)")
            & (BACKTEST_SUMMARY["안전수준"].isin(SAFETY_LEVELS))
        ].copy()
        # Treat safety levels as categories, not numeric x-values.
        # This keeps 95%, 99%, and 99.9% equally spaced and removes empty 96-98 ticks.
        comp[t["safety_col_label"]] = comp["안전수준"].astype(str)
        comp = comp.set_index(t["safety_col_label"]).reindex(SAFETY_LEVELS).reset_index()

        st.subheader(t["comparison_subheader"])
        fig_shortage = px.bar(
            comp,
            x=t["safety_col_label"],
            y="부족지역수",
            text="부족지역수",
            category_orders={t["safety_col_label"]: SAFETY_LEVELS},
            labels={t["safety_col_label"]: t["safety_col_label"], "부족지역수": t["kpi_shortage_areas"]},
        )
        fig_shortage.update_traces(textposition="outside")
        fig_shortage.update_xaxes(
            type="category",
            categoryorder="array",
            categoryarray=SAFETY_LEVELS,
            tickmode="array",
            tickvals=SAFETY_LEVELS,
            ticktext=SAFETY_LEVELS,
        )
        fig_shortage.update_layout(height=235, showlegend=False, margin=dict(t=15, b=10))
        st.plotly_chart(fig_shortage, use_container_width=True)

        fig_surplus = px.bar(
            comp,
            x=t["safety_col_label"],
            y="총잔여량",
            text="총잔여량",
            category_orders={t["safety_col_label"]: SAFETY_LEVELS},
            labels={t["safety_col_label"]: t["safety_col_label"], "총잔여량": t["kpi_surplus_ballots"]},
        )
        fig_surplus.update_traces(texttemplate="%{text:,.0f}", textposition="outside")
        fig_surplus.update_xaxes(
            type="category",
            categoryorder="array",
            categoryarray=SAFETY_LEVELS,
            tickmode="array",
            tickvals=SAFETY_LEVELS,
            ticktext=SAFETY_LEVELS,
        )
        fig_surplus.update_layout(height=235, showlegend=False, margin=dict(t=15, b=10))
        st.plotly_chart(fig_surplus, use_container_width=True)

    st.divider()

    detail_left8, detail_right8 = st.columns([1, 1.4], gap="large")
    selected8 = d8[(d8["시도명"] == sido8) & (d8["구시군명"] == gu8) & (d8["읍면동명"] == dong8)]
    dong8_disp = dong_fmt(dong8)

    with detail_left8:
        st.subheader(t["detail_subheader"].format(dong=dong8_disp))
        if not selected8.empty:
            row = selected8.iloc[0]
            a, b, c = st.columns(3)
            a.metric(t["registered_voters_label"], format_int(row["선거인수_총"]))
            b.metric(t["actual_demand_label"], format_int(row["선거일투표수"]))
            c.metric(t["base_forecast_label"], format_int(row["주모형_예측투표용지수"]))

            d, e, f = st.columns(3)
            d.metric(t["prepared_template"].format(safety=safety8), format_int(row[c8["prepared"]]))
            e.metric(t["shortage_label"], format_int(row[c8["shortage"]]))
            f.metric(t["surplus_label"], format_int(row[c8["surplus"]]))

            levels_df = pd.DataFrame({
                t["safety_col_label"]: SAFETY_LEVELS,
                t["prepared_col_label"]: [row[f"주모형_준비수량_{SAFETY_SUFFIX[l]}"] for l in SAFETY_LEVELS],
            })
            fig_detail = px.bar(
                levels_df,
                x=t["safety_col_label"],
                y=t["prepared_col_label"],
                text=t["prepared_col_label"],
                category_orders={t["safety_col_label"]: SAFETY_LEVELS},
            )
            fig_detail.add_hline(y=row["선거일투표수"], line_dash="dash", annotation_text=t["actual_demand_label"])
            fig_detail.update_traces(texttemplate="%{text:,.0f}", textposition="outside")
            fig_detail.update_xaxes(
                type="category",
                categoryorder="array",
                categoryarray=SAFETY_LEVELS,
                tickmode="array",
                tickvals=SAFETY_LEVELS,
                ticktext=SAFETY_LEVELS,
            )
            fig_detail.update_layout(height=330, showlegend=False, margin=dict(t=25, b=10))
            st.plotly_chart(fig_detail, use_container_width=True)

    with detail_right8:
        table_mode8 = st.radio(
            t["area_list_label"],
            t["table_mode_options"],
            horizontal=True,
            key="table_mode8",
        )
        list_df = d8.copy()
        if lang == "en":
            list_df["Sido"] = list_df["시도명"].map(SIDO_EN)
            list_df["Municipality"] = list_df["구시군명"].map(GU_EN)
            list_df["Dong"] = list_df["읍면동명"].map(DONG_EN)
            region_cols = ["Sido", "Municipality", "Dong"]
        else:
            region_cols = ["시도명", "구시군명", "읍면동명"]
        list_df[t["forecast_error_col_label"]] = list_df["주모형_예측오차_실제minus예측"] * 100

        mode_shortage, mode_underest, mode_surplus = t["table_mode_options"]
        if table_mode8 == mode_shortage:
            list_df = list_df[list_df[c8["shortage"]] > 0].sort_values(c8["shortage"], ascending=False)
            show_cols = region_cols + [c8["prepared"], "선거일투표수", c8["shortage"]]
            rename = {
                c8["prepared"]: t["prepared_col_label"],
                "선거일투표수": t["actual_demand_label"],
                c8["shortage"]: t["shortage_label"],
            }
        elif table_mode8 == mode_underest:
            list_df = list_df.sort_values(t["forecast_error_col_label"], ascending=False)
            show_cols = region_cols + ["주모형_예측투표용지수", "선거일투표수", t["forecast_error_col_label"]]
            rename = {
                "주모형_예측투표용지수": t["base_forecast_label"],
                "선거일투표수": t["actual_demand_label"],
            }
        else:
            list_df = list_df[list_df[c8["surplus"]] > 0].sort_values(c8["surplus"], ascending=False)
            show_cols = region_cols + [c8["prepared"], "선거일투표수", c8["surplus"]]
            rename = {
                c8["prepared"]: t["prepared_col_label"],
                "선거일투표수": t["actual_demand_label"],
                c8["surplus"]: t["surplus_label"],
            }

        st.dataframe(
            list_df[show_cols].rename(columns=rename).head(100),
            use_container_width=True,
            height=455,
            hide_index=True,
        )

# =============================================================================
# 9th Election Forecast
# =============================================================================
with tab9:
    st.subheader(t["forecast_subheader"])

    control9a, control9b = st.columns([1, 1])
    with control9a:
        safety9 = st.select_slider(t["safety_label"], options=SAFETY_LEVELS, value="99%", key="safety9")
    with control9b:
        model9 = st.radio(t["model_label"], t["model_options"], horizontal=True, key="model9")

    national_model = model9 == t["model_options"][0]
    c9 = get_forecast9_columns(safety9, national_model)
    d9 = FORECAST9.copy()

    total_assumed_voters9 = d9["제9회_가정선거인수"].sum()
    total_prepared9 = d9[c9["prepared"]].sum()
    total_base9 = d9[c9["forecast_ballots"]].sum()
    prep_rate9 = total_prepared9 / total_assumed_voters9 if total_assumed_voters9 else np.nan
    extra9 = total_prepared9 - total_base9

    q1, q2, q3, q4 = st.columns(4)
    q1.metric(t["kpi_projected_prepared"], format_int(total_prepared9))
    q2.metric(t["base_forecast_label"], format_int(total_base9))
    q3.metric(t["kpi_additional_safety"], format_int(extra9))
    q4.metric(t["kpi_preparation_rate"], f"{prep_rate9 * 100:.2f}%")

    st.caption(t["denominator_caption"])
    st.divider()

    map9_left, map9_right = st.columns([1.25, 1], gap="large")

    with map9_left:
        gu9 = (
            d9.groupby(["시도명", "구시군명"], as_index=False)
            .agg(
                assumed_voters=("제9회_가정선거인수", "sum"),
                prepared=(c9["prepared"], "sum"),
                base_forecast=(c9["forecast_ballots"], "sum"),
            )
        )
        gu9["preparation_rate"] = gu9["prepared"] / gu9["assumed_voters"] * 100
        gu9["join_key"] = gu9["시도명"] + "_" + gu9["구시군명"]

        if lang == "en":
            gu9["Sido"] = gu9["시도명"].map(SIDO_EN)
            gu9["Municipality"] = gu9["구시군명"].map(GU_EN)
            region_hover9 = {"Sido": True, "Municipality": True}
        else:
            region_hover9 = {"시도명": True, "구시군명": True}

        prep_rate_label, prepared_label = t["map9_metric_options"]
        metric9 = st.selectbox(t["map_metric_label"], t["map9_metric_options"], key="map_metric9")
        color9 = "preparation_rate" if metric9 == prep_rate_label else "prepared"
        fig9 = choropleth_map(
            gu9,
            GEOJSON,
            color9,
            hover_data={
                **region_hover9,
                "preparation_rate": ":.2f",
                "prepared": ":,.0f",
                "base_forecast": ":,.0f",
                "join_key": False,
            },
            labels=t["color_labels"],
        )
        if fig9 is not None:
            st.plotly_chart(fig9, use_container_width=True)
        else:
            st.info(t["map_not_found"])

    with map9_right:
        model_name_ko = "전국 평균변화 가중모형" if national_model else "지역별 변화 가중모형"
        comp9 = FORECAST9_SUMMARY[
            (FORECAST9_SUMMARY["모델명"] == model_name_ko)
            & (FORECAST9_SUMMARY["안전수준"].isin(SAFETY_LEVELS))
        ].copy()
        comp9[t["safety_col_label"]] = pd.Categorical(comp9["안전수준"], SAFETY_LEVELS, ordered=True)
        comp9 = comp9.sort_values(t["safety_col_label"])

        fig9comp = px.bar(
            comp9,
            x="안전수준",
            y="준비수량_합계",
            text="준비수량_합계",
            labels={"안전수준": t["safety_col_label"], "준비수량_합계": t["comparison9_y_label"]},
        )
        fig9comp.update_traces(texttemplate="%{text:,.0f}", textposition="outside")
        fig9comp.update_layout(height=420, showlegend=False, margin=dict(t=15, b=10))
        st.plotly_chart(fig9comp, use_container_width=True)

    st.divider()

    r1, r2, r3 = st.columns(3)
    with r1:
        sido9_list = sorted(d9["시도명"].dropna().unique())
        sido9 = st.selectbox(t["sido_label"], sido9_list, format_func=sido_fmt, key="sido9")
    with r2:
        gu9_list = sorted(d9.loc[d9["시도명"] == sido9, "구시군명"].dropna().unique())
        gu9_sel = st.selectbox(t["gu_label"], gu9_list, format_func=gu_fmt, key="gu9")
    with r3:
        dong9_list = sorted(d9.loc[(d9["시도명"] == sido9) & (d9["구시군명"] == gu9_sel), "읍면동명"].dropna().unique())
        dong9 = st.selectbox(t["dong_label"], dong9_list, format_func=dong_fmt, key="dong9")

    selected9 = d9[(d9["시도명"] == sido9) & (d9["구시군명"] == gu9_sel) & (d9["읍면동명"] == dong9)]
    if not selected9.empty:
        row9 = selected9.iloc[0]
        z1, z2, z3, z4 = st.columns(4)
        z1.metric(t["z_assumed_voters"], format_int(row9["제9회_가정선거인수"]))
        z2.metric(t["z_forecast_demand"], format_int(row9[c9["forecast_ballots"]]))
        z3.metric(t["prepared_template"].format(safety=safety9), format_int(row9[c9["prepared"]]))
        z4.metric(t["kpi_preparation_rate"], f"{row9[c9['prep_rate']] * 100:.2f}%")

        local9 = pd.DataFrame({
            t["safety_col_label"]: SAFETY_LEVELS,
            t["prepared_col_label"]: [
                row9[get_forecast9_columns(level, national_model)["prepared"]]
                for level in SAFETY_LEVELS
            ],
        })
        fig_local9 = px.bar(
            local9,
            x=t["safety_col_label"],
            y=t["prepared_col_label"],
            text=t["prepared_col_label"],
            category_orders={t["safety_col_label"]: SAFETY_LEVELS},
        )
        fig_local9.add_hline(y=row9[c9["forecast_ballots"]], line_dash="dash", annotation_text=t["base_forecast_label"])
        fig_local9.update_traces(texttemplate="%{text:,.0f}", textposition="outside")
        fig_local9.update_xaxes(
            type="category",
            categoryorder="array",
            categoryarray=SAFETY_LEVELS,
            tickmode="array",
            tickvals=SAFETY_LEVELS,
            ticktext=SAFETY_LEVELS,
        )
        fig_local9.update_layout(height=340, showlegend=False, margin=dict(t=25, b=10))
        st.plotly_chart(fig_local9, use_container_width=True)

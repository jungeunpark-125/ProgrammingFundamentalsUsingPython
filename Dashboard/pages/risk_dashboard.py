"""
Ballot Shortage Risk Dashboard (Local Elections, 6th-8th / 2014-2022)
Sido > Gu > Dong drill-down for absolute-risk and rate-risk analysis.

One page, two languages: the ENG/KOR control top-right (common.language_toggle)
switches every label on this page via the STR dict below, instead of being two
separate pages. Registered as a page from app.py (the st.navigation router).

Required files (relative to the project root):
  - Data&Code/05_지방선거_읍면동_통합_6to8회.csv   (required)
  - GeoData/skorea-municipalities-2018-topo-simple.json  (optional -- map renders if present,
    otherwise a placeholder message is shown instead)
"""

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from common import (
    GEOJSON_PATH,
    SIDO_EN,
    build_name_maps,
    choropleth_map,
    language_toggle,
    load_geojson,
    resolve_data_file,
)

DATA_PATH = resolve_data_file("05_", exclude_prefixes=("05b_", "05c_"))

ELECTIONS = ["제6회 전국동시지방선거", "제7회 전국동시지방선거", "제8회 전국동시지방선거"]
ELECTION_LABEL = {
    "en": {ELECTIONS[0]: "6th (2014)", ELECTIONS[1]: "7th (2018)", ELECTIONS[2]: "8th (2022)"},
    "ko": {ELECTIONS[0]: "6회 (2014)", ELECTIONS[1]: "7회 (2018)", ELECTIONS[2]: "8회 (2022)"},
}

STR = {
    "en": dict(
        title="Ballot Shortage Risk Dashboard -- Local Elections 6th-8th (2014-2022)",
        sidebar_header="Region",
        sido_label="Sido (province)",
        gu_label="Gu / Gun / Si",
        dong_label="Dong (detail target)",
        map_election_label="Election shown on map",
        map_metric_label="Map color metric",
        map_metric_help_abs="Absolute risk (day-of votes vs 50%-prepared ballots, signed count)",
        map_metric_help_rate="Rate risk (this gu's Risk_index_50 std vs national average std)",
        map_subheader="Municipality (gu/gun) map -- {e}",
        geo_warning=(
            "Map data (GeoJSON) isn't available yet. "
            "Drop `GeoData/skorea-municipalities-2018-topo-simple.json` into the project folder "
            "and refresh -- the map will appear automatically."
        ),
        map_info_empty="No data available for this selection.",
        selected_caption="Selected: {sido} > {gu} > {dong}",
        turnout_subheader="{gu}: election-day turnout rate by dong (%)",
        dong_index_name="Dong",
        detail_subheader="{dong} -- detail",
        registered_delta="Registered voters",
        registered_suffix="",
        risk_context_caption=(
            "Rate risk (variance-based) is only defined where there are sub-units to compare -- "
            "gu/sido level. Values below are for {gu}'s parent gu and sido."
        ),
        gu_abs_risk_label="Gu absolute risk (ballots)",
        gu_abs_risk_help="Sum of (day-of votes - 50%-prepared ballots) across all dongs in this gu. Positive = net shortage.",
        gu_rate_risk_label="Gu rate risk (std ratio)",
        gu_rate_risk_help="This gu's std of Risk_index_50 across its dongs, divided by the national average of that std. >1 = more internal volatility than average (some dongs may run short even if the gu average looks fine).",
        sido_rate_risk_label="Sido rate risk (std ratio)",
        sido_rate_risk_help="This sido's std of gu-level Risk_index_50, divided by the national average of that std.",
        ratio_unit="x",
        tab_labels=["Trends", "Threshold Simulator", "Radar Profile"],
        trend_abs_title="Absolute risk (day-of votes − 50%-prepared ballots, signed)",
        trend_abs_caption="Positive = shortage (day-of votes exceeded the 50%-prepared ballots). Negative = surplus.",
        trend_rate_title="Rate risk of {gu} (this dong's parent gu)",
        trend_rate_caption=(
            "Std of Risk_index_50 across dongs in this gu, divided by the national average of that std "
            "(dashed line = 1, i.e. national average). >1 means this gu's dongs vary more than typical -- "
            "a high average turnout can still hide one or two dongs that run short."
        ),
        sim_caption="Move the slider to see how many ballots would have been short at this dong under a different preparation threshold.",
        sim_slider_label="Ballot preparation threshold (%)",
        sim_title="{dong}: projected shortage at {threshold}% threshold (ballots)",
        radar_mode_label="Comparison mode",
        radar_mode_options=["a) This dong across 6th-8th", "b) This dong vs gu average vs national average"],
        radar_election_label="Election to compare",
        radar_axes={
            "비율위험_구_pct": "Rate risk (gu variance)",
            "절대량위험_pct": "Absolute risk",
            "선거인수_총_pct": "Population size",
            "관내사전투표비중_pct": "Early-voting preference",
            "당일투표율_변동성_pct": "Forecast volatility",
        },
        gu_avg_label="{gu} average",
        national_avg_label="National average",
        radar_caption="All axes are percentile ranks (0-100) among all dongs nationwide. Closer to 100 means higher relative to the rest of the country.",
        sim_col_prepared="prepared_sim",
        sim_col_shortage="shortage_sim",
        election_axis_label="Election",
        abs_risk_axis_label="Absolute risk (ballots)",
        rate_risk_axis_label="Rate risk (std ratio)",
        shortage_sim_axis_label="Projected shortage (ballots)",
        map_metric_short={"절대량위험": "Absolute risk", "비율위험": "Rate risk"},
    ),
    "ko": dict(
        title="투표용지 부족 위험 대시보드 — 6\\~8회 지방선거 (2014\\~2022)",
        sidebar_header="지역 선택",
        sido_label="시도",
        gu_label="구시군",
        dong_label="읍면동 (상세 분석 대상)",
        map_election_label="지도에 표시할 선거",
        map_metric_label="지도 색상 기준",
        map_metric_help_abs="절대량 위험 (당일투표수 vs 50% 준비 용지, 매수 차이)",
        map_metric_help_rate="비율 위험 (이 구의 Risk_index_50 표준편차 ÷ 전국 평균)",
        map_subheader="구시군 단위 지도 · {e}",
        geo_warning=(
            "지도 데이터(GeoJSON)가 아직 없어요. "
            "`GeoData/skorea-municipalities-2018-topo-simple.json` 파일을 프로젝트 폴더에 넣어주시면 "
            "새로고침 시 자동으로 지도가 나타납니다."
        ),
        map_info_empty="선택한 조건에 해당하는 데이터가 없습니다.",
        selected_caption="현재 선택: {sido} > {gu} > {dong}",
        turnout_subheader="{gu} 읍면동별 당일투표율 (%)",
        dong_index_name="읍면동",
        detail_subheader="{dong} 상세",
        registered_delta="확정선거인수",
        registered_suffix=" 명",
        risk_context_caption=(
            "비율 위험(변동성 기반)은 비교할 하위 단위가 있어야 계산할 수 있어요 — 구/시도 단위 값입니다. "
            "아래는 {dong}이 속한 {gu}(구)와 시도 기준 값이에요."
        ),
        gu_abs_risk_label="구 절대량위험 (매수)",
        gu_abs_risk_help="구 안의 모든 동에서 (당일투표수 - 50%준비용지)를 합산한 값. 양수면 구 전체로도 순부족.",
        gu_rate_risk_label="구 비율위험 (표준편차 비)",
        gu_rate_risk_help="이 구 안 동들의 Risk_index_50 표준편차 ÷ 전국 구 표준편차 평균. 1보다 크면 이 구는 동네 간 격차(변동성)가 전국 평균보다 큼 — 구 평균은 괜찮아 보여도 특정 동에서 몰아서 부족할 위험.",
        sido_rate_risk_label="시도 비율위험 (표준편차 비)",
        sido_rate_risk_help="이 시도 안 구들의 Risk_index_50(구 단위) 표준편차 ÷ 전국 시도 표준편차 평균.",
        ratio_unit="배",
        tab_labels=["추이", "Threshold 시뮬레이터", "레이더 프로파일"],
        trend_abs_title="절대량 위험 (당일투표수 − 50% 준비 용지, 부호 유지)",
        trend_abs_caption="양수 = 부족(당일투표수가 50% 준비량을 초과), 음수 = 여유.",
        trend_rate_title="{gu}(이 동이 속한 구)의 비율 위험",
        trend_rate_caption=(
            "이 구 안 동들의 Risk_index_50 표준편차 ÷ 전국 평균 표준편차 (점선 = 1, 즉 전국 평균). "
            "1보다 크면 이 구는 동네 간 변동성이 평균보다 크다는 뜻 — 평균 투표율이 높아 보여도 "
            "일부 동에서 부족 사태가 날 위험이 있음을 뜻합니다."
        ),
        sim_caption="슬라이더를 움직이면 이 동에서 준비 기준을 바꿨을 때 예상 부족량이 어떻게 달라지는지 바로 계산됩니다.",
        sim_slider_label="투표용지 준비 기준 (%)",
        sim_title="{dong}: 기준 {threshold}%일 때 예상 부족량 (장)",
        radar_mode_label="비교 방식",
        radar_mode_options=["a) 이 동의 6\\~8회 추이", "b) 이 동 vs 구 평균 vs 전국 평균"],
        radar_election_label="비교할 선거",
        radar_axes={
            "비율위험_구_pct": "비율 위험 (구 변동성)",
            "절대량위험_pct": "절대량 위험",
            "선거인수_총_pct": "동네 규모",
            "관내사전투표비중_pct": "사전투표 선호도",
            "당일투표율_변동성_pct": "예측 변동성",
        },
        gu_avg_label="{gu} 평균",
        national_avg_label="전국 평균",
        radar_caption="모든 축은 전국 동 대비 백분위(0\\~100)로 정규화된 값입니다. 100에 가까울수록 전국에서 상위권이라는 뜻이에요.",
        sim_col_prepared="준비량_sim",
        sim_col_shortage="부족_sim",
        election_axis_label="선거",
        abs_risk_axis_label="절대량위험 (매수)",
        rate_risk_axis_label="비율위험 (표준편차 비)",
        shortage_sim_axis_label="예상 부족량 (장)",
        map_metric_short={"절대량위험": "절대량위험", "비율위험": "비율위험"},
    ),
}


@st.cache_data
def load_data():
    return pd.read_csv(DATA_PATH)


df = load_data()
geojson = load_geojson(GEOJSON_PATH)
GU_EN, DONG_EN = build_name_maps(df)

lang = language_toggle()
t = STR[lang]
election_label = ELECTION_LABEL[lang]

sido_fmt = (lambda s: SIDO_EN.get(s, s)) if lang == "en" else (lambda s: s)
gu_fmt = (lambda g: GU_EN.get(g, g)) if lang == "en" else (lambda g: g)
dong_fmt = (lambda d: DONG_EN.get(d, d)) if lang == "en" else (lambda d: d)

# ---------------------------------------------------------------------------
# Sidebar: Sido > Gu > Dong selection
# ---------------------------------------------------------------------------
st.sidebar.header(t["sidebar_header"])

sido_list = sorted(df["시도명"].unique())
sido = st.sidebar.selectbox(
    t["sido_label"], sido_list,
    index=sido_list.index("서울특별시") if "서울특별시" in sido_list else 0,
    format_func=sido_fmt,
)

gu_list = sorted(df.loc[df["시도명"] == sido, "구시군명"].unique())
gu = st.sidebar.selectbox(t["gu_label"], gu_list, format_func=gu_fmt)

dong_list = sorted(df.loc[(df["시도명"] == sido) & (df["구시군명"] == gu), "읍면동명"].unique())
dong = st.sidebar.selectbox(t["dong_label"], dong_list, format_func=dong_fmt)

st.sidebar.divider()
map_election = st.sidebar.radio(t["map_election_label"], ELECTIONS, format_func=lambda e: election_label[e], index=2)
map_metric = st.sidebar.radio(
    t["map_metric_label"],
    ["절대량위험", "비율위험"],
    format_func=lambda m: t["map_metric_help_abs"] if m == "절대량위험" else t["map_metric_help_rate"],
)

st.title(t["title"])

try:
    col_left, col_right = st.columns([1, 1.3], gap="large")
except TypeError:
    col_left, _spacer, col_right = st.columns([1, 0.12, 1.3])

# ---------------------------------------------------------------------------
# Left column: municipality-level map + municipality-level table
# ---------------------------------------------------------------------------
with col_left:
    st.subheader(t["map_subheader"].format(e=election_label[map_election]))

    if geojson is None:
        st.warning(t["geo_warning"])
    else:
        gu_summary = (
            df[df["선거명"] == map_election]
            .groupby(["시도명", "구시군명"])
            .agg(절대량위험=("절대량위험_구", "mean"), 비율위험=("비율위험_구", "mean"))
            .reset_index()
        )
        gu_summary["join_key"] = gu_summary["시도명"] + "_" + gu_summary["구시군명"]

        if lang == "en":
            gu_summary["Sido"] = gu_summary["시도명"].map(SIDO_EN)
            gu_summary["Gu/Gun/Si"] = gu_summary["구시군명"].map(GU_EN)
            hover_cols = ["Sido", "Gu/Gun/Si"]
        else:
            hover_cols = ["시도명", "구시군명"]

        # 절대량위험 is signed (+shortage / -surplus) -> diverging scale centered at 0.
        # 비율위험 is a ratio centered around 1 (>1 = more variance than the national average).
        if map_metric == "절대량위험":
            color_kwargs = dict(color_continuous_scale="RdBu_r", color_continuous_midpoint=0)
        else:
            color_kwargs = dict(color_continuous_scale="Reds")

        fig_map = choropleth_map(
            gu_summary,
            geojson,
            map_metric,
            hover_data=hover_cols,
            opacity=0.75,
            height=430,
            margin=dict(l=0, r=0, t=0, b=0),
            labels={map_metric: t["map_metric_short"][map_metric]},
            **color_kwargs,
        )
        if fig_map is not None:
            st.plotly_chart(fig_map, use_container_width=True)
        else:
            st.info(t["map_info_empty"])

    gu_disp = GU_EN.get(gu, gu) if lang == "en" else gu
    dong_disp = DONG_EN.get(dong, dong) if lang == "en" else dong
    sido_disp = SIDO_EN.get(sido, sido) if lang == "en" else sido

    st.caption(t["selected_caption"].format(sido=sido_disp, gu=gu_disp, dong=dong_disp))

    st.subheader(t["turnout_subheader"].format(gu=gu_disp))
    gu_df = df[(df["시도명"] == sido) & (df["구시군명"] == gu)].copy()
    gu_table = gu_df.pivot_table(index="읍면동명", columns="선거명", values="당일투표율").round(1).rename(columns=election_label)
    if lang == "en":
        gu_table.index = gu_table.index.map(lambda d: DONG_EN.get(d, d))
    gu_table.index.name = t["dong_index_name"]
    st.dataframe(
        gu_table,
        use_container_width=True,
        height=320,
    )

# ---------------------------------------------------------------------------
# Right column: selected-dong detail, organized into tabs
# ---------------------------------------------------------------------------
with col_right:
    st.subheader(t["detail_subheader"].format(dong=dong_disp))

    dong_df = df[(df["시도명"] == sido) & (df["구시군명"] == gu) & (df["읍면동명"] == dong)].copy()
    dong_df["election_label"] = dong_df["선거명"].map(election_label)

    info_cols = st.columns(len(dong_df)) if len(dong_df) > 0 else [st]
    for c, (_, row) in zip(info_cols, dong_df.iterrows()):
        with c:
            st.metric(row["election_label"], f"{row['선거인수_총']:,.0f}{t['registered_suffix']}", t["registered_delta"])

    st.caption(t["risk_context_caption"].format(gu=gu_disp, dong=dong_disp))
    ctx_cols = st.columns(len(dong_df)) if len(dong_df) > 0 else [st]
    for c, (_, row) in zip(ctx_cols, dong_df.iterrows()):
        with c:
            st.markdown(f"**{row['election_label']}**")
            st.metric(
                t["gu_abs_risk_label"],
                f"{row['절대량위험_구']:+,.0f}",
                help=t["gu_abs_risk_help"],
            )
            st.metric(
                t["gu_rate_risk_label"],
                f"{row['비율위험_구']:.2f}{t['ratio_unit']}" if pd.notna(row["비율위험_구"]) else "N/A",
                help=t["gu_rate_risk_help"],
            )
            st.metric(
                t["sido_rate_risk_label"],
                f"{row['비율위험_시도']:.2f}{t['ratio_unit']}" if pd.notna(row["비율위험_시도"]) else "N/A",
                help=t["sido_rate_risk_help"],
            )

    tab_trend, tab_sim, tab_radar = st.tabs(t["tab_labels"])

    # ---- Tab 1: absolute / rate risk trend across elections --------------
    with tab_trend:
        fig = px.bar(
            dong_df, x="election_label", y="절대량위험",
            title=t["trend_abs_title"],
            color="election_label",
            labels={"election_label": t["election_axis_label"], "절대량위험": t["abs_risk_axis_label"]},
        )
        fig.add_hline(y=0, line_dash="dash", line_color="gray")
        fig.update_layout(showlegend=False, height=320)
        st.plotly_chart(fig, use_container_width=True)
        st.caption(t["trend_abs_caption"])

        fig = px.bar(
            dong_df, x="election_label", y="비율위험_구",
            title=t["trend_rate_title"].format(gu=gu_disp),
            color="election_label",
            labels={"election_label": t["election_axis_label"], "비율위험_구": t["rate_risk_axis_label"]},
        )
        fig.add_hline(y=1, line_dash="dash", line_color="red")
        fig.update_layout(showlegend=False, height=320)
        st.plotly_chart(fig, use_container_width=True)
        st.caption(t["trend_rate_caption"])

    # ---- Tab 2: live threshold slider --------------------------------------
    with tab_sim:
        st.caption(t["sim_caption"])
        threshold = st.slider(t["sim_slider_label"], 40, 70, 50, 1)
        sim = dong_df.copy()
        sim[t["sim_col_prepared"]] = sim["선거인수_총"] * threshold / 100
        sim[t["sim_col_shortage"]] = (sim["선거일투표수"] - sim[t["sim_col_prepared"]]).clip(lower=0)
        fig = px.bar(
            sim, x="election_label", y=t["sim_col_shortage"],
            title=t["sim_title"].format(dong=dong_disp, threshold=threshold),
            color="election_label",
            labels={"election_label": t["election_axis_label"], t["sim_col_shortage"]: t["shortage_sim_axis_label"]},
        )
        fig.update_layout(showlegend=False, height=360)
        st.plotly_chart(fig, use_container_width=True)

    # ---- Tab 3: radar / spider chart --------------------------------------
    with tab_radar:
        radar_mode = st.radio(
            t["radar_mode_label"],
            t["radar_mode_options"],
            horizontal=True,
        )

        axis_keys = list(t["radar_axes"].keys())
        axis_labels = list(t["radar_axes"].values())

        fig_radar = go.Figure()

        if radar_mode.startswith("a"):
            for _, row in dong_df.iterrows():
                values = [row[k] if pd.notna(row[k]) else 0 for k in axis_keys]
                fig_radar.add_trace(go.Scatterpolar(
                    r=values + values[:1], theta=axis_labels + axis_labels[:1],
                    fill="toself", name=row["election_label"],
                ))
        else:
            radar_election = st.selectbox(t["radar_election_label"], ELECTIONS, format_func=lambda e: election_label[e], index=2, key="radar_election")
            this_row = dong_df[dong_df["선거명"] == radar_election]
            gu_avg = df[(df["시도명"] == sido) & (df["구시군명"] == gu) & (df["선거명"] == radar_election)][axis_keys].mean()
            national_avg = df[df["선거명"] == radar_election][axis_keys].mean()

            if not this_row.empty:
                v = [this_row.iloc[0][k] if pd.notna(this_row.iloc[0][k]) else 0 for k in axis_keys]
                fig_radar.add_trace(go.Scatterpolar(r=v + v[:1], theta=axis_labels + axis_labels[:1], fill="toself", name=dong_disp))
            fig_radar.add_trace(go.Scatterpolar(
                r=list(gu_avg.values) + [gu_avg.values[0]], theta=axis_labels + axis_labels[:1],
                name=t["gu_avg_label"].format(gu=gu_disp),
            ))
            fig_radar.add_trace(go.Scatterpolar(
                r=list(national_avg.values) + [national_avg.values[0]], theta=axis_labels + axis_labels[:1],
                name=t["national_avg_label"],
            ))

        fig_radar.update_layout(
            polar=dict(radialaxis=dict(visible=True, range=[0, 100])),
            showlegend=True, height=420,
        )
        st.plotly_chart(fig_radar, use_container_width=True)
        st.caption(t["radar_caption"])

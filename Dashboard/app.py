"""
Ballot Shortage Risk Dashboard (Local Elections, 6th-8th / 2014-2022) -- EN
Sido > Gu > Dong drill-down for absolute-risk and rate-risk analysis.

This is the main entry point of the multi-page app:
    streamlit run app.py

Other dashboards (Korean risk dashboard, EN/KO forecast dashboards) live under
pages/ and appear automatically in the sidebar navigator. Shared code lives in
common.py.

Required files (relative to the project root, i.e. one level up from this file):
  - Data&Code/05_지방선거_읍면동_통합_6to8회.csv   (required)
  - GeoData/skorea-municipalities-2018-topo-simple.json  (optional -- map renders if present,
    otherwise a placeholder message is shown instead)
"""

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from common import GEOJSON_PATH, SIDO_EN, build_name_maps, choropleth_map, load_geojson, resolve_data_file

st.set_page_config(page_title="Ballot Shortage Risk Dashboard", layout="wide")

DATA_PATH = resolve_data_file("05_", exclude_prefixes=("05b_", "05c_"))

ELECTIONS = ["제6회 전국동시지방선거", "제7회 전국동시지방선거", "제8회 전국동시지방선거"]
ELECTION_LABEL = {ELECTIONS[0]: "6th (2014)", ELECTIONS[1]: "7th (2018)", ELECTIONS[2]: "8th (2022)"}


@st.cache_data
def load_data():
    return pd.read_csv(DATA_PATH)


df = load_data()
geojson = load_geojson(GEOJSON_PATH)
GU_EN, DONG_EN = build_name_maps(df)

# ---------------------------------------------------------------------------
# Sidebar: Sido > Gu > Dong selection
# ---------------------------------------------------------------------------
st.sidebar.header("Region")

sido_list = sorted(df["시도명"].unique())
sido = st.sidebar.selectbox(
    "Sido (province)", sido_list,
    index=sido_list.index("서울특별시") if "서울특별시" in sido_list else 0,
    format_func=lambda s: SIDO_EN.get(s, s),
)

gu_list = sorted(df.loc[df["시도명"] == sido, "구시군명"].unique())
gu = st.sidebar.selectbox("Gu / Gun / Si", gu_list, format_func=lambda g: GU_EN.get(g, g))

dong_list = sorted(df.loc[(df["시도명"] == sido) & (df["구시군명"] == gu), "읍면동명"].unique())
dong = st.sidebar.selectbox("Dong (detail target)", dong_list, format_func=lambda d: DONG_EN.get(d, d))

st.sidebar.divider()
map_election = st.sidebar.radio("Election shown on map", ELECTIONS, format_func=lambda e: ELECTION_LABEL[e], index=2)
map_metric = st.sidebar.radio(
    "Map color metric",
    ["절대량위험", "비율위험"],
    format_func=lambda m: (
        "Absolute risk (day-of votes vs 50%-prepared ballots, signed count)"
        if m == "절대량위험"
        else "Rate risk (this gu's Risk_index_50 std vs national average std)"
    ),
)

st.title("Ballot Shortage Risk Dashboard -- Local Elections 6th-8th (2014-2022)")

# Wider gap between the two panels; fall back to a spacer column on older Streamlit
# versions that don't support the `gap` argument on st.columns.
try:
    col_left, col_right = st.columns([1, 1.3], gap="large")
except TypeError:
    col_left, _spacer, col_right = st.columns([1, 0.12, 1.3])

# ---------------------------------------------------------------------------
# Left column: municipality-level map + municipality-level table
# ---------------------------------------------------------------------------
with col_left:
    st.subheader(f"Municipality (gu/gun) map -- {ELECTION_LABEL[map_election]}")

    if geojson is None:
        st.warning(
            "Map data (GeoJSON) isn't available yet. "
            "Drop `GeoData/skorea-municipalities-2018-topo-simple.json` into the project folder "
            "and refresh -- the map will appear automatically."
        )
    else:
        gu_summary = (
            df[df["선거명"] == map_election]
            .groupby(["시도명", "구시군명"])
            .agg(절대량위험=("절대량위험_구", "mean"), 비율위험=("비율위험_구", "mean"))
            .reset_index()
        )
        gu_summary["join_key"] = gu_summary["시도명"] + "_" + gu_summary["구시군명"]
        gu_summary["Sido"] = gu_summary["시도명"].map(SIDO_EN)
        gu_summary["Gu/Gun/Si"] = gu_summary["구시군명"].map(GU_EN)

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
            hover_data=["Sido", "Gu/Gun/Si"],
            opacity=0.75,
            height=430,
            margin=dict(l=0, r=0, t=0, b=0),
            **color_kwargs,
        )
        if fig_map is not None:
            st.plotly_chart(fig_map, use_container_width=True)
        else:
            st.info("No data available for this selection.")

    st.caption(f"Selected: {SIDO_EN.get(sido, sido)} > {GU_EN.get(gu, gu)} > {DONG_EN.get(dong, dong)}")

    st.subheader(f"{GU_EN.get(gu, gu)}: election-day turnout rate by dong (%)")
    gu_df = df[(df["시도명"] == sido) & (df["구시군명"] == gu)].copy()
    gu_table = gu_df.pivot_table(index="읍면동명", columns="선거명", values="당일투표율").round(1).rename(columns=ELECTION_LABEL)
    gu_table.index = gu_table.index.map(lambda d: DONG_EN.get(d, d))
    gu_table.index.name = "Dong"
    st.dataframe(
        gu_table,
        use_container_width=True,
        height=320,
    )

# ---------------------------------------------------------------------------
# Right column: selected-dong detail, organized into tabs
# ---------------------------------------------------------------------------
with col_right:
    dong_en = DONG_EN.get(dong, dong)
    gu_en = GU_EN.get(gu, gu)
    st.subheader(f"{dong_en} -- detail")

    dong_df = df[(df["시도명"] == sido) & (df["구시군명"] == gu) & (df["읍면동명"] == dong)].copy()
    dong_df["election_label"] = dong_df["선거명"].map(ELECTION_LABEL)

    info_cols = st.columns(len(dong_df)) if len(dong_df) > 0 else [st]
    for c, (_, row) in zip(info_cols, dong_df.iterrows()):
        with c:
            st.metric(row["election_label"], f"{row['선거인수_총']:,.0f}", "Registered voters")

    st.caption(
        "Rate risk (variance-based) is only defined where there are sub-units to compare -- "
        f"gu/sido level. Values below are for {gu_en}'s parent gu and sido."
    )
    ctx_cols = st.columns(len(dong_df)) if len(dong_df) > 0 else [st]
    for c, (_, row) in zip(ctx_cols, dong_df.iterrows()):
        with c:
            st.markdown(f"**{row['election_label']}**")
            st.metric(
                "Gu absolute risk (ballots)",
                f"{row['절대량위험_구']:+,.0f}",
                help="Sum of (day-of votes - 50%-prepared ballots) across all dongs in this gu. Positive = net shortage.",
            )
            st.metric(
                "Gu rate risk (std ratio)",
                f"{row['비율위험_구']:.2f}x" if pd.notna(row["비율위험_구"]) else "N/A",
                help="This gu's std of Risk_index_50 across its dongs, divided by the national average of that std. >1 = more internal volatility than average (some dongs may run short even if the gu average looks fine).",
            )
            st.metric(
                "Sido rate risk (std ratio)",
                f"{row['비율위험_시도']:.2f}x" if pd.notna(row["비율위험_시도"]) else "N/A",
                help="This sido's std of gu-level Risk_index_50, divided by the national average of that std.",
            )

    tab_trend, tab_sim, tab_radar = st.tabs(["Trends", "Threshold Simulator", "Radar Profile"])

    # ---- Tab 1: absolute / rate risk trend across elections --------------
    with tab_trend:
        fig = px.bar(
            dong_df, x="election_label", y="절대량위험",
            title="Absolute risk (day-of votes − 50%-prepared ballots, signed)",
            color="election_label",
        )
        fig.add_hline(y=0, line_dash="dash", line_color="gray")
        fig.update_layout(showlegend=False, height=320)
        st.plotly_chart(fig, use_container_width=True)
        st.caption("Positive = shortage (day-of votes exceeded the 50%-prepared ballots). Negative = surplus.")

        fig = px.bar(
            dong_df, x="election_label", y="비율위험_구",
            title=f"Rate risk of {gu_en} (this dong's parent gu)",
            color="election_label",
        )
        fig.add_hline(y=1, line_dash="dash", line_color="red")
        fig.update_layout(showlegend=False, height=320)
        st.plotly_chart(fig, use_container_width=True)
        st.caption(
            "Std of Risk_index_50 across dongs in this gu, divided by the national average of that std "
            "(dashed line = 1, i.e. national average). >1 means this gu's dongs vary more than typical -- "
            "a high average turnout can still hide one or two dongs that run short."
        )

    # ---- Tab 2: live threshold slider --------------------------------------
    with tab_sim:
        st.caption("Move the slider to see how many ballots would have been short at this dong under a different preparation threshold.")
        threshold = st.slider("Ballot preparation threshold (%)", 40, 70, 50, 1)
        sim = dong_df.copy()
        sim["prepared_sim"] = sim["선거인수_총"] * threshold / 100
        sim["shortage_sim"] = (sim["선거일투표수"] - sim["prepared_sim"]).clip(lower=0)
        fig = px.bar(
            sim, x="election_label", y="shortage_sim",
            title=f"{dong_en}: projected shortage at {threshold}% threshold (ballots)",
            color="election_label",
        )
        fig.update_layout(showlegend=False, height=360)
        st.plotly_chart(fig, use_container_width=True)

    # ---- Tab 3: radar / spider chart --------------------------------------
    with tab_radar:
        radar_mode = st.radio(
            "Comparison mode",
            ["a) This dong across 6th-8th", "b) This dong vs gu average vs national average"],
            horizontal=True,
        )

        RADAR_AXES = {
            "비율위험_구_pct": "Rate risk (gu variance)",
            "절대량위험_pct": "Absolute risk",
            "선거인수_총_pct": "Population size",
            "관내사전투표비중_pct": "Early-voting preference",
            "당일투표율_변동성_pct": "Forecast volatility",
        }
        axis_keys = list(RADAR_AXES.keys())
        axis_labels = list(RADAR_AXES.values())

        fig_radar = go.Figure()

        if radar_mode.startswith("a"):
            for _, row in dong_df.iterrows():
                values = [row[k] if pd.notna(row[k]) else 0 for k in axis_keys]
                fig_radar.add_trace(go.Scatterpolar(
                    r=values + values[:1], theta=axis_labels + axis_labels[:1],
                    fill="toself", name=row["election_label"],
                ))
        else:
            radar_election = st.selectbox("Election to compare", ELECTIONS, format_func=lambda e: ELECTION_LABEL[e], index=2, key="radar_election")
            this_row = dong_df[dong_df["선거명"] == radar_election]
            gu_avg = df[(df["시도명"] == sido) & (df["구시군명"] == gu) & (df["선거명"] == radar_election)][axis_keys].mean()
            national_avg = df[df["선거명"] == radar_election][axis_keys].mean()

            if not this_row.empty:
                v = [this_row.iloc[0][k] if pd.notna(this_row.iloc[0][k]) else 0 for k in axis_keys]
                fig_radar.add_trace(go.Scatterpolar(r=v + v[:1], theta=axis_labels + axis_labels[:1], fill="toself", name=dong_en))
            fig_radar.add_trace(go.Scatterpolar(
                r=list(gu_avg.values) + [gu_avg.values[0]], theta=axis_labels + axis_labels[:1],
                name=f"{gu_en} average",
            ))
            fig_radar.add_trace(go.Scatterpolar(
                r=list(national_avg.values) + [national_avg.values[0]], theta=axis_labels + axis_labels[:1],
                name="National average",
            ))

        fig_radar.update_layout(
            polar=dict(radialaxis=dict(visible=True, range=[0, 100])),
            showlegend=True, height=420,
        )
        st.plotly_chart(fig_radar, use_container_width=True)
        st.caption("All axes are percentile ranks (0-100) among all dongs nationwide. Closer to 100 means higher relative to the rest of the country.")

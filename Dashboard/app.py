"""
Ballot Shortage Risk Dashboard (Local Elections, 6th-8th / 2014-2022)
Sido > Gu > Dong drill-down for absolute-risk and rate-risk analysis.

Run: streamlit run app.py
Required files (relative to this file):
  - ../Data&Code/05_지방선거_읍면동_통합_6to8회.csv   (required)
  - ../GeoData/skorea-municipalities-2018-topo-simple.json  (optional -- map renders if present,
    otherwise a placeholder message is shown instead)
"""

import json
import os
import re

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from korean_romanizer.romanizer import Romanizer

st.set_page_config(page_title="Ballot Shortage Risk Dashboard", layout="wide")

DATA_PATH = os.path.join(os.path.dirname(__file__), "..", "Data&Code", "05_지방선거_읍면동_통합_6to8회.csv")
GEOJSON_PATH = os.path.join(os.path.dirname(__file__), "..", "GeoData", "skorea-municipalities-2018-topo-simple.json")

ELECTIONS = ["제6회 전국동시지방선거", "제7회 전국동시지방선거", "제8회 전국동시지방선거"]
ELECTION_LABEL = {ELECTIONS[0]: "6th (2014)", ELECTIONS[1]: "7th (2018)", ELECTIONS[2]: "8th (2022)"}

# geojson (southkorea-maps, GADM-based) "code" prefix (first 2 digits) -> sido name.
# This is NOT the official KOSTAT admin code scheme (e.g. "26" here = Ulsan, not Busan) --
# mapping was verified empirically against our own dataset.
SIDO_CODE_PREFIX = {
    "11": "서울특별시", "21": "부산광역시", "22": "대구광역시", "23": "인천광역시",
    "24": "광주광역시", "25": "대전광역시", "26": "울산광역시", "29": "세종특별자치시",
    "31": "경기도", "32": "강원도", "33": "충청북도", "34": "충청남도",
    "35": "전라북도", "36": "전라남도", "37": "경상북도", "38": "경상남도",
    "39": "제주특별자치도",
}
# Name mismatches between geojson and our dataset (renamed/merged districts)
NAME_ALIAS = {"세종시": "세종특별자치시"}

# English display names for sido/gu (for UI labels; underlying joins still use Korean keys)
SIDO_EN = {
    "서울특별시": "Seoul", "부산광역시": "Busan", "대구광역시": "Daegu", "인천광역시": "Incheon",
    "광주광역시": "Gwangju", "대전광역시": "Daejeon", "울산광역시": "Ulsan", "세종특별자치시": "Sejong",
    "경기도": "Gyeonggi", "강원도": "Gangwon", "충청북도": "Chungbuk", "충청남도": "Chungnam",
    "전라북도": "Jeonbuk", "전라남도": "Jeonnam", "경상북도": "Gyeongbuk", "경상남도": "Gyeongnam",
    "제주특별자치도": "Jeju",
}

# ---------------------------------------------------------------------------
# Automatic romanization for gu/dong names (thousands of unique names --
# a hand-built dictionary isn't practical, so we romanize programmatically).
# ---------------------------------------------------------------------------
_SUFFIX_EN = {"시": "si", "군": "gun", "구": "gu", "읍": "eup", "면": "myeon", "동": "dong", "리": "ri"}
_MULTI_LEVEL_CITY = [
    "수원시", "성남시", "안양시", "안산시", "용인시", "청주시", "천안시",
    "전주시", "포항시", "창원시", "고양시", "부천시",
]


def _romanize_simple(name):
    m = re.match(r"^(.*?)(\d*)(시|군|구|읍|면|동|리)$", name)
    if m:
        base, num, suf = m.groups()
        base_r = Romanizer(base).romanize().capitalize() if base else ""
        suf_en = _SUFFIX_EN[suf]
        return f"{base_r} {num}-{suf_en}".strip() if num else f"{base_r}-{suf_en}".strip()
    return Romanizer(name).romanize().capitalize()


def romanize_name(name):
    if not isinstance(name, str) or not name:
        return name
    for city in _MULTI_LEVEL_CITY:
        if name.startswith(city) and name != city:
            base = Romanizer(city[:-1]).romanize().capitalize()
            return f"{base}-si {_romanize_simple(name[len(city):])}"
    return _romanize_simple(name)


@st.cache_data
def build_name_maps(df):
    gu_map = {n: romanize_name(n) for n in df["구시군명"].unique()}
    dong_map = {n: romanize_name(n) for n in df["읍면동명"].unique()}
    return gu_map, dong_map


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------
@st.cache_data
def load_data():
    return pd.read_csv(DATA_PATH)


@st.cache_data
def load_geojson(path):
    """Convert a southkorea-maps-style TopoJSON into a GeoJSON FeatureCollection
    (arc delta-decoding implemented by hand, no extra topojson package needed).
    Returns None if the file isn't there yet -- the rest of the app still works without it.
    """
    if not os.path.exists(path):
        return None

    with open(path, encoding="utf-8") as f:
        topo = json.load(f)

    scale = topo["transform"]["scale"]
    translate = topo["transform"]["translate"]

    def decode_arc(arc):
        x, y = 0, 0
        coords = []
        for dx, dy in arc:
            x += dx
            y += dy
            coords.append([x * scale[0] + translate[0], y * scale[1] + translate[1]])
        return coords

    arcs = [decode_arc(a) for a in topo["arcs"]]

    def resolve_ring(arc_indices):
        ring = []
        for idx in arc_indices:
            seg = arcs[idx] if idx >= 0 else list(reversed(arcs[~idx]))
            if ring and ring[-1] == seg[0]:
                ring.extend(seg[1:])
            else:
                ring.extend(seg)
        return ring

    obj_name = list(topo["objects"].keys())[0]
    geometries = topo["objects"][obj_name]["geometries"]

    features = []
    for g in geometries:
        gtype = g["type"]
        if gtype == "Polygon":
            rings = [resolve_ring(r) for r in g["arcs"]]
            geom = {"type": "Polygon", "coordinates": rings}
        elif gtype == "MultiPolygon":
            polys = [[resolve_ring(r) for r in poly] for poly in g["arcs"]]
            geom = {"type": "MultiPolygon", "coordinates": polys}
        else:
            continue

        props = dict(g.get("properties", {}))
        # Gu/gun names alone repeat across sido (e.g. "동구"/"중구"/"서구" exist in several
        # cities), so we rebuild a "sido_gu" join key from the code prefix to avoid mis-joins.
        code = str(props.get("code", ""))
        sido_name = SIDO_CODE_PREFIX.get(code[:2], "")
        gu_name = NAME_ALIAS.get(props.get("name", ""), props.get("name", ""))
        props["join_key"] = f"{sido_name}_{gu_name}"
        features.append({"type": "Feature", "properties": props, "geometry": geom})

    return {"type": "FeatureCollection", "features": features}


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

        map_kwargs = dict(
            data_frame=gu_summary,
            geojson=geojson,
            locations="join_key",
            featureidkey="properties.join_key",
            color=map_metric,
            zoom=5.8,
            center={"lat": 36.3, "lon": 127.8},
            opacity=0.75,
            hover_data=["Sido", "Gu/Gun/Si"],
            **color_kwargs,
        )
        # Newer plotly has choropleth_map (maplibre); fall back to choropleth_mapbox on older installs
        if hasattr(px, "choropleth_map"):
            fig_map = px.choropleth_map(map_style="carto-positron", **map_kwargs)
        else:
            fig_map = px.choropleth_mapbox(mapbox_style="carto-positron", **map_kwargs)
        fig_map.update_layout(margin=dict(l=0, r=0, t=0, b=0), height=430)
        st.plotly_chart(fig_map, use_container_width=True)

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

"""
투표용지 부족 위험 대시보드 (6~8회 지방선거, 2014~2022) — 한국어 버전
시 > 구 > 동 드릴다운으로 절대량 위험 / 비율 위험을 확인하는 Streamlit 앱.

실행: streamlit run app_ko.py
필요 파일 (같은 폴더 기준 상대 경로):
  - ../Data&Code/05_지방선거_읍면동_통합_6to8회.csv   (필수)
  - ../GeoData/skorea-municipalities-2018-topo-simple.json  (선택 — 있으면 지도가 뜨고, 없으면 지도 자리에 안내 문구만 표시됨)
"""

import json
import os

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

st.set_page_config(page_title="투표용지 부족 위험 대시보드", layout="wide")

DATA_PATH = os.path.join(os.path.dirname(__file__), "..", "Data&Code", "05_지방선거_읍면동_통합_6to8회.csv")
GEOJSON_PATH = os.path.join(os.path.dirname(__file__), "..", "GeoData", "skorea-municipalities-2018-topo-simple.json")

ELECTIONS = ["제6회 전국동시지방선거", "제7회 전국동시지방선거", "제8회 전국동시지방선거"]
ELECTION_LABEL = {ELECTIONS[0]: "6회 (2014)", ELECTIONS[1]: "7회 (2018)", ELECTIONS[2]: "8회 (2022)"}

# geojson(southkorea-maps, GADM 기반)의 "code" 앞 2자리 -> 시도명.
# 통계청 표준 시군구코드와는 다른 자체 numbering이라, 실제 파일을 우리 데이터와
# 대조해서 경험적으로 확인한 매핑임 (표준코드 그대로 쓰면 틀림 — 예: 26이 통계청 기준 부산이 아니라 울산).
SIDO_CODE_PREFIX = {
    "11": "서울특별시", "21": "부산광역시", "22": "대구광역시", "23": "인천광역시",
    "24": "광주광역시", "25": "대전광역시", "26": "울산광역시", "29": "세종특별자치시",
    "31": "경기도", "32": "강원도", "33": "충청북도", "34": "충청남도",
    "35": "전라북도", "36": "전라남도", "37": "경상북도", "38": "경상남도",
    "39": "제주특별자치도",
}
# geojson과 우리 데이터 간 이름 표기가 다른 경우 (구역 통합/개칭 등)
NAME_ALIAS = {"세종시": "세종특별자치시"}


# ---------------------------------------------------------------------------
# 데이터 로딩
# ---------------------------------------------------------------------------
@st.cache_data
def load_data():
    return pd.read_csv(DATA_PATH)


@st.cache_data
def load_geojson(path):
    """TopoJSON(southkorea-maps 포맷)을 GeoJSON FeatureCollection으로 직접 변환.
    (topojson 전용 파이썬 패키지 없이, arcs delta-decoding을 직접 구현)
    파일이 없으면 None을 반환 — 지도 없이도 나머지 기능은 정상 동작함.
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
        # 구시군명만으로는 "동구", "중구", "서구" 등 여러 시도에 걸쳐 이름이 중복돼서
        # 지도 색칠이 엉뚱한 곳으로 섞일 수 있음. code의 앞 2자리로 시도명을 복원해서
        # "시도명_구시군명" 형태의 고유 join key를 만들어 properties에 추가.
        code = str(props.get("code", ""))
        sido_name = SIDO_CODE_PREFIX.get(code[:2], "")
        gu_name = NAME_ALIAS.get(props.get("name", ""), props.get("name", ""))
        props["join_key"] = f"{sido_name}_{gu_name}"
        features.append({"type": "Feature", "properties": props, "geometry": geom})

    return {"type": "FeatureCollection", "features": features}


df = load_data()
geojson = load_geojson(GEOJSON_PATH)

# ---------------------------------------------------------------------------
# 사이드바: 시 > 구 > 동 선택
# ---------------------------------------------------------------------------
st.sidebar.header("지역 선택")

sido_list = sorted(df["시도명"].unique())
sido = st.sidebar.selectbox("시도", sido_list, index=sido_list.index("서울특별시") if "서울특별시" in sido_list else 0)

gu_list = sorted(df.loc[df["시도명"] == sido, "구시군명"].unique())
gu = st.sidebar.selectbox("구시군", gu_list)

dong_list = sorted(df.loc[(df["시도명"] == sido) & (df["구시군명"] == gu), "읍면동명"].unique())
dong = st.sidebar.selectbox("읍면동 (상세 분석 대상)", dong_list)

st.sidebar.divider()
map_election = st.sidebar.radio("지도에 표시할 선거", ELECTIONS, format_func=lambda e: ELECTION_LABEL[e], index=2)
map_metric = st.sidebar.radio(
    "지도 색상 기준",
    ["당일투표율", "부족_50"],
    format_func=lambda m: "비율 위험 (당일투표율 %)" if m == "당일투표율" else "절대량 위험 (부족_50, 장)",
)

st.title("투표용지 부족 위험 대시보드 — 6~8회 지방선거 (2014~2022)")

# 왼쪽/오른쪽 패널 사이 여백을 넓게. gap 옵션을 지원하지 않는 구버전 streamlit이면
# 빈 스페이서 컬럼으로 자동 대체.
try:
    col_left, col_right = st.columns([1, 1.3], gap="large")
except TypeError:
    col_left, _spacer, col_right = st.columns([1, 0.12, 1.3])

# ---------------------------------------------------------------------------
# 왼쪽: 구시군 단위 지도 + 구시군 단위 표
# ---------------------------------------------------------------------------
with col_left:
    st.subheader(f"구시군 단위 지도 · {ELECTION_LABEL[map_election]}")

    if geojson is None:
        st.warning(
            "지도 데이터(GeoJSON)가 아직 없어요. "
            "`GeoData/skorea-municipalities-2018-topo-simple.json` 파일을 프로젝트 폴더에 넣어주시면 "
            "새로고침 시 자동으로 지도가 나타납니다."
        )
    else:
        gu_summary = (
            df[df["선거명"] == map_election]
            .groupby(["시도명", "구시군명"])
            .agg(당일투표율=("당일투표율", "mean"), 부족_50=("부족_50", "sum"))
            .reset_index()
        )
        gu_summary["join_key"] = gu_summary["시도명"] + "_" + gu_summary["구시군명"]

        map_kwargs = dict(
            data_frame=gu_summary,
            geojson=geojson,
            locations="join_key",
            featureidkey="properties.join_key",
            color=map_metric,
            color_continuous_scale="Reds",
            zoom=5.8,
            center={"lat": 36.3, "lon": 127.8},
            opacity=0.75,
            hover_data=["시도명", "구시군명"],
        )
        # plotly 버전에 따라 choropleth_map(신규, maplibre)이 없을 수 있어서
        # 있으면 그걸 쓰고, 없으면 구버전 choropleth_mapbox로 자동 대체
        if hasattr(px, "choropleth_map"):
            fig_map = px.choropleth_map(map_style="carto-positron", **map_kwargs)
        else:
            fig_map = px.choropleth_mapbox(mapbox_style="carto-positron", **map_kwargs)
        fig_map.update_layout(margin=dict(l=0, r=0, t=0, b=0), height=430)
        st.plotly_chart(fig_map, use_container_width=True)

    st.caption(f"현재 선택: {sido} > {gu} > {dong}")

    st.subheader(f"{gu} 읍면동별 당일투표율 (%)")
    gu_df = df[(df["시도명"] == sido) & (df["구시군명"] == gu)].copy()
    st.dataframe(
        gu_df.pivot_table(index="읍면동명", columns="선거명", values="당일투표율").round(1).rename(columns=ELECTION_LABEL),
        use_container_width=True,
        height=320,
    )

# ---------------------------------------------------------------------------
# 오른쪽: 선택한 동 상세 (탭 구성)
# ---------------------------------------------------------------------------
with col_right:
    st.subheader(f"{dong} 상세")

    dong_df = df[(df["시도명"] == sido) & (df["구시군명"] == gu) & (df["읍면동명"] == dong)].copy()
    dong_df["election_label"] = dong_df["선거명"].map(ELECTION_LABEL)

    info_cols = st.columns(len(dong_df)) if len(dong_df) > 0 else [st]
    for c, (_, row) in zip(info_cols, dong_df.iterrows()):
        with c:
            st.metric(row["election_label"], f"{row['선거인수_총']:,.0f} 명", "확정선거인수")

    tab_trend, tab_sim, tab_radar = st.tabs(["추이", "Threshold 시뮬레이터", "레이더 프로파일"])

    # ---- 탭 1: 절대량/비율 위험 추이 --------------------------------------
    with tab_trend:
        fig = px.bar(dong_df, x="election_label", y="부족_50", title="절대량 위험 (부족_50, 장)", color="election_label")
        fig.update_layout(showlegend=False, height=320)
        st.plotly_chart(fig, use_container_width=True)

        fig = px.line(dong_df, x="election_label", y="당일투표율", markers=True, title="비율 위험 (당일투표율 %)")
        fig.add_hline(y=50, line_dash="dash", line_color="red")
        fig.add_hline(y=60, line_dash="dot", line_color="darkred")
        fig.update_layout(height=320)
        st.plotly_chart(fig, use_container_width=True)

    # ---- 탭 2: threshold 슬라이더 — 실시간 예상 부족량 --------------------
    with tab_sim:
        st.caption("슬라이더를 움직이면 이 동에서 준비 기준을 바꿨을 때 예상 부족량이 어떻게 달라지는지 바로 계산됩니다.")
        threshold = st.slider("투표용지 준비 기준 (%)", 40, 70, 50, 1)
        sim = dong_df.copy()
        sim["준비량_sim"] = sim["선거인수_총"] * threshold / 100
        sim["부족_sim"] = (sim["선거일투표수"] - sim["준비량_sim"]).clip(lower=0)
        fig = px.bar(
            sim, x="election_label", y="부족_sim",
            title=f"{dong}: 기준 {threshold}%일 때 예상 부족량 (장)",
            color="election_label",
        )
        fig.update_layout(showlegend=False, height=360)
        st.plotly_chart(fig, use_container_width=True)

    # ---- 탭 3: 스파이더(레이더) 차트 --------------------------------------
    with tab_radar:
        radar_mode = st.radio(
            "비교 방식",
            ["a) 이 동의 6~8회 추이", "b) 이 동 vs 구 평균 vs 전국 평균"],
            horizontal=True,
        )

        RADAR_AXES = {
            "당일투표율_pct": "비율 위험",
            "부족_50_pct": "절대량 위험",
            "선거인수_총_pct": "동네 규모",
            "관내사전투표비중_pct": "사전투표 선호도",
            "당일투표율_변동성_pct": "예측 변동성",
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
            radar_election = st.selectbox("비교할 선거", ELECTIONS, format_func=lambda e: ELECTION_LABEL[e], index=2, key="radar_election")
            this_row = dong_df[dong_df["선거명"] == radar_election]
            gu_avg = df[(df["시도명"] == sido) & (df["구시군명"] == gu) & (df["선거명"] == radar_election)][axis_keys].mean()
            national_avg = df[df["선거명"] == radar_election][axis_keys].mean()

            if not this_row.empty:
                v = [this_row.iloc[0][k] if pd.notna(this_row.iloc[0][k]) else 0 for k in axis_keys]
                fig_radar.add_trace(go.Scatterpolar(r=v + v[:1], theta=axis_labels + axis_labels[:1], fill="toself", name=dong))
            fig_radar.add_trace(go.Scatterpolar(
                r=list(gu_avg.values) + [gu_avg.values[0]], theta=axis_labels + axis_labels[:1],
                name=f"{gu} 평균",
            ))
            fig_radar.add_trace(go.Scatterpolar(
                r=list(national_avg.values) + [national_avg.values[0]], theta=axis_labels + axis_labels[:1],
                name="전국 평균",
            ))

        fig_radar.update_layout(
            polar=dict(radialaxis=dict(visible=True, range=[0, 100])),
            showlegend=True, height=420,
        )
        st.plotly_chart(fig_radar, use_container_width=True)
        st.caption("모든 축은 전국 동 대비 백분위(0~100)로 정규화된 값입니다. 100에 가까울수록 전국에서 상위권이라는 뜻이에요.")

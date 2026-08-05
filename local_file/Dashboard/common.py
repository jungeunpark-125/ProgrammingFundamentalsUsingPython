"""
Shared constants and helpers for the Ballot Dashboard Streamlit apps.

Used by app.py (main entry point) and every page under pages/. Pulling this
code into one module avoids the copy-paste drift risk of having the same
romanization / geojson-decoding / path-resolution logic duplicated across
several dashboard files.
"""

import glob
import os
import json
import re

import pandas as pd
import plotly.express as px
import streamlit as st
from korean_romanizer.romanizer import Romanizer

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
# Anchored on THIS file's own location (Dashboard/common.py), not on the
# caller's __file__. That way pages/*.py -- one directory deeper than
# app.py -- still resolve the project-level Data&Code / GeoData folders
# correctly, without every page needing its own "../.." vs "../../.." logic.
_DASHBOARD_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(_DASHBOARD_DIR, ".."))
DATA_DIR = os.path.join(PROJECT_ROOT, "Data&Code")
GEO_DIR = os.path.join(PROJECT_ROOT, "GeoData")
GEOJSON_PATH = os.path.join(GEO_DIR, "skorea-municipalities-2018-topo-simple.json")


def resolve_data_file(prefix, exclude_prefixes=()):
    """Locate a Data&Code CSV by its numeric filename prefix (e.g. "05_", "07_")
    instead of an exact Korean filename string.

    Some filesystems/uploaders (notably macOS) normalize Korean filenames to a
    decomposed (NFD) Unicode form, which is byte-different from the composed
    (NFC) form a source file might hardcode -- an exact-string path lookup can
    then fail even though the filename displays identically everywhere.
    Globbing on the ASCII numeric prefix avoids depending on that
    normalization entirely.
    """
    matches = sorted(
        p for p in glob.glob(os.path.join(DATA_DIR, f"{prefix}*.csv"))
        if not os.path.basename(p).startswith(exclude_prefixes)
    )
    if not matches:
        raise FileNotFoundError(
            f"'{prefix}*.csv' 패턴의 데이터 파일을 {DATA_DIR} 폴더에서 찾지 못했습니다. "
            "필요한 CSV가 Data&Code 폴더에 있는지 확인해주세요."
        )
    return matches[0]


# ---------------------------------------------------------------------------
# Sido / gu name maps
# ---------------------------------------------------------------------------
# geojson (southkorea-maps, GADM-based) "code" prefix (first 2 digits) -> sido name.
# This is NOT the official KOSTAT admin code scheme (e.g. "26" here = Ulsan, not Busan) --
# mapping was verified empirically against the dataset.
SIDO_CODE_PREFIX = {
    "11": "서울특별시", "21": "부산광역시", "22": "대구광역시", "23": "인천광역시",
    "24": "광주광역시", "25": "대전광역시", "26": "울산광역시", "29": "세종특별자치시",
    "31": "경기도", "32": "강원도", "33": "충청북도", "34": "충청남도",
    "35": "전라북도", "36": "전라남도", "37": "경상북도", "38": "경상남도",
    "39": "제주특별자치도",
}
# Name mismatches between geojson and dataset (renamed/merged districts)
NAME_ALIAS = {"세종시": "세종특별자치시"}

# English display names for sido (for UI labels; underlying joins still use Korean keys)
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
    if not isinstance(name, str) or not name:
        return name
    m = re.match(r"^(.*?)(\d*)(시|군|구|읍|면|동|리)$", name)
    if m:
        base, num, suffix = m.groups()
        base_r = Romanizer(base).romanize().capitalize() if base else ""
        suffix_en = _SUFFIX_EN[suffix]
        return f"{base_r} {num}-{suffix_en}".strip() if num else f"{base_r}-{suffix_en}".strip()
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
def build_name_maps(df, gu_col="구시군명", dong_col="읍면동명"):
    gu_map = {n: romanize_name(n) for n in df[gu_col].dropna().unique()}
    dong_map = {n: romanize_name(n) for n in df[dong_col].dropna().unique()}
    return gu_map, dong_map


# ---------------------------------------------------------------------------
# GeoJSON loading (TopoJSON -> GeoJSON, hand-decoded, no extra topojson package)
# ---------------------------------------------------------------------------
@st.cache_data
def load_geojson(path=GEOJSON_PATH):
    """Convert a southkorea-maps-style TopoJSON into a GeoJSON FeatureCollection.
    Returns None if the file isn't there yet -- callers should handle that by
    showing a placeholder instead of a map.
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
            geom = {"type": "Polygon", "coordinates": [resolve_ring(r) for r in g["arcs"]]}
        elif gtype == "MultiPolygon":
            geom = {
                "type": "MultiPolygon",
                "coordinates": [[resolve_ring(r) for r in poly] for poly in g["arcs"]],
            }
        else:
            continue

        props = dict(g.get("properties", {}))
        # Gu/gun names alone repeat across sido (e.g. "동구"/"중구"/"서구" exist in
        # several cities), so we rebuild a "sido_gu" join key from the code prefix
        # to avoid mis-joins.
        code = str(props.get("code", ""))
        sido_name = SIDO_CODE_PREFIX.get(code[:2], "")
        gu_name = NAME_ALIAS.get(props.get("name", ""), props.get("name", ""))
        props["join_key"] = f"{sido_name}_{gu_name}"
        features.append({"type": "Feature", "properties": props, "geometry": geom})

    return {"type": "FeatureCollection", "features": features}


# ---------------------------------------------------------------------------
# Language toggle
# ---------------------------------------------------------------------------
# One shared session_state key so the ENG/KOR choice is a single app-wide
# setting: it stays put when the user switches between pages via the sidebar
# nav, and each page just renders the same small widget top-right to read/change it.
_LANG_WIDGET_KEY = "lang_widget"


def language_toggle():
    """Render a small KOR/ENG control in the top-right corner of the page and
    return "ko" or "en". Backed by st.session_state so the choice persists
    across reruns and page switches.
    """
    if _LANG_WIDGET_KEY not in st.session_state:
        st.session_state[_LANG_WIDGET_KEY] = "ENG"

    _left, right = st.columns([6, 1])
    with right:
        st.segmented_control(
            "Language",
            ["KOR", "ENG"],
            key=_LANG_WIDGET_KEY,
            label_visibility="collapsed",
        )

    return "ko" if st.session_state[_LANG_WIDGET_KEY] == "KOR" else "en"


def get_current_lang():
    """Read the current language choice without rendering the toggle widget.

    app.py (the router) needs this to localize the sidebar title and the nav
    item labels -- those are drawn before the selected page (and its own
    language_toggle() call) actually runs.
    """
    return "ko" if st.session_state.get(_LANG_WIDGET_KEY, "ENG") == "KOR" else "en"


# ---------------------------------------------------------------------------
# Small formatting / chart helpers
# ---------------------------------------------------------------------------
def format_int(value):
    if pd.isna(value):
        return "-"
    return f"{int(round(value)):,}"


# ---------------------------------------------------------------------------
# Forecast dashboard: safety-level column lookups
# (Shared by pages/2_Forecast_Dashboard.py and pages/3_한국어_예측_대시보드.py --
# the underlying CSV column names are Korean regardless of the page's UI
# language, so this logic doesn't need a language-specific copy.)
# ---------------------------------------------------------------------------
SAFETY_LEVELS = ["95%", "99%", "99.9%"]
SAFETY_SUFFIX = {"95%": "95", "99%": "99", "99.9%": "999"}


def get_backtest_columns(level):
    suffix = SAFETY_SUFFIX[level]
    return {
        "prep_rate": f"주모형_준비율_{suffix}",
        "prepared": f"주모형_준비수량_{suffix}",
        "shortage": f"부족수량_{suffix}",
        "surplus": f"잔여수량_{suffix}",
        "covered": f"충족여부_{suffix}",
    }


def get_forecast9_columns(level, national):
    """national=True -> 전국가중모형 (national weighted model), False -> 지역가중모형 (local weighted model)."""
    suffix = SAFETY_SUFFIX[level]
    prefix = "전국가중모형" if national else "지역가중모형"
    return {
        "forecast_rate": f"{prefix}_예측당일투표율",
        "forecast_ballots": f"{prefix}_예측투표용지수",
        "prep_rate": f"{prefix}_준비율_{suffix}",
        "prepared": f"{prefix}_준비수량_{suffix}",
    }


def choropleth_map(
    data_frame,
    geojson,
    color_col,
    hover_data,
    *,
    color_continuous_scale="Reds",
    color_continuous_midpoint=None,
    title=None,
    height=470,
    opacity=0.78,
    zoom=5.8,
    center=None,
    margin=None,
    labels=None,
):
    """Shared choropleth builder. Handles the plotly-version fallback
    (choropleth_map on newer maplibre-based plotly, choropleth_mapbox on
    older installs) in one place instead of duplicating it per page.
    """
    if geojson is None or data_frame.empty:
        return None

    kwargs = dict(
        data_frame=data_frame,
        geojson=geojson,
        locations="join_key",
        featureidkey="properties.join_key",
        color=color_col,
        color_continuous_scale=color_continuous_scale,
        zoom=zoom,
        center=center or {"lat": 36.3, "lon": 127.8},
        opacity=opacity,
        hover_data=hover_data,
        labels=labels or {},
    )
    if color_continuous_midpoint is not None:
        kwargs["color_continuous_midpoint"] = color_continuous_midpoint

    if hasattr(px, "choropleth_map"):
        fig = px.choropleth_map(map_style="carto-positron", **kwargs)
    else:
        fig = px.choropleth_mapbox(mapbox_style="carto-positron", **kwargs)

    fig.update_layout(
        margin=margin if margin is not None else dict(l=0, r=0, t=35 if title else 0, b=0),
        height=height,
        title=title,
    )
    return fig

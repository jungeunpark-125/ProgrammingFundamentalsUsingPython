"""
Ballot Dashboard -- single entry point.

Run: streamlit run app.py

This file is just the router: it sets the page config once, draws an app
title above the sidebar nav, and wires up the two pages (risk dashboard,
forecast dashboard). Each page renders its own ENG/KOR toggle top-right
(see common.language_toggle) instead of having separate EN/KO page files --
so the sidebar only ever shows two entries, not four.

Page content lives in:
  - pages/risk_dashboard.py
  - pages/forecast_dashboard.py
Shared helpers/constants live in common.py.
"""

import streamlit as st

from common import get_current_lang

st.set_page_config(page_title="Ballot Dashboard", layout="wide")

# The nav is drawn here, before the selected page (and its own language
# toggle) runs -- so it reads the last-known language choice from
# session_state directly instead of rendering the toggle itself.
lang = get_current_lang()

with st.sidebar:
    st.title("🗳️ Ballot Dashboard" if lang == "en" else "🗳️ 투표용지 대시보드")

risk_page = st.Page(
    "pages/risk_dashboard.py",
    title="Risk Dashboard" if lang == "en" else "위험 대시보드",
    icon="⚠️",
    default=True,
)
forecast_page = st.Page(
    "pages/forecast_dashboard.py",
    title="Forecast Dashboard" if lang == "en" else "예측 대시보드",
    icon="🔮",
)

pg = st.navigation([risk_page, forecast_page])
pg.run()

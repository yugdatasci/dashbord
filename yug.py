# app.py
import streamlit as st
import pandas as pd
import numpy as np
import requests
import plotly.express as px
from datetime import datetime
from fredapi import Fred

st.set_page_config(layout="wide", page_title="Monetary Policy & Inflation Dashboard")

# --------------------
# CONFIG / API KEYS
# --------------------
FRED_API_KEY = "YOUR_FRED_API_KEY"   # get one at https://fred.stlouisfed.org
fred = Fred(api_key=FRED_API_KEY)

# --------------------
# HELPERS
# --------------------
@st.cache_data(ttl=3600)
def get_fred_series(series_id, start=None, end=None):
    """Return a pandas Series from FRED as a DataFrame with date index."""
    s = fred.get_series(series_id, observation_start=start, observation_end=end)
    df = s.rename(series_id).to_frame()
    df.index = pd.to_datetime(df.index)
    return df

@st.cache_data(ttl=3600)
def get_worldbank_inflation(country_iso3="WLD", indicator="FP.CPI.TOTL.ZG", startyear=2000, endyear=2025):
    url = f"https://api.worldbank.org/v2/country/{country_iso3}/indicator/{indicator}?date={startyear}:{endyear}&format=json&per_page=200"
    r = requests.get(url)
    data = r.json()
    if not data or len(data) < 2:
        return pd.DataFrame()
    rows = []
    for item in data[1]:
        rows.append({"date": item["date"], "value": item["value"]})
    df = pd.DataFrame(rows)
    df['date'] = pd.to_datetime(df['date'] + "-01-01")
    df = df.sort_values("date").set_index("date")
    return df

def yoy_pct_change(df, periods=12):
    return df.pct_change(periods=periods) * 100

# --------------------
# LAYOUT: Sidebar
# --------------------
st.sidebar.title("Controls")
start_date = st.sidebar.date_input("Start date", value=pd.to_datetime("2015-01-01"))
end_date = st.sidebar.date_input("End date", value=pd.to_datetime(datetime.today()))
country_compare = st.sidebar.multiselect("Compare countries (World Bank 3-letter ISO)", ["IND", "USA", "CHN", "GBR", "WLD"], default=["IND", "USA"])
show_forecast = st.sidebar.checkbox("Show simple CPI nowcast (illustrative)", value=False)

# --------------------
# MAIN: Title + KPI
# --------------------
st.title("Monetary Policy & Inflation Dashboard — Student Project")
st.markdown("Data sources: FRED (US CPI, M2, Fed funds), MoSPI (India CPI), World Bank (cross-country). See code comments for series IDs.")

# Fetch core series (US CPI, US M2, Fed funds, India CPI via FRED if available)
with st.spinner("Fetching data..."):
    # US CPI index (All items) and core CPI
    us_cpi = get_fred_series("CPIAUCSL", start=start_date.isoformat(), end=end_date.isoformat())
    us_core = get_fred_series("CPILFESL", start=start_date.isoformat(), end=end_date.isoformat())
    # US liquidity M2
    us_m2 = get_fred_series("M2SL", start=start_date.isoformat(), end=end_date.isoformat())
    # Fed funds
    fedfunds = get_fred_series("FEDFUNDS", start=start_date.isoformat(), end=end_date.isoformat())
    # Attempt to fetch India CPI from FRED (if present) or fallback to worldbank annual inflation
    try:
        ind_cpi = get_fred_series("INPCPIALLQINMEI", start=start_date.isoformat(), end=end_date.isoformat())
        # (Note: FRED ID may vary; if missing, we get worldbank inflation below)
    except Exception:
        ind_cpi = pd.DataFrame()

# Compute YoY inflation
us_cpi_yoy = yoy_pct_change(us_cpi)
us_core_yoy = yoy_pct_change(us_core)

# KPIs
col1, col2, col3, col4 = st.columns(4)
latest_us_infl = us_cpi_yoy.dropna().iloc[-1,0] if not us_cpi_yoy.dropna().empty else np.nan
latest_us_core = us_core_yoy.dropna().iloc[-1,0] if not us_core_yoy.dropna().empty else np.nan
latest_m2_growth = ((us_m2 / us_m2.shift(12) - 1)*100).dropna().iloc[-1,0] if not us_m2.dropna().empty else np.nan
latest_fedfunds = fedfunds.dropna().iloc[-1,0] if not fedfunds.dropna().empty else np.nan

col1.metric("US CPI YoY (%)", f"{latest_us_infl:.2f}")
col2.metric("US Core CPI YoY (%)", f"{latest_us_core:.2f}")
col3.metric("US M2 YoY (%)", f"{latest_m2_growth:.2f}")
col4.metric("Fed funds rate (%)", f"{latest_fedfunds:.2f}")

# --------------------
# PLOTS
# --------------------
st.header("US: CPI and Core CPI")
fig_us = px.line(pd.concat([us_cpi, us_core], axis=1).dropna(), labels={"index": "Date"}, title="US CPI Index (level)")
st.plotly_chart(fig_us, use_container_width=True)

st.subheader("US: YoY inflation (12-month)")
fig_us_yoy = px.line(pd.concat([us_cpi_yoy, us_core_yoy], axis=1).dropna(), labels={"index":"Date"}, title="US YoY inflation (%)")
st.plotly_chart(fig_us_yoy, use_container_width=True)

st.header("US Liquidity (M2)")
fig_m2 = px.line(us_m2, labels={"index":"Date", "value":"M2 (USD)"}, title="US M2 Money Supply (level)")
st.plotly_chart(fig_m2, use_container_width=True)

st.header("Monetary policy (Fed funds)")
fig_ff = px.line(fedfunds, labels={"index":"Date", "value":"Fed funds (%)"}, title="Effective Federal Funds Rate")
st.plotly_chart(fig_ff, use_container_width=True)

# --------------------
# World / Cross-country inflation
# --------------------
st.header("Cross-country inflation (World Bank)")
wb_frames = []
for iso in country_compare:
    df_wb = get_worldbank_inflation(country_iso3=iso, startyear=2010, endyear=end_date.year)
    df_wb = df_wb.rename(columns={"value": iso})
    wb_frames.append(df_wb[iso])

if wb_frames:
    df_wb_all = pd.concat(wb_frames, axis=1).dropna()
    fig_wb = px.line(df_wb_all, title="Annual inflation (World Bank): selected countries")
    st.plotly_chart(fig_wb, use_container_width=True)

# --------------------
# Simple nowcast (optional)
# --------------------
if show_forecast:
    st.header("Illustrative nowcast: US CPI (simple EWMA)")
    # simple exponential weighted mean of last 3 months growth as naive nowcast
    us_cpi_monthly = us_cpi.resample("M").last().dropna()
    us_cpi_yoy_monthly = us_cpi_monthly.pct_change(12)*100
    last = us_cpi_yoy_monthly.dropna().iloc[-12:]
    nowcast = last.ewm(span=3).mean().iloc[-1,0]
    st.write(f"Simple EWMA nowcast for US CPI YoY (illustrative): **{nowcast:.2f}%**")
    fig_now = px.line(us_cpi_yoy_monthly, title="US CPI YoY (monthly) — with nowcast label")
    st.plotly_chart(fig_now, use_container_width=True)

st.markdown("---")
st.markdown("**Notes & reproducibility**: This dashboard uses FRED API for US series (requires API key). For India CPI use MoSPI/official releases (you can download monthly excel from MOSPI and import). World Bank API is used for cross-country annual inflation. Cite sources when presenting this in your internship submission.")

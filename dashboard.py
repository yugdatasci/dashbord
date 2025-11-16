# dashboard.py
# Fixed and cleaned Monetary Policy & Inflation Dashboard
# Requirements: pip install streamlit pandas plotly requests openpyxl

import streamlit as st
st.set_page_config(page_title="Monetary Policy & Inflation Dashboard", layout="wide")

import pandas as pd
import numpy as np
import requests
import plotly.express as px
from datetime import datetime, timedelta
import io

# -------------------------
# Helper functions
# -------------------------
@st.cache_data(ttl=3600)
def fetch_fred_series(series_id: str, api_key: str, start=None, end=None):
    """Fetch a series from FRED using the HTTP API and return DataFrame with datetime index."""
    if not api_key:
        return pd.DataFrame()
    base = "https://api.stlouisfed.org/fred/series/observations"
    params = {"series_id": series_id, "api_key": api_key, "file_type": "json"}
    if start:
        params["observation_start"] = start
    if end:
        params["observation_end"] = end
    r = requests.get(base, params=params, timeout=30)
    r.raise_for_status()
    payload = r.json()
    obs = payload.get("observations", [])
    rows = []
    for o in obs:
        date = o.get("date")
        val = o.get("value")
        try:
            v = float(val)
        except Exception:
            v = np.nan
        rows.append({"date": date, series_id: v})
    df = pd.DataFrame(rows)
    if df.empty:
        return pd.DataFrame()
    df["date"] = pd.to_datetime(df["date"])
    df = df.set_index("date").sort_index()
    return df

@st.cache_data(ttl=3600)
def fetch_worldbank_inflation(iso3: str, startyear=2010, endyear=None):
    if endyear is None:
        endyear = datetime.today().year
    url = f"https://api.worldbank.org/v2/country/{iso3}/indicator/FP.CPI.TOTL.ZG"
    params = {"date": f"{startyear}:{endyear}", "format": "json", "per_page": 200}
    r = requests.get(url, params=params, timeout=30)
    r.raise_for_status()
    data = r.json()
    if not data or len(data) < 2:
        return pd.DataFrame()
    rows = []
    for item in data[1]:
        year = item.get("date")
        val = item.get("value")
        if val is None:
            continue
        rows.append({"year": int(year), iso3: float(val)})
    df = pd.DataFrame(rows)
    if df.empty:
        return pd.DataFrame()
    df = df.set_index("year").sort_index()
    return df


def yoy_pct_change(df: pd.DataFrame, periods=12):
    return df.pct_change(periods=periods) * 100


def df_to_csv_bytes(df: pd.DataFrame):
    buf = io.StringIO()
    df.to_csv(buf)
    return buf.getvalue().encode('utf-8')

# -------------------------
# Sidebar controls
# -------------------------
st.sidebar.header("Controls & Data sources")

# FRED API key: try secrets then ask user
default_fred = st.secrets.get("FRED_API_KEY") if "FRED_API_KEY" in st.secrets else ""
fred_key = st.sidebar.text_input("FRED API key (optional for US series)", value=default_fred, type="password")

# Date range
today = datetime.today().date()
default_start = (today - timedelta(days=365 * 5)).isoformat()
start_date = st.sidebar.date_input("Start date", value=pd.to_datetime(default_start))
end_date = st.sidebar.date_input("End date", value=pd.to_datetime(today))

# Countries for world comparison
countries = st.sidebar.multiselect("Compare countries (World Bank ISO3)", ["IND", "USA", "CHN", "GBR", "DEU", "WLD"], default=["IND", "USA"])

# India CPI: allow upload
st.sidebar.markdown("---")
st.sidebar.subheader("India CPI (MoSPI)")
uploaded_file = st.sidebar.file_uploader("Upload MoSPI Excel/CSV (monthly CPI index)", type=["xls", "xlsx", "csv"])

# Options
show_forecast = st.sidebar.checkbox("Show simple CPI nowcast (illustrative)", value=False)

# -------------------------
# Main layout
# -------------------------
st.title("Monetary Policy & Inflation Dashboard")
st.caption("Student project — CPI, world inflation, US liquidity, and monetary policy indicators")

col1, col2 = st.columns([3, 1])
with col2:
    st.markdown("**Actions**")
    st.write(" - Provide FRED key to fetch US series.")
    st.write(" - Upload MoSPI CPI if you have the file.")

# -------------------------
# Fetch US series from FRED (if key provided)
# -------------------------
us_cpi = pd.DataFrame()
us_core = pd.DataFrame()
us_m2 = pd.DataFrame()
fedfunds = pd.DataFrame()

if fred_key:
    try:
        st.info("Fetching US series from FRED...")
        us_cpi = fetch_fred_series("CPIAUCSL", fred_key, start=start_date.isoformat(), end=end_date.isoformat())
        us_core = fetch_fred_series("CPILFESL", fred_key, start=start_date.isoformat(), end=end_date.isoformat())
        us_m2 = fetch_fred_series("M2SL", fred_key, start=start_date.isoformat(), end=end_date.isoformat())
        fedfunds = fetch_fred_series("FEDFUNDS", fred_key, start=start_date.isoformat(), end=end_date.isoformat())
        st.success("US series fetched.")
    except Exception as e:
        st.error(f"Error fetching FRED data: {e}")

# -------------------------
# India CPI: upload parsing
# -------------------------
ind_cpi = pd.DataFrame()
if uploaded_file is not None:
    try:
        st.info("Reading uploaded file...")
        if uploaded_file.name.lower().endswith((".xls", ".xlsx")):
            xls = pd.read_excel(uploaded_file, sheet_name=None)
            # pick first sheet and try to parse common patterns
            sheetname = list(xls.keys())[0]
            df_try = xls[sheetname].copy()
            # try to find date and CPI/index columns
            date_col = None
            value_col = None
            for c in df_try.columns:
                # simple heuristics
                if pd.api.types.is_datetime64_any_dtype(df_try[c]):
                    date_col = c
                if isinstance(c, str) and ("cpi" in c.lower() or "index" in c.lower()):
                    value_col = c
            if date_col is None:
                # try first column as date
                df_try.iloc[:, 0] = pd.to_datetime(df_try.iloc[:, 0], errors="coerce")
                date_col = df_try.columns[0]
            if value_col is None and df_try.shape[1] >= 2:
                value_col = df_try.columns[1]
            # build ind_cpi
            df_try = df_try[[date_col, value_col]].dropna()
            df_try.columns = ["date", "CPI"]
            df_try["date"] = pd.to_datetime(df_try["date"], errors="coerce")
            ind_cpi = df_try.set_index("date").sort_index()
        else:
            # csv
            df_try = pd.read_csv(uploaded_file)
            # try common names
            cols_lower = [str(c).lower() for c in df_try.columns]
            date_candidates = [c for c in df_try.columns if str(c).lower() in ("date", "month", "period")]
            value_candidates = [c for c in df_try.columns if "cpi" in str(c).lower() or "index" in str(c).lower()]
            if date_candidates and value_candidates:
                dc = date_candidates[0]
                vc = value_candidates[0]
                df_try[dc] = pd.to_datetime(df_try[dc], errors="coerce")
                ind_cpi = df_try[[dc, vc]].dropna()
                ind_cpi.columns = ["date", "CPI"]
                ind_cpi = ind_cpi.set_index("date").sort_index()
            else:
                # fallback: assume first two columns
                df_try.iloc[:, 0] = pd.to_datetime(df_try.iloc[:, 0], errors="coerce")
                ind_cpi = df_try.iloc[:, :2].dropna()
                ind_cpi.columns = ["date", "CPI"]
                ind_cpi = ind_cpi.set_index("date").sort_index()
        if not ind_cpi.empty:
            st.success("India CPI parsed from uploaded file.")
        else:
            st.warning("Uploaded file could not be parsed automatically; ensure it has a date and CPI/index column.")
    except Exception as e:
        st.error(f"Error reading uploaded file: {e}")

# -------------------------
# Derived series
# -------------------------
us_cpi_yoy = yoy_pct_change(us_cpi) if not us_cpi.empty else pd.DataFrame()
us_core_yoy = yoy_pct_change(us_core) if not us_core.empty else pd.DataFrame()
us_m2_yoy = (us_m2 / us_m2.shift(12) - 1) * 100 if not us_m2.empty else pd.DataFrame()
ind_cpi_yoy = (ind_cpi["CPI"].pct_change(periods=12) * 100).to_frame("India CPI YoY") if not ind_cpi.empty else pd.DataFrame()

# -------------------------
# KPIs
# -------------------------
st.markdown("## Key indicators (latest)")

k1, k2, k3, k4 = st.columns(4)

def last_value(df, col=None):
    if df.empty:
        return None
    if col:
        s = df[col].dropna()
    else:
        s = df.dropna().iloc[:, 0] if not df.empty else pd.Series()
    if s.empty:
        return None
    return s.iloc[-1]

with k1:
    v = last_value(us_cpi_yoy, None)
    if v is not None:
        st.metric("US CPI YoY (%)", f"{v:.2f}")
    else:
        st.write("US CPI YoY: -")

with k2:
    v = last_value(us_core_yoy, None)
    if v is not None:
        st.metric("US Core CPI YoY (%)", f"{v:.2f}")
    else:
        st.write("US Core CPI YoY: -")

with k3:
    v = last_value(us_m2_yoy, None)
    if v is not None:
        # handle Series vs scalar
        try:
            display_val = v.iloc[0] if isinstance(v, pd.Series) else v
            st.metric("US M2 YoY (%)", f"{display_val:.2f}")
        except Exception:
            st.metric("US M2 YoY (%)", f"{v:.2f}")
    else:
        st.write("US M2 YoY: -")

with k4:
    v = last_value(fedfunds, None)
    if v is not None:
        st.metric("Fed funds (%)", f"{v:.2f}")
    else:
        st.write("Fed funds: -")

# -------------------------
# Plots area
# -------------------------
st.markdown("---")
st.header("US: CPI & Liquidity")

# US CPI levels plot
if not us_cpi.empty and not us_core.empty:
    # rename columns to friendly names
    left = us_cpi.copy()
    right = us_core.copy()
    left.columns = ["US CPI"]
    right.columns = ["US

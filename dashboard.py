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
    """Fetch annual inflation from World Bank for a given ISO3 code."""
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
    """12-month year-over-year % change."""
    if df.empty:
        return pd.DataFrame()
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
fred_key = st.sidebar.text_input(
    "FRED API key (optional for US series)",
    value=default_fred,
    type="password"
)

# Date range
today = datetime.today().date()
default_start = (today - timedelta(days=365 * 5)).isoformat()
start_date = st.sidebar.date_input("Start date", value=pd.to_datetime(default_start))
end_date = st.sidebar.date_input("End date", value=pd.to_datetime(today))

# Countries for world comparison
countries = st.sidebar.multiselect(
    "Compare countries (World Bank ISO3)",
    ["IND", "USA", "CHN", "GBR", "DEU", "WLD"],
    default=["IND", "USA"]
)

# India CPI: allow upload
st.sidebar.markdown("---")
st.sidebar.subheader("India CPI (MoSPI)")
uploaded_file = st.sidebar.file_uploader(
    "Upload MoSPI Excel/CSV (monthly CPI index)",
    type=["xls", "xlsx", "csv"]
)

# Options
show_forecast = st.sidebar.checkbox(
    "Show simple CPI nowcast (illustrative)",
    value=False
)

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
        s = df.dropna().iloc[:, 0] if not df.empty else pd.Series(dtype=float)
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
    right.columns = ["US Core CPI"]
    df_levels = pd.concat([left, right], axis=1).dropna()
    fig = px.line(
        df_levels,
        labels={"value": "Index", "index": "Date"},
        title="US CPI (index) and Core CPI"
    )
    st.plotly_chart(fig, use_container_width=True)
elif fred_key:
    st.info("US CPI not available: check your FRED key or date range.")

# US CPI YoY
if not us_cpi.empty:
    if not us_core.empty:
        df_yoy = pd.concat([us_cpi_yoy, us_core_yoy], axis=1).dropna()
        df_yoy.columns = ["US CPI YoY", "US Core CPI YoY"]
        fig = px.line(
            df_yoy,
            labels={"value": "YoY %", "index": "Date"},
            title="US YoY inflation (%)"
        )
        st.plotly_chart(fig, use_container_width=True)
    else:
        if not us_cpi_yoy.empty:
            fig = px.line(
                us_cpi_yoy,
                labels={"value": "YoY %", "index": "Date"},
                title="US CPI YoY (%)"
            )
            st.plotly_chart(fig, use_container_width=True)
else:
    if fred_key:
        st.info("US CPI YoY not available.")

# US Liquidity (M2) and Fed funds
colA, colB = st.columns(2)
with colA:
    if not us_m2.empty:
        fig = px.line(us_m2, title="US M2 Money Supply (Level)")
        st.plotly_chart(fig, use_container_width=True)
        if not us_m2_yoy.empty:
            fig2 = px.line(us_m2_yoy, title="US M2 YoY (%)")
            st.plotly_chart(fig2, use_container_width=True)
    else:
        st.write("US M2 not available.")
with colB:
    if not fedfunds.empty:
        fig = px.line(fedfunds, title="Effective Fed Funds Rate")
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.write("Fed funds not available.")

# -------------------------
# India CPI section
# -------------------------
st.markdown("---")
st.header("India CPI (MoSPI)")

if not ind_cpi.empty:
    st.write("Showing India CPI series parsed from upload.")
    st.line_chart(ind_cpi["CPI"])
    if not ind_cpi_yoy.empty:
        st.line_chart(ind_cpi_yoy)
    csv_bytes = df_to_csv_bytes(ind_cpi)
    st.download_button(
        "Download India CPI (.csv)",
        csv_bytes,
        file_name="india_cpi.csv"
    )
else:
    st.info("India CPI not available. Upload the MoSPI monthly CPI index (Excel/CSV) using the left sidebar.")

# -------------------------
# Cross-country annual inflation (World Bank)
# -------------------------
st.markdown("---")
st.header("Cross-country annual inflation (World Bank)")

if countries:
    wb_frames = []
    for iso in countries:
        try:
            wb = fetch_worldbank_inflation(iso, startyear=2010)
            if not wb.empty:
                wb_frames.append(wb[iso])
        except Exception:
            st.warning(f"World Bank fetch failed for {iso}")
    if wb_frames:
        df_wb = pd.concat(wb_frames, axis=1).dropna()
        fig = px.line(
            df_wb,
            title="Annual inflation (World Bank): selected countries"
        )
        st.plotly_chart(fig, use_container_width=True)
        csv_bytes = df_to_csv_bytes(df_wb)
        st.download_button(
            "Download selected countries inflation (.csv)",
            csv_bytes,
            file_name="worldbank_inflation.csv"
        )
    else:
        st.info("No World Bank data available for selected countries.")
else:
    st.write("Select countries in the sidebar to compare.")

# -------------------------
# Simple nowcast (optional)
# -------------------------
st.markdown("---")
if show_forecast:
    st.header("Illustrative nowcast: US CPI (simple EWMA)")
    if not us_cpi.empty:
        us_cpi_monthly = us_cpi.resample("M").last().dropna()
        us_cpi_yoy_monthly = us_cpi_monthly.pct_change(12) * 100
        us_cpi_yoy_monthly = us_cpi_yoy_monthly.dropna()
        if not us_cpi_yoy_monthly.empty:
            nowcast = us_cpi_yoy_monthly.ewm(span=3).mean().iloc[-1, 0]
            st.write(f"Simple EWMA nowcast for US CPI YoY (illustrative): **{nowcast:.2f}%**")
            fig_now = px.line(
                us_cpi_yoy_monthly,
                title="US CPI YoY (monthly) — with nowcast label"
            )
            st.plotly_chart(fig_now, use_container_width=True)
        else:
            st.info("Not enough monthly US CPI data for nowcast.")
    else:
        st.info("US CPI data required for nowcast. Provide a FRED API key in the sidebar.")

# -------------------------
# Footer
# -------------------------
st.markdown("---")
st.caption(
    "Data sources: FRED (US series), World Bank (country inflation), "
    "MoSPI (India CPI via upload). This dashboard is for educational/demo purposes. "
    "Cite original sources when sharing."
)

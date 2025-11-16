# dashboard.py
# Monetary Policy & Inflation Dashboard (full single-file app)
# Requirements: pip install streamlit pandas plotly requests openpyxl

import streamlit as st
st.set_page_config(page_title="Monetary Policy & Inflation Dashboard", layout="wide")

import pandas as pd
import numpy as np
import requests
from datetime import datetime, timedelta
import plotly.express as px
import io

# -------------------------
# Helper functions
# -------------------------
@st.cache_data(ttl=3600)
def fetch_fred_series(series_id: str, api_key: str, start=None, end=None):
    """Fetch a series from FRED using the HTTP API and return DataFrame with datetime index and a column = series_id"""
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
    """Fetch annual inflation (FP.CPI.TOTL.ZG) from World Bank for given ISO3 country."""
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
st.sidebar.write("If you don't provide a FRED key, US series will not load. You can upload CSVs as alternative.")

# Date range
today = datetime.today().date()
default_start = (today - timedelta(days=365*5)).isoformat()
start_date = st.sidebar.date_input("Start date", value=pd.to_datetime(default_start))
end_date = st.sidebar.date_input("End date", value=pd.to_datetime(today))

# Countries for world comparison
countries = st.sidebar.multiselect("Compare countries (World Bank ISO3)", ["IND", "USA", "CHN", "GBR", "DEU", "WLD"], default=["IND", "USA"])

# India CPI: attempt MOSPI auto-fetch or allow upload
st.sidebar.markdown("---")
st.sidebar.subheader("India CPI (MoSPI)")
mospi_auto = st.sidebar.checkbox("Try auto-download latest MOSPI CPI (best-effort)", value=False)
uploaded_file = st.sidebar.file_uploader("Or upload MoSPI Excel/CSV (monthly CPI index)", type=["xls", "xlsx", "csv"])

# Options
show_forecast = st.sidebar.checkbox("Show simple CPI nowcast (illustrative)", value=False)

# -------------------------
# Main layout
# -------------------------
st.title("Monetary Policy & Inflation Dashboard")
st.caption("Student project — CPI, world inflation, US liquidity, and monetary policy indicators")

col1, col2 = st.columns([3,1])

with col2:
    st.markdown("**Actions**")
    st.write(" - Provide FRED key to fetch US series.")
    st.write(" - Upload MoSPI CPI if auto-download fails.")
    st.write(" - Use date controls and country pickers.")

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
        # series IDs
        us_cpi = fetch_fred_series("CPIAUCSL", fred_key, start=start_date.isoformat(), end=end_date.isoformat())
        us_core = fetch_fred_series("CPILFESL", fred_key, start=start_date.isoformat(), end=end_date.isoformat())
        us_m2 = fetch_fred_series("M2SL", fred_key, start=start_date.isoformat(), end=end_date.isoformat())
        fedfunds = fetch_fred_series("FEDFUNDS", fred_key, start=start_date.isoformat(), end=end_date.isoformat())
        st.success("US series fetched.")
    except Exception as e:
        st.error(f"Error fetching FRED data: {e}")

# -------------------------
# India CPI: MOSPI attempt or upload fallback
# -------------------------
ind_cpi = pd.DataFrame()
if mospi_auto and not uploaded_file:
    try:
        st.info("Attempting to download MOSPI CPI (monthly index) — best-effort")
        # NOTE: MOSPI site structure can change. Attempt common excel URL; keep best-effort.
        mospi_url = "http://mospi.nic.in/sites/default/files/press_release/CPI_India_0.xls"
        r = requests.get(mospi_url, timeout=15)
        r.raise_for_status()
        # read excel into pandas
        ind_cpi = pd.read_excel(io.BytesIO(r.content), engine="openpyxl", header=None)
        st.success("Downloaded a MOSPI file (raw). Please upload your official file for robust parsing if needed.")
        # we do not attempt automatic parsing of unknown-format table here; encourage upload
        ind_cpi = pd.DataFrame()  # leave empty because auto-parsing is risky
    except Exception:
        st.warning("Auto-download failed or MOSPI layout unsupported. Please upload the official MoSPI Excel/CSV below.")
        ind_cpi = pd.DataFrame()

if uploaded_file is not None:
    try:
        st.info("Reading uploaded file...")
        if uploaded_file.name.lower().endswith((".xls", ".xlsx")):
            # Try to guess the sheet and columns that contain monthly CPI index
            xls = pd.read_excel(uploaded_file, sheet_name=None)
            # Try to pick largest sheet
            sheetname = list(xls.keys())[0]
            df_try = xls[sheetname]
            # attempt to find date-like column and CPI column
            # Common MoSPI extracts have "Year" + months or "Month" column; we'll handle a couple patterns
            if "Year" in df_try.columns and any(col for col in df_try.columns if isinstance(col, str) and col.strip()[:3].lower() in ["jan","feb","mar"]):
                # Wide format (Year + month columns)
                df_wide = df_try.set_index("Year")
                rows = []
                for year, row in df_wide.iterrows():
                    for m_name in df_wide.columns:
                        try:
                            # map month names to month numbers
                            month_num = datetime.strptime(m_name[:3], "%b").month
                        except Exception:
                            continue
                        val = row[m_name]
                        rows.append({"date": pd.Timestamp(year=int(year), month=month_num, day=1), "CPI": val})
                ind_cpi = pd.DataFrame(rows).dropna().set_index("date").sort_index()
            else:
                # Try long form: a Date column + Index column
                # heuristics
                date_col = None
                value_col = None
                for c in df_try.columns:
                    if pd.api.types.is_datetime64_any_dtype(df_try[c]) or df_try[c].astype(str).str.match(r"\d{4}-\d{2}-\d{2}").any():
                        date_col = c
                        break
                if date_col is None:
                    # find numeric column and treat first col as year-month
                    for c in df_try.columns:
                        if df_try[c].dtype in [float, int]:
                            value_col = c
                            break
                # try common names
                for possible in ["Date", "Month", "Period"]:
                    if possible in df_try.columns:
                        date_col = possible
                for possible in ["Index", "CPI", "All-India CPI (Base 2012=100)"]:
                    if possible in df_try.columns:
                        value_col = possible
                if date_col and value_col:
                    df_try[date_col] = pd.to_datetime(df_try[date_col], errors="coerce")
                    ind_cpi = df_try[[date_col, value_col]].dropna()
                    ind_cpi.columns = ["date", "CPI"]
                    ind_cpi = ind_cpi.set_index("date").sort_index()
                else:
                    # fallback: try to parse any datetime-like first column and numeric second column
                    df_try = df_try.dropna(how="all")
                    cols = list(df_try.columns)
                    if len(cols) >= 2:
                        try:
                            df_try[cols[0]] = pd.to_datetime(df_try[cols[0]], errors="coerce")
                            ind_cpi = df_try[[cols[0], cols[1]]].dropna()
                            ind_cpi.columns = ["date", "CPI"]
                            ind_cpi = ind_cpi.set_index("date").sort_index()
                        except Exception:
                            ind_cpi = pd.DataFrame()
        else:
            # csv
            df_try = pd.read_csv(uploaded_file)
            # attempt to identify columns
            if "date" in map(str.lower, df_try.columns):
                dc = [c for c in df_try.columns if c.lower()=="date"][0]
                vc = [c for c in df_try.columns if "cpi" in c.lower() or "index" in c.lower()]
                if vc:
                    vc = vc[0]
                    df_try[dc] = pd.to_datetime(df_try[dc], errors="coerce")
                    ind_cpi = df_try[[dc, vc]].dropna()
                    ind_cpi.columns = ["date", "CPI"]
                    ind_cpi = ind_cpi.set_index("date").sort_index()
            # otherwise fallback: try first two cols
            if ind_cpi.empty:
                try:
                    df_try.iloc[:,0] = pd.to_datetime(df_try.iloc[:,0], errors="coerce")
                    ind_cpi = df_try[[df_try.columns[0], df_try.columns[1]]].dropna()
                    ind_cpi.columns = ["date", "CPI"]
                    ind_cpi = ind_cpi.set_index("date").sort_index()
                except Exception:
                    ind_cpi = pd.DataFrame()
        if not ind_cpi.empty:
            st.success("India CPI parsed from uploaded file.")
        else:
            st.warning("Uploaded file could not be parsed automatically; please ensure it contains a date column and a CPI index column.")
    except Exception as e:
        st.error(f"Error reading uploaded file: {e}")

# -------------------------
# Prepare derived series (YoY)
# -------------------------
# US series: compute YoY %
us_cpi_yoy = yoy_pct_change(us_cpi) if not us_cpi.empty else pd.DataFrame()
us_core_yoy = yoy_pct_change(us_core) if not us_core.empty else pd.DataFrame()
us_m2_yoy = (us_m2 / us_m2.shift(12) - 1) * 100 if not us_m2.empty else pd.DataFrame()

# India CPI YoY
ind_cpi_yoy = (ind_cpi["CPI"].pct_change(periods=12) * 100).to_frame("India CPI YoY") if not ind_cpi.empty else pd.DataFrame()

# -------------------------
# KPIs row
# -------------------------
st.markdown("## Key indicators (latest)")
k1, k2, k3, k4 = st.columns(4)

def last_value(df, col=None):
    if df.empty:
        return None
    if col:
        s = df[col].dropna()
    else:
        s = df.dropna().iloc[:,0] if not df.empty else pd.Series()
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
        st.metric("US M2 YoY (%)", f"{v.iloc[0]:.2f}" if isinstance(v, pd.Series) else f"{v:.2f}")
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
    df_levels = pd.concat([us_cpi.rename(columns={"CPIAUCSL":"US CPI"} if "CPIAUCSL" in us_cpi.columns else {us_cpi.columns[0]:"US CPI"}),
                           us_core.rename(columns={"CPILFESL":"US Core CPI"} if "CPILFESL" in us_core.columns else {us_core.columns[0]:"US Core CPI"})], axis=1)
    df_levels.columns = ["US CPI" if "CPIAUCSL" in df_levels.columns or df_levels.columns[0] else df_levels.columns[0], "US Core CPI"]
    df_levels = df_levels.dropna()
    fig = px.line(df_levels, labels={"index":"Date"}, title="US CPI (index) and Core CPI")
    st.plotly_chart(fig, use_container_width=True)
elif fred_key:
    st.info("US CPI not available: check your FRED key or date range.")

# US CPI YoY
if not us_cpi.empty:


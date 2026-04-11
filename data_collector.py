"""
Macro Variable Data Collector for BISTRO-XAI Feature Selection
==============================================================
FRED API + 기존 BIS 데이터를 활용하여 ~30개 거시 변수를 수집,
월별 주기로 정리하여 /tmp/bistro-repo/data/macro_panel.csv 에 저장.

실행:
    .venv-bistro/bin/python3 data_collector.py
"""

import os
import io
import time
import warnings
import numpy as np
import pandas as pd
import requests

warnings.filterwarnings("ignore")

DATA_DIR = "/tmp/bistro-repo/data"
OUTPUT_CSV = os.path.join(DATA_DIR, "macro_panel.csv")

# 최소 기간: 2013-01 ~ 2023-12 (CTX=120개월 + PDT=12개월)
DATE_START = "2000-01-01"
DATE_END = "2025-12-31"


# ============================================================
# FRED Series Definitions
# ============================================================
# (series_id, display_name, frequency, transform)
# frequency: "d" = daily → monthly mean, "m" = already monthly
# transform: "raw" = 그대로, "yoy" = 전년동월비(%), "diff" = 전월차

FRED_SERIES = [
    # 환율
    ("DEXKOUS",        "USD_KRW",       "d", "raw"),      # 원/달러 환율
    ("DEXJPUS",        "JPY_USD",       "d", "raw"),      # 엔/달러
    ("DEXCHUS",        "CNY_USD",       "d", "raw"),      # 위안/달러
    ("DTWEXBGS",       "DXY_Broad",     "d", "raw"),      # 달러 무역가중지수

    # 원자재
    ("DCOILBRENTEU",   "Oil_Brent",     "d", "raw"),      # 브렌트유
    ("DHHNGSP",        "NatGas_HH",     "d", "raw"),      # 천연가스(Henry Hub)
    ("GOLDAMGBD228NLBM","Gold",         "d", "raw"),      # 금 가격
    ("PCOPPUSDM",      "Copper",        "m", "raw"),      # 구리 가격
    ("PWHEAMTUSDM",    "Wheat",         "m", "raw"),      # 밀 가격
    ("PMAIZMTUSDM",    "Corn",          "m", "raw"),      # 옥수수

    # 미국 거시
    ("FEDFUNDS",       "FedFunds",      "m", "raw"),      # 연방기금금리
    ("CPIAUCSL",       "US_CPI_idx",    "m", "yoy"),      # 미국 CPI (YoY로 변환)
    ("CPILFESL",       "US_CoreCPI_idx","m", "yoy"),      # 미국 Core CPI (YoY)
    ("PPIACO",         "US_PPI",        "m", "yoy"),      # 미국 PPI All Commodities
    ("UNRATE",         "US_Unemp",      "m", "raw"),      # 미국 실업률
    ("UMCSENT",        "US_ConsConf",   "m", "raw"),      # 미시간 소비자심리
    ("M2SL",           "US_M2",         "m", "yoy"),      # M2 통화량 (YoY)
    ("T10Y2Y",         "US_YieldSpread","d", "raw"),      # 10Y-2Y 스프레드
    ("VIXCLS",         "VIX",           "d", "raw"),      # VIX 변동성

    # 글로벌 물가/교역
    ("GSCPI",          "GSCPI",         "m", "raw"),      # Global Supply Chain Pressure
    ("IR3TIB01KRM156N","KR_Interbank3M","m", "raw"),      # 한국 3개월 금리 (OECD)
    ("LRUN64TTKSM156S","KR_Unemp",     "m", "raw"),      # 한국 실업률 (OECD)
    ("IRSTCI01KRM156N","KR_LongRate",   "m", "raw"),      # 한국 장기금리

    # 수입물가/교역조건
    ("IR3TIB01JPM156N","JP_Interbank3M","m", "raw"),      # 일본 3개월 금리
    ("CHNCPIALLMINMEI","China_CPI",     "m", "yoy"),      # 중국 CPI (YoY)
    ("XTIMVA01KRM667S","KR_Imports",    "m", "yoy"),      # 한국 수입액 (YoY)
    ("XTEXVA01KRM667S","KR_Exports",    "m", "yoy"),      # 한국 수출액 (YoY)
]


def download_fred(series_id: str) -> pd.DataFrame:
    """FRED에서 CSV 직접 다운로드."""
    url = (
        f"https://fred.stlouisfed.org/graph/fredgraph.csv"
        f"?id={series_id}&cosd={DATE_START}&coed={DATE_END}"
    )
    resp = requests.get(url, timeout=30)
    resp.raise_for_status()
    df = pd.read_csv(io.StringIO(resp.text), index_col=0, parse_dates=True)
    df.columns = [series_id]
    # "." 등 비숫자 → NaN
    df[series_id] = pd.to_numeric(df[series_id], errors="coerce")
    return df


def download_all_fred():
    """모든 FRED 시리즈 다운로드 → 월별 DataFrame 반환."""
    monthly_frames = {}

    for series_id, name, freq, transform in FRED_SERIES:
        print(f"  Downloading {series_id} ({name})...", end=" ")
        try:
            raw = download_fred(series_id)
            raw = raw.dropna()

            if freq == "d":
                # daily → monthly mean
                monthly = raw.resample("M").mean()
            else:
                monthly = raw.copy()
                monthly.index = pd.to_datetime(monthly.index)

            monthly.index = monthly.index.to_period("M")
            monthly.columns = [name]

            if transform == "yoy":
                monthly[name] = monthly[name].pct_change(12) * 100  # 전년동월비 %
                monthly = monthly.dropna()

            monthly_frames[name] = monthly
            print(f"OK ({len(monthly)} months, {monthly.index[0]}~{monthly.index[-1]})")
        except Exception as e:
            print(f"FAILED: {e}")

        time.sleep(0.3)  # rate limiting

    return monthly_frames


def load_bis_data():
    """기존 BIS/FRED 데이터 로딩."""
    frames = {}

    # Korean CPI YoY (TARGET) — 별도 처리
    cpi = pd.read_csv(f"{DATA_DIR}/bis_cpi_kr_yoy_m.csv", index_col=0)
    cpi.index = pd.to_datetime(cpi.index).to_period("M")
    cpi.columns = ["CPI_KR_YoY"]
    frames["CPI_KR_YoY"] = cpi

    # Korean Policy Rate
    rate_kr = pd.read_csv(f"{DATA_DIR}/bis_cbpol_kr_m.csv", index_col=0)
    rate_kr.index = pd.to_datetime(rate_kr.index).to_period("M")
    rate_kr.columns = ["Rate_KR"]
    frames["Rate_KR"] = rate_kr

    # US Policy Rate (from BIS — may overlap with FRED FedFunds)
    # Skip: we'll use FRED FedFunds instead

    # ECB/XM Policy Rate
    rate_xm = pd.read_csv(f"{DATA_DIR}/bis_cbpol_xm_m.csv", index_col=0)
    rate_xm.index = pd.to_datetime(rate_xm.index).to_period("M")
    rate_xm.columns = ["Rate_ECB"]
    frames["Rate_ECB"] = rate_xm

    # Oil WTI (daily → monthly)
    oil = pd.read_csv(f"{DATA_DIR}/fred_oil_price_d.csv", index_col=0)
    oil.index = pd.to_datetime(oil.index)
    oil.columns = ["Oil_WTI"]
    oil = oil["Oil_WTI"].resample("M").mean().to_frame("Oil_WTI")
    oil.index = oil.index.to_period("M")
    frames["Oil_WTI"] = oil

    # US CPI YoY (from BIS)
    cpi_us = pd.read_csv(f"{DATA_DIR}/bis_cpi_us_yoy_m.csv", index_col=0)
    cpi_us.index = pd.to_datetime(cpi_us.index).to_period("M")
    cpi_us.columns = ["CPI_US_YoY"]
    frames["CPI_US_YoY"] = cpi_us

    # XM CPI YoY (from BIS)
    cpi_xm = pd.read_csv(f"{DATA_DIR}/bis_cpi_xm_yoy_m.csv", index_col=0)
    cpi_xm.index = pd.to_datetime(cpi_xm.index).to_period("M")
    cpi_xm.columns = ["CPI_XM_YoY"]
    frames["CPI_XM_YoY"] = cpi_xm

    return frames


def build_panel():
    """전체 데이터 패널 구축."""
    print("=" * 60)
    print("Step 1: Loading BIS data...")
    print("=" * 60)
    bis_frames = load_bis_data()
    print(f"  Loaded {len(bis_frames)} BIS series")

    print("\n" + "=" * 60)
    print("Step 2: Downloading FRED data...")
    print("=" * 60)
    fred_frames = download_all_fred()

    # 병합
    all_frames = {**bis_frames, **fred_frames}

    # FRED US_CPI_idx 가 있으면 BIS CPI_US_YoY 와 중복 → FRED 버전 제거
    if "US_CPI_idx" in all_frames and "CPI_US_YoY" in all_frames:
        del all_frames["US_CPI_idx"]
        print("\n  Note: Dropped FRED US_CPI_idx (using BIS CPI_US_YoY)")

    # 모든 시리즈를 하나의 DataFrame으로 병합
    target = all_frames.pop("CPI_KR_YoY")
    panel = target.copy()
    for name, df in sorted(all_frames.items()):
        panel = panel.join(df, how="outer")

    # 공통 기간 필터 (2013-01 이전 컨텍스트 포함을 위해 2003-01부터)
    panel = panel.loc["2003-01":"2025-12"]

    # 결측치 처리: forward fill → backward fill (최대 3개월)
    panel = panel.ffill(limit=3).bfill(limit=3)

    # 여전히 NaN이 많은 열 제거 (50% 이상 결측)
    threshold = len(panel) * 0.5
    before = panel.columns.tolist()
    panel = panel.dropna(axis=1, thresh=int(threshold))
    dropped = set(before) - set(panel.columns.tolist())
    if dropped:
        print(f"\n  Dropped columns with >50% NaN: {dropped}")

    # 나머지 NaN → 선형 보간
    panel = panel.interpolate(method="linear").bfill().ffill()

    print(f"\n" + "=" * 60)
    print(f"Final panel: {panel.shape[0]} months × {panel.shape[1]} variables")
    print(f"Period: {panel.index[0]} ~ {panel.index[-1]}")
    print(f"Variables: {panel.columns.tolist()}")
    print("=" * 60)

    # 저장
    panel.to_csv(OUTPUT_CSV)
    print(f"\nSaved: {OUTPUT_CSV}")

    return panel


OUTPUT_DAILY_CSV = os.path.join(DATA_DIR, "macro_panel_daily.csv")
OUTPUT_FREQ_CSV  = os.path.join(DATA_DIR, "variable_freq.csv")

# 변수별 원본 주기 매핑
VARIABLE_FREQ = {}
for series_id, name, freq, transform in FRED_SERIES:
    VARIABLE_FREQ[name] = "daily" if freq == "d" else "monthly"
# BIS 변수
VARIABLE_FREQ["CPI_KR_YoY"]  = "monthly"
VARIABLE_FREQ["Rate_KR"]     = "monthly"
VARIABLE_FREQ["Rate_ECB"]    = "monthly"
VARIABLE_FREQ["Oil_WTI"]     = "daily"
VARIABLE_FREQ["CPI_US_YoY"]  = "monthly"
VARIABLE_FREQ["CPI_XM_YoY"]  = "monthly"


def build_daily_panel():
    """
    일별 패널 구축.
    - 원본이 일별인 변수: 일별 그대로 유지
    - 원본이 월별인 변수: forward-fill로 일별 확장
    결과: macro_panel_daily.csv + variable_freq.csv (주기 메타)
    """
    print("=" * 60)
    print("Building DAILY panel")
    print("=" * 60)

    # ── 1. BIS 데이터 (월별) ────────────────────────────────
    bis_monthly = {}

    cpi = pd.read_csv(f"{DATA_DIR}/bis_cpi_kr_yoy_m.csv", index_col=0)
    cpi.index = pd.to_datetime(cpi.index)
    cpi.columns = ["CPI_KR_YoY"]
    bis_monthly["CPI_KR_YoY"] = cpi

    rate_kr = pd.read_csv(f"{DATA_DIR}/bis_cbpol_kr_m.csv", index_col=0)
    rate_kr.index = pd.to_datetime(rate_kr.index)
    rate_kr.columns = ["Rate_KR"]
    bis_monthly["Rate_KR"] = rate_kr

    rate_xm = pd.read_csv(f"{DATA_DIR}/bis_cbpol_xm_m.csv", index_col=0)
    rate_xm.index = pd.to_datetime(rate_xm.index)
    rate_xm.columns = ["Rate_ECB"]
    bis_monthly["Rate_ECB"] = rate_xm

    # Oil_WTI (daily)
    oil = pd.read_csv(f"{DATA_DIR}/fred_oil_price_d.csv", index_col=0)
    oil.index = pd.to_datetime(oil.index)
    oil.columns = ["Oil_WTI"]
    bis_daily = {"Oil_WTI": oil}

    # BIS CPI US/XM (monthly)
    cpi_us = pd.read_csv(f"{DATA_DIR}/bis_cpi_us_yoy_m.csv", index_col=0)
    cpi_us.index = pd.to_datetime(cpi_us.index)
    cpi_us.columns = ["CPI_US_YoY"]
    bis_monthly["CPI_US_YoY"] = cpi_us

    cpi_xm = pd.read_csv(f"{DATA_DIR}/bis_cpi_xm_yoy_m.csv", index_col=0)
    cpi_xm.index = pd.to_datetime(cpi_xm.index)
    cpi_xm.columns = ["CPI_XM_YoY"]
    bis_monthly["CPI_XM_YoY"] = cpi_xm

    # ── 2. FRED 데이터 ──────────────────────────────────────
    fred_daily = {}
    fred_monthly = {}

    for series_id, name, freq, transform in FRED_SERIES:
        print(f"  Downloading {series_id} ({name})...", end=" ")
        try:
            raw = download_fred(series_id)
            raw = raw.dropna()
            raw.columns = [name]

            if transform == "yoy":
                if freq == "d":
                    # 일별 데이터의 YoY: 월별로 집계 후 YoY 계산 → 다시 월별
                    monthly = raw.resample("M").mean()
                    monthly[name] = monthly[name].pct_change(12) * 100
                    monthly = monthly.dropna()
                    fred_monthly[name] = monthly
                    VARIABLE_FREQ[name] = "monthly"  # YoY 변환하면 월별로 취급
                else:
                    monthly = raw.copy()
                    monthly[name] = monthly[name].pct_change(12) * 100
                    monthly = monthly.dropna()
                    fred_monthly[name] = monthly
            elif freq == "d":
                fred_daily[name] = raw
            else:
                fred_monthly[name] = raw

            n = len(raw)
            print(f"OK ({n} rows)")
        except Exception as e:
            print(f"FAILED: {e}")
        time.sleep(0.3)

    # FRED US_CPI_idx 중복 제거
    if "US_CPI_idx" in fred_monthly and "CPI_US_YoY" in bis_monthly:
        del fred_monthly["US_CPI_idx"]
        if "US_CPI_idx" in VARIABLE_FREQ:
            del VARIABLE_FREQ["US_CPI_idx"]
        print("  Note: Dropped FRED US_CPI_idx (using BIS CPI_US_YoY)")

    # ── 3. 일별 인덱스 생성 ──────────────────────────────────
    start = pd.Timestamp("2003-01-01")
    end = pd.Timestamp("2025-12-31")
    daily_idx = pd.date_range(start, end, freq="D")
    panel = pd.DataFrame(index=daily_idx)

    # 일별 변수 합치기
    all_daily = {**bis_daily, **fred_daily}
    for name, df in sorted(all_daily.items()):
        df = df.loc[start:end]
        panel = panel.join(df, how="left")
        print(f"  [daily]   {name}: {df.dropna().shape[0]} raw daily points")

    # 월별 변수 → 일별 forward-fill
    all_monthly = {**bis_monthly, **fred_monthly}
    for name, df in sorted(all_monthly.items()):
        df = df.loc[start:end]
        # 월별 데이터를 일별 인덱스에 reindex → forward fill
        df_daily = df.reindex(daily_idx).ffill()
        panel[name] = df_daily[name]
        print(f"  [monthly] {name}: {df.dropna().shape[0]} monthly points → daily ffill")

    # CPI_KR_YoY를 첫 번째 열로
    cols = panel.columns.tolist()
    if "CPI_KR_YoY" in cols:
        cols.remove("CPI_KR_YoY")
        cols = ["CPI_KR_YoY"] + cols
        panel = panel[cols]

    # 결측치 처리
    # 일별 데이터: 주말/공휴일 NaN → forward fill
    panel = panel.ffill(limit=5).bfill(limit=5)

    # 50% 이상 NaN인 열 제거
    threshold = len(panel) * 0.5
    before = panel.columns.tolist()
    panel = panel.dropna(axis=1, thresh=int(threshold))
    dropped = set(before) - set(panel.columns.tolist())
    if dropped:
        print(f"\n  Dropped columns with >50% NaN: {dropped}")

    # 나머지 NaN → 선형 보간
    panel = panel.interpolate(method="linear").bfill().ffill()

    # ── 4. 주기 메타데이터 저장 ──────────────────────────────
    freq_df = pd.DataFrame([
        {"variable": col, "original_freq": VARIABLE_FREQ.get(col, "unknown")}
        for col in panel.columns
    ])
    freq_df.to_csv(OUTPUT_FREQ_CSV, index=False)

    print(f"\n{'=' * 60}")
    print(f"Daily panel: {panel.shape[0]} days × {panel.shape[1]} variables")
    print(f"Period: {panel.index[0].strftime('%Y-%m-%d')} ~ {panel.index[-1].strftime('%Y-%m-%d')}")
    n_daily = sum(1 for c in panel.columns if VARIABLE_FREQ.get(c) == "daily")
    n_monthly = sum(1 for c in panel.columns if VARIABLE_FREQ.get(c) == "monthly")
    print(f"  Daily vars: {n_daily}, Monthly vars: {n_monthly}")
    print(f"Variables: {panel.columns.tolist()}")
    print("=" * 60)

    panel.to_csv(OUTPUT_DAILY_CSV)
    print(f"\nSaved: {OUTPUT_DAILY_CSV}")
    print(f"Saved: {OUTPUT_FREQ_CSV}")

    return panel


OUTPUT_TOURNAMENT_DAILY_CSV = os.path.join(
    os.path.dirname(__file__), "data", "macro_panel_tournament_daily.csv"
)
OUTPUT_TOURNAMENT_FREQ_CSV = os.path.join(
    os.path.dirname(__file__), "data", "variable_freq_tournament.csv"
)


def build_tournament_daily_panel():
    """
    토너먼트용 ~300개 변수 일별 패널 구축.
    tournament_config.py의 FRED_SERIES를 사용.
    BIS 데이터(CPI_KR_YoY 타겟 포함)는 기존과 동일하게 로딩.
    """
    from tournament_config import FRED_SERIES as TOURNAMENT_FRED_SERIES

    print("=" * 60)
    print("Building TOURNAMENT DAILY panel (~300 variables)")
    print("=" * 60)

    # ── 1. BIS 데이터 (기존과 동일) ──────────────────────────
    bis_monthly = {}
    bis_daily = {}

    cpi = pd.read_csv(f"{DATA_DIR}/bis_cpi_kr_yoy_m.csv", index_col=0)
    cpi.index = pd.to_datetime(cpi.index)
    cpi.columns = ["CPI_KR_YoY"]
    bis_monthly["CPI_KR_YoY"] = cpi

    rate_kr = pd.read_csv(f"{DATA_DIR}/bis_cbpol_kr_m.csv", index_col=0)
    rate_kr.index = pd.to_datetime(rate_kr.index)
    rate_kr.columns = ["Rate_KR"]
    bis_monthly["Rate_KR"] = rate_kr

    rate_xm = pd.read_csv(f"{DATA_DIR}/bis_cbpol_xm_m.csv", index_col=0)
    rate_xm.index = pd.to_datetime(rate_xm.index)
    rate_xm.columns = ["Rate_ECB"]
    bis_monthly["Rate_ECB"] = rate_xm

    oil = pd.read_csv(f"{DATA_DIR}/fred_oil_price_d.csv", index_col=0)
    oil.index = pd.to_datetime(oil.index)
    oil.columns = ["Oil_WTI"]
    bis_daily["Oil_WTI"] = oil

    cpi_us = pd.read_csv(f"{DATA_DIR}/bis_cpi_us_yoy_m.csv", index_col=0)
    cpi_us.index = pd.to_datetime(cpi_us.index)
    cpi_us.columns = ["CPI_US_YoY"]
    bis_monthly["CPI_US_YoY"] = cpi_us

    cpi_xm = pd.read_csv(f"{DATA_DIR}/bis_cpi_xm_yoy_m.csv", index_col=0)
    cpi_xm.index = pd.to_datetime(cpi_xm.index)
    cpi_xm.columns = ["CPI_XM_YoY"]
    bis_monthly["CPI_XM_YoY"] = cpi_xm

    print(f"  Loaded {len(bis_monthly)} BIS monthly + {len(bis_daily)} BIS daily")

    # ── 2. FRED 다운로드 (300개) ─────────────────────────────
    fred_daily = {}
    fred_monthly = {}
    freq_map = {}  # name → "daily" / "monthly"
    failed = []

    total = len(TOURNAMENT_FRED_SERIES)
    for idx, (series_id, name, freq, transform) in enumerate(TOURNAMENT_FRED_SERIES, 1):
        print(f"  [{idx:>3}/{total}] {series_id} ({name})...", end=" ")
        try:
            raw = download_fred(series_id)
            raw = raw.dropna()
            raw.columns = [name]

            if transform == "yoy":
                if freq == "d":
                    monthly = raw.resample("M").mean()
                    monthly[name] = monthly[name].pct_change(12) * 100
                    monthly = monthly.dropna()
                    fred_monthly[name] = monthly
                    freq_map[name] = "monthly"
                else:
                    monthly = raw.copy()
                    monthly[name] = monthly[name].pct_change(12) * 100
                    monthly = monthly.dropna()
                    fred_monthly[name] = monthly
                    freq_map[name] = "monthly"
            elif transform == "diff":
                if freq == "d":
                    monthly = raw.resample("M").mean()
                    monthly[name] = monthly[name].diff()
                    monthly = monthly.dropna()
                    fred_monthly[name] = monthly
                    freq_map[name] = "monthly"
                else:
                    raw[name] = raw[name].diff()
                    raw = raw.dropna()
                    fred_monthly[name] = raw
                    freq_map[name] = "monthly"
            elif freq == "d":
                fred_daily[name] = raw
                freq_map[name] = "daily"
            else:
                fred_monthly[name] = raw
                freq_map[name] = "monthly"

            print(f"OK ({len(raw)} rows)")
        except Exception as e:
            print(f"FAILED: {e}")
            failed.append((series_id, name, str(e)))
        time.sleep(0.3)

    if failed:
        print(f"\n  ⚠ {len(failed)} series failed:")
        for sid, name, err in failed:
            print(f"    {sid} ({name}): {err}")

    # BIS 주기 메타 추가
    for name in bis_monthly:
        freq_map[name] = "monthly"
    for name in bis_daily:
        freq_map[name] = "daily"

    # ── 3. 일별 패널 조립 ────────────────────────────────────
    start = pd.Timestamp("2003-01-01")
    end = pd.Timestamp("2025-12-31")
    daily_idx = pd.date_range(start, end, freq="D")
    panel = pd.DataFrame(index=daily_idx)

    # 일별 변수
    all_daily = {**bis_daily, **fred_daily}
    for name, df in sorted(all_daily.items()):
        df = df.loc[start:end]
        panel = panel.join(df, how="left")

    # 월별 변수 → forward-fill
    all_monthly = {**bis_monthly, **fred_monthly}
    for name, df in sorted(all_monthly.items()):
        df = df.loc[start:end]
        df_daily = df.reindex(daily_idx).ffill()
        panel[name] = df_daily[name]

    # CPI_KR_YoY를 첫 번째 열로
    cols = panel.columns.tolist()
    if "CPI_KR_YoY" in cols:
        cols.remove("CPI_KR_YoY")
        cols = ["CPI_KR_YoY"] + cols
        panel = panel[cols]

    # 결측치 처리
    panel = panel.ffill(limit=5).bfill(limit=5)

    # 50% 이상 NaN인 열 제거
    before = set(panel.columns.tolist())
    threshold = len(panel) * 0.5
    panel = panel.dropna(axis=1, thresh=int(threshold))
    dropped = before - set(panel.columns.tolist())
    if dropped:
        print(f"\n  Dropped {len(dropped)} columns with >50% NaN: {sorted(dropped)}")

    # 나머지 NaN → 선형 보간
    panel = panel.interpolate(method="linear").bfill().ffill()

    # ── 4. 저장 ──────────────────────────────────────────────
    freq_df = pd.DataFrame([
        {"variable": col, "original_freq": freq_map.get(col, "unknown")}
        for col in panel.columns
    ])
    freq_df.to_csv(OUTPUT_TOURNAMENT_FREQ_CSV, index=False)

    panel.to_csv(OUTPUT_TOURNAMENT_DAILY_CSV)

    print(f"\n{'=' * 60}")
    print(f"Tournament daily panel: {panel.shape[0]} days × {panel.shape[1]} variables")
    print(f"Period: {panel.index[0].strftime('%Y-%m-%d')} ~ {panel.index[-1].strftime('%Y-%m-%d')}")
    n_d = sum(1 for c in panel.columns if freq_map.get(c) == "daily")
    n_m = sum(1 for c in panel.columns if freq_map.get(c) == "monthly")
    print(f"  Daily vars: {n_d}, Monthly vars: {n_m}")
    print(f"  Failed downloads: {len(failed)}")
    print(f"Saved: {OUTPUT_TOURNAMENT_DAILY_CSV}")
    print(f"Saved: {OUTPUT_TOURNAMENT_FREQ_CSV}")
    print("=" * 60)

    return panel


if __name__ == "__main__":
    import sys
    if "--tournament" in sys.argv:
        build_tournament_daily_panel()
    elif "--daily" in sys.argv:
        build_daily_panel()
    else:
        build_panel()
        print("\nTip: use --daily for daily panel, --tournament for 300-var panel")

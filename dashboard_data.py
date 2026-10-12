"""Data loading, validation, and filtering for the AgriSight Dash application.

Only the four CSV datasets displayed by the app are read at startup. The large
wind-tower Excel workbook remains in the repository for the original notebooks.
"""
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

BASE_DIR = Path(__file__).resolve().parent


@dataclass(frozen=True)
class Datasets:
    harvest: pd.DataFrame
    erosion: pd.DataFrame
    weather: pd.DataFrame
    cover_crop: pd.DataFrame


def load_datasets() -> Datasets:
    harvest = pd.read_csv(BASE_DIR / "h5_i2_2019-2022_hand_harvest.csv")
    erosion = pd.read_csv(BASE_DIR / "Mass Flux values 2019_2022 updated.csv")
    weather = pd.read_csv(BASE_DIR / "h5-i2_2016-2021_daily-weather.csv")
    cover_crop = pd.read_csv(BASE_DIR / "I2_CC_AGB_2020-2022.csv")

    for frame in (harvest, erosion, weather, cover_crop):
        frame.columns = frame.columns.str.strip()

    harvest["Date"] = pd.to_datetime(harvest["Date"], errors="coerce")
    harvest["Year"] = pd.to_numeric(harvest["Year"], errors="coerce").astype("Int64")
    harvest["Crop"] = harvest["Crop"].replace({"SpringWheat": "Spring wheat"})
    for col in ("Grain_yield_kg_ha", "AGB_g_m2", "Harvest_index"):
        harvest[col] = pd.to_numeric(harvest[col], errors="coerce")
    harvest = harvest.dropna(subset=["Year", "Grain_yield_kg_ha"])

    erosion = erosion.rename(columns={"total.flux": "total_flux"})
    erosion["Year"] = pd.to_numeric(erosion["Year"], errors="coerce").astype("Int64")
    erosion["total_flux"] = pd.to_numeric(erosion["total_flux"], errors="coerce")
    # The erosion 'Date' field is MMDD, e.g. 613 means June 13.
    mmdd = pd.to_numeric(erosion["Date"], errors="coerce").astype("Int64").astype(str).str.zfill(4)
    erosion["Event_date"] = pd.to_datetime(
        erosion["Year"].astype(str) + mmdd, format="%Y%m%d", errors="coerce"
    )
    erosion["Month"] = erosion["Event_date"].dt.month
    erosion = erosion.dropna(subset=["Year", "total_flux"])

    # Original field name is 'Date ' with trailing whitespace.
    weather["Date"] = pd.to_datetime(
        weather["Date"].astype(str), format="%Y%m%d", errors="coerce"
    )
    weather["Year"] = pd.to_numeric(weather["Year"], errors="coerce").astype("Int64")
    weather["Month"] = weather["Date"].dt.month
    for col in ("Wind_spd_m_s", "PCPN_mm_d", "T_min_C", "T_max_C"):
        weather[col] = pd.to_numeric(weather[col], errors="coerce")
    weather = weather.dropna(subset=["Date"])

    # Dates use month_day_year, *not* a standard slash-separated date.
    cover_crop["Date"] = pd.to_datetime(
        cover_crop["DATE"], format="%m_%d_%Y", errors="coerce"
    )
    cover_crop["Year"] = cover_crop["Date"].dt.year.astype("Int64")
    cover_crop["CC_AGB"] = pd.to_numeric(cover_crop["CC_AGB"], errors="coerce")
    cover_crop = cover_crop.dropna(subset=["Date", "CC_AGB"])

    return Datasets(harvest=harvest, erosion=erosion, weather=weather, cover_crop=cover_crop)


def apply_filters(data: Datasets, year="all", treatment="all", crop="all", site="all") -> Datasets:
    """Apply only meaningful filters to each dataset.

    Crop is only recorded in harvest data; site refers to Mandan/Morton erosion
    sites and does not match harvest plot identifiers or NP weather station.
    """
    harvest = data.harvest.copy()
    erosion = data.erosion.copy()
    weather = data.weather.copy()
    cover_crop = data.cover_crop.copy()

    if year not in (None, "all"):
        selected_year = int(year)
        harvest = harvest.loc[harvest["Year"] == selected_year]
        erosion = erosion.loc[erosion["Year"] == selected_year]
        weather = weather.loc[weather["Year"] == selected_year]
        cover_crop = cover_crop.loc[cover_crop["Year"] == selected_year]
    if treatment not in (None, "all"):
        harvest = harvest.loc[harvest["Treatment"] == treatment]
        erosion = erosion.loc[erosion["Treatment"] == treatment]
        cover_crop = cover_crop.loc[cover_crop["TRT"] == treatment]
    if crop not in (None, "all"):
        harvest = harvest.loc[harvest["Crop"] == crop]
    if site not in (None, "all"):
        erosion = erosion.loc[erosion["Site"] == site]

    return Datasets(harvest=harvest, erosion=erosion, weather=weather, cover_crop=cover_crop)


def records_for_export(data: Datasets, section: str) -> pd.DataFrame:
    """Return the observed, filtered data behind the active view."""
    if section in ("yield", "treatment"):
        return data.harvest
    if section == "erosion":
        return data.erosion
    if section == "weather":
        return data.weather
    if section == "trends":
        # Label the source because the datasets have different measurements.
        frames = []
        for name, frame in (("harvest", data.harvest), ("erosion", data.erosion)):
            if not frame.empty:
                copy = frame.copy()
                copy.insert(0, "Dataset", name)
                frames.append(copy)
        return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    # Overview export: observed, filtered harvest records; other pages export
    # their own underlying data (avoids joining unmatched observations).
    return data.harvest

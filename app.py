import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

from dash import Dash, dcc, html, Input, Output

# ---------- 1. LOAD DATA (use local files in ADV Project folder) ----------

harvest = pd.read_csv("h5_i2_2019-2022_hand_harvest.csv")
weather = pd.read_csv("h5-i2_2016-2021_daily-weather.csv")
agb_cc = pd.read_csv("I2_CC_AGB_2020-2022.csv")
massflux = pd.read_csv("Mass Flux values 2019_2022 updated.csv")

# Excel with wind tower averages (Mandan & Morton)
wind_sheets = pd.ExcelFile("avg wind speed nwern.xlsx")
mandan_daily = pd.read_excel("avg wind speed nwern.xlsx", sheet_name="MandanDailyAVG")
morton_daily = pd.read_excel("avg wind speed nwern.xlsx", sheet_name="MortonDailyAVG")

# ---------- 2. CLEAN COLUMN NAMES ----------

def clean_columns(df):
    df = df.copy()
    df.columns = [c.strip().replace(" ", "_") for c in df.columns]
    return df

harvest = clean_columns(harvest)
weather = clean_columns(weather)
agb_cc = clean_columns(agb_cc)
massflux = clean_columns(massflux)
mandan_daily = clean_columns(mandan_daily)
morton_daily = clean_columns(morton_daily)

# ---------- 3. FIX DATES & NUMBERS ----------

# --- Harvest (yield, biomass) ---
harvest["Date"] = pd.to_datetime(harvest["Date"])
for col in ["AGB_g_m2", "Grain_yield_g_m2", "Grain_yield_kg_ha",
            "Stover_g_m2", "Harvest_index", "Percent_H2O"]:
    harvest[col] = pd.to_numeric(harvest[col], errors="coerce")

# --- Cover crop biomass ---
agb_cc["DATE"] = pd.to_datetime(agb_cc["DATE"], errors="coerce")
agb_cc["CC_AGB"] = pd.to_numeric(agb_cc["CC_AGB"], errors="coerce")
agb_cc["Year"] = agb_cc["DATE"].dt.year

# --- Weather (note: FIeld, Date / Date_) ---
weather.rename(columns={"FIeld": "Field"}, inplace=True)
if "Date_" in weather.columns:
    weather.rename(columns={"Date_": "Date"}, inplace=True)
if "Date" not in weather.columns and "Date_" in weather.columns:
    weather.rename(columns={"Date_": "Date"}, inplace=True)

weather["Date"] = pd.to_datetime(weather["Date"].astype(str), format="%Y%m%d", errors="coerce")
for col in ["Sol_Rad_MJ_m2_d", "T_min_C", "T_max_C",
            "PCPN_mm_d", "RH_f", "Wind_spd_m_s"]:
    weather[col] = pd.to_numeric(weather[col], errors="coerce")

# --- Mass flux / wind erosion ---
massflux["Year"] = pd.to_numeric(massflux["Year"], errors="coerce")
massflux["Date"] = massflux["Date"].astype(str).str.zfill(4)
massflux["Month"] = massflux["Date"].str[:2].astype(int)
massflux["Day"] = massflux["Date"].str[2:].astype(int)

massflux["Date_full"] = pd.to_datetime(
    dict(year=massflux["Year"], month=massflux["Month"], day=massflux["Day"]),
    errors="coerce"
)

# column names are "Days.After.Planting" and "total.flux" after cleaning
massflux.rename(columns={"total.flux": "total_flux"}, inplace=True)
massflux["total_flux"] = pd.to_numeric(massflux["total_flux"], errors="coerce")
if "Days.After.Planting" in massflux.columns:
    massflux["Days_After_Planting"] = pd.to_numeric(
        massflux["Days.After.Planting"], errors="coerce"
    )

# --- Wind tower daily average speed ---
for df, site_name in [(mandan_daily, "Mandan"), (morton_daily, "Morton")]:
    # Try to standardize the date column
    # (after clean_columns, it's usually 'date')
    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df["site"] = site_name

    # Try to find any column that looks like wind speed
    wind_col = None
    for c in df.columns:
        if "wind" in c.lower():
            wind_col = c
            break

    # If we find a wind column, convert it; otherwise just create an empty one
    if wind_col is not None:
        df["wind_m_s"] = pd.to_numeric(df[wind_col], errors="coerce")
    else:
        # Not used anywhere else right now, so just fill with NaN
        df["wind_m_s"] = np.nan

# ---------- 4. SUMMARY TABLES FOR THEMES ----------

# Theme 1 – yield & biomass by crop/treatment/year
harvest_theme = (harvest
                 .groupby(["Year", "Crop", "Treatment"], as_index=False)
                 .agg(mean_yield=("Grain_yield_kg_ha", "mean"),
                      mean_agb=("AGB_g_m2", "mean"),
                      mean_hi=("Harvest_index", "mean")))

# Theme 2 – erosion by site/treatment
erosion_theme = (massflux
                 .groupby(["Site", "Treatment"], as_index=False)
                 .agg(mean_flux=("total_flux", "mean"),
                      max_flux=("total_flux", "max")))

# Theme 3 – weather summaries
weather_theme = (weather
                 .groupby(["Year", "Field"], as_index=False)
                 .agg(mean_wind=("Wind_spd_m_s", "mean"),
                      mean_pcpn=("PCPN_mm_d", "mean"),
                      mean_tmin=("T_min_C", "mean"),
                      mean_tmax=("T_max_C", "mean")))

# Theme 4 – trends
yield_by_year = (harvest
                 .groupby(["Year", "Treatment"], as_index=False)
                 .agg(mean_yield=("Grain_yield_kg_ha", "mean")))

flux_by_year = (massflux
                .groupby(["Year", "Site", "Treatment"], as_index=False)
                .agg(mean_flux=("total_flux", "mean")))

# Theme 5 – treatment performance
treatment_perf = (harvest
                  .groupby("Treatment", as_index=False)
                  .agg(mean_yield=("Grain_yield_kg_ha", "mean"),
                       mean_agb=("AGB_g_m2", "mean"),
                       mean_hi=("Harvest_index", "mean")))

cc_perf = (agb_cc
           .groupby(["TRT", "Year"], as_index=False)
           .agg(mean_cc_agb=("CC_AGB", "mean")))

# ---------- THEME 1: Yield & Biomass ----------

def fig_yield_by_crop_treatment(harvest_theme, year=None):
    df = harvest_theme.copy()
    if year is not None:
        df = df[df["Year"] == year]

    fig = px.bar(
        df,
        x="Crop",
        y="mean_yield",
        color="Treatment",
        barmode="group",
        facet_col="Year" if year is None else None,
        title="Grain yield (kg/ha) by crop and treatment",
        labels={"mean_yield": "Mean grain yield (kg/ha)"}
    )
    return fig


def fig_yield_by_crop_treatment(harvest_theme, year=None):
    """
    Theme 1 – Question 1:
    Average grain yield by crop–treatment combination, horizontal,
    sorted in descending order, with labels on bars.
    If a year is selected, show that year's averages only.
    Otherwise, average across all years.
    """
    df = harvest_theme.copy()

    # Filter by year if provided
    if year is not None:
        df = df[df["Year"] == year]

    # Aggregate to crop–treatment level
    summary = (
        df.groupby(["Crop", "Treatment"], as_index=False)
          .agg(avg_yield=("mean_yield", "mean"))
    )

    # Combined label "Crop – Treatment"
    summary["CropTreatment"] = summary["Crop"] + " – " + summary["Treatment"]

    # Sort in descending order of avg yield
    summary = summary.sort_values("avg_yield", ascending=False)
    display_order = summary["CropTreatment"].tolist()

    fig = px.bar(
    summary,
    x="avg_yield",
    y="CropTreatment",
    orientation="h",
    color="Treatment",
    category_orders={"CropTreatment": display_order},
    labels={
        "avg_yield": "Average grain yield (kg/ha)",
        "CropTreatment": "Crop – Treatment"
    },
    title="Average Grain Yield by Crop–Treatment Combination",
    text="avg_yield"      # <- each bar gets its own value
)
    # Format and position the existing text
    fig.update_traces(
    texttemplate="%{text:.0f}",
    textposition="inside"
)


    fig.update_layout(
        hovermode="closest",
        margin=dict(t=80, l=140, r=40, b=60)
    )

    return fig


def fig_agb_vs_yield_scatter(harvest_theme, year=None):
    df = harvest_theme.copy()

    # Filter by year if requested
    if year is not None and year != "all":
        df = df[df["Year"] == year]

    df["TreatCrop"] = df["Treatment"] + ", " + df["Crop"]

    fig = px.line(
        df,
        x="Year",
        y="mean_agb",
        color="TreatCrop",
        markers=True,
        title="AGB Variation Over Years by Crop and Treatment",
        labels={
            "Year": "Year",
            "mean_agb": "Average Aboveground Biomass (g/m²)",
            "TreatCrop": "Treatment, Crop"
        }
    )

    fig.update_layout(
        hovermode="x unified",
        margin=dict(t=80, l=80, r=40, b=60)
    )
    return fig



# ---------- THEME 2: Wind Erosion ----------

def fig_flux_by_site_treatment(erosion_theme):
    """(You’re not using this in the dashboard right now,
    but I’m leaving it here in case you still need it.)"""
    fig = px.bar(
        erosion_theme,
        x="Site",
        y="mean_flux",
        color="Treatment",
        barmode="group",
        title="Average total mass flux by site and treatment",
        labels={"mean_flux": "Mean total flux (g m⁻¹ d⁻¹)"}
    )
    return fig


def fig_flux_location_site_year(massflux_df):
    """
    Theme 2 – Question 3

    Make Dash show the SAME figure as your .ipynb:

    'Mean Wind Erosion (Total Flux) Variation by Location, Site, and Year'

    - Horizontal stacked bars
    - One bar per Location
    - Colors = Site (Mandan / Morton)
    - Faceted by Year (2x2 layout)
    - Numeric labels inside bars
    """

    # If nothing to plot, return an empty figure
    if massflux_df is None or massflux_df.empty:
        return go.Figure()

    # Work on a copy and use the cleaned column name total_flux
    df = massflux_df.copy()

    # 1. Aggregate mean mass flux by Year, Site, and Location
    flux_summary = (
        df
        .groupby(["Year", "Site", "Location"])["total_flux"]
        .agg(mean_flux="mean", n_days="size")
        .reset_index()
    )

    # 2. Order locations by TOTAL (Mandan + Morton) mean flux, descending
    location_order = (
        flux_summary
        .groupby("Location")["mean_flux"]
        .sum()
        .sort_values(ascending=False)
        .index
        .tolist()
    )

    # 3. Horizontal stacked bars: one bar per Location, colours = Site
    fig = px.bar(
        flux_summary,
        x="mean_flux",
        y="Location",          # one bar per sampling location (A1, A2, …)
        color="Site",          # Mandan vs Morton segments
        orientation="h",
        facet_col="Year",
        facet_col_wrap=2,      # 2x2 layout
        barmode="stack",       # stack Mandan + Morton in same bar
        category_orders={
            "Location": location_order,
            "Year": sorted(flux_summary["Year"].unique())
        },
        labels={
            "mean_flux": "Mean Total Mass Flux (g m⁻¹ d⁻¹)",
            "Location": "Sampling Location",
            "Year": "Year",
            "Site": "Site"
        },
        title="Mean Wind Erosion (Total Flux) Variation by Location, Site, and Year",
        text="mean_flux",      # numeric labels on each coloured segment
        height=900,
        width=1300,
        # Metadata for custom hover
        custom_data=["Location", "Site", "Year", "n_days"],
        color_discrete_map={
            "Mandan": "#636EFA",   # blue
            "Morton": "#EF553B"    # red
        }
    )

    # 4. Put numbers inside each segment & custom hover
    fig.update_traces(
        texttemplate="%{text:.2f}",
        textposition="inside",
        hovertemplate=(
            "<b>%{customdata[1]} – %{customdata[0]}</b><br>"  # Site – Location
            "Year: %{customdata[2]}<br>"
            "Mean mass flux: %{x:.2f}<br>"
            "Number of days: %{customdata[3]}<extra></extra>"
        )
    )

    # 5. Layout polish
    fig.update_layout(
        hovermode="closest",
        margin=dict(t=80, l=120, r=40, b=60),
        bargap=0.1,
        bargroupgap=0.0,
        legend_title_text="Site"
    )

    return fig


def fig_flux_time_series(massflux, site=None, treatment=None):
    """
    Theme 2 – Time-series view (radio option 'Time-series by site & treatment').

    Annual mean total flux by site, with optional filters.
    """
    # Start from valid rows
    df = massflux.dropna(subset=["Year", "total_flux"]).copy()

    # Optional filters from dropdowns
    if site:
        df = df[df["Site"] == site]
    if treatment:
        df = df[df["Treatment"] == treatment]

    # Aggregate: mean total flux per Year and Site
    annual = (
        df.groupby(["Year", "Site"], as_index=False)["total_flux"]
          .mean()
          .rename(columns={"total_flux": "mean_flux"})
    )

    fig = px.line(
        annual,
        x="Year",
        y="mean_flux",
        color="Site",
        markers=True,
        title="Annual Mean Wind Erosion Mass Flux by Site",
        labels={
            "Year": "Year",
            "mean_flux": "Mean total flux (g m⁻¹ d⁻¹)",
            "Site": "Site"
        },
        color_discrete_map={
            "Mandan": "#636EFA",
            "Morton": "#EF553B"
        }
    )

    fig.update_layout(
    hovermode="x unified",
    height=450,     # smaller height
    margin=dict(t=60, l=60, r=20, b=60),
    legend_title_text="Site",
    xaxis=dict(tickangle=0, automargin=True),
    yaxis=dict(automargin=True)
)

    return fig



# ---------- THEME 3: Weather Impacts ----------

def fig_theme3_q1_wind_vs_flux(weather, massflux, year=None):
    """
    THEME 3 – QUESTION 1:
    How does wind speed influence wind erosion (mass flux)?

    - x = Wind_spd_m_s (annual mean from daily weather)
    - y = total.flux (mean annual mass flux by Site)
    - color = Site
    - symbol = Treatment
    - red dashed trend line
    """

    # ---- 1. Weather summary by Year ----
    wf = weather.copy()
    wf["Year"] = wf["Date"].dt.year

    weather_by_year = (
        wf.groupby("Year", as_index=False)
          .agg({
              "Wind_spd_m_s": "mean",
              "T_min_C": "mean",
              "T_max_C": "mean",
              "PCPN_mm_d": "mean",
              "Sol_Rad_MJ_m2_d": "mean"
          })
    )

    # ---- 2. Mass flux summary by Site & Year ----
    mf = massflux.copy()

    # Ensure we have a 'total.flux' column (to match notebook)
    if "total.flux" in mf.columns:
        pass
    else:
        mf["total.flux"] = mf["total_flux"]

    flux_by_site_year = (
        mf.groupby(["Site", "Year"], as_index=False)
          .agg({
              "total.flux": "mean",
              "Days_After_Planting": "mean",
              "Treatment": "first"
          })
    )

    # ---- 3. Optional filter by year (AFTER creating the summaries) ----
    if year is not None:
        weather_by_year = weather_by_year[weather_by_year["Year"] == year]
        flux_by_site_year = flux_by_site_year[flux_by_site_year["Year"] == year]

    # ---- 4. Merge on Year ----
    merged_env_data = pd.merge(
        flux_by_site_year,
        weather_by_year[["Year", "Wind_spd_m_s"]],
        on="Year",
        how="left"
    )

    plot_data = merged_env_data[
        ["total.flux", "Wind_spd_m_s", "Site", "Treatment"]
    ].dropna()

    if plot_data.empty:
        return go.Figure(
            layout=dict(
                title="Relationship between Wind Speed and Wind Erosion (Mass Flux)",
                xaxis_title="Wind Speed (m/s)",
                yaxis_title="Total Mass Flux (g m⁻¹ d⁻¹)",
                annotations=[dict(
                    text="No data available for the selected year.",
                    x=0.5, y=0.5, xref="paper", yref="paper",
                    showarrow=False
                )]
            )
        )

    # ---- 5. Scatter plot ----
    fig = px.scatter(
        plot_data,
        x="Wind_spd_m_s",
        y="total.flux",
        color="Site",
        symbol="Treatment",
        title="Relationship between Wind Speed and Wind Erosion (Mass Flux)",
        labels={
            "Wind_spd_m_s": "Wind Speed (m/s)",
            "total.flux": "Total Mass Flux (g m⁻¹ d⁻¹)",
            "Site": "Site",
            "Treatment": "Treatment"
        },
        hover_data={
            "Wind_spd_m_s": ":.2f",
            "total.flux": ":.2f",
            "Site": True,
            "Treatment": True
        }
    )

    # ---- 6. Red dashed trend line ----
    if len(plot_data) > 1:
        x = plot_data["Wind_spd_m_s"].to_numpy()
        y = plot_data["total.flux"].to_numpy()
        z = np.polyfit(x, y, 1)
        p = np.poly1d(z)
        x_trend = np.linspace(x.min(), x.max(), 100)

        fig.add_trace(
            go.Scatter(
                x=x_trend,
                y=p(x_trend),
                mode="lines",
                name="Trend Line",
                line=dict(color="red", dash="dash")
            )
        )

    fig.update_layout(
        height=600,
        width=1000,
        hovermode="closest",
        margin=dict(t=80, l=80, r=40, b=60),
        legend_title_text="Site"
    )

    return fig



def fig_precip_vs_flux(weather, massflux, year=None):
    """
    Precipitation vs wind erosion flux (scatter).
    """

    wf = weather.copy()
    wf["Year"] = wf["Date"].dt.year

    mf = massflux.copy()
    mf["Year"] = mf["Year"].astype(int)

    if year is not None:
        wf = wf[wf["Year"] == year]
        mf = mf[mf["Year"] == year]

    merged = pd.merge(
        mf,
        wf,
        left_on=["Year", "Date_full"],
        right_on=["Year", "Date"],
        how="left"
    ).dropna(subset=["total_flux", "PCPN_mm_d"])

    if merged.empty:
        return go.Figure(
            layout=dict(
                title="Precipitation vs wind erosion flux",
                xaxis_title="Precipitation (mm/day)",
                yaxis_title="Total flux (g m⁻¹ d⁻¹)",
                annotations=[dict(
                    text="No data available for the selected year.",
                    x=0.5, y=0.5, xref="paper", yref="paper",
                    showarrow=False
                )]
            )
        )

    fig = px.scatter(
        merged,
        x="PCPN_mm_d",
        y="total_flux",
        color="Site",
        title="Precipitation vs wind erosion flux",
        labels={
            "PCPN_mm_d": "Precipitation (mm/day)",
            "total_flux": "Total flux (g m⁻¹ d⁻¹)"
        }
    )
    return fig



def fig_wind_vs_flux(weather, massflux, year=None):
    """
    Daily wind speed vs wind erosion flux (scatter).
    """

    wf = weather.copy()
    wf["Year"] = wf["Date"].dt.year

    mf = massflux.copy()
    mf["Year"] = mf["Year"].astype(int)

    if year is not None:
        wf = wf[wf["Year"] == year]
        mf = mf[mf["Year"] == year]

    merged = pd.merge(
        mf,
        wf,
        left_on=["Year", "Date_full"],
        right_on=["Year", "Date"],
        how="left"
    ).dropna(subset=["total_flux", "Wind_spd_m_s"])

    if merged.empty:
        return go.Figure(
            layout=dict(
                title="Wind speed vs wind erosion flux",
                xaxis_title="Wind speed (m/s)",
                yaxis_title="Total flux (g m⁻¹ d⁻¹)",
                annotations=[dict(
                    text="No data available for the selected year.",
                    x=0.5, y=0.5, xref="paper", yref="paper",
                    showarrow=False
                )]
            )
        )

    fig = px.scatter(
        merged,
        x="Wind_spd_m_s",
        y="total_flux",
        color="Site",
        title="Wind speed vs wind erosion flux",
        labels={
            "Wind_spd_m_s": "Wind speed (m/s)",
            "total_flux": "Total flux (g m⁻¹ d⁻¹)"
        }
    )
    return fig



# ---------- THEME 4: Time Trends & Seasonality ----------
def fig_yield_trend(yield_by_year, treatment=None):
    """
    Multi-year trend in mean grain yield by treatment.
    Optional: filter by treatment.
    """
    df = yield_by_year.copy()

    if treatment and treatment != "all":
        df = df[df["Treatment"] == treatment]

    fig = px.line(
        df,
        x="Year",
        y="mean_yield",
        color="Treatment",
        markers=True,
        title="Trend in mean grain yield over time",
        labels={"mean_yield": "Mean grain yield (kg/ha)", "Treatment": "Treatment"}
    )
    fig.update_layout(hovermode="x unified")
    return fig



def fig_flux_trend(flux_by_year, site=None, treatment=None):
    """
    Multi-year trend in mean total mass flux by site & treatment.
    Optional filters by site and/or treatment.
    """
    df = flux_by_year.copy()

    if treatment and treatment != "all":
        df = df[df["Treatment"] == treatment]

    if site and site != "all":
        df = df[df["Site"] == site]

    fig = px.line(
        df,
        x="Year",
        y="mean_flux",
        color="Treatment",
        facet_row="Site",
        facet_row_spacing=0.12,
        markers=True,
        title="Trend in mean total mass flux by site and treatment",
    )

    # Remove facet titles like "Site=Mandan"
    fig.for_each_annotation(lambda a: a.update(text=""))

    # Remove per-row y titles
    fig.update_yaxes(title_text="", row=1, col=1)
    fig.update_yaxes(title_text="", row=2, col=1)

    # One global y-axis label for the whole figure
    fig.add_annotation(
        text="Mean total flux (g m⁻¹ d⁻¹)",
        x=-0.08,
        y=0.5,
        xref="paper",
        yref="paper",
        showarrow=False,
        textangle=-90,
        font=dict(size=14)
    )

    fig.update_layout(
        margin=dict(l=120, r=40, t=80, b=60),
        hovermode="x unified"
    )

    return fig



    # Move facet titles (Site=Mandan, Site=Morton) to left for readability
    fig.for_each_annotation(
        lambda a: a.update(x=0, xanchor="left", font=dict(size=12))
    )

    return fig



def fig_erosion_seasonality(massflux):
    """
    Seasonal pattern of wind erosion (monthly mean total flux).
    Answers:
    - Are there seasonal peaks in wind erosion (e.g., planting or dry periods)?
    """
    df = massflux.copy()
    df["Month_name"] = df["Month"].apply(
        lambda m: pd.Timestamp(year=2000, month=int(m), day=1).strftime("%b")
    )

    df = (
        df.groupby(["Site", "Month", "Month_name"], as_index=False)["total_flux"]
          .mean()
    )

    # Order months Jan–Dec
    month_order = ["Jan", "Feb", "Mar", "Apr", "May", "Jun",
                   "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]

    fig = px.line(
        df,
        x="Month_name",
        y="total_flux",
        color="Site",
        category_orders={"Month_name": month_order},
        title="Seasonal pattern of wind erosion (mean monthly flux)",
        labels={
            "total_flux": "Mean total flux (g m⁻¹ d⁻¹)",
            "Month_name": "Month",
            "Site": "Site"
        },
        markers=True
    )
    fig.update_layout(hovermode="x unified")
    return fig


def fig_weather_yearly_trends(weather):
    """
    NEW: Year-over-year weather trends.
    Answers:
    - How does weather vary year-over-year?
    Includes:
    - Mean annual wind speed (m/s)
    - Mean annual precipitation (mm/day)
    """
    df = weather.copy()
    df["Year"] = df["Date"].dt.year

    yearly = (
        df.groupby("Year", as_index=False)
          .agg(
              mean_wind=("Wind_spd_m_s", "mean"),
              mean_pcpn=("PCPN_mm_d", "mean")
          )
    )

    # Long format for Plotly Express
    long_df = yearly.melt(
        id_vars="Year",
        value_vars=["mean_wind", "mean_pcpn"],
        var_name="Variable",
        value_name="Mean_value"
    )

    var_label_map = {
        "mean_wind": "Mean wind speed (m/s)",
        "mean_pcpn": "Mean precipitation (mm/day)"
    }
    long_df["Variable_label"] = long_df["Variable"].map(var_label_map)

    fig = px.line(
        long_df,
        x="Year",
        y="Mean_value",
        color="Variable_label",
        markers=True,
        title="Yearly Weather Trends: Wind Speed and Precipitation",
        labels={
            "Year": "Year",
            "Mean_value": "Mean value",
            "Variable_label": "Weather variable"
        }
    )

    fig.update_layout(hovermode="x unified")
    return fig


# ---------- THEME 5: Treatment Performance ----------

def fig_treatment_overall_yield(treatment_perf):
    """
    Theme 5 – Plot 1
    Overall treatment performance: mean grain yield and AGB by treatment.
    Answers:
      - Which treatment system has higher yield?
      - Which treatment system has greater biomass accumulation?
    """
    df = treatment_perf.copy()

    # Long format: one row per Treatment × Metric
    long_df = df.melt(
        id_vars="Treatment",
        value_vars=["mean_yield", "mean_agb"],
        var_name="Metric",
        value_name="Mean_value"
    )

    metric_label = {
        "mean_yield": "Grain yield (kg/ha)",
        "mean_agb": "Aboveground biomass (g/m²)"
    }
    long_df["Metric_label"] = long_df["Metric"].map(metric_label)

    fig = px.bar(
        long_df,
        x="Treatment",
        y="Mean_value",
        color="Metric_label",
        barmode="group",
        title="Overall treatment performance: yield and biomass",
        labels={
            "Treatment": "Treatment",
            "Mean_value": "Mean value",
            "Metric_label": "Metric"
        }
    )
    return fig


def fig_treatment_cc_biomass(cc_perf):
    """
    Theme 5 – Plot 2
    Cover crop biomass by treatment and year.
    """
    # Work on a copy
    df = cc_perf.copy()

    # Drop rows where the mean is NaN
    df = df.dropna(subset=["mean_cc_agb"])

    # If still nothing left, return an empty figure with a message
    if df.empty:
        import plotly.graph_objects as go
        fig = go.Figure()
        fig.update_layout(
            title="Cover crop aboveground biomass by treatment and year",
            xaxis_title="Treatment",
            yaxis_title="Mean CC AGB (g/m²)",
            annotations=[
                dict(
                    text="No cover crop biomass data available",
                    x=0.5,
                    y=0.5,
                    xref="paper",
                    yref="paper",
                    showarrow=False,
                    font=dict(size=14)
                )
            ],
        )
        return fig

    # Normal bar plot if we have data
    fig = px.bar(
        df,
        x="TRT",
        y="mean_cc_agb",
        color="Year",
        barmode="group",
        title="Cover crop aboveground biomass by treatment and year",
        labels={"mean_cc_agb": "Mean CC AGB (g/m²)", "TRT": "Treatment"}
    )
    return fig




def fig_yield_by_crop_treatment_rank(harvest_theme):
    """
    Theme 5 – Plot 3 (updated)
    Mean grain yield by crop and treatment, side-by-side bars.

    Answers:
      - For each crop (corn, soybean, spring wheat),
        which treatment (BAU vs ASP) gives higher average yield?
    """
    df = harvest_theme.copy()

    # Aggregate over years: mean of mean_yield for each Crop–Treatment combo
    summary = (
        df.groupby(["Crop", "Treatment"], as_index=False)
          .agg(mean_yield=("mean_yield", "mean"))
    )

    # Optional: control crop order on x-axis
    crop_order = ["Corn", "Soybean", "SpringWheat"]
    summary["Crop"] = pd.Categorical(summary["Crop"], categories=crop_order, ordered=True)
    summary = summary.sort_values("Crop")

    fig = px.bar(
        summary,
        x="Crop",
        y="mean_yield",
        color="Treatment",
        barmode="group",
        title="Mean grain yield by crop and treatment",
        labels={
            "Crop": "Crop",
            "mean_yield": "Mean grain yield (kg/ha)",
            "Treatment": "Treatment"
        }
    )

    return fig







# ---------- DASH APP LAYOUT ----------
app = Dash(__name__)

available_years = sorted(harvest["Year"].dropna().unique())
available_sites = sorted(massflux["Site"].dropna().unique())
available_treatments = sorted(harvest["Treatment"].dropna().unique())

app.layout = html.Div([
    # --------- TOP HEADER BAR ----------
    html.Div(
        [
            html.H1(
                "Crop & Weather-Dependent Yield and Wind Erosion Benefits From a Conservation Practices System",
                style={
                    "margin": 0,
                    "fontSize": "32px",
                    "fontWeight": "700",
                },
            ),
            html.P(
                "Integrated view of crop yield, cover crops, weather, and wind erosion at Mandan & Morton",
                style={
                    "margin": "4px 0 0 0",
                    "fontSize": "14px",
                    "opacity": 0.9,
                },
            ),
        ],
        style={
            "backgroundColor": "#2e7d32",        # deep green
            "color": "white",                    # white text
            "textAlign": "center",
            "padding": "16px 10px",
            "marginBottom": "20px",
            "boxShadow": "0 2px 6px rgba(0, 0, 0, 0.15)",
        },
    ),
    
    
    

    # ---------------- KPI ROW ----------------
    html.Div([
        html.Div([
            html.H4("Avg Grain Yield (kg/ha)", style={"marginBottom": "0"}),
            html.H2(f"{harvest['Grain_yield_kg_ha'].mean():.0f}")
        ], style={"flex": 1, "padding": "10px",
                  "border": "1px solid #ddd", "borderRadius": "8px"}),

        html.Div([
            html.H4("Avg Mass Flux (g m⁻¹ d⁻¹)", style={"marginBottom": "0"}),
            html.H2(f"{massflux['total_flux'].mean():.2f}")
        ], style={"flex": 1, "padding": "10px",
                  "border": "1px solid #ddd", "borderRadius": "8px"}),

        html.Div([
            html.H4("Avg CC Biomass (g/m²)", style={"marginBottom": "0"}),
            html.H2(f"{agb_cc['CC_AGB'].mean():.0f}")
        ], style={"flex": 1, "padding": "10px",
                  "border": "1px solid #ddd", "borderRadius": "8px"}),
    ], style={
        "display": "flex",
        "gap": "20px",
        "marginBottom": "20px"
    }),

    # ---------------- TABS FOR THEMES ----------------
    dcc.Tabs([

        # ===== THEME 1 =====
        dcc.Tab(label="Theme 1 – Yield & Biomass", children=[
            html.Div([
                html.H2("Theme 1 – Crop Yield and Aboveground Biomass: How does Crop yield & biomass productivity vary by year, crop type, and management practice?"),
                html.P(
                    "This theme compares grain yield and aboveground biomass across crops "
                    "and management treatments over multiple years. Use the controls below "
                    "to explore how yield varies by crop, treatment, and year, or to see the "
                    "relationship between biomass and yield."
                ),

                html.Div([
                    html.Div([
                        html.Label("Year"),
                        dcc.Dropdown(
                            id="theme1-year",
                            options=[{"label": "All years", "value": "all"}] +
                                    [{"label": str(int(y)), "value": int(y)}
                                     for y in sorted(harvest_theme["Year"].unique())],
                            value="all",
                            clearable=False
                        )
                    ], style={"width": "220px"}),

                    html.Div([
                        html.Label("View"),
                        dcc.RadioItems(
                            id="theme1-question",
                            options=[
                                {"label": "Q1 – Yield by crop & treatment", "value": "q1"},
                                {"label": "Q2 – AGB over years by crop & treatment", "value": "q2"},
                            ],
                            value="q1",
                            labelStyle={"display": "block"}
                        )
                    ], style={"marginLeft": "40px"}),
                ], style={"display": "flex",
                          "alignItems": "flex-start",
                          "marginTop": "10px"}),
                
                
                            dcc.Graph(id="theme1-main-graph", style={"marginTop": "10px"}),

                # ---- SUMMARY BOX FOR THEME 1 (Q1 & Q2) ----
                html.Div(
                    id="theme1-summary",
                    style={
                        "backgroundColor": "#f8f9fa",
                        "padding": "12px 16px",
                        "borderRadius": "8px",
                        "border": "1px solid #ddd",
                        "width": "90%",
                        "margin": "15px auto 0",
                        "fontSize": "13px",
                    },
                )
            ], style={"padding": "20px"})
        ]),


        # ===== THEME 2 =====
        dcc.Tab(label="Theme 2 – Wind Erosion", children=[
            html.Div([
                html.H2("Theme 2 – Wind Erosion Mass Flux: How does mass flux vary across sampling locations and years?"),
                html.P(
                    "This theme summarizes wind erosion (mass flux) across sampling locations, "
                    "sites (Mandan vs Morton), and years. The main chart shows mean total flux "
                    "stacked by site for each sampling location, faceted by year. You can also "
                    "view a time-series of flux for a chosen site and treatment."
                ),

                html.Div([
                    html.Div([
                        html.Label("Site (for time-series view)"),
                        dcc.Dropdown(
                            id="theme2-site",
                            options=[{"label": s, "value": s} for s in available_sites],
                            value=None
                        )
                    ], style={"width": "220px"}),

                    html.Div([
                        html.Label("Treatment (for time-series view)"),
                        dcc.Dropdown(
                            id="theme2-treatment",
                            options=[{"label": t, "value": t} for t in available_treatments],
                            value=None
                        )
                    ], style={"width": "260px", "marginLeft": "20px"}),

                    html.Div([
                        html.Label("View"),
                        dcc.RadioItems(
                            id="theme2-question",
                            options=[
                                {"label": "Q3 – Flux by location, site, and year", "value": "q3"},
                                {"label": "Time-series by site & treatment", "value": "ts"},
                            ],
                            value="q3",
                            labelStyle={"display": "block"}
                        )
                    ], style={"marginLeft": "40px"}),
                ], style={"display": "flex",
                          "alignItems": "flex-start",
                          "marginTop": "10px"}),
                
                                dcc.Graph(id="theme2-main-graph", style={"marginTop": "10px"}),

                # ---- SUMMARY BOX FOR THEME 2 ----
                html.Div(
                    id="theme2-summary",
                    style={
                        "backgroundColor": "#f8f9fa",
                        "padding": "12px 16px",
                        "borderRadius": "8px",
                        "border": "1px solid #ddd",
                        "width": "90%",
                        "margin": "15px auto 0",
                        "fontSize": "13px",
                    },
                )
            ], style={"padding": "20px"})
        ]),


 # ===== THEME 3 =====
dcc.Tab(label="Theme 3 – Weather & Environment", children=[
    html.Div([
        html.H2("Theme 3 – Weather Conditions and Erosion Drivers: How do wind speed patterns/Precipitation influence daily or seasonal erosion?"),
        html.P(
            "This theme connects weather variables to wind erosion. "
            "The first chart (Q1) shows how growing-season wind speed is related "
            "to mean annual wind erosion (mass flux) by site and treatment. "
            "The other charts explore relationships between precipitation, wind "
            "speed, and erosion flux."
        ),

        # ---------- THEME 3 SUMMARY (NOW ON TOP) ----------
        html.Div(
            id="theme3-summary",
            children=[
                html.H4("Summary of Key Insights", style={"marginBottom": "6px"}),

                html.P(
                    "• Higher wind speeds are generally associated with stronger wind erosion, "
                    "especially during peak erosion years.",
                    style={"margin": "2px 0"}
                ),
                html.P(
                    "• Precipitation events tend to suppress erosion, with most high-flux days "
                    "occurring when rainfall is minimal or absent.",
                    style={"margin": "2px 0"}
                ),
                html.P(
                    "• Mandan and Morton differ in how strongly weather influences erosion, "
                    "showing how local conditions modify climate impacts.",
                    style={"margin": "2px 0"}
                ),
                html.P(
                    "• Overall, variability in wind intensity and rainfall plays a major role "
                    "in shaping erosion patterns across the landscape.",
                    style={"margin": "2px 0"}
                ),
            ],
            style={
                "backgroundColor": "#f8f9fa",
                "padding": "12px 16px",
                "borderRadius": "8px",
                "border": "1px solid #ddd",
                "width": "90%",
                "margin": "15px auto 25px auto",
                "fontSize": "13px",
            },
        ),

        # ---------- FILTERS ----------
        html.Div([
            html.Div([
                html.Label("Select Year"),
                dcc.Dropdown(
                    id="theme3-year",
                    options=[{"label": "All years", "value": "all"}] +
                            [{"label": str(int(y)), "value": int(y)}
                             for y in sorted(weather["Date"].dt.year.dropna().unique())],
                    value="all",
                    clearable=False
                )
            ], style={"width": "220px"}),
        ], style={"display": "flex", "marginBottom": "20px"}),

        # ---------- GRAPHS ----------
        dcc.Graph(id="theme3-weather-field"),
        dcc.Graph(id="theme3-precip-flux"),
        dcc.Graph(id="theme3-wind-flux"),

    ], style={"padding": "20px"})
]),




  # ===== THEME 4 =====
dcc.Tab(label="Theme 4 – Time Trends", children=[
    html.Div([
        html.H2("Theme 4 – Multi-year Trends and Seasonal Patterns"),
        html.P(
            "This theme highlights how crop yield, erosion, and weather change over years "
            "and how wind erosion is distributed through the year. It also helps reveal "
            "whether conservation treatments improve outcomes over time."
        ),

        # ---------- THEME 4 SUMMARY (NOW AT THE TOP) ----------
        html.Div([
            html.H4("Summary of Key Insights", style={"marginBottom": "6px"}),

            html.P(
                "• Grain yield shows clear year-to-year variability, with both ASP and BAU dipping sharply "
                "in 2021 before recovering in 2022.",
                style={"margin": "2px 0"}
            ),
            html.P(
                "• Wind erosion trends differ by treatment, with BAU showing stronger fluctuations while "
                "ASP remains more stable across years.",
                style={"margin": "2px 0"}
            ),
            html.P(
                "• Weather patterns shift over time—wind speeds oscillate while precipitation drops in "
                "drier years, influencing erosion conditions.",
                style={"margin": "2px 0"}
            ),
            html.P(
                "• Seasonal erosion peaks in early summer and steadily declines into fall, following the "
                "typical dry-windy seasonal pattern.",
                style={"margin": "2px 0"}
            ),
        ], style={
            "backgroundColor": "#f4f4f4",
            "padding": "12px 18px",
            "borderRadius": "8px",
            "marginTop": "10px",
            "marginBottom": "20px",
            "border": "1px solid #ddd"
        }),

        # ---------- FILTER CONTROLS ----------
        html.Div([
            html.Div([
                html.Label("Treatment"),
                dcc.Dropdown(
                    id="theme4-treatment",
                    options=[{"label": "All treatments", "value": "all"}] +
                            [{"label": t, "value": t} for t in available_treatments],
                    value="all",
                    clearable=False
                )
            ], style={"width": "220px"}),

            html.Div([
                html.Label("Site (for flux trend)"),
                dcc.Dropdown(
                    id="theme4-site",
                    options=[{"label": "All sites", "value": "all"}] +
                            [{"label": s, "value": s} for s in available_sites],
                    value="all",
                    clearable=False
                )
            ], style={"width": "220px", "marginLeft": "20px"}),
        ], style={
            "display": "flex",
            "marginTop": "10px",
            "marginBottom": "20px"
        }),

        # ---------- GRAPHS ----------
        dcc.Graph(id="theme4-yield-trend"),
        dcc.Graph(id="theme4-flux-trend"),
        dcc.Graph(id="theme4-weather-trend"),
        dcc.Graph(id="theme4-seasonality"),

    ], style={"padding": "20px"})
]),


         # ===== THEME 5 =====
        dcc.Tab(label="Theme 5 – Treatment Performance", children=[
            html.Div([
                html.H2("Theme 5 – Conservation Treatment Performance"),
                html.P(
                    "This theme compares conservation practices (ASP vs BAU) in terms of grain yield, "
                    "cover-crop biomass, and crop-specific treatment performance."
                ),

                # ---------- THEME 5 SUMMARY BOX ----------
                html.Div(
                    [
                        html.H4("Summary of Key Insights", style={"marginBottom": "6px"}),
                        html.P(
                            "• BAU generally produces higher grain yield than ASP across most crops, especially corn and spring wheat.\n"
                            "• Aboveground biomass is also higher under BAU, indicating stronger total plant growth.\n"
                            "• Crop-specific responses vary: ASP performs closer to BAU in soybean but lags behind in high-yield crops.\n"
                            "• Overall, BAU remains more productive, while ASP shows moderate, crop-dependent outcomes.",
                            style={"whiteSpace": "pre-line", "opacity": 0.85}
                        )
                    ],
                    style={
                        "backgroundColor": "#f8f9fa",
                        "border": "1px solid #ddd",
                        "padding": "12px 16px",
                        "borderRadius": "6px",
                        "marginTop": "10px",
                        "marginBottom": "20px"
                    }
                ),

                html.Div([
                    html.Div([
                        html.Label("View"),
                        dcc.Dropdown(
                            id="theme5-view",
                            options=[
                                {"label": "Q1 – Overall treatment performance (yield + biomass)", "value": "q1"},
                                # Q2 removed because no data is available
                                {"label": "Q3 – Crop × treatment yield comparison", "value": "q3"},
                            ],
                            value="q1",
                            clearable=False,
                            style={"width": "420px"}
                        )
                    ]),
                ], style={"marginTop": "10px", "marginBottom": "20px"}),

                dcc.Graph(id="theme5-main-graph")
            ], style={"padding": "20px"})
        ]),
    ])
])






# ---------- CALLBACKS ----------

# THEME 1
@app.callback(
    [
        Output("theme1-main-graph", "figure"),
        Output("theme1-summary", "children"),
    ],
    [
        Input("theme1-year", "value"),
        Input("theme1-question", "value"),
    ],
)
def update_theme1(year, question):
    year_filter = None if (year is None or year == "all") else year

    if question == "q1":
        # ---- Q1: Grain yield by crop & treatment ----
        fig = fig_yield_by_crop_treatment(harvest_theme, year_filter)

        summary_children = [
            html.H4("Summary for Grain Yield Visualization (Theme 1 – Q1)"),
            html.P("• Corn under BAU has the highest average grain yield across all years."),
            html.P("• For both Corn and Spring Wheat, BAU slightly outperforms ASP, indicating a yield advantage under BAU."),
            html.P("• Soybean has the lowest yields overall, with ASP performing slightly better than BAU for soybean."),
            html.P("• Overall, cereals respond strongly to BAU, while ASP narrows the gap for soybean, highlighting crop-specific treatment responses."),
        ]

    else:
        # ---- Q2: AGB variation over years ----
        fig = fig_agb_vs_yield_scatter(harvest_theme, year_filter)

        summary_children = [
            html.H4("Summary for AGB Visualization (Theme 1 – Q2)"),
            html.P("• Above-ground biomass generally declines from 2019 to 2022 across most crops."),
            html.P("• Corn shows the highest biomass, while Soybean remains consistently low."),
            html.P("• BAU management produces higher biomass than ASP for Wheat and Corn, while ASP slightly favors Soybean."),
            html.P("• Overall, the patterns highlight crop-specific responses to management practices and changing yearly conditions."),
        ]

    return fig, summary_children




# THEME 2
@app.callback(
    [
        Output("theme2-main-graph", "figure"),
        Output("theme2-summary", "children"),
    ],
    [
        Input("theme2-site", "value"),
        Input("theme2-treatment", "value"),
        Input("theme2-question", "value"),
    ]
)
def update_theme2(site, treatment, question):
    if question == "q3":
        # ---- Q3: Flux by location, site, and year (stacked bars) ----
        df = massflux.copy()
        if site:
            df = df[df["Site"] == site]
        if treatment:
            df = df[df["Treatment"] == treatment]

        fig = fig_flux_location_site_year(df)

        summary_children = [
            html.H4("Summary for Flux by Location, Site, and Year (Theme 2 – Q3)"),
            html.P("• Wind erosion intensity varies widely among sampling locations, with a few hotspots driving much of the total flux."),
            html.P("• Morton often has higher mean flux than Mandan at the same location, especially in high-flux years such as 2020."),
            html.P("• Some locations switch which site dominates from year to year, showing how local surface conditions and weather interact."),
            html.P("• Overall, erosion risk is highly localized, so conservation practices should target the highest-flux locations rather than be applied uniformly."),
        ]

    else:
        # ---- Time-series by site & treatment ----
        fig = fig_flux_time_series(massflux, site=site, treatment=treatment)

        summary_children = [
            html.H4("Summary for Time-series Flux by Site (Theme 2 – Time-series)"),
            html.P("• Annual mean flux is generally higher at Morton in 2019–2020, while Mandan shows a sharp spike around 2021 before dropping again."),
            html.P("• Morton follows a smoother, more stable pattern across years, whereas Mandan is more variable."),
            html.P("• The strong 2021 peak at Mandan highlights how a single windy or dry year can dominate the erosion record."),
            html.P("• Using the site and treatment filters lets you zoom in on when and where erosion is greatest to better time conservation actions."),
        ]

    return fig, summary_children




# THEME 3
@app.callback(
    Output("theme3-weather-field", "figure"),
    Output("theme3-precip-flux", "figure"),
    Output("theme3-wind-flux", "figure"),
    Input("theme3-year", "value")
)
def update_theme3(selected_year):
    # "all" or None  → no filter
    year_filter = None if (selected_year is None or selected_year == "all") else int(selected_year)

    fig1 = fig_theme3_q1_wind_vs_flux(weather, massflux, year_filter)
    fig2 = fig_precip_vs_flux(weather, massflux, year_filter)
    fig3 = fig_wind_vs_flux(weather, massflux, year_filter)

    return fig1, fig2, fig3




# THEME 4
@app.callback(
    Output("theme4-yield-trend", "figure"),
    Output("theme4-flux-trend", "figure"),
    Output("theme4-weather-trend", "figure"),
    Output("theme4-seasonality", "figure"),
    Input("theme4-treatment", "value"),
    Input("theme4-site", "value")
)
def update_theme4(selected_treatment, selected_site):
    fig1 = fig_yield_trend(yield_by_year, treatment=selected_treatment)
    fig2 = fig_flux_trend(flux_by_year, site=selected_site, treatment=selected_treatment)
    fig3 = fig_weather_yearly_trends(weather)
    fig4 = fig_erosion_seasonality(massflux)
    return fig1, fig2, fig3, fig4



# THEME 5
@app.callback(
    Output("theme5-main-graph", "figure"),
    Input("theme5-view", "value")
)
def update_theme5(view):
    """
    Switch Theme 5 figure based on the selected question/view.
    """
    if view == "q1":
        # Overall treatment performance: yield + biomass
        return fig_treatment_overall_yield(treatment_perf)

    elif view == "q2":
        # Cover crop biomass by treatment and year
        return fig_treatment_cc_biomass(cc_perf)

    else:  # "q3"
        # Crop × treatment mean yield (BAU vs ASP side-by-side for each crop)
        return fig_yield_by_crop_treatment_rank(harvest_theme)




if __name__ == "__main__":
    app.run(debug=True)


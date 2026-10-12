"""Consistent, descriptive Plotly figures for the AgriSight dashboard.

The chart captions in app.py explain important limitations of the source data.
No causal conclusions or modelled forecasts are implied by the charts.
"""
from calendar import month_abbr

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

NAVY = "#162846"
BLUE = "#2677E8"
TEAL = "#16A19A"
GOLD = "#E4A03D"
MUTED = "#6F8096"
TREATMENT = {"BAU": BLUE, "ASP": TEAL}
SITE = {"Mandan": BLUE, "Morton": GOLD}
CROP_ORDER = ["Corn", "Spring wheat", "Soybean"]
MONTH_ORDER = list(range(1, 13))


def style_figure(fig, height=440, legend=True):
    fig.update_layout(
        template="plotly_white",
        height=height,
        autosize=True,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="Inter, Segoe UI, Arial, sans-serif", color=NAVY, size=14),
        margin=dict(l=86, r=38, t=66, b=80),
        legend=dict(
            orientation="h", x=0, y=1.03, yanchor="bottom",
            title_text="", font=dict(size=16, color=MUTED),
            bgcolor="rgba(255,255,255,0.85)"
        ),
        showlegend=legend,
        hoverlabel=dict(bgcolor="white", font_color=NAVY, font_size=14),
        hovermode="closest",
    )
    fig.update_xaxes(showline=False, zeroline=False, showgrid=False,
                     tickfont=dict(color=MUTED, size=13), title_font=dict(color=MUTED, size=13),
                     automargin=True)
    fig.update_yaxes(showline=False, zeroline=False, gridcolor="#E9EEF6",
                     tickfont=dict(color=MUTED, size=13), title_font=dict(color=MUTED, size=13),
                     automargin=True)
    # Plotly bar labels were also too small relative to chart axis values.
    fig.update_traces(textfont_size=13, selector=dict(type="bar"))
    return fig


def empty_figure(message="No observations match these filters."):
    fig = go.Figure()
    fig.add_annotation(x=0.5, y=0.5, xref="paper", yref="paper", text=message,
                       showarrow=False, font=dict(size=16, color=MUTED), align="center")
    fig.update_xaxes(visible=False)
    fig.update_yaxes(visible=False)
    return style_figure(fig)


def yield_by_crop(harvest):
    df = harvest.dropna(subset=["Grain_yield_kg_ha"])
    if df.empty:
        return empty_figure()
    summary = df.groupby(["Crop", "Treatment"], as_index=False).agg(
        value=("Grain_yield_kg_ha", "mean"), n=("Grain_yield_kg_ha", "size")
    )
    fig = px.bar(summary, x="Crop", y="value", color="Treatment", barmode="group",
                 color_discrete_map=TREATMENT, category_orders={"Crop": CROP_ORDER},
                 custom_data=["n"], text_auto=",.0f")
    fig.update_traces(marker_line_width=0, textposition="outside", cliponaxis=False,
                      hovertemplate="%{x}<br>Average yield: %{y:,.0f} kg/ha<br>Observations: %{customdata[0]}<extra>%{fullData.name}</extra>")
    fig.update_yaxes(title_text="Mean yield (kg/ha)", rangemode="tozero")
    fig.update_xaxes(title_text="")
    return style_figure(fig, legend=summary["Treatment"].nunique() > 1)


def yield_by_year(harvest):
    if harvest.empty:
        return empty_figure()
    df = harvest.groupby(["Year", "Treatment"], as_index=False).agg(
        value=("Grain_yield_kg_ha", "mean"), n=("Grain_yield_kg_ha", "size")
    )
    fig = px.line(df, x="Year", y="value", color="Treatment", markers=True,
                  color_discrete_map=TREATMENT, custom_data=["n"])
    fig.update_traces(line_width=3, marker=dict(size=9, line=dict(color="white", width=2)),
                      hovertemplate="Year %{x}<br>Mean yield: %{y:,.0f} kg/ha<br>Observations: %{customdata[0]}<extra>%{fullData.name}</extra>")
    fig.update_xaxes(dtick=1, title_text="Year")
    fig.update_yaxes(title_text="Mean yield (kg/ha)", rangemode="tozero")
    return style_figure(fig, legend=df["Treatment"].nunique() > 1)


def biomass_relationship(harvest):
    df = harvest.dropna(subset=["AGB_g_m2", "Grain_yield_kg_ha"])
    if df.empty:
        return empty_figure()
    df = df.groupby(["Year", "Crop", "Treatment"], as_index=False).agg(
        biomass=("AGB_g_m2", "mean"), grain=("Grain_yield_kg_ha", "mean"), n=("Grain_yield_kg_ha", "size")
    )
    fig = px.scatter(df, x="biomass", y="grain", color="Treatment", symbol="Crop",
                     size="n", size_max=17, color_discrete_map=TREATMENT,
                     custom_data=["Year", "Crop", "n"])
    fig.update_traces(marker=dict(opacity=0.88, line=dict(color="white", width=1.5)),
                      hovertemplate="%{customdata[1]} · %{customdata[0]}<br>Mean biomass: %{x:,.0f} g/m²<br>Mean grain yield: %{y:,.0f} kg/ha<br>Observations: %{customdata[2]}<extra>%{fullData.name}</extra>")
    fig.update_xaxes(title_text="Mean biomass (g/m²)")
    fig.update_yaxes(title_text="Mean grain yield (kg/ha)")
    return style_figure(fig, height=460)


def flux_by_year(erosion):
    if erosion.empty:
        return empty_figure()
    df = erosion.groupby(["Year", "Site"], as_index=False).agg(
        value=("total_flux", "mean"), n=("total_flux", "size")
    )
    fig = px.line(df, x="Year", y="value", color="Site", markers=True,
                  color_discrete_map=SITE, custom_data=["n"])
    fig.update_traces(line_width=3, marker=dict(size=9, line=dict(color="white", width=2)),
                      hovertemplate="Year %{x}<br>Mean flux: %{y:.2f}<br>Records: %{customdata[0]}<extra>%{fullData.name}</extra>")
    fig.update_xaxes(dtick=1, title_text="Year")
    fig.update_yaxes(title_text="Mean mass flux (source units)", rangemode="tozero")
    return style_figure(fig, legend=df["Site"].nunique() > 1)


def flux_by_site(erosion):
    if erosion.empty:
        return empty_figure()
    df = erosion.groupby(["Site", "Treatment"], as_index=False).agg(
        value=("total_flux", "mean"), n=("total_flux", "size")
    )
    df["Group"] = df["Site"] + " · " + df["Treatment"]
    fig = px.bar(df, x="Group", y="value", color="Site", color_discrete_map=SITE,
                 custom_data=["n", "Treatment"], text_auto=".2f")
    fig.update_traces(marker_line_width=0, textposition="outside", cliponaxis=False,
                      hovertemplate="%{x}<br>Mean mass flux: %{y:.2f}<br>Records: %{customdata[0]}<extra></extra>")
    fig.update_xaxes(title_text="Site and observed treatment")
    fig.update_yaxes(title_text="Mean mass flux (source units)", rangemode="tozero")
    return style_figure(fig, legend=False)


def flux_hotspots(erosion):
    if erosion.empty:
        return empty_figure()
    df = erosion.groupby(["Site", "Location"], as_index=False).agg(
        value=("total_flux", "mean"), n=("total_flux", "size")
    ).sort_values("value", ascending=False).head(10)
    df["Label"] = df["Site"] + " · " + df["Location"]
    df = df.sort_values("value")
    fig = px.bar(df, x="value", y="Label", color="Site", orientation="h",
                 color_discrete_map=SITE, text_auto=".1f", custom_data=["n"])
    fig.update_traces(marker_line_width=0, textposition="outside", cliponaxis=False,
                      hovertemplate="%{y}<br>Mean mass flux: %{x:.2f}<br>Records: %{customdata[0]}<extra></extra>")
    fig.update_xaxes(title_text="Mean mass flux (source units)")
    fig.update_yaxes(title_text="")
    return style_figure(fig, height=540, legend=df["Site"].nunique() > 1)


def flux_seasonality(erosion):
    df = erosion.dropna(subset=["Month"])
    if df.empty:
        return empty_figure()
    df = df.groupby(["Month", "Site"], as_index=False).agg(value=("total_flux", "mean"), n=("total_flux", "size"))
    df["Month_name"] = df["Month"].astype(int).map(lambda m: month_abbr[m])
    fig = px.line(df, x="Month_name", y="value", color="Site", markers=True,
                  category_orders={"Month_name": [month_abbr[i] for i in MONTH_ORDER]},
                  color_discrete_map=SITE, custom_data=["n"])
    fig.update_traces(line_width=3, marker=dict(size=8),
                      hovertemplate="%{x}<br>Mean mass flux: %{y:.2f}<br>Records: %{customdata[0]}<extra>%{fullData.name}</extra>")
    fig.update_xaxes(title_text="Month")
    fig.update_yaxes(title_text="Mean mass flux (source units)", rangemode="tozero")
    return style_figure(fig, legend=df["Site"].nunique() > 1)


def weather_monthly(weather, metric, label, color):
    df = weather.dropna(subset=["Month", metric])
    if df.empty:
        return empty_figure("No NP weather observations for this selection.")
    df = df.groupby("Month", as_index=False).agg(value=(metric, "mean"), n=(metric, "size"))
    df["Month_name"] = df["Month"].astype(int).map(lambda m: month_abbr[m])
    fig = px.area(df, x="Month_name", y="value", category_orders={"Month_name": [month_abbr[i] for i in MONTH_ORDER]})
    fig.update_traces(line=dict(color=color, width=3), fillcolor="rgba(38,119,232,0.10)" if color == BLUE else "rgba(22,161,154,0.10)",
                      mode="lines+markers", marker=dict(size=7, color=color),
                      hovertemplate="%{x}<br>" + label + ": %{y:.2f}<extra></extra>")
    fig.update_xaxes(title_text="Month")
    fig.update_yaxes(title_text=label, rangemode="tozero")
    return style_figure(fig, legend=False)


def weather_annual(weather):
    df = weather.dropna(subset=["Wind_spd_m_s"])
    if df.empty:
        return empty_figure("No NP weather observations for this selection.")
    df = df.groupby("Year", as_index=False).agg(value=("Wind_spd_m_s", "mean"), n=("Wind_spd_m_s", "size"))
    fig = px.bar(df, x="Year", y="value", text_auto=".2f", custom_data=["n"])
    fig.update_traces(marker_color=BLUE, marker_line_width=0, textposition="outside",
                      hovertemplate="Year %{x}<br>Mean daily wind: %{y:.2f} m/s<br>Field-day rows: %{customdata[0]}<extra></extra>")
    fig.update_xaxes(type="category", title_text="Year")
    fig.update_yaxes(title_text="Mean daily wind speed (m/s)", rangemode="tozero")
    return style_figure(fig, legend=False)


def weather_flux_context(weather, erosion):
    # These are annual-context comparisons, not station/site-matched or causal.
    wx = weather.groupby("Year", as_index=False)["Wind_spd_m_s"].mean().rename(columns={"Wind_spd_m_s": "wind"})
    fx = erosion.groupby(["Year", "Site"], as_index=False)["total_flux"].mean().rename(columns={"total_flux": "flux"})
    df = fx.merge(wx, on="Year", how="inner").dropna(subset=["wind", "flux"])
    if df.empty:
        return empty_figure("No overlapping weather and erosion years for this selection.")
    fig = px.scatter(df, x="wind", y="flux", color="Site", text="Year", color_discrete_map=SITE)
    fig.update_traces(marker=dict(size=12, line=dict(color="white", width=2)), textposition="top center",
                      hovertemplate="Regional wind: %{x:.2f} m/s<br>Mean flux: %{y:.2f}<extra>%{fullData.name}</extra>")
    fig.update_xaxes(title_text="Annual NP mean wind speed (m/s)")
    fig.update_yaxes(title_text="Erosion mean flux (source units)")
    return style_figure(fig, legend=df["Site"].nunique() > 1)


def crop_year_heatmap(harvest):
    if harvest.empty:
        return empty_figure()
    df = harvest.pivot_table(index="Crop", columns="Year", values="Grain_yield_kg_ha", aggfunc="mean")
    df = df.reindex([c for c in CROP_ORDER if c in df.index])
    if df.empty:
        return empty_figure()
    fig = go.Figure(go.Heatmap(
        z=df.values, x=df.columns.astype(int).astype(str).tolist(), y=df.index.tolist(),
        colorscale=[[0,"#E8F3FC"],[0.5,"#9BC9F4"],[1,"#2369C9"]],
        text=np.where(np.isfinite(df.values), np.round(df.values).astype(str), ""),
        texttemplate="%{z:,.0f}", textfont=dict(size=14),
        hovertemplate="%{y} · %{x}<br>Mean yield: %{z:,.0f} kg/ha<extra></extra>",
        colorbar=dict(title=dict(text="kg/ha", side="top"), thickness=12, len=0.72)
    ))
    fig.update_xaxes(title_text="Year", type="category")
    fig.update_yaxes(title_text="")
    return style_figure(fig, legend=False)


def treatment_gap(harvest):
    if harvest.empty or harvest["Treatment"].nunique() < 2:
        return empty_figure("Select both treatments to compare observed yield differences.")
    df = harvest.groupby(["Crop", "Treatment"])["Grain_yield_kg_ha"].mean().unstack()
    df = df.dropna(subset=["BAU", "ASP"], how="any")
    if df.empty:
        return empty_figure("No crops have observations for both selected treatments.")
    df["gap"] = (df["ASP"] - df["BAU"]) / df["BAU"] * 100
    df = df.reset_index().sort_values("gap")
    colors = [TEAL if v >= 0 else BLUE for v in df["gap"]]
    fig = go.Figure(go.Bar(
        x=df["gap"], y=df["Crop"], orientation="h", marker_color=colors,
        text=[f"{v:+.1f}%" for v in df["gap"]], textposition="outside",
        customdata=np.stack([df["ASP"], df["BAU"]], axis=-1),
        hovertemplate="%{y}<br>ASP vs BAU: %{x:+.1f}%<br>ASP: %{customdata[0]:,.0f} kg/ha<br>BAU: %{customdata[1]:,.0f} kg/ha<extra></extra>"
    ))
    fig.add_vline(x=0, line_width=1, line_color="#8C9CAF")
    fig.update_xaxes(title_text="Observed ASP difference from BAU (%)", zeroline=False)
    fig.update_yaxes(title_text="")
    return style_figure(fig, legend=False)


def cover_crop_history(cover_crop):
    if cover_crop.empty:
        return empty_figure("No measured ASP cover-crop biomass for this selection.")
    df = cover_crop.groupby("Year", as_index=False).agg(value=("CC_AGB", "mean"), n=("CC_AGB", "size"))
    fig = px.bar(df, x="Year", y="value", text_auto=".1f", custom_data=["n"])
    fig.update_traces(marker_color=TEAL, marker_line_width=0, textposition="outside", cliponaxis=False,
                      hovertemplate="Year %{x}<br>Mean cover-crop biomass: %{y:.1f} g/m²<br>Measurements: %{customdata[0]}<extra>ASP only</extra>")
    fig.update_xaxes(type="category", title_text="Year")
    fig.update_yaxes(title_text="Mean cover-crop biomass (g/m²)", rangemode="tozero")
    return style_figure(fig, legend=False)

"""AgriSight | Interactive agricultural performance and erosion analytics.

Start with: python app.py
Dash auto-loads assets/styles.css from the assets directory.
"""

from dash import Dash, Input, Output, State, dcc, html

import dashboard_charts as charts
from dashboard_data import apply_filters, load_datasets, records_for_export

DATA = load_datasets()
app = Dash(__name__, title="AgriSight | Field Performance Analytics")
server = app.server

PAGES = {
    "overview": ("Performance overview", "Yield, erosion, and management observations in one decision-ready view"),
    "yield": ("Yield & biomass", "Explore differences across crops, years, and management systems"),
    "erosion": ("Wind erosion", "Locate mass-flux hotspots and inspect seasonal variation"),
    "weather": ("Weather context", "Track NP field-weather conditions and explore cross-dataset context"),
    "trends": ("Time trends", "Find year-to-year patterns without mixing measurement units"),
    "treatment": ("Treatment performance", "Compare observed harvest outcomes and examine ASP cover-crop records"),
}

CHART_CONFIG = {
    "displaylogo": False,
    "responsive": True,
    "scrollZoom": False,
    "modeBarButtonsToRemove": ["lasso2d", "select2d", "autoScale2d"],
    "toImageButtonOptions": {"format": "png", "scale": 2, "filename": "agrisight-chart"},
}


def metric_card(label, number, context, badge=None):
    return html.Div(
        [
            html.Div([html.Span(label, className="metric-label"),
                      html.Span(badge, className="metric-badge") if badge else None],
                     className="metric-topline"),
            html.Div(number, className="metric-number"),
            html.Div(context, className="metric-context"),
        ], className="metric-card"
    )


# Short reading guides make the visuals understandable without requiring users to
# recognize chart types or infer what the height of a line/bar represents.
CHART_GUIDES = {
    "Grain yield by crop": "Read it: compare the colored bars within each crop. Higher bars mean higher average grain yield (kg/ha).",
    "Erosion across years": "Read it: each point shows the average recorded mass flux at that site in that year; higher points mean a higher observed mean.",
    "Yield over time": "Read it: follow each treatment line across years to see how average grain yield changes.",
    "Crop and treatment comparison": "Read it: each color represents a management treatment. Compare the bars for the same crop.",
    "Yield trajectory": "Read it: compare the lines within the same year, then follow them to see how averages change over time.",
    "Biomass vs. grain yield": "Read it: each bubble summarizes a crop, year and treatment. Its size represents the number of observations.",
    "Mean flux by site": "Read it: taller bars show larger average recorded flux. These site differences do not isolate treatment effects.",
    "Annual erosion trend": "Read it: compare year-to-year mean mass flux for each site, not the total amount across all samples.",
    "Top sampling hotspots": "Read it: the longest bars mark the sampled locations with the highest average recorded mass flux.",
    "Monthly erosion profile": "Read it: each point shows the mean flux of recorded events in that month, grouped by site.",
    "Seasonal wind profile": "Read it: a higher point indicates a month with higher observed average wind speed.",
    "Seasonal precipitation": "Read it: a higher point indicates a month with greater mean daily rainfall, not total monthly rainfall.",
    "Yearly regional wind": "Read it: each bar summarizes the regional weather observations for that year, not a specific erosion site.",
    "Weather vs. erosion: exploratory": "Read it: each dot combines a year's regional average wind and a site's average erosion. This is not a matched-site or causal comparison.",
    "Annual yield trend": "Read it: each line follows the observed mean yield for a treatment across years.",
    "Crop–year yield heatmap": "Read it: darker cells indicate higher average yield. Compare crops and years; blank cells have no observations.",
    "Seasonal erosion": "Read it: follow monthly averages for each site to identify months with higher recorded erosion.",
    "Observed crop outcomes": "Read it: compare the two treatment bars for the same crop, rather than the totals across different crops.",
    "ASP compared with BAU": "Read it: bars to the right of zero mean ASP had a higher observed average; left means lower. These are not causal effects.",
    "Cover-crop biomass": "Read it: each bar is average measured cover-crop biomass under ASP in that year. BAU measurements were not provided.",
}


def chart_card(title, subtitle, figure, note=None, class_name=""):
    # Give each Graph an explicit, stable height. Using responsive=True without
    # a sized parent made Plotly collapse into very short chart areas in CSS grid.
    figure_height = int(figure.layout.height or 450)
    contents = [
        html.Div([html.H3(title, className="card-title"),
                  html.P(subtitle, className="card-subtitle")], className="card-heading"),
        dcc.Graph(figure=figure, config=CHART_CONFIG, className="chart",
                  style={"height": f"{figure_height}px", "width": "100%"}),
    ]
    if title in CHART_GUIDES:
        contents.append(html.P(CHART_GUIDES[title], className="chart-guide"))
    if note:
        contents.append(html.Div(note, className="chart-note"))
    return html.Section(contents, className="chart-card " + class_name)


def narrative(title, text, variant="normal", eyebrow="ANALYST NOTE"):
    if isinstance(text, str):
        text = [text]
    return html.Section(
        [html.Div(eyebrow, className="note-eyebrow"),
         html.H3(title, className="note-title")]
        + [html.P(item) for item in text],
        className="narrative-card " + ("narrative-warning" if variant == "warning" else "")
    )


def format_number(value, digits=0):
    return f"{value:,.{digits}f}" if value is not None else "—"


def summarize_kpis(d):
    h, e, c = d.harvest, d.erosion, d.cover_crop
    return html.Div([
        metric_card("Average grain yield",
                    format_number(h["Grain_yield_kg_ha"].mean()) if not h.empty else "—",
                    "kg/ha · observed harvest mean", "YIELD"),
        metric_card("Average mass flux",
                    format_number(e["total_flux"].mean(), 2) if not e.empty else "—",
                    "Source flux units · event mean", "EROSION"),
        metric_card("Harvest observations", f"{len(h):,}",
                    "Records matching harvest filters", "SAMPLE"),
        metric_card("ASP cover-crop biomass",
                    format_number(c["CC_AGB"].mean(), 1) if not c.empty else "—",
                    "g/m² · ASP only, where available", "BIOMASS"),
    ], className="kpi-grid")


def overview_content(d):
    h, e = d.harvest, d.erosion
    if not h.empty:
        crop_yield = h.groupby("Crop")["Grain_yield_kg_ha"].mean().sort_values(ascending=False)
        yield_sentence = f"{crop_yield.index[0]} has the highest observed mean grain yield ({crop_yield.iloc[0]:,.0f} kg/ha) within your harvest filters."
    else:
        yield_sentence = "No harvest records match the current combination of filters. Clear a filter to restore the yield view."
    if not e.empty:
        peak = e.groupby("Site")["total_flux"].mean().sort_values(ascending=False)
        erosion_sentence = f"{peak.index[0]} has the highest mean mass flux ({peak.iloc[0]:.2f} in source units) within the erosion selection."
    else:
        erosion_sentence = "No erosion records match this selection. Site and treatment are linked in the supplied data."
    return [
        html.Div([
            chart_card("Grain yield by crop", "Average harvest yield, split by treatment", charts.yield_by_crop(h)),
            chart_card("Erosion across years", "Observed annual average mass flux by field site", charts.flux_by_year(e)),
        ], className="two-column"),
        html.Div([
            chart_card("Yield over time", "Monitor changes in average yield rather than total sample volume", charts.yield_by_year(h)),
            narrative("What stands out", [yield_sentence, erosion_sentence,
                "Use this dashboard to locate patterns for follow-up investigation, not to infer a causal treatment effect from site-level differences."]),
        ], className="two-column"),
    ]


def yield_content(d):
    return [
        html.Div([
            chart_card("Crop and treatment comparison", "Mean yield by observed crop–treatment group", charts.yield_by_crop(d.harvest)),
            chart_card("Yield trajectory", "Annual means using the same grain-yield unit", charts.yield_by_year(d.harvest)),
        ], className="two-column"),
        html.Div([
            chart_card("Biomass vs. grain yield", "Each bubble represents one crop, year, and treatment; size reflects sample count",
                       charts.biomass_relationship(d.harvest)),
            narrative("How to interpret", [
                "Grain yield is measured in kg/ha; aboveground biomass is measured in g/m². They belong on separate axes.",
                "These comparisons are observational. Year and crop composition can affect treatment averages; filter them before interpreting a difference.",
            ]),
        ], className="two-column"),
    ]


def erosion_content(d):
    return [
        html.Div([
            chart_card("Mean flux by site", "The source includes only the observed site–treatment combinations", charts.flux_by_site(d.erosion)),
            chart_card("Annual erosion trend", "Mean mass flux per observed record, not a cumulative total", charts.flux_by_year(d.erosion)),
        ], className="two-column"),
        html.Div([
            chart_card("Top sampling hotspots", "Ten site–location pairs with the highest observed mean flux", charts.flux_hotspots(d.erosion)),
            chart_card("Monthly erosion profile", "Monthly mean based on dated erosion records", charts.flux_seasonality(d.erosion)),
        ], className="two-column"),
        narrative("Important comparison limitation", [
            "In these source records, Mandan appears with BAU and Morton appears with ASP. Site and treatment are therefore confounded.",
            "Differences between those bars may reflect location, treatment, weather, or other conditions. Do not label them as a standalone treatment effect.",
        ], "warning", eyebrow="DATA QUALITY & INTERPRETATION"),
    ]


def weather_content(d):
    return [
        html.Div([
            chart_card("Seasonal wind profile", "Mean daily wind speed by month in NP field weather records",
                       charts.weather_monthly(d.weather, "Wind_spd_m_s", "Wind speed (m/s)", charts.BLUE)),
            chart_card("Seasonal precipitation", "Mean daily rainfall by month, not the sum across duplicated field rows",
                       charts.weather_monthly(d.weather, "PCPN_mm_d", "Precipitation (mm/day)", charts.TEAL)),
        ], className="two-column"),
        html.Div([
            chart_card("Yearly regional wind", "Mean wind speed by calendar year from NP field observations",
                       charts.weather_annual(d.weather)),
            chart_card("Weather vs. erosion: exploratory", "Annual NP wind averages plotted against site-level annual erosion means",
                       charts.weather_flux_context(d.weather, d.erosion),
                       "Context only: weather and erosion records do not share the same site identifier; matching by year is not a site-matched test."),
        ], className="two-column"),
        narrative("Coverage and comparability", [
            "Weather observations cover 2016–2021, whereas the erosion dataset covers 2019–2022. Weather data for 2022 is not supplied.",
            "The NP weather station is not a direct Mandan/Morton site match. A year-to-year association alone cannot establish that weather caused changes in erosion.",
        ], "warning", eyebrow="SOURCE COVERAGE"),
    ]


def trends_content(d):
    return [
        html.Div([
            chart_card("Annual yield trend", "Treatment-group grain yield, kg/ha", charts.yield_by_year(d.harvest)),
            chart_card("Annual erosion trend", "Site-group mass flux, source units", charts.flux_by_year(d.erosion)),
        ], className="two-column"),
        html.Div([
            chart_card("Crop–year yield heatmap", "Within-cell color indicates mean grain yield in kg/ha", charts.crop_year_heatmap(d.harvest)),
            chart_card("Seasonal erosion", "See when mean recorded flux tends to peak", charts.flux_seasonality(d.erosion)),
        ], className="two-column"),
        narrative("A better decision question", [
            "Which crop–year combinations deserve investigation because yield falls while erosion rises? Compare the two trends without putting unlike units on one axis.",
            "Sample sizes and measurement coverage vary across years; observed trends should be reviewed before making recommendations.",
        ]),
    ]


def treatment_content(d):
    return [
        html.Div([
            chart_card("Observed crop outcomes", "Side-by-side mean harvest yield for each treatment", charts.yield_by_crop(d.harvest)),
            chart_card("ASP compared with BAU", "Percentage difference in crop-level mean yield; descriptive, not causal",
                       charts.treatment_gap(d.harvest)),
        ], className="two-column"),
        html.Div([
            chart_card("Cover-crop biomass", "Average ASP observations by year (no BAU cover-crop data supplied)",
                       charts.cover_crop_history(d.cover_crop)),
            narrative("What the evidence supports", [
                "The harvest file permits descriptive yield comparisons for BAU and ASP within crop and year.",
                "The separate cover-crop biomass file contains only ASP treatment measurements. It cannot support a BAU-versus-ASP cover-crop chart.",
                "A practical next step is to test whether yield differences persist within comparable crops, years, and experimental blocks.",
            ], "warning", eyebrow="EVIDENCE CHECK"),
        ], className="two-column"),
    ]


app.layout = html.Div([
    html.A("Skip to dashboard", href="#main-content", className="skip-link"),
    html.Aside([
        html.Div([
            html.Div("A", className="brand-symbol"),
            html.Div([html.Div("AgriSight", className="brand-name"),
                      html.Div("FIELD INTELLIGENCE", className="brand-caption")]),
        ], className="brand"),
        html.Div("WORKSPACE", className="sidebar-caption"),
        dcc.Tabs(id="section-tabs", value="overview", vertical=True,
                 parent_className="nav-tabs", className="nav-tabs",
                 children=[dcc.Tab(label=label, value=key,
                         className="nav-tab", selected_className="nav-tab nav-tab-active")
                           for key, label in [
                               ("overview", "Overview"), ("yield", "Yield & biomass"),
                               ("erosion", "Wind erosion"), ("weather", "Weather context"),
                               ("trends", "Time trends"), ("treatment", "Treatment performance")
                           ]]),
        html.Div("FILTER THE ANALYSIS", className="sidebar-caption filter-heading"),
        html.Div([
            html.Label("Year", htmlFor="filter-year"),
            dcc.Dropdown(id="filter-year", options=[{"label": "All years", "value": "all"}] +
                         [{"label": str(y), "value": str(y)} for y in sorted(set(DATA.harvest["Year"].dropna().tolist()) | set(DATA.weather["Year"].dropna().tolist()))],
                         value="all", clearable=False),
            html.Label("Treatment", htmlFor="filter-treatment"),
            dcc.Dropdown(id="filter-treatment", options=[{"label": "All treatments", "value": "all"},
                         {"label": "BAU", "value": "BAU"}, {"label": "ASP", "value": "ASP"}],
                         value="all", clearable=False),
            html.Label("Crop", htmlFor="filter-crop"),
            dcc.Dropdown(id="filter-crop", options=[{"label": "All crops", "value": "all"}] +
                         [{"label": crop, "value": crop} for crop in charts.CROP_ORDER],
                         value="all", clearable=False),
            html.Label("Erosion site", htmlFor="filter-site"),
            dcc.Dropdown(id="filter-site", options=[{"label": "All sites", "value": "all"},
                         {"label": "Mandan", "value": "Mandan"}, {"label": "Morton", "value": "Morton"}],
                         value="all", clearable=False),
            html.Button("Reset filters", id="reset-filters", n_clicks=0, className="reset-button"),
            html.P("Crop filters harvest data; site filters erosion data. Weather refers to NP records.",
                   className="filter-help"),
        ], className="filter-panel"),
        html.Div([html.Div("DATA COVERAGE", className="sidebar-caption"),
                  html.Div("Yield & erosion: 2019–2022", className="sidebar-footer-text"),
                  html.Div("NP weather: 2016–2021", className="sidebar-footer-text"),
                  html.Div("Cover crop: 2020–2022", className="sidebar-footer-text")],
                 className="sidebar-footer"),
    ], className="sidebar"),
    html.Main([
        html.Div([
            html.Div([
                html.Div([html.Span("ANALYTICS", className="topbar-tag"),
                          html.Span("/", className="topbar-separator"),
                          html.Span("CONSERVATION PRACTICES", className="topbar-section")],
                         className="breadcrumbs"),
                html.H1(id="page-title", className="page-title"),
                html.P(id="page-subtitle", className="page-subtitle"),
            ]),
            html.Button([html.Span("↓", className="export-icon"), " Export filtered CSV"],
                        id="export-button", className="export-button", n_clicks=0),
        ], className="page-header"),
        dcc.Loading(html.Div(id="page-content", className="page-content"),
                    type="circle", color=charts.BLUE),
        dcc.Download(id="download-data"),
        html.Footer("AgriSight · Built for agricultural observations · Descriptive analytics, not causal inference",
                    className="footer"),
    ], className="main", id="main-content"),
], className="app-shell")


@app.callback(
    Output("page-title", "children"), Output("page-subtitle", "children"),
    Output("page-content", "children"),
    Input("section-tabs", "value"), Input("filter-year", "value"),
    Input("filter-treatment", "value"), Input("filter-crop", "value"),
    Input("filter-site", "value"),
)
def render_page(section, year, treatment, crop, site):
    selected = apply_filters(DATA, year, treatment, crop, site)
    title, subtitle = PAGES.get(section, PAGES["overview"])
    content = {
        "overview": overview_content, "yield": yield_content,
        "erosion": erosion_content, "weather": weather_content,
        "trends": trends_content, "treatment": treatment_content,
    }.get(section, overview_content)(selected)
    return title, subtitle, [summarize_kpis(selected), *content]


@app.callback(
    Output("filter-year", "value"), Output("filter-treatment", "value"),
    Output("filter-crop", "value"), Output("filter-site", "value"),
    Input("reset-filters", "n_clicks"), prevent_initial_call=True,
)
def reset_filters(_):
    return "all", "all", "all", "all"


@app.callback(
    Output("download-data", "data"), Input("export-button", "n_clicks"),
    State("section-tabs", "value"), State("filter-year", "value"),
    State("filter-treatment", "value"), State("filter-crop", "value"),
    State("filter-site", "value"), prevent_initial_call=True,
)
def export_csv(_clicks, section, year, treatment, crop, site):
    selected = apply_filters(DATA, year, treatment, crop, site)
    frame = records_for_export(selected, section)
    return dcc.send_data_frame(frame.to_csv, f"agrisight_{section}_filtered.csv", index=False)


if __name__ == "__main__":
    app.run(debug=True)

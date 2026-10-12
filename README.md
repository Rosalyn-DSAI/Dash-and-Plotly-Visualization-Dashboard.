# AgriSight | Agricultural Performance & Wind Erosion Analytics

**AgriSight** is an interactive analytics dashboard built with **Python, pandas, Dash, and Plotly**. It brings historical crop yield, plant biomass, wind erosion, and weather observations into one place to make field performance easier to explore and understand.

The goal is not simply to display charts. AgriSight helps turn separate agricultural records into understandable findings, while making clear where additional evidence is needed before drawing conclusions.

## The problem and the approach

Agricultural measurements were stored in different datasets, covering different years, field locations, and management practices. Looking at each file separately made it difficult to compare crop performance, recognize high-erosion areas, understand seasonal patterns, or see when data was missing.

AgriSight combines these sources in a single dashboard with **six analysis views**, consistent charts, interactive filters, and explanations of what the results mean. Instead of assuming every dataset can be merged directly, the dashboard keeps different measurement units and locations distinct.

This analysis is designed to help answer practical questions:

- Which crops have the highest and lowest **observed average grain yields**?
- In which years or sampled locations are **wind erosion measurements** highest?
- How do **yield, erosion, and weather patterns** change over time?
- How do observed outcomes differ between **BAU** and **ASP** management groups?
- Which findings deserve further investigation, and where is the evidence too limited to support a conclusion?

## Dashboard walkthrough and findings

The screenshots below show the dashboard with its full-data selections, unless otherwise indicated. Each view uses the same navigation and filter controls, making it easy to move from a broad summary to a more focused analysis.

### 1. Performance overview

![AgriSight overview with KPI cards, crop yield comparisons and annual wind erosion trends](images/Tab%201.png)

The overview brings together four headline measures: average grain yield, average recorded erosion mass flux, the number of harvest observations, and average ASP cover-crop biomass. Underneath, crop-level bars and annual erosion lines make the differences visible without requiring users to inspect the raw files.

**What the results show:** Across the displayed harvest records, **corn has the highest average grain yield** among the three crops shown. Wind erosion also changes from year to year, and the two sampled sites do not follow identical patterns. These summaries help identify which crops, years, and sites deserve closer examination.

### 2. Yield and biomass

![AgriSight yield and biomass charts showing management-group yields, year-to-year changes and a biomass scatter plot](images/Tab%202.png)

This view compares grain yields by crop and management group, tracks yield across years, and explores how aboveground biomass relates to grain yield. Bubble sizes represent the number of observations behind a summarized point.

**What the results show:** In the observed records, **BAU has a higher average corn and spring-wheat yield**, while **ASP has a higher average soybean yield**. Average yields also fall sharply in **2021** before increasing in **2022**. These are observed differences, not proof that the management practice alone produced them. They suggest useful follow-up questions about crop, year, and field conditions.

### 3. Wind erosion

![AgriSight wind erosion dashboard with highest-flux sampling locations and monthly erosion trends](images/Tab%203.png)

The wind erosion section identifies sampled locations with high recorded **mass flux**, a measure of material moving in wind erosion observations. A separate chart compares average erosion readings across the months for Mandan and Morton.

**What the results show:** Some sampling locations record much higher average flux than others. In the displayed monthly trends, **June has the highest mean readings**, followed by generally lower values later in the season. These patterns help prioritize locations and periods for closer field investigation. Because site and treatment are linked in the supplied records, the site differences cannot be interpreted as independent treatment effects.

### 4. Weather context

![AgriSight weather context with annual wind speed and an exploratory weather-versus-erosion comparison](images/Tab%204.png)

This section examines monthly wind speed, rainfall patterns, annual average wind speed, and an exploratory comparison between regional wind conditions and erosion observations.

**What the results show:** Rainfall is more concentrated in the warmer months, while average wind speed changes across months and years. The regional weather data provides context for investigating erosion patterns, but **the weather station is not confirmed to match the erosion sampling sites**. The scatter plot is therefore a starting point for further analysis, not evidence that the recorded wind speed caused an erosion result.

### 5. Time trends

![AgriSight time trends with a crop-year yield heatmap and seasonal wind erosion trends](images/Tab%205.png)

The time trends view tracks changes in grain yield and erosion separately. Its crop-year heatmap highlights high and low yield averages, while leaving cells blank where no relevant observations are available. A seasonal erosion chart shows how average readings differ by month and site.

**What the results show:** Yield varies substantially by year, with a visible drop in 2021 in the annual trends. The heatmap also shows that **not every crop has measurements for every year**. Showing these gaps matters: a missing observation should not be mistaken for zero yield or filled in without evidence. The view helps identify unusual periods and gaps worth investigating.

### 6. Treatment performance

![AgriSight treatment performance with yield comparisons, percentage differences and ASP cover-crop biomass](images/Tab%206.png)

This view compares observed BAU and ASP grain yields within each crop, displays the percentage difference in those averages, and summarizes **ASP cover-crop biomass** by year. The accompanying evidence notes explain which comparisons the data can and cannot support.

**What the results show:** ASP's observed soybean yield average is higher than BAU's, while its corn and spring-wheat averages are lower. The available ASP cover-crop biomass averages also vary substantially between **2020 and 2022**. Since the cover-crop source contains **no BAU observations**, it would be misleading to present those bars as proof of an ASP-versus-BAU cover-crop benefit. Differences in crop, year, experimental block, and sample coverage need further investigation.

## Interactive features

- **Six analysis sections:** Overview, Yield & Biomass, Wind Erosion, Weather Context, Time Trends, and Treatment Performance.
- **Context-aware filters:** Select year, treatment, crop, or erosion site; each filter operates only on the datasets to which it applies.
- **Persistent navigation:** The sidebar remains accessible while longer chart pages scroll, and can scroll independently on smaller screens.
- **Interactive Plotly charts:** Hover for measured values, units, and sample information; zoom or download a chart image from the toolbar.
- **Export filtered data:** Download CSV records for the selected analysis section and active filters.
- **Clear chart explanations:** Short reading guides explain the values, trends, and important interpretation limits.

## Approach and tools

**Data preparation:** pandas is used to load and clean source records, handle dates and numeric columns, and apply filters without mixing incompatible datasets.

**Analysis and visualization:** Plotly creates interactive bar charts, line charts, scatter plots, and heatmaps. Dash connects the charts, KPI cards, filters, and CSV export into one application. Custom CSS provides a consistent visual design and readable chart layout.

**Analytical judgment:** The dashboard prioritizes observed averages, makes missing coverage visible, keeps unlike units separate, and communicates limitations alongside the results. It is designed as a **descriptive decision-support tool**, not a system for proving cause and effect.

**Technology:** Python · pandas · Dash · Plotly · HTML/CSS · Jupyter Notebook · Git/GitHub

## Key findings and next analytical steps

The dashboard surfaces several opportunities for focused analysis:

| Finding from the supplied records | Useful next step |
| --- | --- |
| Corn has the highest observed average grain yield among the displayed crops. | Examine whether that pattern holds across comparable years and plots. |
| Yield drops in 2021 and recovers in 2022 in the displayed annual averages. | Review sample composition, crop mix, and available field conditions before attributing the change to any one factor. |
| Erosion readings vary substantially across sampling locations and are generally higher at the beginning of the observed June–October period. | Prioritize higher-flux locations and months for further investigation or monitoring. |
| BAU and ASP yield averages differ by crop. | Compare matched crops, years, and experimental blocks before making management recommendations. |
| Some crop-year combinations and treatment comparisons have missing data. | Identify additional measurements needed for a more reliable comparison. |

These are **investigation priorities**, not recommendations to change farming practices without further evidence.

## Data quality and interpretation limits

The source datasets do not all measure the same location or period. To avoid presenting unsupported conclusions, AgriSight highlights the following limitations:

1. **Wind erosion site and treatment are linked:** Mandan appears with **BAU** and Morton with **ASP** in the supplied erosion records. The data cannot separate site effects from treatment effects.
2. **Cover-crop comparison is incomplete:** The cover-crop biomass file includes **ASP observations only**. It cannot support a BAU-versus-ASP cover-crop comparison.
3. **Weather coverage is different:** NP weather covers **2016–2021**, while harvest and erosion records cover **2019–2022**. Weather data for 2022 is not supplied, and the station is not confirmed to be the same location as Mandan or Morton.
4. **Location IDs are not interchangeable:** Harvest plots, erosion sites, and weather-field identifiers represent different sources and are not treated as identical locations.
5. **Group sizes can differ:** Some averages are based on more records than others. Hover information and visible data gaps help put comparisons in context.
6. **Erosion measurement units require confirmation:** The original mass-flux column is named `total.flux` without a physical unit in its header. The dashboard labels these measurements as **source flux units** rather than assigning an unverified unit.

## Repository structure

The Dash application and original exploratory notebooks are both kept in the repository. The notebooks document earlier analysis work; the dashboard provides the interactive presentation of the cleaned and summarized data.

| Path | Purpose |
| --- | --- |
| `app.py` | Dashboard layout, navigation, callbacks, interactive filters, and exports |
| `dashboard_data.py` | Loading, cleaning, filtering, and selecting source observations |
| `dashboard_charts.py` | Plotly charts used in the six dashboard sections |
| `assets/styles.css` | Dashboard layout, responsive styling, and fixed sidebar behavior |
| `images/Tab 1.png` – `images/Tab 6.png` | Screenshots of the six dashboard sections shown above |
| `Theme1.ipynb` – `Theme5.ipynb`, `Plotly_Charts.ipynb` | Earlier exploratory and visualization notebooks |
| Source `*.csv` and `*.xlsx` files | Original agricultural datasets |
| `requirements.txt` | Python dependencies |
| `tests/test_dashboard.py` | Automated checks for data, filters, chart creation, and evidence constraints |
| `.gitignore` | Local files excluded from version control |

The source datasets are read **relative to the project directory** and are not overwritten by the application. The historical wind-speed Excel workbook is retained for earlier notebook work but is not required to load the dashboard.

## Run locally

Requires **Python 3.10 or later**.

1. Download or clone the repository. Open a terminal in the folder containing `app.py`.
2. Create a virtual environment (**activation is not required** for the commands below).

   **Windows PowerShell:**
   ```powershell
   py -m venv .venv
   ```
3. Install dependencies:
   ```powershell
   .\.venv\Scripts\python.exe -m pip install -r requirements.txt
   ```
4. Start the dashboard:
   ```powershell
   .\.venv\Scripts\python.exe app.py
   ```
5. Open **http://127.0.0.1:8050/** in your browser.

6. Run regression checks (optional):
   ```powershell
   .\.venv\Scripts\python.exe -m unittest discover -s tests -v
   ```

**macOS/Linux** (from the project folder):
```bash
python3 -m venv .venv
./.venv/bin/python -m pip install -r requirements.txt
./.venv/bin/python app.py
```

**Deployment note:** The application uses `debug=True` for local development; disable debugging if deploying it to a public server.

## Source datasets

| Dataset | Period | Measures |
| --- | --- | --- |
| `h5_i2_2019-2022_hand_harvest.csv` | 2019–2022 | Grain yield, aboveground biomass, crop, treatment, plot |
| `Mass Flux values 2019_2022 updated.csv` | 2019–2022 | Mass flux, sampling location, year, site, treatment |
| `h5-i2_2016-2021_daily-weather.csv` | 2016–2021 | Wind speed, precipitation, daily temperature at NP |
| `I2_CC_AGB_2020-2022.csv` | 2020–2022 | ASP cover-crop biomass measurements |

The project uses historical observations for exploratory field analysis; it does not provide real-time environmental monitoring.

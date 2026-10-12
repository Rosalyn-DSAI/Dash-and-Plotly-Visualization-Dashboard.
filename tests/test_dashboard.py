"""Minimal regression checks for data quality and Plotly renderability.

Run: python -m unittest discover -s tests -v
"""
import unittest

from dashboard_data import apply_filters, load_datasets
from dashboard_charts import (
    biomass_relationship, cover_crop_history, crop_year_heatmap,
    flux_by_site, flux_by_year, flux_hotspots, treatment_gap,
    weather_flux_context, yield_by_crop,
    yield_by_year, flux_seasonality, weather_monthly, weather_annual,
)


class DashboardTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.raw = load_datasets()

    def test_expected_data_coverage(self):
        self.assertEqual(len(self.raw.harvest), 260)
        self.assertEqual(len(self.raw.erosion), 563)
        self.assertEqual(len(self.raw.weather), 4384)
        # One missing measurement and one blank row in the supplied 31-line file.
        self.assertEqual(len(self.raw.cover_crop), 29)
        self.assertEqual(sorted(self.raw.cover_crop["Year"].unique().tolist()), [2020, 2021, 2022])

    def test_erosion_site_and_treatment_are_confounded(self):
        combos = set(map(tuple, self.raw.erosion[["Site", "Treatment"]].drop_duplicates().to_numpy()))
        self.assertEqual(combos, {("Mandan", "BAU"), ("Morton", "ASP")})

    def test_filters_apply_to_correct_datasets(self):
        filtered = apply_filters(self.raw, year="2021", treatment="ASP", crop="Corn", site="Morton")
        self.assertTrue((filtered.harvest["Crop"] == "Corn").all())
        self.assertTrue((filtered.harvest["Treatment"] == "ASP").all())
        self.assertTrue((filtered.erosion["Site"] == "Morton").all())
        self.assertTrue((filtered.weather["Year"] == 2021).all())
        self.assertEqual(set(filtered.cover_crop["TRT"]), {"ASP"})
        # Crop choice must never wrongly drop flux/weather observations.
        other_crop = apply_filters(self.raw, year="2021", crop="Soybean", site="Morton")
        self.assertEqual(len(filtered.erosion), len(other_crop.erosion))

    def test_missing_2022_weather_is_handled(self):
        selected = apply_filters(self.raw, year="2022")
        self.assertTrue(selected.weather.empty)
        self.assertIsNotNone(weather_flux_context(selected.weather, selected.erosion).to_json())

    def test_chart_typography_and_plot_heights_are_readable(self):
        # Prevent the short, overlapping chart panels seen in the first redesign.
        chart_set = (
            yield_by_crop(self.raw.harvest),
            yield_by_year(self.raw.harvest),
            flux_by_site(self.raw.erosion),
            flux_by_year(self.raw.erosion),
            flux_hotspots(self.raw.erosion),
            flux_seasonality(self.raw.erosion),
            weather_monthly(self.raw.weather, "Wind_spd_m_s", "Wind speed (m/s)", "#2677E8"),
            weather_annual(self.raw.weather),
            crop_year_heatmap(self.raw.harvest),
            cover_crop_history(self.raw.cover_crop),
        )
        for fig in chart_set:
            self.assertGreaterEqual(fig.layout.height, 440)
            self.assertGreaterEqual(fig.layout.font.size, 14)
            self.assertGreaterEqual(fig.layout.margin.l, 75)
            self.assertGreaterEqual(fig.layout.margin.b, 70)
            self.assertGreaterEqual(fig.layout.xaxis.tickfont.size, 13)
            self.assertGreaterEqual(fig.layout.yaxis.tickfont.size, 13)

    def test_charts_serialize_for_full_and_empty_filters(self):
        for year in ("all", "2016", "2020", "2022"):
            selected = apply_filters(self.raw, year=year)
            charts = (
                yield_by_crop(selected.harvest), biomass_relationship(selected.harvest),
                flux_by_site(selected.erosion), flux_by_year(selected.erosion),
                flux_hotspots(selected.erosion), crop_year_heatmap(selected.harvest),
                treatment_gap(selected.harvest), cover_crop_history(selected.cover_crop),
            )
            for chart in charts:
                self.assertIn('"data"', chart.to_json())


if __name__ == "__main__":
    unittest.main()

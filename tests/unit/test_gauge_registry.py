import datetime
import unittest
from unittest import mock

import numpy

import naaulu.gauge


class TestListNetworks(unittest.TestCase):

    def test_returns_all_when_no_filter(self):
        nets = naaulu.gauge.list_networks()
        self.assertGreater(len(nets), 0)

    def test_each_entry_has_required_fields(self):
        required = {
            "organisation", "network", "temporal_resolution",
            "provider_module", "provider_function",
            "weather", "climat",
        }
        for n in naaulu.gauge.list_networks():
            with self.subTest(network=f"{n.get('organisation')}/{n.get('network')}"):
                self.assertTrue(required.issubset(n.keys()))

    def test_filter_by_organisation(self):
        nets = naaulu.gauge.list_networks(organisation="fmi")
        self.assertGreater(len(nets), 0)
        for n in nets:
            self.assertEqual(n["organisation"], "fmi")

    def test_filter_by_country(self):
        nets = naaulu.gauge.list_networks(country="FIN")
        self.assertGreater(len(nets), 0)
        for n in nets:
            c = n.get("country")
            self.assertTrue(c is None or c == "FIN")

    def test_filter_by_duration(self):
        nets = naaulu.gauge.list_networks(duration=datetime.timedelta(minutes=10))
        # FMI/aws PT10M should be in the registry
        self.assertGreater(len(nets), 0)
        for n in nets:
            self.assertEqual(n["temporal_resolution"], "PT10M")

    def test_filter_by_duration_no_match(self):
        nets = naaulu.gauge.list_networks(duration=datetime.timedelta(seconds=37))
        self.assertEqual(nets, [])

    def test_combined_filters(self):
        nets = naaulu.gauge.list_networks(
            organisation="fmi", duration=datetime.timedelta(hours=1),
        )
        self.assertEqual(len(nets), 1)
        self.assertEqual(nets[0]["organisation"], "fmi")
        self.assertEqual(nets[0]["temporal_resolution"], "PT1H")


class TestPath(unittest.TestCase):

    def test_path_changes_with_provider_args(self):
        t = datetime.datetime(2024, 1, 15, 12, 0)
        d = datetime.timedelta(hours=1)
        base = naaulu.gauge.path(t, d, "iem", "asos")
        with_ia = naaulu.gauge.path(t, d, "iem", "asos", {"network": "IA_ASOS"})
        with_wi = naaulu.gauge.path(t, d, "iem", "asos", {"network": "WI_ASOS"})
        self.assertNotEqual(base, with_ia)
        self.assertNotEqual(with_ia, with_wi)

    def test_path_stable_with_kwarg_order(self):
        t = datetime.datetime(2024, 1, 15, 12, 0)
        d = datetime.timedelta(hours=1)
        a = naaulu.gauge.path(t, d, "x", "y", {"a": 1, "b": 2})
        b = naaulu.gauge.path(t, d, "x", "y", {"b": 2, "a": 1})
        self.assertEqual(a, b)


class TestGetForwardsProviderArgs(unittest.TestCase):

    def test_provider_args_are_forwarded(self):
        captured = {}

        def fake_provider(time, duration, **kwargs):
            captured.update(kwargs)
            return ([0.0], ["S1"], numpy.array([[0.0, 0.0]]))

        import types
        fake_module = types.ModuleType("naaulu.providers.fakeorg")
        fake_module.fakenet = fake_provider

        real_import = naaulu.gauge.importlib.import_module

        def fake_import(name, *args, **kwargs):
            if name == "naaulu.providers.fakeorg":
                return fake_module
            return real_import(name, *args, **kwargs)

        with mock.patch.object(naaulu.gauge.importlib, "import_module", side_effect=fake_import), \
             mock.patch("os.path.exists", return_value=False), \
             mock.patch("xarray.Dataset.to_netcdf"), \
             mock.patch("os.makedirs"):
            naaulu.gauge.get(
                time=datetime.datetime(2024, 1, 15, 12, 0),
                duration=datetime.timedelta(hours=1),
                organisation="fakeorg",
                network="fakenet",
                provider_args={"network": "WI_ASOS", "extra": "foo"},
            )

        self.assertEqual(captured, {"network": "WI_ASOS", "extra": "foo"})


if __name__ == "__main__":
    unittest.main()

"""Secret lookup and the parse_area window source.

Secrets: environment variable, then keyring, then the secrets file.
Windows: --adm 1 clips and draws with the same Natural Earth geometry
(what you see is the mask edge); every other detail, and every CLI whose
args carry no --adm at all (estimate, combine, ...), keeps the OSM window.
"""

import argparse

import keyring.errors
import pytest
import shapely.geometry

import naaulu.config
from naaulu import geography


# ─── secret lookup ──────────────────────────────────────────────────────────

KEY = "KNMI_API_KEY"


@pytest.fixture
def secrets(tmp_path, monkeypatch):
    """An empty secrets file and a clean environment and keyring."""
    monkeypatch.delenv(KEY, raising=False)
    monkeypatch.setattr(naaulu.config, "_secret_cache", {})
    path = tmp_path / ".naaulu_secrets"
    monkeypatch.setattr(naaulu.config, "SECRETS_FILE", path)
    return path


def test_environment_variable_wins(secrets, monkeypatch):
    """The env var is enough: neither keyring nor the file is consulted."""
    monkeypatch.setenv(KEY, "from-the-environment")

    def explode(*args, **kwargs):
        raise AssertionError("the keyring should not be reached")

    monkeypatch.setattr(naaulu.config.keyring, "get_password", explode)

    assert naaulu.config.get_secret(KEY) == "from-the-environment"


def test_keyring_is_used_when_the_environment_is_empty(secrets, monkeypatch):
    monkeypatch.setattr(naaulu.config.keyring, "get_password",
                        lambda service, name: "from-the-keyring")

    assert naaulu.config.get_secret(KEY) == "from-the-keyring"


def test_a_broken_keyring_falls_back_to_the_file(secrets, monkeypatch):
    """A host without a keyring backend must not raise: the file still works."""
    secrets.write_text(f"{KEY}=from-the-file\n")

    def unavailable(service, name):
        raise keyring.errors.NoKeyringError("no backend")

    monkeypatch.setattr(naaulu.config.keyring, "get_password", unavailable)

    assert naaulu.config.get_secret(KEY) == "from-the-file"


def test_the_file_is_used_when_the_keyring_has_nothing(secrets, monkeypatch):
    secrets.write_text(f"{KEY}=from-the-file\n")
    monkeypatch.setattr(naaulu.config.keyring, "get_password",
                        lambda service, name: None)

    assert naaulu.config.get_secret(KEY) == "from-the-file"


def test_no_secret_anywhere_is_none(secrets, monkeypatch):
    monkeypatch.setattr(naaulu.config.keyring, "get_password",
                        lambda service, name: None)

    assert naaulu.config.get_secret("NOT_A_SECRET_AT_ALL") is None


def test_the_file_is_read_as_key_value_lines(secrets, monkeypatch):
    secrets.write_text("# a comment\nSPW_API_KEY = spw-value\n")
    monkeypatch.setattr(naaulu.config.keyring, "get_password",
                        lambda service, name: None)

    assert naaulu.config.get_secret("SPW_API_KEY") == "spw-value"


def test_a_secret_stored_in_the_keyring_is_not_in_the_file(secrets, monkeypatch):
    """`naaulu set` writes to the keyring only, and that still comes back."""
    monkeypatch.setattr(naaulu.config.keyring, "set_password",
                        lambda service, name, value: None)
    monkeypatch.setattr(naaulu.config.keyring, "get_password",
                        lambda service, name: "stored-on-the-desktop")

    assert naaulu.config.get_secret(KEY) == "stored-on-the-desktop"


# ─── window source per --adm detail ─────────────────────────────────────────

def _args(**kw):
    base = dict(name=None, window=None, country=None, geofile=None,
                buffer=None, chunk=2, no_mainland=False, adm=2)
    base.update(kw)
    return argparse.Namespace(**base)


def test_adm1_country_window_is_natural_earth(monkeypatch):
    """--adm 1 clips with the very geometry it draws; the OSM path must
    not run at all."""
    def boom(*args, **kwargs):
        raise AssertionError("adm 1 windows must not go through OSM")
    monkeypatch.setattr(geography, "import_country", boom)

    area, _, geometry = naaulu.config.parse_area(
        _args(country="bel", adm=1))

    assert area == "bel"
    assert geometry.equals(geography.import_country_ne10m("BEL")), \
        "the clip geometry is the NE10m outline"


def test_other_details_and_other_clis_keep_the_osm_window(monkeypatch):
    """--adm >= 2, and any CLI whose args carry no --adm at all, use the
    OSM country outline — Natural Earth must not be consulted."""
    called = {}

    def fake_osm(iso_code, mainland_only=True, **kw):
        called["osm"] = iso_code
        return shapely.geometry.box(2, 50, 6, 52)

    def boom_ne(*args, **kwargs):
        raise AssertionError("the OSM window must not use Natural Earth")
    monkeypatch.setattr(geography, "import_country", fake_osm)
    monkeypatch.setattr(geography, "import_country_ne10m", boom_ne)

    _, _, geometry = naaulu.config.parse_area(_args(country="bel", adm=2))
    assert called == {"osm": "BEL"}
    assert geometry.bounds == (2, 50, 6, 52)

    called.clear()
    # simulate estimate/combine: same namespace, no --adm attribute
    args = argparse.Namespace(
        name=None, window=None, country="bel", geofile=None,
        buffer=None, chunk=2, no_mainland=False)
    naaulu.config.parse_area(args)
    assert called == {"osm": "BEL"}, "CLIs without --adm keep the OSM window"


def test_buffer_wraps_the_swapped_window(monkeypatch):
    """--buffer applies after the source swap: the buffered NE window is
    what gets clipped and drawn at --adm 1."""
    monkeypatch.setattr(geography, "import_country",
                        lambda *a, **k: (_ for _ in ()).throw(
                            AssertionError("adm 1 must not use OSM")))

    _, _, geometry = naaulu.config.parse_area(
        _args(country="bel", adm=1, buffer="10km"))

    plain = geography.import_country_ne10m("BEL")
    assert geometry.bounds[0] < plain.bounds[0], \
        "the buffer must widen the NE window, not be lost in the swap"

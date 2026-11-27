"""--no-network / NAAULU_NO_NETWORK: cache-only operation.

The flag (each CLI main calls config.set_no_network) and the env var both
flip config.no_network(); the single gate in network.runtime_session() then
refuses every HTTP call while cache hits keep working. OfflineError is a
plain Exception on purpose: the retry/backoff loops (Postpass, RangeReader,
the OPERA download retries) catch request errors and would sleep through an
offline miss instead of failing fast.
"""

import argparse

import pytest
import requests

import naaulu.config
import naaulu.errors
import naaulu.geography
import naaulu.network
import naaulu.radar


@pytest.fixture
def offline(monkeypatch):
    """no-network mode on via the flag, with a clean environment."""
    monkeypatch.delenv("NAAULU_NO_NETWORK", raising=False)
    monkeypatch.setattr(naaulu.config, "_NO_NETWORK", True)


@pytest.fixture
def no_http(monkeypatch):
    """Any attempted HTTP connection fails the test immediately."""
    def explode(*args, **kwargs):
        raise AssertionError("HTTP attempted while --no-network is active")
    monkeypatch.setattr(requests.sessions.Session, "request", explode)


# ─── flag and env ───────────────────────────────────────────────────────────

def test_off_by_default(monkeypatch):
    monkeypatch.delenv("NAAULU_NO_NETWORK", raising=False)
    monkeypatch.setattr(naaulu.config, "_NO_NETWORK", False)
    assert not naaulu.config.no_network()


def test_enabled_by_env(monkeypatch):
    monkeypatch.setattr(naaulu.config, "_NO_NETWORK", False)
    monkeypatch.setenv("NAAULU_NO_NETWORK", "1")
    assert naaulu.config.no_network()


def test_env_off_values(monkeypatch):
    monkeypatch.setattr(naaulu.config, "_NO_NETWORK", False)
    for value in ("", "0", "false", "no"):
        monkeypatch.setenv("NAAULU_NO_NETWORK", value)
        assert not naaulu.config.no_network(), value


def test_system_args_offer_the_flag():
    parser = argparse.ArgumentParser()
    naaulu.config.add_system_args(parser)
    assert parser.parse_args([]).no_network is False
    assert parser.parse_args(["--no-network"]).no_network is True


def test_set_no_network_flips_the_helper(monkeypatch):
    monkeypatch.delenv("NAAULU_NO_NETWORK", raising=False)
    monkeypatch.setattr(naaulu.config, "_NO_NETWORK", False)
    naaulu.config.set_no_network(value=True)
    try:
        assert naaulu.config.no_network()
    finally:
        naaulu.config.set_no_network(value=False)


# ─── the single gate ────────────────────────────────────────────────────────

def test_runtime_session_refuses(offline, no_http):
    with pytest.raises(naaulu.errors.OfflineError):
        naaulu.network.runtime_session()


def test_offline_error_is_plain_exception():
    """Not a RequestException: retry loops must not catch-and-sleep it."""
    assert issubclass(naaulu.errors.OfflineError, Exception)
    assert not issubclass(naaulu.errors.OfflineError, requests.RequestException)


# ─── cache-only behaviour ───────────────────────────────────────────────────

def test_fetch_serves_cache_without_http(tmp_path, monkeypatch, offline, no_http):
    monkeypatch.setattr(naaulu.config, "get_download_dir", lambda *a: str(tmp_path))
    (tmp_path / "cached.bin").write_bytes(b"payload")
    content = naaulu.network.fetch("http://example.invalid/cached.bin",
                                   filename="cached.bin")
    assert content == b"payload"


def test_fetch_miss_raises_offline(tmp_path, monkeypatch, offline, no_http):
    monkeypatch.setattr(naaulu.config, "get_download_dir", lambda *a: str(tmp_path))
    with pytest.raises(naaulu.errors.OfflineError):
        naaulu.network.fetch("http://example.invalid/missing.bin",
                             filename="missing.bin")


def test_download_serves_cache_without_http(tmp_path, monkeypatch, offline, no_http):
    monkeypatch.setattr(naaulu.config, "get_download_dir", lambda *a: str(tmp_path))
    (tmp_path / "radar.h5").write_bytes(b"volume")
    path = naaulu.network.download("http://example.invalid/radar.h5")
    assert open(path, "rb").read() == b"volume"


def test_download_miss_raises_offline(tmp_path, monkeypatch, offline, no_http):
    monkeypatch.setattr(naaulu.config, "get_download_dir", lambda *a: str(tmp_path))
    with pytest.raises(naaulu.errors.OfflineError):
        naaulu.network.download("http://example.invalid/absent.h5")


def test_s3_listing_raises_offline(offline, no_http):
    with pytest.raises(naaulu.errors.OfflineError):
        naaulu.radar._s3_list("2026/01/01/")


# ─── boundary fetches stay offline ──────────────────────────────────────────

def test_no_network_implies_boundaries_offline(monkeypatch):
    monkeypatch.delenv("NAAULU_BOUNDARIES_OFFLINE", raising=False)
    monkeypatch.setattr(naaulu.config, "_NO_NETWORK", True)
    assert naaulu.geography._boundaries_offline()

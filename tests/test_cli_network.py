"""Offline tests for the gauge network key the CLI accepts.

verify --network is both repeatable and space separated, which argparse
only supports as action="append" with nargs="+". With nargs="+" alone, a
second --network replaces the first, silently dropping networks from the
verification - so both groupings have to arrive, in order.
"""

import argparse

from naaulu import gauge
from naaulu.cli import verify


def parse(args):
    parser = argparse.ArgumentParser()
    verify.add_network_args(parser)
    return verify.flatten_networks(parser.parse_args(args).network)


def test_repeated_flags_keep_every_network():
    """--network bel --network bel_spw --network bel_vmm"""
    assert parse(["--network", "bel", "--network", "bel_spw",
                  "--network", "bel_vmm"]) == ["bel", "bel_spw", "bel_vmm"]


def test_space_separated_groups_still_work():
    """--network bel bel_spw --network deu"""
    assert parse(["--network", "bel", "bel_spw", "--network", "deu"]) == [
        "bel", "bel_spw", "deu",
    ]


def test_keys_are_lowercased_for_the_dispatch_table():
    assert parse(["--network", "BEL", "BEL_VMM"]) == ["bel", "bel_vmm"]
    assert set(gauge._GAUGE) >= {"bel", "bel_spw", "bel_vmm"}


def test_a_single_network_is_the_common_case():
    assert parse(["--network", "bel_spw"]) == ["bel_spw"]

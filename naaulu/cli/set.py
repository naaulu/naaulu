#!/usr/bin/env python

"""Set a secret value (e.g. API token) in the system keyring"""

import getpass
import sys

import naaulu.config


def main():
    if len(sys.argv) < 2 or sys.argv[1] in ("-h", "--help"):
        print("Usage: naaulu set <KEY>")
        print()
        print("Stores a secret in the system keyring.")
        print()
        print("Examples:")
        print("  naaulu set NOAA_API_TOKEN")
        print("  naaulu set SPW_API_KEY")
        sys.exit(0)

    key = sys.argv[1]
    value = getpass.getpass(f"Enter value for {key}: ")

    if not value:
        print("Error: empty value", file=sys.stderr)
        sys.exit(1)

    naaulu.config.set_secret(key, value)
    print(f"Stored {key} in keyring")

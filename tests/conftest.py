import datetime

# ---------------------------------------------------------------------------
# Radars confirmed with data in the OPERA S3 bucket (openradar-24h)
#   key = country alpha-2 code
#   value = (WIGOS ID, ODIM code / node name)
#
# Data is stored under PVOL/ or SCAN/ depending on the operator.  All
# entries below have been verified to return data at COVERED_TIME (≈1 h
# ago rounded to 5 min).
#
# Known caveats:
#   DE – S3 data is often several hours stale (~02:40 on most days).
#        Included for reference but time‑sensitive tests may be skipped.
# ---------------------------------------------------------------------------
RADARS = {
    "BE": ("0-20000-0-06482", "bewid"),
    "CH": ("0-20000-0-06661", "chalb"),
    "FI": ("0-20010-0-02954", "fianj"),
    "FR": ("0-20010-0-07005", "frabb"),
    "IS": ("0-352-6-iskef", "iskef"),
    "NL": ("0-20010-0-06234", "nldhl"),
    "PL": ("0-20000-0-12568", "plbrz"),
}

# DE has very stale data on openradar S3 (latest ~02:40 on most days).
# Kept separately so time‑sensitive tests can opt out.
RADARS_DE = { "DE": ("0-20010-0-10103", "deasb") }

# ---------------------------------------------------------------------------
# One hour ago, rounded to the nearest 5 minutes — always within the
# rolling S3 window for most countries (except DE).
# ---------------------------------------------------------------------------
_COVERED_RAW = datetime.datetime.now(datetime.timezone.utc) - datetime.timedelta(hours=1)
COVERED_TIME = datetime.datetime.fromtimestamp(
    round(_COVERED_RAW.timestamp() / 300) * 300, tz=datetime.timezone.utc
)
COVERED_DURATION = datetime.timedelta(minutes=30)

SHORT_DURATION = datetime.timedelta(minutes=5)

TOO_OLD_TIME = datetime.datetime(2020, 6, 1, 12, 0, tzinfo=datetime.timezone.utc)
TOO_OLD_DURATION = datetime.timedelta(hours=1)

UNKNOWN_WSI = "0-000-0-00000"

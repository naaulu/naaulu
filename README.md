# Naaulu

Naaulu is a Python toolbox for estimating, combining, ploting and verifying rainfall datasets.

> **Note:** This project is currently a **Work In Progress (WIP)** preview. Features may change and stability is not guaranteed.

## For users

General information is available on the [website](https://naaulu.org).

Please open issues on [GitHub](https://github.com/naaulu/naaulu/issues).

## For developers

### Collaboration

Development occurs on the [self-hosted forge](https://git.naaulu.org).

Please send an email to hello@naaulu.org.

### Install on Linux

```sh
cat install.sh
./install.sh
```

### Run

```sh
source .venv/bin/activate
naaulu estimate
naaulu combine
naaulu verify
naaulu plot
```

## For maintainers

To rebuild bundled reference data (requires internet access):

```sh
python build_data.py
```

## Data sources

naaulu is based on meteorological and geographical data that are public domain, CC BY 4.0 or ODbL: 

- **Radar volumes**: EUMETNET OPERA (BE, DE, CZ, NL, FR, FI/EE recent),
  FMI radar on AWS (FI archive), Keskkonnaportaal (EE archive), NOAA
  NEXRAD/NCEI (US), Bureau of Meteorology via NCI/AURA (AU).
- **Rain gauges**: KNMI (NL), RMI, SPW Wallonie and VMM (BE), CHMI (CZ),
  Keskkonnaagentuur (EE), FMI (FI), NOAA GHCN-hourly (historical), Iowa
  Environmental Mesonet (FR, DE, US).
- **Boundaries**: OpenStreetMap for countries, provinces and cities. Natural Earth for the `--adm 1` window and the
  bundled country outlines (`naaulu/data/ne_*_admin_0_countries.geojson`).
- **Database**: `radars_wmo.json` derives from the WMO Weather Radar Database
  (CC BY-SA 4.0, © World Meteorological Organization (WMO),
  https://wrd.mgm.gov.tr)


## License

This project is licensed under the GNU Affero General Public License v3.0 (AGPL-3.0). Works based on naaulu cite **© naaulu**.

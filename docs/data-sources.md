# Data sources and licences

naaulu itself is AGPL-3.0, but it is a *bridge*: it combines data published by
national meteorological services and boundary providers, each with its own
terms. This page lists them, with what naaulu uses them for and what you owe
them when you republish maps or figures.

Most European weather data are **high-value datasets** under [Commission
Implementing Regulation (EU) 2023/138](https://eur-lex.europa.eu/eli/reg_impl/2023/138/oj)
(the annex of [Directive (EU) 2019/1024](https://eur-lex.europa.eu/eli/dir/2019/1024)):
free of charge, via API, machine-readable, under an open licence — CC BY 4.0
or equivalent. Reuse, including commercial reuse and derivatives, is
therefore legally guaranteed; attribution is the obligation that remains.

## Radar volumes

| Source | Used for | Licence | Obligation |
|---|---|---|---|
| [EUMETNET OPERA](https://eumetnet.github.io/openradardata-documentation/1-ORD-API-overview/) | BE, DE, CZ, NL, FR, FI/EE recent | CC BY 4.0 (per-provider exceptions noted in the dataset metadata) | credit the provider; no implied endorsement |
| [FMI radar on AWS](https://registry.opendata.aws/fmi-radar) | FI archive | CC BY 4.0 | credit FMI |
| [Keskkonnaportaal / Estonian Environment Agency](https://keskkonnaportaal.ee/en/open-data) | EE archive | CC BY 4.0 | credit the Estonian Environment Agency |
| [NEXRAD via Unidata](https://registry.opendata.aws/noaa-nexrad/) and [NCEI HOMR](https://www.ncei.noaa.gov/access/homr/file/nexrad-stations.txt) | US | NOAA data: open, public domain | attribution requested; must not imply NOAA endorsement; modified data must not be presented as original |
| [BoM radar via NCI/AURA](https://opus.nci.org.au/spaces/NDP/pages/399803630/Operational+Radars+rq0) (`rq0`, Level 1 ODIM HDF5) | AU | CC BY 4.0 | "Bureau of Meteorology, © Commonwealth of Australia, licensed under CC BY 4.0". **Level 1 only**: Level 1b, Level 2 and Rainfields are CC BY-NC 4.0 and are not used |

## Rain gauges

| Source | Used for | Licence | Obligation |
|---|---|---|---|
| [KNMI Data Platform](https://developer.dataplatform.knmi.nl/open-data-api) | NL | CC BY 4.0 | "Source: KNMI" plus the dataset link; mark derived data as derived |
| [RMI Belgium](https://opendata.meteo.be/termsandconditions) | BE | CC BY 4.0 (automatic weather station WFS) | place the source immediately next to the information; mention RMI in references/imprint when adapted. RMI's gridded 5×5 km product is *not* reusable and is not used |
| [SPW Wallonie](https://hydrometrie.wallonie.be/mentions-legales.html) | BE_SPW | custom terms; intellectual property remains with SPW | link to hydrometrie.wallonie.be |
| [VMM / waterinfo.be](https://waterinfo.vlaanderen.be/default.aspx?path=NL/Algemene_Info/Disclaimer) | BE_VMM | the site disclaimer describes the data as "information and non-commercial purposes", while the catalogue records apply the Flemish Open Data licence v1.0 | to be resolved with VMM before commercial republishing |
| [CHMI](https://www.chmi.cz/o-chmu/produkty-a-sluzby/data-a-vyhodnoceni) and its station GeoPackage | CZ | CC BY 4.0 | "ČHMÚ, opendata.chmi.cz, CC BY 4.0" |
| [Keskkonnaagentuur / ilmateenistus.ee](https://www.ilmateenistus.ee/teenused/ilmainfo/eesti-vaatlusandmed-xml) | EE | open, with an explicit condition | cite Keskkonnaagentuur as the source **and** link to ilmateenistus.ee |
| [FMI open data](https://en.ilmatieteenlaitos.fi/open-data) | FI | CC BY 4.0 | credit FMI |
| [NOAA GHCN-hourly](https://www.ncei.noaa.gov/oa/global-historical-climatology-network/hourly/) | historical, any country | NOAA data: open, public domain | attribution requested |
| [Iowa Environmental Mesonet](https://mesonet.agron.iastate.edu/disclaimer.php) (ASOS) | FR, DE, US and others | public domain ("may be used freely by anyone for any lawful purpose") | attribution appreciated |

## Administrative boundaries

| Source | Used for | Licence |
|---|---|---|
| [OpenStreetMap](https://www.openstreetmap.org/copyright) (queried at prefetch through [Geofabrik Postpass](https://wiki.openstreetmap.org/wiki/Postpass), SQL over OSM's planet database) | everything at detail ≥ 2: the `--country` window that clips the data, countries (`--adm 2`), the province tier (`--adm 2`) and cities (`--adm 3`); at `--adm 1` both window and outlines come from Natural Earth below | **ODbL** — figures must credit "© OpenStreetMap contributors" |
| [Natural Earth](https://www.naturalearthdata.com/about/terms-of-use/) admin-0 (bundled 50m + 10m) | the `--adm 1` data window *and* country outlines (10m — one geometry, so the visible border is the mask edge), the window→country tile lookup (50m), and the attribution/scope fallbacks for fetched OSM data (units without ISO tags, Åland) | public domain, no credit required |

Each cached boundary file is written with a `.meta.json` recording its
source, licence and vintage, so the terms of any country can be checked with:

```sh
cat ~/.local/share/naaulu/boundaries/*.meta.json
```

Drawing a map from a database is a "produced work" under ODbL, so a rendered
figure carries the attribution obligation above; the cached OSM geometry
itself is a local derivative database — redistribute *that* file only under
the ODbL.

### Why every drawn border is OpenStreetMap

Earlier versions drew boundaries from geoBoundaries gbOpen, which is built
per country: France standardised from a Wikipedia outline (2016), Belgium
from Eurostat (2022), Germany from the federal BKG (2021). Each unit is fine
on its own, but two neighbours describe their *shared* border differently —
measured on the 2023 release: ADM0 France/Belgium overlapped by ~650 km² and
diverged by up to ~7 km along their border, ADM1 (régions/gewesten) still
overlapped by ~236 km². On a map the border is drawn twice and the two lines
visibly diverge. Natural Earth (10m) was tried as a fix — its borders do
match exactly, since it is one file — but its coastline detail is too coarse
for a regional radar map (Belgium: 648 vertices; estuaries and islands
smoothed away). The split today is deliberate and consistent per detail:
`--adm 1` draws *and clips* with the bundled Natural Earth 10m outlines —
one geometry for borders and data mask alike, nothing fetched — while
`--adm 2` and up stay OpenStreetMap end to end (window, borders, provinces),
where coastline detail matters and shared linework keeps every seam exact.

OpenStreetMap gets both properties at every tier: adjacent units reference
the *same* boundary ways in one database, so France/Belgium and
France/Germany overlap by exactly 0 km² and share one line, while the
coastline keeps full detail (the residual overlaps that remain — ~16 km² at
the Dollart, ~10 km² on the German-Belgian border — are places where OSM
itself double-maps the border, present at both tiers alike).

Two mechanics make this work, plus one rule that keeps it working:

- **Territorial sea**: OSM closes a boundary relation with `maritime=yes`
  coastal segments instead of the physical coastline. naaulu draws every
  relation as stored: the outline follows the ~12 nm territorial line rather
  than the physical shore (province outlines therefore stop short of the
  country outline at coasts), and no `natural=coastline` is ever fetched —
  nothing network-heavy left to fail.
- **Per-country admin_level table**: OSM has no ADM numbering — the province
  tier sits at a different level per country (Dutch provincie, Swiss cantons,
  Finnish maakunnat, German Länder, Croatian županije, Romanian județe,
  Moldovan raions, Polish województwa and Czech kraje at 4; Latvian novadi
  at 5;
  French départements, Estonian maakonnad and Luxemburgish cantons at 6;
  Danish, Norwegian and Swedish kommuner, Macedonian opštini and Andorran
  parroquies at 7; Åland's, Guernsey's and Jersey's parishes, Slovenian
  občine, Maltese local councils, Liechtenstein Gemeinden and Sammarinese
  castelli at 8; Monaco's quartiers at 10). naaulu draws exactly this one
  sub-national tier below the country (the GADM ADM2 notion); the
  state/region tier (ADM1: régions, Länder, gewesten) is not drawn. The
  mapping lives in `OSM_PROVINCE_LEVEL` in `naaulu/geography.py`, verified
  against OSM relation counts (and a prefetch warns when a country's mapped
  level yields no units); a wrong or missing level degrades to no lines
  for that country, never to an error.
- **No simplification**: cached geometry is stored exactly as served, at
  full OSM detail. Simplifying two neighbouring units independently is what
  opens hairline seams between them — only untouched shared linework stays
  bit-identical on both sides of a border — so nothing simplifies a unit
  polygon on its own, at fetch time or at draw time.

### Cities

`--adm 3` labels the window's major cities from OSM populated places —
black square markers with the city name, from `place=city` and `place=town`
nodes ranked by the `population` tag (a place-class default when the tag is
missing, capitals boosted so they survive the label budget). Each country's
nodes are queried within its own ADM0 extent and re-checked against that
geometry, so neighbouring places never leak in — and where OSM has no
country-level relation at all (Åland is modelled only as a level-4 region
plus its municipalities), the bundled Natural Earth country scopes and
attributes the labels instead. Drawn borders never fall back to NE. The
nodes are
cached beside the boundaries as `openstreetmap-{ISO}-CITIES.geojson` with
the same ODbL sidecar; like every other layer it prefetches
(`prefetch_boundaries.py FRA,BEL --levels CITIES`) and renders offline under
`NAAULU_BOUNDARIES_OFFLINE=1`. A plot fills a missing cache itself, once,
and reads it from then on — under `NAAULU_BOUNDARIES_OFFLINE=1` it never
fetches and missing labels are simply omitted.

### Fetching

All tiers fill through `prefetch_boundaries.py`, which queries Geofabrik
Postpass: SQL over OSM's full planet database, served as GeoJSON. One
country is a handful of small queries — its own outline by ISO3166-1 tag,
its province units and its cities bounded by that outline — so a whole
prefetch costs megabytes of traffic and never a planet file on disk.
Postpass is the same underlying database the planet.osm mirrors hold, free
of the rate limits that made Overpass unusable for bulk fills (its bans
cost this project entire days of blocked retries). A render fills exactly
the window's countries on its first run and reads the cache from then on;
`NAAULU_BOUNDARIES_OFFLINE=1` keeps serves strictly local — a gap stays a
gap instead of a network call mid-serve.

### Why not GADM (or geoBoundaries)

Earlier versions used [GADM](https://gadm.org/license.html) for the same
layers. GADM is restricted to academic and non-commercial use, with
redistribution and commercial use requiring prior permission — a
non-commercial-only boundary source is not usable by everyone naaulu is for —
and it publishes no per-country provenance. geoBoundaries replaced it:
commercial-friendly, documented per-country sources — but that same
per-country standardisation is what breaks shared seams (see above), which
is why the drawn borders now come from OpenStreetMap's single database.
Eurostat GISCO, the other harmonised candidate, carries the same
non-commercial restriction that ruled GADM out.

## Bundled and derived files

| File | Derived from | Licence |
|---|---|---|
| `naaulu/data/radars_wmo.json`, `radars.json` | [WMO Weather Radar Database](https://wrd.mgm.gov.tr/Home/About) | **CC BY-SA 4.0**, attribute "World Meteorological Organization (WMO)" |
| `naaulu/data/radars_usa.json` | NCEI HOMR / NOAA | public domain |
| `naaulu/data/radars_aus.json` | NCI/AURA radar site list (BoM) | CC BY 4.0 |
| `naaulu/data/radars_opera.json` | EUMETNET OPERA | CC BY 4.0 |
| `naaulu/data/chmi_aws_stations.json` | CHMI station GeoPackage | CC BY 4.0 |
| `naaulu/data/ne_50m_admin_0_countries.geojson` | Natural Earth | public domain |
| `naaulu/data/ne_10m_admin_0_countries.geojson` | Natural Earth | public domain |

The share-alike on the WMO-derived files is why they are covered by the
project `NOTICE` rather than by the AGPL of the code.

## Republishing naaulu output

Figures and NetCDF tiles produced by naaulu are derivatives of the sources
above. A single credit line covers most of the obligations:

> Radar: EUMETNET OPERA / FMI / NOAA (NEXRAD) / Bureau of Meteorology.
> Gauges: KNMI, RMI, SPW, VMM, CHMI, Keskkonnaagentuur, FMI, NOAA (GHCN),
> Iowa Environmental Mesonet. Boundaries: © OpenStreetMap contributors
> (ODbL), Natural Earth.

Add the WMO line when a map uses the bundled radar database, and keep the
VMM caveat in mind for anything commercial.

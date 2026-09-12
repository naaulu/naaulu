# Naaulu

Naaulu is a Python toolbox for building, combining, verifying, and visualizing rainfall estimation datasets.

> **Note:** This project is currently a **Work In Progress (WIP)** preview. Features may change and stability is not guaranteed.

## For users

See [website](https://naaulu.org).

## For developers

### Install on windows

```powershell
type install.ps1
.\install.ps1
```

### Install on Linux

```sh
cat install.sh
./install.sh
```

## Run

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

## License

This project is licensed under the GNU Affero General Public License v3.0 (AGPL-3.0).

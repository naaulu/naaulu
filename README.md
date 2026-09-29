# Naaulu

Naaulu is a Python toolbox for building, combining, verifying, and visualizing rainfall estimation datasets.

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

## License

This project is licensed under the GNU Affero General Public License v3.0 (AGPL-3.0).

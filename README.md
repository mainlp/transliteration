# Linear Script Representations in Speech Foundation Models

Code for [*Linear Script Representations in Speech Foundation Models Enable Zero-Shot Transliteration*](https://aclanthology.org/2026.findings-acl.464/).

## Installation

Python 3.10 or newer and `ffmpeg` are required.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .
```

For the OWSM-CTC experiment, install the optional ESPnet dependencies:

```bash
pip install -e '.[owsm]'
```

## Reproducing the paper

Run the experiment groups in this order:

```bash
bash experiments/script_confusion.sh
bash experiments/transliteration.sh
bash experiments/probe.sh
bash experiments/encoder_and_owsm.sh
```

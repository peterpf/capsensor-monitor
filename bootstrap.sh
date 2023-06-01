#!/usr/bin/env sh

# Switch to the directory of this script
cd "$(dirname "$0")"

# set pipefail
set -e

# Create and activate the virtual environment (works on linux and windows)
venv_name=".venv"
python3 -m venv $venv_name 2> /dev/null || python -m venv $venv_name
[ -f $venv_name/bin/activate ] && . $venv_name/bin/activate
[ -f $venv_name/Scripts/activate ] && . $venv_name/Scripts/activate

# Install pip and poetry (for dependency management)
python3 -m pip install -U 2> /dev/null || python -m pip install -U pip
pip install poetry

# Install dependencies
poetry install

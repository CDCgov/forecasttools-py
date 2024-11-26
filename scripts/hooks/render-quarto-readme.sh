#!/bin/bash

set -e # halt execution if non-zero status

# is quarto command available?
if ! command -v quarto &> /dev/null; then
  echo "Quarto is not installed. Please install it to use this hook."
  exit 1
fi

# is poetry command available?
if ! command -v poetry &> /dev/null; then
  echo "Poetry is not installed. Please install it to use this hook."
  exit 1
fi

# set up poetry env
echo "Installing dependencies with Poetry..."
poetry install --with dev

echo "Rendering README.md from README.qmd..."
poetry run quarto render README.qmd

# # add the generated README.md to the commit
# git add README.md

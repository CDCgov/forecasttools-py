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

# prompt user to stage/commit README
# if there is a change
if [[ -f README.md ]]; then
  if ! git diff --quiet README.md; then
    echo "README.md has been modified."
    git add README.md
    echo "Changes staged. Please commit the updated README.md."
    exit 0
  else
    echo "README.md is up to date."
    exit 0
  fi
else
  echo "README.md not generated. Something went wrong with render."
  exit 1
fi

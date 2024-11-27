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
poetry run quarto render README.qmd --no-cache

# prompt user to stage/commit README
# if there is a change
if [[ -f README.md ]]; then

  # make sure considered in diff but not
  # actually staged
  git add -N README.md

  # dont want --cached because want unstaged
  # stages to show up as a diff
  if ! git diff --quiet --ignore-all-space --ignore-space-change --exit-code README.md; then

    echo "README.md has been modified by this hook."
    echo "Changes staged. Please commit the updated README.md."
    # the presence of "exit 1" here has been
    # debated
  else
    echo "README.md is up to date."
  fi
else
  echo "README.md not found. Something went wrong."
  exit 1
fi

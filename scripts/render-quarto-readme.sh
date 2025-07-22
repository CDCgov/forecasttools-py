#!/bin/bash

set -e # halt execution if non-zero status

# is quarto command available?
if ! command -v quarto &>/dev/null; then
  echo "Quarto is not installed. Please install it to use this hook."
  exit 1
fi

echo "Rendering README.md from README.qmd..."
quarto render README.qmd

# prompt user to stage/commit README
# if there is a change
if [[ -f README.md ]]; then

  # make sure considered in diff but not
  # actually staged
  git add --intent-to-add README.md

  # dont want --cached because want unstaged
  # stages to show up as a diff
  if ! git diff --quiet --exit-code README.md; then

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

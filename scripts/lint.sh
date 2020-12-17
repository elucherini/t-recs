#! /bin/bash

set -ex # exit script upon first error, print commands line-by-line when running

SRC_DIR=${SRC_DIR:-$(pwd)} # if SRC_DIR is not defined, then execute pwd command 

echo "Checking code style with black..."
python -m black --line-length 100 --check "${SRC_DIR}"
echo "Success!"

echo "Type checking with mypy..."
mypy --ignore-missing-imports trecs
echo "Success!"

echo "Checking code style with pylint..."
python -m pylint "${SRC_DIR}"/trecs/
echo "Success!"

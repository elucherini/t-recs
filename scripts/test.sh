#! /bin/bash

set -ex # fail on first error, print commands

SRC_DIR=${SRC_DIR:-$(pwd)}

pytest -vx --cov trecs
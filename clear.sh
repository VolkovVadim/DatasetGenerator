#!/bin/bash

ROOT_DIR="$( cd "$(dirname "$0")" ; pwd -P )"

rm -rfv ${ROOT_DIR}/Classification/*.csv
rm -rfv ${ROOT_DIR}/Regression/*.csv

echo "Done"
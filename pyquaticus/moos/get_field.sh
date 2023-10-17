#!/bin/bash
SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
DEPS=$(realpath $SCRIPT_DIR/../../../)
MISSION_DIR="$DEPS/moos-ivp-aquaticus/missions/jervis-2023"
echo "Using Mission directory: $MISSION_DIR"
pattern="$MISSION_DIR/surveyor/region_info_*.txt"
files=( $pattern )
REGION_INFO="${files[0]}"
echo "Using region info file: $REGION_INFO"
grep -A 2 REGION $REGION_INFO > field.txt
grep -A 1 BLUE_FLAG $REGION_INFO > flags.txt

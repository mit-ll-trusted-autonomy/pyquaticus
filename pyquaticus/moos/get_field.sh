#!/bin/bash
SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
DEPS=$(realpath $SCRIPT_DIR/../../../)
MISSION_DIR="$DEPS/moos-ivp-aquaticus/missions/jervis-2023"
echo "Using Mission directory: $MISSION_DIR"
grep -A 2 REGION $MISSION_DIR/surveyor/region_info_red_one.txt > field.txt
grep -A 1 BLUE_FLAG $MISSION_DIR/surveyor/region_info_red_one.txt > flags.txt

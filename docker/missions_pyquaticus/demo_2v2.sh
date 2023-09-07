#!/bin/bash

TIME_WARP=4

./launch_demo.sh --team-size=2 ${TIME_WARP} &
# wait for everything to start up
sleep 1

cd shoreside
uPokeDB targ_shoreside.moos ACTION_BLUE_ONE=CONTROL DEPLOY_BLUE_ONE=true MOOS_MANUAL_OVERRIDE_BLUE_ONE=false RETURN_BLUE_ONE=false >/dev/null 2>&1
uPokeDB targ_shoreside.moos ACTION_RED_ONE=CONTROL DEPLOY_RED_ONE=true MOOS_MANUAL_OVERRIDE_RED_ONE=false RETURN_RED_ONE=false >/dev/null 2>&1

cd ..

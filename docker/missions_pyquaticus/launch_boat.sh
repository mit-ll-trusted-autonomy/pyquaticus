#!/bin/bash
TIME_WARP=1

LOGDIR="../logs"

echo Logging to $LOGDIR

#-------------------------------------------------------
#  Launching herons
#-------------------------------------------------------
# Access the first argument ($1) as the boat name
BOAT=$1
COLOR=$2

if [[ "$COLOR" == "red" ]]; then
    ./launch_surveyor.sh $BOAT r1 r2 1 $LOGDIR --role=CONTROL > /dev/null &
    sleep 1

else
    ./launch_surveyor.sh $BOAT b1 b2 1 $LOGDIR --role=CONTROL > /dev/null &
    sleep 1

cd ..


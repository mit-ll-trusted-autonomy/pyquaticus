#!/bin/bash
TIME_WARP=1

LOGPATH="../logs"

echo Logging to $LOGDIR

#-------------------------------------------------------
#  Launching herons
#-------------------------------------------------------
if [[ -z $NO_HERON ]]; then
  cd ./surveyor
  # Gus Red
  ./launch_surveyor.sh s r1 r2 1 $LOGDIR --role=CONTROL > /dev/null &
  sleep 1
  # Luke Red
  # ./launch_surveyor.sh t r2 r1 $TIME_WARP $LOGDIR -s --start-x=262 --start-y=156 --start-a=240 --role=DEFEND_E > /dev/null &
  # sleep 1
  # Kirk Blue
  # ./launch_surveyor.sh u b1 b2 $TIME_WARP $LOGDIR -s --start-x=230 --start-y=78 --start-a=60 --role=ATTACK_MED > /dev/null &
  # sleep 1
  # Jing Blue
  # ./launch_surveyor.sh v b2 b1 $TIME_WARP $LOGDIR -s --start-x=220 --start-y=80 --start-a=60 --role=DEFEND_MED > /dev/null &
  # sleep 1
  cd ..
fi

#!/bin/bash
TIME_WARP=1

CMD_ARGS=""
NO_HERON=""
NO_MOKAI=""
NO_SHORESIDE=""
LOGPATH=""
NUM_AGENTS="1"

#-------------------------------------------------------
#  Part 1: Check for and handle command-line arguments
#-------------------------------------------------------
for ARGI; do
    if [ "${ARGI}" = "--help" -o "${ARGI}" = "-h" ] ; then
        HELP="yes"
    elif [ "${ARGI}" = "--no_shoreside" -o "${ARGI}" = "-ns" ] ; then
        NO_SHORESIDE="true"
    elif [ "${ARGI//[^0-9]/}" = "$ARGI" -a "$TIME_WARP" = 1 ]; then
        TIME_WARP=$ARGI
        echo "Time warp set up to $TIME_WARP."
    elif [ "${ARGI}" = "--just_build" -o "${ARGI}" = "-j" ] ; then
        JUST_BUILD="yes"
        echo "Just building files; no vehicle launch."
    elif [ "${ARGI:0:10}" = "--logpath=" ]; then
        LOGPATH="${ARGI#--logpath=*}"
    elif [ "${ARGI:0:12}" = "--team-size=" ]; then
        NUM_AGENTS="${ARGI#--team-size=*}"
	if [ "${NUM_AGENTS}" -ne "1" -a "${NUM_AGENTS}" -ne "2" ]; then
            HELP="yes"
        fi
    else
        CMD_ARGS=$CMD_ARGS" "$ARGI
    fi
done


if [ "${HELP}" = "yes" ]; then
  echo "$0 [SWITCHES]"
  echo "  XX                  : Time warp"
  echo "  --no_shoreside, -ns"
  echo "  --just_build, -j"
  echo "  --logpath="
  echo "  --team-size,   {options: 1 or 2, default: 1}"
  echo "  --help, -h"
  exit 0;
fi

if [ -n "${LOGPATH}" ]; then
  LOGDIR=--logpath=${LOGPATH}
fi

echo Logging to $LOGDIR

#-------------------------------------------------------
#  Part 2: Launching herons
#-------------------------------------------------------
echo "Launching ${NUM_AGENTS} heron(s) per team"
if [[ -z $NO_HERON ]]; then
  cd ./surveyor
  # Gus Red
  ./launch_surveyor.sh s r1 r2 $TIME_WARP $LOGDIR -s --start-x=250.23 --start-y=156.03 --start-a=201.3 --role=ATTACK_E > /dev/null &
  sleep 1
  # Kirk Blue
  ./launch_surveyor.sh u b1 b2 $TIME_WARP $LOGDIR -s --start-x=221.17 --start-y=81.49 --start-a=21.3 --role=ATTACK_MED > /dev/null &
  sleep 1

  if [ "${NUM_AGENTS}" = "2" ]; then
      # Luke Red
      ./launch_surveyor.sh t r2 r1 $TIME_WARP $LOGDIR -s --start-x=259.55 --start-y=152.39 --start-a=201.3 --role=DEFEND_E > /dev/null &
      sleep 1
      # Jing Blue
      ./launch_surveyor.sh v b2 b1 $TIME_WARP $LOGDIR -s --start-x=230.49 --start-y=77.85 --start-a=21.3 --role=DEFEND_MED > /dev/null &
      sleep 1
  fi
  cd ..
fi



#-------------------------------------------------------
#  Part 3: Launching shoreside
#-------------------------------------------------------
if [[ -z $NO_SHORESIDE ]]; then
  cd ./shoreside
  ./launch_shoreside.sh $TIME_WARP >& /dev/null &
  cd ..
fi

#-------------------------------------------------------
#  Part 4: Launching uMAC
#-------------------------------------------------------
uMAC shoreside/targ_shoreside.moos

#-------------------------------------------------------
#  Part 5: Killing all processes launched from script
#-------------------------------------------------------
echo "Killing Simulation..."
kill -- -$$
# sleep is to give enough time to all processes to die
sleep 3
echo "All processes killed"

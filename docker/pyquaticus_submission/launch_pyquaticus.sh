#!/bin/bash

# To run just launch_pyquatcius use:
# ./launch_pyquaticus.sh --boat-name u --boat-role blue_one --logpath=./logs/ --time-warp 4 --policy-dir ./policies/u/ --num-players 2 --sim
# if not running in simulation, also pass --shore-ip <ip address> and --host-op <ip address>

THISDIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"

SHORE_IP="$(python-c 'import config;print(config.get_shore_ip())')"

HOST_IP="$(python-c 'import config;print(config.get_boat_ip())')"


# Typical format for SHORE/HOST IP
# SHORE_IP="192.168.1.111"
# HOST_IP="192.168.1.42" # where boat number is 4 (the last 4) and 2 means backseat computer

TIME_WARP=4
CMD_ARGS=""
NO_HERON=""
LOGPATH=""
COLOR=""
SIMULATION=""

#-------------------------------------------------------
#  Part 1: Check for and handle command-line arguments
#-------------------------------------------------------
for ARGI; do
    case "$ARGI" in
        -h|--help)
            HELP="yes"
            ;;
        -n|--num-players)
            # shift
            NUM_PLAYERS="$1"
            ;;
        --logpath=*)

            LOGPATH="${ARGI#--logpath=}"
            ;;
        -b|--boat-name)
            shift
            if [[ "s t u v w x y z" =~ "$1" ]]; then
                BOAT_NAME="$1"
            else
                echo "Error: Invalid boat name. Accepted names are: s, t, u, v, w, x, y, z."
                exit 1
            fi
            ;;
        -s|--sim)
            SIMULATION="--sim"
            ;;
        -r|--boat-role)
            if [[ "red_one blue_one red_two blue_two" =~ "$1" ]]; then
                BOAT_ROLE="$1"
                # Infer the color from the boat role
                if [[ "$BOAT_ROLE" =~ "red" ]]; then
                    COLOR="red"
                else
                    COLOR="blue"
                fi
            else
                echo "Error: Invalid boat role. Accepted roles are: red_one, blue_one, red_two, blue_two."
                exit 1
            fi
            ;;
        -t|--time-warp)
            TIME_WARP="$1"
            ;;
	--shore-ip)
	    SHORE_IP="$1"
	    ;;
	--host-ip)
	    HOST_IP="$1"
	    ;;
        *)
            CMD_ARGS="$CMD_ARGS $ARGI"
            ;;
    esac
    shift
done

echo ""
echo "Logpath: ${LOGPATH}"
echo "Num players: ${NUM_PLAYERS}"
echo "Boat name: ${BOAT_NAME}"
echo "Boat role: ${BOAT_ROLE}"
echo "Time warp: ${TIME_WARP}"
echo "Simulation: ${SIMULATION}"
echo "Shore IP: ${SHORE_IP}"
echo "Host IP: ${HOST_IP}"
echo "Color: ${COLOR}"
echo ""

if [ "${HELP}" = "yes" ]; then
  echo "Usage: $0 [SWITCHES]"
  echo "  -t, --time-warp XX      : Set the time warp (default is 4)."
  echo "  -p, --policy-dir PATH   : Specify the policy directory (default is ./policies/BOAT_NAME/)."
  echo "  -n, --num-players NUM   : Specify the number of players per team (default is 2)."
  echo "  -ns, --no_shoreside     : Do not launch the shoreside."
  echo "  -j, --just_build        : Just build files; no vehicle launch."
  echo "  -l, --logpath=PATH      : Specify the log path."
  echo "  -b, --boat-name NAME    : Specify the boat name (e.g., u, v, s, t)."
  echo "  -s, --sim               : Launch in simulation mode."
  echo "  --shore-ip IP           : Provide the shore IP address if not running in simulation mode."
  echo "  --host-ip IP            : Provide the host IP address if not running in simulation mode."
  echo "  -r, --boat-role ROLE    : Specify the boat role (e.g., red_one, blue_one, red_two, blue_two). Color is derived from the role."
  echo "  -h, --help              : Display this help and exit."
  exit 0;
fi

if [[ "$SIMULATION" == "" && ("${SHORE_IP}" == "localhost" || "${HOST_IP}" == "localhost") ]]; then
  echo "Must provide both shore IP and host IP if not running in simulation mode."
  exit 1
fi


#-------------------------------------------------------
#  Part 2: Launching herons
#-------------------------------------------------------
current_path=$(pwd)

mkdir $current_path/$LOGPATH

if [[ -z $NO_HERON ]]; then
  cd /home/john/moos-ivp-aquaticus/missions/wp_2024/surveyor/
  if [ "$BOAT_ROLE" == "blue_one" ]; then
    # New
    ./launch_surveyor.sh -v$BOAT_NAME -b1 $TIME_WARP --logpath=$current_path/$LOGPATH $SIMULATION --start=100,81.49,21.3 --role=CONTROL --shore=$SHORE_IP --ip=$HOST_IP > /dev/null &
  elif [ "$BOAT_ROLE" == "blue_two" ]; then
    # New
    ./launch_surveyor.sh -v$BOAT_NAME -b2 $TIME_WARP --logpath=$current_path/$LOGPATH $SIMULATION --start=100,77.85,21.3 --role=CONTROL --shore=$SHORE_IP --ip=$HOST_IP > /dev/null &
  fi

  if [ "$BOAT_ROLE" == "red_one" ]; then
    # New
    ./launch_surveyor.sh -v$BOAT_NAME -r1 $TIME_WARP --logpath=$current_path/$LOGPATH $SIMULATION --start=70,156.03,201.3 --role=CONTROL --shore=$SHORE_IP --ip=$HOST_IP > /dev/null &
  elif [ "$BOAT_ROLE" == "red_two" ]; then
      # New
    ./launch_surveyor.sh -v$BOAT_NAME -r2 $TIME_WARP --logpath=$current_path/$LOGPATH $SIMULATION --start=70,152.39,201.3 --role=CONTROL --shore=$SHORE_IP --ip=$HOST_IP > /dev/null &
  fi
fi
sleep 3

cd $current_path
#-------------------------------------------------------
#  Part 3: Launch the python script
#-------------------------------------------------------

echo "Running pyquaticus_moos_bridge.py"

# echo "python3 solution.py $([ "$SIMULATION" == "true" ] && echo "--sim") --color $COLOR --policy-dir $POLICY_DIR --boat_id $BOAT_ROLE --num-players $NUM_PLAYERS --boat_name $BOAT_NAME --timewarp $TIME_WARP"
python3 game_loop.py $SIMULATION --color $COLOR --boat_id $BOAT_ROLE --num-players $NUM_PLAYERS --boat_name $BOAT_NAME --timewarp $TIME_WARP

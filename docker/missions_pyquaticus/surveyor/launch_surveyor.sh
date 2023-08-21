#!/bin/bash
TIME_WARP=1

#SHORE_IP=192.168.1.252
SHORE_IP=localhost
SHORE_LISTEN="9300"

TRAIL_RANGE="3"
TRAIL_ANGLE="330"
HELP="no"
JUST_BUILD="no"
VTEAM=""
VNAME=""
RNAME=""
VMODEL="M1_8"

START_POS="140,45,240"
RETURN_POS="5,0"
LOITER_POS="x=100,y=-180"
GRAB_POS=""
GRABR_POS=""
GRABL_POS=""
UNTAG_POS=""

HERON_TEAMMATE=""
HERON_TEAMMATE_VTEAM=""

CID=000 #COMP ID
LOGPATH=./

START_ACTION="PROTECT"

source ../aquaticus_params.txt
echo $CORNER_1
echo $MIDFIELD_1
TEST=$MIDFIELD_1,$CORNER_1
echo $TEST
#exit

function help(){
    echo ""
    echo "USAGE: $0 <heron_vehicle_name> <vehicle_role> <heron_teammate_vehicle_role> [SWITCHES]"
    
    echo ""
    echo "POSSIBLE SURVEYOR VEHICLE NAMES:"
    echo "  scott,         s   : Scott surveyor."
    echo "  thomas,        t   : Thomas surveyor."
    echo "  ursula,        u   : Ursula surveyor."
    echo "  valhalla,      v   : Valhalla surveyor."
    echo "  walter,        w   : Walter surveyor."
    echo "  xavier,        x   : Xavier surveyor."
    echo "  yolanda,       y   : Yolanda surveyor."
    echo "  zach,          z   : Zach surveyor."

    echo ""
    echo "POSSIBLE ROLES (and heron teammate_roles):"
    echo "  blue_one,     b1  : Vehicle one on blue team."
    echo "  blue_two,     b2  : Vehicle two on blue team."
    echo "  red_one,      r1  : Vehicle one on red team."
    echo "  red_two,      r2  : Vehicle two on red team."

    echo ""
    echo "POSSIBLE SWITCHES:"
    echo "  --role,           : Autonomy startup roles: ATTACK/DEFEND _E/_MED."
    echo "  --sim,        -s  : Simulation mode."
    echo "  --start-x=        : Start from x position (requires x y a)."
    echo "  --start-y=        : Start from y position (requires x y a)."
    echo "  --start-a=        : Start from angle (requires x y a)."
    echo "  --cid=            : Competition ID (for log file)"
    echo "  --logpath=        : Log path"
    echo "  --just_build, -J  : Just build targ files."
    echo "  --help,       -H  : Display this message."
    echo "  #                 : Moos Time Warp."
    exit 0
}

#-------------------------------------------------------
#  Part 1: Check for and handle command-line arguments
#-------------------------------------------------------
case "$1" in
    s|scott)
        SURVEYOR_IP=192.168.1.11
        VNAME="SCOTT"
        echo "SCOTT surveyor selected."
        ;;
    t|thomas)
        SURVEYOR_IP=192.168.1.21
        VNAME="THOMAS"
        echo "THOMAS surveyor selected."
        ;;
    u|ursula)
        SURVEYOR_IP=192.168.1.31
        VNAME="URSULA"
        echo "URSULA surveyor selected."
        ;;
    v|valhalla)
        SURVEYOR_IP=192.168.1.41
        VNAME="VALHALLA"
        echo "VALHALLA surveyor selected."
        ;;
    w|walter)
        SURVEYOR_IP=192.168.1.51
        VNAME="WALTER"
        echo "WALTER surveyor selected."
        ;;
    x|xavier)
        SURVEYOR_IP=192.168.1.61
        VNAME="XAVIER"
        echo "XAVIER surveyor selected."
        ;;
    y|yolanda)
        SURVEYOR_IP=192.168.1.71
        VNAME="YOLANDA"
        echo "YOLANDA surveyor selected."
        ;;
    z|zach)
	SURVEYOR_IP=192.168.1.81
        VNAME="ZACH"
	    echo "ZACH surveyor selected."
	    ;;
    *)
        echo "!!! Error invalid positional argument $1 !!!"
        ;;
esac

case "$2" in
    r1|red_one)
        VTEAM="red"
        RNAME="red_one"
        VPORT="9011"
	VR_PORT="9811"
        SHARE_LISTEN="9311"
	START_ACTION="DEFEND_E"
        echo "Vehicle set to red one."
        ;;
    r2|red_two)
        VTEAM="red"
        RNAME="red_two"
        VPORT="9012"
	VR_PORT="9812"
        SHARE_LISTEN="9312"
	START_ACTION="ATTACK_MED"
        echo "Vehicle set to red two."
        ;;
    b1|blue_one)
        VTEAM="blue"
        RNAME="blue_one"
        VPORT="9015"
	VR_PORT="9815"
        SHARE_LISTEN="9315"
	START_ACTION="DEFEND_MED"
        echo "Vehicle set to blue one."
        ;;
    b2|blue_two)
        VTEAM="blue"
        RNAME="blue_two"
        VPORT="9016"
	VR_PORT="9816"
        SHARE_LISTEN="9316"
	PLAYERS="b1,b3,b4"
	START_ACTION="ATTACK_E"
        echo "Vehicle set to blue two."
        ;;
    *)
        echo "!!! Error invalid positional argument $2 !!!"
        help
        ;;
esac

	
case "$3" in
    r1|red_one)
        HERON_TEAMMATE="red_one"
        HERON_TEAMMATE_VTEAM="red"
        echo "Vehicle set to red one."
        ;;
    r2|red_two)
        HERON_TEAMMATE="red_two"
        HERON_TEAMMATE_VTEAM="red"
        echo "Vehicle set to red two."
        ;;
    b1|blue_one)
        HERON_TEAMMATE="blue_one"
        HERON_TEAMMATE_VTEAM="blue"
        echo "Vehicle set to blue one."
        ;;
    b2|blue_two)
        HERON_TEAMMATE="blue_two"
        HERON_TEAMMATE_VTEAM="blue"
        echo "Vehicle set to blue two."
        ;;
    *)
        echo "!!! Error invalid positional argument $3 !!!"
        ;;
esac

if [[ "$HERON_TEAMMATE_VTEAM" != "$VTEAM" ]]; then
    echo "!!! Error teammate team can not be different then vehicle team !!!"
    help
fi

for arg in "${@:4}"; do
    if [ "${arg}" = "--help" -o "${arg}" = "-H" ]; then
        help
    elif [ "${arg//[^0-9]/}" = "$arg" -a "$TIME_WARP" = 1 ]; then
        TIME_WARP=$arg
        echo "Time warp set to: " $arg
    elif [ "${arg}" = "--just_build" -o "${arg}" = "-J" ] ; then
        JUST_BUILD="yes"
        echo "Just building files; no vehicle launch."
    elif [ "${arg}" = "--sim" -o "${arg}" = "-s" ] ; then
        SIM="SIM"
        echo "Simulation mode ON."
    elif [ "${arg:0:10}" = "--start-x=" ] ; then
        START_POS_X="${arg#--start-x=*}"
    elif [ "${arg:0:10}" = "--start-y=" ] ; then
        START_POS_Y="${arg#--start-y=*}"
    elif [ "${arg:0:10}" = "--start-a=" ] ; then
        START_POS_A="${arg#--start-a=*}"
    elif [ "${arg:0:7}" = "--role=" ] ; then
        START_ACTION="${arg#--role=*}"
    elif [ "${arg:0:6}" = "--cid=" ] ; then
        CID="${arg#--cid=*}"
        CID=$(printf "%03d" $CID)
    elif [ "${arg:0:10}" = "--logpath=" ]; then
        LOGPATH="${arg#--logpath=*}"
    else
        echo "Undefined switch:" $arg
        help
    fi
done

#HERON_NAME=`get_vname.sh`
#echo "Heron name:"$HERON_NAME
echo $LOGPATH



if [ "${VTEAM}" = "red" ]; then
    GRAB_POS=$BLUE_FLAG_X,$BLUE_FLAG_Y
    GRABR_POS="20,20"
    GRABL_POS="30,30"
    UNTAG_POS=$RED_FLAG_X,$RED_FLAG_Y
    RETURN_POS="5,0"
    START_POS="140,45,240"
    STATION_KEEP_AGGRESSIVE="249.44,140.23"
    OPFOR="blue"
    OPFOR_ZONE=$MIDFIELD_1,$CORNER_2,$CORNER_3,$MIDFIELD_2
    echo $OPFOR_ZONE
    echo "Red team selected."
elif [ "${VTEAM}" = "blue" ]; then
    GRAB_POS=$RED_FLAG_X,$RED_FLAG_Y
    GRABR_POS="130,30"
    GRABL_POS="130,50"
    UNTAG_POS=$BLUE_FLAG_X,$BLUE_FLAG_Y
    RETURN_POS="5,0"
    START_POS="20,35,60"
    STATION_KEEP_AGGRESSIVE="231.28,93.65"
    OPFOR="red"
    OPFOR_ZONE=$CORNER_1,$MIDFIELD_1,$MIDFIELD_2,$CORNER_4
    echo "Blue team selected."
fi
   
#-------------------------------------------------------
#  Part 2: Create the .moos and .bhv files.
#-------------------------------------------------------

if [[ -n $START_POS_X && (-n $START_POS_Y && -n $START_POS_A)]]; then
  START_POS="$START_POS_X,$START_POS_Y,$START_POS_A"
  echo "Starting from " $START_POS
elif [[ -z $START_POS_X && (-z $START_POS_Y && -z $START_POS_A) ]]; then
  echo "Starting from default postion: " $START_POS
else [[ -z $START_POS_X || (-z $START_POS_Y || -z $START_POS_A) ]]
  echo "When specifying a starting coordinate, all 3 should be specified (x,y,a)."
  echo "See help (-h)."
  exit 1
fi

echo "Assembling MOOS file targ_${RNAME}.moos"


nsplug meta_surveyor.moos targ_${RNAME}.moos -f \
    VNAME=$VNAME                 \
    RNAME=$RNAME                 \
    VPORT=$VPORT                 \
    VR_PORT=$VR_PORT             \
    WARP=$TIME_WARP              \
    SHARE_LISTEN=$SHARE_LISTEN   \
    SHORE_LISTEN=$SHORE_LISTEN   \
    SHORE_IP=$SHORE_IP           \
    SURVEYOR_IP=$SURVEYOR_IP     \
    HOSTIP_FORCE="localhost"     \
    LOITER_POS=$LOITER_POS       \
    VARIATION=$VARIATION         \
    VMODEL=$VMODEL               \
    VTYPE="kayak"                \
    VTEAM=$VTEAM                 \
    START_POS=$START_POS         \
    CID=$CID                     \
    LOGPATH=$LOGPATH             \
    UNTAG_POS=$UNTAG_POS         \
    OPFOR=$OPFOR                 \
    OPFOR_ZONE=$OPFOR_ZONE       \
    $SIM                         

echo "Assembling BHV file targ_${RNAME}.bhv"
nsplug meta_surveyor.bhv targ_${RNAME}.bhv -f  \
    RETURN_POS=${RETURN_POS}       \
    TRAIL_RANGE=$TRAIL_RANGE       \
    TRAIL_ANGLE=$TRAIL_ANGLE       \
    VTEAM=$VTEAM                   \
    VNAME=$VNAME                   \
    RNAME=$RNAME                   \
    GRAB_POS=$GRAB_POS             \
    GRABR_POS=$GRABR_POS           \
    GRABL_POS=$GRABL_POS           \
    UNTAG_POS=$UNTAG_POS           \
    HERON_TEAMMATE=$HERON_TEAMMATE \
    STATION_KEEP_AGGRESSIVE=$STATION_KEEP_AGGRESSIVE \
    START_ACTION=$START_ACTION     \
    CORNER_1=$CORNER_1             \
    CORNER_2=$CORNER_2             \
    CORNER_3=$CORNER_3             \
    CORNER_4=$CORNER_4             


if [ ${JUST_BUILD} = "yes" ] ; then
    echo "Files assembled; vehicle not launched; exiting per request."
    exit 0
fi

#-------------------------------------------------------
#  Part 3: Launch the processes
#-------------------------------------------------------

echo "Launching $RNAME MOOS Community "
pAntler targ_${RNAME}.moos >& /dev/null &

uMAC targ_${RNAME}.moos

echo "Killing all processes ..."
kill -- -$$
echo "Done killing processes."

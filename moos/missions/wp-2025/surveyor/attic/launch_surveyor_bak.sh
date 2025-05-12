#!/bin/bash
#---------------------------------------------------------------
#   Script: launch.sh
#  Mission: a1_aquaticus
#   Author: Mike Benjamin
#   LastEd: Oct 2023
#-------------------------------------------------------------- 
#  Part 1: Define a convenience function for producing terminal
#          debugging/status output depending on the verbosity.
#-------------------------------------------------------------- 
vecho() { if [ "$VERBOSE" != "" ]; then echo "$ME: $1"; fi }

#---------------------------------------------------------------
#  Part 2: Set global var defaults
#---------------------------------------------------------------
ME=`basename "$0"`
TIME_WARP=1
VERBOSE=""
JUST_MAKE="no"
AUTO_LAUNCHED="no"

#SHORE_IP=192.168.1.252
SHORE_IP=localhost
SHORE_LISTEN="9300"

TRAIL_RANGE="3"
TRAIL_ANGLE="330"
HELP="no"
VTEAM=""
VNAME=""
RNAME=""
VMODEL="M1_8"
START_POS=""

CID=000 #COMP ID
LOGPATH=./

START_ACTION="PROTECT"

./jerop.sh > region_info.txt
source region_info.txt


function help(){
    echo ""
    echo "USAGE: $ME <heron_vehicle_name> <vehicle_role> <heron_teammate_vehicle_role> [SWITCHES]"
    
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
    echo "  --start=<X,Y,H>                                " 
    echo "    Start position chosen by script launching    "
    echo "    this script (to ensure separation)           "
    echo "  --cid=            : Competition ID (for log file)"
    echo "  --logpath=        : Log path"
    echo "  --just_build, -J  : Just build targ files."
    echo "  --help,       -H  : Display this message."
    echo "  --verbose,    -V  : Verbose launch."
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
        vecho "SCOTT surveyor selected."
        ;;
    t|thomas)
        SURVEYOR_IP=192.168.1.21
        VNAME="THOMAS"
        vecho "THOMAS surveyor selected."
        ;;
    u|ursula)
        SURVEYOR_IP=192.168.1.31
        VNAME="URSULA"
        vecho "URSULA surveyor selected."
        ;;
    v|valhalla)
        SURVEYOR_IP=192.168.1.41
        VNAME="VALHALLA"
        vecho "VALHALLA surveyor selected."
        ;;
    w|walter)
        SURVEYOR_IP=192.168.1.51
        VNAME="WALTER"
        vecho "WALTER surveyor selected."
        ;;
    x|xavier)
        SURVEYOR_IP=192.168.1.61
        VNAME="XAVIER"
        vecho "XAVIER surveyor selected."
        ;;
    y|yolanda)
        SURVEYOR_IP=192.168.1.71
        VNAME="YOLANDA"
        vecho "YOLANDA surveyor selected."
        ;;
    z|zach)
	SURVEYOR_IP=192.168.1.81
        VNAME="ZACH"
	vecho "ZACH surveyor selected."
	    ;;
    *)
        vecho "!!! Error invalid positional argument $1 !!!"
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
        vecho "Vehicle set to red one."
        ;;
    r2|red_two)
        VTEAM="red"
        RNAME="red_two"
        VPORT="9012"
	VR_PORT="9812"
        SHARE_LISTEN="9312"
	START_ACTION="ATTACK_MED"
        vecho "Vehicle set to red two."
        ;;
    b1|blue_one)
        VTEAM="blue"
        RNAME="blue_one"
        VPORT="9015"
	VR_PORT="9815"
        SHARE_LISTEN="9315"
	START_ACTION="DEFEND_MED"
        vecho "Vehicle set to blue one."
        ;;
    b2|blue_two)
        VTEAM="blue"
        RNAME="blue_two"
        VPORT="9016"
	VR_PORT="9816"
        SHARE_LISTEN="9316"
	PLAYERS="b1,b3,b4"
	START_ACTION="ATTACK_E"
        vecho "Vehicle set to blue two."
        ;;
    *)
        vecho "!!! Error invalid positional argument $2 !!!"
        help
        ;;
esac


for arg in "${@:4}"; do
    CMD_ARGS+="${arg} "
    if [ "${arg}" = "--help" -o "${arg}" = "-H" ]; then
        help
    elif [ "${arg//[^0-9]/}" = "$arg" -a "$TIME_WARP" = 1 ]; then
        TIME_WARP=$arg
        echo "Time warp set to: " $arg
    elif [ "${arg}" = "--just_build" -o "${arg}" = "-J" ] ; then
        JUST_MAKE="yes"
    elif [ "${arg}" = "--sim" -o "${arg}" = "-s" ] ; then
        SIM="SIM"
        echo "Simulation mode ON."
    elif [ "${arg:0:8}" = "--start=" ]; then
        START_POS="${arg#--start=*}"
    elif [ "${arg:0:7}" = "--role=" ] ; then
        START_ACTION="${arg#--role=*}"
    elif [ "${arg:0:6}" = "--cid=" ] ; then
        CID="${arg#--cid=*}"
        CID=$(printf "%03d" $CID)
    elif [ "${arg}" = "--auto" -o "${arg}" = "-a" ]; then
        AUTO_LAUNCHED="yes" 
    elif [ "${arg:0:10}" = "--logpath=" ]; then
        LOGPATH="${arg#--logpath=*}"
    elif [ "${arg}" = "--verbose" -o "${arg}" = "-V" ]; then
	VERBOSE="yes"
    else
        echo "$ME Bad Arg:" $arg
	exit 1
    fi
done

echo $LOGPATH

if [ "${VTEAM}" = "red" ]; then
    OPFOR="blue"
    OPFOR_ZONE=$BLUE_ZONE
    HFLD=$HOME_IS_NORTH
elif [ "${VTEAM}" = "blue" ]; then
    OPFOR="red"
    OPFOR_ZONE=$RED_ZONE
    HFLD=$HOME_IS_SOUTH
fi
   
#---------------------------------------------------------------
#  Part 4: If verbose, show vars and confirm before launching
#---------------------------------------------------------------
echo "launch_suveyor.sh [$VERBOSE]"
if [ "${VERBOSE}" != "" ]; then 
    echo "=================================="
    echo "   launch_surveyor.sh SUMMARY     "
    echo "=================================="
    echo "$ME"
    echo "CMD_ARGS =      [${CMD_ARGS}]     "
    echo "TIME_WARP =     [${TIME_WARP}]    "
    echo "JUST_MAKE =     [${JUST_MAKE}]    "
    echo "SIM =           [${SIM}]          "
    echo "----------------------------------"
    echo "IP_ADDR =       [${IP_ADDR}]      "
    echo "MOOS_PORT =     [${MOOS_PORT}]    "
    echo "PSHARE_PORT =   [${PSHARE_PORT}]  "
    echo "----------------------------------"
    echo "SHORE_IP =      [${SHORE_IP}]     "
    echo "SHORE_PSHARE =  [${SHORE_PSHARE}] "
    echo "VNAME =         [${VNAME}]        "
    echo "VTEAM =         [${TEAM}]         "
    echo "COLOR =         [${COLOR}]        "
    echo "----------------------------------"
    echo "XMODE =         [${XMODE}]        "
    echo "FSEAT_IP =      [${FSEAT_IP}]     "
    echo "----------------------------------"
    echo "REGION =        [${REGION}]       "
    echo "WANG =          [${WANG}]         "
    echo "DANG =          [${DANG}]         "
    echo "RED_ZONE =      [${RED_ZONE}]     "
    echo "BLUE_ZONE =     [${BLUE_ZONE}]    "
    echo "START_POS =     [${START_POS}]    "
    echo -n "Hit any key to continue with launching ${VNAME}"
    read ANSWER
fi

echo "Assembling MOOS file targ_${RNAME}.moos"

#-------------------------------------------------------
#  Part 2: Create the .moos and .bhv files.
#-------------------------------------------------------

nsplug meta_surveyor.moos targ_${RNAME}.moos -f \
    VNAME=$VNAME                RNAME=$RNAME         \
    VPORT=$VPORT                VR_PORT=$VR_PORT     \
    SHARE_LISTEN=$SHARE_LISTEN  WARP=$TIME_WARP      \
    SHORE_LISTEN=$SHORE_LISTEN  SHORE_IP=$SHORE_IP   \
    SURVEYOR_IP=$SURVEYOR_IP    VTEAM=$VTEAM         \
    HOSTIP_FORCE="localhost"    VMODEL=$VMODEL       \
    LOITER_POS=$LOITER_POS      VARIATION=$VARIATION \
    START_POS=$START_POS        VTYPE="kayak"        \
    CID=$CID                    LOGPATH=$LOGPATH     \
    OPFOR=$OPFOR         \
    OPFOR_ZONE=$OPFOR_ZONE      $SIM $HFLD $FLDS

echo "Assembling BHV file targ_${RNAME}.bhv"
nsplug meta_surveyor.bhv targ_${RNAME}.bhv -f  \
    TRAIL_RANGE=$TRAIL_RANGE       \
    TRAIL_ANGLE=$TRAIL_ANGLE       \
    VTEAM=$VTEAM                   \
    VNAME=$VNAME                   \
    RNAME=$RNAME                   \
    START_ACTION=$START_ACTION     \
    $HFLD $FLDS

if [ ${JUST_MAKE} = "yes" ] ; then
    echo "Files made; vehicle not launched; exiting per request."
    exit 0
fi

#-------------------------------------------------------
#  Part 3: Launch the processes
#-------------------------------------------------------

echo "Launching $RNAME MOOS Community "
pAntler targ_${RNAME}.moos >& /dev/null &

#---------------------------------------------------------------
#  Part 7: If launched from script, we're done, exit now
#---------------------------------------------------------------
if [ "${AUTO_LAUNCHED}" = "yes" ]; then
    exit 0
fi

uMAC targ_${RNAME}.moos

echo "Killing all processes ..."
kill -- -$$
echo "Done killing processes."

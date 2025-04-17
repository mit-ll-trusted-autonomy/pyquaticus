#!/bin/bash
#---------------------------------------------------------------
#   Script: launch.sh
#  Mission: jervis-2023
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
CMD_ARGS=""

IP_ADDR="localhost"
MOOS_PORT="9001"
PSHARE_PORT="9201"

SHORE_IP=localhost
SHORE_PSHARE="9300"

TRAIL_RANGE="3"
TRAIL_ANGLE="330"
VTEAM=""
VNAME=""
RNAME=""
VMODEL="M300"
START_POS=""
RETURN_POS=""
RLA_ENABLE="yes"

CID=000 #COMP ID
LOGPATH="./"
FLD_ANG=""
NSFLAGS="-f"

START_ACTION="PROTECT"
ENTRY=""

#-------------------------------------------------------
#  Part 3: Check for and handle command-line arguments
#-------------------------------------------------------
for ARGI; do
    CMD_ARGS+="${ARGI} "
    if [ $ARGI = "--help" -o $ARGI = "-H" -o $ARGI = "-h" ]; then
	echo "$ME [OPTIONS] [time_warp]                   "
	echo "                                            "
	echo "Options (vehicle name):                     "
	echo "  NOTE: On MIT Herons, name is auto-detected.    "
	echo "  --abe, -va, -v1       Vehicle name is Abe      "
	echo "  --ben, -vb, -v2       Vehicle name is Ben      "
	echo "  --cal, -vc, -v3       Vehicle name is Cal      "
	echo "  --deb, -vd, -v4       Vehicle name is Deb      "
	echo "  --eve, -ve, -v5       Vehicle name is Eve      "
	echo "  --fin, -vf, -v6       Vehicle name is Fin      "
	echo "  --max, -vm, -v7       Vehicle name is Max      "
	echo "  --ned, -vn, -v8       Vehicle name is Ned      "
	echo "  --oak, -vo, -v9       Vehicle name is Oak      "
	echo "                                                 "
	echo "Options (vehicle role):                          "
	echo "  --red_one, -r1   Vehicle role is Red-One    "
	echo "  --red_two, -r2   Vehicle role is Red-Two    "
    echo "  --red_two, -r3   Vehicle role is Red-Two    "
	echo "  --blue_one, -b1  Vehicle role is Blue-One   "
	echo "  --blue_two, -b2  Vehicle role is Blue-Two   "
    echo "  --blue_two, -b3  Vehicle role is Blue-Three "
	echo "                                              "
	echo "Options (other):                              "
	echo "  --role,           Startup roles: CONTROL or ATTACK/DEFEND _E/_MED."
	echo "  --sim, -s         Simulation mode."
	echo "  --shore=<localhost>                            "
    echo "    IP address location of shoreside             "
	echo "    Shortcut: --sip same as --shore=192.168.1.199"
	echo "  --ip=<localhost>                               " 
	echo "    Force pHostInfo to use this IP Address       "
	echo "  --entry=<entry>                                " 
	echo "  --no_rla          Disable RLAgent behavior     " 
	echo "  --start=<X,Y,H>                                " 
	echo "    Start position chosen by script launching    "
	echo "    this script (to ensure separation)           "
	echo "  --logpath=        Log path                     "
	echo "  --ang=<fld_ang>                                "
	echo "  --just_make, -j   Just make targ files.        "
	echo "  --help, -H,-h     Display this message.        "
	echo "  --verbose,-V,-v   Verbose launch.              "
	exit 0
    elif [ $ARGI = "-va" -o $ARGI = "-v1" -o $ARGI = "--abe" ]; then
        FSEAT_IP=192.168.14.1;   VNAME="abe";  IP_ADDR=192.168.14.100
    elif [ $ARGI = "-vb" -o $ARGI = "-v2" -o $ARGI = "--ben" ]; then
        FSEAT_IP=192.168.15.1;   VNAME="ben";  IP_ADDR=192.168.15.100
    elif [ $ARGI = "-vc" -o $ARGI = "-v3" -o $ARGI = "--cal" ]; then
        FSEAT_IP=192.168.16.1;   VNAME="cal";  IP_ADDR=192.168.16.100
    elif [ $ARGI = "-vd" -o $ARGI = "-v4" -o $ARGI = "--deb" ]; then
        FSEAT_IP=192.168.17.1;   VNAME="deb";  IP_ADDR=192.168.17.100
    elif [ $ARGI = "-ve" -o $ARGI = "-v5" -o $ARGI = "--eve" ]; then
        FSEAT_IP=192.168.18.1;   VNAME="eve";  IP_ADDR=192.168.18.100
    elif [ $ARGI = "-vf" -o $ARGI = "-v6" -o $ARGI = "--fin" ]; then
        FSEAT_IP=192.168.19.1;   VNAME="fin";  IP_ADDR=192.168.19.100
    elif [ $ARGI = "-vm" -o $ARGI = "-v7" -o $ARGI = "--max" ]; then
        FSEAT_IP=192.168.20.1;   VNAME="max";  IP_ADDR=192.168.20.100
    elif [ $ARGI = "-vn" -o $ARGI = "-v8" -o $ARGI = "--ned" ]; then
        FSEAT_IP=192.168.21.1;   VNAME="ned";  IP_ADDR=192.168.21.100
    elif [ $ARGI = "-vo" -o $ARGI = "-v9" -o $ARGI = "--oak" ]; then
        FSEAT_IP=192.168.22.1;  VNAME="oak";  IP_ADDR=192.168.22.100


    elif [ "${ARGI}" = "-r1" -o "${ARGI}" = "--red_one" ]; then
        PSHARE_PORT="9311";         VTEAM="red"
        SHARE_ACTION="DEFEND_E";    RNAME="red_one"
        MOOS_PORT="9011"               
    elif [ "${ARGI}" = "-r2" -o "${ARGI}" = "--red_two" ]; then
        PSHARE_PORT="9312";         VTEAM="red"
        SHARE_ACTION="ATTACK_MED";  RNAME="red_two"
        MOOS_PORT="9012"
    elif [ "${ARGI}" = "-r3" -o "${ARGI}" = "--red_three" ]; then
        PSHARE_PORT="9313";         VTEAM="red"
        SHARE_ACTION="ATTACK_MED";    RNAME="red_three"
        MOOS_PORT="9013"               
    elif [ "${ARGI}" = "-b1" -o "${ARGI}" = "--blue_one" ]; then
        PSHARE_PORT="9315";         VTEAM="blue"
        SHARE_ACTION="DEFEND_MED";  RNAME="blue_one"
        MOOS_PORT="9015"               
    elif [ "${ARGI}" = "-b2" -o "${ARGI}" = "--blue_two" ]; then
        PSHARE_PORT="9316";         VTEAM="blue"
        SHARE_ACTION="ATTACK_E";    RNAME="blue_two"
        MOOS_PORT="9016"
    elif [ "${ARGI}" = "-b3" -o "${ARGI}" = "--blue_three" ]; then
        PSHARE_PORT="9317";         VTEAM="blue"
        SHARE_ACTION="ATTACK_MED";    RNAME="blue_three"
        MOOS_PORT="9017"


    elif [ "${ARGI//[^0-9]/}" = "$ARGI" -a "$TIME_WARP" = 1 ]; then
        TIME_WARP=$ARGI
    elif [ $ARGI = "--just_build" -o $ARGI = "-J" -o $ARGI = "-j" ]; then
        JUST_MAKE="yes"
    elif [ "${ARGI}" = "--sim" -o "${ARGI}" = "-s" ]; then
        SIM="SIM"
    elif [ "${ARGI:0:8}" = "--shore=" ]; then
        SHORE_IP="${ARGI#--shore=*}"
    elif [ "${ARGI:0:5}" = "--ip=" ]; then
        IP_ADDR="${ARGI#--ip=*}"
    elif [ "${ARGI}" = "--sip" -o "${ARGI}" = "-sip" ]; then
        SHORE_IP="192.168.1.199"
    elif [ "${ARGI:0:8}" = "--start=" ]; then
        START_POS="${ARGI#--start=*}"
    elif [ "${ARGI:0:7}" = "--role=" ]; then
        START_ACTION="${ARGI#--role=*}"
    elif [ "${ARGI}" = "--auto" -o "${ARGI}" = "-a" ]; then
        AUTO_LAUNCHED="yes" 
    elif [ "${ARGI:0:10}" = "--logpath=" ]; then
        LOGPATH="${ARGI#--logpath=*}"
    elif [ "${ARGI:0:6}" = "--ang=" ]; then
        FLD_ANG="${ARGI}"
    elif [ "${ARGI:0:8}" = "--entry=" ]; then
        ENTRY="_${ARGI#--entry=*}"
    elif [ $ARGI = "--no_rla" ]; then
        RLA_ENABLED="no"
    elif [ $ARGI = "--verbose" -o $ARGI = "-V" -o $ARGI = "-v" ]; then
	VERBOSE="yes"
	NSFLAGS+=" -i"
    else
        echo "$ME Bad Arg: [$ARGI]. Exit Code 1"
	exit 1
    fi
done

if [ "${AUTO_LAUNCHED}" != "yes" ]; then
    ./genop.sh $FLD_ANG > targ_region.txt
fi
source targ_region.txt

if [ "${VTEAM}" = "red" ]; then
    OPFOR="blue"
    OPFOR_ZONE=$BLUE_ZONE
    HFLD=$HOME_IS_NORTH
elif [ "${VTEAM}" = "blue" ]; then
    OPFOR="red"
    OPFOR_ZONE=$RED_ZONE
    HFLD=$HOME_IS_SOUTH
fi

if [ "${RNAME}" = "red_one" ]; then
    RETURN_POS=$XNWF
elif [ "${RNAME}" = "red_two" ]; then
    RETURN_POS=$XNWM
elif [ "${RNAME}" = "blue_one" ]; then
    RETURN_POS=$XSWF
elif [ "${RNAME}" = "blue_two" ]; then
    RETURN_POS=$XSWM
else
    RETURN_POS=$XWT
fi
     
#---------------------------------------------------------------
#  Part 4: Ensure Mandatory Settings Specified
#---------------------------------------------------------------
if [ "${VNAME}" = "" -o "${RNAME}" = "" ]; then
    echo "VNAME and VROLE must be set. See --help "
    echo "RNAME =         [${RNAME}]              "
    echo "VNAME =         [${VNAME}]              "
    exit 1
fi

#---------------------------------------------------------------
#  Part 5: If verbose, show vars and confirm before launching
#---------------------------------------------------------------
if [ "${VERBOSE}" != "" ]; then 
    echo "=================================="
    echo "   launch_vehicle.sh SUMMARY      "
    echo "=================================="
    echo "$ME"
    echo "CMD_ARGS =      [${CMD_ARGS}]     "
    echo "TIME_WARP =     [${TIME_WARP}]    "
    echo "JUST_MAKE =     [${JUST_MAKE}]    "
    echo "SIM =           [${SIM}]          "
    echo "NSFLAGS =       [$NSFLAGS]        "
    echo "FLD_ANG =       [${FLD_ANG}]      "
    echo "----------------------------------"
    echo "IP_ADDR =       [${IP_ADDR}]      "
    echo "MOOS_PORT =     [${MOOS_PORT}]    "
    echo "PSHARE_PORT =   [${PSHARE_PORT}]  "
    echo "----------------------------------"
    echo "SHORE_IP =      [${SHORE_IP}]     "
    echo "SHORE_PSHARE =  [${SHORE_PSHARE}] "
    echo "RNAME =         [${RNAME}]        "
    echo "VNAME =         [${VNAME}]        "
    echo "VTEAM =         [${VTEAM}]        "
    echo "ENTRY =         [${ENTRY}]        "
    echo "RLA_ENABLED =   [${RLA_ENABLED}]  "
    echo "----------------------------------"
    echo "FSEAT_IP =      [${FSEAT_IP}]     "
    echo "LOGPATH =       [${LOGPATH}]      "
    echo "----------------------------------"
    echo "REGION =        [${REGION}]       "
    echo "WANG =          [${WANG}]         "
    echo "DANG =          [${DANG}]         "
    echo "RED_ZONE =      [${RED_ZONE}]     "
    echo "BLUE_ZONE =     [${BLUE_ZONE}]    "
    echo "START_POS =     [${START_POS}]    "
    echo "RETURN_POS =    [${RETURN_POS}]   "
    echo -n "Hit any key to continue with launching ${VNAME}"
    read ANSWER
fi

echo "Assembling MOOS file targ_${RNAME}.moos"

#-------------------------------------------------------
#  Part 6: Create the .moos and .bhv files.
#-------------------------------------------------------
nsplug meta_vehicle.moos targ_${RNAME}.moos $NSFLAGS WARP=$TIME_WARP  \
       PSHARE_PORT=$PSHARE_PORT    VNAME=$VNAME         \
       IP_ADDR=$IP_ADDR            SHORE_IP=$SHORE_IP   \
       SHORE_PSHARE=$SHORE_PSHARE  MOOS_PORT=$MOOS_PORT \
       VTEAM=$VTEAM                RNAME=$RNAME         \
       START_POS=$START_POS        VTYPE="kayak"        \
       CID=$CID                    LOGPATH=$LOGPATH     \
       OPFOR_ZONE=$OPFOR_ZONE      OPFOR=$OPFOR         \
       $SIM $HFLD $FLDS            FSEAT_IP=$FSEAT_IP

echo "Assembling BHV file targ_${RNAME}.bhv"
nsplug "meta_vehicle${ENTRY}.bhv" targ_${RNAME}.bhv $NSFLAGS \
       TRAIL_RANGE=$TRAIL_RANGE    VTEAM=$VTEAM   \
       TRAIL_ANGLE=$TRAIL_ANGLE    VNAME=$VNAME   \
       START_ACTION=$START_ACTION  RNAME=$RNAME   \
       RETURN_POS=$RETURN_POS      $HFLD $FLDS    \
       RLA_ENABLED=$RLA_ENABLED

#-------------------------------------------------------
#  Part 7: Possibly exit now if we're just building† targ files
#-------------------------------------------------------
if [ ${JUST_MAKE} = "yes" ]; then
    echo "Vehicle targ files made. Nothing Launched per request"
    exit 0
fi

#-------------------------------------------------------
#  Part 8: Launch the processes
#-------------------------------------------------------

echo "Launching $RNAME MOOS Community "
pAntler targ_${RNAME}.moos >& /dev/null &

#---------------------------------------------------------------
#  Part 9: If launched from script, we're done, exit now
#---------------------------------------------------------------
if [ "${AUTO_LAUNCHED}" = "yes" ]; then
    exit 0
fi

uMAC targ_${RNAME}.moos

sleep 2 # Give them a chance to exit with grace
echo "Killing all processes ..."
kill -- -$$
echo "Done killing processes."

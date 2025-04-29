#!/bin/bash -e
#---------------------------------------------------------------
#   Script: launch.sh
#  Mission: jervis-2023
#   Author: Mike Benjamin
#   LastEd: Oct 2023
#-------------------------------------------------------
#  Part 1: Set global var defaults
#-------------------------------------------------------
ME=`basename "$0"`
TIME_WARP=1
JUST_MAKE="no"
VERBOSE=""
AUTO_LAUNCHED="no"
CMD_ARGS=""

IP_ADDR="localhost"
MOOS_PORT="9000"
PSHARE_PORT="9300"

VTEAM1="red"
VTEAM2="blue"

CID=000 # Competiton id
LOGPATH=./
FLD_ANG=""
VNAMES="red_one:red_two:red_three:blue_one:blue_two:blue_three"

#-------------------------------------------------------
#  Part 2: Check for and handle command-line arguments
#-------------------------------------------------------
for ARGI; do
    CMD_ARGS+="${ARGI} "
    if [ "${ARGI}" = "--help" -o "${ARGI}" = "-h" ] ; then
	echo "$ME [OPTIONS] [time_warp]                     "
        echo "  --help, -h                                  "
        echo "  --shore-port=<PORT>  Shore pShare port      "
	echo "                       Default $SHORE_LISTEN  "
        echo "  --ip=<localhost>     Shore IP address       "
        echo "  --mport=<9000>       Shoreside MOOS port    "
        echo "  --pshare=<9300>      The pShare Listen port "
	echo "                                              "
        echo "  --cid=<ID>           Competition id (for log file)"
        echo "  --logpath=<PATH>     Log path              "
	echo "  --ang=<field_angle>                         "
        echo "  --just_make, -j      Just make targ files   "
        echo "  --verbose, -v        Verbose Launch         "
        echo "  --auto, -a                                  "
        echo "     Auto-launched by a script.               "
        echo "     Will not launch uMAC as the final step.  "
	echo "  --vnames=<vname:vname:vname>                " 
	echo "    List of anticipated vehicle names         "
        exit 0
    elif [ "${ARGI//[^0-9]/}" = "$ARGI" -a "$TIME_WARP" = 1 ]; then
        TIME_WARP=$ARGI
    elif [ "${ARGI}" = "--just_make" -o "${ARGI}" = "-j" ]; then
        JUST_MAKE="yes"
    elif [ $ARGI = "--verbose" -o $ARGI = "-V" -o $ARGI = "-v" ]; then
	VERBOSE="yes"
    elif [ "${ARGI}" = "--auto" -o "${ARGI}" = "-a" ]; then
        AUTO_LAUNCHED="yes"

    elif [ "${ARGI:0:5}" = "--ip=" ]; then
        IP_ADDR="${ARGI#--ip=*}"
        FORCE_IP="yes"
    elif [ "${ARGI}" = "-sip" -o "${ARGI}" = "--sip" ]; then
	IP_ADDR="192.168.1.37"
        FORCE_IP="yes"

    elif [ "${ARGI:0:11}" = "--shore-ip=" ]; then
        SHORE_IP="${ARGI#--shore-ip=*}"
    elif [ "${ARGI:0:13}" = "--shore-port=" ]; then
        PSHARE_PORT=${ARGI#--shore-port=*}
    elif [ "${ARGI:0:9}" = "--pshare=" ]; then
        PSHARE_PORT=${ARGI#--pshare=*}
    elif [ "${ARGI:0:6}" = "--cid=" ]; then
        CID="${ARGI#--cid=*}"
        CID=$(printf "%03d" $CID)
    elif [ "${ARGI:0:10}" = "--logpath=" ]; then
        LOGPATH="${ARGI#--logpath=*}"
    elif [ "${ARGI:0:6}" = "--ang=" ]; then
        FLD_ANG="${ARGI}"
    elif [ "${ARGI:0:9}" = "--vnames=" ]; then
        VNAMES="${ARGI#--vnames=*}"
    else
        echo "$ME: Bad Arg: [$ARGI] Exit code 1."
        exit 1
    fi
done

if [ "${AUTO_LAUNCHED}" != "yes" ]; then
    ./genop.sh $FLD_ANG > targ_region.txt
fi
source targ_region.txt

#---------------------------------------------------------------
#  Part 3: If verbose, show vars and confirm before launching
#---------------------------------------------------------------
if [ "${VERBOSE}" != "" ]; then 
    echo "======================================================"
    echo "        launch_shoreside.sh SUMMARY                   "
    echo "======================================================"
    echo "$ME"
    echo "CMD_ARGS =      [${CMD_ARGS}]      "
    echo "TIME_WARP =     [${TIME_WARP}]     "
    echo "AUTO_LAUNCHED = [${AUTO_LAUNCHED}] "
    echo "JUST_MAKE =     [${JUST_MAKE}]     "
    echo "FLD_ANG =       [${FLD_ANG}]       "
    echo "---------------------------------- "
    echo "IP_ADDR =       [${IP_ADDR}]       "
    echo "MOOS_PORT =     [${MOOS_PORT}]     "
    echo "PSHARE_PORT =   [${PSHARE_PORT}]   "
    echo "---------------------------------- "
    echo "XMODE =         [${XMODE}]         "
    echo "REGION =        [${REGION}]        "
    echo "VNAMES =        [${VNAMES}]        "
    echo "VTEAM1 =        [${VTEAM1}]        "
    echo "VTEAM2 =        [${VTEAM2}]        "
    echo "---------------------------------- "
    echo "BLUE_ZONE =     [${BLUE_ZONE}]     "
    echo "RED_ZONE =      [${RED_ZONE}]      "
    echo "BLUE_FLAG =     [${BLUE_FLAG}]     "
    echo "RED_FLAG =      [${RED_FLAG}]      "
    echo -n "Hit any key to continue with launching SHORESIDE"
    read ANSWER
fi

#-------------------------------------------------------
#  Part 4: Create the Shoreside MOOS file
#-------------------------------------------------------
nsplug meta_shoreside.moos targ_shoreside.moos -f WARP=$TIME_WARP  \
       IP_ADDR=$IP_ADDR     PSHARE_PORT=$PSHARE_PORT  \
       MOOS_PORT="9000"     VNAMES=$VNAMES            \
       LOGPATH=$LOGPATH     CID=$CID                  \
       VTEAM1=$VTEAM1       VTEAM2=$VTEAM2            \
       DANGX=$DANGX         $FLDS
       
if [ ! -e targ_shoreside.moos ]; then
    echo "no targ_shoreside.moos";
    exit 1;
fi

#-------------------------------------------------------
#  Part 5: Possibly exit now if we're just building targ files
#-------------------------------------------------------
if [ ${JUST_MAKE} = "yes" ]; then
    echo "Shoreside targ files made. Nothing launched per request."
    exit 0
fi

#-------------------------------------------------------
#  Part 6: Launch the Shoreside
#-------------------------------------------------------
echo "Launching shoreside MOOS Community (WARP=$TIME_WARP)"
pAntler targ_shoreside.moos >& /dev/null &
echo "Done Launching Shoreside "

#---------------------------------------------------------------
#  Part 7: If launched from script, we're done, exit now
#---------------------------------------------------------------
if [ "${AUTO_LAUNCHED}" = "yes" ]; then
    exit 0
fi

uMAC targ_shoreside.moos

sleep 2 # Give them a chance to exit with grace
echo "Killing all processes ... "
kill -- -$$
echo "Done killing processes.   "

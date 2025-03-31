#!/bin/bash
#------------------------------------------------------
#   Script: launch.sh
#  Mission: jervis-2023
#   Author: Mike Benjamin
#   LastEd: Oct 2023
#------------------------------------------------------
#  Part 1: Set global var defaults
#------------------------------------------------------
ME=`basename "$0"`
TIME_WARP=1
JUST_MAKE=""
VERBOSE=""

CMD_ARGS=""
NO_SHORESIDE=""
LOGPATH=""
RLA=""
BENTRY=""
RENTRY=""
FLD_ANG=""
VNAMES="red_one:red_two:red_three:blue_one:blue_two:blue_three"

#-------------------------------------------------------
#  Part 2: Check for and handle command-line arguments
#-------------------------------------------------------
for ARGI; do
    CMD_ARGS+="${ARGI} "
    if [ "${ARGI}" = "--help" -o "${ARGI}" = "-h" ]; then
	echo "$ME [OPTIONS] [time_warp]                     "
	echo "                                              "
	echo "Options:                                      "
	echo "  --help, -h                                  "
	echo "    Show this help message                    "
	echo "  --just_make, -j                             "
	echo "    Just make targ files, no launch           "
	echo "  --verbose, -v                               "
	echo "    Increase verbosity, confirm before launch "
	echo "  --no_shoreside, -ns                         "
	echo "  --logpath=<path>                            "
	echo "  --ang=<field_angle>                         "
	echo "  --no_rla                                    "
	echo "    Do not use RLAgent behavior               "
	echo "                                              "
	echo "  --bentry=<entry>                            "
	echo "    Example: --bentry=mitx                    "
	echo "  --rentry=<entry>                            "
	echo "    Example: --rentry=team                    "
	exit 0
    elif [ "${ARGI//[^0-9]/}" = "$ARGI" -a "$TIME_WARP" = 1 ]; then
        TIME_WARP=$ARGI
    elif [ "${ARGI}" = "--just_make" -o "${ARGI}" = "-j" ]; then
        JUST_MAKE="-j"
    elif [ $ARGI = "--verbose" -o $ARGI = "-V"  -o $ARGI = "-v" ]; then
	VERBOSE="-V"
    elif [ "${ARGI}" = "--no_shoreside" -o "${ARGI}" = "-ns" ]; then
        NO_SHORESIDE="true"
    elif [ "${ARGI}" = "--no_rla" ]; then
        RLA=$ARGI
    elif [ "${ARGI:0:10}" = "--logpath=" ]; then
        LOGPATH="${ARGI#--logpath=*}"
    elif [ "${ARGI:0:6}" = "--ang=" ]; then
        FLD_ANG="${ARGI}"
    elif [ "${ARGI:0:9}" = "--bentry=" ]; then
        BENTRY="--entry=${ARGI#--bentry=*}"
    elif [ "${ARGI:0:9}" = "--rentry=" ]; then
        RENTRY="--entry=${ARGI#--rentry=*}"
    else
        echo "$ME Bad Arg: [$ARGI]. Exit Code 1"
	exit 1
    fi
done

./genop.sh $FLD_ANG > targ_region.txt
source targ_region.txt

if [ -n "${LOGPATH}" ]; then
  LOGDIR=--logpath=${LOGPATH}
fi

#-----------------------------------------------
# Part 3: Set the Vehicle random positions
#-----------------------------------------------
POS=(`pickpos --amt=3 --polygon=$RED_ZONE --hdg=$CCT,0 --format=terse` )
ABE_POS=${POS[0]}
BEN_POS=${POS[1]}
CAL_POS=${POS[2]}
POS=(`pickpos --amt=3 --polygon=$BLUE_ZONE --hdg=$CCT,0 --format=terse`)
DEB_POS=${POS[0]}
EVE_POS=${POS[1]}
FIN_POS=${POS[2]}

#---------------------------------------------------------------
#  Part 4: If verbose, show vars and confirm before launching
#---------------------------------------------------------------
if [ "${VERBOSE}" != "" ]; then 
    echo "======================================="
    echo "  launch.sh SUMMARY                    "
    echo "======================================="
    echo "CMD_ARGS =  [${CMD_ARGS}]              "
    echo "TIME_WARP = [${TIME_WARP}]             "
    echo "JUST_MAKE = [${JUST_MAKE}]             "
    echo "---------------------------------------"
    echo "WANG =      [${WANG}]                  "
    echo "DANG =      [${DANG}]                  "
    echo "FLD_ANG =   [${FLD_ANG}]               "
    echo "---------------------------------------"
    echo "BENTRY =    [${BENTRY}]                "
    echo "RENTRY =    [${RENTRY}]                "
    echo "LOGDIR =    [${LOGDIR}]                "
    echo "RLA =       [${RLA}]                   "
    echo "---------------------------------------"
    echo "VNAMES = [${VNAMES}]                   "
    echo -n "Hit the RETURN key to continue with launching"
    read ANSWER
fi

#-------------------------------------------------------
#  Part 5: Launching vehicles
#-------------------------------------------------------
VARGS=" $VERBOSE $JUST_MAKE $TIME_WARP $LOGDIR $FLD_ANG $RLA --sim --auto  --shore=localhost --ip=localhost"
BVARGS="$VARGS $BENTRY"
RVARGS="$VARGS $RENTRY"

echo "Launching Abe Red-One"
./launch_vehicle.sh -va -r1 $RVARGS --start=$ABE_POS --role=CONTROL
echo "Launching Ben Red-Two"
./launch_vehicle.sh -vb -r2 $RVARGS --start=$BEN_POS --role=CONTROL
echo "Launching Cal Red-Three"
./launch_vehicle.sh -vc -r3 $RVARGS --start=$CAL_POS --role=CONTROL
echo "Launching Deb Blue-One"
./launch_vehicle.sh -vd -b1 $BVARGS --start=$DEB_POS --role=CONTROL
echo "Launching Eve Blue-Two"
./launch_vehicle.sh -ve -b2 $BVARGS --start=$EVE_POS --role=CONTROL
echo "Launching Fin Blue-Three"
./launch_vehicle.sh -vf -b3 $BVARGS --start=$FIN_POS --role=CONTROL



#-------------------------------------------------------
#  Part 6: Launching shoreside
#-------------------------------------------------------
SARGS=" $VERBOSE $JUST_MAKE $TIME_WARP $LOGDIR $FLD_ANG --auto"
SARGS+=" --vnames=$VNAMES"

if [[ -z $NO_SHORESIDE ]]; then
  ./launch_shoreside.sh $SARGS
fi

#-------------------------------------------------------
#  Part 7: Possibly exit now if we're just building targ files
#-------------------------------------------------------
if [[ ${JUST_MAKE} != "" ]]; then
    exit 0
fi

#-------------------------------------------------------
#  Part 8: Launching uMAC
#-------------------------------------------------------
uMAC targ_shoreside.moos

#-------------------------------------------------------
#  Part 9: Killing all processes launched from script
#-------------------------------------------------------
echo "Killing Simulation..."

kill -- -$$
# sleep is to give enough time to all processes to die
sleep 3
echo "All processes killed"

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
#VNAMES="red_one:red_two:blue_one:blue_two"
VNAMES="red_one:blue_one"

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

./jerop.sh $FLD_ANG > region_info.txt
source region_info.txt

if [ -n "${LOGPATH}" ]; then
  LOGDIR=--logpath=${LOGPATH}
fi

#-----------------------------------------------
# Part 3: Set the Vehicle random positions
#-----------------------------------------------
POS=(`pickpos --amt=1 --polygon=$RED_ZONE --hdg=$CCT,0 --format=terse` )
SCO_POS=${POS[0]}
#THO_POS=${POS[1]}
POS=(`pickpos --amt=1 --polygon=$BLUE_ZONE --hdg=$CCT,0 --format=terse`)
URS_POS=${POS[0]}
#VAL_POS=${POS[1]}

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
    echo "SCOTT_POS    = [${SCO_POS}]            "
    echo "THOMAS_POS   = [${THO_POS}]            "
    echo "URSULA_POS   = [${URS_POS}]            "
    echo "VALHALLA_POS = [${VAL_POS}]            "
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

cd ./surveyor
echo "Launching Scott Red-One[$RVARGS]"
./launch_surveyor.sh -vs -r1 $RVARGS --start=$SCO_POS --role=ATTACK_E --no_rla 
#echo "Launching Thomas Red-Two"
#./launch_surveyor.sh -vt -r2 $RVARGS --start=$THO_POS --role=DEFEND_E 
echo "Launching Ursula Blue-One"
./launch_surveyor.sh -vu -b1 $BVARGS --start=$URS_POS --role=ATTACK_MED --no_rla
#echo "Launching Valhall Blue-Two"
#./launch_surveyor.sh -vv -b2 $BVARGS --start=$VAL_POS --role=DEFEND_MED 
cd ..


#-------------------------------------------------------
#  Part 6: Launching shoreside
#-------------------------------------------------------
SARGS=" $VERBOSE $JUST_MAKE $TIME_WARP $LOGDIR $FLD_ANG --auto"
SARGS+=" --vnames=$VNAMES"

if [[ -z $NO_SHORESIDE ]]; then
  cd ./shoreside
  ./launch_shoreside.sh $SARGS
  cd ..
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
uMAC shoreside/targ_shoreside.moos

#-------------------------------------------------------
#  Part 9: Killing all processes launched from script
#-------------------------------------------------------
echo "Killing Simulation..."

kill -- -$$
# sleep is to give enough time to all processes to die
sleep 3
echo "All processes killed"

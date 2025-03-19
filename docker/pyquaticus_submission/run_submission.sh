#!/bin/bash
# This script is meant to be run from the root of the pyquaticus_submission directory
# Here is an example of how to run it:
# ./run_submission.sh --boat-name=u --boat-role=blue_one --logpath=./logs/ --time-warp=4 --num-players=1 --sim
SIMULATION="false"
INTERACTIVE=false

HELP=false

# Check for -i and -h flags first
for arg in "$@"; do
    if [ "$arg" == "-i" ]; then
        INTERACTIVE=true
    elif [ "$arg" == "-h" ]; then
        HELP=true
    fi
done

if $HELP; then
    echo "Usage: $0 [OPTIONS]"
    echo ""
    echo "Options:"
    echo "  -i               Interactive mode. Will prompt for each setting."
    echo "  -h               Show this help message and exit."
    echo "  --boat-name=NAME Specify the boat name (e.g., u, v, s, t)."
    echo "  --boat-role=ROLE Specify the boat role (e.g., red_one, blue_one, red_two, blue_two)."
    echo "  --logpath=PATH   Specify the log path relative to this directory (default is ./logs/)."
    echo "  --time-warp=WARP Specify the time warp (default is 4)."
    echo "  --num-players=NUM Specify the number of players per team (default is 2)."
    echo "  --sim            Indicate that this is a simulation."
    exit 0
fi

# If in interactive mode, prompt the user for each variable
if $INTERACTIVE; then
    read -p "Enter boat name (e.g., u, v, s, t): " BOAT_NAME
    read -p "Enter boat role (e.g., red_one, blue_one, red_two, blue_two): " BOAT_ROLE
    read -p "Enter log path (default is ./logs/): " LOGPATH
    read -p "Enter time warp (default 4): " TIME_WARP
    read -p "Enter number of players per team (default 2): " NUM_PLAYERS
    read -p "Is it a simulation? (yes/no, default no): " SIM_INPUT
    
    # Convert simulation input from the prompt to true/false for the command
    if [ "$SIM_INPUT" == "yes" ]; then
        SIMULATION="true"
    fi
else
    # Process command line arguments
    for arg in "$@"; do
        case $arg in
            --boat-name=*)
            BOAT_NAME="${arg#*=}"
            ;;
            --boat-role=*)
            BOAT_ROLE="${arg#*=}"
            ;;
            --logpath=*)
            LOGPATH="${arg#*=}"
            ;;
            --time-warp=*)
            TIME_WARP="${arg#*=}"
            ;;
            --num-players=*)
            NUM_PLAYERS="${arg#*=}"
            ;;
            --sim)
            SIMULATION="true"
            ;;
        esac
    done
fi

# Default values if the user left any field blank or didn't provide them as arguments
TIME_WARP=${TIME_WARP:-4}
NUM_PLAYERS=${NUM_PLAYERS:-2}
LOGPATH=${LOGPATH:-./logs/}

# Run your command using the provided values
./launch_pyquaticus.sh --boat-name $BOAT_NAME --boat-role $BOAT_ROLE --logpath=$LOGPATH --time-warp $TIME_WARP  --num-players $NUM_PLAYERS $([ "$SIMULATION" == "true" ] && echo "--sim")

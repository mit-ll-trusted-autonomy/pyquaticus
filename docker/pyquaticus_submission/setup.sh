#!/bin/bash

# Any other packages that you want to install using pip can be added here
echo "Setting up..."

# Add the RL_agent behavior to the jervis mission
#UPDATE STHIS
FILE="/home/john/moos-ivp-aquaticus/missions/jervis-2023/surveyor/meta_surveyor.bhv"


if ! grep -q "name\s*=\s*BHV_RLAgent" "$FILE"; then
    echo "BHV_RLAgent not found. Adding it to $FILE..."


    cat <<EOF >> "$FILE"

Behavior = BHV_RLAgent
{
  name                    = rlagent_attacker
  pwt                     = 50
  perpetual               = true

  condition               = (MODE == CONTROLLED)
  runflag                 = BOT_DIALOG_STATUS=Controlled
}
EOF

    echo "BHV_RLAgent added."
else
    echo "BHV_RLAgent is already present in $FILE."
fi

#!/bin/bash

# DISTRIBUTION STATEMENT A. Approved for public release. Distribution is unlimited.
#
# This material is based upon work supported by the Under Secretary of Defense for
# Research and Engineering under Air Force Contract No. FA8702-15-D-0001. Any opinions,
# findings, conclusions or recommendations expressed in this material are those of the
# author(s) and do not necessarily reflect the views of the Under Secretary of Defense
# for Research and Engineering.
#
# (C) 2023 Massachusetts Institute of Technology.
#
# The software/firmware is provided to you on an As-Is basis
#
# Delivered to the U.S. Government with Unlimited Rights, as defined in DFARS
# Part 252.227-7013 or 7014 (Feb 2014). Notwithstanding any copyright notice, U.S.
# Government rights in this work are defined by DFARS 252.227-7013 or DFARS
# 252.227-7014 as detailed above. Use of this work other than as specifically
# authorized by the U.S. Government may violate any copyrights that exist in this
# work.

# SPDX-License-Identifier: BSD-3-Clause

# get top-level directory of repo
REPODIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"
ENV_NAME="$REPODIR/env"


usage() {
    echo "$0 [light|full]"
    echo -e "\t\t     light: build the lightweight environment -- only what's needed to run the gymnasium"
    echo -e "\t\t     full: build the full environment -- includes RLlib and Pytorch"
    exit 0
}

if command -v mamba &> /dev/null; then
    CONDATYPE="mamba"
    MAMBABIN=`which mamba`
    MAMBA_ROOT_PREFIX=${MAMBABIN%condabin/mamba}
    source $MAMBA_ROOT_PREFIX/etc/profile.d/mamba.sh
elif command -v conda &> /dev/null; then
    CONDATYPE="conda"
else
    echo "Missing conda. Please install the latest Anaconda or Miniconda"
    exit
fi
echo "Using $CONDATYPE"

if [[ "$1" == "--help" ]]; then
    usage
fi

if [[ $# == 1 && $1 == "light" ]]; then
    ENV_NAME="${ENV_NAME}-light"
elif [[ $# == 1 && $1 == "full" ]]; then
    ENV_NAME="${ENV_NAME}-full"
else
    usage
fi
ENVTYPE="$1"

cd $REPODIR
echo "Creating conda virtual env at: ${ENV_NAME}"
eval "$(conda shell.bash hook)"
${CONDATYPE} create --prefix=${ENV_NAME} -y python=3.10
conda activate ${ENV_NAME}
conda install -y -c conda-forge libstdcxx-ng
conda env config vars set LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$CONDA_PREFIX/lib/

if [[ "$ENVTYPE" == "light" ]]; then
    pip install -e .
elif [[ "$ENVTYPE" == "full" ]]; then
    # Note: there is an unresolvable dependency conflict between pettingzoo and ray
    # Here is an example error message
    # ERROR: pip's dependency resolver does not currently take into account all the packages that are installed. This behaviour is the source of the following dependency conflicts.
    # pettingzoo 1.23.0 requires gymnasium>=0.28.0, but you have gymnasium 0.26.3 which is incompatible.
    # we ignore that warning and install gymnasium==0.28.1 which works in practice
    # see
    #   https://github.com/ray-project/ray/pull/34696
    #   https://github.com/ray-project/ray/pull/35698
    # instead we just install ray[rllib]==2.4.0 separately and ignore the warning
    # this only works because the current version of pip does *not* fail for dependency conflicts with previously installed packages
    pip install -e .[torch]
    pip install ray[rllib]==2.4.0
    pip install gymnasium==0.28.1
fi

echo ""
echo "You may now activate the ${ENV_NAME} environment with: $CONDATYPE activate ${ENV_NAME}"
echo "Note: if you see an error above about pettingzoo, ray[rllib], and gymnasium dependency conflicts, you may ignore it"
echo "      there is an unresolvable dependency conflict that does not seem to affect our code"
echo "      see the comments at the end of this script ($0) for more info"


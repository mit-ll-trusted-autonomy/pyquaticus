#!/bin/bash

MOOSPID=`ps -ef | egrep '\w\.moos' | awk '{print $2}'`
[[ ! -z $MOOSPID ]] && kill -9 $MOOSPID

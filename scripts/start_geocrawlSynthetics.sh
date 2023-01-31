#!/bin/bash

echo "-----------------------------------------------------------------------------"
echo " "
echo " "
echo " Setting anaconda environment for synthetic datageneration"
echo " "
echo " "
echo " If it doesn't work, I offer limited support........"
echo " "
echo " Sincerely,"
echo " Tom Merrifield"
echo " "
echo "-----------------------------------------------------------------------------"
OPTS=`getopt -o c:n:r:t: --long config:,num_run:,run_id:,test_mode: -n 'parse-options' -- "$@"`

if [ $? != 0 ] ; then echo "Failed parsing options." >&2 ; exit 1 ; fi

eval set -- "$OPTS"

CONFIG="config/example.json"
NUM_RUN=5
RUN_ID=""
TEST_MODE=""
while true; do
  case "$1" in
    -c | --config    ) CONFIG=$2;shift 2 ;;
    -n | --num_run   ) NUM_RUN=$2;shift 2 ;;
    -r | --run_id    ) RUN_ID=$2;shift 2 ;;
    -t | --test_mode ) TEST_MODE=$2;shift 2 ;;
    -- ) shift;break ;;
    * ) break ;;
  esac
done

conda activate synthoseis

echo " "
echo " PATH variable is:"
echo $PATH
echo " "
echo " Environment variables containing PATH:"
env |grep PATH
echo " "
echo " "
echo $PYTHONPATH

if [[ -z "$TEST_MODE" ]]; then
  X_CMD python ../main.py -c $CONFIG -n $NUM_RUN -r $RUN_ID
else
  $X_CMD python ../main.py -c $CONFIG -n $NUM_RUN -r $RUN_ID -t $TEST_MODE
fi

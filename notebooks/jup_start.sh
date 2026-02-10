#!/bin/bash

#. /etc/profile.d/modules.sh
#[ -r "${HOME}/.jupyterrunrc" ] && . ${HOME}/.jupyterrunrc

port=$(($RANDOM % 10000 + 10000))
host=$(hostname)
#unset XDG_RUNTIME_DIR

localhost=$(($RANDOM % 10000 + 10000))
localhost=8891

cd $5
echo Enter $5

echo To enable port forwarding, execute this command in another terminal:
echo ssh -N -v -l $USER -L ${localhost}:$host:$port ansible
echo Then, you can connect to http://localhost:${localhost}
echo

[ "$2" != "(null)" ] && eval module load "${2//:/ }"
#[ "$3" != "(null)" ] && . $3/bin/activate

which python

if [ -x "${4}-lab" ]; then
    $4 lab --no-browser --port=${port} --ip=0.0.0.0
else
    $4 notebook --no-browser --ip=0.0.0.0 --port=$port
fi


#!/bin/bash
# if necessary, load conda environment
#eval "$(conda shell.bash hook)"
#conda activate jupyter
#jupyter lab --no-browser --port=49000 --ip=`hostname`

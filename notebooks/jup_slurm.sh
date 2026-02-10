#!/bin/bash

srcdir=$PWD
prjdir=$PWD

if [ -z "${prjdir}" ]; then
    echo "Please provide a working directory"
    echo "bash jup_slurm.sh <DIR>"
    exit 1
fi

# activate the environment
eval "$(mamba shell hook --shell bash)"
#eval "$(mamba shell.bash hook)"
mamba activate jupyter_torch_cpu

# check jupyter exist
jupyter=$(which jupyter 2> /dev/null)
if [ "$?" -gt 0 ]; then
    echo 'Please activate environment with jupyter' 1>&2
    exit 1
fi

if [ ! -d ${prjdir} ]; then
    echo "cannot find ${prjdir}"
    exit 1
fi

# run script
srun -p qcpu --cpus-per-task=4 --time=72:00:00 --pty bash ${srcdir}/jup_start.sh "$PWD" "$LOADEDMODULES" "$VIRTUAL_ENV" $jupyter $prjdir

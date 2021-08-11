#!/bin/bash

experiment="experiments/$1"

if [ ! -L "experiments/datasets" ];
then
    echo $(pwd)/experiments/datasets
    ln -s "../../datasets";
fi

# $1 is the experiment directory
if [ ! -d $experiment ];
then
    echo "Missing experiments directory $experiment";
elif [ -d "$experiment/results" ] || [ -d "$experiment/summaries" ];
then
	echo "Can't be run with previous results. Please removed the summaries and results directories";
elif [ ! -f "$experiment/pipeline.params" ];
then
    echo "Missing parameters file $experiment/pipeline.params";
else
    [[ -d $experiment/logs ]] || mkdir $experiment/logs
    for dir in results summaries;
    do
        mkdir $experiment/$dir;
    done;

    docker run --rm -v $(pwd)/experiments/datasets:/datasets \
        -v $(pwd)/experiments/$1:/experiment \
        -v ~/.compi:/root/.compi \
        -v ~/.mxnet:/root/.mxnet \
        --gpus device=0 \
        -ti polydeep/object-detection \
        /compi run -p train.xml \
        --params /experiment/pipeline.params \
        -l /experiment/logs \
        -o
fi;

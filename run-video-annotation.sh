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
elif [ ! -f "$experiment/pipeline.params" ];
then
    echo "Missing parameters file $experiment/pipeline.params";
else
     command="xhost +"
     eval $command

     docker run --rm --net=host -e DISPLAY  -v $(pwd)/experiments/datasets:/datasets \
        -v $(pwd)/experiments/$1:/experiment \
        -v $(pwd)/downloads-cache:/downloads-cache \
        -v ~/.compi:/root/.compi \
        -v ~/.mxnet:/root/.mxnet \
        -v $(pwd)/cad:/cad \
        --gpus device=0 \
        -t polydeep/object-detection \
        /compi run -p video-annotation.xml \
        --params /experiment/pipeline.params \
 	    -l /experiment/logs \
        -o
fi;

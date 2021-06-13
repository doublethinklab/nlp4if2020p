#!/bin/bash

if [ -z $1 ]
then
    docker run \
        --rm \
        -v ${PWD}:/ccpalign \
        -w /ccpalign \
        doublethinklab/nlp4if2020p:latest \
            python -m unittest discover
else
    docker run \
        --rm \
        -v ${PWD}:/ccpalign \
        -w /ccpalign \
        doublethinklab/nlp4if2020p:latest \
            python -m unittest $1
fi

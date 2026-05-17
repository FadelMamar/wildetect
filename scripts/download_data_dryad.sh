#!/bin/bash
# Download data from Dryad

cd /home/fadelmamar.seydou/workspace/wildetect/data/
nohup curl -X GET "https://datadryad.org/api/v2/files/4557066/download" \
    -H "Authorization: Bearer g1oPKnRseuY6e_89-o_gm1EvvKLLsK11NeHnPumGe80" \
    -O -J -L \
    > dryad.log 2>&1 &
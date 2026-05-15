#!/bin/bash
# Download data from Dryad

nohup curl -X GET "https://datadryad.org/api/v2/files/4557087/download" \
    -H "Authorization: Bearer YOUR_DRYAD_API_TOKEN" \
    -O -J -L \
    > dryad.log 2>&1 &
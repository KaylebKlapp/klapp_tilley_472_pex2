#!/bin/bash
if [ -d "/media/usafa/extern_data/Team Just Kidding/Collections" ]; then
    cp "/media/usafa/data/telemetry/*.bag" "/media/usafa/extern_data/Team Just Kidding/Collections"
    cp "/media/usafa/data/telemetry/csvs/*" "/media/usafa/extern_data/Team Just Kidding/Collections/csvs/"
    rm "/media/usafa/data/telemetry/*.bag"
    rm "/media/usafa/data/telemetry/csvs/*"
    echo "File copy complete"
else
    echo "Path not found."
fi




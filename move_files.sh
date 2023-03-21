#!/bin/bash
if [ -d "/media/usafa/extern_data/Team Just Kidding/Collections" ]; then
    cp "/media/usafa/PycharmProjects/USAFA/pex2/*.bag" "/media/usafa/extern_data/Team Just Kidding/Collections"
    cp "/media/usafa/PycharmProjects/USAFA/pex2/csvs/*" "/media/usafa/extern_data/Team Just Kidding/Collections/csvs/"
    rm "/media/usafa/PycharmProjects/USAFA/pex2/*.bag"
    rm "/media/usafa/PycharmProjects/USAFA/pex2/csvs/*"
    echo "File copy complete"
else
    echo "Path not found."
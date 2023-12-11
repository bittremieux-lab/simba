#sCRIPT FOR SENDING DATA`
#/scratch/antwerpen/209/vsc20939/globusconnectpersonal-3.2.3/globusconnect -start &
#cp /scratch/antwerpen/209/vsc20939/metabolomics/scatter_plot.png $VSC_HOME/
#globus transfer dff8c41a-9419-11ee-83dc-d5484943e99a:/user/antwerpen/209/vsc20939/scatter_plot.png ddb59aef-6d04-11e5-ba46-22000b92c6ec:~/scatter_plot.png


#!/bin/bash

# Check if the file argument is provided
if [ -z "$1" ]; then
    echo "Usage: $0 <file_to_send>"
    exit 1
fi

# Start Globus Connect
/scratch/antwerpen/209/vsc20939/globusconnectpersonal-3.2.3/globusconnect -start &

# Copy the specified file to $VSC_HOME
cp "$1" $VSC_HOME/

# Transfer the file using Globus
globus transfer dff8c41a-9419-11ee-83dc-d5484943e99a:/user/antwerpen/209/vsc20939/"$(basename "$1")" ddb59aef-6d04-11e5-ba46-22000b92c6ec:~/"$(basename "$1")"


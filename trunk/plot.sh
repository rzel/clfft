#!/bin/bash 

if [ "$#" -lt 4 ] ; then
    echo " Usage plot.sh pow_n start_n end_n config"
    exit 1
fi

input=$( echo "2^$1"| bc)   
count=$2
while [ $count -lt $3 ] ;
do
  pown=$(echo " 2^$count" | bc)
  output=`./clfft $input $pown $4 | grep Time | awk -F: '{print $2}'| awk  'BEGIN{myvar=""} { myvar=myvar":"$1} END{print myvar}'`
  echo "$input $pown $output"
  count=`expr $count + 1`
done

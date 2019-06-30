#!/bin/bash
#

file1=$1
shuffed_file1=${file1}.shuf
echo "shuf $file1 into ${shuffed_file1}"
shuf $file1 -o ${shuffed_file1}
#line_number=$2
folder=${shuffed_file1}_splits/
if [ -e $folder ]; then
   echo "$folder exists!, please remove that to re splits"
   exit $?
else
   mkdir -p $folder
   #split --verbose -d -l51200 ${shuffed_file1} $folder/split- 
   split --verbose -d -l12800 ${shuffed_file1} $folder/split- 
fi

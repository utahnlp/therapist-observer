#!/bin/bash
file1=$1
shuffed_file1=${file1}.shuf
echo "shuf $file1 into ${shuffed_file1}
shuf $file1 -o ${shuffed_file1}

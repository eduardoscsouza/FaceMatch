#!/bin/bash

find tensorboard_logs/ -mindepth 3 -maxdepth 3 -type d -name "validation" | cut -d\/ -f2-3 > aux.txt
find tensorboard_logs/ -mindepth 2 -maxdepth 2 -type d -name "validation" | cut -d\/ -f2   >> aux.txt
cat aux.txt | sort > temp_3.txt

find results/ -mindepth 3 -maxdepth 3 -type f -name "best_model.h5" | cut -d\/ -f2-3 > aux.txt
find results/ -mindepth 2 -maxdepth 2 -type f -name "best_model.h5" | cut -d\/ -f2   >> aux.txt
cat aux.txt | sort > temp_2.txt

find results/ -mindepth 3 -maxdepth 3 -type f -name "metrics.csv" | cut -d\/ -f2-3 > aux.txt
find results/ -mindepth 2 -maxdepth 2 -type f -name "metrics.csv" | cut -d\/ -f2   >> aux.txt
cat aux.txt | sort > temp_1.txt

diff temp_1.txt temp_2.txt
diff temp_1.txt temp_3.txt

rm temp_1.txt temp_2.txt temp_3.txt aux.txt
#!/bin/bash

datasetName=sampleData
numVgaViews=10 # Up to 480
mkdir -p ${datasetName}
cd ${datasetName}

# Download vga videos
mkdir -p vgaVideos

# Uniformly sampled order
panels=(1 19 14 6 16 9 5 10 18 15 3 8 4 20 11 13 7 2 17 12 9 5 6 3 15 2 12 14 16 10 4 13 20 8 17 19 18 9 4 6 1 20 1 11 7 7 14 15 3 2 16 13 3 15 17 9 20 19 8 11 5 8 18 10 12 19 5 6 16 12 4 6 20 13 4 10 15 12 17 17 16 1 5 3 2 18 13 16 8 19 13 11 10 7 3 2 18 10 1 17 10 15 14 4 7 9 11 7 20 14 1 12 1 6 11 18 7 8 9 3 15 19 4 16 18 1 11 8 4 10 20 13 6 16 7 6 16 17 12 5 17 4 8 20 12 17 14 2 19 14 18 15 11 11 9 9 2 13 5 15 20 18 8 3 19 11 9 2 13 14 5 9 17 9 7 6 12 16 18 17 13 15 17 20 4 2 2 12 4 1 16 4 11 1 16 12 18 9 7 20 1 10 10 19 5 8 14 8 4 2 9 20 14 17 11 3 12 3 13 6 5 16 3 5 10 19 1 11 13 17 18 2 5 14 19 15 8 8 9 3 6 16 15 18 20 4 13 2 11 20 7 13 15 18 10 20 7 5 2 15 6 13 4 17 7 3 19 19 3 10 2 12 10 7 7 12 11 19 8 9 6 10 6 15 10 11 3 16 1 5 14 6 5 13 20 14 4 18 10 14 14 1 19 8 14 19 3 6 6 3 13 17 8 20 15 18 2 2 16 5 19 15 9 12 19 17 8 9 3 7 1 12 7 13 1 14 5 12 11 2 16 1 18 4 18 10 16 11 7 5 1 16 9 4 15 1 7 10 14 3 2 17 13 19 20 15 10 4 8 16 14 5 6 20 12 5 18 7 1 8 11 5 13 1 16 14 18 12 15 2 12 3 8 12 17 8 20 9 2 6 9 6 12 3 20 15 20 13 3 14 1 4 8 6 10 7 17 13 18 19 10 20 12 19 2 15 10 8 19 11 19 11 2 4 6 2 11 8 7 18 14 4 12 14 7 9 7 11 18 16 16 17 16 15 4 15 9 17 13 3 6 17 17 20 19 11 5 3 1 18 4 10 5 9 13 1 5 9 6 14 )
nodes=(1 14 3 15 12 12 8 6 13 12 12 17 7 17 21 17 4 6 12 18 2 18 5 4 2 17 12 10 18 8 18 5 10 10 17 1 18 7 12 9 13 5 6 18 16 9 16 8 8 10 21 22 16 16 21 16 14 6 14 11 11 20 4 22 4 22 20 19 15 15 15 12 2 2 3 3 20 22 5 9 3 16 23 22 20 8 8 9 2 16 14 16 16 14 1 13 16 12 10 15 18 6 13 10 7 10 4 1 7 21 8 6 4 7 9 10 11 8 4 6 10 4 5 6 21 21 6 6 19 20 20 20 14 19 22 22 23 19 9 15 23 23 23 23 19 2 8 2 8 19 19 23 23 19 19 23 24 24 2 14 12 2 12 14 12 2 14 15 11 6 6 21 4 5 5 4 2 10 5 10 7 3 7 9 8 9 3 7 9 9 7 2 5 5 5 5 7 8 8 4 7 11 9 7 5 3 5 7 6 8 9 8 7 8 8 3 8 7 6 11 7 2 9 9 2 11 12 7 4 6 6 7 4 4 9 18 1 5 6 5 10 11 5 9 6 11 12 1 10 11 6 9 7 11 5 1 2 12 11 11 3 3 21 11 10 2 3 10 11 19 5 11 13 12 20 13 3 5 9 11 8 4 6 4 7 12 10 8 11 19 14 23 10 1 3 12 4 3 10 9 2 3 20 4 11 2 20 20 2 23 10 3 22 22 1 12 12 21 4 22 23 22 18 10 18 22 11 3 18 13 18 3 3 13 2 1 3 20 20 4 20 14 14 20 20 14 14 22 18 21 20 22 20 22 9 22 21 21 22 21 22 20 21 21 21 21 23 17 21 13 20 13 13 15 17 1 23 23 23 18 13 16 15 19 17 17 22 21 17 14 1 13 13 14 14 16 19 17 18 1 13 18 24 19 16 13 18 18 15 23 17 14 19 17 1 19 13 19 1 15 17 13 23 13 19 24 15 15 19 15 17 1 16 24 21 23 14 24 15 24 24 1 16 15 24 1 17 17 15 24 1 16 16 19 13 15 22 24 23 17 16 18 1 24 24 24 17 24 24 17 16 24 14 15 16 15 24 24 24 18)

for (( c=0; c<$numVgaViews; c++))
do
	cmd=$(printf "wget -c -t 0 -O vgaVideos/vga_%02d_%02d.mp4 http://domedb.perception.cs.cmu.edu/webdata/dataset/$datasetName/videos/vga_shared_crf10/ve%02d_%02d.mp4" ${panels[c]} ${nodes[c]} ${panels[c]} ${nodes[c]})
	eval $cmd
done

# Download a script for vga image extraction
wget -c -t 0 -O vgaImgsExtractor.sh http://domedb.perception.cs.cmu.edu/webdata/scripts/vgaImgsExtractor.sh

# Download calibration data
wget -c -t 0 -O calibration_${datasetName}.json http://domedb.perception.cs.cmu.edu/webdata/dataset/$datasetName/calibration_${datasetName}.json

# Download 3D pose reconstruction results (by vga index)
wget -c -t 0 -O vgaPose3d_stage1.tar http://domedb.perception.cs.cmu.edu/webdata/dataset/$datasetName/vgaPose3d_stage1.tar

tar -xf vgaPose3d_stage1.tar

. vgaImgsExtractor.sh

# Download hd videos
# Download 3D pose reconstruction results (by hd index)
# Download kinect-rgb videos
# Download point cloud data


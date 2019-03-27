#!/bin/bash
	cd ~/polybench/OpenCL/
	for file in $(ls); do
		if [ -d $file ]; then	
			cd $file
			make clean
			cd ..	
		fi	
	done


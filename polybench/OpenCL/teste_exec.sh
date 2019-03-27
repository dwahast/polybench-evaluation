#!/bin/bash

declare -a A7_frequencias=(400000 1000000 1400000)
declare -a A7_cores=(0 1 2 3)

declare -a A15_frequencias=(400000 1000000 1400000 2000000)
declare -a A15_cores=(4 5 6 7)

device="1"

 for freq in "${A7_frequencias[@]}"; do
 	for core in "${A7_cores[@]}"; do
 		sudo cpufreq-set --min $freq --max $freq -c $core -g performance
 	done

 	for file in $(ls); do
 		if [ -d $file ]; then
 			cd $(pwd)/$file
 			make
 			for i in $(seq 1 30); do
 				make run proc=A7 frequenc="$freq" c0="${A7_cores[0]}" c1="${A7_cores[1]}" c2="${A7_cores[2]}" c3="${A7_cores[3]}" dev="${device}"
 			done
 				cd ..
 		fi
 	done
 done

for freq in "${A15_frequencias[@]}"; do
	for core in "${A15_cores[@]}"; do
		sudo cpufreq-set --min $freq --max $freq -c $core -g performance
	done

	for file in $(ls); do
		if [ -d $file ]; then
			cd $(pwd)/$file
			make
			for i in $(seq 1 30); do
				make run proc=A15 frequenc="$freq" f="${file}" c0="${A15_cores[0]}" c1="${A15_cores[1]}" c2="${A15_cores[2]}" c3="${A15_cores[3]}" dev="${device}"
			done
				cd ..
		fi
	done
done

device="0"

for file in $(ls); do
	if [ -d $file ]; then
		cd $(pwd)/$file
		make
		for i in $(seq 1 30); do
			make run2 proc=GPU dev="${device}" f="${file}"
		done
			cd ..
	fi
done

# ## declare an array variable
# declare -a arr=(10 20 30)

# ## now loop through the above array
# for i in "${arr[@]}"
# do
#    echo "$i"
#    # or do whatever with individual element of the array
# done

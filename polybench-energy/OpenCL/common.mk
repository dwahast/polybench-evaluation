OpenCL_SDK=/usr/local/cuda
INCLUDE=-L ${OpenCL_SDK}/include
LIBPATH=-L ${OpenCL_SDK}/lib64
LIB=-lOpenCL -lm -w
#./${EXECUTABLE} >> ~/Documents/polybench/Logs/log_${EXECUTABLE}.txt

all:
	gcc -O3 ${INCLUDE} ${LIBPATH} ${CFILES} -o ${EXECUTABLE} ${LIB} 

run:
	taskset -c ${c0},${c1},${c2},${c3} sudo sh ~/run.sh ~/energyMonitor ~/Douglas/polybench-energy/Logs/${f}_$(proc)_$(frequenc)_log_energia.txt "./${EXECUTABLE} $(dev)"
	
run2:
	sudo sh  ~/run.sh ~/energyMonitor ~/Douglas/polybench-energy/Logs/${f}_GPU_600000_log_energia.txt "./${EXECUTABLE} $(dev)"

clean:
	rm -f *~ *.exe

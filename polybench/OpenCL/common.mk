OpenCL_SDK=/usr/local/cuda
INCLUDE=-L ${OpenCL_SDK}/include
LIBPATH=-L ${OpenCL_SDK}/lib64
LIB=-lOpenCL -lm -w
#taskset -c ${c0},${c1},${c2},${c3} sudo sh ~/run.sh ~/energyMonitor ~/Douglas/polybench-energy/Logs/${EXECUTABLE}_$(proc)_$(frequenc)_log_energia.txt "./${EXECUTABLE} $(dev)"

all:
	gcc -O3 ${INCLUDE} ${LIBPATH} ${CFILES} -o ${EXECUTABLE} ${LIB} 

run:	
	taskset -c ${c0},${c1},${c2},${c3} ./${EXECUTABLE} $(dev) >> ~/Douglas/polybench/Logs/${f}_$(proc)_$(frequenc)_log_tempo.txt

run2:
	./${EXECUTABLE} $(dev) >> ~/Douglas/polybench/Logs/${f}_GPU_600000_log_tempo.txt

clean:
	rm -f *~ *.exe

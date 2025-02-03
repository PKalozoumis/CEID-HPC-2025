.PHONY: clean

all: clean cuda

cuda:cuda.cu
	nvcc -Xcompiler -fopenmp cuda.cu -o cuda

clean:
	@if [ -f "cuda" ]; then rm cuda; fi


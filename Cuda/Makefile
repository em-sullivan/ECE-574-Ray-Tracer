CUDA_PATH     ?= /usr/local/cuda
HOST_COMPILER  = g++
NVCC           = $(CUDA_PATH)/bin/nvcc -ccbin $(HOST_COMPILER)

OBJS = main.o Vec3.o Ray.o Color.o Sphere.o Hittable_List.o Material.o  Aabb.o Texture.o Camera.o #Moving_Sphere.o Bvh_Node.o Aarect.o Box.o Translate.o Constant_Medium.o

# select one of these for Debug vs. Release
#NVCC_DBG       = -g -G
NVCC_DBG       =

NVCCFLAGS      = $(NVCC_DBG) -m64 -Xptxas -O3 -use_fast_math #--ftz=true --prec-div=false --prec-sqrt=false --fmad=true.

# For RTX20 series, Turing
GENCODE_FLAGS  = -gencode arch=compute_75,code=sm_75

cudart: $(OBJS)
	$(NVCC) $(NVCCFLAGS) $(GENCODE_FLAGS) $(OBJS) -o cudart

%.o: %.cpp
	$(NVCC) $(NVCCFLAGS) $(GENCODE_FLAGS) -x cu -dc $< -o $@

create-image:
	./cudart
	convert out.ppm out.jpg

profile_basic: cudart
	nvprof --unified-memory-profiling off ./cudart

# use nvprof --query-metrics
profile_metrics: cudart
	nvprof --unified-memory-profiling off --metrics achieved_occupancy,inst_executed,inst_fp_32,inst_fp_64,inst_integer ./cudart

clean:
	rm -f cudart *.o *.ppm out.jpg *.elf *core
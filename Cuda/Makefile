CUDA_PATH     ?= /usr/local/cuda
HOST_COMPILER  = g++
NVCC           = $(CUDA_PATH)/bin/nvcc -ccbin $(HOST_COMPILER)

CFLAGS = -Wall -g -O2
OBJS = main.o Vec3.o Ray.o Color.o Sphere.o Hittable_List.o Material.o  Aabb.o Texture.o Perlin.o Camera.o #Moving_Sphere.o Bvh_Node.o Aarect.o Box.o Translate.o Constant_Medium.o

# select one of these for Debug vs. Release
#NVCC_DBG       = -g -G
NVCC_DBG       =

NVCCFLAGS      = $(NVCC_DBG) -m64  #--expt-relaxed-constexpr

# For RTX20 series, Turing
GENCODE_FLAGS  = -gencode arch=compute_75,code=sm_75

cudart: $(OBJS)
	$(NVCC) $(NVCCFLAGS) $(GENCODE_FLAGS) $(OBJS) -o cudart

%.o: %.cpp
	$(NVCC) $(NVCCFLAGS) $(GENCODE_FLAGS) -x cu -dc $< -o $@
	
#cudart: cudart.o $(OBJS)
#	$(HOST_COMPILER) $(CFLAGS)  -o cudart cudart.o $(OBJS)

#cudart.o: main.cu
#	$(NVCC) $(NVCCFLAGS) $(GENCODE_FLAGS) -c main.cu Vec3.cu 

create-image:
	./cudart
	convert out.ppm out.jpg

profile_basic: cudart
	nvprof ./cudart

# use nvprof --query-metrics
profile_metrics: cudart
	nvprof --metrics achieved_occupancy,inst_executed,inst_fp_32,inst_fp_64,inst_integer ./cudart

clean:
	rm -f cudart *.o *.ppm out.jpg *.elf *core
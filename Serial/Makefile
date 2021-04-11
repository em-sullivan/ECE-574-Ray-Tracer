CC = g++
CFLAGS = -Wall -o2
LDFLAGS = 
OBJS = Color.o Vec3.o Ray.o Sphere.o Hittable_List.o Camera.o Material.o Moving_Sphere.o Bvh_Node.o Aabb.o Texture.o Perlin.o Aarect.o Box.o Translate.o Constant_Medium.o render.o

N = 8


sample: main.o $(OBJS)
	$(CC) $(LDFLAGS) -o rayer-tracer.elf main.o $(OBJS)

main.o:
	$(CC) $(CLFAGS) -c main.cpp

%.o : %.cpp
	$(CC) $(CFLAGS) -o $@ -c $<

create-image:
	./rayer-tracer.elf $(N)
	convert out.ppm out.jpg

clean:
	rm -f *.o *.ppm out.jpg *.elf *core

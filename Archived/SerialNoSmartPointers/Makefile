CC = g++
CFLAGS = -Wall -g -O3
OBJS = Color.o Vec3.o Ray.o Sphere.o Hittable_List.o Camera.o Material.o Moving_Sphere.o Bvh_Node.o Aabb.o Texture.o Perlin.o Aarect.o Box.o Translate.o Constant_Medium.o

N = 7 


sample: main.o $(OBJS)
	$(CC) $(CFLAGS) -o sample.elf main.o $(OBJS)

#main.o: main.cpp
#	$(CC) $(CFLAGS) -c main.cpp

create-image:
	./sample.elf $(N)
	convert out.ppm out.jpg
	#rm out.ppm

clean:
	rm -f *.o *.ppm out.jpg *.elf *core
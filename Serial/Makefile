CC = g++
CFLAGS = -Wall -g
OBJS = Color.o Vec3.o Ray.o Sphere.o Hittable_List.o Camera.o Material.o Moving_Sphere.o

sample: main.o $(OBJS)
	$(CC) $(CFLAGS) -o sample.elf main.o $(OBJS)

#main.o: main.cpp
#	$(CC) $(CFLAGS) -c main.cpp

create-image:
	./sample.elf
	convert out.ppm out.jpg
	#rm out.ppm

clean:
	rm -f *.o *.ppm out.jpg *.elf *core

# Dependencies
Ray.o: Ray.h
Vec.o: Vec.h
Color.o: Color.h
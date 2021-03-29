CC = g++
CFLAGS = -Wall -g
OBJS = Color.o Vec3.o Ray.o Sphere.o Hittable_List.o Camera.o Material.o

sample: main.o $(OBJS)
	$(CC) $(CFLAGS) -o sample main.o $(OBJS)

#main.o: main.cpp
#	$(CC) $(CFLAGS) -c main.cpp

create-image:
	./sample
	convert out.ppm out.jpg
	#rm out.ppm

clean:
	rm -f *.o *.ppm out.jpg sample *core

# Dependencies
Ray.o: Ray.h
Vec.o: Vec.h
Color.o: Color.h
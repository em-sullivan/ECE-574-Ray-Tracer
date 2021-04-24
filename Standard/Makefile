CPP = g++
CPPFLAGS = -Wall -O2 -g
LDFLAGS = 
OBJS = Color.o Vec3.o Ray.o Sphere.o Hittable_List.o Camera.o Material.o Moving_Sphere.o Bvh_Node.o Aabb.o Texture.o Perlin.o Aarect.o Box.o Translate.o Constant_Medium.o render.o worlds.o
INC = -I../Common/include


rayer-tracer: main.o $(OBJS)
	$(CPP) $(LDFLAGS) -o rayer-tracer.elf main.o $(OBJS)

main.o: main.cpp
	$(CPP) $(CPPFLAGS) $(INC) -o main.o -c main.cpp

%.o : %.cpp
	$(CPP) $(CPPFLAGS) $(INC) -o $@ -c $<

create-image:
	./rayer-tracer.elf $(N) $(W) $(H) $(S)
	convert out.ppm out.jpg

clean:
	rm -f *.o *.ppm out.jpg *.elf *core

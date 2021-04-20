CC = cl.exe
LINK = link.exe
CCFLAGS = /EHsc /o2
CPPFLAGS = /EHsc /O2
LDFLAGS = 
OBJS = main.obj Color.obj Vec3.obj Ray.obj Sphere.obj Hittable_List.obj Camera.obj Material.obj Moving_Sphere.obj Bvh_Node.obj Aabb.obj Texture.obj Perlin.obj Aarect.obj Box.obj Translate.obj Constant_Medium.obj render.obj worlds.obj

N = 8

#build: main

#main: $(OBJS)
#    $(CC) $(CFLAGS) $@ $**

rayer-tracer: $(OBJS)
	$(LINK) $(LDFLAGS) /OUT:rayer-tracer.exe $(OBJS)

#main.obj: main.cpp
#	$(CC) $(CFLAGS) /c main.cpp /Fomain.obj

%.obj : %.cpp
	$(CPP) $<.cpp /Fo$@.obj

create-image:
	.\rayer-tracer.exe $(N)
	convert out.ppm out.jpg

clean:
	del -f *.obj *.ppm out.jpg *.exe

ROOTDIR    ?= /opt/opencl-sdk/sdk

CXXWARN_FLAGS := \
        -W -Wall \
        -Wimplicit \
        -Wswitch \
        -Wformat \
        -Wchar-subscripts \
        -Wparentheses \
        -Wmultichar \
        -Wtrigraphs \
        -Wpointer-arith \
        -Wcast-align \
        -Wreturn-type \
        -Wno-unused-function \
	$(SPACE)

CXXFLAGS  := $(CXXWARN_FLAGS)
INCLUDES  := -I/opt/opencl-sdk/shared/inc -I/opt/opencl-sdk/OpenCL/common/inc

# Libs
LIB  := -L/usr/lib64/ -L/opt/opencl-sdk/OpenCL/common/lib  -L/opt/opencl-sdk/shared/lib/ 

LIB += -lOpenCL -loclUtil -lshrutil

OBJFILES := $(patsubst %.c,%.o, $(wildcard *.c))

clfft: $(OBJFILES)
	echo "Object files:" $(OBJFILES)
	g++ $(CXXFLAGS)  -o clfft  $(INCLUDES) $(LIB) $(OBJFILES) 

%.o: %.c
	g++ $(CXXFLAGS) -o $@ $< $(INCLUDES) $(LIB)
	echo "Object files:" $(OBJFILES)


all: clfft


clean:
	rm *.o a.out

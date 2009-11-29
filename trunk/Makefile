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

CXXFLAGS  := -fopenmp $(CXXWARN_FLAGS)
INCLUDES  := -I/opt/opencl-sdk/shared/inc -I/opt/opencl-sdk/OpenCL/common/inc

# Libs
LIB  := -L/usr/lib64/ -L/opt/opencl-sdk/OpenCL/common/lib  -L/opt/opencl-sdk/shared/lib/ 

LIB += -lOpenCL -loclUtil -lshrutil

OBJFILES := $(patsubst %.cpp,%.o, $(wildcard *.cpp))
HFILES := $(wildcard *.h)

clfft: $(OBJFILES)
	@g++ $(CXXFLAGS)  -o clfft $(OBJFILES)  $(INCLUDES) $(LIB)  

%.o: %.cpp $(HFILES) 
	@g++ $(CXXFLAGS) -ggdb3 -c -o $@ $< $(INCLUDES) 

all:clfft


clean:
	rm *.o clfft


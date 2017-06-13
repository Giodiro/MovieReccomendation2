UNAME := $(shell (lsb_release -ds 2>/dev/null || cat /etc/*release 2>/dev/null | head -n1 || uname -om) | cut -d' ' -f 1)

ifeq ($(UNAME), Linux)
	CC=g++      # Used for compiling SGD
	#CC=mpic++  # Used for compiling MPI_SGD

	WARNINGS=-Wall -Wextra -Wcast-align -Wcast-qual -Wctor-dtor-privacy \
		     -Wdisabled-optimization -Wformat=2 -Winit-self -Wlogical-op -Wmissing-declarations \
		     -Wmissing-include-dirs -Wnoexcept -Woverloaded-virtual -Wredundant-decls \
		     -Wshadow -Wsign-promo -Wstrict-null-sentinel -Wold-style-cast \
		     -Wswitch-default -Wundef -Wno-unused -pedantic

	# Use of -isystem instead of -I avoids printing tons of errors from Eigen
	CFLAGS=-std=c++14 -O3 -march=native -fopenmp -isystem ../lib/  $(WARNINGS)
	LDFLAGS=
else
	CC=icpc
	#CC=mpic++
	CFLAGS=-Wall -std=c++14 -O3 -no-prec-div -fp-model precise -ipo -qopenmp -isystem lib/ -isystem ~/local/include
	LDFLAGS=
endif


EXECUTABLE=SGD
#EXECUTABLE=MPI_SGD
SOURCES=variant.cpp sgd_util.cpp sgd.cpp ioutil.cpp evaluation.cpp main.cpp
#SOURCES=variant.cpp sgd_util.cpp sgd.cpp ioutil.cpp evaluation.cpp mpi_main.cpp

OBJECTS=$(SOURCES:.cpp=.o)

all: $(SOURCES) $(EXECUTABLE)

$(EXECUTABLE): $(OBJECTS)
	$(CC) $(CFLAGS) $(OBJECTS) $(LDFLAGS) -o $@

.cpp.o:
	$(CC) $(CFLAGS) -c $< -o $@

clean:
	rm $(EXECUTABLE) *.o

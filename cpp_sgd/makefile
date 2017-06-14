#######
# Compilation:
# - different compilers if on Euler (use the intel compilers)
# - 2 different targets: MPI_SGD or SGD. They use 2 different compilers as well!
#

# UNAME is used to detect the system (works on Ubuntu and Euler)
UNAME := $(shell (lsb_release -ds 2>/dev/null || cat /etc/*release 2>/dev/null | head -n1 || uname -om) | cut -d' ' -f 1)

ifeq ($(UNAME), Linux)
mpi: CC=mpic++   # Used for compiling MPI_SGD
sgd: CC=g++      # Used for compiling SGD

	WARNINGS=-Wall -Wextra -Wcast-align -Wcast-qual -Wctor-dtor-privacy \
		     -Wdisabled-optimization -Wformat=2 -Winit-self -Wlogical-op -Wmissing-declarations \
		     -Wmissing-include-dirs -Wnoexcept -Woverloaded-virtual -Wredundant-decls \
		     -Wshadow -Wsign-promo -Wstrict-null-sentinel \
		     -Wswitch-default -Wundef -Wno-unused -pedantic

	# Use of -isystem instead of -I avoids printing tons of errors from Eigen
	LDFLAGS=-fopenmp -isystem ../lib/
	CFLAGS=-std=c++14 -O3 -g3 -march=native $(LDFLAGS) $(WARNINGS)
else
mpi: CC=mpic++
sgd: CC=icpc
	LDFLAGS=-isystem ../lib/ -isystem ~/local/include
	CFLAGS=-Wall -std=c++14 -O3 -no-prec-div -fp-model precise -ipo -qopenmp $(LDFLAGS)
endif

COMMON_SOURCES = variant.cpp sgd_util.cpp sgd.cpp ioutil.cpp evaluation.cpp
MPI_SOURCES = mpi_main.cpp
SGD_SOURCES = main.cpp

COMMON_OBJECTS = $(COMMON_SOURCES:.cpp=.o)
MPI_OBJECTS = $(MPI_SOURCES:.cpp=.o)
SGD_OBJECTS = $(SGD_SOURCES:.cpp=.o)

MPI_EXE = MPI_SGD
SGD_EXE = SGD

# No idea what this is
.PHONY: all mpi sgd

all: mpi sgd

mpi: $(MPI_EXE)

sgd: $(SGD_EXE)

$(MPI_EXE): $(COMMON_OBJECTS) $(MPI_OBJECTS)
	$(CC) $(LDFLAGS) $^ -o $@

$(SGD_EXE): $(COMMON_OBJECTS) $(SGD_OBJECTS)
	$(CC) $(LDFLAGS) $^ -o $@

.cpp.o:
	$(CC) $(CFLAGS) -c $< -o $@

clean:
	rm $(MPI_EXE) $(SGD_EXE) *.o

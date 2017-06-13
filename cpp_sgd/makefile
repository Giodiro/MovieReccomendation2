UNAME := $(shell (lsb_release -ds 2>/dev/null || cat /etc/*release 2>/dev/null | head -n1 || uname -om) | cut -d' ' -f 1)

ifeq ($(UNAME), Linux)
	CC=g++
	#CC=mpic++
	# Compiler options
	#CFLAGS=-std=c++14 -O3 -g3 -march=native -fopenmp -isystem lib/ \
	#	   -pedantic -Wall -Wextra -Wcast-align -Wcast-qual -Wctor-dtor-privacy \
	#	   -Wdisabled-optimization -Wformat=2 -Winit-self -Wlogical-op -Wmissing-declarations \
	#	   -Wmissing-include-dirs -Wnoexcept -Wold-style-cast -Woverloaded-virtual -Wredundant-decls \
	#	   -Wshadow -Wsign-promo -Wstrict-null-sentinel -Wuseless-cast \
	#	   -Wswitch-default -Wundef -Wno-unused
	CFLAGS=-std=c++14 -O3 -march=native -fopenmp -isystem ../lib/ \
		   -pedantic -Wall -Wextra -Wcast-align -Wcast-qual -Wctor-dtor-privacy \
		   -Wdisabled-optimization -Wformat=2 -Winit-self -Wlogical-op -Wmissing-declarations \
		   -Wmissing-include-dirs -Wnoexcept -Woverloaded-virtual -Wredundant-decls \
		   -Wshadow -Wsign-promo -Wstrict-null-sentinel \
		   -Wswitch-default -Wundef -Wno-unused
	LDFLAGS=-L../xgboost/lib -L../xgboost/dmlc-core -L../xgboost/rabit/lib -lxgboost -lrabit -ldmlc -lgomp
else
	CC=icpc
	#CC=mpic++
	CFLAGS=-Wall -std=c++14 -O3 -no-prec-div -fp-model precise -ipo -qopenmp -isystem lib/ -isystem ~/local/include
	LDFLAGS=-L/cluster/home/gmeanti/local/lib -lxgboost -lrabit -ldmlc
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

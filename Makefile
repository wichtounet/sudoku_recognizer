default: bin/binarize

.PHONY: default

C_FLAGS=-Iinclude -std=c++1y -Ofast -g -Wextra -Wall -Wno-unused-function
LD_FLAGS=$(C_FLAGS) -lopencv_core -lopencv_imgproc -lopencv_highgui

#TODO Add flags from dbn
src/binarize.cpp.o: src/binarize.cpp
	clang++ $(C_FLAGS) -o src/binarize.cpp.o -c src/binarize.cpp

bin/binarize: src/binarize.cpp.o
	mkdir -p bin/
	clang++ $(LD_FLAGS) -o bin/binarize src/binarize.cpp.o

clean:
	rm -rf src/*.cpp.o
	rm -rf bin
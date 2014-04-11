default: bin/binarize

.PHONY: default

CPP_FILES=$(wildcard src/*.cpp)
O_FILES=$(CPP_FILES:%.cpp=%.cpp.o)

CC=clang++
LD=clang++
CXX_FLAGS=-Iinclude -std=c++1y -Ofast -g -Wextra -Wall -Wno-unused-function
LD_FLAGS=$(CXX_FLAGS) -lopencv_core -lopencv_imgproc -lopencv_highgui

#TODO Add flags from dbn
src/binarize.cpp.o: src/binarize.cpp
	$(CC) $(CXX_FLAGS) -o src/binarize.cpp.o -c src/binarize.cpp

src/%.cpp.o: src/%.cpp
	$(CC) $(CXX_FLAGS) -o $@ -c $<

bin/binarize: $(O_FILES)
	mkdir -p bin/
	$(LD) $(LD_FLAGS) -o bin/binarize $(O_FILES)

clean:
	rm -rf $(O_FILES)
	rm -rf bin
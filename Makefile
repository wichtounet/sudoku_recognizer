default: bin/binarize

.PHONY: default

CPP_FILES=$(wildcard src/*.cpp)
D_FILES=$(CPP_FILES:%.cpp=%.cpp.d)
O_FILES=$(CPP_FILES:%.cpp=%.cpp.o)

CC=clang++
LD=clang++
CXX_FLAGS=-Iinclude -std=c++1y -Ofast -g -Wextra -Wall -Wno-unused-function
LD_FLAGS=$(CXX_FLAGS) -lopencv_core -lopencv_imgproc -lopencv_highgui

src/%.cpp.d: $(CPP_FILES)
	@ $(CC) $(CXX_FLAGS) -MM -MT src/$*.cpp.o src/$*.cpp | sed -e 's@^\(.*\)\.o:@\1.d \1.o:@' > $@

include $(D_FILES)

#TODO Add flags from dbn
src/%.cpp.o:
	$(CC) $(CXX_FLAGS) -o $@ -c $<

bin/binarize: $(O_FILES)
	mkdir -p bin/
	$(LD) $(LD_FLAGS) -o bin/binarize $(O_FILES)

clean:
	rm -rf $(O_FILES)
	rm -rf $(D_FILES)
	rm -rf bin
default: release

.PHONY: default release debug all clean

OUTPUT=sudoku

CPP_FILES=$(wildcard src/*.cpp)

DEBUG_D_FILES=$(CPP_FILES:%.cpp=debug/%.cpp.d)
RELEASE_D_FILES=$(CPP_FILES:%.cpp=release/%.cpp.d)

DEBUG_O_FILES=$(CPP_FILES:%.cpp=debug/%.cpp.o)
RELEASE_O_FILES=$(CPP_FILES:%.cpp=release/%.cpp.o)

CC=clang++
LD=clang++

#TODO Add flags from dbn

CXX_FLAGS=-Iinclude -std=c++1y -Wextra -Wall -Wno-unused-function
LD_FLAGS=$(CXX_FLAGS) -lopencv_core -lopencv_imgproc -lopencv_highgui

DEBUG_FLAGS=-g
RELEASE_FLAGS=-g -Ofast -march=native -fvectorize -fslp-vectorize-aggressive -fomit-frame-pointer

debug/src/%.cpp.o: src/%.cpp
	@ mkdir -p debug/src/
	$(CC) $(CXX_FLAGS) $(DEBUG_FLAGS) -o $@ -c $<

release/src/%.cpp.o: src/%.cpp
	@ mkdir -p release/src/
	$(CC) $(CXX_FLAGS) $(RELEASE_FLAGS) -o $@ -c $<

debug/bin/$(OUTPUT): $(DEBUG_O_FILES)
	@ mkdir -p debug/bin/
	$(LD) $(LD_FLAGS) $(DEBUG_FLAGS) -o debug/bin/$(OUTPUT) $(DEBUG_O_FILES)

release/bin/$(OUTPUT): $(RELEASE_O_FILES)
	@ mkdir -p release/bin/
	$(LD) $(LD_FLAGS) $(RELEASE_FLAGS) -o release/bin/$(OUTPUT) $(RELEASE_O_FILES)

debug/src/%.cpp.d: $(CPP_FILES)
	@ mkdir -p debug/src/
	@ $(CC) $(CXX_FLAGS) $(DEBUG_FLAGS) -MM -MT debug/src/$*.cpp.o src/$*.cpp | sed -e 's@^\(.*\)\.o:@\1.d \1.o:@' > $@

release/src/%.cpp.d: $(CPP_FILES)
	@ mkdir -p release/src/
	@ $(CC) $(CXX_FLAGS) $(RELEASE_FLAGS) -MM -MT release/src/$*.cpp.o src/$*.cpp | sed -e 's@^\(.*\)\.o:@\1.d \1.o:@' > $@

release: release/bin/$(OUTPUT)
debug: debug/bin/$(OUTPUT)

all: release debug

clean:
	rm -rf release/
	rm -rf debug/

-include $(DEBUG_D_FILES)
-include $(RELEASE_D_FILES)
default: binarize

.PHONY: default

#TODO Add flags from dbn
src/binarize.cpp.o: src/binarize.cpp
	clang++ -o src/binarize.cpp.o -Ofast -g -c src/binarize.cpp

binarize: src/binarize.cpp.o
	mkdir -p bin/
	clang++ -o bin/binarize -lopencv_core -lopencv_imgproc -lopencv_highgui src/binarize.cpp.o
default: release

.PHONY: default release release_debug debug all clean

include make-utils/flags.mk
include make-utils/cpp-utils.mk

$(eval $(call use_libcxx))

CXX_FLAGS += -Idbn/etl/include -Ihmm/include -Idbn/nice_svm/include -Idbn/include -Imnist/include -Iinclude/cpp_utils
LD_FLAGS  += -lopencv_core -lopencv_imgproc -lopencv_highgui -lsvm -pthread

$(eval $(call auto_folder_compile,src))
$(eval $(call auto_simple_c_folder_compile,hmm/src))
$(eval $(call auto_add_executable,sudoku))

release_debug: release_debug_sudoku
release: release_sudoku
debug: debug_sudoku

all: release debug release_debug

cppcheck:
	cppcheck --enable=all --std=c++11 -I include src

clean: base_clean

include make-utils/cpp-utils-finalize.mk

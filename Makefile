default: release

.PHONY: default release debug all clean

include make-utils/flags.mk
include make-utils/cpp-utils.mk

$(eval $(call use_libcxx))

CXX_FLAGS += -Idbn/etl/include -Ihmm/include -Idbn/nice_svm/include -Idbn/include -Imnist/include -Iinclude/cpp_utils
LD_FLAGS  += -lopencv_core -lopencv_imgproc -lopencv_highgui -lsvm -pthreads

$(eval $(call auto_folder_compile,src))
$(eval $(call auto_simple_c_folder_compile,hmm/src))
$(eval $(call auto_add_executable,sudoku))

release: release_sudoku
debug: debug_sudoku

all: release debug

cppcheck:
	cppcheck --enable=all --std=c++11 -I include src

sonar: release
	cppcheck --xml-version=2 --enable=all --std=c++11 -I include src 2> cppcheck_report.xml
	/opt/sonar-runner/bin/sonar-runner

clean: base_clean

include make-utils/cpp-utils-finalize.mk

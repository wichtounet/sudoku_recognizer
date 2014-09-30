default: release

.PHONY: default release debug all clean

include make-utils/flags.mk
include make-utils/cpp-utils.mk

CXX_FLAGS += -Idbn/etl/include -Idbn/include -std=c++1y -stdlib=libc++
LD_FLAGS  += -lopencv_core -lopencv_imgproc -lopencv_highgui

$(eval $(call auto_folder_compile,src))
$(eval $(call auto_add_executable,sudoku))

release: release_sudoku
debug: debug_sudoku

all: release debug

sonar: release
	cppcheck --xml-version=2 --enable=all --std=c++11 -I include src 2> cppcheck_report.xml
	/opt/sonar-runner/bin/sonar-runner

clean: base_clean

include make-utils/cpp-utils-finalize.mk
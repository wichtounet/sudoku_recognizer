default: release

.PHONY: default release release_debug debug all clean

include make-utils/flags.mk
include make-utils/cpp-utils.mk

$(eval $(call use_libcxx))

CXX_FLAGS += -I dbn/etl/lib/include -Idbn/etl/include -Idbn/include -Ihmm/include -Idbn/nice_svm/include -Imnist/include
LD_FLAGS  += -lopencv_core -lopencv_imgproc -lopencv_highgui -lsvm -pthread

# Let ETL vectorize as much as possible
CXX_FLAGS += -DETL_VECTORIZE_FULL

# Activate BLAS mode on demand
ifneq (,$(ETL_MKL))
CXX_FLAGS += -DETL_MKL_MODE $(shell pkg-config --cflags $(DLL_BLAS_PKG))
LD_FLAGS += $(shell pkg-config --libs $(DLL_BLAS_PKG))

# Disable warning for MKL
ifneq (,$(findstring clang,$(CXX)))
CXX_FLAGS += -Wno-tautological-compare
endif

else
ifneq (,$(ETL_BLAS))
CXX_FLAGS += -DETL_BLAS_MODE $(shell pkg-config --cflags cblas)
LD_FLAGS += $(shell pkg-config --libs cblas)

# Disable warning for MKL
ifneq (,$(findstring clang,$(CXX)))
CXX_FLAGS += -Wno-tautological-compare
endif

endif
endif

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

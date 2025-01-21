 .PHONY : build clean test

TYPE ?= Release

build:
	mkdir -p build
	cmake -Bbuild \
      -DCMAKE_BUILD_TYPE=$(TYPE) \
      -DUSE_CUDA=ON \
      -DCUDNN_INCLUDE_DIR=/usr/local/cuda/include \
      -DCUDNN_LIBRARY=/usr/local/cuda/lib64/libcudnn.so \
	  -DUSE_CUDA=ON
	make -j -C build

clean:
	rm -rf build

test:
	python3 test/gather.py --device cuda
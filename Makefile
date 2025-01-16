# 指定Python解释器路径
PYTHON_EXECUTABLE := C:/Users/sword/anaconda3/python.exe

# 定义构建类型，默认为Release
TYPE ?= Release

# 定义.PHONY目标，确保这些命令总是被执行，即使有同名文件存在
.PHONY: build clean test

# 构建规则
build:
	mkdir -p build
	cmake -Bbuild -G Ninja -DCMAKE_BUILD_TYPE=$(TYPE) -DUSE_CUDA=ON -DPYTHON_EXECUTABLE=$(PYTHON_EXECUTABLE)
	$(MAKE) -j -C build

# 清理规则
clean:
	rm -rf build

# 测试规则
test:
	"$(PYTHON_EXECUTABLE)" test/gather.py --device cuda
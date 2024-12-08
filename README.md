# About CHLOE
CHLOE is a end-to-end Fully Homomorphic Encryption (FHE) compiler to compile 
high-level input C programs into efficient FHE implementations. In general, 
CHLOE uses both arithmetic and logic FHE encryption schemes to transform 
programs with loop structures to efficient algorithms over fully 
homomorphic encryption (FHE). CHLOE's designs are discribed in our [paper](https://www.computer.org/csdl/proceedings-article/sp/2025/223600a035/21B7QoGZAGc).
# Structure of this Repository
The repository is organized as follow:
```
cmake             – configuration files for the CMake build system
include           – header (.h) and TableGen (.td) files
 └ IR               – contains HEIR dialect definitions
 └ Passes           – contains the definitions of the different transformations
src               – source files (.cpp)
 └ IR               – implementations of additional dialect-specific functionality
 └ Passes           – implementations of the different transformations
 └ tools            – sources for the main commandline interface
benchmarks        – benchmark evaluation
```
# Structure of this Repository
# Using CHLOE
### Front-End
CHLOE uses [Polygeist](https://github.com/llvm/Polygeist) CLI `cgeist` 
as the Front-End to transform
the input C program into `*.mlir` file. Please use the 
following default parameters to execute the tool.
```sh
./{Polygeitst_BUILD_DIR}/cgeist $fileName$.c \
  -function=$functionName$ -S \
  -raise-scf-to-affine \
  --memref-fullrank -O0
```
### Middle-End
In Middle-End, HEIR uses `chloe-opt` CLI to transform the input
MLIR program into programs with homomorphic operators 
reprsented in `emitc` dialect. There are three parameters for 
`chloe-opt`:

+ **--branch**: Add this parameter when `if` insturction is 
called in the input C program.
+ **--affine-loop-unroll="unroll-full"**: 
Add this parameter to unroll all the `for` loop in the 
input program.
+ **--loop-analyze**: A pass to segment a mixed-operation loop structure
to sub loops with pure arithmetic/logic operations.
+ **--loop-batch**: A pass to transform loop structures into a batched version.
+ **--full-pass**: A full pass pipeline to transform the program in 
standard MLIR dialects into optimized FHE implementations.  

Next, CHLOE uses `emitc-translate` to transform the MLIR file
into a C++ file:
```sh
./tools/emitc-translate $fileName$.mlir --mlir-to-cpp
```

# A Guide For HEIR Installation
## Build Polygeist Front-End
Start with ``CHLOE`` directory.

Clone Polygeist from Github.
```sh
cd ..
git clone -b dev --recursive https://github.com/heir-compiler/Polygeist
cd Polygeist
```
Using unified LLVM, MLIR, Clang, and Polygeist build.
```sh
mkdir build
cd build
cmake -G Ninja ../llvm-project/llvm \
  -DLLVM_ENABLE_PROJECTS="clang;mlir" \
  -DLLVM_EXTERNAL_PROJECTS="polygeist" \
  -DLLVM_EXTERNAL_POLYGEIST_SOURCE_DIR=.. \
  -DLLVM_TARGETS_TO_BUILD="host" \
  -DLLVM_ENABLE_ASSERTIONS=ON \
  -DCMAKE_BUILD_TYPE=DEBUG
ninja
```

## Build HEIR Middle-End
Start with ``CHLOE`` directory.

Clone llvm-15 from Github. Note that LLVM-15 used for HEIR Middle-End is not compatiable with LLVM for building Polygeist.
```sh
cd ..
git clone -b release/19.x https://github.com/llvm/llvm-project.git
cd llvm-project
```
Build LLVM/MLIR.
```sh
mkdir build && cd build
cmake -G Ninja ../llvm -DLLVM_ENABLE_PROJECTS="mlir" -DLLVM_BUILD_EXAMPLES=ON -DLLVM_TARGETS_TO_BUILD="X86" -DCMAKE_BUILD_TYPE=Release -DLLVM_ENABLE_ASSERTIONS=ON -DLLVM_INSTALL_UTILS=ON
ninja -j N
```

Build CHLOE.
```sh
cd ../../HEIR
mkdir build && cd build
cmake .. -DMLIR_DIR=/home/NDSS_Artifact/llvm-project/build/lib/cmake/mlir
cmake --build . --target all
```


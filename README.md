### ptMgrad = pt (PyTorch) + Mgrad (micrograd)

This is a fun project inspired by the [PyTorch](https://github.com/pytorch/pytorch)
and Andrej Karpathy's [micrograd](https://github.com/karpathy/micrograd).

***Note:*** Took help from AI assistance for C++ memory management and
the iterative `backward()` topological sort.

#### Install from source

```
git clone https://github.com/khushi-411/ptMgrad.git
cd ptMgrad

git submodule sync
git submodule update --init --recursive
```

#### Build

```
mkdir build && cd build && cmake .. && make && cd ..
```

#### Run tests

```
./build/ptMgrad_tests

# or
make test
```

#### Run neural network model

```
g++ -O3 -std=c++17 -Isrc model.cpp -o model && ./model
```

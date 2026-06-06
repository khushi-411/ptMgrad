### ptMgrad = pt (PyTorch) + Mgrad (micrograd)

This is a simple neural network library written from scratch in
in modern C++, inspired by the [PyTorch](https://github.com/pytorch/pytorch)
and Andrej Karpathy's [micrograd](https://github.com/karpathy/micrograd).

The goal was to explore how deep-learning frameworks work under the hood
by implementing automatic differentiation, computational graphs, and
neural-network building blocks from first principles, without relying on
existing libraries. It also supports complex neural-networks
architectures.

**Note**: `std::complex` and `std::vector`/`std::array` are
intentionally avoided in favor of custom-built `complex<T>` and `Array<T>`
types, written entirely from scratch.

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

***Note:*** I took help from AI assistance for C++ memory management issues and
the iterative `backward()` topological sort.

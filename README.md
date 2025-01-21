### ptMgrad = pt (PyTorch) + Mgrad (micrograd)

This project is inspired by the [PyTorch](https://github.com/pytorch/pytorch)
and Andrej Karpathy's [microgrid](https://github.com/karpathy/micrograd).

#### Install from source

```
git clone https://github.com/khushi-411/ptMgrad.git
cd ptMgrad

git submodule sync
git submodule update --init --recursive
```

#### Build

```
mkdir build && cd build && cmake .. && make
```

#### Run tests

```
./build/ptMgrad_tests
```
or
```
make test
```
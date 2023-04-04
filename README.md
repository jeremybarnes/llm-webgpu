# Large Language Models on WebGPU

Personal experiment to see how it's possible to convert LLMs to WebGPU and run them,
comparing with native runtimes.

It's being done on an M1 Max machine, so part of it is debugging MPS (the Metal Performance Shaders)
so that they can run those models reliably.

## Setup

As of 2022-12-30, PyTorch isn't packaged for Python 3.11.

```
virtualenv -p python310 virtualenv
pip install transformers torch accelerate ansi natsort
```

# Running

Currently:

```
. ./virtualenv/bin/activate
python ./gptj.py
```

# Building a PyTorch version for debugging

(from the pytorch source directory)

```
pip install transformers torch accelerate
DEBUG=1 USE_DISTRIBUTED=1 MACOSX_DEPLOYMENT_TARGET=10.9 CC=clang CXX=clang++ python3 setup.py develop
```


#!/usr/bin/env python
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

import glob
import os
from setuptools import find_packages, setup
import torch
from torch.utils.cpp_extension import CUDA_HOME, CppExtension, CUDAExtension

torch_ver = [int(x) for x in torch.__version__.split(".")[:2]]
assert torch_ver >= [1, 3], "Requires PyTorch >= 1.3"

def get_extensions():
    this_dir = os.path.dirname(os.path.abspath(__file__))
    extensions_dir = os.path.join("bua", "caffe", "modeling", "layers", "csrc")

    sources_set = set()

    main_source = os.path.join(extensions_dir, "vision.cpp")
    sources_set.add(main_source)

    other_sources = glob.glob(os.path.join(extensions_dir, "**", "*.cpp"), recursive=True)
    for source in other_sources:
        rel_source = os.path.relpath(source, this_dir) if os.path.isabs(source) else source
        sources_set.add(rel_source)

    source_cuda = set()
    cuda_files = glob.glob(os.path.join(extensions_dir, "**", "*.cu"), recursive=True) + glob.glob(
        os.path.join(extensions_dir, "*.cu")
    )
    for cuda_file in cuda_files:
        rel_cuda = os.path.relpath(cuda_file, this_dir) if os.path.isabs(cuda_file) else cuda_file
        source_cuda.add(rel_cuda)

    sources = list(sources_set)

    extension = CppExtension

    extra_compile_args = {"cxx": []}
    define_macros = []

    if (torch.cuda.is_available() and CUDA_HOME is not None) or os.getenv("FORCE_CUDA", "0") == "1":
        extension = CUDAExtension
        sources += list(source_cuda)
        define_macros += [("WITH_CUDA", None)]
        extra_compile_args["nvcc"] = [
            "-DCUDA_HAS_FP16=1",
            "-D__CUDA_NO_HALF_OPERATORS__",
            "-D__CUDA_NO_HALF_CONVERSIONS__",
            "-D__CUDA_NO_HALF2_OPERATORS__",
        ]

    include_dirs = [extensions_dir]

    print("Sources to compile:")
    for source in sources:
        print(f"  {source}")

    ext_modules = [
        extension(
            "bua.caffe.modeling._C",
            sources,
            include_dirs=include_dirs,
            define_macros=define_macros,
            extra_compile_args=extra_compile_args,
        )
    ]

    return ext_modules


setup(
    name="bottom-up-attention.pytorch",
    packages=find_packages(exclude=("configs", "tests")),
    python_requires=">=3.6",
    ext_modules=get_extensions(),
    cmdclass={"build_ext": torch.utils.cpp_extension.BuildExtension},
)

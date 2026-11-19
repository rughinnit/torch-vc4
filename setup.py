import os

from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CppExtension

ENGINE = os.environ.get("VC4_ENGINE_DIR", "/home/raspi/work/engine-daemon")

setup(
    name="torch_vc4",
    version="0.1.0",
    description="Minimal honest PyTorch PrivateUse1 backend for VideoCore IV (VC4)",
    packages=["torch_vc4"],
    ext_modules=[
        CppExtension(
            "torch_vc4._C",
            ["csrc/torch_vc4.cpp"],
            include_dirs=[os.path.join(ENGINE, "include")],
            library_dirs=[os.path.join(ENGINE, "build")],
            libraries=["vc4infer"],
            # -O0/-g0: torch's headers are big; on this 1 GB board the
            # compile peaks near 1 GB of address space (measured), so keep
            # codegen cheap.  The hot code is on the QPU, not here.
            extra_compile_args=["-O0", "-g0", "-std=c++20",
                                "-fno-strict-aliasing"],
            extra_link_args=[
                "-Wl,-rpath," + os.path.join(ENGINE, "build"),
                "-Wl,-rpath-link," + os.path.join(ENGINE, "build"),
            ],
        )
    ],
    cmdclass={"build_ext": BuildExtension},
    python_requires=">=3.9",
)

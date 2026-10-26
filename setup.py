"""Builds the optional C++ BM25 extension. Project metadata lives in pyproject.toml."""

from pybind11.setup_helpers import Pybind11Extension, build_ext
from setuptools import setup


class OptionalBuildExt(build_ext):
    """Never fail the install over the extension: conduit falls back to pure Python."""

    def run(self):
        try:
            super().run()
        except Exception as exc:  # noqa: BLE001 - any compiler failure means "use the fallback"
            print(f"WARNING: C++ BM25 extension not built ({exc}); using the pure-Python fallback.")

    def build_extension(self, ext):
        try:
            super().build_extension(ext)
        except Exception as exc:  # noqa: BLE001
            print(f"WARNING: could not build {ext.name} ({exc}); using the pure-Python fallback.")


setup(
    ext_modules=[Pybind11Extension("conduit._bm25", ["cpp/bm25.cpp"], cxx_std=17)],
    cmdclass={"build_ext": OptionalBuildExt},
)

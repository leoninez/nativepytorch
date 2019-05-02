from setuptools import setup, find_packages, Extension

emb_module = Extension('emb',
                        sources=['main.cpp'])

setup(
    name="emb",
    version="1.0.0",
    packages=find_packages(),
    ext_modules=[emb_module]
)

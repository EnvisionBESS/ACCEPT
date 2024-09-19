from setuptools import setup, find_packages

setup(
    name="accept",
    py_modules=["accept"],
    version="1.0",
    description="",
    author="EnvisionEnergy",
    packages=find_packages(exclude=["tests*"]),
    install_requires=[],
    include_package_data=True,
    extras_require={'dev': ['pytest']},
)
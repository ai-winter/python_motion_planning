from setuptools import setup, find_packages

setup(
    name="python_motion_planning",
    version="1.1",
    author="Winter",
    packages=find_packages(),
    install_requires=[
        "numpy",
        "matplotlib",
        "cvxopt"
    ]
)
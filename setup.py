"""
setup.py - FedGuard Package Setup
"""
from setuptools import setup, find_packages

with open("requirements.txt") as f:
    requirements = f.read().splitlines()

with open("README.md", "r", encoding="utf-8") as f:
    long_description = f.read()

setup(
    name="fedguard",
    version="1.0.0",
    author="FedGuard Team",
    author_email="contact@fedguard.dev",
    description="Privacy-Preserving Adversarially Robust IDS for SDN",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/amanverma420/fedguard",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Security",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    python_requires=">=3.8",
    install_requires=requirements,
    entry_points={
        "console_scripts": [
            "fedguard=main:main",
            "fedguard-gui=app:main",
        ],
    },
    include_package_data=True,
    zip_safe=False,
)
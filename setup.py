import sys
from setuptools import setup, find_packages
from os import path as osp

if sys.version_info < (3, 6):
    sys.exit("Sorry, Python < 3.6 is not supported")

BASE_DIR = osp.abspath(osp.dirname(__file__))

with open("README.md", "r", encoding="utf-8") as fp:
    long_description = fp.read()

with open("requirements.txt") as fp:
    packages = fp.readlines()
packages = [x.strip() for x in packages]

setup(
    name="nmag",
    version="0.1.0",
    author="sichaoy",
    author_email="sichao.young@gmail.com",
    license="MIT Licence",
    description=("A collection of useful commands for note management"),
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="",
    packages=find_packages(),
    package_data={
        "examples": ["*.sh", "*.pdf", "*.md", "*.png", "*.txt"],
    },
    python_requires='>=3.6',
    install_requires=packages,
    entry_points={
        # 'console_scripts': ['mycli=mymodule:cli'],
        "console_scripts": [
            "nmag=nmag.cli:run",
        ],
    },
    include_package_data=True,
    zip_safe=False,
    classifiers=[
        "Programming Language :: Python :: 3.6",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
    ],
)

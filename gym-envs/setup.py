import setuptools
from pathlib import Path

setuptools.setup(
    name='gym_envs',
    version='0.0.1',
    description="My test envs",
    long_description=Path("README.md").read_text(),
    long_description_content_type="text/markdown",
    packages=setuptools.find_packages(include="gym_envs*"),
    install_requires=['gymnasium']
)
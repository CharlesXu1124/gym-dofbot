import setuptools
from pathlib import Path

setuptools.setup(
    name='gym_dofbot',
    version='0.0.2',
    description="A OpenAI Gym Env for Yahboom DOFBOT robotic arm",
    long_description=Path("README.md").read_text(),
    long_description_content_type="text/markdown",
    packages=setuptools.find_packages(include="gym_dofbot*"),
    install_requires=['gym']  # And any other dependencies foo needs
)
from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="luckyrobots",
    version="0.1.35",
    description="Robotics-AI Training in Hyperrealistic Game Environments",
    long_description=long_description,
    long_description_content_type="text/markdown",
    author="Devrim Yasar",
    author_email="braces.verbose03@icloud.com",
    url="https://github.com/lucky-robots/lucky-robots",
    packages=find_packages(),
    install_requires=[
        "watchdog",
        "fastapi",
        "uvicorn",
        "requests",
        "tqdm",
        "beautifulsoup4",
        "psutil",
        "packaging",
    ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.8',
)
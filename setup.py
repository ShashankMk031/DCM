from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

with open("requirements.txt", "r") as f:
    requirements = [line.strip() for line in f if line.strip() and not line.startswith("#")]

setup(
    name="dcm",
    version="0.1.0",
    author="Your Name",
    author_email="your.email@example.com",
    description="DCM (Desktop/Mobile Music Assistant) - A privacy-focused, offline music player with AI-powered recommendations",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/dcm",
    packages=find_packages(exclude=["tests"]),
    package_data={
        "dcm": ["py.typed"],
    },
    install_requires=requirements,
    entry_points={
        "console_scripts": [
            "dcm=dcm.__main__:cli",
        ],
    },
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: End Users/Desktop",
        "License :: OSI Approved :: GNU General Public License v3 or later (GPLv3+)",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Operating System :: OS Independent",
        "Topic :: Multimedia :: Sound/Audio :: Players",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    python_requires=">=3.8",
    keywords="music audio player offline ai recommendations",
    project_urls={
        "Bug Reports": "https://github.com/yourusername/dcm/issues",
        "Source": "https://github.com/yourusername/dcm",
    },
)

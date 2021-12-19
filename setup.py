import setuptools


with open("README.md", "r", encoding="UTF-8") as readme:
    long_description = readme.read()
with open("requirements.txt", "r", encoding="UTF-8") as requirements:
    install_requires = requirements.read().split("\n")


setuptools.setup(
    name="ads-evt",
    version="0.0.1",
    author="limjcst",
    description="Anomaly Detection in Streams with Extreme Value Theory",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/limjcst/ads-evt",
    project_urls={
        "Bug Tracker": "https://github.com/limjcst/ads-evt/issues",
    },
    packages=["ads_evt"],
    install_requires=install_requires,
    classifiers=[
        "Programming Language :: Python :: 3.6",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.6",
    license="GNU GPLv3",
)

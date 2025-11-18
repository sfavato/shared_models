from setuptools import setup, find_packages

setup(
    name="shared-models",  # Unique name for the package
    version="0.1.15",       # Package version
    packages=find_packages(exclude=["tests*"]),
    include_package_data=True,
    install_requires=[],   # Add any dependencies here
    description="Shared model classes for my harmonic project usage",
    author="Vincent Schmitt",
    author_email="vsc77420@gmail.com",
    url="https://github.com/sfavato/shared-models",  # Repository URL
)

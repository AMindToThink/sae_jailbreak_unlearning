from setuptools import setup, find_packages

setup(
    name="sae_jailbreak_unlearning",
    version="0.1",
    package_dir={"": "src"},
    packages=find_packages(where="src"),
    install_requires=[
        # Add your dependencies here
    ],
)
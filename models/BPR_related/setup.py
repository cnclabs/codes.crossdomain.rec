import setuptools

with open('requirements.txt') as f:
    requirements = f.read().splitlines()

setuptools.setup(
        name="CPR-BPR_related",
        install_requires=requirements,
        
        )

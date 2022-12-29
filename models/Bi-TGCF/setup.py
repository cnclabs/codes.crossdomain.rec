import setuptools

with open('requirements.txt') as f:
    requirements = f.read().splitlines()

setuptools.setup(
        name="CPR-BiTGCF",
        install_requires=requirements,
        python_requires='>=3.7'
        
        )

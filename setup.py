import setuptools

with open('requirements.txt') as f:
    install_requires = f.read().splitlines()

setuptools.setup(
    name='two_stage_pipeliner',
    version='0.1.0',
    include_package_data=True,
    packages=setuptools.find_packages('two_stage_pipeliner'),
    install_requires=install_requires,
    python_requires='>=3.8'
)

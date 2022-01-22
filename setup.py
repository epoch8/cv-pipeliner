import setuptools


def get_version(rel_path):
    for line in open(rel_path, 'r').read().splitlines():
        if line.startswith('__version__'):
            delim = '"' if '"' in line else "'"
            return line.split(delim)[1]
    else:
        raise RuntimeError("Unable to find version string.")


with open('requirements.txt') as f:
    install_requires = f.read().splitlines()

setuptools.setup(
    name='cv_pipeliner',
    version=get_version('cv_pipeliner/__init__.py'),
    include_package_data=True,
    packages=setuptools.find_packages(),
    install_requires=install_requires,
    python_requires='>=3.8'
)

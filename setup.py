#!/usr/bin/env python3
import setuptools

# load the README file and use it as the long_description for PyPI
with open('README.md', 'r') as f:
    readme = f.read()

# package configuration - for reference see:
# https://setuptools.readthedocs.io/en/latest/setuptools.html#id9
setuptools.setup(
    name='autofocus',
    description='Toolbox for autofocusing.',
    long_description=readme,
    long_description_content_type='text/markdown',
    version='0.1.0',
    author='Issam Manai, Viktor Truderung',
    author_email='issam.manai@itwm-extern.fraunhofer.de, viktor@truderung.com',
    url='',
    namespace_packages=['autofocus'],
    packages=setuptools.find_packages(),
    include_package_data=True,
    python_requires=">=3.6.*",
    install_requires=[],
    license='FREE',
    classifiers=[
        'Development Status :: 1',
        'Intended Audience :: Developers',
        'Programming Language :: Python :: 3.6',
    ],
    keywords=''
)

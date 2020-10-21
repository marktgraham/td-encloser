import setuptools


setuptools.setup(
    name='td-encloser',
    version='0.0.2',
    description='A set of tools to optimise the allocation of counterparties.',
    author='Mark Graham',
    license='MIT License',
    packages=setuptools.find_packages(),
    install_requires=[
        'numpy>=1.19.1',
        'pandas>=1.1.0',
        'matplotlib>=3.3.0',
        'tqdm>=4.48.2',
        'scipy>=1.1.0'
    ]
)
from setuptools import setup

setup(
    name='OS_model',

    version='0.1',

    description='Code to simulate cell-cell interactions and differential adhesion experiments'
                'via an overlapping spheres model with linear force laws and a forward Euler time step',

    license='MIT',

    packages=['OS_model'],

    install_requires=[
        'numpy==1.19.3',
        'scipy',
        'matplotlib'
    ],

    extras_require={
        'docs': [
            'sphinx'
        ],
        'dev': [
            'flake8',
            'pytest',
            'coverage',
            'codecov'
        ],
    },

    zip_safe=False
)

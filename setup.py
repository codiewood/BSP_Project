from setuptools import setup

setup(
    name='ci_course',

    version='0.1',

    description='Demonstration of setting up continuous integration for a Python project',

    license='MIT',

    packages=['ci_course'],

    install_requires=[
        'numpy==1.19.3'
    ],

    extras_require={
        'docs': [
            'sphinx'
        ],
        'dev': [
            'flake8',
            'pytest',
        ],
    },

    zip_safe=False
)

![Test on Python 3.6-3.9 & Windows, Linux and macOS.](https://github.com/codiewood/BSP_Project/workflows/Test%20on%20Python%203.6-3.9%20&%20Windows,%20Linux%20and%20macOS./badge.svg)

[![codecov](https://codecov.io/gh/codiewood/BSP_Project/branch/main/graph/badge.svg?token=Z5LVN7MTB4)](https://codecov.io/gh/codiewood/BSP_Project)

[![BCH compliance](https://bettercodehub.com/edge/badge/codiewood/BSP_Project?branch=main)](https://bettercodehub.com/)

[![Documentation Status](https://readthedocs.org/projects/part-b-structured-project/badge/?version=latest)](https://part-b-structured-project.readthedocs.io/en/latest/?badge=latest)

# OxRSE Continuous Integration course

This project contains a small Python project. We are going to use free cloud services to automate:

- unit testing on multiple Python versions
- unit testing on multiple operating systems
- coverage testing
- static analysis
- documentation generation

To make sure all dependencies are installed, we recommend creating a new virtual environment.
From the directory containing this file:

```bash
python3 -m pip install --user virtualenv
python3 -m venv venv
```

Activate the virtual environment:

Linux / macOS:
```bash
source venv/bin/activate
```

Windows cmd.exe:
```bash
venv\Scripts\activate.bat
```

Windows PowerShell:
```bash
venv\Scripts\Activate.ps1
```

Windows using Git Bash:
```bash
source venv\Scripts\activate
```

Upgrade the build tools and install this project:

```bash
pip install --upgrade pip setuptools wheel
pip install -e .[dev,docs]
```

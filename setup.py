# Copyright 2023 Maya Bhat
# (see accompanying license files for details).
from setuptools import setup

setup(name='doe',
      version='0.0.1',
      description='sequential batch sampling - 3 methods',
      url='',
      maintainer='Maya Bhat',
      maintainer_email='mayabhat@andrew.cmu.edu',
      license='GPL',
      platforms=['linux'],
      packages=['sbs'],
      setup_requires=[],
      data_files=[],
      install_requires=['autograd', 'pyDOE2', 'sklearn', 'matplotlib', 'numpy', 'pycse', 'pandas', 'shapely'],
      long_description='''Python module containing 3 methods of sequential batch sampling - broad initial sampling, low point-density sampling, and path tracing.''')

# (shell-command "python setup.py register") to setup user
# to push to pypi - (shell-command "python setup.py sdist upload")


# Set TWINE_USERNAME and TWINE_PASSWORD in .bashrc
# python setup.py sdist bdist_wheel
# twine upload dist/*

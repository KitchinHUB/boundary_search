# Copyright 2023 Maya Bhat
# (see accompanying license files for details).
from setuptools import setup

setup(name='sbs',
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
      install_requires=['autograd', 'pyDOE2', 'scikit-learn',
                        'matplotlib', 'numpy', 'pycse', 'pandas',
                        'shapely'],
      long_description='''Python module containing 3 methods of sequential batch sampling - broad initial sampling, low point-density sampling, and path tracing.''')


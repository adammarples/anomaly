# encoding: utf-8

from setuptools import find_packages, setup

with open('requirements.txt', 'r') as fi:
    required_packages = fi.readlines()

setup(name='anomaly',
      description='An anomaly detection module for pandas DateTimeSeries objects.',
      author='Adam Marples',
      author_email='adam.marples@iprospect.com',
      version='0.0.1',
      packages=find_packages(),
      keywords='anomaly detection pandas datetimeindex',
      install_requires=required_packages,
      test_suite='tests'
      )
from setuptools import setup, find_packages


setup(name='omnicalib',
      version='1.0',
      description='Omnidirectional camera calibration',
      author='Thomas Pönitz',
      author_email='tasptz@gmail.com',
      url='https://github.com/tasptz/py-omnicalib',
      packages=find_packages(exclude=['omnicalib_tests']),
      license='MIT',
      install_requires=[
          'autograd',
          'matplotlib',
          'opencv-python-headless',
          'pyyaml',
          'scipy',
          'torch>=1.10',
          'tqdm'
      ]
      )

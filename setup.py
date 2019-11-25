from setuptools import setup

setup(name='lupi_svm',
      version='1.0',
      description='LUPI SVM library in Python',
      long_description=open("README.rst").read(),
      url='http://github.com/perspectalabs/lupi_svm',
      author='Perspecta Labs',
      license='MIT',
      packages=['lupi_svm'],
      install_requires=[
          'scikit-learn==0.19.1',
          'pandas>=0.22.0',
          'scipy>=1.2.1',
          'numpy>=1.14.3'
      ],
      zip_safe=False)

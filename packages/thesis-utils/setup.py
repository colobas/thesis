from setuptools import setup

setup(name='thesis-utils',
      version='0.0',
      description='Guilherme Pires MSc Thesis',
      author='Guilherme Pires',
      author_email='mail@gpir.es',
      packages=['thesis_utils'],
      entry_points={
          'console_scripts': ['tboardX_extract_images=thesis_utils.extract_images:main'],
      },
      zip_safe=False)

from setuptools import setup

setup(name='heudiconv_helpers',
      version='0.0.5',
      description='Some packages to help get datasets bids compliant after heudiconv processing',
      url='https://github.com/nih-fmrif/heudiconv_helpers',
      author='John Lee, Dylan Nielson',
      author_email='nimhdsst@gmail.com',
      license='Public Domain',
      packages=['heudiconv_helpers'],
      install_requires=[
          'numpy',
          'pandas',
          'pytest',
          'heudiconv'
      ],
      zip_safe=False)

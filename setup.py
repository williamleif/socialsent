from setuptools import setup

setup(name='socialsent',
      version='0.1.2',
      description='Algorithms for inducing domain-specific sentiment lexicons from unlabeled corpora.',
      url='https://github.com/williamleif/socialsent',
      author='William Hamilton',
      author_email='wleif@stanford.edu',
      license='Apache Version 2',
      packages=['socialsent'],
      install_requires = ['numpy',
                          'keras==0.3',
                          'sklearn',
                          'theano'],
      package_data= {'socialsent' : ['data/lexicons/*.json']},
      zip_safe=False)

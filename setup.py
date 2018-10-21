from distutils.core import setup
import setuptools

short_description = "Create, visualise and leverage networks of ANPR cameras on the road network."

long_description = \
"""
**ANPRx** is a package for traffic analytics using networks of automatic number plate cameras.
"""

classifiers = ['Development Status :: 1 - Planning',
               'License :: OSI Approved :: MIT License',
               'Operating System :: OS Independent',
               'Intended Audience :: Science/Research',
               'Topic :: Scientific/Engineering :: GIS',
               'Topic :: Scientific/Engineering :: Visualization',
               'Topic :: Scientific/Engineering :: Physics',
               'Topic :: Scientific/Engineering :: Mathematics',
               'Topic :: Scientific/Engineering :: Information Analysis',
               'Programming Language :: Python',
               'Programming Language :: Python :: 2',
               'Programming Language :: Python :: 2.7',
               'Programming Language :: Python :: 3',
               'Programming Language :: Python :: 3.4',
               'Programming Language :: Python :: 3.5',
               'Programming Language :: Python :: 3.6']

with open('requirements.txt') as f:
    requirements_lines = f.readlines()
install_requires = [r.strip() for r in requirements_lines]

# now call setup
setup(name='anprx',
      version='0.1.0',
      description=short_description,
      long_description=long_description,
      classifiers=classifiers,
      url='https://github.com/pedroswits/anprx',
      author='Pedro Pinto da Silva',
      author_email='ppintodasilva@gmail.com',
      license='MIT',
      platforms='any',
      packages=['anprx'],
      install_requires=install_requires)

from distutils.core import setup
import setuptools

short_description = "Create, visualise and leverage networks of ANPR cameras on the road network."

long_description = \
"""
**ANPRx** is a package for traffic analytics using networks of automatic number plate cameras.
"""

classifiers = ['Development Status :: 3 - Alpha',
               'License :: OSI Approved :: Apache Software License',
               'Operating System :: OS Independent',
               'Intended Audience :: Science/Research',
               'Topic :: Scientific/Engineering :: GIS',
               'Topic :: Scientific/Engineering :: Visualization',
               'Topic :: Scientific/Engineering :: Physics',
               'Topic :: Scientific/Engineering :: Mathematics',
               'Topic :: Scientific/Engineering :: Information Analysis',
               'Programming Language :: Python',
               'Programming Language :: Python :: 3.6']

install_requires = [
    'numpy >= 1.15.1',
    'statistics >= 1.0.3',
    'requests >= 2.19.1',
    'Shapely >= 1.6',
    'pandas >= 0.23',
    'geopandas >= 0.4',
    'matplotlib >= 2.2',
    'networkx >= 2.2',
    'osmnx >= 0.8.1',
    'scikit-learn >= 0.20.0',
    'adjustText >= 0.7.3',
    'progress >= 1.4']

extras_require = {
    'tests': [
       'tox >= 3.2.1',
       'pytest >= 3.8.2'],
    'docs': [
       'sphinx >= 1.4',
       'sphinx_rtd_theme'],
    'examples': [
       'ipykernel']}

dependency_links = [
    'http://github.com/pedroswits/osmnx/tarball/master#egg=package-1.0']

# now call setup
setup(name = 'anprx',
      version = '0.1.3',
      description = short_description,
      long_description = long_description,
      classifiers = classifiers,
      url = 'https://github.com/pedroswits/anprx',
      author = 'Pedro Pinto da Silva',
      author_email = 'ppintodasilva@gmail.com',
      license = 'Apache License 2.0',
      platforms = 'any',
      packages = ['anprx'],
      install_requires = install_requires,
      extras_require = extras_require,
      dependency_links = dependency_links)

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
               'Programming Language :: Python :: 3.6']

with open('requirements.txt') as f:
    requirements_lines = f.readlines()
install_requires = [r.strip() for r in requirements_lines]

# now call setup
setup(name = 'anprx',
      version = '0.1.1',
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
      extras_require = {
        'tests': [
            'tox >= 3.2.1',
            'pytest >= 3.8.2'],
        'docs': [
            'sphinx >= 1.4',
            'sphinx_rtd_theme'],
        'examples': [
            'ipykernel']})

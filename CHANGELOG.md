# Changelog

## [0.1.1] - 22/10/2018

Patch that implements fixes/improvements to docs, CI and packaging.

### Added
- Added changelog.
- Added mention of important C-library requirement for OSMnx to readme.

### Changed
- The wording of 'fake' to 'mock' cameras on readme. Figures changed accordingly.

## Removed
- Removed old, unused python 2 dependencies from requirements.txt

### Fixed
- Travis now runs without errors:
	- Added tests/data/.gitkeep.
	- Removed py27 target
- Docs now compile properly on readthedocs:
	- Added autodoc_mock_imports to docs/conf.py
- setup.py now displays the correct information:
	- no support for python 2
	- added extra_requires field

## [0.1.0] - 21/10/2018

First minor release.

### Added
- Implemented the following main features:
	- Get the drivable street network (nx.MultiDiGraph) that encompasses a set of points (lat, lng).
	- Instantiate Camera objects capable of computing the likelihood of neighbouring edges as the true edge observed by the camera.
	- Candidate edges can be filtered based on address.
	- Plot the street network around the camera and highlight nearby nodes and candidate edges based on their computed probability.
- Plots are saved to images folder in app's folder.

### Changed
- LICENSE - now Apache v2.0
- OSMnx is now configured initially by anprx, and everytime app_folder is chaged, and exists as a subfolder of app_folder. Its cache is set true by default
- Log formatting

### Removed
- Removed support for python 2.

## [0.0.0] - 5/09/2018

Bare bones project skeleton.

### Added
- Sphinx and autodoc as build engines for docs. Added automethod example using a dummy method.
- Tests run using pytest. Added dummy test example.
- OSMnx as the main package dependency.
- Tox as the main driver of testing.
- Continuous integration with .travis.yml and support for tox.
- Logging ability.

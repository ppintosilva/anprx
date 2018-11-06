# Changelog

## [0.1.2] - 31/10/2018

Refactoring patch - plot, animate and edge attributes.

### Added

- Animation method to show, with explanation, how the camera's edge is being estimated.
- Method that combines Edges in the network can now be 'enriched' with additional attributes: bearing, grade (calculated from node elevation), address details (road, suburb, postcode - outer and inner, OSM importance metric, type)
- Address details can be added as attributes to every edge in the network
- Address details, of up to 50 osmids, can be retrived from Nominatim
- Dead end nodes can be retrieved and removed from the network graph.

### Changed
- Increased default value for 'mean_area' to 0.3 km^2 in `anprx.navigation.get_surrounding_network`
- Adjusted log indentation
- Moved camera.plot method to a separate source file - the method is now called plot_camera

### Removed
- Removed import * from anprx submodules `__init__.py`

### Refactored
- navigation.py source file - moved to core.py
- anprx imports in test files
- `__init__.py` now imports anprx submodules

### Fixed
- Errors and typos in docstrings

## [0.1.1] - 22/10/2018

Patch that implements fixes/improvements to docs, CI and packaging.

### Added
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

[![Build Status](https://travis-ci.org/PedrosWits/anprx.svg?branch=master)](https://travis-ci.org/PedrosWits/anprx)
[![Build Status - master(https://anprx.readthedocs.io/en/latest/?badge=latest)](anprx.readthedocs.io)

# ANPRx

Traffic analysis using networks of ANPR cameras (Automatic Number Plate Recognition).

## Installation

ANPRX is available through pypi:
```
pip install anprx
```

See `requirements.txt` for a complete list of dependencies.

## Features

#### Stable

Given a pair of latitude and longitude coordinates for each camera:

- Obtain a model of the drivable street network, using [osmnx](https://github.com/gboeing/osmnx) and [networkx](https://networkx.github.io/documentation/stable/index.html), that encompasses the cameras (coordinate points).
- Compute the likelihood of neighbouring edges (road segments) as the true edge observed by the camera. Filter out candidate edges by address.
- Visualise camera placement, nearby nodes, and the likelihood of candidate edges.

#### Under development

Among others:

- Enrich the road network by adding further attributes to the edges (address details, elevation, bearing).
- Filter/compress the road network based on edge attributes.
- Batch analysis of ANPR data: trip identification and inference.

## Documentation

All modules and methods are documented in anprx.readthedocs.io

## License
Apache v2.0. See LICENSE


## Copyright(c) 2024, 2025 Yoann Robin
## 
## This file is part of zxarray.
## 
## zxarray is free software: you can redistribute it and/or modify
## it under the terms of the GNU General Public License as published by
## the Free Software Foundation, either version 3 of the License, or
## (at your option) any later version.
## 
## zxarray is distributed in the hope that it will be useful,
## but WITHOUT ANY WARRANTY; without even the implied warranty of
## MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
## GNU General Public License for more details.
## 
## You should have received a copy of the GNU General Public License
## along with zxarray.  If not, see <https://www.gnu.org/licenses/>.

###############
## Libraries ##
###############

import setuptools
from setuptools import setup
from pathlib import Path


############################
## Python path resolution ##
############################

cpath = Path(__file__).parent


########################
## Infos from release ##
########################

list_packages = setuptools.find_packages()
package_dir = { "zxarray" : "zxarray" }
release = {}
exec( (cpath / "zxarray" / "__release.py").read_text() , {} , release )


#################
## Description ##
#################
long_description = (cpath / "README.md").read_text()


## Required elements
requires    = [
               "numpy",
               "xarray",
               "zarr",
               "netCDF4",
               "cftime"
              ]
keywords    = []
platforms   = ["linux","macosx"]


#######################
## And now the setup ##
#######################

setup(  name         = release['name'],
	version          = release['version'],
	description      = release['description'],
	long_description = long_description,
	long_description_content_type = 'text/markdown',
	author           = release['author'],
	author_email     = release['author_email'],
	url              = release['src_url'],
	packages         = list_packages,
	package_dir      = package_dir,
	requires         = requires,
	license          = release['license'],
	keywords         = keywords,
	platforms        = platforms,
	classifiers      = [
		"License :: OSI Approved :: GNU General Public License v3 or later (GPLv3+)",
		"Natural Language :: English",
		"Operating System :: MacOS :: MacOS X",
		"Operating System :: POSIX :: Linux",
		"Programming Language :: Python :: 3",
		"Programming Language :: Python :: 3.11",
		"Programming Language :: Python :: 3.12",
		"Programming Language :: Python :: 3.13",
		"Topic :: Scientific/Engineering :: Mathematics"
	],
		include_package_data = True
    )



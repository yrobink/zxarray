
## Copyright(c) 2024 Yoann Robin
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


## Start by import release details
import os

cpath = os.path.dirname(os.path.abspath(__file__)) ## current-path
with open( os.path.join( cpath , "zxarray" , "__release.py" ) , "r" ) as f:
    lines = f.readlines()
exec("".join(lines))

## Required elements
package_dir = { "zxarray" : "zxarray" }
requires    = [
               "numpy",
               "xarray",
               "zarr",
               "netCDF4",
               "cftime"
              ]
keywords    = []
platforms   = ["linux","macosx"]
packages    = [
    "zxarray",
    ]

## Now the setup
from distutils.core import setup

setup(  name             = name,
        version          = version,
        description      = description,
        long_description = long_description,
        author           = author,
        author_email     = author_email,
        url              = src_url,
        packages         = packages,
        package_dir      = package_dir,
        requires         = requires,
        license          = license,
        keywords         = keywords,
        platforms        = platforms,
		include_package_data = True
    )



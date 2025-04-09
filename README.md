# zxarray

## Description

zxarray is an interface of zarr similar to xarray.  The idea is to benefit from
the advantages of xarray (coordinate access, parallel apply), while keeping
memory usage low by storing data in zarr. To achieve this:
- zxarray.ZXArray have accessors similar to those of xarray, such as `.loc`,
  `.sel`, `.isel`.
- an `apply_ufunc` function calling the xarray function enables parallel
  calculation via dask.


## Install

zxarray can be installed via pip:

~~~bash
pip install zxarray
~~~


## License

Copyright(c) 2024, 2025 Yoann Robin

This file is part of zxarray.

zxarray is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

zxarray is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with zxarray.  If not, see <https://www.gnu.org/licenses/>.


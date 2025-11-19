
## Copyright(c) 2025 Yoann Robin
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

#############
## Imports ##
#############


from .__apply_ufunc import apply_ufunc

import numpy as np

from typing import Sequence


###############
## Functions ##
###############

def anomaly( X: np.ndarray , axis: int | Sequence = -1 ) -> np.ndarray:##{{{
    
    if isinstance(axis,int):
        axis = (axis,)
    axis = tuple(axis)

    shp = list(X.shape)
    for i in axis:
        shp[i] = 1

    A = X - X.mean( axis = axis ).reshape(shp)

    return A
##}}}

def zanomaly( zX , dims = None, **kwargs ):##{{{

    if dims is None:
        dims = zX.dims
    bdims = tuple([d for d in zX.dims if d not in dims])
    axis  = sorted([len(zX.dims) - 1 -i for i in range(len(dims))])
    
    dask_kwargs = {
        "input_core_dims" : [dims],
        "output_core_dims": [dims],
        "dask": "parallelized",
        "kwargs": { "axis": axis }
    }
    zA = apply_ufunc( anomaly, zX,
                     block_dims = bdims,
                     output_dims = [zX.dims],
                     output_coords = [{ d: zX[d] for d in zX.dims}],
                     dask_kwargs = dask_kwargs,
                     **kwargs
                     )
    
    return zA
##}}}



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

#############
## Imports ##
#############

import itertools as itt
import psutil

import numpy  as np
import xarray as xr

from .__DMUnit import DMUnit


###############
## Functions ##
###############

def apply_ufunc( func , *args , bdims : list | tuple = [] ,
					bsizes : tuple[int] | list[int] | None = None,
					max_mem : str | DMUnit | None = None ,
					fb_mem = None ,
					rb_mem : int = 3,
					output_coords : dict | list | None = None ,
					output_dims : list | tuple | None = None ,
					output_zfile : list | tuple | None = None ,
					output_dtypes : list | None = None ,
					dask_kwargs : dict = {} ,
					zarr_kwargs : dict = {} ):
	"""
	zxarray.apply_ufunc
	===================
	This function is an overlay to 'xarray.apply_ufunc'. The aim is to build
	blocks along the dimensions defined by 'bdims' in such a way that the call
	to 'xarray.apply_ufunc' does not exceed the maximum memory defined by
	'max_mem'.
	
	Parameters
	----------
	func: callable
		The function to apply.
	*args: tuple of zxarray.ZXArray
		Array to apply the function 'func'.
	bdims: list | tuple
		block-dims, list of dimensions to construct the blocks.
	bsizes: tuple or None
		block-sizes, the size of the block of length bdims. It is assumed that
		blocks defined in this way will not cause a memory overflow. Otherwise
		automatically estimated by 'fb_mem'.
	max_mem: zxarray.DMUnit | str | None
		Maximal memory available. Default is 80% of the total memory.
	fb_mem: callable
		Function that takes a block size as input and returns the amount of
		memory used by 'func'. Default is
			'rbmem' * size_of_input * size_of_output.
	rbmem: int
		Multiplicative factor used in 'fb_mem'.
	output_coords:
		List of output coordinates.
	output_dims:
		List of output dimensions.
	output_zfile:
		List of output zarr files.
	output_dtypes:
		List of output dtypes.
	dask_kwargs:
		Arguments passed to xarray.apply_ufunc.
	zarr_kwargs:
		Arguments passed to zarr.
	
	
	Returns
	-------
	zX: zxarray.ZXArray | tuple[zxarray.ZXArray]
		The result of the computation.
	"""
	## ZXArray class, to remove the import
	ZXArray = type(args[0])
	
	## Check output coordinates
	output_coords = list(output_coords)
	n_out = len(output_coords)
	if output_dims is None:
		output_dims = []
		for i in range(n_out):
			if not isinstance( output_coords[i] , dict ):
				raise ValueError( "If output_dims is not given, output_coords must be a list of a dict of coordinates" )
			output_dims.append(list(output_coords))
	else:
		for i in range(n_out):
			if not isinstance( output_coords[i] , dict ):
				output_coords[i] = { d : c for d,c in zip(output_dims[i],output_coords[i]) }
	
	## Check other ZXArray parameters
	if output_zfile is None:
		output_zfile = [ None for _ in range(n_out) ]
	if output_dtypes is None:
		output_dtypes = ["float32" for _ in range(n_out)]
	
	## Create output ZXArray
	zout = [ ZXArray( np.nan , dims = output_dims[i] , coords = output_coords[i] , zfile = output_zfile[i] , dtype = output_dtypes[i] , zarr_kwargs = zarr_kwargs ) for i in range(n_out) ]
	
	## Special case, len(bdims) == 0
	if len(bdims) == 0:
		xargs = [ arg.dataarray.values for arg in args ]
		xout  = func( *xargs , **dask_kwargs.get( "kwargs" , {} ) )
		for i in range(len(zout)):
			zout[i]._internal.zdata[:] = xout[i][:]
		return zout
	
	## Find block coords
	bcoords = []
	for d in bdims:
		for zX in args:
			if d in zX.dims:
				bcoords.append(zX.coords[d])
				break
	
	## Find block sizes
	if bsizes is None:
		
		## Memory used
		if max_mem is None:
			max_mem = DMUnit( n = int( 0.8 * psutil.virtual_memory().total ) , unit = 'B' )
		elif isinstance( max_mem , str ):
			max_mem = DMUnit( s = max_mem )
		elif not isinstance( max_mem , tuple([str,DMUnit]) ):
			raise ValueError( "'max_mem' argument class must be a string or zxarray.DMUnit" )
		
		## function_block_memory
		if fb_mem is None:
			total_unit_block = DMUnit.zero()
			for Z in itt.chain( args , zout ):
				try:
					nbits = np.finfo(Z.dtype).bits
				except:
					try:
						nbits = np.iinfo(Z.dtype).bits
					except:
						nbits = 64
				sizeZ = DMUnit( n = nbits // DMUnit.bits_per_octet , unit = 'o' )
				for d in Z.dims:
					if d in bdims: continue
					sizeZ *= Z.coords[d].size
				total_unit_block = total_unit_block + sizeZ
			fb_mem = lambda b: np.prod(b) * total_unit_block * rb_mem
		
		## Find block size
		nbdims  = len(bdims)
		bsizes  = [1 for _ in range(nbdims)]
		notfind = [True for _ in range(nbdims)]
		fsizes  = [bcoords[i].size for i in range(nbdims)]
		ssizes  = [bcoords[i].size for i in range(nbdims)]
		
		while any(notfind):
			i = np.argmin(ssizes)
			bsizes[i] = fsizes[i]
			while fb_mem(bsizes) > max_mem:
				if bsizes[i] < 2:
					bsizes[i] = 1
					break
				bsizes[i] = bsizes[i] // 2
			notfind[i] = False
			ssizes[i]  = np.inf
		mem_need = fb_mem(bsizes)
		
		if mem_need > max_mem:
			raise MemoryError( f"Insufficient memory, maximal memory lower than memory needed: max_mem = {max_mem} < {mem_need} = mem_need" )
	
	## Find dimensions of chunks
	chunks = [ { d : 1 for d in Z.dims if d not in icd } for Z,icd in zip(args,dask_kwargs["input_core_dims"]) ]
	if not len(chunks) == len(args):
		raise ValueError( f"Len of input_core_dims must match the numbers of input array" )
	
	## Loop on blocks
	for bx in itt.product(*[range(0,c.size,b) for c,b in zip(bcoords,bsizes)]):
		
		## Block indexes
		bidx = { bdims[i] : slice(bx[i],bx[i]+bsizes[i],1) for i in range(len(bsizes)) }
		
		## Extract array
		xargs = [ Z.isel( drop = False , **{ **{ d : slice(None) for d in Z.dims if d not in bdims } , **{ d : bidx[d] for d in bdims if d in Z.dims } } ).chunk(chunk) for Z,chunk in zip(args,chunks) ]
		
		## Apply
		res = xr.apply_ufunc( func , *xargs , **dask_kwargs )
		
		## Compute and transpose
		if isinstance(res,xr.DataArray):
			res = [res]
		else:
			res = list(res)
		res = [ R.compute().transpose(*Z.dims) for R,Z in zip(res,zout) ]
		
		## And save
		for i in range(n_out):
			xrcoords = { **{ d : slice(None) for d in zout[i].dims if d not in bdims } , **{ d : bidx[d] for d in bdims if d in zout[i].dims } }
			zidx     = [ xrcoords[d] for d in zout[i].dims ]
			zout[i][*zidx] = res[i].values
	
	if n_out == 1:
		return zout[0]
	
	return zout

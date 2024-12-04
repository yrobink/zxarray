
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

import logging
import warnings
import itertools as itt
import psutil
import multiprocessing
import gc

import numpy  as np
import xarray as xr

from .__DMUnit import DMUnit

##################
## Init logging ##
##################

logging.captureWarnings(True)
logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())

###############
## Functions ##
###############


class P_Func:
	
	def __init__( self , func , **kwargs ):
		self.func = func
		self.kwargs = kwargs
	
	def __call__( self , args ):
		return self.func( *args , **self.kwargs )

## map_ufunc {{{

def map_ufunc( func , *args , block_dims : list | tuple = [] ,
					total_memory : DMUnit | str = None,
					block_memory = None ,
					chunk_memory = None ,
					output_coords : dict | list | None = None ,
					output_dims : list | tuple | None = None ,
					output_dtypes : list | None = None ,
					output_zfile : list | tuple | None = None ,
					dask_kwargs : dict = {} ,
					zarr_kwargs : dict = {} ,
					n_workers : int = 1,
					threads_per_worker : int = 1,
					memory_per_worker : DMUnit | str = None,
					**kwargs
					):
	"""
	zxarray.map_ufunc
	===================
	This function is an overlay to 'xarray.apply_ufunc'. The aim is to build
	blocks along the dimensions defined by 'block_dims' in such a way that the call
	to 'xarray.apply_ufunc' does not exceed the maximum memory defined by
	'total_memory'.
	
	Parameters
	----------
	func: callable
		The function to apply.
	*args: tuple of zxarray.ZXArray
		Array to apply the function 'func'.
	block_dims: list | tuple
		block-dims, list of dimensions to construct the blocks.
	total_memory: zxarray.DMUnit | str | None
		Maximal memory available. Default is 80% of the total memory.
	block_memory: callable
		Function that takes a block size as input and returns the amount of
		memory used by 'func'. Default is
			4 * size_of_input * size_of_output.
	output_coords:
		List of output coordinates.
	output_dims:
		List of output dimensions.
	output_dtypes:
		List of output dtypes.
	output_zfile:
		List of output zarr files.
	dask_kwargs:
		Arguments passed to xarray.apply_ufunc.
	zarr_kwargs:
		Arguments passed to zarr.
	n_workers:
		Number of workers
	memory_per_worker:
		Can replace total_memory, the both can not be simultaneously set
	
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
	
	## Special case, len(block_dims) == 0
	if len(block_dims) == 0:
		xargs = [ arg.dataarray.values for arg in args ]
		xout  = func( *xargs , **dask_kwargs.get( "kwargs" , {} ) )
		for i in range(len(zout)):
			zout[i]._internal.zdata[:] = xout[i][:]
		return zout
	
	## Find block coords
	bcoords = []
	for d in block_dims:
		for zX in args:
			if d in zX.dims:
				bcoords.append(zX.coords[d])
				break
	
	## Find parallel parameters
	if total_memory is not None and memory_per_worker is not None:
		raise ValueError( "total_memory and memory_per_worker can not be set simultaneously" )
	if total_memory is None and memory_per_worker is None:
		total_memory = DMUnit( n = int( 0.8 * psutil.virtual_memory().total ) , unit = 'B' )
	if memory_per_worker is not None:
		memory_per_worker = DMUnit(memory_per_worker)
		total_memory = n_workers * memory_per_worker
	else:
		total_memory = DMUnit(total_memory)
	
	## function_block_memory
	if block_memory is None:
		logger.debug( "block_memory not given, infer it" )
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
				if d in block_dims: continue
				sizeZ *= Z.coords[d].size
			total_unit_block = total_unit_block + sizeZ
		block_memory = lambda b: np.prod(b) * total_unit_block * 4
	
	## Find block size
	nblock_dims  = len(block_dims)
	bsizes  = [1 for _ in range(nblock_dims)]
	notfind = [True for _ in range(nblock_dims)]
	fsizes  = [bcoords[i].size for i in range(nblock_dims)]
	ssizes  = [bcoords[i].size for i in range(nblock_dims)]
	
	while any(notfind):
		i = np.argmin(ssizes)
		bsizes[i] = fsizes[i]
		while block_memory(bsizes) > total_memory:
			if bsizes[i] < 2:
				bsizes[i] = 1
				break
			bsizes[i] = bsizes[i] - 1
		notfind[i] = False
		ssizes[i]  = np.inf
	memory_needed = block_memory(bsizes)
	
	if memory_needed > total_memory:
		raise MemoryError( f"Insufficient memory, maximal memory lower than memory needed: total_memory = {total_memory} < {memory_needed} = memory_needed" )
	logger.debug( f"Block size found: {bsizes}, memory_needed: {memory_needed}" )
	
	## Find dimensions of chunks
	chunks = [ { d : 1 for d in Z.dims if d not in icd } for Z,icd in zip(args,dask_kwargs["input_core_dims"]) ]
	chunked_dims = []
	for c in chunks:
		chunked_dims = chunked_dims + list(c)
	chunked_dims = set(chunked_dims)
	w_ratio = max( 1 , int(np.ceil( np.power( n_workers , 1 / len(chunked_dims) )  )) )
	chunks = [ { d : Z[d].size // w_ratio for d in Z.dims if d not in icd } for Z,icd in zip(args,dask_kwargs["input_core_dims"]) ]
	
	if not len(chunks) == len(args):
		raise ValueError( f"Len of input_core_dims must match the numbers of input array" )
	logger.debug( f"Chunk dimensions: {chunks}" )
	
	nblocks = len([ k for k in itt.product(*[range(0,c.size,b) for c,b in zip(bcoords,bsizes)])])
	iblock  = 0
	
	## Loop on blocks
	for bx in itt.product(*[range(0,c.size,b) for c,b in zip(bcoords,bsizes)]):
		
		iblock += 1
		logger.debug( f"| Block {iblock} / {nblocks}" )
		
		## Block indexes
		bidx = { block_dims[i] : slice(bx[i],bx[i]+bsizes[i],1) for i in range(len(bsizes)) }
		
		## Extract array
		logger.debug( "| | => From disk to memory" )
		xargs = [ Z.isel( drop = False , **{ **{ d : slice(None) for d in Z.dims if d not in block_dims } , **{ d : bidx[d] for d in block_dims if d in Z.dims } } ) for Z in args ]
		
		## Find chunks
		to_slice = lambda l: [ slice(l[i],l[i+1]) for i in range(len(l)-1) ]
		reverse  = lambda l: [ tuple([l[i][j] for i in range(len(l))]) for j in range(len(l[0])) ]
		xcoords  = reverse( [ [ idx for idx in itt.product( *[ to_slice( [0] + np.cumsum(c).tolist() ) for c in x.chunk( chunk ).chunks] ) ] for x,chunk in zip(xargs,chunks) ] )
		
		## Apply
		logger.debug( f"| | => Parallel apply with {len(xcoords)} partitions" )
		p_func = P_Func( func , **dask_kwargs.get("kwargs",{}) )
		with multiprocessing.Pool(n_workers) as p:
			p_res = p.map( p_func , [ tuple(x[idx] for x,idx in zip(xargs,xcoords[i])) for i in range(len(xcoords)) ] )
		
		## Loop to store
		logger.debug( "| | => From memory to disk" )
		for res,idxs in zip(p_res,xcoords):
			
			if isinstance(res,xr.DataArray):
				res = [res]
			else:
				res = list(res)
			
			##
			for arr,idx,Z in zip(res,idxs,zout):
				arr = arr.assign_coords( { d : Z.coords.coords[d][idx[i_d]] for i_d,d in enumerate(arr.dims) } ).transpose( *Z.dims )
				Z.loc[*[arr[d] for d in arr.dims]] = arr
		
		## Clean memory
		logger.debug( "| | => Clean memory" )
		del xargs
		del res
		del arr
		gc.collect()
	
	if n_out == 1:
		return zout[0]
	
	return zout
##}}}



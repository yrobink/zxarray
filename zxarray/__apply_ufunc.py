
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

#############
## Imports ##
#############

import typing
import logging
import itertools as itt
import psutil
import gc

import numpy  as np
import xarray as xr
import distributed

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

def apply_ufunc( func , *args , block_dims : list | tuple = [] ,
					total_memory : DMUnit | str = None,
					block_memory = None ,
					output_coords : dict | list | None = None ,
					output_dims : list | tuple | None = None ,
					output_dtypes : list | None = None ,
					output_zfile : list | tuple | None = None ,
					dask_kwargs : dict = {} ,
					zarr_kwargs : dict = {} ,
					n_workers : int = 1,
					threads_per_worker : int = 1,
					memory_per_worker : DMUnit | str = None,
					client : distributed.Client | None = None,
					cluster : typing.Any = None,
					**kwargs
					):
	"""
	zxarray.apply_ufunc
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
	threads_per_worker:
		Number of threads per worker
	memory_per_worker:
		Can replace total_memory, the both can not be simultaneously set
	client:
		The client for parallel computing, use 'processes' if not given and
		cluster not given
	cluster:
		The cluster used, default is distributed.LocalCluster.
	
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
		memory_per_worker = total_memory // n_workers
	
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
	
	## Size of all dimensions
	size_alldims = {}
	for Z in args:
		for d in Z.dims:
			size_alldims[d] = Z[d].size
			if d in block_dims:
				size_alldims[d] = bsizes[block_dims.index(d)]
	logger.debug( f"Size of all dimensions: {size_alldims}" )
	
	## Find dimensions of chunks
	chunks = [ { d : 1 for d in Z.dims if d not in icd } for Z,icd in zip(args,dask_kwargs["input_core_dims"]) ]
	if kwargs.get("chunks") is not None:
		logger.debug( "Use user chunks" )
		user_chunks = kwargs.get("chunks")
		for i in range(len(chunks)):
			for d in chunks[i]:
				if d in user_chunks:
					chunks[i][d] = user_chunks[d]
	else:
		logger.debug( "Infer chunk" )
		user_chunks = kwargs.get("chunks")
		chunked_dims = []
		for c in chunks:
			chunked_dims = chunked_dims + list(c)
		chunked_dims = set(chunked_dims)
		w_ratio = max( 2 , int(np.ceil( np.power( n_workers , 1 / len(chunked_dims) )  )) )
		chunks = [ { d : int(max( 1 , size_alldims[d] // w_ratio )) for d in Z.dims if d not in icd } for Z,icd in zip(args,dask_kwargs["input_core_dims"]) ]
		
		if not len(chunks) == len(args):
			raise ValueError( f"Len of input_core_dims must match the numbers of input array" )
	logger.debug( f"Chunk dimensions: {chunks}" )
	
	## Create the cluster / client
	manage_client = False
	if client is None and cluster is None:
		cluster = distributed.LocalCluster( n_workers  = n_workers , threads_per_worker = threads_per_worker , memory_limit = f"{memory_per_worker.B}B" , processes = False )
		client  = distributed.Client(cluster)
		manage_client = True
	else:
		if client is not None:
			cluster = client.cluster
		elif cluster is not None:
			client = distributed.Client(cluster)
	logger.debug( f"client : {client}" )
	logger.debug( f"cluster: {cluster}" )
	
	##
	logger.debug("dask_kwargs:" )
	for key in dask_kwargs:
		logger.debug( f" * {key} : {dask_kwargs[key]}" )
	
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
		xargs = [ Z.isel( drop = False , **{ **{ d : slice(None) for d in Z.dims if d not in block_dims } , **{ d : bidx[d] for d in block_dims if d in Z.dims } } ).chunk(chunk) for Z,chunk in zip(args,chunks) ]
		
		## Apply
		logger.debug( "| | => Create apply" )
		ores = xr.apply_ufunc( func , *xargs , **dask_kwargs )
		
		## Transform in list, and in dataset
		if isinstance(ores,xr.DataArray):
			ores = [ores]
		else:
			ores = list(ores)
		ores = xr.Dataset( { f"xarr{i}" : res for i,res in enumerate(ores) } )
		
		logger.debug( "| | => Transpose" )
		for i,Z in enumerate(zout):
			ores[f"xarr{i}"] = ores[f"xarr{i}"].transpose(*Z.dims)
		
		logger.debug( "| | => Compute" )
		ores = ores.persist().compute( scheduler = client )
		
		## Clean memory
		logger.debug( "| | => Clean memory of input" )
		del xargs
		gc.collect()
		
		## And save
		logger.debug( "| | => From memory to disk" )
		for i in range(n_out):
			xrcoords = { **{ d : slice(None) for d in zout[i].dims if d not in block_dims } , **{ d : bidx[d] for d in block_dims if d in zout[i].dims } }
			zidx     = [ xrcoords[d] for d in zout[i].dims ]
			zout[i][*zidx] = ores[f"xarr{i}"].values
		
		## Clean memory
		logger.debug( "| | => Clean memory of output" )
		del ores
		gc.collect()
	
	if manage_client:
		cluster.close()
		client.shutdown()
		client.close()
	
	if n_out == 1:
		return zout[0]
	
	return zout

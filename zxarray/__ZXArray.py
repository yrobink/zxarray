
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

from .__zxParams import zxParams
from dataclasses import dataclass

from .__sys import random_zfile

import os
import pathlib
import numpy as np
import xarray as xr
import zarr
import netCDF4
import cftime


#############
## Classes ##
#############

## Coordinates class

class ZXArrayCoords:##{{{
	"""
	zxarray.ZXArrayCoords
	=====================
	Class to manage coordinates.
	"""
	
	def __init__( self , coords : list | dict , dims : list | tuple | None = None ):##{{{
		"""
		zxarray.ZXArrayCoords.__init__
		==============================
		Constructor of the class.
		
		Parameters
		----------
		coords:
			Coordinates, can be a list, and in this case 'dims' must be set, or
			a dict.
		dims:
			List or tuple of coordinates. Can be None if coords is a dict.
		"""
		
		## Check coherence between dims and coords
		if dims is not None:
			if not len(dims) == len(coords):
				raise ValueError( f"Inconsistence size between dims and coords (ndim = {len(dims)} != {len(coords)} = ncoord)" )
		else:
			if isinstance(coords,list):
				raise ValueError( "coords can not be a list if dims is not given" )
			dims = tuple(coords)
		
		## Change coords to be a dict of xarray.DataArray
		if isinstance(coords,list):
			coords = { d : c for d,c in zip(dims,coords) }
			dims   = tuple(coords)
		for d in dims:
			if not isinstance(coords[d],xr.DataArray):
				coords[d] = xr.DataArray( coords[d] , dims = [d] , coords = [coords[d]] )
		
		## Final copy
		self._coords = { d : coords[d].copy() for d in coords }
		self._dims   = tuple(coords)
		self._shape  = tuple(list([len(coords[d]) for d in self._dims]))
	##}}}
	
	def __repr__(self):##{{{
		return self.__str__()
	##}}}
	
	def __str__(self):##{{{
		out = "Coordinates:"
		size = max([len(d) for d in self.dims])
		for d in self.dims:
			out = out + "\n"
			out = out + f"  * " + "{:{fill}{align}{n}}".format( d , fill = " ", align = "<" , n = size ) + f" ({','.join(self.coords[d].coords.dims)}) {self.coords[d].dtype}"
		return out
	##}}}
	
	def coords_to_index( self , drop = True , **kwargs ):##{{{
		"""
		zxarray.ZXArrayCoords.coords_to_index
		=====================================
		Function which transform a mapping of coordinates in indexes.
		
		Parameters
		----------
		drop: bool
			drop or not 1-dim coordinates. Default is True.
		kwargs: dict
			A dict of mapping dimension -> coords to find the index.
		
		Returns
		-------
		index: tuple
			Indexes defines by kwargs
		dims:
			New dimensions
		Coords:
			New coords
		"""
		if not isinstance(drop,bool):
			raise ValueError("drop argument must be a boolean" )
		index  = []
		coords = {}
		dims   = []
		for d in self.dims:
			if d in kwargs:
				idx_val = xr.DataArray( range(self._coords[d].size) , dims = [d] , coords = [self.coords[d].copy()] ).sel( { d : kwargs[d] } ).values
				if idx_val.size == 1:
					if idx_val.ndim > 0:
						idx_val = idx_val[0]
					idx_val = int(idx_val)
					if not drop:
						dims.append(d)
						idx_val   = np.array([idx_val])
						coords[d] = self.coords[d][idx_val].copy()
				else:
					dims.append(d)
					coords[d] = self.coords[d][idx_val].copy()
				index.append(idx_val)
			else:
				dims.append(d)
				index.append(slice(None))
				coords[d] = self.coords[d][index[-1]].copy()
		return tuple(index),dims,coords
	##}}}
	
	def __getitem__( self , key ):##{{{
		"""
		zxarray.ZXArrayCoords.__getitem__
		=================================
		Return the coordinate of the dimension define by 'key'.
		
		Parameters
		----------
		key: str
			Dimension to return
		
		Returns
		-------
		coord: array_like
			Coordinates of 'key'
		"""
		if key not in self._dims:
			raise ValueError( f"dimension '{key}' not found" )
		return self._coords[key].copy()
	##}}}
	
	def rename( self , *args , **kwargs ):##{{{
		
		## Merge args and kwargs
		mname = kwargs
		if len(args) > 0:
			for arg in args:
				if not isinstance(arg,dict):
					raise ValueError("args must be a tuple of dict wich map old name to new name")
				mname = { **mname , **arg }
		
		## Check the key
		for key in mname:
			if key not in self._dims:
				raise ValueError(f"Invalid dimension name: '{key}'")
		
		## Add the current key
		new_dims = []
		for d in self._dims:
			if d not in mname:
				mname[d] = d
				new_dims.append(d)
			else:
				new_dims.append(mname[d])
		new_dims = tuple(new_dims)
		
		## Create the new coordinates
		new_coords = {}
		for ok,nk in zip(self._dims,new_dims):
			new_coords[nk] = xr.DataArray( self._coords[ok].values , dims = [nk] , coords = [self._coords[ok].values] )
		
		## And set
		self._coords = new_coords
		self._dims   = new_dims
		self._shape  = tuple(list([len(new_coords[d]) for d in new_dims]))
	##}}}
	
	def assign_coords( self , *args , **kwargs ):##{{{
		
		## Merge args and kwargs
		mname = kwargs
		if len(args) > 0:
			for arg in args:
				if not isinstance(arg,dict):
					raise ValueError("args must be a tuple of dict wich map dim to new coordinates")
				mname = { **mname , **arg }
		
		## Check the key
		for key in mname:
			if key not in self._dims:
				raise ValueError(f"Invalid dimension name: '{key}'")
		
		## Create the new coordinates
		new_coords = {}
		for key in self._dims:
			if key in mname:
				c = mname[key]
				if not isinstance(c,xr.DataArray):
					c = xr.DataArray( c , dims = [key] , coords = [c] )
				new_coords[key] = c.copy()
			else:
				new_coords[key] = self._coords[key].copy()
		
		## And set
		self._coords = new_coords
	##}}}
	
	## Properties ##{{{
	
	@property
	def ndim(self):
		"""
		Return the number of dimensions.
		"""
		return len(self._dims)
	
	@property
	def dims(self):
		"""
		Return the tuple of dimensions.
		"""
		return self._dims
	
	@property
	def shape(self):
		"""
		Return the shape, a tuple of size of each dimensions.
		"""
		return self._shape
	
	@property
	def coords(self):
		"""
		Return a copy of the coordinates.
		"""
		return { d : self._coords[d].copy() for d in self.dims }
	
	##}}}
	
##}}}


## Attributes class

## dataclass.ZXArrayAttributes ##{{{

if zarr.__version__ < '3.0.0':
	ZarrArrayClass = zarr.core.Array
else:
	ZarrArrayClass = zarr.Array

@dataclass
class ZXArrayAttributes:
	coords : ZXArrayCoords  | None = None
	zdata  : ZarrArrayClass | None = None
##}}}


## Locators

class ZXArrayLocator:##{{{
	"""
	zxarray.ZXArrayLocator
	======================
	Class to manage the '.loc' method of ZXArray. Return the data selected
	by coordinates (not the index) given to the locator. Data returned are
	a xarray.DataArray class, so the data are back to the memory.
	"""
	
	def __init__( self , zxarr ):
		self._zxarr = zxarr
	
	def __getitem__( self , args ):
		if not ( len(args) == self._zxarr.ndim or len(args) == self._zxarr.ndim + 1 ):
			raise ValueError( f"ZXArray.ZXArrayLocator: Bad number arguments")
		drop = bool(args[-1]) if len(args) == self._zxarr.ndim + 1 else True
		sel  = { d : arg for d,arg in zip(self._zxarr.dims,args) }
		return self._zxarr.sel( drop = drop , **sel )
	
	def __setitem__( self , args , data ):
		if not len(args) == self._zxarr.ndim:
			raise ValueError( f"ZXArray.ZXArrayLocator: Bad number arguments")
		sel = { d : arg for d,arg in zip(self._zxarr.dims,args) }
		index,dims,coords = self._zxarr._internal.coords.coords_to_index( drop = False , **sel )
		shape = tuple([coords[d].size for d in dims])
		data = np.asarray( data , dtype = self._zxarr.dtype ).reshape(shape)
		if data.ndim == 0:
			data = np.zeros(shape) + data
		self._zxarr._internal.zdata.set_orthogonal_selection( index , data[:] )
##}}}

class ZXArrayZLocator:##{{{
	"""
	zxarray.ZXArrayZLocator
	=======================
	Class to manage the '.zloc' method of ZXArray. Return the data selected
	by coordinates (not the index) given to the locator. Data returned are
	a zxarray.ZXArray class, so the data are only on the disk.
	"""
	def __init__( self , zxarr ):
		self._zxarr = zxarr
	
	def __getitem__( self , args ):
		if not ( len(args) == self._zxarr.ndim or len(args) == self._zxarr.ndim + 1 ):
			raise ValueError( f"ZXArray.ZXArrayLocator: Bad number arguments")
		drop = bool(args[-1]) if len(args) == self._zxarr.ndim + 1 else True
		sel  = { d : arg for d,arg in zip(self._zxarr.dims,args) }
		return self._zxarr.zsel( drop = drop , **sel )
	
	def __setitem__( self , args , data ):
		if not len(args) == self._zxarr.ndim:
			raise ValueError( f"ZXArray.ZXArrayLocator: Bad number arguments")
		sel = { d : arg for d,arg in zip(self._zxarr.dims,args) }
		index,dims,coords = self._zxarr._internal.coords.coords_to_index(**sel)
		
		data = np.asarray( data , dtype = self._zxarr.dtype )
		if data.ndim == 0:
			data = np.zeros( [coords[d].size for d in dims] ) + data
		self._zxarr._internal.zdata.set_orthogonal_selection( index , data[:] )
##}}}


## Main class

class ZXArray:##{{{
	"""
	zxarray.ZXArray
	===============
	Main class similar to xarray.DataArray, BUT:
	- data are stored in a zarr file, so no memory are used (except the memory
	  used by coordinates)
	- The classic accessors of (.loc, .sel, .isel) xarray.DataArray can be
	  used, and data are returned as a xarray.DataArray (so data are go back to
	  the memory)
	- New accessors which mimics the xarray.DataArray accessors can be used:
	  .zloc, .zsel, .zisel. Data are returned as a zxarray.ZXArray, and no
	  memory are used.
	"""
	
	## I/O ##{{{
	
	def __init__( self , data = None , dims = None , coords = None , zfile = None , dtype = "float32" , zarr_kwargs = { "compressor" : None } ): ##{{{
		"""
		zxarray.ZXArray.__init__
		========================
		Init a zxarray.ZXArray.
		
		Arguments
		---------
		data: scalar | array_like
			Data to initialize
		dims: tuple
			Dimensions names. If not given, 'coords' must be a dict where the
			key will be the dimensions
		coords: list | dict
			Coordinates. Can be a list only if 'dims' is given.
		zfile: str | None
			Path to the zarr file for storage. Default path is given by
			zxarray.zxParams.tmp_folder
		dtype: data_type
			Data type
		zarr_kwargs: dict
			Keywords arguments given to zarr.open
		
		Returns
		-------
		zX: zxarray.ZXArray
		"""
		
		## Internal attributes
		self._internal = ZXArrayAttributes()
		
		## Init the coords class
		self._internal.coords = ZXArrayCoords( coords , dims )
		
		## Open the zarr file
		if zfile is None:
			zfile = random_zfile( dir = zxParams.tmp_folder )
		self._internal.zdata = zarr.open( zfile , mode = "w" , shape = self._internal.coords.shape , dtype = dtype , **zarr_kwargs )
		
		## Set data
		if data is not None:
			data = np.asarray( data , dtype = self.dtype )
			if data.ndim == 0:
				self._internal.zdata[:] = data.dtype.type(data)
			elif data.shape == self.shape:
				self._internal.zdata[:] = data[:]
			else:
				raise ValueError( f"zxarray.ZXArray.__init__: incoherent dimensions between 'data' (ndim = {data.ndim}, shape = {data.shape}) and 'dims' / 'coords' (ndim = {self.ndim}, shape = {self.shape})" )
		
		
	##}}}
	
	## static.from_xarray ## {{{
	@staticmethod
	def from_xarray( xX : xr.DataArray , zfile : str | None = None , zarr_kwargs : dict = {"compressor" : None} ):
		"""
		zxarray.ZXArray@static.from_xarray
		==================================
		Init a zxarray.ZXArray from a xarray.DataArray.
		
		Arguments
		---------
		xX: xarray.DataArray
			The array to transform
		zfile: str | None
			Path to the zarr file for storage. Default path is given by
			zxarray.zxParams.tmp_folder
		zarr_kwargs: dict
			Keywords arguments given to zarr.open
		
		Returns
		-------
		zX: zxarray.ZXArray
		"""
		return ZXArray( data = xX.values , dims = tuple(xX.dims) , coords = { d : xX[d] for d in xX.dims } , zfile = zfile , dtype = xX.dtype , zarr_kwargs = zarr_kwargs )
	##}}}
	
	## static.from_ncfiles ## {{{
	@staticmethod
	def from_ncfiles( *args , dims : tuple[str] | list[str] | None = None , var : str | None = None , concat_var : dict | None = None , transform_coords : dict = {} , infer_cftime : bool = True , zfile : str | None = None , dtype : str = "float32" , zarr_kwargs : dict = {"compressor": None} ):
		"""
		zxarray.ZXArray@static.from_ncfiles
		===================================
		Init a zxarray.ZXArray from a set of netcdf files.
		
		Arguments
		---------
		args: list[str | path]
			List or tuple of path to netcdf files, the joker '*' is allowed.
		dims: tuple
			Dimensions names of the variable(s), coordinates are infered from
			the netcdf files
		var: str | None
			Common variable of all netcdf file to read. Can be None only if
			'concat_var' is given.
		concat_var: dict
			Multiple variables to read and to stack in a new dimensions, added
			after 'dims'. This parameter must be a dict with one key (the new
			dimension name), and the values is the list of variables (used as
			coordinates)
		transform_coords: dict
			Dict of functions used to apply a transformation to coordinates
			when they are read from netcdf files. Keywords must be the
			dimension, and values must be callable.
		infer_cftime: bool
			If a dimension has a calendar and units attributes, use cftime to
			infer a time axis.
		zfile: str | None
			Path to the zarr file for storage. Default path is given by
			zxarray.zxParams.tmp_folder
		dtype: data_type
			Data type
		zarr_kwargs: dict
			Keywords arguments given to zarr.open
		
		Returns
		-------
		zX: zxarray.ZXArray
		"""
		
		## Build the file list
		ifiles = []
		for p in args:
			d = os.path.dirname(p)
			r = os.path.basename(p)
			ifiles = ifiles + sorted(pathlib.Path(d).glob(r))
		
		if len(ifiles) == 0:
			raise ValueError("No files found")
		
		## Build the variables list
		if concat_var is not None:
			dvar  = list(concat_var)[0]
			lvars = np.asarray( concat_var[dvar] ).tolist()
			has_vars = True
		elif var is not None:
			lvars = [var]
			has_vars = False
		else:
			raise ValueError( f"Incoherent input var '{var}' or concat_var '{concat_var}'" )
		
		## Transform
		for d in dims:
			if d not in transform_coords:
				transform_coords[d] = lambda c: c
		
		## Loop on files to find coordinates
		coords = { d : [] for d in dims }
		for ifile in ifiles:
			with netCDF4.Dataset( ifile , 'r' ) as incf:
				
				## Check if variable in the file
				if not len(list(set(list(incf.variables)) & set(lvars))) > 0:
					raise ValueError( f"No input variables found in input file '{ifile}'" )
				
				## Find coordinates
				for d in dims:
					if d not in list(incf.dimensions):
						raise ValueError( f"Dimension '{d}' not in the file '{ifile}'" )
					v = incf.variables[d]
					a = v.ncattrs()
					c = v[:]
					if infer_cftime and "calendar" in a and "units" in a:
						c = cftime.num2date( c , units = v.getncattr("units") , calendar = v.getncattr("calendar") )
					c = transform_coords[d](c)
					coords[d] = coords[d] + c.tolist()
		
		for d in dims:
			c = sorted(list(set(coords[d])))
			coords[d] = xr.DataArray( c , dims = [d] , coords = [c] )
		
		## Init ZXArray
		zdims   = tuple(dims)
		zcoords = dict(coords)
		if has_vars:
			zdims = tuple(zdims) + (dvar,)
			zcoords[dvar] = xr.DataArray( lvars , dims = [dvar] , coords = [lvars] )
		zX = ZXArray( np.nan , dims = zdims , coords = zcoords , zfile = zfile , dtype = dtype , zarr_kwargs = zarr_kwargs )
		
		## And new loop on files to copy data
		for ifile in ifiles:
			with netCDF4.Dataset( ifile , 'r' ) as incf:
				
				## Find variable
				for var in lvars:
					if var in incf.variables:
						break
				
				## Find coordinates
				coords = []
				shape  = []
				for d in dims:
					ncd = incf.variables[d]
					a   = ncd.ncattrs()
					c   = ncd[:]
					if infer_cftime and "calendar" in a and "units" in a:
						c = cftime.num2date( c , units = ncd.getncattr("units") , calendar = ncd.getncattr("calendar") )
					c   = transform_coords[d](c)
					c   = c.tolist()
					coords.append( xr.DataArray( c , dims = [d] , coords = [c] ) )
					shape.append( len(c) )
				
				if has_vars:
					coords.append( xr.DataArray( [var] , dims = [dvar] , coords = [ [var] ] ) )
					shape.append(1)
				
				zX.loc[*coords] = incf.variables[var][:].reshape(shape)
		
		return zX
	##}}}
	
	def to_netcdf( self , ofile , name ):##{{{
		"""
		zxarray.ZXArray.to_netcdf
		=========================
		Store the data in a netcdf file
		
		Arguments
		---------
		ofile: str
			output file
		name: str
			Name of the variables
		"""
		raise NotImplementedError
	##}}}
	
	def copy( self , zfile = None , zarr_kwargs = { "compressor" : None } ): ##{{{
		"""
		zxarray.ZXArray.copy
		====================
		Return a copy of the current zxarray.ZXArray.
		
		Arguments
		---------
		zfile: str | None
			Path to the zarr file for storage. Default path is given by
			zxarray.zxParams.tmp_folder
		zarr_kwargs: dict
			Keywords arguments given to zarr.open
		
		Returns
		-------
		zX: zxarray.ZXArray
		"""
		return self.zsel( zfile = zfile , zarr_kwargs = zarr_kwargs , **{ d : self._internal.coords[d] for d in self.dims } )
	##}}}
	
	def __repr__(self):##{{{
		return self.__str__()
	##}}}
	
	def __str__(self):##{{{
		out = "\n".join( ["<zxarray.ZXArray> (" + ", ".join( f"{d}: {s}" for d,s in zip(self.dims,self.shape) ) + ")" , str(self.zinfo)[:-1] , str(self.coords)] )
		return out
	##}}}
	
	##}}}
	
	## Others methods ##{{{
	
	def rename( self , *args , **kwargs ):
		self._internal.coords.rename(*args,**kwargs)
		return self
	
	def assign_coords( self , *args , **kwargs ):
		self._internal.coords.assign_coords(*args,**kwargs)
		return self
	
	##}}}
	
	## Properties ##{{{
	
	@property
	def dims(self):
		"""
		Return a tuple of dimensions.
		"""
		return self._internal.coords.dims
	
	@property
	def coords(self):
		"""
		Return a dict of coordinates.
		"""
		return self._internal.coords
	
	@property
	def shape(self):
		"""
		Return a tuple of shape.
		"""
		return self._internal.zdata.shape
	
	@property
	def size(self):
		"""
		Return the size.
		"""
		return self._internal.zdata.size
	
	@property
	def ndim(self):
		"""
		Return the number of dimension.
		"""
		return self._internal.zdata.ndim
	
	@property
	def dtype(self):
		"""
		Return the datatype.
		"""
		return self._internal.zdata.dtype
	
	@property
	def zinfo(self):
		"""
		Return the zarr informations.
		"""
		return self._internal.zdata.info
	
	@property
	def path(self):
		"""
		Return the path of the zarr storage.
		"""
		return self._internal.zdata.store.path
	
	##}}}
	
	## Data accessors ## {{{
	
	def sel( self , drop = True , **kwargs ):##{{{
		"""
		zxarray.ZXArray.sel
		===================
		Return a xarray.DataArray defined by the keyword coordinates. Note that
		the data of the return are going back to the memory.
		
		Arguments
		---------
		drop: bool
			drop or not 1-dim coordinates. Default is True.
		kwargs: dict
			mapping { dim : sub-coord } of a coordinates.
		
		Returns
		-------
		xX: xarray.DataArray
		"""
		
		index,dims,coords = self._internal.coords.coords_to_index( drop = drop , **kwargs )
		
		return xr.DataArray( self._internal.zdata.get_orthogonal_selection(index) , dims = dims , coords = coords )
	##}}}
	
	def zsel( self , drop = True , zfile = None , zarr_kwargs = {"compressor" : None} , **kwargs ): ##{{{
		"""
		zxarray.ZXArray.zsel
		====================
		Return a zxarray.ZXArray defined by the keyword coordinates. Note that
		the data stay in a zarr file.
		
		Arguments
		---------
		drop: bool
			drop or not 1-dim coordinates. Default is True.
		zfile: str | None
			Path to the zarr file for storage. Default path is given by
			zxarray.zxParams.tmp_folder
		zarr_kwargs: dict
			Keywords arguments given to zarr.open
		kwargs: dict
			mapping { dim : sub-coord } of a coordinates.
		
		Returns
		-------
		zX: zxarray.ZXArray
		"""
		
		index,dims,coords = self._internal.coords.coords_to_index( drop = drop , **kwargs )
		xzarr = ZXArray( dims = dims , coords = coords , zfile = zfile , dtype = self.dtype , zarr_kwargs = zarr_kwargs )
		xzarr._internal.zdata[:] = self._internal.zdata.get_orthogonal_selection(index)
		
		return xzarr
	##}}}
	
	def isel( self , drop = True , **kwargs ):##{{{
		"""
		zxarray.ZXArray.isel
		====================
		Return a xarray.DataArray defined by the keyword indexes. Note that
		the data of the return are going back to the memory.
		
		Arguments
		---------
		drop: bool
			drop or not 1-dim coordinates. Default is True.
		kwargs: dict
			mapping { dim : sub-index } of index. Integer or slices are allowed.
		
		Returns
		-------
		xX: xarray.DataArray
		"""
		return self.sel( drop = drop , **{ d : self._internal.coords[d][kwargs[d]] for d in kwargs } )
	##}}}
	
	def zisel( self , drop = True , zfile = None , zarr_kwargs = {"compressor" : None} , **kwargs ): ##{{{
		"""
		zxarray.ZXArray.zisel
		=====================
		Return a zxarray.ZXArray defined by the index coordinates. Note that
		the data stay in a zarr file.
		
		Arguments
		---------
		drop: bool
			drop or not 1-dim coordinates. Default is True.
		zfile: str | None
			Path to the zarr file for storage. Default path is given by
			zxarray.zxParams.tmp_folder
		zarr_kwargs: dict
			Keywords arguments given to zarr.open
		kwargs: dict
			mapping { dim : sub-index } of index. Integer or slices are allowed.
		
		Returns
		-------
		zX: zxarray.ZXArray
		"""
		return self.zsel( drop = drop , zfile = zfile , zarr_kwargs = zarr_kwargs , **{ d : self._internal.coords[d][kwargs[d]] for d in kwargs } )
	##}}}
	
	## property.loc ##{{{
	@property
	def loc(self):
		"""
		zxarray.ZXArray.loc
		===================
		Return a locator smimilar to xarray.DataArray.loc
		
		Returns
		-------
		loc: zxarray.__ZXArray.ZXArrayLocator
		"""
		return ZXArrayLocator(self)
	##}}}
	
	## property.zloc ##{{{
	@property
	def zloc(self):
		"""
		zxarray.ZXArray.zloc
		====================
		Return a locator smimilar to xarray.DataArray.loc
		
		Returns
		-------
		loc: zxarray.__ZXArray.ZXArrayZLocator
		"""
		return ZXArrayZLocator(self)
	##}}}
	
	## property.datarray ##{{{
	@property
	def dataarray( self ):
		"""
		zxarray.ZXArray.dataarray
		=========================
		Return all the data as a datarray. All data are going back to the memory
		
		Returns
		-------
		xarray.DataArray
		"""
		idx = [slice(None) for _ in range(self.ndim)] + [False]
		return self.loc[idx]
	##}}}
	
	def __getitem__( self , *args ): ##{{{
		"""
		zxarray.ZXArray.__getitem__
		===========================
		If args is a single string, return the coordinates of the dimension.
		Else 'args' must be a list of index, and the method '.isel' is called.
		
		Arguments
		---------
		args: str | tuple[index]
			a single string, or index of corresponding to coordinates. If index
			is given, the values of args can be 'x' or 'z' to return a
			xarray.DataArray or a zxarray.ZXArray, or True or False to drop or
			not the 1-d coordinates
		
		Returns
		-------
		xarray.DataArray
		"""
		
		## Ask a coordinate
		if isinstance(args[0],str):
			if len(args) > 1:
				raise TypeError( "Invalid indexer, must be a string (for coordinates) or array indexer" ) 
			return self.coords[args[0]]
		
		## Check if option for x or z array, or drop:
		index = args[0]
		mode  = "z"
		drop  = False
		if len(index) > self.ndim:
			for md in index[self.ndim:]:
				if md in ["z","x"]:
					mode = md
				if isinstance(md,bool):
					drop = md
			index = index[:self.ndim]
		
		## Check size
		if not len(index) == self.ndim:
			raise ValueError( f"Invalid number of dimension {len(index)} != {self.ndim}" )
		
		## output coords
		ocoords = {}
		for d,i in zip(self.dims,index):
			if isinstance(i,int):
				ocoords[d] = np.array( [i] , dtype = int )
			else:
				ocoords[d] = i
		
		## In fact just a call to the zisel / isel method
		if mode == "z":
			return self.zisel( drop = drop , **ocoords )
		
		return self.isel( drop = drop , **ocoords )
	##}}}
	
	def __setitem__( self , args , data ): ##{{{
		"""
		zxarray.ZXArray.__setitem__
		===========================
		'args' must be a list of index, and 'data' is used to update the dataset
		
		Arguments
		---------
		args: str | tuple[index]
			a single string, or index of corresponding to coordinates.
		data: array_like
		
		Returns
		-------
		"""
		data = np.asarray( data , dtype = self.dtype )
		if data.ndim == 0:
			self._internal.zdata.set_orthogonal_selection( args , data )
		else:
			self._internal.zdata.set_orthogonal_selection( args , data[:] )
	##}}}
	
	## }}}
	
	## Stats ##{{{
	
	def _ufunc( self , func , dims = None , **kwargs ):##{{{
		raise NotImplementedError
	##}}}
	
	def sum( self , dims = None ):##{{{
		raise NotImplementedError
	##}}}
	
	def prod( self , dims = None ):##{{{
		raise NotImplementedError
	##}}}
	
	def mean( self , dims = None ):##{{{
		raise NotImplementedError
	##}}}
	
	def median( self , dims = None , **kwargs ):##{{{
		return self.quantile( 0.5 , dims = dims , method = method )
	##}}}
	
	def std( self , dims = None ):##{{{
		raise NotImplementedError
	##}}}
	
	def min( self , dims = None ):##{{{
		raise NotImplementedError
	##}}}
	
	def max( self , dims = None ):##{{{
		raise NotImplementedError
	##}}}
	
	def quantile( self , p , dims = None , **kwargs ):##{{{
		raise NotImplementedError
	##}}}
	
	def abs( self ):##{{{
		raise NotImplementedError
	##}}}
	
	##}}}
	
##}}}


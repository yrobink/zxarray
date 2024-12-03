#!/usr/bin/env python3 -m unittest

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

import os
import unittest

import numpy as np
import xarray as xr
import zxarray as zr


###############
## Functions ##
###############

def build_data_example( ndata ): ##{{{
	
	## Define coordinates
	sample = [ "S{:{fill}{align}{n}}".format( s , fill = "0" , align = ">" , n = 3 ) for s in range(100) ]
	time   = xr.date_range("2000-01-01","2003-12-31").values
	time   = xr.DataArray( time , dims = ["time"] , coords = [time] )
	y      = np.linspace( -90 , 90 , 15 )
	x      = np.linspace( -180 , 180 , 15 + 1 )[:-1]
	y      = xr.DataArray( y , dims = ["y"] , coords = [y] )
	x      = xr.DataArray( x , dims = ["x"] , coords = [x] )
	dims   = ["sample","time","y","x"]
	coords = [sample,time,y,x]
	shape  = [len(c) for c in coords]
	
	## Define numpy data
	data = [ np.random.normal( size = shape ).astype("float32") for _ in range(ndata) ]
	
	## Define multiple xarray data
	xdata = [ xr.DataArray( X , dims = dims , coords = coords ) for X in data ]
	
	## Define a zxarray
	zdata = [ zr.ZXArray( X.values , dims = X.dims , coords = X.coords ) for X in xdata ]
	
	
	return data,xdata,zdata
##}}}

def func_1return( *args ):##{{{
	
	shp = args[0].shape
	
	res = np.zeros( (shp[0],shp[1],len(args),4) )
	for iarg,arg in enumerate(args):
		res[:,:,iarg,0] = np.mean( arg , axis = (2,3) )
		res[:,:,iarg,1] = np.std(  arg , axis = (2,3) )
		res[:,:,iarg,2] = np.min(  arg , axis = (2,3) )
		res[:,:,iarg,3] = np.max(  arg , axis = (2,3) )
	
	return res
##}}}

def func_Nreturn( *args ):##{{{
	
	shp = args[0].shape
	
	res0 = np.zeros( (shp[0],shp[1],len(args),4) )
	res1 = np.zeros( (shp[0],shp[1],len(args),4) )
	res2 = np.zeros( (shp[0],shp[1],len(args),4) )
	for iarg,arg in enumerate(args):
		res0[:,:,iarg,0] = np.mean( arg , axis = (2,3) )
		res0[:,:,iarg,1] = np.std(  arg , axis = (2,3) )
		res0[:,:,iarg,2] = np.min(  arg , axis = (2,3) )
		res0[:,:,iarg,3] = np.max(  arg , axis = (2,3) )
		
		res1[:,:,iarg,0] = np.quantile( arg , 0.1 , axis = (2,3) )
		res1[:,:,iarg,1] = np.quantile( arg , 0.3 , axis = (2,3) )
		res1[:,:,iarg,2] = np.quantile( arg , 0.7 , axis = (2,3) )
		res1[:,:,iarg,3] = np.quantile( arg , 0.9 , axis = (2,3) )
		
		res2[:,:,iarg,0] = np.exp( np.mean( arg , axis = (2,3) ) )
		res2[:,:,iarg,1] = np.cos( np.mean( arg , axis = (2,3) ) )
		res2[:,:,iarg,2] = np.sin( np.mean( arg , axis = (2,3) ) )
		res2[:,:,iarg,3] = np.abs( np.mean( arg , axis = (2,3) ) )
	
	return res0,res1,res2
##}}}

###########
## Tests ##
###########

class Test__apply_ufunc(unittest.TestCase):
	
	def test_memory_error(self):##{{{
		
		## Data
		ndata = 3
		data,xdata,zdata = build_data_example(ndata)
		
		## Set parameters
		input_core_dims    = [["sample","time"] for _ in range(len(xdata))]
		output_core_dims   = [["array","stats"]]
		output_dtypes      = [float for _ in range(ndata)]
		vectorize          = False
		dask               = "parallelized"
		output_coords      = { "stats" : ["m","s","n","x"] , "array" : range(ndata) }
		output_sizes       = { "stats" : 4 , "array" : ndata }
		dask_gufunc_kwargs = { "output_sizes" : output_sizes }
		transpose          = ("stats","array","y","x")
		dask_kwargs        = { "input_core_dims" : input_core_dims , "output_core_dims" : output_core_dims , "output_dtypes" : output_dtypes , "vectorize" : vectorize , "dask" : dask , "dask_gufunc_kwargs" : dask_gufunc_kwargs }
		
		output_dims   = [["stats","array","y","x"]]
		output_coords = [ { **output_coords , **{ "y" : xdata[0].y , "x" : xdata[0].x } } ]
		bdims         = ("y","x")
		args   = [func_1return] + zdata
		kwargs = { "block_dims" : bdims , "total_memory" : "1Mo" , "output_coords" : output_coords , "output_dims" : output_dims , "dask_kwargs" : dask_kwargs }
		self.assertRaises( MemoryError ,  zr.apply_ufunc , *args , **kwargs ) 
		
	##}}}
	
	def test_comparison_xarray_1return(self):##{{{
		
		## Data
		ndata = 3
		data,xdata,zdata = build_data_example(ndata)
		
		## Set parameters
		input_core_dims    = [["sample","time"] for _ in range(len(xdata))]
		output_core_dims   = [["array","stats"]]
		output_dtypes      = [float]
		vectorize          = False
		dask               = "parallelized"
		output_coords      = { "stats" : ["m","s","n","x"] , "array" : range(ndata) }
		output_sizes       = { "stats" : 4 , "array" : ndata }
		dask_gufunc_kwargs = { "output_sizes" : output_sizes }
		transpose          = ("stats","array","y","x")
		dask_kwargs        = { "input_core_dims" : input_core_dims , "output_core_dims" : output_core_dims , "output_dtypes" : output_dtypes , "vectorize" : vectorize , "dask" : dask , "dask_gufunc_kwargs" : dask_gufunc_kwargs }
		
		xS = xr.apply_ufunc( func_1return , *xdata ,
		                    input_core_dims    = input_core_dims,
		                    output_core_dims   = output_core_dims,
		                    output_dtypes      = output_dtypes,
		                    vectorize          = vectorize,
		                    dask               = dask,
		                    dask_gufunc_kwargs = dask_gufunc_kwargs
		                    ).assign_coords(output_coords).transpose(*transpose)
		
		output_dims   = [["stats","array","y","x"]]
		output_coords = [ { **output_coords , **{ "y" : xdata[0].y , "x" : xdata[0].x } } ]
		bdims         = ("y","x")
		zS            = zr.apply_ufunc( func_1return , *zdata , block_dims = bdims , output_coords = output_coords , output_dims = output_dims , dask_kwargs = dask_kwargs , n_workers = 4 ) 
		
		self.assertTrue( np.abs( zS.dataarray - xS ).max() < 1e-3 )
	##}}}
	
	def test_comparison_xarray_Nreturn(self):##{{{
		
		## Data
		ndata = 3
		data,xdata,zdata = build_data_example(ndata)
		
		## Set parameters
		input_core_dims    = [["sample","time"] for _ in range(len(xdata))]
		output_core_dims   = [["array","stats"],["array","quantile"],["array","ufunc"]]
		output_dtypes      = [float for _ in range(3)]
		vectorize          = False
		dask               = "parallelized"
		output_coords      = [{ "stats" : ["m","s","n","x"] , "array" : range(ndata) } , { "quantile" : ["Q10","Q30","Q70","Q90"] , "array" : range(ndata) }, { "ufunc" : ["exp","cos","sin","abs"] , "array" : range(ndata) }]
		output_sizes       = { "stats" : 4 , "array" : ndata , "quantile" : 4 , "ufunc" : 4 }
		dask_gufunc_kwargs = { "output_sizes" : output_sizes }
		transpose          = [["stats","array","y","x"],["quantile","array","y","x"],["ufunc","array","y","x"]]
		dask_kwargs        = { "input_core_dims" : input_core_dims , "output_core_dims" : output_core_dims , "output_dtypes" : output_dtypes , "vectorize" : vectorize , "dask" : dask , "dask_gufunc_kwargs" : dask_gufunc_kwargs }
		
		##
		lxS = xr.apply_ufunc( func_Nreturn , *xdata ,
		                    input_core_dims    = input_core_dims,
		                    output_core_dims   = output_core_dims,
		                    output_dtypes      = output_dtypes,
		                    vectorize          = vectorize,
		                    dask               = dask,
		                    dask_gufunc_kwargs = dask_gufunc_kwargs
		                    )
		lxS = list(lxS)
		for i in range(len(lxS)):
			lxS[i] = lxS[i].compute().assign_coords(output_coords[i]).transpose(*transpose[i])
		
		output_dims   = [["stats","array","y","x"],["quantile","array","y","x"],["ufunc","array","y","x"]]
		output_coords = [ { **output_coords[i] , **{ "y" : xdata[0].y , "x" : xdata[0].x } }  for i in range(3)]
		bdims         = ("y","x")
		lzS           = zr.apply_ufunc( func_Nreturn , *zdata , block_dims = bdims , total_memory = zr.DMUnit("1Go") , output_coords = output_coords , output_dims = output_dims , dask_kwargs = dask_kwargs , n_workers = 4 ) 
		
		lerr = []
		for xS,zS in zip(lxS,lzS):
			lerr.append( np.abs( zS.dataarray - xS ).max() )
		
		self.assertTrue( np.mean(lerr) < 1e-3 )
	##}}}
	

##########
## main ##
##########

if __name__ == "__main__":
	unittest.main()

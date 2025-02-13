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
import string
import random
import unittest

import numpy   as np
import xarray  as xr
import zxarray as zr

from zxarray.__ZXArray import ZXArrayCoords


###############
## Functions ##
###############

def build_coords():##{{{
	sample = [ "S{:{fill}{align}{n}}".format( s , fill = "0" , align = ">" , n = 2 ) for s in range(10) ]
	time   = xr.date_range("2000-01-01","2003-12-31").values
	time   = xr.DataArray( time , dims = ["time"] , coords = [time] )
	y      = np.linspace( -90 , 90 , 15 )
	x      = np.linspace( -180 , 180 , 15 + 1 )[:-1]
	y      = xr.DataArray( y , dims = ["y"] , coords = [y] )
	x      = xr.DataArray( x , dims = ["x"] , coords = [x] )
	dims   = ["sample","time","y","x"]
	coords = [sample,time,y,x]
	shape  = [len(c) for c in coords]
	
	lcoords = coords
	dcoords = { d : c for d,c in zip(dims,coords) }
	
	return dims,lcoords,dcoords,shape
##}}}


###########
## Tests ##
###########

class Test__ZXArrayCoords(unittest.TestCase):##{{{
	
	def test__init__( self ):##{{{
		dims,lcoords,dcoords,_ = build_coords()
		
		self.assertRaises( ValueError , ZXArrayCoords , coords = lcoords , dims = dims[:2] )
		self.assertRaises( ValueError , ZXArrayCoords , coords = lcoords , dims = None )
	##}}}
	
	def test__getitem__(self):##{{{
		dims,lcoords,dcoords,_ = build_coords()
		c = ZXArrayCoords( coords = dcoords )
		
		self.assertRaises( ValueError , c.__getitem__ , "K" )
	##}}}
	
##}}}

class Test__ZXArray(unittest.TestCase):##{{{
	
	def test__init__(self):##{{{
		
		dims,_,coords,shape = build_coords()
		
		## Initialization with nan value
		zX = zr.ZXArray( np.nan , dims = dims , coords = coords )
		self.assertTrue( np.isnan(zX.dataarray).all() )
		
		## Initialization with a 0 value
		zX = zr.ZXArray( 0. , dims = dims , coords = coords )
		self.assertTrue( (np.abs(zX.dataarray) < 1e-6 ).all() )
		
		## Incoherent data
		self.assertRaises( ValueError , zr.ZXArray , data = 0. , dims = dims[:2] , coords = coords )
	##}}}
	
	def test__from_xarray(self):##{{{
		
		dims,_,coords,shape = build_coords()
		
		xX = xr.DataArray( np.random.normal( size = shape ) , dims = dims , coords = coords )
		zX = zr.ZXArray.from_xarray(xX)
		self.assertTrue( ( np.abs( xX - zX.dataarray ) < 1e-6 ).all() )
		
	##}}}
	
	def test__rename(self):##{{{
		dims,_,coords,shape = build_coords()
		
		xX = xr.DataArray( np.random.normal( size = shape ) , dims = dims , coords = coords )
		zX = zr.ZXArray.from_xarray(xX)
		zX = zX.rename( { 'y' : "lat" } , x = 'lon' )
		self.assertTrue( "lat" in zX.dims )
		self.assertTrue( "lon" in zX.dims )
		self.assertFalse( "x" in zX.dims )
		self.assertFalse( "y" in zX.dims )
	##}}}
	
	def test__assign_coords(self):##{{{
		dims,_,coords,shape = build_coords()
		
		xX = xr.DataArray( np.random.normal( size = shape ) , dims = dims , coords = coords )
		zX = zr.ZXArray.from_xarray(xX)
		zX = zX.assign_coords( { "y" : np.arange(coords['y'].size) } , x = np.arange(coords['x'].size) )
		for d in ["y","x"]:
			self.assertAlmostEqual( np.sqrt( np.sum(np.abs(zX[d] - np.arange(coords[d].size))**2) ) , 0. )
	##}}}
	
	def test__copy(self):##{{{
		
		dims,_,coords,shape = build_coords()
		
		xX = xr.DataArray( np.random.normal( size = shape ) , dims = dims , coords = coords )
		zX = zr.ZXArray.from_xarray(xX)
		zY = zX.copy()
		self.assertTrue( ( np.abs( zY.dataarray - zX.dataarray ) < 1e-6 ).all() )
		
	##}}}
	
	def test__sel(self):##{{{
		dims,_,coords,shape = build_coords()
		
		xX = xr.DataArray( np.random.normal( size = shape ) , dims = dims , coords = coords )
		zX = zr.ZXArray.from_xarray(xX)
		
		scoords = { "time" : "2002" }
		
		sxX = xX.sel( **scoords )
		szX = zX.sel( **scoords )
		
		self.assertTrue( ( np.abs( sxX - szX ) < 1e-6 ).all() )
	##}}}
	
	def test__zsel(self):##{{{
		dims,_,coords,shape = build_coords()
		
		xX = xr.DataArray( np.random.normal( size = shape ) , dims = dims , coords = coords )
		zX = zr.ZXArray.from_xarray(xX)
		
		scoords = { "time" : "2002" }
		
		sxX = xX.sel( **scoords )
		szX = zX.zsel( **scoords )
		
		self.assertTrue( ( np.abs( sxX - szX.dataarray ) < 1e-6 ).all() )
	##}}}
	
	def test__isel(self):##{{{
		dims,_,coords,shape = build_coords()
		
		xX = xr.DataArray( np.random.normal( size = shape ) , dims = dims , coords = coords )
		zX = zr.ZXArray.from_xarray(xX)
		
		sicoords = { "time" : np.arange(0,100,2,dtype = int) }
		
		sxX = xX.isel( **sicoords )
		szX = zX.isel( **sicoords )
		
		self.assertTrue( ( np.abs( sxX - szX ) < 1e-6 ).all() )
	##}}}
	
	def test__zisel(self):##{{{
		dims,_,coords,shape = build_coords()
		
		xX = xr.DataArray( np.random.normal( size = shape ) , dims = dims , coords = coords )
		zX = zr.ZXArray.from_xarray(xX)
		
		sicoords = { "time" : np.arange(0,100,2,dtype = int) }
		
		sxX = xX.isel( **sicoords )
		szX = zX.zisel( **sicoords )
		
		self.assertTrue( ( np.abs( sxX - szX.dataarray ) < 1e-6 ).all() )
	##}}}
	
	def test__loc(self):##{{{
		dims,_,coords,shape = build_coords()
		
		xX = xr.DataArray( np.random.normal( size = shape ) , dims = dims , coords = coords )
		zX = zr.ZXArray.from_xarray(xX)
		
		sxX = xX.loc[:,"2002",:,:]
		szX = zX.loc[:,"2002",:,:]
		
		self.assertTrue( ( np.abs( sxX - szX ) < 1e-6 ).all() )
		
		sxX = xX.loc[:,"2002-01-01",:,:]
		szX = zX.loc[:,"2002-01-01",:,:]
		
		self.assertTrue( ( np.abs( sxX - szX ) < 1e-6 ).all() )
	##}}}
	
	def test__zloc(self):##{{{
		dims,_,coords,shape = build_coords()
		
		xX = xr.DataArray( np.random.normal( size = shape ) , dims = dims , coords = coords )
		zX = zr.ZXArray.from_xarray(xX)
		
		sxX =  xX.loc[:,"2002",:,:]
		szX = zX.zloc[:,"2002",:,:]
		
		self.assertTrue( ( np.abs( sxX - szX.dataarray ) < 1e-6 ).all() )
		
		sxX =  xX.loc[:,"2002-01-01",:,:]
		szX = zX.zloc[:,"2002-01-01",:,:]
		
		self.assertTrue( ( np.abs( sxX - szX.dataarray ) < 1e-6 ).all() )
	##}}}
	
	def test__getitem__(self):##{{{
		dims,_,coords,shape = build_coords()
		
		xX = xr.DataArray( np.random.normal( size = shape ) , dims = dims , coords = coords )
		zX = zr.ZXArray.from_xarray(xX)
		
		for d in zX.dims:
			self.assertTrue( (zX["time"] == zX.coords["time"]).all() )
		
		self.assertTrue( ( np.abs( zX[0,:,:,:,"x"] -xX[0,:,:,:] ) < 1e-6 ).all() )
		self.assertTrue( ( np.abs( zX[0,:,:,:,"z"].dataarray -xX[0,:,:,:] ) < 1e-6 ).all() )
		
	##}}}
	
	def test__setitem__(self):##{{{
		dims,_,coords,shape = build_coords()
		
		xX = xr.DataArray( np.random.normal( size = shape ) , dims = dims , coords = coords )
		zX = zr.ZXArray.from_xarray(xX)
		xY = xr.DataArray( np.random.uniform( size = shape ) , dims = dims , coords = coords )
		
		zX[0,:,:,:] = xY[0,:,:,:]
		
		self.assertTrue( ( np.abs( zX[0,:,:,:,"x"] - xY[0,:,:,:] ) < 1e-6 ).all() )
		
	##}}}
	
##}}}


##########
## main ##
##########

if __name__ == "__main__":
	unittest.main()

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
	
	lcoords = coords
	dcoords = { d : c for d,c in zip(dims,coords) }
	
	return dims,lcoords,dcoords
##}}}


###########
## Tests ##
###########

class Test__ZXArrayCoords(unittest.TestCase):##{{{
	
	def test__init__( self ):##{{{
		dims,lcoords,dcoords = build_coords()
		
		self.assertRaises( ValueError , ZXArrayCoords , coords = lcoords , dims = dims[:2] )
		self.assertRaises( ValueError , ZXArrayCoords , coords = lcoords , dims = None )
	##}}}
	
	def test__getitem__(self):##{{{
		dims,lcoords,dcoords = build_coords()
		c = ZXArrayCoords( coords = dcoords )
		
		self.assertRaises( ValueError , c.__getitem__ , "K" )
	##}}}
	
##}}}

class Test__ZXArray(unittest.TestCase):##{{{
	pass
##}}}


##########
## main ##
##########

if __name__ == "__main__":
	unittest.main()

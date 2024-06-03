
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

###############
## Functions ##
###############

def random_zfile( prefix = "ZXARRAY_RANDOM_NAME_" , dir = "." ):##{{{
	
	"""
	zxarray.random_zfile
	====================
	Return a random zarr file name prefixed by 'prefix' in the directory 'dir'.
	
	Arguments
	---------
	prefix: str
		The prefix of the file name. Default is 'ZXARRAY_RANDOM_NAME_'.
	dir: str | path
		Directory of the file.
	
	Return
	------
	zfile: str
		A path to a non-existing file.
	
	"""
	
	if not os.path.exists(dir):
		raise NotADirectoryError( f"The directory {dir} does not exists" )
	
	zfile = os.path.join( dir , prefix + ''.join(random.choice(string.ascii_uppercase + string.digits) for _ in range(30)) + ".zarr" )
	while os.path.exists(zfile):
		zfile = os.path.join( dir , prefix + ''.join(random.choice(string.ascii_uppercase + string.digits) for _ in range(30)) + ".zarr" )
	
	return zfile
##}}}


#############
## Classes ##
#############



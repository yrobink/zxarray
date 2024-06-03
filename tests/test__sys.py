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

import zxarray as zr


###########
## Tests ##
###########

class Test__misc(unittest.TestCase):
	
	def test_random_zfile_invalidPath(self): ##{{{
		
		## Build an invalid path
		dir = "."
		while os.path.exists(dir):
			dir = ''.join(random.choice(string.ascii_uppercase + string.digits) for _ in range(30))
		
		## And test if an exception is raised
		self.assertRaises( NotADirectoryError , zr.random_zfile , dir = dir )
	##}}}
	
	def test_random_zfile_validZfile(self): ##{{{
		zfile = zr.random_zfile()
		self.assertFalse( os.path.exists(zfile) )
		self.assertTrue( os.path.exists(os.path.dirname(zfile)) )
	##}}}
	

##########
## main ##
##########

if __name__ == "__main__":
	unittest.main()


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
import tempfile


#############
## Classes ##
#############

class ZxParams:##{{{
	
	## I/O ##{{{
	
	def __init__(self):
		self._tmp_folder_gen = tempfile.TemporaryDirectory()
		self._tmp_folder     = self._tmp_folder_gen.name
	
	def __repr__(self):
		raise NotImplementedError
	
	def __str__(self):
		raise NotImplementedError
	
	##}}}
	
	## Properties ##{{{
	
	@property
	def tmp_folder(self):
		return self._tmp_folder
	
	@tmp_folder.setter
	def tmp_folder( self , path ):
		self._tmp_folder_gen = tempfile.TemporaryDirectory( dir = os.path.abspath(path) )
		self._tmp_folder     = self._tmp_folder_gen.name
	##}}}
	
##}}}

zxParams = ZxParams()


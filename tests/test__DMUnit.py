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
import itertools as itt
import unittest

import numpy as np
import zxarray as zr


###########
## Tests ##
###########

class Test__DMUnit(unittest.TestCase):
	
	def test_init(self):##{{{
		self.assertRaises( ValueError , zr.DMUnit )
		self.assertRaises( ValueError , zr.DMUnit , s = "3o" , n = 3 )
	##}}}
	
	def test_init_str(self):##{{{
		self.assertRaises( ValueError , zr.DMUnit , n = 3 , unit = "W" )
		self.assertRaises( ValueError , zr.DMUnit , s = "AAo" )
		self.assertRaises( ValueError , zr.DMUnit , s = "0.3b" )
		self.assertEqual( zr.DMUnit("0.001Go") , zr.DMUnit("1Mo") )
	##}}}
	
	def test_init_value(self):##{{{
		self.assertRaises( ValueError , zr.DMUnit , n = 0.1 )
	##}}}
	
	def test_zero(self):##{{{
		self.assertTrue( zr.DMUnit.zero().b == 0 )
	##}}}
	
	def test_eq(self):##{{{
		
		s = [zr.DMUnit.zero(),zr.DMUnit("0o") ]
		for sa,sb in itt.combinations_with_replacement(s,2):
			self.assertEqual(sa,sb)
		
		s = [zr.DMUnit( "1Go" ),
		     zr.DMUnit( "1000Mo" ),
		     zr.DMUnit( "1000000ko" ),
		     zr.DMUnit( "1000000000o" ),
		     zr.DMUnit( n = 1 , unit = "Go" ),
		     zr.DMUnit( n = 1000 , unit = "Mo" ),
		     zr.DMUnit( n = 1000000 , unit = "ko" ),
		     zr.DMUnit( n = 1000000000 , unit = "o" ),
		]
		for sa,sb in itt.combinations_with_replacement(s,2):
			self.assertEqual(sa,sb)
	##}}}
	
	def test_lt(self):##{{{
		s0,s1 = sorted(np.random.choice( 1000000 , 2 ).tolist())
		self.assertLess( zr.DMUnit(f"{s0}o") , zr.DMUnit(f"{s1}o") )
	##}}}
	
	def test_gt(self):##{{{
		s0,s1 = sorted(np.random.choice( 1000000 , 2 ).tolist())
		self.assertGreater( zr.DMUnit(f"{s1}o") , zr.DMUnit(f"{s0}o") )
	##}}}
	
	def test_le(self):##{{{
		s0,s1 = sorted(np.random.choice( 1000000 , 2 ).tolist())
		self.assertLessEqual( zr.DMUnit(f"{s0}o") , zr.DMUnit(f"{s1}o") )
	##}}}
	
	def test_ge(self):##{{{
		s0,s1 = sorted(np.random.choice( 1000000 , 2 ).tolist())
		self.assertGreaterEqual( zr.DMUnit(f"{s1}o") , zr.DMUnit(f"{s0}o") )
	##}}}
	
	def test_add(self):##{{{
		self.assertEqual( zr.DMUnit("1ko") + "100o" , "1.1ko" )
	##}}}
	
	def test_radd(self):##{{{
		self.assertEqual( "100o" + zr.DMUnit("1ko") , "1.1ko" )
	##}}}
	
	def test_mul(self):##{{{
		self.assertEqual( zr.DMUnit("1ko") * 4 , "4ko" )
		self.assertRaises( ValueError , zr.DMUnit.zero().__mul__ , 1.5 )
	##}}}
	
	def test_rmul(self):##{{{
		self.assertEqual( 4 * zr.DMUnit("1ko") , "4ko" )
		self.assertRaises( ValueError , zr.DMUnit.zero().__rmul__ , 1.5 )
	##}}}
	
	def test_floordiv(self):##{{{
		self.assertEqual( zr.DMUnit("3kb") // 17 , zr.DMUnit( n = 3000 // 17 , unit = "b" ) )
		self.assertRaises( ValueError , zr.DMUnit.zero().__floordiv__ , 1.5 )
	##}}}
	
	def test_mod(self):##{{{
		self.assertEqual( zr.DMUnit("3kb") % 17 , zr.DMUnit( n = 3000 % 17 , unit = "b" ) )
		self.assertRaises( ValueError , zr.DMUnit.zero().__mod__ , 1.5 )
	##}}}
	

##########
## main ##
##########

if __name__ == "__main__":
	unittest.main()

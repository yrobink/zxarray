
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

import numpy as np


#############
## Classes ##
#############

class DMUnit:
	"""
	zxarray.DMUnit
	==============
	
	Data and Memory Unit (DMUnit): class used to convert between different
	units of computer data size.
	
	Conversions are carried out according to the following rules:
	
	1o == 1octet == 1Byte == 1B == 8 bits == 8b
	
	Two types of prefixes (units) can be used, those in base 1000:
	
	- 1kB = 1000^1B,
	- 1MB = 1000^2B,
	- 1GB = 1000^3B,
	- 1TB = 1000^4B;
	- 1PB = 1000^5B;
	- 1HB = 1000^6B;
	
	and those in base 1024:
	
	- 1kiB = 1024^1B,
	- 1MiB = 1024^2B,
	- 1GiB = 1024^3B,
	- 1TiB = 1024^4B.
	- 1PiB = 1024^5B,
	- 1HiB = 1024^6B.
	
	Conversions can be made using the corresponding properties. Comparison and
	arithmetic operators are overloaded:
	
	>>> s0 = zxarray.DMUnit('3ko')
	>>> s1 = zxarray.DMUnit('3000o')
	>>> s2 = zxarrau.DMUnit('3kio')
	>>> s0 == s1 ## True
	>>> s0 <= s1 ## True
	>>> s0 < s2  ## True
	>>> s0 == "1Go" ## False
	>>> s0 + s1 == "6ko" ## True
	>>> 2 * s0  == "6ko" ## True
	
	Notes that two constants are also defined:
	
	- 'zxarray.DMUnit.bits_per_byte  == 8
	- 'zxarray.DMUnit.bits_per_octet == 8
	
	As example, theses constants can be used to find the memory size of a numpy
	array:
	
	>>> x = np.zeros( (2,3,4) )
	>>> size_x = zxarray.DMUnit( n = x.size * np.finfo(x.dtype).bits // zxarray.DMUnit.bits_per_octet , unit = "o" )
	
	which is equivalent to :
	
	>>> size_x = zxarray.DMUnit.sizeof_array(x)
	
	"""
	
	## Constant ##{{{
	
	bits_per_byte  : int = 8
	bits_per_octet : int = 8
	
	##}}}
	
	## attributes ##{{{
	
	_unit   : str = 'b'
	_prefix : str = ''
	_value  : int | float = 0
	_bits   : int = 0
	_ibase  : str = ''
	_base   : int = 1000
	
	##}}}
	
	def __init__( self , s : str | None = None , n : int | None = None , unit : str = "b" ):##{{{
		
		"""
		zxarray.DMUnit.__init__
		=======================
		
		Constructor of the zxarray.DMUnit class
		
		Parameters
		----------
		s: str | None = None
			A string representing a data size, such as '8kb', '9Mb', etc.
			Equivalent to passing the value f"{n}{unit}". If 's' is not None,
			'n' must be None.
		n: int | None = None
			An integer representing a data size, with unit 'unit'. If 'n' is
			not None, 's' must be None.
		unit: str = 'b'
			The unit, see the class description for the available units.
		
		Returns
		-------
		self: zxarray.DMUnit
			A zxarray.DMUnit object initialized
		"""
		
		if s is not None and n is None:
			self._init_from_str(s)
		elif n is not None and s is None:
			self._init_from_value( n , unit )
		elif s is None and n is None:
			raise ValueError( f"'s' or 'n' must be set!" )
		else:
			raise ValueError( f"s = {s} and n = {n} can not be set simultaneously!" )
		
	##}}}
	
	## @static.zero ##{{{
	@staticmethod
	def zero():
		"""
		zxarray.DMUnit@static.zero
		==========================
		Static method that returns zero.
		
		Parameters
		----------
		
		Returns
		-------
		s: zxarray.DMUnit
			A zxarrau.DMUnit of size 0.
		"""
		return DMUnit( n = 0 )
	##}}}
	
	## @static.one ##{{{
	@staticmethod
	def one():
		"""
		zxarray.DMUnit@static.one
		=========================
		Static method that returns one.
		
		Parameters
		----------
		
		Returns
		-------
		s: zxarray.DMUnit
			A zxarrau.DMUnit of size 1b.
		"""
		return DMUnit( n = 1 , unit = 'b' )
	##}}}
	
	## @static.sizeof_array ##{{{
	@staticmethod
	def sizeof_array(X):
		"""
		zxarray.DMUnit@static.sizeof_array
		==================================
		Static method that returns the memory size of an array.
		
		Parameters
		----------
		X: array_like as np.ndarray
			The attributes 'X.size' and 'X.dtype' must exist, and the function
			'numpy.finfo' must be able to calculate the number of bits of
			'X.dtype'.
		
		Returns
		-------
		s: zxarray.DMUnit
			Memory size of X
		"""
		try:
			bits = np.finfo(X).bits
		except:
			try:
				bits = np.iinfo(X).bits
			except:
				bits = 64
		return DMUnit( n = int( X.size * (bits // DMUnit.bits_per_octet) ) , unit = "o" )
	##}}}
	
	def _init_from_str( self , s ):##{{{
		
		if isinstance( s , type(self) ):
			s = f"{s.b}b"
		
		s = s.replace(" ","")
		
		## Start with unit
		self._unit = s[-1]
		if not self._unit.lower() in ["b","o"]:
			raise ValueError(f"Bad unit: {self._unit}")
		
		## Others values
		if s[-2] == "i":
			self._ibase = "i"
			self._base = 1024
			if s[-3].lower() in ["k","m","g","t"]:
				prefix = s[-3]
				value = s[:-3]
			else:
				prefix = ""
				value = s[:-2]
		else:
			self._ibase = ""
			self._base = 1000
			if s[-2].lower() in ["k","m","g","t"]:
				prefix = s[-2]
				value = s[:-2]
			else:
				prefix = ""
				value = s[:-1]
		
		## Check value
		try:
			try:
				value = int(value)
			except:
				value = float(value)
		except:
			raise ValueError(f"Value {value} non castable to int or float")
		
		self._prefix = prefix
		self._value = value
		
		bits = self.value * self._base**self.iprefix
		if not self._unit == "b":
			bits = bits * self.bits_per_byte
		if not 10 * int(bits) == int(10*bits):
			raise ValueError(f"Value is a subdivision of a bit, it is not possible! b = {bits}" )
		self._bits = int(bits)
	##}}}
	
	def _init_from_value( self , n , unit ):##{{{
		
		if not isinstance( n , int ):
			raise ValueError( f"n = {n} must be an integer" )
		self._init_from_str( f"{n}{unit}" )
	##}}}
	
	def __repr__( self ):##{{{
		return self.__str__()
	##}}}
	
	def __str__( self ):##{{{
		
		if int(self.o) == 0:
			return "{:.2f}o".format(self.o)
		elif int(self.ko) == 0:
			return "{:.2f}o".format(self.o)
		elif int(self.Mo) == 0:
			return "{:.2f}ko".format(self.ko)
		elif int(self.Go) == 0:
			return "{:.2f}Mo".format(self.Mo)
		elif int(self.To) == 0:
			return "{:.2f}Go".format(self.Go)
		elif int(self.Po) == 0:
			return "{:.2f}To".format(self.To)
		elif int(self.Ho) == 0:
			return "{:.2f}Po".format(self.Po)
		
		return "{:.2f}Ho".format(self.Ho)
		
	##}}}
	
	## Properties ##{{{
	
	@property
	def unit(self):
		"""
		Return the 'unit' in the decomposition:
		zxarray.DMUnit( s = f"{self.value}{self.prefix}{self.unit}" )
		"""
		return self._unit
	
	@property
	def prefix(self):
		"""
		Return the 'prefix' in the decomposition:
		zxarray.DMUnit( s = f"{self.value}{self.prefix}{self.unit}" )
		"""
		return self._prefix
	
	@property
	def value(self):
		"""
		Return the 'value' in the decomposition:
		zxarray.DMUnit( s = f"{self.value}{self.prefix}{self.unit}" )
		"""
		return self._value
	
	@property
	def bits(self):
		"""
		Return number of bits.
		"""
		return self._bits
	
	##}}}
	
	## property.iprefix ##{{{
	@property
	def iprefix(self):
		"""
		Return the exponent of the prefix:
		- if 'prefix == k', return 1
		- if 'prefix == M', return 2
		- if 'prefix == G', return 3
		- if 'prefix == T', return 4
		- if 'prefix == P', return 5
		- if 'prefix == E', return 6
		- else return 0
		"""
		match self.prefix.lower():
			case 'k': return 1
			case 'm': return 2
			case 'g': return 3
			case 't': return 4
			case 'p': return 5
			case 'e': return 6
			case  _ : return 0
		return 0
	
	##}}}
	
	## Octet properties ##{{{
	
	@property
	def o( self ):
		"""
		Return number of octets (unit prefix = 1000**0).
		"""
		return self.bits / self.bits_per_octet / 1000**0
	
	@property
	def ko( self ):
		"""
		Return number of kilo-octets (unit prefix = 1000**1).
		"""
		return self.bits / self.bits_per_octet / 1000**1
	
	@property
	def Mo( self ):
		"""
		Return number of mega-octets (unit prefix = 1000**2).
		"""
		return self.bits / self.bits_per_octet / 1000**2
	
	@property
	def Go( self ):
		"""
		Return number of giga-octets (unit prefix = 1000**3).
		"""
		return self.bits / self.bits_per_octet / 1000**3
	
	@property
	def To( self ):
		"""
		Return number of tera-octets (unit prefix = 1000**4).
		"""
		return self.bits / self.bits_per_octet / 1000**4
	
	@property
	def Po( self ):
		"""
		Return number of peta-octets (unit prefix = 1000**5).
		"""
		return self.bits / self.bits_per_octet / 1000**5
	
	@property
	def Eo( self ):
		"""
		Return number of hexa-octets (unit prefix = 1000**6).
		"""
		return self.bits / self.bits_per_octet / 1000**6
	
	##}}}
	
	## iOctet properties ##{{{
	
	@property
	def io( self ):
		"""
		Return number of i-octets (unit prefix = 1024**0).
		"""
		return self.bits / self.bits_per_octet / 1024**0
	
	@property
	def kio( self ):
		"""
		Return number of kilo-i-octets (unit prefix = 1024**1).
		"""
		return self.bits / self.bits_per_octet / 1024**1
	
	@property
	def Mio( self ):
		"""
		Return number of mega-i-octets (unit prefix = 1024**2).
		"""
		return self.bits / self.bits_per_octet / 1024**2
	
	@property
	def Gio( self ):
		"""
		Return number of giga-i-octets (unit prefix = 1024**3).
		"""
		return self.bits / self.bits_per_octet / 1024**3
	
	@property
	def Tio( self ):
		"""
		Return number of tera-i-octets (unit prefix = 1024**4).
		"""
		return self.bits / self.bits_per_octet / 1024**4
	
	@property
	def Pio( self ):
		"""
		Return number of peta-i-octets (unit prefix = 1024**5).
		"""
		return self.bits / self.bits_per_octet / 1024**5
	
	@property
	def Eio( self ):
		"""
		Return number of hexa-i-octets (unit prefix = 1024**6).
		"""
		return self.bits / self.bits_per_octet / 1024**6
	
	##}}}
	
	## Byte properties ##{{{
	
	@property
	def B( self ):
		"""
		Return number of bytes (unit prefix = 1000**0).
		"""
		return self.bits / self.bits_per_byte / 1000**0
	
	@property
	def kB( self ):
		"""
		Return number of kilo-bytes (unit prefix = 1000**1).
		"""
		return self.bits / self.bits_per_byte / 1000**1
	
	@property
	def MB( self ):
		"""
		Return number of mega-bytes (unit prefix = 1000**2).
		"""
		return self.bits / self.bits_per_byte / 1000**2
	
	@property
	def GB( self ):
		"""
		Return number of giga-bytes (unit prefix = 1000**3).
		"""
		return self.bits / self.bits_per_byte / 1000**3
	
	@property
	def TB( self ):
		"""
		Return number of tera-bytes (unit prefix = 1000**4).
		"""
		return self.bits / self.bits_per_byte / 1000**4
	
	
	@property
	def PB( self ):
		"""
		Return number of peta-bytes (unit prefix = 1000**5).
		"""
		return self.bits / self.bits_per_byte / 1000**5
	
	@property
	def EB( self ):
		"""
		Return number of hexa-bytes (unit prefix = 1000**6).
		"""
		return self.bits / self.bits_per_byte / 1000**6
	##}}}
	
	## iByte properties ##{{{
	
	@property
	def iB( self ):
		"""
		Return number of i-bytes (unit prefix = 1024**0).
		"""
		return self.bits / self.bits_per_byte / 1024**0
	
	@property
	def kiB( self ):
		"""
		Return number of kilo-i-bytes (unit prefix = 1024**1).
		"""
		return self.bits / self.bits_per_byte / 1024**1
	
	@property
	def MiB( self ):
		"""
		Return number of mega-i-bytes (unit prefix = 1024**2).
		"""
		return self.bits / self.bits_per_byte / 1024**2
	
	@property
	def GiB( self ):
		"""
		Return number of giga-i-bytes (unit prefix = 1024**3).
		"""
		return self.bits / self.bits_per_byte / 1024**3
	
	@property
	def TiB( self ):
		"""
		Return number of tera-i-bytes (unit prefix = 1024**4).
		"""
		return self.bits / self.bits_per_byte / 1024**4
	
	
	@property
	def PiB( self ):
		"""
		Return number of peta-i-bytes (unit prefix = 1024**5).
		"""
		return self.bits / self.bits_per_byte / 1024**5
	
	@property
	def EiB( self ):
		"""
		Return number of hexa-i-bytes (unit prefix = 1024**6).
		"""
		return self.bits / self.bits_per_byte / 1024**6
	
	##}}}
	
	## bits properties ##{{{
	
	@property
	def b( self ):
		"""
		Return number of bits (unit prefix = 1000**0).
		"""
		return self.bits / 1000**0
	
	@property
	def kb( self ):
		"""
		Return number of kilo-bits (unit prefix = 1000**1).
		"""
		return self.bits / 1000**1
	
	@property
	def Mb( self ):
		"""
		Return number of mega-bits (unit prefix = 1000**2).
		"""
		return self.bits / 1000**2
	
	@property
	def Gb( self ):
		"""
		Return number of giga-bits (unit prefix = 1000**3).
		"""
		return self.bits / 1000**3
	
	@property
	def Tb( self ):
		"""
		Return number of tera-bits (unit prefix = 1000**4).
		"""
		return self.bits / 1000**4
	
	@property
	def Pb( self ):
		"""
		Return number of peta-bits (unit prefix = 1000**5).
		"""
		return self.bits / 1000**5
	
	@property
	def Eb( self ):
		"""
		Return number of hexa-bits (unit prefix = 1000**6).
		"""
		return self.bits / 1000**6
	
	##}}}
	
	## ibits properties ##{{{
	
	@property
	def ib( self ):
		"""
		Return number of i-bits (unit prefix = 1024**0).
		"""
		return self.bits / 1024**0
	
	@property
	def kib( self ):
		"""
		Return number of kilo-i-bits (unit prefix = 1024**1).
		"""
		return self.bits / 1024**1
	
	@property
	def Mib( self ):
		"""
		Return number of mega-i-bits (unit prefix = 1024**2).
		"""
		return self.bits / 1024**2
	
	@property
	def Gib( self ):
		"""
		Return number of giga-i-bits (unit prefix = 1024**3).
		"""
		return self.bits / 1024**3
	
	@property
	def Tib( self ):
		"""
		Return number of tera-i-bits (unit prefix = 1024**4).
		"""
		return self.bits / 1024**4
	
	@property
	def Pib( self ):
		"""
		Return number of peta-i-bits (unit prefix = 1024**5).
		"""
		return self.bits / 1024**5
	
	@property
	def Eib( self ):
		"""
		Return number of hexa-i-bits (unit prefix = 1024**6).
		"""
		return self.bits / 1024**6
	
	##}}}
	
	## Comparison operators overload ##{{{
	
	def __eq__( self , other ):##{{{
		"""
		Equality comparison operator. 'other' can be a zxarray.DMUnit or a str.
		"""
		if isinstance(other,str):
			other = DMUnit(other)
		
		return self.bits == other.bits
	##}}}
	
	def __ne__( self , other ):##{{{
		"""
		Not equal comparison operator. 'other' can be a zxarray.DMUnit or a str.
		"""
		
		if isinstance(other,str):
			other = DMUnit(other)
		
		return self.bits != other.bits
	##}}}
	
	def __lt__( self , other ):##{{{
		"""
		Less than comparison operator. 'other' can be a zxarray.DMUnit or a str.
		"""
		
		if isinstance(other,str):
			other = DMUnit(other)
		
		return self.bits < other.bits
	##}}}
	
	def __gt__( self , other ):##{{{
		"""
		Greater than comparison operator. 'other' can be a zxarray.DMUnit or a str.
		"""
		
		if isinstance(other,str):
			other = DMUnit(other)
		
		return self.bits > other.bits
	##}}}
	
	def __le__( self , other ):##{{{
		"""
		Less than or equal comparison operator. 'other' can be a zxarray.DMUnit
		or a str.
		"""
		
		if isinstance(other,str):
			other = DMUnit(other)
		
		return self.bits <= other.bits
	##}}}
	
	def __ge__( self , other ):##{{{
		"""
		Greater than or equal comparison operator. 'other' can be a
		zxarray.DMUnit or a str.
		"""
		
		if isinstance(other,str):
			other = DMUnit(other)
		
		return self.bits >= other.bits
	##}}}
	
	##}}}
	
	## Arithmetic operators overload ##{{{
	
	def __add__( self , other ):##{{{
		"""
		Addition operator. 'other' can be a zxarray.DMUnit or a str.
		"""
		if isinstance(other,str):
			other = DMUnit(other)
		
		return DMUnit( n = self.bits + other.bits , unit = "b" )
	##}}}
	
	def __radd__( self , other ):##{{{
		"""
		Right addition operator. 'other' can be a zxarray.DMUnit or a str.
		"""
		if isinstance(other,str):
			other = DMUnit(other)
		
		return DMUnit( n = self.bits + other.bits , unit = "b" )
	##}}}
	
	def __mul__( self , x ):##{{{
		"""
		Multiplication operator. 'x' must be an integer.
		"""
		if not isinstance(x,int):
			raise ValueError( "Only multiplication by an integer is allowed" )
		
		return DMUnit( n = self.bits * x , unit = "b" )
	##}}}
	
	def __rmul__( self , x ):##{{{
		"""
		Right multiplication operator. 'x' must be an integer.
		"""
		if not isinstance(x,int):
			raise ValueError( "Only multiplication by an integer is allowed" )
		
		return DMUnit( n = self.bits * x , unit = "b" )
	##}}}
	
	def __floordiv__( self , x ):##{{{
		"""
		Integer division operator. 'x' must be an integer.
		"""
		if not isinstance(x,int):
			raise ValueError( "Only division by an integer is allowed" )
		
		return DMUnit( n = self.bits // x , unit = "b" )
	##}}}
	
	def __mod__( self , x ):##{{{
		"""
		Modulo operator. 'x' must be an integer.
		"""
		if not isinstance(x,int):
			raise ValueError( "Only modulo operator by an integer is allowed" )
		
		return DMUnit( n = self.bits % x , unit = "b" )
	##}}}
	
	##}}}
	


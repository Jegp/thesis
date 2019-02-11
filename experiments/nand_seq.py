import sys
import numpy as np
import ctypes as ct
import time
import argparse
# Hacky parser/reader/writer for values written in Futhark syntax.
# Used for reading stdin when compiling standalone programs with the
# Python code generator.

import numpy as np
import string
import struct
import sys

class ReaderInput:
    def __init__(self, f):
        self.f = f
        self.lookahead_buffer = []

    def get_char(self):
        if len(self.lookahead_buffer) == 0:
            return self.f.read(1)
        else:
            c = self.lookahead_buffer[0]
            self.lookahead_buffer = self.lookahead_buffer[1:]
            return c

    def unget_char(self, c):
        self.lookahead_buffer = [c] + self.lookahead_buffer

    def get_chars(self, n):
        n1 = min(n, len(self.lookahead_buffer))
        s = b''.join(self.lookahead_buffer[:n1])
        self.lookahead_buffer = self.lookahead_buffer[n1:]
        n2 = n - n1
        if n2 > 0:
            s += self.f.read(n2)
        return s

    def peek_char(self):
        c = self.get_char()
        if c:
            self.unget_char(c)
        return c

def skip_spaces(f):
    c = f.get_char()
    while c != None:
        if c.isspace():
            c = f.get_char()
        elif c == b'-':
          # May be line comment.
          if f.peek_char() == b'-':
            # Yes, line comment. Skip to end of line.
            while (c != b'\n' and c != None):
              c = f.get_char()
          else:
            break
        else:
          break
    if c:
        f.unget_char(c)

def parse_specific_char(f, expected):
    got = f.get_char()
    if got != expected:
        f.unget_char(got)
        raise ValueError
    return True

def parse_specific_string(f, s):
    # This funky mess is intended, and is caused by the fact that if `type(b) ==
    # bytes` then `type(b[0]) == int`, but we need to match each element with a
    # `bytes`, so therefore we make each character an array element
    b = s.encode('utf8')
    bs = [b[i:i+1] for i in range(len(b))]
    read = []
    try:
        for c in bs:
            parse_specific_char(f, c)
            read.append(c)
        return True
    except ValueError:
        map(f.unget_char, read[::-1])
        raise

def optional(p, *args):
    try:
        return p(*args)
    except ValueError:
        return None

def optional_specific_string(f, s):
    c = f.peek_char()
    # This funky mess is intended, and is caused by the fact that if `type(b) ==
    # bytes` then `type(b[0]) == int`, but we need to match each element with a
    # `bytes`, so therefore we make each character an array element
    b = s.encode('utf8')
    bs = [b[i:i+1] for i in range(len(b))]
    if c == bs[0]:
        return parse_specific_string(f, s)
    else:
        return False

def sepBy(p, sep, *args):
    elems = []
    x = optional(p, *args)
    if x != None:
        elems += [x]
        while optional(sep, *args) != None:
            x = p(*args)
            elems += [x]
    return elems

# Assumes '0x' has already been read
def parse_hex_int(f):
    s = b''
    c = f.get_char()
    while c != None:
        if c in string.hexdigits:
            s += c
            c = f.get_char()
        elif c == '_':
            c = f.get_char() # skip _
        else:
            f.unget_char(c)
            break
    return str(int(s, 16))


def parse_int(f):
    s = b''
    c = f.get_char()
    if c == b'0' and f.peek_char() in [b'x', b'X']:
        c = f.get_char() # skip X
        s += parse_hex_int(f)
    else:
        while c != None:
            if c.isdigit():
                s += c
                c = f.get_char()
            elif c == '_':
                c = f.get_char() # skip _
            else:
                f.unget_char(c)
                break
    if len(s) == 0:
        raise ValueError
    return s

def parse_int_signed(f):
    s = b''
    c = f.get_char()

    if c == b'-' and f.peek_char().isdigit():
      s = c + parse_int(f)
    else:
      if c != b'+':
          f.unget_char(c)
      s = parse_int(f)

    return s

def read_str_comma(f):
    skip_spaces(f)
    parse_specific_char(f, b',')
    return b','

def read_str_int(f, s):
    skip_spaces(f)
    x = int(parse_int_signed(f))
    optional_specific_string(f, s)
    return x

def read_str_uint(f, s):
    skip_spaces(f)
    x = int(parse_int(f))
    optional_specific_string(f, s)
    return x

def read_str_i8(f):
    return np.int8(read_str_int(f, 'i8'))
def read_str_i16(f):
    return np.int16(read_str_int(f, 'i16'))
def read_str_i32(f):
    return np.int32(read_str_int(f, 'i32'))
def read_str_i64(f):
    return np.int64(read_str_int(f, 'i64'))

def read_str_u8(f):
    return np.uint8(read_str_int(f, 'u8'))
def read_str_u16(f):
    return np.uint16(read_str_int(f, 'u16'))
def read_str_u32(f):
    return np.uint32(read_str_int(f, 'u32'))
def read_str_u64(f):
    return np.uint64(read_str_int(f, 'u64'))

def read_char(f):
    skip_spaces(f)
    parse_specific_char(f, b'\'')
    c = f.get_char()
    parse_specific_char(f, b'\'')
    return c

def read_str_hex_float(f, sign):
    int_part = parse_hex_int(f)
    parse_specific_char(f, b'.')
    frac_part = parse_hex_int(f)
    parse_specific_char(f, b'p')
    exponent = parse_int(f)

    int_val = int(int_part, 16)
    frac_val = float(int(frac_part, 16)) / (16 ** len(frac_part))
    exp_val = int(exponent)

    total_val = (int_val + frac_val) * (2.0 ** exp_val)
    if sign == b'-':
        total_val = -1 * total_val

    return float(total_val)


def read_str_decimal(f):
    skip_spaces(f)
    c = f.get_char()
    if (c == b'-'):
      sign = b'-'
    else:
      f.unget_char(c)
      sign = b''

    # Check for hexadecimal float
    c = f.get_char()
    if (c == '0' and (f.peek_char() in ['x', 'X'])):
        f.get_char()
        return read_str_hex_float(f, sign)
    else:
        f.unget_char(c)

    bef = optional(parse_int, f)
    if bef == None:
        bef = b'0'
        parse_specific_char(f, b'.')
        aft = parse_int(f)
    elif optional(parse_specific_char, f, b'.'):
        aft = parse_int(f)
    else:
        aft = b'0'
    if (optional(parse_specific_char, f, b'E') or
        optional(parse_specific_char, f, b'e')):
        expt = parse_int_signed(f)
    else:
        expt = b'0'
    return float(sign + bef + b'.' + aft + b'E' + expt)

def read_str_f32(f):
    skip_spaces(f)
    try:
        parse_specific_string(f, 'f32.nan')
        return np.float32(np.nan)
    except ValueError:
        try:
            parse_specific_string(f, 'f32.inf')
            return np.float32(np.inf)
        except ValueError:
            try:
               parse_specific_string(f, '-f32.inf')
               return np.float32(-np.inf)
            except ValueError:
               x = read_str_decimal(f)
               optional_specific_string(f, 'f32')
               return x

def read_str_f64(f):
    skip_spaces(f)
    try:
        parse_specific_string(f, 'f64.nan')
        return np.float64(np.nan)
    except ValueError:
        try:
            parse_specific_string(f, 'f64.inf')
            return np.float64(np.inf)
        except ValueError:
            try:
               parse_specific_string(f, '-f64.inf')
               return np.float64(-np.inf)
            except ValueError:
               x = read_str_decimal(f)
               optional_specific_string(f, 'f64')
               return x

def read_str_bool(f):
    skip_spaces(f)
    if f.peek_char() == b't':
        parse_specific_string(f, 'true')
        return True
    elif f.peek_char() == b'f':
        parse_specific_string(f, 'false')
        return False
    else:
        raise ValueError

def read_str_empty_array(f, type_name, rank):
    parse_specific_string(f, 'empty')
    parse_specific_char(f, b'(')
    for i in range(rank):
        parse_specific_string(f, '[]')
    parse_specific_string(f, type_name)
    parse_specific_char(f, b')')

    return None

def read_str_array_elems(f, elem_reader, type_name, rank):
    skip_spaces(f)
    try:
        parse_specific_char(f, b'[')
    except ValueError:
        return read_str_empty_array(f, type_name, rank)
    else:
        xs = sepBy(elem_reader, read_str_comma, f)
        skip_spaces(f)
        parse_specific_char(f, b']')
        return xs

def read_str_array_helper(f, elem_reader, type_name, rank):
    def nested_row_reader(_):
        return read_str_array_helper(f, elem_reader, type_name, rank-1)
    if rank == 1:
        row_reader = elem_reader
    else:
        row_reader = nested_row_reader
    return read_str_array_elems(f, row_reader, type_name, rank-1)

def expected_array_dims(l, rank):
  if rank > 1:
      n = len(l)
      if n == 0:
          elem = []
      else:
          elem = l[0]
      return [n] + expected_array_dims(elem, rank-1)
  else:
      return [len(l)]

def verify_array_dims(l, dims):
    if dims[0] != len(l):
        raise ValueError
    if len(dims) > 1:
        for x in l:
            verify_array_dims(x, dims[1:])

def read_str_array(f, elem_reader, type_name, rank, bt):
    elems = read_str_array_helper(f, elem_reader, type_name, rank)
    if elems == None:
        # Empty array
        return np.empty([0]*rank, dtype=bt)
    else:
        dims = expected_array_dims(elems, rank)
        verify_array_dims(elems, dims)
        return np.array(elems, dtype=bt)

################################################################################

READ_BINARY_VERSION = 2

# struct format specified at
# https://docs.python.org/2/library/struct.html#format-characters

def mk_bin_scalar_reader(t):
    def bin_reader(f):
        fmt = FUTHARK_PRIMTYPES[t]['bin_format']
        size = FUTHARK_PRIMTYPES[t]['size']
        return struct.unpack('<' + fmt, f.get_chars(size))[0]
    return bin_reader

read_bin_i8 = mk_bin_scalar_reader('i8')
read_bin_i16 = mk_bin_scalar_reader('i16')
read_bin_i32 = mk_bin_scalar_reader('i32')
read_bin_i64 = mk_bin_scalar_reader('i64')

read_bin_u8 = mk_bin_scalar_reader('u8')
read_bin_u16 = mk_bin_scalar_reader('u16')
read_bin_u32 = mk_bin_scalar_reader('u32')
read_bin_u64 = mk_bin_scalar_reader('u64')

read_bin_f32 = mk_bin_scalar_reader('f32')
read_bin_f64 = mk_bin_scalar_reader('f64')

read_bin_bool = mk_bin_scalar_reader('bool')

def read_is_binary(f):
    skip_spaces(f)
    c = f.get_char()
    if c == b'b':
        bin_version = read_bin_u8(f)
        if bin_version != READ_BINARY_VERSION:
            panic(1, "binary-input: File uses version %i, but I only understand version %i.\n",
                  bin_version, READ_BINARY_VERSION)
        return True
    else:
        f.unget_char(c)
        return False

FUTHARK_PRIMTYPES = {
    'i8':  {'binname' : b"  i8",
            'size' : 1,
            'bin_reader': read_bin_i8,
            'str_reader': read_str_i8,
            'bin_format': 'b',
            'numpy_type': np.int8 },

    'i16': {'binname' : b" i16",
            'size' : 2,
            'bin_reader': read_bin_i16,
            'str_reader': read_str_i16,
            'bin_format': 'h',
            'numpy_type': np.int16 },

    'i32': {'binname' : b" i32",
            'size' : 4,
            'bin_reader': read_bin_i32,
            'str_reader': read_str_i32,
            'bin_format': 'i',
            'numpy_type': np.int32 },

    'i64': {'binname' : b" i64",
            'size' : 8,
            'bin_reader': read_bin_i64,
            'str_reader': read_str_i64,
            'bin_format': 'q',
            'numpy_type': np.int64},

    'u8':  {'binname' : b"  u8",
            'size' : 1,
            'bin_reader': read_bin_u8,
            'str_reader': read_str_u8,
            'bin_format': 'B',
            'numpy_type': np.uint8 },

    'u16': {'binname' : b" u16",
            'size' : 2,
            'bin_reader': read_bin_u16,
            'str_reader': read_str_u16,
            'bin_format': 'H',
            'numpy_type': np.uint16 },

    'u32': {'binname' : b" u32",
            'size' : 4,
            'bin_reader': read_bin_u32,
            'str_reader': read_str_u32,
            'bin_format': 'I',
            'numpy_type': np.uint32 },

    'u64': {'binname' : b" u64",
            'size' : 8,
            'bin_reader': read_bin_u64,
            'str_reader': read_str_u64,
            'bin_format': 'Q',
            'numpy_type': np.uint64 },

    'f32': {'binname' : b" f32",
            'size' : 4,
            'bin_reader': read_bin_f32,
            'str_reader': read_str_f32,
            'bin_format': 'f',
            'numpy_type': np.float32 },

    'f64': {'binname' : b" f64",
            'size' : 8,
            'bin_reader': read_bin_f64,
            'str_reader': read_str_f64,
            'bin_format': 'd',
            'numpy_type': np.float64 },

    'bool': {'binname' : b"bool",
             'size' : 1,
             'bin_reader': read_bin_bool,
             'str_reader': read_str_bool,
             'bin_format': 'b',
             'numpy_type': np.bool }
}

def read_bin_read_type(f):
    read_binname = f.get_chars(4)

    for (k,v) in FUTHARK_PRIMTYPES.items():
        if v['binname'] == read_binname:
            return k
    panic(1, "binary-input: Did not recognize the type '%s'.\n", read_binname)

def numpy_type_to_type_name(t):
    for (k,v) in FUTHARK_PRIMTYPES.items():
        if v['numpy_type'] == t:
            return k
    raise Exception('Unknown Numpy type: {}'.format(t))

def read_bin_ensure_scalar(f, expected_type):
  dims = read_bin_i8(f)

  if dims != 0:
      panic(1, "binary-input: Expected scalar (0 dimensions), but got array with %i dimensions.\n", dims)

  bin_type = read_bin_read_type(f)
  if bin_type != expected_type:
      panic(1, "binary-input: Expected scalar of type %s but got scalar of type %s.\n",
            expected_type, bin_type)

# ------------------------------------------------------------------------------
# General interface for reading Primitive Futhark Values
# ------------------------------------------------------------------------------

def read_scalar(f, ty):
    if read_is_binary(f):
        read_bin_ensure_scalar(f, ty)
        return FUTHARK_PRIMTYPES[ty]['bin_reader'](f)
    return FUTHARK_PRIMTYPES[ty]['str_reader'](f)

def read_array(f, expected_type, rank):
    if not read_is_binary(f):
        str_reader = FUTHARK_PRIMTYPES[expected_type]['str_reader']
        return read_str_array(f, str_reader, expected_type, rank,
                              FUTHARK_PRIMTYPES[expected_type]['numpy_type'])

    bin_rank = read_bin_u8(f)

    if bin_rank != rank:
        panic(1, "binary-input: Expected %i dimensions, but got array with %i dimensions.\n",
              rank, bin_rank)

    bin_type_enum = read_bin_read_type(f)
    if expected_type != bin_type_enum:
        panic(1, "binary-input: Expected %iD-array with element type '%s' but got %iD-array with element type '%s'.\n",
              rank, expected_type, bin_rank, bin_type_enum)

    shape = []
    elem_count = 1
    for i in range(rank):
        bin_size = read_bin_u64(f)
        elem_count *= bin_size
        shape.append(bin_size)

    bin_fmt = FUTHARK_PRIMTYPES[bin_type_enum]['bin_format']

    # We first read the expected number of types into a bytestring,
    # then use np.fromstring.  This is because np.fromfile does not
    # work on things that are insufficiently file-like, like a network
    # stream.
    bytes = f.get_chars(elem_count * FUTHARK_PRIMTYPES[expected_type]['size'])
    arr = np.fromstring(bytes, dtype='<'+bin_fmt)
    arr.shape = shape

    return arr

if sys.version_info >= (3,0):
    input_reader = ReaderInput(sys.stdin.buffer)
else:
    input_reader = ReaderInput(sys.stdin)

import re

def read_value(type_desc, reader=input_reader):
    """Read a value of the given type.  The type is a string
representation of the Futhark type."""
    m = re.match(r'((?:\[\])*)([a-z0-9]+)$', type_desc)
    if m:
        dims = int(len(m.group(1))/2)
        basetype = m.group(2)
        assert basetype in FUTHARK_PRIMTYPES, "Unknown type: {}".format(type_desc)
        if dims > 0:
            return read_array(reader, basetype, dims)
        else:
            return read_scalar(reader, basetype)
        return (dims, basetype)

def write_value(v, out=sys.stdout):
    if type(v) == np.uint8:
        out.write("%uu8" % v)
    elif type(v) == np.uint16:
        out.write("%uu16" % v)
    elif type(v) == np.uint32:
        out.write("%uu32" % v)
    elif type(v) == np.uint64:
        out.write("%uu64" % v)
    elif type(v) == np.int8:
        out.write("%di8" % v)
    elif type(v) == np.int16:
        out.write("%di16" % v)
    elif type(v) == np.int32:
        out.write("%di32" % v)
    elif type(v) == np.int64:
        out.write("%di64" % v)
    elif type(v) in [np.bool, np.bool_]:
        if v:
            out.write("true")
        else:
            out.write("false")
    elif type(v) == np.float32:
        if np.isnan(v):
            out.write('f32.nan')
        elif np.isinf(v):
            if v >= 0:
                out.write('f32.inf')
            else:
                out.write('-f32.inf')
        else:
            out.write("%.6ff32" % v)
    elif type(v) == np.float64:
        if np.isnan(v):
            out.write('f64.nan')
        elif np.isinf(v):
            if v >= 0:
                out.write('f64.inf')
            else:
                out.write('-f64.inf')
        else:
            out.write("%.6ff64" % v)
    elif type(v) == np.ndarray:
        if np.product(v.shape) == 0:
            tname = numpy_type_to_type_name(v.dtype)
            out.write('empty({}{})'.format(''.join(['[]' for _ in v.shape[1:]]), tname))
        else:
            first = True
            out.write('[')
            for x in v:
                if not first: out.write(', ')
                first = False
                write_value(x, out=out)
            out.write(']')
    else:
        raise Exception("Cannot print value of type {}: {}".format(type(v), v))

################################################################################
### end of values.py
################################################################################
# Helper functions dealing with memory blocks.

import ctypes as ct

def addressOffset(x, offset, bt):
  return ct.cast(ct.addressof(x.contents)+int(offset), ct.POINTER(bt))

def allocateMem(size):
  return ct.cast((ct.c_byte * max(0,size))(), ct.POINTER(ct.c_byte))

# Copy an array if its is not-None.  This is important for treating
# Numpy arrays as flat memory, but has some overhead.
def normaliseArray(x):
  if (x.base is x) or (x.base is None):
    return x
  else:
    return x.copy()

def unwrapArray(x):
  return normaliseArray(x).ctypes.data_as(ct.POINTER(ct.c_byte))

def createArray(x, dim):
  return np.ctypeslib.as_array(x, shape=dim)

def indexArray(x, offset, bt, nptype):
  return nptype(addressOffset(x, offset, bt)[0])

def writeScalarArray(x, offset, v):
  ct.memmove(ct.addressof(x.contents)+int(offset), ct.addressof(v), ct.sizeof(v))

# An opaque Futhark value.
class opaque(object):
  def __init__(self, desc, *payload):
    self.data = payload
    self.desc = desc

  def __repr__(self):
    return "<opaque Futhark value of type {}>".format(self.desc)
def panic(exitcode, fmt, *args):
    sys.stderr.write('%s: ' % sys.argv[0])
    sys.stderr.write(fmt % args)
    sys.exit(exitcode)
# Scalar functions.

import numpy as np
import struct

def signed(x):
  if type(x) == np.uint8:
    return np.int8(x)
  elif type(x) == np.uint16:
    return np.int16(x)
  elif type(x) == np.uint32:
    return np.int32(x)
  else:
    return np.int64(x)

def unsigned(x):
  if type(x) == np.int8:
    return np.uint8(x)
  elif type(x) == np.int16:
    return np.uint16(x)
  elif type(x) == np.int32:
    return np.uint32(x)
  else:
    return np.uint64(x)

def shlN(x,y):
  return x << y

def ashrN(x,y):
  return x >> y

def sdivN(x,y):
  return x // y

def smodN(x,y):
  return x % y

def udivN(x,y):
  return signed(unsigned(x) // unsigned(y))

def umodN(x,y):
  return signed(unsigned(x) % unsigned(y))

def squotN(x,y):
  return np.floor_divide(np.abs(x), np.abs(y)) * np.sign(x) * np.sign(y)

def sremN(x,y):
  return np.remainder(np.abs(x), np.abs(y)) * np.sign(x)

def sminN(x,y):
  return min(x,y)

def smaxN(x,y):
  return max(x,y)

def uminN(x,y):
  return signed(min(unsigned(x),unsigned(y)))

def umaxN(x,y):
  return signed(max(unsigned(x),unsigned(y)))

def fminN(x,y):
  return min(x,y)

def fmaxN(x,y):
  return max(x,y)

def powN(x,y):
  return x ** y

def fpowN(x,y):
  return x ** y

def sleN(x,y):
  return x <= y

def sltN(x,y):
  return x < y

def uleN(x,y):
  return unsigned(x) <= unsigned(y)

def ultN(x,y):
  return unsigned(x) < unsigned(y)

def lshr8(x,y):
  return np.int8(np.uint8(x) >> np.uint8(y))

def lshr16(x,y):
  return np.int16(np.uint16(x) >> np.uint16(y))

def lshr32(x,y):
  return np.int32(np.uint32(x) >> np.uint32(y))

def lshr64(x,y):
  return np.int64(np.uint64(x) >> np.uint64(y))

def sext_T_i8(x):
  return np.int8(x)

def sext_T_i16(x):
  return np.int16(x)

def sext_T_i32(x):
  return np.int32(x)

def sext_T_i64(x):
  return np.int64(x)

def itob_T_bool(x):
  return np.bool(x)

def btoi_bool_i8(x):
  return np.int8(x)

def btoi_bool_i16(x):
  return np.int8(x)

def btoi_bool_i32(x):
  return np.int8(x)

def btoi_bool_i64(x):
  return np.int8(x)

def zext_i8_i8(x):
  return np.int8(np.uint8(x))

def zext_i8_i16(x):
  return np.int16(np.uint8(x))

def zext_i8_i32(x):
  return np.int32(np.uint8(x))

def zext_i8_i64(x):
  return np.int64(np.uint8(x))

def zext_i16_i8(x):
  return np.int8(np.uint16(x))

def zext_i16_i16(x):
  return np.int16(np.uint16(x))

def zext_i16_i32(x):
  return np.int32(np.uint16(x))

def zext_i16_i64(x):
  return np.int64(np.uint16(x))

def zext_i32_i8(x):
  return np.int8(np.uint32(x))

def zext_i32_i16(x):
  return np.int16(np.uint32(x))

def zext_i32_i32(x):
  return np.int32(np.uint32(x))

def zext_i32_i64(x):
  return np.int64(np.uint32(x))

def zext_i64_i8(x):
  return np.int8(np.uint64(x))

def zext_i64_i16(x):
  return np.int16(np.uint64(x))

def zext_i64_i32(x):
  return np.int32(np.uint64(x))

def zext_i64_i64(x):
  return np.int64(np.uint64(x))

shl8 = shl16 = shl32 = shl64 = shlN
ashr8 = ashr16 = ashr32 = ashr64 = ashrN
sdiv8 = sdiv16 = sdiv32 = sdiv64 = sdivN
smod8 = smod16 = smod32 = smod64 = smodN
udiv8 = udiv16 = udiv32 = udiv64 = udivN
umod8 = umod16 = umod32 = umod64 = umodN
squot8 = squot16 = squot32 = squot64 = squotN
srem8 = srem16 = srem32 = srem64 = sremN
smax8 = smax16 = smax32 = smax64 = smaxN
smin8 = smin16 = smin32 = smin64 = sminN
umax8 = umax16 = umax32 = umax64 = umaxN
umin8 = umin16 = umin32 = umin64 = uminN
pow8 = pow16 = pow32 = pow64 = powN
fpow32 = fpow64 = fpowN
fmax32 = fmax64 = fmaxN
fmin32 = fmin64 = fminN
sle8 = sle16 = sle32 = sle64 = sleN
slt8 = slt16 = slt32 = slt64 = sltN
ule8 = ule16 = ule32 = ule64 = uleN
ult8 = ult16 = ult32 = ult64 = ultN
sext_i8_i8 = sext_i16_i8 = sext_i32_i8 = sext_i64_i8 = sext_T_i8
sext_i8_i16 = sext_i16_i16 = sext_i32_i16 = sext_i64_i16 = sext_T_i16
sext_i8_i32 = sext_i16_i32 = sext_i32_i32 = sext_i64_i32 = sext_T_i32
sext_i8_i64 = sext_i16_i64 = sext_i32_i64 = sext_i64_i64 = sext_T_i64
itob_i8_bool = itob_i16_bool = itob_i32_bool = itob_i64_bool = itob_T_bool

def ssignum(x):
  return np.sign(x)

def usignum(x):
  if x < 0:
    return ssignum(-x)
  else:
    return ssignum(x)

def sitofp_T_f32(x):
  return np.float32(x)
sitofp_i8_f32 = sitofp_i16_f32 = sitofp_i32_f32 = sitofp_i64_f32 = sitofp_T_f32

def sitofp_T_f64(x):
  return np.float64(x)
sitofp_i8_f64 = sitofp_i16_f64 = sitofp_i32_f64 = sitofp_i64_f64 = sitofp_T_f64

def uitofp_T_f32(x):
  return np.float32(unsigned(x))
uitofp_i8_f32 = uitofp_i16_f32 = uitofp_i32_f32 = uitofp_i64_f32 = uitofp_T_f32

def uitofp_T_f64(x):
  return np.float64(unsigned(x))
uitofp_i8_f64 = uitofp_i16_f64 = uitofp_i32_f64 = uitofp_i64_f64 = uitofp_T_f64

def fptosi_T_i8(x):
  return np.int8(np.trunc(x))
fptosi_f32_i8 = fptosi_f64_i8 = fptosi_T_i8

def fptosi_T_i16(x):
  return np.int16(np.trunc(x))
fptosi_f32_i16 = fptosi_f64_i16 = fptosi_T_i16

def fptosi_T_i32(x):
  return np.int32(np.trunc(x))
fptosi_f32_i32 = fptosi_f64_i32 = fptosi_T_i32

def fptosi_T_i64(x):
  return np.int64(np.trunc(x))
fptosi_f32_i64 = fptosi_f64_i64 = fptosi_T_i64

def fptoui_T_i8(x):
  return np.uint8(np.trunc(x))
fptoui_f32_i8 = fptoui_f64_i8 = fptoui_T_i8

def fptoui_T_i16(x):
  return np.uint16(np.trunc(x))
fptoui_f32_i16 = fptoui_f64_i16 = fptoui_T_i16

def fptoui_T_i32(x):
  return np.uint32(np.trunc(x))
fptoui_f32_i32 = fptoui_f64_i32 = fptoui_T_i32

def fptoui_T_i64(x):
  return np.uint64(np.trunc(x))
fptoui_f32_i64 = fptoui_f64_i64 = fptoui_T_i64

def fpconv_f32_f64(x):
  return np.float64(x)

def fpconv_f64_f32(x):
  return np.float32(x)

def futhark_log64(x):
  return np.float64(np.log(x))

def futhark_log2_64(x):
  return np.float64(np.log2(x))

def futhark_log10_64(x):
  return np.float64(np.log10(x))

def futhark_sqrt64(x):
  return np.sqrt(x)

def futhark_exp64(x):
  return np.exp(x)

def futhark_cos64(x):
  return np.cos(x)

def futhark_sin64(x):
  return np.sin(x)

def futhark_tan64(x):
  return np.tan(x)

def futhark_acos64(x):
  return np.arccos(x)

def futhark_asin64(x):
  return np.arcsin(x)

def futhark_atan64(x):
  return np.arctan(x)

def futhark_atan2_64(x, y):
  return np.arctan2(x, y)

def futhark_round64(x):
  return np.round(x)

def futhark_isnan64(x):
  return np.isnan(x)

def futhark_isinf64(x):
  return np.isinf(x)

def futhark_to_bits64(x):
  s = struct.pack('>d', x)
  return np.int64(struct.unpack('>q', s)[0])

def futhark_from_bits64(x):
  s = struct.pack('>q', x)
  return np.float64(struct.unpack('>d', s)[0])

def futhark_log32(x):
  return np.float32(np.log(x))

def futhark_log2_32(x):
  return np.float32(np.log2(x))

def futhark_log10_32(x):
  return np.float32(np.log10(x))

def futhark_sqrt32(x):
  return np.float32(np.sqrt(x))

def futhark_exp32(x):
  return np.exp(x)

def futhark_cos32(x):
  return np.cos(x)

def futhark_sin32(x):
  return np.sin(x)

def futhark_tan32(x):
  return np.tan(x)

def futhark_acos32(x):
  return np.arccos(x)

def futhark_asin32(x):
  return np.arcsin(x)

def futhark_atan32(x):
  return np.arctan(x)

def futhark_atan2_32(x, y):
  return np.arctan2(x, y)

def futhark_round32(x):
  return np.round(x)

def futhark_isnan32(x):
  return np.isnan(x)

def futhark_isinf32(x):
  return np.isinf(x)

def futhark_to_bits32(x):
  s = struct.pack('>f', x)
  return np.int32(struct.unpack('>l', s)[0])

def futhark_from_bits32(x):
  s = struct.pack('>l', x)
  return np.float32(struct.unpack('>f', s)[0])
class nand:
  entry_points = {"main": (["[][]f64", "[][]f64"], ["f64",
                                                    "(([][]f64, []f64), ([][]f64, []f64))"])}
  def __init__(self):
    pass
  def futhark_main(self, input_mem_sizze_36616, input_mem_36617,
                   labels_mem_sizze_36618, labels_mem_36619, sizze_35973,
                   sizze_35974, sizze_35975, sizze_35976):
    dim_zzero_35979 = (np.int32(0) == sizze_35975)
    dim_zzero_35980 = (np.int32(0) == sizze_35976)
    old_empty_35981 = (dim_zzero_35979 or dim_zzero_35980)
    dim_zzero_35982 = (np.int32(0) == sizze_35973)
    new_empty_35983 = (dim_zzero_35980 or dim_zzero_35982)
    both_empty_35984 = (old_empty_35981 and new_empty_35983)
    dim_match_35985 = (sizze_35973 == sizze_35975)
    empty_or_match_35986 = (both_empty_35984 or dim_match_35985)
    empty_or_match_cert_35987 = True
    assert empty_or_match_35986, ("Error at nand.fut:8:1-20:23: %s" % ("function arguments of wrong shape",))
    res_35988 = sitofp_i32_f64(sizze_35973)
    arg_35989 = (np.float64(0.8) * res_35988)
    res_35990 = fptosi_f64_i32(arg_35989)
    y_35991 = srem32(res_35990, np.int32(128))
    res_35992 = (res_35990 - y_35991)
    arg_35993 = (np.float64(0.2) * res_35988)
    res_35994 = fptosi_f64_i32(arg_35993)
    y_35995 = srem32(res_35994, np.int32(128))
    res_35996 = (res_35994 - y_35995)
    mem_36622 = allocateMem(np.int64(64))
    i_36357 = np.int32(0)
    one_36905 = np.int32(1)
    for counter_36904 in range(np.int32(8)):
      x_36000 = (np.int32(8) + i_36357)
      res_36001 = (np.int32(1) + x_36000)
      arg_36002 = (np.int32(5461) ^ res_36001)
      arg_36003 = (np.int32(1) ^ arg_36002)
      arg_36004 = (np.int32(48271) * arg_36003)
      arg_36005 = umod32(arg_36004, np.int32(2147483647))
      arg_36006 = (np.int32(48271) * arg_36005)
      arg_36007 = umod32(arg_36006, np.int32(2147483647))
      res_36008 = uitofp_i32_f64(arg_36007)
      res_36009 = (res_36008 / np.float64(2.147483647e9))
      res_36010 = (np.float64(2.0) * res_36009)
      res_36011 = (np.float64(-1.0) + res_36010)
      writeScalarArray(mem_36622, (i_36357 * np.int32(8)),
                       ct.c_double(res_36011))
      i_36357 += one_36905
    mem_36629 = allocateMem(np.int64(32))
    i_36851 = np.int32(0)
    one_36907 = np.int32(1)
    for counter_36906 in range(np.int32(4)):
      writeScalarArray(mem_36629, (i_36851 * np.int32(8)),
                       ct.c_double(np.float64(0.0)))
      i_36851 += one_36907
    mem_36632 = allocateMem(np.int64(64))
    i_36361 = np.int32(0)
    one_36909 = np.int32(1)
    for counter_36908 in range(np.int32(8)):
      x_36016 = (np.int32(8) + i_36361)
      res_36017 = (np.int32(2) + x_36016)
      arg_36018 = (np.int32(5461) ^ res_36017)
      arg_36019 = (np.int32(1) ^ arg_36018)
      arg_36020 = (np.int32(48271) * arg_36019)
      arg_36021 = umod32(arg_36020, np.int32(2147483647))
      arg_36022 = (np.int32(48271) * arg_36021)
      arg_36023 = umod32(arg_36022, np.int32(2147483647))
      res_36024 = uitofp_i32_f64(arg_36023)
      res_36025 = (res_36024 / np.float64(2.147483647e9))
      res_36026 = (np.float64(2.0) * res_36025)
      res_36027 = (np.float64(-1.0) + res_36026)
      writeScalarArray(mem_36632, (i_36361 * np.int32(8)),
                       ct.c_double(res_36027))
      i_36361 += one_36909
    mem_36639 = allocateMem(np.int64(16))
    i_36853 = np.int32(0)
    one_36911 = np.int32(1)
    for counter_36910 in range(np.int32(2)):
      writeScalarArray(mem_36639, (i_36853 * np.int32(8)),
                       ct.c_double(np.float64(0.0)))
      i_36853 += one_36911
    empty_slice_36030 = (res_35992 == np.int32(0))
    m_36031 = (res_35992 - np.int32(1))
    zzero_leq_i_p_m_t_s_36032 = sle32(np.int32(0), m_36031)
    i_p_m_t_s_leq_w_36033 = slt32(m_36031, sizze_35973)
    i_lte_j_36034 = sle32(np.int32(0), res_35992)
    y_36035 = (zzero_leq_i_p_m_t_s_36032 and i_p_m_t_s_leq_w_36033)
    y_36036 = (i_lte_j_36034 and y_36035)
    ok_or_empty_36037 = (empty_slice_36030 or y_36036)
    index_certs_36038 = True
    assert ok_or_empty_36037, ("Error at nand.fut:8:1-20:23 -> nand.fut:16:13-25: %s%s%s%d%s%d%s" % ("Index [",
                                                                                                     "",
                                                                                                     ":",
                                                                                                     res_35992,
                                                                                                     "] out of bounds for array of shape [",
                                                                                                     sizze_35973,
                                                                                                     "]."))
    loop_cond_36039 = slt32(np.int32(0), res_35992)
    dim_match_36040 = (np.int32(2) == sizze_35974)
    empty_or_match_cert_36041 = True
    if loop_cond_36039:
      x_36042 = True
      assert dim_match_36040, ("Error at nand.fut:8:1-20:23 -> nand.fut:15:13-17:64 -> ../lib/github.com/HnimNart/deeplearning/optimizers/optimizers.fut:27:5-47 -> ../lib/github.com/HnimNart/deeplearning/optimizers/gradient_descent.fut:46:53-69 -> ../lib/github.com/HnimNart/deeplearning/neural_network.fut:73:46-68 -> ../lib/github.com/HnimNart/deeplearning/layers/dense.fut:47:20-50: %s" % ("function arguments of wrong shape",))
      empty_or_match_cert_36041 = x_36042
    else:
      empty_or_match_cert_36041 = True
    dim_match_36043 = (np.int32(2) == sizze_35976)
    empty_or_match_cert_36044 = True
    if loop_cond_36039:
      x_36045 = True
      assert dim_match_36043, ("Error at nand.fut:8:1-20:23 -> nand.fut:15:13-17:64 -> ../lib/github.com/HnimNart/deeplearning/optimizers/optimizers.fut:27:5-47 -> ../lib/github.com/HnimNart/deeplearning/optimizers/gradient_descent.fut:47:42-79 -> /futlib/soacs.fut:51:3-37 -> /futlib/soacs.fut:51:19-23 -> ../lib/github.com/HnimNart/deeplearning/optimizers/gradient_descent.fut:47:56-64: %s" % ("function arguments of wrong shape",))
      empty_or_match_cert_36044 = x_36045
    else:
      empty_or_match_cert_36044 = True
    empty_or_match_cert_36046 = True
    if loop_cond_36039:
      x_36047 = True
      assert dim_match_36040, ("Error at nand.fut:8:1-20:23 -> nand.fut:15:13-17:64 -> ../lib/github.com/HnimNart/deeplearning/optimizers/optimizers.fut:27:5-47 -> ../lib/github.com/HnimNart/deeplearning/optimizers/gradient_descent.fut:48:42-70 -> ../lib/github.com/HnimNart/deeplearning/neural_network.fut:78:47-66 -> ../lib/github.com/HnimNart/deeplearning/layers/dense.fut:65:20-53 -> ../lib/github.com/HnimNart/deeplearning/optimizers/gradient_descent.fut:27:21-47: %s" % ("function arguments of wrong shape",))
      empty_or_match_cert_36046 = x_36047
    else:
      empty_or_match_cert_36046 = True
    mem_36650 = allocateMem(np.int64(4096))
    mem_36667 = allocateMem(np.int64(2048))
    mem_36672 = allocateMem(np.int64(16))
    mem_36675 = allocateMem(np.int64(16))
    mem_36686 = allocateMem(np.int64(16))
    convop_x_36728 = (np.int32(4) * sizze_35974)
    binop_x_36729 = sext_i32_i64(convop_x_36728)
    bytes_36727 = (np.int64(8) * binop_x_36729)
    mem_36730 = allocateMem(bytes_36727)
    mem_36735 = allocateMem(np.int64(1024))
    w_mem_36641 = mem_36622
    w_mem_36643 = mem_36629
    w_mem_36645 = mem_36632
    w_mem_36647 = mem_36639
    loop_while_36054 = loop_cond_36039
    i_36059 = np.int32(0)
    while loop_while_36054:
      j_36060 = (np.int32(128) + i_36059)
      i_p_m_t_s_36061 = (np.int32(127) + i_36059)
      zzero_leq_i_p_m_t_s_36062 = sle32(np.int32(0), i_p_m_t_s_36061)
      i_p_m_t_s_leq_w_36063 = slt32(i_p_m_t_s_36061, res_35992)
      zzero_lte_i_36064 = sle32(np.int32(0), i_36059)
      y_36066 = (i_p_m_t_s_leq_w_36063 and zzero_lte_i_36064)
      y_36067 = (zzero_leq_i_p_m_t_s_36062 and y_36066)
      forwards_ok_36069 = (zzero_lte_i_36064 and y_36067)
      index_certs_36070 = True
      assert forwards_ok_36069, ("Error at nand.fut:8:1-20:23 -> nand.fut:15:13-17:64 -> ../lib/github.com/HnimNart/deeplearning/optimizers/optimizers.fut:27:5-47 -> ../lib/github.com/HnimNart/deeplearning/optimizers/gradient_descent.fut:44:42-60: %s%d%s%d%s%d%s" % ("Index [",
                                                                                                                                                                                                                                                                           i_36059,
                                                                                                                                                                                                                                                                           ":",
                                                                                                                                                                                                                                                                           j_36060,
                                                                                                                                                                                                                                                                           "] out of bounds for array of shape [",
                                                                                                                                                                                                                                                                           res_35992,
                                                                                                                                                                                                                                                                           "]."))
      i_36371 = np.int32(0)
      one_36917 = np.int32(1)
      for counter_36916 in range(np.int32(4)):
        x_36078 = indexArray(w_mem_36643, (i_36371 * np.int32(8)), ct.c_double,
                             np.float64)
        i_36367 = np.int32(0)
        one_36915 = np.int32(1)
        for counter_36914 in range(np.int32(128)):
          j_p_i_t_s_36489 = (i_36059 + i_36367)
          redout_36363 = np.float64(0.0)
          i_36364 = np.int32(0)
          one_36913 = np.int32(1)
          for counter_36912 in range(np.int32(2)):
            x_36085 = indexArray(w_mem_36641,
                                 (((i_36371 * np.int32(2)) + i_36364) * np.int32(8)),
                                 ct.c_double, np.float64)
            x_36086 = indexArray(input_mem_36617,
                                 (((j_p_i_t_s_36489 * sizze_35974) + i_36364) * np.int32(8)),
                                 ct.c_double, np.float64)
            res_36087 = (x_36085 * x_36086)
            res_36084 = (res_36087 + redout_36363)
            redout_tmp_36866 = res_36084
            redout_36363 = redout_tmp_36866
            i_36364 += one_36913
          res_36081 = redout_36363
          res_36088 = (x_36078 + res_36081)
          res_36089 = fmax64(np.float64(0.0), res_36088)
          writeScalarArray(mem_36650,
                           (((i_36371 * np.int32(128)) + i_36367) * np.int32(8)),
                           ct.c_double(res_36089))
          i_36367 += one_36915
        i_36371 += one_36917
      i_36394 = np.int32(0)
      one_36927 = np.int32(1)
      for counter_36926 in range(np.int32(128)):
        redout_36377 = -np.inf
        i_36380 = np.int32(0)
        one_36921 = np.int32(1)
        for counter_36920 in range(np.int32(2)):
          x_wasfree_36104 = indexArray(w_mem_36647, (i_36380 * np.int32(8)),
                                       ct.c_double, np.float64)
          redout_36373 = np.float64(0.0)
          i_36374 = np.int32(0)
          one_36919 = np.int32(1)
          for counter_36918 in range(np.int32(4)):
            x_36109 = indexArray(w_mem_36645,
                                 (((i_36380 * np.int32(4)) + i_36374) * np.int32(8)),
                                 ct.c_double, np.float64)
            x_36110 = indexArray(mem_36650,
                                 (((i_36374 * np.int32(128)) + i_36394) * np.int32(8)),
                                 ct.c_double, np.float64)
            res_36111 = (x_36109 * x_36110)
            res_36108 = (res_36111 + redout_36373)
            redout_tmp_36871 = res_36108
            redout_36373 = redout_tmp_36871
            i_36374 += one_36919
          res_36105 = redout_36373
          res_36112 = (x_wasfree_36104 + res_36105)
          res_36113 = fmax64(np.float64(0.0), res_36112)
          res_36101 = fmax64(res_36113, redout_36377)
          writeScalarArray(mem_36672, (i_36380 * np.int32(8)),
                           ct.c_double(res_36113))
          writeScalarArray(mem_36675, (i_36380 * np.int32(8)),
                           ct.c_double(res_36112))
          redout_tmp_36868 = res_36101
          redout_36377 = redout_tmp_36868
          i_36380 += one_36921
        res_36096 = redout_36377
        redout_36384 = np.float64(0.0)
        i_36386 = np.int32(0)
        one_36923 = np.int32(1)
        for counter_36922 in range(np.int32(2)):
          x_36119 = indexArray(mem_36672, (i_36386 * np.int32(8)), ct.c_double,
                               np.float64)
          res_36120 = (x_36119 - res_36096)
          res_36121 = futhark_exp64(res_36120)
          res_36118 = (res_36121 + redout_36384)
          writeScalarArray(mem_36686, (i_36386 * np.int32(8)),
                           ct.c_double(res_36121))
          redout_tmp_36872 = res_36118
          redout_36384 = redout_tmp_36872
          i_36386 += one_36923
        res_36114 = redout_36384
        j_p_i_t_s_36505 = (i_36059 + i_36394)
        i_36390 = np.int32(0)
        one_36925 = np.int32(1)
        for counter_36924 in range(np.int32(2)):
          x_36124 = indexArray(mem_36686, (i_36390 * np.int32(8)), ct.c_double,
                               np.float64)
          x_36125 = indexArray(labels_mem_36619,
                               (((j_p_i_t_s_36505 * sizze_35976) + i_36390) * np.int32(8)),
                               ct.c_double, np.float64)
          x_36126 = indexArray(mem_36675, (i_36390 * np.int32(8)), ct.c_double,
                               np.float64)
          res_36127 = (x_36124 / res_36114)
          res_36128 = (res_36127 - x_36125)
          res_36129 = (x_36126 <= np.float64(0.0))
          if res_36129:
            res_36130 = np.float64(0.0)
          else:
            res_36130 = np.float64(1.0)
          res_36131 = (res_36128 * res_36130)
          writeScalarArray(mem_36667,
                           (((i_36394 * np.int32(2)) + i_36390) * np.int32(8)),
                           ct.c_double(res_36131))
          i_36390 += one_36925
        i_36394 += one_36927
      mem_36705 = allocateMem(np.int64(16))
      mem_36708 = allocateMem(np.int64(64))
      i_36408 = np.int32(0)
      one_36935 = np.int32(1)
      for counter_36934 in range(np.int32(2)):
        x_36137 = indexArray(w_mem_36647, (i_36408 * np.int32(8)), ct.c_double,
                             np.float64)
        redout_36396 = np.float64(0.0)
        i_36397 = np.int32(0)
        one_36929 = np.int32(1)
        for counter_36928 in range(np.int32(128)):
          x_36142 = indexArray(mem_36667,
                               (((i_36397 * np.int32(2)) + i_36408) * np.int32(8)),
                               ct.c_double, np.float64)
          res_36141 = (x_36142 + redout_36396)
          redout_tmp_36877 = res_36141
          redout_36396 = redout_tmp_36877
          i_36397 += one_36929
        res_36138 = redout_36396
        res_36143 = (res_36138 / np.float64(128.0))
        res_36144 = (np.float64(0.1) * res_36143)
        i_36402 = np.int32(0)
        one_36933 = np.int32(1)
        for counter_36932 in range(np.int32(4)):
          x_36147 = indexArray(w_mem_36645,
                               (((i_36408 * np.int32(4)) + i_36402) * np.int32(8)),
                               ct.c_double, np.float64)
          redout_36398 = np.float64(0.0)
          i_36399 = np.int32(0)
          one_36931 = np.int32(1)
          for counter_36930 in range(np.int32(128)):
            x_36152 = indexArray(mem_36667,
                                 (((i_36399 * np.int32(2)) + i_36408) * np.int32(8)),
                                 ct.c_double, np.float64)
            x_36153 = indexArray(mem_36650,
                                 (((i_36402 * np.int32(128)) + i_36399) * np.int32(8)),
                                 ct.c_double, np.float64)
            res_36154 = (x_36152 * x_36153)
            res_36151 = (res_36154 + redout_36398)
            redout_tmp_36879 = res_36151
            redout_36398 = redout_tmp_36879
            i_36399 += one_36931
          res_36148 = redout_36398
          res_36155 = (res_36148 / np.float64(128.0))
          res_36156 = (np.float64(0.1) * res_36155)
          res_36157 = (x_36147 - res_36156)
          writeScalarArray(mem_36708,
                           (((i_36408 * np.int32(4)) + i_36402) * np.int32(8)),
                           ct.c_double(res_36157))
          i_36402 += one_36933
        res_36158 = (x_36137 - res_36144)
        writeScalarArray(mem_36705, (i_36408 * np.int32(8)),
                         ct.c_double(res_36158))
        i_36408 += one_36935
      i_36427 = np.int32(0)
      one_36947 = np.int32(1)
      for counter_36946 in range(np.int32(4)):
        x_36162 = indexArray(w_mem_36643, (i_36427 * np.int32(8)), ct.c_double,
                             np.float64)
        i_36417 = np.int32(0)
        one_36941 = np.int32(1)
        for counter_36940 in range(np.int32(128)):
          j_p_i_t_s_36519 = (i_36059 + i_36417)
          redout_36411 = np.float64(0.0)
          i_36412 = np.int32(0)
          one_36937 = np.int32(1)
          for counter_36936 in range(np.int32(2)):
            x_36171 = indexArray(w_mem_36641,
                                 (((i_36427 * np.int32(2)) + i_36412) * np.int32(8)),
                                 ct.c_double, np.float64)
            x_36172 = indexArray(input_mem_36617,
                                 (((j_p_i_t_s_36519 * sizze_35974) + i_36412) * np.int32(8)),
                                 ct.c_double, np.float64)
            res_36173 = (x_36171 * x_36172)
            res_36170 = (res_36173 + redout_36411)
            redout_tmp_36882 = res_36170
            redout_36411 = redout_tmp_36882
            i_36412 += one_36937
          res_36167 = redout_36411
          res_36174 = (x_36162 + res_36167)
          redout_36413 = np.float64(0.0)
          i_36414 = np.int32(0)
          one_36939 = np.int32(1)
          for counter_36938 in range(np.int32(2)):
            x_36179 = indexArray(w_mem_36645,
                                 (((i_36414 * np.int32(4)) + i_36427) * np.int32(8)),
                                 ct.c_double, np.float64)
            x_36180 = indexArray(mem_36667,
                                 (((i_36417 * np.int32(2)) + i_36414) * np.int32(8)),
                                 ct.c_double, np.float64)
            res_36181 = (x_36179 * x_36180)
            res_36178 = (res_36181 + redout_36413)
            redout_tmp_36883 = res_36178
            redout_36413 = redout_tmp_36883
            i_36414 += one_36939
          res_36175 = redout_36413
          res_36182 = (res_36174 <= np.float64(0.0))
          if res_36182:
            res_36183 = np.float64(0.0)
          else:
            res_36183 = np.float64(1.0)
          res_36184 = (res_36175 * res_36183)
          writeScalarArray(mem_36735, (i_36417 * np.int32(8)),
                           ct.c_double(res_36184))
          i_36417 += one_36941
        i_36423 = np.int32(0)
        one_36945 = np.int32(1)
        for counter_36944 in range(sizze_35974):
          redout_36419 = np.float64(0.0)
          i_36420 = np.int32(0)
          one_36943 = np.int32(1)
          for counter_36942 in range(np.int32(128)):
            x_36191 = indexArray(mem_36735, (i_36420 * np.int32(8)),
                                 ct.c_double, np.float64)
            j_p_i_t_s_36531 = (i_36059 + i_36420)
            x_36192 = indexArray(input_mem_36617,
                                 (((j_p_i_t_s_36531 * sizze_35974) + i_36423) * np.int32(8)),
                                 ct.c_double, np.float64)
            res_36193 = (x_36191 * x_36192)
            res_36190 = (res_36193 + redout_36419)
            redout_tmp_36885 = res_36190
            redout_36419 = redout_tmp_36885
            i_36420 += one_36943
          res_36187 = redout_36419
          res_36194 = (res_36187 / np.float64(128.0))
          res_36195 = (np.float64(0.1) * res_36194)
          writeScalarArray(mem_36730,
                           (((i_36427 * sizze_35974) + i_36423) * np.int32(8)),
                           ct.c_double(res_36195))
          i_36423 += one_36945
        i_36427 += one_36947
      mem_36754 = allocateMem(np.int64(32))
      mem_36757 = allocateMem(np.int64(64))
      i_36443 = np.int32(0)
      one_36957 = np.int32(1)
      for counter_36956 in range(np.int32(4)):
        x_36200 = indexArray(w_mem_36643, (i_36443 * np.int32(8)), ct.c_double,
                             np.float64)
        redout_36433 = np.float64(0.0)
        i_36434 = np.int32(0)
        one_36953 = np.int32(1)
        for counter_36952 in range(np.int32(128)):
          j_p_i_t_s_36537 = (i_36059 + i_36434)
          redout_36429 = np.float64(0.0)
          i_36430 = np.int32(0)
          one_36949 = np.int32(1)
          for counter_36948 in range(np.int32(2)):
            x_36213 = indexArray(w_mem_36641,
                                 (((i_36443 * np.int32(2)) + i_36430) * np.int32(8)),
                                 ct.c_double, np.float64)
            x_36214 = indexArray(input_mem_36617,
                                 (((j_p_i_t_s_36537 * sizze_35974) + i_36430) * np.int32(8)),
                                 ct.c_double, np.float64)
            res_36215 = (x_36213 * x_36214)
            res_36212 = (res_36215 + redout_36429)
            redout_tmp_36889 = res_36212
            redout_36429 = redout_tmp_36889
            i_36430 += one_36949
          res_36209 = redout_36429
          res_36216 = (x_36200 + res_36209)
          redout_36431 = np.float64(0.0)
          i_36432 = np.int32(0)
          one_36951 = np.int32(1)
          for counter_36950 in range(np.int32(2)):
            x_36221 = indexArray(w_mem_36645,
                                 (((i_36432 * np.int32(4)) + i_36443) * np.int32(8)),
                                 ct.c_double, np.float64)
            x_36222 = indexArray(mem_36667,
                                 (((i_36434 * np.int32(2)) + i_36432) * np.int32(8)),
                                 ct.c_double, np.float64)
            res_36223 = (x_36221 * x_36222)
            res_36220 = (res_36223 + redout_36431)
            redout_tmp_36890 = res_36220
            redout_36431 = redout_tmp_36890
            i_36432 += one_36951
          res_36217 = redout_36431
          res_36224 = (res_36216 <= np.float64(0.0))
          if res_36224:
            res_36225 = np.float64(0.0)
          else:
            res_36225 = np.float64(1.0)
          res_36226 = (res_36217 * res_36225)
          res_36206 = (res_36226 + redout_36433)
          redout_tmp_36888 = res_36206
          redout_36433 = redout_tmp_36888
          i_36434 += one_36953
        res_36203 = redout_36433
        res_36227 = (res_36203 / np.float64(128.0))
        res_36228 = (np.float64(0.1) * res_36227)
        i_36437 = np.int32(0)
        one_36955 = np.int32(1)
        for counter_36954 in range(np.int32(2)):
          x_36230 = indexArray(w_mem_36641,
                               (((i_36443 * np.int32(2)) + i_36437) * np.int32(8)),
                               ct.c_double, np.float64)
          x_36231 = indexArray(mem_36730,
                               (((i_36443 * sizze_35974) + i_36437) * np.int32(8)),
                               ct.c_double, np.float64)
          res_36232 = (x_36230 - x_36231)
          writeScalarArray(mem_36757,
                           (((i_36443 * np.int32(2)) + i_36437) * np.int32(8)),
                           ct.c_double(res_36232))
          i_36437 += one_36955
        res_36233 = (x_36200 - res_36228)
        writeScalarArray(mem_36754, (i_36443 * np.int32(8)),
                         ct.c_double(res_36233))
        i_36443 += one_36957
      loop_cond_36234 = slt32(j_36060, res_35992)
      w_mem_tmp_36854 = mem_36757
      w_mem_tmp_36855 = mem_36754
      w_mem_tmp_36856 = mem_36708
      w_mem_tmp_36857 = mem_36705
      loop_while_tmp_36858 = loop_cond_36234
      i_tmp_36863 = j_36060
      w_mem_36641 = w_mem_tmp_36854
      w_mem_36643 = w_mem_tmp_36855
      w_mem_36645 = w_mem_tmp_36856
      w_mem_36647 = w_mem_tmp_36857
      loop_while_36054 = loop_while_tmp_36858
      i_36059 = i_tmp_36863
    res_mem_36777 = w_mem_36641
    res_mem_36779 = w_mem_36643
    res_mem_36781 = w_mem_36645
    res_mem_36783 = w_mem_36647
    res_36048 = loop_while_36054
    res_36053 = i_36059
    mem_36622 = None
    mem_36629 = None
    mem_36632 = None
    mem_36639 = None
    mem_36650 = None
    mem_36667 = None
    mem_36672 = None
    mem_36675 = None
    mem_36686 = None
    mem_36730 = None
    mem_36735 = None
    j_36235 = (res_35992 + res_35996)
    empty_slice_36236 = (res_35996 == np.int32(0))
    m_36237 = (res_35996 - np.int32(1))
    i_p_m_t_s_36238 = (res_35992 + m_36237)
    zzero_leq_i_p_m_t_s_36239 = sle32(np.int32(0), i_p_m_t_s_36238)
    i_p_m_t_s_leq_w_36240 = slt32(i_p_m_t_s_36238, sizze_35973)
    i_lte_j_36241 = sle32(res_35992, j_36235)
    y_36242 = (i_lte_j_36034 and i_p_m_t_s_leq_w_36240)
    y_36243 = (zzero_leq_i_p_m_t_s_36239 and y_36242)
    y_36244 = (i_lte_j_36241 and y_36243)
    forwards_ok_36245 = (i_lte_j_36034 and y_36244)
    ok_or_empty_36246 = (empty_slice_36236 or forwards_ok_36245)
    index_certs_36247 = True
    assert ok_or_empty_36246, ("Error at nand.fut:8:1-20:23 -> nand.fut:18:32-60: %s%d%s%d%s%d%s" % ("Index [",
                                                                                                     res_35992,
                                                                                                     ":",
                                                                                                     j_36235,
                                                                                                     "] out of bounds for array of shape [",
                                                                                                     sizze_35973,
                                                                                                     "]."))
    dim_zzero_36251 = (np.int32(0) == sizze_35974)
    dim_zzero_36252 = (np.int32(0) == res_35996)
    old_empty_36253 = (dim_zzero_36251 or dim_zzero_36252)
    both_empty_36254 = (dim_zzero_36252 and old_empty_36253)
    empty_or_match_36255 = (dim_match_36040 or both_empty_36254)
    empty_or_match_cert_36256 = True
    assert empty_or_match_36255, ("Error at nand.fut:8:1-20:23 -> nand.fut:18:13-19:62 -> ../lib/github.com/HnimNart/deeplearning/neural_network.fut:114:24-54 -> ../lib/github.com/HnimNart/deeplearning/neural_network.fut:104:23-37 -> ../lib/github.com/HnimNart/deeplearning/neural_network.fut:73:46-68 -> ../lib/github.com/HnimNart/deeplearning/layers/dense.fut:47:20-50: %s" % ("function arguments of wrong shape",))
    convop_x_36785 = (np.int32(4) * res_35996)
    binop_x_36786 = sext_i32_i64(convop_x_36785)
    bytes_36784 = (np.int64(8) * binop_x_36786)
    mem_36787 = allocateMem(bytes_36784)
    i_36454 = np.int32(0)
    one_36963 = np.int32(1)
    for counter_36962 in range(np.int32(4)):
      x_36261 = indexArray(res_mem_36779, (i_36454 * np.int32(8)), ct.c_double,
                           np.float64)
      i_36450 = np.int32(0)
      one_36961 = np.int32(1)
      for counter_36960 in range(res_35996):
        j_p_i_t_s_36553 = (res_35992 + i_36450)
        redout_36446 = np.float64(0.0)
        i_36447 = np.int32(0)
        one_36959 = np.int32(1)
        for counter_36958 in range(np.int32(2)):
          x_36268 = indexArray(res_mem_36777,
                               (((i_36454 * np.int32(2)) + i_36447) * np.int32(8)),
                               ct.c_double, np.float64)
          x_36269 = indexArray(input_mem_36617,
                               (((j_p_i_t_s_36553 * sizze_35974) + i_36447) * np.int32(8)),
                               ct.c_double, np.float64)
          res_36270 = (x_36268 * x_36269)
          res_36267 = (res_36270 + redout_36446)
          redout_tmp_36894 = res_36267
          redout_36446 = redout_tmp_36894
          i_36447 += one_36959
        res_36264 = redout_36446
        res_36271 = (x_36261 + res_36264)
        res_36272 = fmax64(np.float64(0.0), res_36271)
        writeScalarArray(mem_36787,
                         (((i_36454 * res_35996) + i_36450) * np.int32(8)),
                         ct.c_double(res_36272))
        i_36450 += one_36961
      i_36454 += one_36963
    mem_36804 = allocateMem(np.int64(16))
    mem_36811 = allocateMem(np.int64(16))
    mem_36818 = allocateMem(np.int64(16))
    redout_36482 = np.int32(0)
    i_36483 = np.int32(0)
    one_36977 = np.int32(1)
    for counter_36976 in range(res_35996):
      redout_36459 = -np.inf
      i_36461 = np.int32(0)
      one_36967 = np.int32(1)
      for counter_36966 in range(np.int32(2)):
        x_wasfree_36305 = indexArray(res_mem_36783, (i_36461 * np.int32(8)),
                                     ct.c_double, np.float64)
        redout_36456 = np.float64(0.0)
        i_36457 = np.int32(0)
        one_36965 = np.int32(1)
        for counter_36964 in range(np.int32(4)):
          x_36310 = indexArray(res_mem_36781,
                               (((i_36461 * np.int32(4)) + i_36457) * np.int32(8)),
                               ct.c_double, np.float64)
          x_36311 = indexArray(mem_36787,
                               (((i_36457 * res_35996) + i_36483) * np.int32(8)),
                               ct.c_double, np.float64)
          res_36312 = (x_36310 * x_36311)
          res_36309 = (res_36312 + redout_36456)
          redout_tmp_36898 = res_36309
          redout_36456 = redout_tmp_36898
          i_36457 += one_36965
        res_36306 = redout_36456
        res_36313 = (x_wasfree_36305 + res_36306)
        res_36314 = fmax64(np.float64(0.0), res_36313)
        res_36302 = fmax64(res_36314, redout_36459)
        writeScalarArray(mem_36804, (i_36461 * np.int32(8)),
                         ct.c_double(res_36314))
        redout_tmp_36896 = res_36302
        redout_36459 = redout_tmp_36896
        i_36461 += one_36967
      res_36298 = redout_36459
      redout_36464 = np.float64(0.0)
      i_36466 = np.int32(0)
      one_36969 = np.int32(1)
      for counter_36968 in range(np.int32(2)):
        x_36320 = indexArray(mem_36804, (i_36466 * np.int32(8)), ct.c_double,
                             np.float64)
        res_36321 = (x_36320 - res_36298)
        res_36322 = futhark_exp64(res_36321)
        res_36319 = (res_36322 + redout_36464)
        writeScalarArray(mem_36811, (i_36466 * np.int32(8)),
                         ct.c_double(res_36322))
        redout_tmp_36899 = res_36319
        redout_36464 = redout_tmp_36899
        i_36466 += one_36969
      res_36315 = redout_36464
      i_36470 = np.int32(0)
      one_36971 = np.int32(1)
      for counter_36970 in range(np.int32(2)):
        x_36324 = indexArray(mem_36811, (i_36470 * np.int32(8)), ct.c_double,
                             np.float64)
        res_36325 = (x_36324 / res_36315)
        writeScalarArray(mem_36818, (i_36470 * np.int32(8)),
                         ct.c_double(res_36325))
        i_36470 += one_36971
      j_p_i_t_s_36581 = (res_35992 + i_36483)
      redout_36472 = np.int32(0)
      i_36473 = np.int32(0)
      one_36973 = np.int32(1)
      for counter_36972 in range(sizze_35976):
        arg_36329 = indexArray(labels_mem_36619,
                               (((j_p_i_t_s_36581 * sizze_35976) + redout_36472) * np.int32(8)),
                               ct.c_double, np.float64)
        arg_36330 = indexArray(labels_mem_36619,
                               (((j_p_i_t_s_36581 * sizze_35976) + i_36473) * np.int32(8)),
                               ct.c_double, np.float64)
        res_36331 = (arg_36330 < arg_36329)
        if res_36331:
          res_36332 = redout_36472
        else:
          res_36332 = i_36473
        redout_tmp_36902 = res_36332
        redout_36472 = redout_tmp_36902
        i_36473 += one_36973
      res_36326 = redout_36472
      redout_36476 = np.int32(0)
      i_36479 = np.int32(0)
      one_36975 = np.int32(1)
      for counter_36974 in range(np.int32(2)):
        arg_36339 = indexArray(mem_36818, (redout_36476 * np.int32(8)),
                               ct.c_double, np.float64)
        arg_36340 = indexArray(mem_36818, (i_36479 * np.int32(8)), ct.c_double,
                               np.float64)
        res_36341 = (arg_36340 < arg_36339)
        if res_36341:
          res_36342 = redout_36476
        else:
          res_36342 = i_36479
        redout_tmp_36903 = res_36342
        redout_36476 = redout_tmp_36903
        i_36479 += one_36975
      res_36334 = redout_36476
      res_36350 = (res_36326 == res_36334)
      res_36351 = btoi_bool_i32(res_36350)
      res_36293 = (res_36351 + redout_36482)
      redout_tmp_36895 = res_36293
      redout_36482 = redout_tmp_36895
      i_36483 += one_36977
    res_36290 = redout_36482
    mem_36787 = None
    mem_36804 = None
    mem_36811 = None
    mem_36818 = None
    res_36352 = sitofp_i32_f64(res_36290)
    res_36353 = sitofp_i32_f64(res_35996)
    res_36354 = (res_36352 / res_36353)
    out_arrsizze_36838 = np.int32(4)
    out_arrsizze_36839 = np.int32(2)
    out_arrsizze_36842 = np.int32(4)
    out_arrsizze_36845 = np.int32(2)
    out_arrsizze_36846 = np.int32(4)
    out_arrsizze_36849 = np.int32(2)
    out_memsizze_36837 = np.int64(64)
    out_mem_36836 = res_mem_36777
    out_memsizze_36841 = np.int64(32)
    out_mem_36840 = res_mem_36779
    out_memsizze_36844 = np.int64(64)
    out_mem_36843 = res_mem_36781
    out_memsizze_36848 = np.int64(16)
    out_mem_36847 = res_mem_36783
    scalar_out_36835 = res_36354
    return (scalar_out_36835, out_memsizze_36837, out_mem_36836,
            out_arrsizze_36838, out_arrsizze_36839, out_memsizze_36841,
            out_mem_36840, out_arrsizze_36842, out_memsizze_36844,
            out_mem_36843, out_arrsizze_36845, out_arrsizze_36846,
            out_memsizze_36848, out_mem_36847, out_arrsizze_36849)
  def main(self, input_mem_36617_ext, labels_mem_36619_ext):
    if not(((type(input_mem_36617_ext) in [np.ndarray]) and (input_mem_36617_ext.dtype == np.float64))):
      raise TypeError("Argument #0 has invalid value\nFuthark type: {}\nArgument has Python type {} and value: {}\n".format("[][]f64",
                                                                                                                            type(input_mem_36617_ext),
                                                                                                                            input_mem_36617_ext))
    sizze_35973 = np.int32(input_mem_36617_ext.shape[0])
    sizze_35974 = np.int32(input_mem_36617_ext.shape[1])
    input_mem_sizze_36616 = np.int32(input_mem_36617_ext.nbytes)
    input_mem_36617 = unwrapArray(input_mem_36617_ext)
    if not(((type(labels_mem_36619_ext) in [np.ndarray]) and (labels_mem_36619_ext.dtype == np.float64))):
      raise TypeError("Argument #1 has invalid value\nFuthark type: {}\nArgument has Python type {} and value: {}\n".format("[][]f64",
                                                                                                                            type(labels_mem_36619_ext),
                                                                                                                            labels_mem_36619_ext))
    sizze_35975 = np.int32(labels_mem_36619_ext.shape[0])
    sizze_35976 = np.int32(labels_mem_36619_ext.shape[1])
    labels_mem_sizze_36618 = np.int32(labels_mem_36619_ext.nbytes)
    labels_mem_36619 = unwrapArray(labels_mem_36619_ext)
    (scalar_out_36835, out_memsizze_36837, out_mem_36836, out_arrsizze_36838,
     out_arrsizze_36839, out_memsizze_36841, out_mem_36840, out_arrsizze_36842,
     out_memsizze_36844, out_mem_36843, out_arrsizze_36845, out_arrsizze_36846,
     out_memsizze_36848, out_mem_36847,
     out_arrsizze_36849) = self.futhark_main(input_mem_sizze_36616,
                                             input_mem_36617,
                                             labels_mem_sizze_36618,
                                             labels_mem_36619, sizze_35973,
                                             sizze_35974, sizze_35975,
                                             sizze_35976)
    return (np.float64(scalar_out_36835),
            opaque("(([][]f64, []f64), ([][]f64, []f64))",
                   createArray(ct.cast(out_mem_36836, ct.POINTER(ct.c_double)),
                               (out_arrsizze_36838, out_arrsizze_36839)),
                   createArray(ct.cast(out_mem_36840, ct.POINTER(ct.c_double)),
                               (out_arrsizze_36842,)),
                   createArray(ct.cast(out_mem_36843, ct.POINTER(ct.c_double)),
                               (out_arrsizze_36845, out_arrsizze_36846)),
                   createArray(ct.cast(out_mem_36847, ct.POINTER(ct.c_double)),
                               (out_arrsizze_36849,))))
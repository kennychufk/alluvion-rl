import numpy as np
import struct


def read_file_int(filename):
    with open(filename) as f:
        return int(f.read())


def read_file_float(filename):
    with open(filename) as f:
        return float(f.read())


def read_alu(filename):
    with open(filename, 'rb') as f:
        shape = []
        while (True):
            shape_item = np.frombuffer(f.read(np.dtype(np.uint32).itemsize),
                                       dtype=np.uint32)[0]
            if (shape_item == 0):
                break
            shape.append(shape_item)
        num_primitives_per_unit = np.frombuffer(f.read(
            np.dtype(np.uint32).itemsize),
                                                dtype=np.uint32)[0]
        numeric_type_label = f.read(1).decode('ascii')
        if (numeric_type_label == 'f'):
            nominal_dtype = np.float32
        elif (numeric_type_label == 'd'):
            nominal_dtype = np.float64
        elif (numeric_type_label == 'u'):
            nominal_dtype = np.uint32
        elif (numeric_type_label == 'i'):
            nominal_dtype = np.int32
        np_dtype = np.dtype(nominal_dtype)
        num_bytes = np.prod(
            shape,
            dtype=np.uint32) * num_primitives_per_unit * np_dtype.itemsize
        return np.frombuffer(f.read(num_bytes),
                             dtype=np_dtype).reshape(*shape,
                                                     num_primitives_per_unit)

import numpy as np
import struct
from pathlib import Path


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


def read_pile(filename):
    with open(filename, 'rb') as f:
        f.seek(0, 2)  #Jumps to the end
        end_of_file = f.tell(
        )  #Give you the end location (characters from start)
        f.seek(0)  #Jump to the beginning of the file again
        xs = []
        vs = []
        qs = []
        omegas = []
        while (True):
            x = np.frombuffer(f.read(np.dtype(np.float32).itemsize * 3),
                              dtype=np.float32)
            v = np.frombuffer(f.read(np.dtype(np.float32).itemsize * 3),
                              dtype=np.float32)
            q = np.frombuffer(f.read(np.dtype(np.float32).itemsize * 4),
                              dtype=np.float32)
            omega = np.frombuffer(f.read(np.dtype(np.float32).itemsize * 3),
                                  dtype=np.float32)

            xs.append(x)
            vs.append(v)
            qs.append(q)
            omegas.append(omega)
            if f.tell() == end_of_file:
                break
    return np.array(xs), np.array(vs), np.array(qs), np.array(omegas)


def get_agitator_offset(trajectory_filename, agitator_option):
    trajectory_path = Path(trajectory_filename)
    offset_dir = trajectory_path.parent.joinpath("offsets")
    agitator_postfix = agitator_option.replace('/', '-')
    offset_path = offset_dir.joinpath(
        f'{trajectory_path.stem}.{agitator_postfix}.npy')
    return np.load(str(offset_path))

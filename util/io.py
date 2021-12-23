def read_file_int(filename):
    with open(filename) as f:
        return int(f.read())


def read_file_float(filename):
    with open(filename) as f:
        return float(f.read())

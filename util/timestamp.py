import xxhash
import time


def get_timestamp_and_hash():
    hasher = xxhash.xxh32()
    hasher.reset()
    timestamp = time.time()
    timestamp_str = time.strftime('%m%d.%H.%M.%S', time.localtime(timestamp))
    hasher.update(bytearray("{}".format(timestamp), 'utf8'))
    timestamp_hash = hasher.hexdigest()
    return timestamp_str, timestamp_hash

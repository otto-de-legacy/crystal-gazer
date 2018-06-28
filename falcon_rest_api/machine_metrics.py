import os
import sys

import psutil

def memory_usage_psutil():
    """STILL FAST, return the memory usage of a Python module in MB (>~ 10000 per second)"""
    process = psutil.Process(os.getpid())
    mem = process.memory_info()[0] / float(2 ** 20)
    return mem

def memory_usage_resource():
    """FASTEST disatvantage: no idea of output unit and whats measured (>~ 10000 per second)"""
    import resource
    rusage_denom = 1024.
    if sys.platform == 'darwin':
        # ... it seems that in OSX the output is different units ...
        rusage_denom = rusage_denom * rusage_denom
    mem = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / rusage_denom
    return mem

def memory_usage_ps():
    """SLOWEST (ca 100 per second)"""
    import subprocess
    out = subprocess.Popen(['ps', 'v', '-p', str(os.getpid())],
                           stdout=subprocess.PIPE).communicate()[0].split(b'\n')
    vsz_index = out[0].split().index(b'RSS')
    mem = float(out[1].split()[vsz_index]) / 1024
    return mem

def cpu_usage_percent():
    """ """
    cpu_perc = psutil.cpu_percent()
    return cpu_perc
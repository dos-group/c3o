#!/usr/bin/python

import sys
from ClusterConfiguration import Configure

if __name__ == "__main__":

    if sys.version_info < (3,6,0):
        print("Python version >= 3.6 required!"); exit(1)

    usage = "\nUsage:\nc3o <job_name> <max_runtime (seconds)> [*context_args]"

    args = sys.argv[1:]
    if len(args) < 2: print(usage); exit(1)

    for i, arg in enumerate(args):
        try: args[i] = int(arg)
        except:
            try: args[i] = float(arg)
            except: pass

    Configure.get_configuration(*args)


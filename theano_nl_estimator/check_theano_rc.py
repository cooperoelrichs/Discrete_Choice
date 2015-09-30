import os

file = os.path.expanduser('~') + '/.theanorc'

theanorc_with_gpu = (
    "[global]\n"
    "device = gpu\n"
    "floatX = float32\n"
    "numpy.seterr_all = raise\n"
    "assert_no_cpu_op = warn\n"
    "\n"
    "[cuda]\n"
    "root = /Developer/NVIDIA/CUDA-7.5"
)

theanorc_with_cpu = (
    "[global]\n"
    "device = cpu\n"
    "floatX = float32\n"
    "numpy.seterr_all = raise"
)


def check_file(theanorc_str):
    with open(file, "r") as f:
        if f.read() == theanorc_str:
            print('theanorc file did match string, not changing anything')
            return True
        else:
            print('theanorc file did NOT match string')
            return False

def replace_file_with(theanorc_str):
    print('replacing file: ' + file)

    if os.path.exists(file):
        with open(file, "r") as f:
            print('theanorc file was:\n==============')
            print(f.read())
            print('==============')
        os.remove(file)

    with open(file, "w+") as f:
        f.write(theanorc_str)

def check_and_maybe_replace_file(theanorc_str):
    if os.path.exists(file) and check_file(theanorc_str):
        pass
    else:
        replace_file_with(theanorc_str)

def use_gpu():
    print('Run Theano on the GPU')
    check_and_maybe_replace_file(theanorc_with_gpu)

def use_cpu():
    print('Run Theano on the CPU')
    check_and_maybe_replace_file(theanorc_with_cpu)

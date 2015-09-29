import os

file = os.path.expanduser('~') + '/.theanorc'

theanorc_string = (
    "[global]\n"
    "device = gpu\n"
    "floatX = float32\n"
    "config.numpy.seterr_all = raise\n"
    "\n"
    "[cuda]\n"
    "root = /Developer/NVIDIA/CUDA-7.5\n"
)


def check_file():
    with open(file, "r") as f:
        if f.read() == theanorc_string:
            print('theanorc file is standard')
            return True
        else:
            print('theanorc file is NOT standard')
            return False

def replace_file_with_standard():
    print('replacing file: ' + file)

    if os.path.exists(file):
        with open(file, "r") as f:
            print('theanorc file was:\n==============')
            print(f.read())
            print('==============')
        os.remove(file)

    with open(file, "w+") as f:
        f.write(theanorc_string)

def check_and_maybe_replace_file():
    if os.path.exists(file) and check_file():
        pass
    else:
        replace_file_with_standard()

check_and_maybe_replace_file()

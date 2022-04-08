from pyinstrument import Profiler
import numpy as np
import sys


def main():

    profiler = Profiler()
    profiler.start()

    # Function 1
    print('Function 1')
    func1()
    # Function 2
    print('Function 2')
    func2()
    # Function 3
    print('Function 3')
    func3()

    profiler.stop()

    profiler.print()


def func1():

    a = 0
    for ii in range(10000):
        a += 1


def func2():

    a = 3240
    for ii in range(100):
        a = np.sqrt(a)


def func3():

    a = np.random.random((100, 100))
    for ii in range(10):
        a = a @ a




# ------------------------------------ RUN SCRIPT ------------------------------------
if __name__ == '__main__':
    sys.exit(main())
# ------------------------------------------------------------------------------------
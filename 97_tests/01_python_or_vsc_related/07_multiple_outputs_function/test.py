# Purpose of script:
# Test the multiple outputs format of a function.
#
# (c) Paul Didier, SOUNDS ETN, KU Leuven ESAT STADIUS

import sys

def main():
    """Main function (called by default when running script)."""
    a, b, c = my_function()
    d = my_function()
    print(a, b, c)
    print(d)

def my_function():
    """Function with multiple outputs."""
    return 1, '2', [3]

if __name__ == '__main__':
    sys.exit(main())
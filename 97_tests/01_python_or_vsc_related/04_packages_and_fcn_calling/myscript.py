import sys
from mypackage import functions


def main():

    # Calling a function from a package directly
    a = functions.a_function()
    # Calling a function from a package that calls another
    b = functions.other_functions.another_function()

    print(a)
    print(b)

    return None


# ------------------------------------ RUN SCRIPT ------------------------------------
if __name__ == '__main__':
    sys.exit(main())
# ------------------------------------------------------------------------------------
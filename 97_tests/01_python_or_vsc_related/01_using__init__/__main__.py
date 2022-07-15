# from package import file_1, file_2 # This imports the files because package/ includes an __init__.py file
# from package2 import file_1, file_2 # This does not import because package2/ does not includes an __init__.py file
from package import * # This imports the files because package/ includes an __init__.py file

script1.file_1() # This is my file 1!
script2.file_2() # And this is file 2!


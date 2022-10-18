
# ----- 18/10/2022 [creation date]
#
# Question to answer:
# - Can I work using DANSE (from the <danse> repository) in a different repository (say, a branch of <sounds-phd>)? 
#
# Dummy rep test:
# - Create another repository <test>
# - Fill it up with a couple of packages (folders with an __init__.py file + some other .py files)
# - Understand how to work with versions (v==1.x.x)

# Potential solution: https://git-scm.com/book/en/v2/Git-Tools-Submodules

# Testing SUBMODULES..

import sys
from pathlib import Path, PurePath
# Find path to root folder
rootFolder = 'sounds-phd'
pathToRoot = Path(__file__)
while PurePath(pathToRoot).name != rootFolder:
    pathToRoot = pathToRoot.parent
sys.path.append(f'{pathToRoot}')
print(sys.path)

from danse.utils.hello import hello

hello()

stop = 1
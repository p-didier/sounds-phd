# Purpose of script:
# Test the class methods in the `class_methods` folder.
#
# (c) Paul Didier, SOUNDS ETN, KU Leuven ESAT STADIUS

import sys
import numpy as np
from pathlib import Path
import dataclass_methods as dcm
from dataclasses import dataclass, field

# FUNCTION_TO_TEST = 'save'
# FUNCTION_TO_TEST = 'dump_to_yaml_template'
FUNCTION_TO_TEST = 'load_from_yaml'
EXPORT_PATH_YAML = f'{Path(__file__).parent}/testing_out/test.yaml'

def main():
    """Main function (called by default when running script)."""
    # Check if function to test exists in `dataclass_methods.py`
    if not hasattr(dcm, FUNCTION_TO_TEST):
        raise ValueError(f'Function "{FUNCTION_TO_TEST}" not found in "dataclass_methods.py".')
    
    # Run test
    if FUNCTION_TO_TEST == 'save':
        # test_save()  # Test the `save` function TODO: implement
        pass
    elif FUNCTION_TO_TEST == 'dump_to_yaml_template':
        test_dump_to_yaml_template(path=EXPORT_PATH_YAML)
    elif FUNCTION_TO_TEST == 'load_from_yaml':
        out = test_load_from_yaml(path=EXPORT_PATH_YAML)
        stop = 1
    else:
        raise ValueError(f'Function "{FUNCTION_TO_TEST}" not implemented in unit test script.')


# Create sub-dataclass
@dataclass
class TestSubClass:
    a: int = 1
    b: float = 2.0
    c: str = 'foo'
    d: list = field(default_factory=list)
    # e: dict = field(default_factory=dict)
    f: np.ndarray = np.array([[1, 2, 3], [4, 5, 6]])
    
    def __post_init__(self):
        if self.d is None:
            self.d = []
        # if self.e is None:
        #     self.e = {}
# Create dataclass
@dataclass
class TestClass:
    a: int = 1
    b: float = 2.0
    c: str = 'foo'
    d: list = field(default_factory=list)
    # e: dict = field(default_factory=dict)
    f: np.ndarray = np.array([[1, 2, 3], [4, 5, 6]])
    g: TestSubClass = TestSubClass()

    def __post_init__(self):
        if self.d is None:
            self.d = []
        # if self.e is None:
        #     self.e = {}

# Write unit test for load_from_yaml
def test_load_from_yaml(path='test.yaml'):
    """
    Unit test for the `load_from_yaml` function.
    """
    return dcm.load_from_yaml(path=path, myDataclass=TestClass())


def test_dump_to_yaml_template(path='test.yaml'):
    """
    Unit test for the `dump_to_yaml_template` function.
    """
    # Create instance of dataclass
    mycls = TestClass(
        a=3,
        b=4.0,
        c='bar',
        d=[1, 2, 3],
        # e={'foo': 1, 'bar': 2, 'baz': 3},
        f=np.array([[1, 2, 3], [4, 5, 6]]),
        g=TestSubClass(
            a=3,
            b=4.0,
            c='bar',
            d=[1, 2, 3],
            # e={'foo': 1, 'bar': 2, 'baz': 3},
            f=np.array([[1, 2, 3], [4, 5, 6]])
        )
    )

    # Dump to YAML template
    dcm.dump_to_yaml_template(mycls, path=path)

if __name__ == '__main__':
    sys.exit(main())
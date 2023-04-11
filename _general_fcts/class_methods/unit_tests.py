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
FUNCTION_TO_TEST = 'dataclasses_equal'
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
    elif FUNCTION_TO_TEST == 'dataclasses_equal':
        test_dataclasses_equal()
    else:
        raise ValueError(f'Function "{FUNCTION_TO_TEST}" not implemented in unit test script.')


# Create sub-dataclass
@dataclass
class TestSubClass:
    aB: int = 1
    Ab: float = 2.0
    c_e: str = 'foo'
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
            aB=3,
            Ab=4.0,
            c_e='bar',
            d=[1, 2, 3],
            # e={'foo': 1, 'bar': 2, 'baz': 3},
            f=np.array([[1, 2, 3], [4, 5, 6]])
        )
    )

    # Dump to YAML template
    dcm.dump_to_yaml_template(mycls, path=path)


# Unit test for the `dataclasses_equal` function
def test_dataclasses_equal():
    d1 = TestClass()
    d2 = TestClass()
    d3 = TestClass(a=3)
    d4 = TestClass(a=3, b=4.0)
    d5 = TestClass(a=3, b=4.0, c='bar')
    d6 = TestClass(a=3, b=4.0, c='bar', d=[1, 2, 3])
    d7 = TestClass(a=3, b=4.0, c='bar', d=[1, 2, 3], f=np.array([[1, 2, 3, 4], [4, 5, 6, 7]]))
    d8 = TestClass(a=3, b=4.0, c='bar', d=[1, 2, 3], f=np.array([[1, 2, 3, 4], [4, 5, 6, 7]]), g=TestSubClass(aB=3))

    assert dcm.dataclasses_equal(d1, d2)
    assert not dcm.dataclasses_equal(d1, d3)
    assert not dcm.dataclasses_equal(d1, d4)
    assert not dcm.dataclasses_equal(d1, d5)
    assert not dcm.dataclasses_equal(d1, d6)
    assert not dcm.dataclasses_equal(d1, d7)
    assert not dcm.dataclasses_equal(d1, d8)

    assert not dcm.dataclasses_equal(d3, d4)
    assert not dcm.dataclasses_equal(d3, d5)
    assert not dcm.dataclasses_equal(d3, d6)
    assert not dcm.dataclasses_equal(d3, d7)
    assert not dcm.dataclasses_equal(d3, d8)

    assert not dcm.dataclasses_equal(d4, d5)
    assert not dcm.dataclasses_equal(d4, d6)
    assert not dcm.dataclasses_equal(d4, d7)
    assert not dcm.dataclasses_equal(d4, d8)

    assert not dcm.dataclasses_equal(d5, d6)
    assert not dcm.dataclasses_equal(d5, d7)
    assert not dcm.dataclasses_equal(d5, d8)

    assert not dcm.dataclasses_equal(d6, d7)
    assert not dcm.dataclasses_equal(d6, d8)

    assert not dcm.dataclasses_equal(d7, d8)

    print('All tests passed.')


if __name__ == '__main__':
    sys.exit(main())
"""Quick script to ensure correct matrices dimensions in DANSE
algorithm implementation"""
#%% IMPORT EXPORTS USING PICKLE

from dataclasses import dataclass
import numpy as np
import pickle, gzip

sigDuration = 10
samplingFreq = 16e3
numSensors = 10
numSamples = int(sigDuration * samplingFreq)
seed = 12345
rng = np.random.default_rng(seed)

# signal matrix at node k
yk = rng.random((numSamples, numSensors))

@dataclass
class TestClass:
    mydata: np.ndarray

@dataclass
class TestClass2:
    mydata2: TestClass

myObject = TestClass2(mydata2=TestClass(mydata=yk))

pickle.dump(myObject, gzip.open(f'.testExportPickle.pkl.gz', 'wb'))

reloaded = pickle.load(gzip.open(f'.testExportPickle.pkl.gz', 'r'))


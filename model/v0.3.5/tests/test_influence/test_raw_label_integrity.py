import math
import numpy as np

def test_per_time_rms_reconstructs_normalized_label():
    diff=np.array([.1,.2,.3]);base=np.array([1.,2.,3.])
    label=float(np.sqrt(np.mean(diff**2))/max(float(np.sqrt(np.mean(base**2))),1e-12))
    assert math.isclose(label,float(np.sqrt(np.mean(diff**2))/np.sqrt(np.mean(base**2))),rel_tol=1e-14)

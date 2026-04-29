import numpy as np

from spikingjelly.activation_based import functional, neuron

# SpikingJelly's auto-CUDA path still references deprecated NumPy aliases such
# as `np.int` on some releases. Restore the alias when running against NumPy 2.
if not hasattr(np, "int"):
    np.int = int

try:
    import cupy  # noqa: F401
except Exception:
    _CUPY_AVAILABLE = False
else:
    _CUPY_AVAILABLE = True


SPIKING_NEURON_TYPES = (neuron.IFNode, neuron.LIFNode)


def default_backend_for_steps(T: int) -> str:
    if T > 0 and _CUPY_AVAILABLE:
        return "cupy"
    return "torch"


def set_default_backend(net, T: int) -> str:
    backend = default_backend_for_steps(T)
    functional.set_backend(net, backend, instance=SPIKING_NEURON_TYPES)
    return backend

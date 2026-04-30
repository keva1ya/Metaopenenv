# audioop.py – stub for the removed stdlib module (Python 3.13)
# Only the functions referenced by pydub are defined. They raise
# NotImplementedError to make failures explicit if audio handling
# is ever invoked. The current application does not use audio features.

def lin2lin(fragment, width, new_width):
    raise NotImplementedError(
        "audioop.lin2lin is not available in this stub. "
        "Audio processing is not required for the current app."
    )

def max(fragment, width):
    raise NotImplementedError("audioop.max is not available in this stub.")

def rms(fragment, width):
    raise NotImplementedError("audioop.rms is not available in this stub.")

def avg(fragment, width):
    raise NotImplementedError("audioop.avg is not available in this stub.")

def mul(fragment, width, factor):
    raise NotImplementedError("audioop.mul is not available in this stub.")

def add(fragment1, fragment2, width):
    raise NotImplementedError("audioop.add is not available in this stub.")

def bias(fragment, width, bias):
    raise NotImplementedError("audioop.bias is not available in this stub.")

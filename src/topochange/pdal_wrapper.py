"""
PDAL Wrapper for Google Colab Compatibility.

This module provides a drop-in replacement for the pdal module that works
in Google Colab where the native Python 3.12 kernel cannot use conda-installed
PDAL bindings (which are built for Python 3.11).

Usage:
    # Instead of: import pdal
    from .pdal_wrapper import pdal

    # Then use normally:
    pipeline = pdal.Pipeline(json.dumps({...}))
    pipeline.execute()

The wrapper automatically detects the environment:
- In Colab with condacolab: Uses subprocess to call Python 3.11
- Locally or when native pdal works: Uses native pdal module
"""

import subprocess
import json
import os
import sys
import tempfile
import numpy as np
from typing import Optional, List, Dict, Any

# Detect environment
IN_COLAB = 'google.colab' in sys.modules
CONDA_PYTHON = '/usr/local/bin/python'
PROJ_LIB = '/usr/local/share/proj/'
CONDA_LIB = '/usr/local/lib'


def _get_conda_env() -> Dict[str, str]:
    """Get environment variables needed for conda PDAL to work."""
    env = {**os.environ}
    env['PROJ_LIB'] = PROJ_LIB
    # Prepend conda lib path to ensure conda's SQLite/GDAL are used
    existing_ld_path = os.environ.get('LD_LIBRARY_PATH', '')
    env['LD_LIBRARY_PATH'] = f"{CONDA_LIB}:{existing_ld_path}" if existing_ld_path else CONDA_LIB
    return env

# Try to import native pdal first
_NATIVE_PDAL_AVAILABLE = False
_native_pdal = None

try:
    import pdal as _native_pdal
    _NATIVE_PDAL_AVAILABLE = True
except ImportError:
    pass


def _check_conda_pdal_available() -> bool:
    """Check if PDAL is available via conda's Python."""
    if not os.path.exists(CONDA_PYTHON):
        return False
    try:
        result = subprocess.run(
            [CONDA_PYTHON, '-c', 'import pdal; print(pdal.__version__)'],
            capture_output=True, text=True, timeout=10,
            env=_get_conda_env()
        )
        return result.returncode == 0
    except Exception:
        return False


_CONDA_PDAL_AVAILABLE = _check_conda_pdal_available() if IN_COLAB else False


class PipelineWrapper:
    """
    Wrapper around pdal.Pipeline that uses subprocess when needed.

    Mimics the pdal.Pipeline API for compatibility with existing code.
    """

    def __init__(self, pipeline_json: str, arrays: Optional[List[np.ndarray]] = None):
        """
        Initialize pipeline with JSON string.

        Args:
            pipeline_json: JSON string defining the PDAL pipeline
            arrays: Optional list of numpy structured arrays to use as input
                   (for pipelines that start with a writer instead of reader)
        """
        self.pipeline_json = pipeline_json
        self._input_arrays = arrays  # Arrays to pass TO the pipeline
        self._count = 0
        self._arrays: List[np.ndarray] = []  # Arrays returned FROM the pipeline
        self._metadata: Dict[str, Any] = {}
        self._log = ""

    def execute(self) -> int:
        """
        Execute the pipeline.

        Returns:
            Number of points processed
        """
        if _NATIVE_PDAL_AVAILABLE:
            return self._execute_native()
        elif _CONDA_PDAL_AVAILABLE:
            return self._execute_subprocess()
        else:
            raise RuntimeError(
                "PDAL is not available. In Colab, run the condacolab setup first. "
                "Locally, install pdal via: pip install pdal"
            )

    def execute_streaming(self, chunk_size: int = 1000000) -> int:
        """
        Execute the pipeline in streaming mode.

        Args:
            chunk_size: Number of points to process at a time

        Returns:
            Number of points processed
        """
        if _NATIVE_PDAL_AVAILABLE:
            return self._execute_streaming_native(chunk_size)
        elif _CONDA_PDAL_AVAILABLE:
            return self._execute_streaming_subprocess(chunk_size)
        else:
            raise RuntimeError("PDAL is not available.")

    def _execute_native(self) -> int:
        """Execute using native pdal module."""
        if self._input_arrays is not None:
            pipeline = _native_pdal.Pipeline(self.pipeline_json, arrays=self._input_arrays)
        else:
            pipeline = _native_pdal.Pipeline(self.pipeline_json)
        self._count = pipeline.execute()
        self._arrays = list(pipeline.arrays)
        self._metadata = pipeline.metadata
        self._log = getattr(pipeline, 'log', '')
        return self._count

    def _execute_streaming_native(self, chunk_size: int) -> int:
        """Execute streaming using native pdal module."""
        if self._input_arrays is not None:
            pipeline = _native_pdal.Pipeline(self.pipeline_json, arrays=self._input_arrays)
        else:
            pipeline = _native_pdal.Pipeline(self.pipeline_json)
        self._count = pipeline.execute_streaming(chunk_size=chunk_size)
        self._arrays = list(pipeline.arrays) if hasattr(pipeline, 'arrays') else []
        self._metadata = pipeline.metadata if hasattr(pipeline, 'metadata') else {}
        return self._count

    def _execute_subprocess(self) -> int:
        """Execute using subprocess with conda's Python."""
        # If we have input arrays, we need to serialize them and pass to subprocess
        if self._input_arrays is not None:
            return self._execute_subprocess_with_arrays()

        # Create a temporary script that runs the pipeline and outputs results
        script = f'''
import pdal
import json
import numpy as np
import sys

pipeline_json = {repr(self.pipeline_json)}
pipeline = pdal.Pipeline(pipeline_json)
count = pipeline.execute()

# Get arrays and convert to lists for JSON serialization
arrays_data = []
for arr in pipeline.arrays:
    # Convert structured array to dict of lists
    arr_dict = {{name: arr[name].tolist() for name in arr.dtype.names}}
    arrays_data.append(arr_dict)

# Get metadata
metadata = pipeline.metadata

result = {{
    "count": count,
    "arrays": arrays_data,
    "metadata": metadata,
    "log": getattr(pipeline, 'log', '')
}}
print(json.dumps(result))
'''

        result = subprocess.run(
            [CONDA_PYTHON, '-c', script],
            capture_output=True, text=True, env=_get_conda_env()
        )

        if result.returncode != 0:
            raise RuntimeError(f"PDAL pipeline failed: {result.stderr}")

        try:
            data = json.loads(result.stdout)
        except json.JSONDecodeError as e:
            raise RuntimeError(f"Failed to parse PDAL output: {e}\nOutput: {result.stdout}\nStderr: {result.stderr}")

        self._count = data['count']
        self._metadata = data['metadata']
        self._log = data.get('log', '')

        # Reconstruct numpy structured arrays
        self._arrays = []
        for arr_dict in data['arrays']:
            if arr_dict:
                # Determine dtype from the data
                dtype = [(name, np.array(values).dtype) for name, values in arr_dict.items()]
                n_points = len(next(iter(arr_dict.values())))
                arr = np.empty(n_points, dtype=dtype)
                for name, values in arr_dict.items():
                    arr[name] = values
                self._arrays.append(arr)

        return self._count

    def _execute_subprocess_with_arrays(self) -> int:
        """Execute using subprocess with input arrays passed via temp file."""
        # Save input arrays to a temporary numpy file to pass to subprocess
        with tempfile.NamedTemporaryFile(suffix='.npz', delete=False) as tmp:
            arrays_file = tmp.name

        try:
            # Save each input array with metadata about its dtype
            save_dict = {}
            for i, arr in enumerate(self._input_arrays):
                # Save the array data
                for name in arr.dtype.names:
                    save_dict[f'arr{i}_{name}'] = arr[name]
                # Save dtype info as string
                save_dict[f'arr{i}_dtype'] = str(arr.dtype)

            save_dict['num_arrays'] = np.array([len(self._input_arrays)])
            np.savez(arrays_file, **save_dict)

            # Create script that loads arrays and runs pipeline
            script = f'''
import pdal
import json
import numpy as np

# Load input arrays from temp file
data = np.load({repr(arrays_file)}, allow_pickle=True)
num_arrays = int(data['num_arrays'][0])

input_arrays = []
for i in range(num_arrays):
    # Reconstruct structured array
    dtype_str = str(data[f'arr{{i}}_dtype'])
    # Parse dtype string to get field names
    import re
    fields = re.findall(r"\\('(\\w+)'", dtype_str)

    # Get first field to determine length
    first_field = data[f'arr{{i}}_{{fields[0]}}']
    n_points = len(first_field)

    # Build dtype from actual data
    dtype_list = []
    for name in fields:
        arr_data = data[f'arr{{i}}_{{name}}']
        dtype_list.append((name, arr_data.dtype))

    arr = np.empty(n_points, dtype=dtype_list)
    for name in fields:
        arr[name] = data[f'arr{{i}}_{{name}}']
    input_arrays.append(arr)

pipeline_json = {repr(self.pipeline_json)}
pipeline = pdal.Pipeline(pipeline_json, arrays=input_arrays)
count = pipeline.execute()

# Get output arrays (usually empty for writers)
arrays_data = []
for arr in pipeline.arrays:
    arr_dict = {{name: arr[name].tolist() for name in arr.dtype.names}}
    arrays_data.append(arr_dict)

result = {{
    "count": count,
    "arrays": arrays_data,
    "metadata": pipeline.metadata,
    "log": getattr(pipeline, 'log', '')
}}
print(json.dumps(result))
'''

            result = subprocess.run(
                [CONDA_PYTHON, '-c', script],
                capture_output=True, text=True, env=_get_conda_env()
            )

            if result.returncode != 0:
                raise RuntimeError(f"PDAL pipeline failed: {result.stderr}")

            try:
                data = json.loads(result.stdout)
            except json.JSONDecodeError as e:
                raise RuntimeError(f"Failed to parse PDAL output: {e}\nOutput: {result.stdout}\nStderr: {result.stderr}")

            self._count = data['count']
            self._metadata = data['metadata']
            self._log = data.get('log', '')
            self._arrays = []

            return self._count

        finally:
            # Clean up temp file
            if os.path.exists(arrays_file):
                os.remove(arrays_file)

    def _execute_streaming_subprocess(self, chunk_size: int) -> int:
        """Execute streaming using subprocess with conda's Python."""
        script = f'''
import pdal
import json

pipeline_json = {repr(self.pipeline_json)}
pipeline = pdal.Pipeline(pipeline_json)
count = pipeline.execute_streaming(chunk_size={chunk_size})

result = {{"count": count}}
print(json.dumps(result))
'''

        result = subprocess.run(
            [CONDA_PYTHON, '-c', script],
            capture_output=True, text=True, env=_get_conda_env()
        )

        if result.returncode != 0:
            raise RuntimeError(f"PDAL streaming pipeline failed: {result.stderr}")

        data = json.loads(result.stdout)
        self._count = data['count']
        self._arrays = []
        self._metadata = {}

        return self._count

    def execute_metadata_only(self) -> int:
        """
        Execute pipeline and return only metadata, skipping array serialization.

        This is much more memory-efficient for pipelines where you only need
        metadata (e.g., filters.stats, filters.info) and not the point arrays.

        Returns:
            Number of points processed
        """
        if _NATIVE_PDAL_AVAILABLE:
            return self._execute_native()  # Native is efficient enough
        elif _CONDA_PDAL_AVAILABLE:
            return self._execute_metadata_only_subprocess()
        else:
            raise RuntimeError("PDAL is not available.")

    def _execute_metadata_only_subprocess(self) -> int:
        """Execute using subprocess, returning only metadata (no array serialization)."""
        script = f'''
import pdal
import json

pipeline_json = {repr(self.pipeline_json)}
pipeline = pdal.Pipeline(pipeline_json)
count = pipeline.execute()

# Only return metadata - skip expensive array serialization
result = {{
    "count": count,
    "metadata": pipeline.metadata,
    "log": getattr(pipeline, 'log', '')
}}
print(json.dumps(result))
'''

        result = subprocess.run(
            [CONDA_PYTHON, '-c', script],
            capture_output=True, text=True, env=_get_conda_env()
        )

        if result.returncode != 0:
            raise RuntimeError(f"PDAL pipeline failed: {result.stderr}")

        try:
            data = json.loads(result.stdout)
        except json.JSONDecodeError as e:
            raise RuntimeError(f"Failed to parse PDAL output: {e}\nOutput: {result.stdout}\nStderr: {result.stderr}")

        self._count = data['count']
        self._metadata = data['metadata']
        self._log = data.get('log', '')
        self._arrays = []  # No arrays returned

        return self._count

    def execute_streaming_metadata(self, chunk_size: int = 100000) -> int:
        """
        Execute pipeline in streaming mode for memory-efficient metadata extraction.

        This processes points in chunks without loading the entire point cloud
        into memory. Useful for very large files that would otherwise cause
        out-of-memory errors.

        Note: Not all filters support streaming mode. filters.stats and
        filters.hexbin do support streaming.

        Args:
            chunk_size: Number of points to process at a time (default 100000)

        Returns:
            Number of points processed
        """
        if _NATIVE_PDAL_AVAILABLE:
            return self._execute_streaming_metadata_native(chunk_size)
        elif _CONDA_PDAL_AVAILABLE:
            return self._execute_streaming_metadata_subprocess(chunk_size)
        else:
            raise RuntimeError("PDAL is not available.")

    def _execute_streaming_metadata_native(self, chunk_size: int) -> int:
        """Execute streaming with native PDAL, extracting only metadata."""
        pipeline = _native_pdal.Pipeline(self.pipeline_json)
        count = pipeline.execute_streaming(chunk_size=chunk_size)
        self._count = count
        self._metadata = pipeline.metadata
        self._log = getattr(pipeline, 'log', '')
        self._arrays = []
        return count

    def _execute_streaming_metadata_subprocess(self, chunk_size: int) -> int:
        """Execute streaming using subprocess, returning only metadata."""
        script = f'''
import pdal
import json

pipeline_json = {repr(self.pipeline_json)}
pipeline = pdal.Pipeline(pipeline_json)
count = pipeline.execute_streaming(chunk_size={chunk_size})

# Only return metadata - no arrays in streaming mode anyway
result = {{
    "count": count,
    "metadata": pipeline.metadata,
    "log": getattr(pipeline, 'log', '')
}}
print(json.dumps(result))
'''

        result = subprocess.run(
            [CONDA_PYTHON, '-c', script],
            capture_output=True, text=True, env=_get_conda_env()
        )

        if result.returncode != 0:
            raise RuntimeError(f"PDAL streaming pipeline failed: {result.stderr}")

        try:
            data = json.loads(result.stdout)
        except json.JSONDecodeError as e:
            raise RuntimeError(f"Failed to parse PDAL output: {e}\nOutput: {result.stdout}\nStderr: {result.stderr}")

        self._count = data['count']
        self._metadata = data['metadata']
        self._log = data.get('log', '')
        self._arrays = []

        return self._count

    @property
    def arrays(self) -> List[np.ndarray]:
        """Get the point arrays from the pipeline."""
        return self._arrays

    @property
    def metadata(self) -> Dict[str, Any]:
        """Get the pipeline metadata."""
        return self._metadata

    @property
    def log(self) -> str:
        """Get the pipeline log."""
        return self._log


class PdalModule:
    """
    Drop-in replacement for the pdal module.

    Provides the same interface as the pdal module but uses subprocess
    when running in Google Colab with condacolab.
    """

    def __init__(self):
        self._version = None

    @property
    def __version__(self) -> str:
        """Get PDAL version."""
        if self._version is not None:
            return self._version

        if _NATIVE_PDAL_AVAILABLE:
            self._version = _native_pdal.__version__
        elif _CONDA_PDAL_AVAILABLE:
            result = subprocess.run(
                [CONDA_PYTHON, '-c', 'import pdal; print(pdal.__version__)'],
                capture_output=True, text=True, env=_get_conda_env()
            )
            self._version = result.stdout.strip() if result.returncode == 0 else "unknown"
        else:
            self._version = "not installed"

        return self._version

    def Pipeline(self, pipeline_json: str, arrays: Optional[List[np.ndarray]] = None) -> PipelineWrapper:
        """
        Create a new Pipeline.

        Args:
            pipeline_json: JSON string defining the PDAL pipeline
            arrays: Optional list of numpy structured arrays to use as input

        Returns:
            PipelineWrapper instance
        """
        return PipelineWrapper(pipeline_json, arrays=arrays)


# Create the module-level instance
pdal = PdalModule()


def get_pdal_status() -> Dict[str, Any]:
    """
    Get information about PDAL availability.

    Returns:
        Dict with status information
    """
    return {
        "in_colab": IN_COLAB,
        "native_pdal_available": _NATIVE_PDAL_AVAILABLE,
        "conda_pdal_available": _CONDA_PDAL_AVAILABLE,
        "conda_python_path": CONDA_PYTHON,
        "version": pdal.__version__,
        "mode": "native" if _NATIVE_PDAL_AVAILABLE else ("subprocess" if _CONDA_PDAL_AVAILABLE else "unavailable")
    }


# For backwards compatibility, also expose Pipeline at module level
Pipeline = pdal.Pipeline

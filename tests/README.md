# Tests for Topographic Differencing Uncertainty

This directory contains comprehensive tests for the Option 1 workflow from `my_implementation.ipynb`.

## Test Coverage

The test suite covers the following components:

### 1. Point Cloud Metadata ([test_pointcloud_metadata.py](test_pointcloud_metadata.py))
- Loading point clouds from `.laz` files
- Extracting metadata from point cloud headers
- Updating compound, horizontal, and vertical CRS
- Setting geoid models
- Updating epochs (single dates and date ranges)
- Verifying metadata consistency

### 2. Point Cloud Transformation ([test_pointcloud_transformation.py](test_pointcloud_transformation.py))
- Creating `PointCloudPair` objects
- Transforming compare cloud to match reference CRS
- Handling different CRS systems (horizontal and vertical)
- Skipping epoch transformation for faster testing
- Verbose output validation

### 3. Point Cloud Alignment ([test_alignment.py](test_alignment.py))
- VGICP (Voxelized Generalized ICP) alignment
- GICP (Generalized ICP) alignment
- Standard ICP alignment
- Memory-constrained alignment with downsampling
- Alignment result validation (RMSE, fitness, convergence)
- Transformation matrix properties
- Custom configuration options

### 4. DEM Creation ([test_dem_creation.py](test_dem_creation.py))
- Creating DTM pairs from aligned point clouds
- Setting DEM resolution
- Validating DEM properties (CRS, bounds, units)
- Setting vertical and horizontal units
- Output file management

### 5. Integration Tests ([test_option1_integration.py](test_option1_integration.py))
- Complete end-to-end workflow
- Step-by-step workflow validation
- Error handling
- Performance testing

## Running the Tests

### Prerequisites

Install test dependencies:
```bash
pip install pytest pytest-cov
```

Install the package with alignment support:
```bash
pip install -e ".[alignment]"
```

### Run All Tests

```bash
pytest tests/
```

### Run Specific Test Files

```bash
# Test metadata handling
pytest tests/test_pointcloud_metadata.py -v

# Test transformation
pytest tests/test_pointcloud_transformation.py -v

# Test alignment
pytest tests/test_alignment.py -v

# Test DEM creation
pytest tests/test_dem_creation.py -v

# Test complete workflow
pytest tests/test_option1_integration.py -v
```

### Run with Coverage

```bash
pytest tests/ --cov=topochange --cov-report=html
```

### Run Specific Test Classes or Methods

```bash
# Run a specific test class
pytest tests/test_alignment.py::TestAlignmentBasics -v

# Run a specific test method
pytest tests/test_alignment.py::TestAlignmentBasics::test_align_point_clouds_vgicp -v
```

## Test Data

Tests expect point cloud data in the `test_data/` directory:
- `test_data/compare.laz` - Compare (older) point cloud
- `test_data/reference.laz` - Reference (newer) point cloud

If test data is not available, tests will be skipped automatically.

## Notes

- **Velocity transformation tests are excluded** as requested
- Tests use `skip_epoch=True` for faster execution
- Alignment tests require `small_gicp` library
- Tests use temporary directories for output files (cleaned up automatically)
- Memory constraints are applied in alignment tests (reduced point counts)

## CI/CD Integration

To run tests in continuous integration:

```bash
# Install with test dependencies
pip install -e ".[dev,alignment]"

# Run tests with coverage
pytest tests/ --cov=topochange --cov-report=term --cov-report=xml

# Generate coverage badge (optional)
coverage-badge -o coverage.svg -f
```

## Troubleshooting

### Import Errors
If you get import errors for `small_gicp`, install it:
```bash
pip install small_gicp
```

### Missing Test Data
If tests are skipped due to missing data, ensure your test data is in the correct location:
```bash
ls test_data/compare.laz
ls test_data/reference.laz
```

### Slow Tests
Alignment tests can be slow. To run only fast tests:
```bash
pytest tests/ -m "not slow"
```

Or skip integration tests:
```bash
pytest tests/ --ignore=tests/test_option1_integration.py
```

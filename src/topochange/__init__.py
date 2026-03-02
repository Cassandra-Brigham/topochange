"""topographic change detection and uncertainty quantification."""

__version__ = "0.1.0"

from .raster import Raster
from .rasterpair import RasterPair

from .pointcloud import PointCloud
from .pointcloudpair import PointCloudPair

from .variogram import (
    RasterDataHandler,
    SingleVariogram,
    GridVariogram,
    KrigingLOOCVResult,
    AggregatedLOOCVResult,
    # backward-compatibility stubs
    VariogramAnalysis,
    FittedVariogramModel,
    EmpiricalVariogram,
    StatisticalAnalysis,
)
from .uncertainty import RegionalUncertaintyEstimator, DerivativeUncertaintyEstimator
from .variogram_models import MODEL_REGISTRY, VariogramModelRegistry
from .composite_variogram import CompositeVariogramModel

from .stable_area_analysis import (
    TopoMapInteractor,
    StableAreaRasterizer,
    StableAreaAnalyzer,
)

from .crs_history import CRSHistory
from .pipeline_builder import CRSState, build_vertical_pipeline

# data_access requires GDAL (osgeo) which may not be installed — guard the import
# so the rest of the package works without it. Users who need data_access should
# install with: pip install topochange[data_access]  (plus conda install gdal)
try:
    from .data_access import DataAccess, OpenTopographyQuery, GetDEMs
except ImportError:
    pass

from .alignment import (
    LandscapeAligner,
    RegistrationConfig,
    RegistrationResult,
    RegistrationMethod,
    align_point_clouds,
)

from .alignment_utils import (
    load_points_from_las,
    save_transformed_las,
    compute_alignment_quality,
    PointCloudPreprocessor,
    AlignmentQualityMetrics,
)

__all__ = [
    "__version__",
    "Raster",
    "RasterPair",
    "PointCloud",
    "PointCloudPair",
    "LandscapeAligner",
    "RegistrationConfig",
    "RegistrationResult",
    "RegistrationMethod",
    "align_point_clouds",
    "PointCloudPreprocessor",
    "AlignmentQualityMetrics",
    "load_points_from_las",
    "save_transformed_las",
    "compute_alignment_quality",
    "RasterDataHandler",
    "SingleVariogram",
    "GridVariogram",
    "KrigingLOOCVResult",
    "AggregatedLOOCVResult",
    "MODEL_REGISTRY",
    "VariogramModelRegistry",
    "CompositeVariogramModel",
    "RegionalUncertaintyEstimator",
    "DerivativeUncertaintyEstimator",
    "TopoMapInteractor",
    "StableAreaRasterizer",
    "StableAreaAnalyzer",
    "CRSHistory",
    "CRSState",
    "build_vertical_pipeline",
    "VariogramAnalysis",
    "FittedVariogramModel",
    "EmpiricalVariogram",
    "StatisticalAnalysis",
    "DataAccess",
    "OpenTopographyQuery",
    "GetDEMs",
]


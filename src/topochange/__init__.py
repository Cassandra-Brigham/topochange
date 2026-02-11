"""topographic change detection and uncertainty quantification."""

__version__ = "0.1.0"

from .raster import Raster
from .rasterpair import RasterPair

from .pointcloud import PointCloud
from .pointcloudpair import PointCloudPair

from .variogram import (
    RasterDataHandler,
    StatisticalAnalysis,
    VariogramAnalysis,
    VariogramModelSelector,
    FittedVariogramModel,
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

from .alignment import (
    LandscapeAligner,
    RegistrationConfig,
    RegistrationResult,
    RegistrationMethod,
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
    "PointCloudPreprocessor",
    "AlignmentQualityMetrics",
    "load_points_from_las",
    "save_transformed_las",
    "compute_alignment_quality",
    "RasterDataHandler",
    "StatisticalAnalysis",
    "VariogramAnalysis",
    "VariogramModelSelector",
    "FittedVariogramModel",
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
]


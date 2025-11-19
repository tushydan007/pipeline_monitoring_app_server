import os
from pathlib import Path
import rasterio
from rasterio.enums import Resampling
from rasterio.warp import (
    calculate_default_transform,
    reproject,
    transform_geom,
    transform_bounds,
)
from shapely.geometry import shape
from shapely.ops import transform
import pyproj
from rasterio.shutil import copy
from rasterio.io import MemoryFile
from rasterio.crs import CRS
import numpy as np
from typing import Dict, Any, List, Tuple
from skimage import filters, feature, morphology, segmentation
from skimage.feature import graycomatrix, graycoprops
from scipy import ndimage, signal
import json


def calculate_geojson_properties(geojson_data: dict) -> dict:
    """
    Calculate geometric properties from GeoJSON data

    Args:
        geojson_data: GeoJSON dictionary

    Returns:
        Dictionary with centroid, area, and perimeter
    """
    try:
        # Handle both Feature and FeatureCollection
        if geojson_data["type"] == "FeatureCollection":
            features = geojson_data["features"]
            if not features:
                return None
            # Use first feature
            geometry = features[0]["geometry"]
        elif geojson_data["type"] == "Feature":
            geometry = geojson_data["geometry"]
        else:
            geometry = geojson_data

        # Create shapely geometry
        geom = shape(geometry)

        # Get centroid in WGS84
        centroid = geom.centroid
        centroid_lat = centroid.y
        centroid_lon = centroid.x

        # Calculate area and perimeter in meters
        # Transform to appropriate projected CRS for accurate measurements
        # Use UTM zone based on centroid longitude
        utm_zone = int((centroid_lon + 180) / 6) + 1
        utm_crs = f"+proj=utm +zone={utm_zone} +datum=WGS84"

        project = pyproj.Transformer.from_crs(
            "EPSG:4326", utm_crs, always_xy=True  # WGS84
        ).transform

        geom_projected = transform(project, geom)

        area_m2 = geom_projected.area
        perimeter_m = geom_projected.length

        return {
            "centroid_lat": centroid_lat,
            "centroid_lon": centroid_lon,
            "area_m2": area_m2,
            "perimeter_m": perimeter_m,
        }
    except Exception as e:
        import logging

        logger = logging.getLogger(__name__)
        logger.error(f"Error calculating GeoJSON properties: {str(e)}")
        return None


def convert_to_cog(
    input_path: str | Path, output_path: str | Path | None = None
) -> Path:
    input_path = Path(input_path)
    if output_path is None:
        output_path = input_path.with_name(f"{input_path.stem}_cog.tif")

    with rasterio.open(input_path) as src:
        profile = src.profile.copy()
        profile.update(
            {
                "driver": "GTiff",
                "tiled": True,
                "blockxsize": 512,
                "blockysize": 512,
                "compress": "lzw",
                "interleave": "band",
            }
        )

        with rasterio.open(output_path, "w", **profile) as dst:
            for i in range(1, src.count + 1):
                dst.write(src.read(i), i)
            overviews = [2, 4, 8, 16]
            dst.build_overviews(overviews, Resampling.average)
            dst.update_tags(ns="rio_overview", resampling="average")

    return output_path  # Returns Path


def extract_geotiff_bbox(geotiff_path: str) -> Dict[str, float] | None:
    """
    Extract bounding box from GeoTIFF file and transform to WGS84 (EPSG:4326)
    for use with Leaflet maps

    Args:
        geotiff_path: Path to GeoTIFF file

    Returns:
        Dictionary with minx, miny, maxx, maxy in WGS84 (lat/lon) or None if error
    """
    try:
        with rasterio.open(geotiff_path) as src:
            # Get bounds in the source CRS
            bounds = src.bounds
            src_crs = src.crs

            # If source CRS is None, assume WGS84
            if src_crs is None:
                src_crs = CRS.from_epsg(4326)

            # Define target CRS (WGS84 for Leaflet)
            dst_crs = CRS.from_epsg(4326)

            # Transform bounds to WGS84 if needed
            if src_crs != dst_crs:
                minx, miny, maxx, maxy = transform_bounds(
                    src_crs,
                    dst_crs,
                    bounds.left,
                    bounds.bottom,
                    bounds.right,
                    bounds.top,
                )
            else:
                # Already in WGS84
                minx, miny, maxx, maxy = (
                    bounds.left,
                    bounds.bottom,
                    bounds.right,
                    bounds.top,
                )

            return {"minx": minx, "miny": miny, "maxx": maxx, "maxy": maxy}
    except Exception as e:
        import logging

        logger = logging.getLogger(__name__)
        logger.error(f"Error extracting bbox from {geotiff_path}: {str(e)}")
        return None


def run_pipeline_analysis(cog_path: str, satellite_image) -> Dict[str, Dict[str, Any]]:
    """
    Run comprehensive SAR-based pipeline monitoring analysis on satellite imagery

    Args:
        cog_path: Path to Cloud Optimized GeoTIFF
        satellite_image: SatelliteImage model instance

    Returns:
        Dictionary of analysis results by analysis type
    """
    results = {}

    try:
        with rasterio.open(cog_path) as src:
            # Read image data
            if src.count >= 3:
                # Multi-band image (e.g., RGB or multi-polarization SAR)
                band1 = src.read(1)
                band2 = src.read(2)
                band3 = src.read(3)
                multi_band = np.stack([band1, band2, band3], axis=0)
            elif src.count == 1:
                # Single band
                band1 = src.read(1)
                multi_band = np.stack([band1] * 3, axis=0)
            else:
                # Use first band
                band1 = src.read(1)
                multi_band = np.stack([band1] * 3, axis=0)

            # Use first band as primary for analysis
            primary_band = multi_band[0]

            # Normalize image
            image_normalized = (primary_band - primary_band.min()) / (
                primary_band.max() - primary_band.min() + 1e-8
            )

            # Run different SAR analysis types
            results["backscatter_threshold"] = backscatter_thresholding(
                image_normalized, src
            )
            results["glcm_texture"] = glcm_texture_analysis(image_normalized, src)
            results["polarimetric_decomp"] = (
                polarimetric_decomposition(multi_band, src) if src.count >= 3 else {}
            )
            results["time_series"] = time_series_tracking(
                image_normalized, src, satellite_image
            )
            results["insar"] = insar_analysis(image_normalized, src)
            results["dinsar"] = dinsar_analysis(image_normalized, src)
            results["psinsar"] = psinsar_analysis(image_normalized, src)
            results["sbas_insar"] = sbas_insar_analysis(image_normalized, src)
            results["phase_unwrap"] = phase_unwrapping(image_normalized, src)
            results["atmospheric_corr"] = atmospheric_correction(image_normalized, src)
            results["cfar"] = cfar_detection(image_normalized, src)
            results["polarimetric_scatter"] = (
                polarimetric_scattering_analysis(multi_band, src)
                if src.count >= 3
                else {}
            )
            results["texture_shape"] = texture_shape_metrics(image_normalized, src)
            results["coherence_change"] = coherence_change_detection(
                image_normalized, src
            )
            results["deep_learning"] = deep_learning_analysis(image_normalized, src)

    except Exception as e:
        # Return empty results on error
        for analysis_type in [
            "backscatter_threshold",
            "glcm_texture",
            "polarimetric_decomp",
            "time_series",
            "insar",
            "dinsar",
            "psinsar",
            "sbas_insar",
            "phase_unwrap",
            "atmospheric_corr",
            "cfar",
            "polarimetric_scatter",
            "texture_shape",
            "coherence_change",
            "deep_learning",
        ]:
            results[analysis_type] = {
                "confidence_score": 0.0,
                "severity": "low",
                "results": {},
                "metadata": {"error": str(e)},
                "processing_time": 0,
                "anomalies": [],
            }

    return results


def backscatter_thresholding(
    image: np.ndarray, src: rasterio.DatasetReader
) -> Dict[str, Any]:
    """Detect anomalies using backscatter intensity thresholding"""
    start_time = __import__("time").time()

    # Calculate backscatter statistics
    mean_backscatter = np.mean(image)
    std_backscatter = np.std(image)

    # Define thresholds (low and high backscatter anomalies)
    low_threshold = mean_backscatter - 2 * std_backscatter
    high_threshold = mean_backscatter + 2 * std_backscatter

    # Detect anomalous regions
    low_backscatter = image < low_threshold
    high_backscatter = image > high_threshold

    anomalies = []

    # Process low backscatter regions
    labeled_low = ndimage.label(low_backscatter)[0]
    regions_low = ndimage.find_objects(labeled_low)

    for region in regions_low:
        if region is not None:
            y_coords, x_coords = region
            area = (y_coords.stop - y_coords.start) * (x_coords.stop - x_coords.start)

            if area > 100:  # Minimum area threshold
                y_center = (y_coords.start + y_coords.stop) / 2
                x_center = (x_coords.start + x_coords.stop) / 2
                lon, lat = rasterio.transform.xy(src.transform, y_center, x_center)

                intensity = np.mean(image[y_coords, x_coords])
                confidence = min(
                    0.95,
                    0.6 + abs(intensity - mean_backscatter) / std_backscatter * 0.2,
                )

                severity = (
                    "critical" if area > 5000 else "high" if area > 2000 else "medium"
                )

                anomalies.append(
                    {
                        "type": "intensity_change",
                        "severity": severity,
                        "location": [float(lat), float(lon)],
                        "area_m2": float(area * 0.1),
                        "confidence": float(confidence),
                        "description": f"Low backscatter anomaly detected (intensity: {intensity:.3f})",
                        "metadata": {
                            "backscatter_value": float(intensity),
                            "type": "low",
                        },
                    }
                )

    # Process high backscatter regions
    labeled_high = ndimage.label(high_backscatter)[0]
    regions_high = ndimage.find_objects(labeled_high)

    for region in regions_high:
        if region is not None:
            y_coords, x_coords = region
            area = (y_coords.stop - y_coords.start) * (x_coords.stop - x_coords.start)

            if area > 100:
                y_center = (y_coords.start + y_coords.stop) / 2
                x_center = (x_coords.start + x_coords.stop) / 2
                lon, lat = rasterio.transform.xy(src.transform, y_center, x_center)

                intensity = np.mean(image[y_coords, x_coords])
                confidence = min(
                    0.95,
                    0.6 + abs(intensity - mean_backscatter) / std_backscatter * 0.2,
                )

                severity = "high" if area > 2000 else "medium"

                anomalies.append(
                    {
                        "type": "intensity_change",
                        "severity": severity,
                        "location": [float(lat), float(lon)],
                        "area_m2": float(area * 0.1),
                        "confidence": float(confidence),
                        "description": f"High backscatter anomaly detected (intensity: {intensity:.3f})",
                        "metadata": {
                            "backscatter_value": float(intensity),
                            "type": "high",
                        },
                    }
                )

    confidence_score = min(0.9, 0.5 + len(anomalies) * 0.08)
    overall_severity = (
        "critical"
        if len(anomalies) > 10
        else "high" if len(anomalies) > 5 else "medium" if len(anomalies) > 0 else "low"
    )

    processing_time = __import__("time").time() - start_time

    return {
        "confidence_score": float(confidence_score),
        "severity": overall_severity,
        "results": {
            "anomaly_count": len(anomalies),
            "mean_backscatter": float(mean_backscatter),
            "std_backscatter": float(std_backscatter),
        },
        "metadata": {
            "detection_method": "dinsar_analysis",
            "note": "Requires two SAR acquisitions for full DInSAR",
        },
        "processing_time": processing_time,
        "anomalies": anomalies,
    }


def psinsar_analysis(image: np.ndarray, src: rasterio.DatasetReader) -> Dict[str, Any]:
    """Persistent Scatterer InSAR analysis"""
    start_time = __import__("time").time()

    # Identify stable scatterers (high amplitude, low temporal variation)
    # Simulate temporal stability using spatial statistics
    amplitude = image
    local_std = ndimage.generic_filter(image, np.std, size=5)

    # Persistent scatterers have high amplitude and low variation
    ps_candidates = (amplitude > np.percentile(amplitude, 70)) & (
        local_std < np.percentile(local_std, 30)
    )

    # Analyze PS points for displacement
    ps_labeled = ndimage.label(ps_candidates)[0]
    ps_regions = ndimage.find_objects(ps_labeled)

    anomalies = []

    for region in ps_regions:
        if region is not None:
            y_coords, x_coords = region
            area = (y_coords.stop - y_coords.start) * (x_coords.stop - x_coords.start)

            if area > 50:
                y_center = (y_coords.start + y_coords.stop) / 2
                x_center = (x_coords.start + x_coords.stop) / 2
                lon, lat = rasterio.transform.xy(src.transform, y_center, x_center)

                # Simulate displacement measurement
                displacement = np.mean(image[y_coords, x_coords]) - np.mean(image)

                if abs(displacement) > 0.1:
                    confidence = min(0.92, 0.7 + abs(displacement) * 0.5)
                    severity = (
                        "critical"
                        if abs(displacement) > 0.3
                        else "high" if abs(displacement) > 0.2 else "medium"
                    )

                    anomalies.append(
                        {
                            "type": "displacement",
                            "severity": severity,
                            "location": [float(lat), float(lon)],
                            "area_m2": float(area * 0.1),
                            "confidence": float(confidence),
                            "description": f"PS-InSAR displacement detected ({displacement:.3f})",
                            "metadata": {
                                "displacement_mm": float(displacement * 1000),
                                "ps_count": 1,
                            },
                        }
                    )

    confidence_score = min(0.9, 0.65 + len(anomalies) * 0.1)
    overall_severity = (
        "critical"
        if len(anomalies) > 10
        else "high" if len(anomalies) > 5 else "medium" if len(anomalies) > 0 else "low"
    )

    processing_time = __import__("time").time() - start_time

    return {
        "confidence_score": float(confidence_score),
        "severity": overall_severity,
        "results": {
            "anomaly_count": len(anomalies),
            "ps_candidate_count": int(np.sum(ps_candidates)),
        },
        "metadata": {
            "detection_method": "psinsar_analysis",
            "note": "Requires time series of SAR acquisitions",
        },
        "processing_time": processing_time,
        "anomalies": anomalies,
    }


def sbas_insar_analysis(
    image: np.ndarray, src: rasterio.DatasetReader
) -> Dict[str, Any]:
    """Small Baseline Subset InSAR analysis"""
    start_time = __import__("time").time()

    # SBAS looks at small baseline interferograms
    # Simulate by analyzing local deformation patterns

    # Calculate gradient-based deformation proxy
    grad_y, grad_x = np.gradient(image)
    deformation_proxy = np.sqrt(grad_y**2 + grad_x**2)

    # Detect significant deformation areas
    deformation_threshold = np.percentile(deformation_proxy, 88)
    deformation_areas = deformation_proxy > deformation_threshold

    # Apply morphological operations to get coherent areas
    kernel = morphology.disk(3)
    deformation_areas = morphology.closing(deformation_areas, kernel)

    anomalies = []
    labeled = ndimage.label(deformation_areas)[0]
    regions = ndimage.find_objects(labeled)

    for region in regions:
        if region is not None:
            y_coords, x_coords = region
            area = (y_coords.stop - y_coords.start) * (x_coords.stop - x_coords.start)

            if area > 200:
                y_center = (y_coords.start + y_coords.stop) / 2
                x_center = (x_coords.start + x_coords.stop) / 2
                lon, lat = rasterio.transform.xy(src.transform, y_center, x_center)

                deform_val = np.mean(deformation_proxy[y_coords, x_coords])
                confidence = min(
                    0.89, 0.62 + (deform_val / deformation_threshold) * 0.22
                )

                severity = (
                    "critical" if area > 5000 else "high" if area > 2000 else "medium"
                )

                anomalies.append(
                    {
                        "type": "deformation",
                        "severity": severity,
                        "location": [float(lat), float(lon)],
                        "area_m2": float(area * 0.1),
                        "confidence": float(confidence),
                        "description": "SBAS deformation detected",
                        "metadata": {"deformation_rate": float(deform_val)},
                    }
                )

    confidence_score = min(0.87, 0.6 + len(anomalies) * 0.085)
    overall_severity = (
        "critical"
        if len(anomalies) > 7
        else "high" if len(anomalies) > 3 else "medium" if len(anomalies) > 0 else "low"
    )

    processing_time = __import__("time").time() - start_time

    return {
        "confidence_score": float(confidence_score),
        "severity": overall_severity,
        "results": {
            "anomaly_count": len(anomalies),
            "mean_deformation_rate": float(np.mean(deformation_proxy)),
        },
        "metadata": {
            "detection_method": "sbas_insar_analysis",
            "note": "Requires multiple SAR acquisitions with small baselines",
        },
        "processing_time": processing_time,
        "anomalies": anomalies,
    }


def phase_unwrapping(image: np.ndarray, src: rasterio.DatasetReader) -> Dict[str, Any]:
    """Phase unwrapping analysis"""
    start_time = __import__("time").time()

    # Simulate wrapped phase from image
    wrapped_phase = np.angle(np.exp(1j * image * 2 * np.pi))

    # Simple phase unwrapping using gradient-based approach
    unwrapped_phase = np.unwrap(np.unwrap(wrapped_phase, axis=0), axis=1)

    # Detect phase inconsistencies (unwrapping errors or real anomalies)
    phase_diff = np.abs(np.diff(unwrapped_phase, axis=0))
    phase_diff_padded = np.pad(phase_diff, ((0, 1), (0, 0)), mode="edge")

    phase_threshold = np.percentile(phase_diff_padded, 90)
    phase_inconsistencies = phase_diff_padded > phase_threshold

    anomalies = []
    labeled = ndimage.label(phase_inconsistencies)[0]
    regions = ndimage.find_objects(labeled)

    for region in regions:
        if region is not None:
            y_coords, x_coords = region
            area = (y_coords.stop - y_coords.start) * (x_coords.stop - x_coords.start)

            if area > 80:
                y_center = (y_coords.start + y_coords.stop) / 2
                x_center = (x_coords.start + x_coords.stop) / 2
                lon, lat = rasterio.transform.xy(src.transform, y_center, x_center)

                phase_jump = np.mean(phase_diff_padded[y_coords, x_coords])
                confidence = min(0.86, 0.58 + (phase_jump / phase_threshold) * 0.23)

                severity = "high" if phase_jump > phase_threshold * 1.3 else "medium"

                anomalies.append(
                    {
                        "type": "phase_anomaly",
                        "severity": severity,
                        "location": [float(lat), float(lon)],
                        "area_m2": float(area * 0.1),
                        "confidence": float(confidence),
                        "description": "Phase discontinuity detected",
                        "metadata": {"phase_jump": float(phase_jump)},
                    }
                )

    confidence_score = min(0.83, 0.57 + len(anomalies) * 0.08)
    overall_severity = (
        "high" if len(anomalies) > 8 else "medium" if len(anomalies) > 0 else "low"
    )

    processing_time = __import__("time").time() - start_time

    return {
        "confidence_score": float(confidence_score),
        "severity": overall_severity,
        "results": {
            "anomaly_count": len(anomalies),
            "phase_range": float(np.ptp(unwrapped_phase)),
        },
        "metadata": {
            "detection_method": "phase_unwrapping",
            "unwrapping_algorithm": "gradient_based",
        },
        "processing_time": processing_time,
        "anomalies": anomalies,
    }


def atmospheric_correction(
    image: np.ndarray, src: rasterio.DatasetReader
) -> Dict[str, Any]:
    """Atmospheric correction analysis"""
    start_time = __import__("time").time()

    # Estimate atmospheric effects using spatial low-pass filtering
    from scipy.ndimage import gaussian_filter

    # Simulate atmospheric phase screen
    atmospheric_component = gaussian_filter(image, sigma=20)

    # Corrected image
    corrected = image - atmospheric_component

    # Analyze residual after correction
    residual_std = np.std(corrected)
    residual_mean = np.mean(np.abs(corrected))

    # Detect areas with strong atmospheric effects
    atmospheric_threshold = np.percentile(np.abs(atmospheric_component), 85)
    strong_atmospheric = np.abs(atmospheric_component) > atmospheric_threshold

    anomalies = []
    labeled = ndimage.label(strong_atmospheric)[0]
    regions = ndimage.find_objects(labeled)

    for region in regions:
        if region is not None:
            y_coords, x_coords = region
            area = (y_coords.stop - y_coords.start) * (x_coords.stop - x_coords.start)

            if area > 500:  # Large-scale atmospheric effects
                y_center = (y_coords.start + y_coords.stop) / 2
                x_center = (x_coords.start + x_coords.stop) / 2
                lon, lat = rasterio.transform.xy(src.transform, y_center, x_center)

                atm_effect = np.mean(atmospheric_component[y_coords, x_coords])
                confidence = min(
                    0.8, 0.5 + abs(atm_effect) / atmospheric_threshold * 0.25
                )

                severity = (
                    "medium" if abs(atm_effect) > atmospheric_threshold * 1.2 else "low"
                )

                anomalies.append(
                    {
                        "type": "phase_anomaly",
                        "severity": severity,
                        "location": [float(lat), float(lon)],
                        "area_m2": float(area * 0.1),
                        "confidence": float(confidence),
                        "description": "Strong atmospheric effect detected",
                        "metadata": {"atmospheric_magnitude": float(atm_effect)},
                    }
                )

    confidence_score = min(0.78, 0.55 + len(anomalies) * 0.06)
    overall_severity = "medium" if len(anomalies) > 3 else "low"

    processing_time = __import__("time").time() - start_time

    return {
        "confidence_score": float(confidence_score),
        "severity": overall_severity,
        "results": {
            "anomaly_count": len(anomalies),
            "residual_std": float(residual_std),
            "correction_quality": float(1.0 - residual_std),
        },
        "metadata": {
            "detection_method": "atmospheric_correction",
            "correction_type": "spatial_filtering",
        },
        "processing_time": processing_time,
        "anomalies": anomalies,
    }


def cfar_detection(image: np.ndarray, src: rasterio.DatasetReader) -> Dict[str, Any]:
    """Constant False Alarm Rate (CFAR) detection"""
    start_time = __import__("time").time()

    # CFAR detector for target detection in SAR imagery
    window_size = 15
    guard_size = 3
    threshold_factor = 3.0

    height, width = image.shape
    detection_map = np.zeros_like(image, dtype=bool)

    # Simplified CFAR (cell-averaging CFAR)
    for i in range(window_size, height - window_size):
        for j in range(window_size, width - window_size):
            # Test cell
            test_cell = image[i, j]

            # Background window (excluding guard cells)
            window = image[
                i - window_size : i + window_size + 1,
                j - window_size : j + window_size + 1,
            ].copy()
            window[
                window_size - guard_size : window_size + guard_size + 1,
                window_size - guard_size : window_size + guard_size + 1,
            ] = 0

            # Calculate background statistics
            background = window[window > 0]
            if len(background) > 0:
                bg_mean = np.mean(background)
                bg_std = np.std(background)
                threshold = bg_mean + threshold_factor * bg_std

                if test_cell > threshold:
                    detection_map[i, j] = True

    anomalies = []
    labeled = ndimage.label(detection_map)[0]
    regions = ndimage.find_objects(labeled)

    for region in regions:
        if region is not None:
            y_coords, x_coords = region
            area = (y_coords.stop - y_coords.start) * (x_coords.stop - x_coords.start)

            if area > 30:  # Small targets
                y_center = (y_coords.start + y_coords.stop) / 2
                x_center = (x_coords.start + x_coords.stop) / 2
                lon, lat = rasterio.transform.xy(src.transform, y_center, x_center)

                target_intensity = np.mean(image[y_coords, x_coords])
                confidence = min(
                    0.92,
                    0.7 + (target_intensity - np.mean(image)) / np.std(image) * 0.15,
                )

                severity = (
                    "high" if target_intensity > np.percentile(image, 98) else "medium"
                )

                anomalies.append(
                    {
                        "type": "structural_change",
                        "severity": severity,
                        "location": [float(lat), float(lon)],
                        "area_m2": float(area * 0.1),
                        "confidence": float(confidence),
                        "description": "CFAR target detected",
                        "metadata": {"target_intensity": float(target_intensity)},
                    }
                )

    confidence_score = min(0.9, 0.65 + len(anomalies) * 0.09)
    overall_severity = (
        "high" if len(anomalies) > 15 else "medium" if len(anomalies) > 5 else "low"
    )

    processing_time = __import__("time").time() - start_time

    return {
        "confidence_score": float(confidence_score),
        "severity": overall_severity,
        "results": {
            "anomaly_count": len(anomalies),
            "detection_rate": len(anomalies) / (height * width) * 10000,
        },
        "metadata": {
            "detection_method": "cfar_detection",
            "cfar_type": "cell_averaging",
            "threshold_factor": threshold_factor,
        },
        "processing_time": processing_time,
        "anomalies": anomalies,
    }


def polarimetric_scattering_analysis(
    multi_band: np.ndarray, src: rasterio.DatasetReader
) -> Dict[str, Any]:
    """Polarimetric scattering mechanism analysis"""
    start_time = __import__("time").time()

    # Analyze scattering mechanisms from multi-polarization data
    hh = multi_band[0]
    hv = multi_band[1] if multi_band.shape[0] > 1 else multi_band[0] * 0.5
    vv = multi_band[2] if multi_band.shape[0] > 2 else multi_band[0] * 0.7

    # Calculate polarimetric ratios
    hh_vv_ratio = (hh + 1e-10) / (vv + 1e-10)
    cross_pol_ratio = (hv + 1e-10) / (hh + vv + 1e-10)

    # Detect anomalous scattering patterns
    ratio_mean = np.mean(hh_vv_ratio)
    ratio_std = np.std(hh_vv_ratio)

    anomalous_ratio = np.abs(hh_vv_ratio - ratio_mean) > 2 * ratio_std

    anomalies = []
    labeled = ndimage.label(anomalous_ratio)[0]
    regions = ndimage.find_objects(labeled)

    for region in regions:
        if region is not None:
            y_coords, x_coords = region
            area = (y_coords.stop - y_coords.start) * (x_coords.stop - x_coords.start)

            if area > 150:
                y_center = (y_coords.start + y_coords.stop) / 2
                x_center = (x_coords.start + x_coords.stop) / 2
                lon, lat = rasterio.transform.xy(src.transform, y_center, x_center)

                ratio_val = np.mean(hh_vv_ratio[y_coords, x_coords])
                cross_pol_val = np.mean(cross_pol_ratio[y_coords, x_coords])

                confidence = min(
                    0.88, 0.65 + abs(ratio_val - ratio_mean) / ratio_std * 0.15
                )
                severity = (
                    "high" if abs(ratio_val - ratio_mean) > 3 * ratio_std else "medium"
                )

                anomalies.append(
                    {
                        "type": "scattering_change",
                        "severity": severity,
                        "location": [float(lat), float(lon)],
                        "area_m2": float(area * 0.1),
                        "confidence": float(confidence),
                        "description": "Anomalous polarimetric scattering detected",
                        "metadata": {
                            "hh_vv_ratio": float(ratio_val),
                            "cross_pol_ratio": float(cross_pol_val),
                        },
                    }
                )

    confidence_score = min(0.86, 0.6 + len(anomalies) * 0.08)
    overall_severity = (
        "high" if len(anomalies) > 7 else "medium" if len(anomalies) > 0 else "low"
    )

    processing_time = __import__("time").time() - start_time

    return {
        "confidence_score": float(confidence_score),
        "severity": overall_severity,
        "results": {
            "anomaly_count": len(anomalies),
            "mean_hh_vv_ratio": float(ratio_mean),
            "mean_cross_pol_ratio": float(np.mean(cross_pol_ratio)),
        },
        "metadata": {
            "detection_method": "polarimetric_scattering_analysis",
        },
        "processing_time": processing_time,
        "anomalies": anomalies,
    }


def texture_shape_metrics(
    image: np.ndarray, src: rasterio.DatasetReader
) -> Dict[str, Any]:
    """Extract and analyze texture and shape metrics"""
    start_time = __import__("time").time()

    # Calculate multiple texture features
    from scipy.stats import entropy

    window_size = 32
    height, width = image.shape

    # Initialize feature maps
    entropy_map = np.zeros_like(image)
    variance_map = np.zeros_like(image)

    # Calculate local features
    for i in range(0, height - window_size, window_size // 2):
        for j in range(0, width - window_size, window_size // 2):
            window = image[i : i + window_size, j : j + window_size]

            # Entropy
            hist, _ = np.histogram(window, bins=32, range=(0, 1))
            local_entropy = entropy(hist + 1e-10)
            entropy_map[i : i + window_size, j : j + window_size] = local_entropy

            # Variance
            local_var = np.var(window)
            variance_map[i : i + window_size, j : j + window_size] = local_var

    # Combine features
    combined_feature = entropy_map * variance_map

    # Detect anomalous texture/shape regions
    feature_threshold = np.percentile(combined_feature, 90)
    anomalous_texture = combined_feature > feature_threshold

    anomalies = []
    labeled = ndimage.label(anomalous_texture)[0]
    regions = ndimage.find_objects(labeled)

    for region in regions:
        if region is not None:
            y_coords, x_coords = region
            area = (y_coords.stop - y_coords.start) * (x_coords.stop - x_coords.start)

            if area > 200:
                y_center = (y_coords.start + y_coords.stop) / 2
                x_center = (x_coords.start + x_coords.stop) / 2
                lon, lat = rasterio.transform.xy(src.transform, y_center, x_center)

                # Calculate shape metrics
                aspect_ratio = (y_coords.stop - y_coords.start) / (
                    x_coords.stop - x_coords.start + 1e-10
                )
                compactness = area / (4 * np.pi * ((area / np.pi) ** 0.5) ** 2 + 1e-10)

                feature_val = np.mean(combined_feature[y_coords, x_coords])
                confidence = min(0.87, 0.62 + (feature_val / feature_threshold) * 0.2)

                severity = "high" if area > 3000 else "medium"

                anomalies.append(
                    {
                        "type": "texture_anomaly",
                        "severity": severity,
                        "location": [float(lat), float(lon)],
                        "area_m2": float(area * 0.1),
                        "confidence": float(confidence),
                        "description": "Texture/shape anomaly detected",
                        "metadata": {
                            "aspect_ratio": float(aspect_ratio),
                            "compactness": float(compactness),
                            "texture_score": float(feature_val),
                        },
                    }
                )

    confidence_score = min(0.85, 0.58 + len(anomalies) * 0.08)
    overall_severity = (
        "high" if len(anomalies) > 8 else "medium" if len(anomalies) > 0 else "low"
    )

    processing_time = __import__("time").time() - start_time

    return {
        "confidence_score": float(confidence_score),
        "severity": overall_severity,
        "results": {
            "anomaly_count": len(anomalies),
            "mean_entropy": float(np.mean(entropy_map)),
            "mean_variance": float(np.mean(variance_map)),
        },
        "metadata": {
            "detection_method": "texture_shape_metrics",
            "window_size": window_size,
        },
        "processing_time": processing_time,
        "anomalies": anomalies,
    }


def coherence_change_detection(
    image: np.ndarray, src: rasterio.DatasetReader
) -> Dict[str, Any]:
    """Detect changes in interferometric coherence"""
    start_time = __import__("time").time()

    # Simulate coherence from intensity (real coherence needs complex SAR data pairs)
    # High coherence = stable areas, low coherence = changed areas

    # Calculate local similarity as coherence proxy
    window_size = 16
    height, width = image.shape
    coherence_map = np.ones_like(image)

    for i in range(0, height - window_size, window_size // 2):
        for j in range(0, width - window_size, window_size // 2):
            window = image[i : i + window_size, j : j + window_size]

            # Coherence proxy: inverse of normalized variance
            window_var = np.var(window)
            window_mean = np.mean(window)
            coherence_proxy = 1.0 / (1.0 + window_var / (window_mean + 1e-10))

            coherence_map[i : i + window_size, j : j + window_size] = coherence_proxy

    # Detect low coherence areas (potential changes)
    coherence_threshold = np.percentile(coherence_map, 20)  # Low coherence
    low_coherence = coherence_map < coherence_threshold

    anomalies = []
    labeled = ndimage.label(low_coherence)[0]
    regions = ndimage.find_objects(labeled)

    for region in regions:
        if region is not None:
            y_coords, x_coords = region
            area = (y_coords.stop - y_coords.start) * (x_coords.stop - x_coords.start)

            if area > 250:
                y_center = (y_coords.start + y_coords.stop) / 2
                x_center = (x_coords.start + x_coords.stop) / 2
                lon, lat = rasterio.transform.xy(src.transform, y_center, x_center)

                coherence_val = np.mean(coherence_map[y_coords, x_coords])
                confidence = min(0.88, 0.65 + (1.0 - coherence_val) * 0.2)

                severity = (
                    "critical"
                    if coherence_val < coherence_threshold * 0.5
                    else "high" if area > 3000 else "medium"
                )

                anomalies.append(
                    {
                        "type": "coherence_loss",
                        "severity": severity,
                        "location": [float(lat), float(lon)],
                        "area_m2": float(area * 0.1),
                        "confidence": float(confidence),
                        "description": f"Coherence loss detected (coherence: {coherence_val:.3f})",
                        "metadata": {"coherence_value": float(coherence_val)},
                    }
                )

    confidence_score = min(0.87, 0.6 + len(anomalies) * 0.085)
    overall_severity = (
        "critical"
        if len(anomalies) > 8
        else "high" if len(anomalies) > 4 else "medium" if len(anomalies) > 0 else "low"
    )

    processing_time = __import__("time").time() - start_time

    return {
        "confidence_score": float(confidence_score),
        "severity": overall_severity,
        "results": {
            "anomaly_count": len(anomalies),
            "mean_coherence": float(np.mean(coherence_map)),
            "low_coherence_percentage": float(
                np.sum(low_coherence) / low_coherence.size * 100
            ),
        },
        "metadata": {
            "detection_method": "coherence_change_detection",
            "note": "Requires SAR image pairs for true coherence calculation",
        },
        "processing_time": processing_time,
        "anomalies": anomalies,
    }


def deep_learning_analysis(
    image: np.ndarray, src: rasterio.DatasetReader
) -> Dict[str, Any]:
    """Deep learning-based anomaly detection (simplified feature-based approach)"""
    start_time = __import__("time").time()

    # Simulate deep learning features using traditional computer vision
    # In production, this would use a trained CNN/transformer model

    # Multi-scale feature extraction
    scales = [1, 2, 4]
    features = []

    for scale in scales:
        if scale > 1:
            scaled_image = ndimage.zoom(image, 1.0 / scale, order=1)
        else:
            scaled_image = image

        # Extract edges at this scale
        edges = feature.canny(scaled_image, sigma=scale)
        features.append(edges)

    # Combine multi-scale features
    combined_features = np.zeros_like(image)
    for i, scale_features in enumerate(features):
        if scales[i] > 1:
            resized = ndimage.zoom(scale_features.astype(float), scales[i], order=1)
            # Ensure same size
            if resized.shape != image.shape:
                resized = np.pad(
                    resized,
                    [
                        (0, max(0, image.shape[0] - resized.shape[0])),
                        (0, max(0, image.shape[1] - resized.shape[1])),
                    ],
                    mode="constant",
                )[: image.shape[0], : image.shape[1]]
        else:
            resized = scale_features.astype(float)
        combined_features += resized

    # Detect anomalies using combined features
    feature_threshold = np.percentile(combined_features, 92)
    anomalous_regions = combined_features > feature_threshold

    # Apply morphological operations
    kernel = morphology.disk(2)
    anomalous_regions = morphology.closing(anomalous_regions, kernel)

    anomalies = []
    labeled = ndimage.label(anomalous_regions)[0]
    regions = ndimage.find_objects(labeled)

    for region in regions:
        if region is not None:
            y_coords, x_coords = region
            area = (y_coords.stop - y_coords.start) * (x_coords.stop - x_coords.start)

            if area > 100:
                y_center = (y_coords.start + y_coords.stop) / 2
                x_center = (x_coords.start + x_coords.stop) / 2
                lon, lat = rasterio.transform.xy(src.transform, y_center, x_center)

                feature_strength = np.mean(combined_features[y_coords, x_coords])
                confidence = min(
                    0.93, 0.75 + (feature_strength / feature_threshold - 1.0) * 0.15
                )

                # Classify anomaly type based on feature patterns
                intensity_val = np.mean(image[y_coords, x_coords])
                if intensity_val < np.percentile(image, 30):
                    anomaly_type = "intensity_change"
                    description = "Low intensity anomaly detected by DL"
                elif intensity_val > np.percentile(image, 70):
                    anomaly_type = "structural_change"
                    description = "High intensity structure detected by DL"
                else:
                    anomaly_type = "texture_anomaly"
                    description = "Texture pattern anomaly detected by DL"

                severity = (
                    "critical" if area > 5000 else "high" if area > 2000 else "medium"
                )

                anomalies.append(
                    {
                        "type": anomaly_type,
                        "severity": severity,
                        "location": [float(lat), float(lon)],
                        "area_m2": float(area * 0.1),
                        "confidence": float(confidence),
                        "description": description,
                        "metadata": {
                            "feature_strength": float(feature_strength),
                            "dl_model": "multi_scale_feature_extraction",
                        },
                    }
                )

    confidence_score = min(0.91, 0.7 + len(anomalies) * 0.07)
    overall_severity = (
        "critical"
        if len(anomalies) > 10
        else "high" if len(anomalies) > 5 else "medium" if len(anomalies) > 0 else "low"
    )

    processing_time = __import__("time").time() - start_time

    return {
        "confidence_score": float(confidence_score),
        "severity": overall_severity,
        "results": {
            "anomaly_count": len(anomalies),
            "feature_extraction_scales": len(scales),
        },
        "metadata": {
            "detection_method": "deep_learning_analysis",
            "model_type": "multi_scale_cnn_simulation",
            "note": "Production version would use trained deep learning model",
        },
        "processing_time": processing_time,
        "anomalies": anomalies,
    }


# Helper functions for advanced SAR processing


def calculate_sar_statistics(image: np.ndarray) -> Dict[str, float]:
    """Calculate statistical properties of SAR image"""
    return {
        "mean": float(np.mean(image)),
        "std": float(np.std(image)),
        "min": float(np.min(image)),
        "max": float(np.max(image)),
        "median": float(np.median(image)),
        "percentile_25": float(np.percentile(image, 25)),
        "percentile_75": float(np.percentile(image, 75)),
        "variance": float(np.var(image)),
        "coefficient_of_variation": float(np.std(image) / (np.mean(image) + 1e-10)),
    }


def apply_speckle_filter(
    image: np.ndarray, filter_type: str = "lee", window_size: int = 5
) -> np.ndarray:
    """
    Apply speckle filtering to SAR image

    Args:
        image: Input SAR image
        filter_type: Type of filter ('lee', 'frost', 'kuan', 'median')
        window_size: Size of filtering window

    Returns:
        Filtered image
    """
    if filter_type == "median":
        return ndimage.median_filter(image, size=window_size)
    elif filter_type == "lee":
        # Lee filter implementation
        mean = ndimage.uniform_filter(image, size=window_size)
        sqr_mean = ndimage.uniform_filter(image**2, size=window_size)
        variance = sqr_mean - mean**2
        variance = np.maximum(variance, 0)  # Ensure non-negative

        # Calculate weights
        overall_variance = np.var(image)
        weights = variance / (variance + overall_variance + 1e-10)

        filtered = mean + weights * (image - mean)
        return filtered
    elif filter_type == "frost":
        # Simplified Frost filter
        mean = ndimage.uniform_filter(image, size=window_size)
        std = ndimage.generic_filter(image, np.std, size=window_size)

        k = 1.0  # Damping factor
        weights = np.exp(-k * std / (mean + 1e-10))

        filtered = weights * image + (1 - weights) * mean
        return filtered
    elif filter_type == "kuan":
        # Kuan filter (simplified)
        mean = ndimage.uniform_filter(image, size=window_size)
        sqr_mean = ndimage.uniform_filter(image**2, size=window_size)
        variance = sqr_mean - mean**2
        variance = np.maximum(variance, 0)

        overall_variance = np.var(image)
        cu = np.sqrt(overall_variance) / (np.mean(image) + 1e-10)
        cu_sqr = cu**2

        ci_sqr = variance / (mean**2 + 1e-10)
        weights = 1 - cu_sqr / (ci_sqr + 1e-10)
        weights = np.clip(weights, 0, 1)

        filtered = mean + weights * (image - mean)
        return filtered
    else:
        return image


def extract_pipeline_buffer_zone(
    image: np.ndarray,
    src: rasterio.DatasetReader,
    pipeline_coords: List[Tuple[float, float]],
    buffer_m: float = 100,
) -> np.ndarray:
    """
    Extract region around pipeline from SAR image

    Args:
        image: Input SAR image
        src: Rasterio dataset
        pipeline_coords: List of (lat, lon) coordinates defining pipeline
        buffer_m: Buffer distance in meters

    Returns:
        Masked image with only pipeline buffer zone
    """
    from shapely.geometry import LineString, Point
    from shapely.ops import transform
    import pyproj

    # Create mask
    mask = np.zeros_like(image, dtype=bool)

    # Convert pipeline coordinates to pixel coordinates
    pixel_coords = []
    for lat, lon in pipeline_coords:
        row, col = rasterio.transform.rowcol(src.transform, lon, lat)
        pixel_coords.append((row, col))

    # Create buffer around pipeline in pixel space
    # Approximate buffer in pixels (simplified)
    buffer_pixels = int(buffer_m / 10)  # Rough approximation

    for row, col in pixel_coords:
        y_min = max(0, row - buffer_pixels)
        y_max = min(image.shape[0], row + buffer_pixels)
        x_min = max(0, col - buffer_pixels)
        x_max = min(image.shape[1], col + buffer_pixels)

        mask[y_min:y_max, x_min:x_max] = True

    # Apply mask
    masked_image = np.copy(image)
    masked_image[~mask] = 0

    return masked_image


def calculate_coherence_map(
    image1: np.ndarray, image2: np.ndarray, window_size: int = 5
) -> np.ndarray:
    """
    Calculate interferometric coherence between two SAR images

    Args:
        image1: First SAR image (complex or intensity)
        image2: Second SAR image (complex or intensity)
        window_size: Window size for coherence estimation

    Returns:
        Coherence map
    """
    # Convert to complex if needed (simulate from intensity)
    if not np.iscomplexobj(image1):
        image1_complex = image1 * np.exp(
            1j * np.random.uniform(0, 2 * np.pi, image1.shape)
        )
    else:
        image1_complex = image1

    if not np.iscomplexobj(image2):
        image2_complex = image2 * np.exp(
            1j * np.random.uniform(0, 2 * np.pi, image2.shape)
        )
    else:
        image2_complex = image2

    # Calculate coherence
    conjugate_product = image1_complex * np.conj(image2_complex)

    # Windowed averaging
    numerator = np.abs(ndimage.uniform_filter(conjugate_product, size=window_size))

    denominator1 = ndimage.uniform_filter(np.abs(image1_complex) ** 2, size=window_size)
    denominator2 = ndimage.uniform_filter(np.abs(image2_complex) ** 2, size=window_size)
    denominator = np.sqrt(denominator1 * denominator2 + 1e-10)

    coherence = numerator / denominator
    coherence = np.clip(coherence, 0, 1)

    return coherence


def estimate_displacement_from_phase(
    phase_diff: np.ndarray, wavelength: float = 0.056
) -> np.ndarray:
    """
    Estimate displacement from interferometric phase

    Args:
        phase_diff: Phase difference image (radians)
        wavelength: SAR wavelength in meters (default: C-band ~5.6 cm)

    Returns:
        Displacement map in meters
    """
    # Line-of-sight displacement
    displacement = (phase_diff * wavelength) / (4 * np.pi)

    return displacement


def detect_linear_features(image: np.ndarray, min_length: int = 50) -> List[np.ndarray]:
    """
    Detect linear features (e.g., pipelines) in SAR image using Hough transform

    Args:
        image: Input SAR image
        min_length: Minimum length of lines to detect (pixels)

    Returns:
        List of detected line segments
    """
    from skimage.transform import probabilistic_hough_line

    # Edge detection
    edges = feature.canny(image, sigma=2.0)

    # Hough line detection
    lines = probabilistic_hough_line(
        edges, threshold=10, line_length=min_length, line_gap=5
    )

    return lines


def calculate_polarimetric_indices(
    hh: np.ndarray, hv: np.ndarray, vv: np.ndarray
) -> Dict[str, np.ndarray]:
    """
    Calculate polarimetric indices from dual/quad-pol SAR data

    Args:
        hh: HH polarization channel
        hv: HV polarization channel (cross-pol)
        vv: VV polarization channel

    Returns:
        Dictionary of polarimetric indices
    """
    indices = {}

    # Span (total power)
    indices["span"] = hh + 2 * hv + vv

    # Polarimetric radar vegetation index (RVI)
    indices["rvi"] = (4 * hv) / (hh + vv + 1e-10)

    # Cross-polarization ratio
    indices["cpr"] = hv / (hh + vv + 1e-10)

    # Co-polarization ratio
    indices["co_pol_ratio"] = hh / (vv + 1e-10)

    # Depolarization ratio
    indices["depol_ratio"] = (hv + vv) / (hh + 1e-10)

    # Normalized difference polarization index
    indices["ndpi"] = (hh - vv) / (hh + vv + 1e-10)

    # Entropy (simplified)
    total = hh + hv + vv + 1e-10
    p1 = hh / total
    p2 = hv / total
    p3 = vv / total
    indices["entropy"] = -(
        p1 * np.log(p1 + 1e-10) + p2 * np.log(p2 + 1e-10) + p3 * np.log(p3 + 1e-10)
    )

    return indices


def classify_scattering_mechanism(
    hh: np.ndarray, hv: np.ndarray, vv: np.ndarray
) -> np.ndarray:
    """
    Classify dominant scattering mechanism

    Args:
        hh: HH polarization
        hv: HV polarization
        vv: VV polarization

    Returns:
        Classification map (0=surface, 1=double-bounce, 2=volume)
    """
    # Freeman-Durden components
    surface = np.abs(hh + vv) ** 2
    double_bounce = np.abs(hh - vv) ** 2
    volume = 4 * np.abs(hv) ** 2

    # Stack and find argmax
    mechanisms = np.stack([surface, double_bounce, volume], axis=0)
    classification = np.argmax(mechanisms, axis=0)

    return classification


def temporal_coherence_analysis(image_stack: List[np.ndarray]) -> Dict[str, np.ndarray]:
    """
    Analyze temporal coherence across multiple acquisitions

    Args:
        image_stack: List of SAR images from different dates

    Returns:
        Dictionary with temporal statistics
    """
    stack_array = np.stack(image_stack, axis=0)

    results = {
        "mean": np.mean(stack_array, axis=0),
        "std": np.std(stack_array, axis=0),
        "temporal_coherence": 1.0 / (1.0 + np.std(stack_array, axis=0)),
        "change_intensity": np.max(stack_array, axis=0) - np.min(stack_array, axis=0),
        "temporal_variance": np.var(stack_array, axis=0),
    }

    return results


def apply_radiometric_calibration(
    image: np.ndarray, calibration_type: str = "sigma0"
) -> np.ndarray:
    """
    Apply radiometric calibration to SAR image

    Args:
        image: Input SAR image (DN values)
        calibration_type: Type of calibration ('sigma0', 'gamma0', 'beta0')

    Returns:
        Calibrated image
    """
    # Simplified calibration (in production, use actual calibration parameters)
    # Convert from DN to backscatter coefficient (dB)

    # Avoid log of zero
    image_safe = np.maximum(image, 1e-10)

    # Convert to dB
    image_db = 10 * np.log10(image_safe)

    # Apply calibration offset (simplified - use actual cal constants in production)
    if calibration_type == "sigma0":
        calibrated = image_db  # Sigma nought
    elif calibration_type == "gamma0":
        calibrated = image_db + 3  # Gamma nought (simplified)
    elif calibration_type == "beta0":
        calibrated = image_db - 3  # Beta nought (simplified)
    else:
        calibrated = image_db

    return calibrated


def detect_sar_shadows(
    image: np.ndarray, threshold_percentile: float = 10
) -> np.ndarray:
    """
    Detect radar shadows in SAR image

    Args:
        image: Input SAR image
        threshold_percentile: Percentile threshold for shadow detection

    Returns:
        Binary mask of shadow areas
    """
    threshold = np.percentile(image, threshold_percentile)
    shadows = image < threshold

    # Remove small isolated pixels
    shadows = morphology.remove_small_objects(shadows, min_size=50)

    return shadows


def calculate_sar_texture_features(
    image: np.ndarray, window_size: int = 11
) -> Dict[str, np.ndarray]:
    """
    Calculate comprehensive texture features from SAR image

    Args:
        image: Input SAR image
        window_size: Window size for feature calculation

    Returns:
        Dictionary of texture feature maps
    """
    features = {}

    # Convert to uint8 for GLCM
    image_uint8 = (
        (image - image.min()) / (image.max() - image.min() + 1e-10) * 255
    ).astype(np.uint8)

    # Local statistics
    features["local_mean"] = ndimage.uniform_filter(image, size=window_size)
    features["local_variance"] = ndimage.generic_filter(image, np.var, size=window_size)
    features["local_std"] = np.sqrt(features["local_variance"])

    # Range (max - min in window)
    features["local_range"] = ndimage.maximum_filter(
        image, size=window_size
    ) - ndimage.minimum_filter(image, size=window_size)

    # Coefficient of variation
    features["local_cv"] = features["local_std"] / (features["local_mean"] + 1e-10)

    # Gradient magnitude
    grad_y, grad_x = np.gradient(image)
    features["gradient_magnitude"] = np.sqrt(grad_y**2 + grad_x**2)

    # Laplacian (second derivative)
    features["laplacian"] = ndimage.laplace(image)

    return features


def perform_change_detection(
    image_before: np.ndarray, image_after: np.ndarray, method: str = "ratio"
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Perform change detection between two SAR images

    Args:
        image_before: SAR image from earlier date
        image_after: SAR image from later date
        method: Change detection method ('ratio', 'difference', 'log_ratio')

    Returns:
        Tuple of (change_map, change_mask)
    """
    if method == "ratio":
        change_map = (image_after + 1e-10) / (image_before + 1e-10)
    elif method == "log_ratio":
        change_map = np.log((image_after + 1e-10) / (image_before + 1e-10))
    elif method == "difference":
        change_map = image_after - image_before
    else:
        change_map = image_after - image_before

    # Threshold to get binary change mask
    # Use Otsu's method for automatic thresholding
    threshold = filters.threshold_otsu(np.abs(change_map))
    change_mask = np.abs(change_map) > threshold

    return change_map, change_mask

    # return change_map, change_maskmethod': 'backscatter_thresholding',
    #         'thresholds': {'low': float(low_threshold), 'high': float(high_threshold)}
    #     },
    #     'processing_time': processing_time,
    #     'anomalies': anomalies
    # }


def glcm_texture_analysis(
    image: np.ndarray, src: rasterio.DatasetReader
) -> Dict[str, Any]:
    """Analyze texture using Gray Level Co-occurrence Matrix (GLCM)"""
    start_time = __import__("time").time()

    # Convert to uint8 for GLCM
    image_uint8 = (image * 255).astype(np.uint8)

    # Calculate GLCM for multiple directions
    distances = [1, 3, 5]
    angles = [0, np.pi / 4, np.pi / 2, 3 * np.pi / 4]

    Calculate GLCM
    glcm = graycomatrix(
        image_uint8,
        distances=distances,
        angles=angles,
        levels=256,
        symmetric=True,
        normed=True,
    )

    # Extract texture properties
    contrast = graycoprops(glcm, "contrast").mean()
    dissimilarity = graycoprops(glcm, "dissimilarity").mean()
    homogeneity = graycoprops(glcm, "homogeneity").mean()
    energy = graycoprops(glcm, "energy").mean()
    correlation = graycoprops(glcm, "correlation").mean()

    # Compute local texture variation
    window_size = 32
    height, width = image.shape
    texture_map = np.zeros_like(image)

    anomalies = []

    for i in range(0, height - window_size, window_size // 2):
        for j in range(0, width - window_size, window_size // 2):
            window = image_uint8[i : i + window_size, j : j + window_size]

            if window.shape[0] == window_size and window.shape[1] == window_size:
                local_glcm = graycomatrix(
                    window,
                    distances=[1],
                    angles=[0],
                    levels=256,
                    symmetric=True,
                    normed=True,
                )
                local_contrast = graycoprops(local_glcm, "contrast")[0, 0]
                texture_map[i : i + window_size, j : j + window_size] = local_contrast

    # Detect texture anomalies
    texture_threshold = np.percentile(texture_map, 95)
    texture_anomalies = texture_map > texture_threshold

    labeled = ndimage.label(texture_anomalies)[0]
    regions = ndimage.find_objects(labeled)

    for region in regions:
        if region is not None:
            y_coords, x_coords = region
            area = (y_coords.stop - y_coords.start) * (x_coords.stop - x_coords.start)

            if area > 200:
                y_center = (y_coords.start + y_coords.stop) / 2
                x_center = (x_coords.start + x_coords.stop) / 2
                lon, lat = rasterio.transform.xy(src.transform, y_center, x_center)

                local_contrast_val = np.mean(texture_map[y_coords, x_coords])
                confidence = min(
                    0.92, 0.65 + (local_contrast_val / texture_threshold) * 0.2
                )

                severity = "high" if area > 3000 else "medium"

                anomalies.append(
                    {
                        "type": "texture_anomaly",
                        "severity": severity,
                        "location": [float(lat), float(lon)],
                        "area_m2": float(area * 0.1),
                        "confidence": float(confidence),
                        "description": f"Texture anomaly detected (contrast: {local_contrast_val:.2f})",
                        "metadata": {"local_contrast": float(local_contrast_val)},
                    }
                )

    confidence_score = min(0.88, 0.6 + len(anomalies) * 0.07)
    overall_severity = (
        "high" if len(anomalies) > 8 else "medium" if len(anomalies) > 0 else "low"
    )

    processing_time = __import__("time").time() - start_time

    return {
        "confidence_score": float(confidence_score),
        "severity": overall_severity,
        "results": {
            "anomaly_count": len(anomalies),
            "contrast": float(contrast),
            "dissimilarity": float(dissimilarity),
            "homogeneity": float(homogeneity),
            "energy": float(energy),
            "correlation": float(correlation),
        },
        "metadata": {
            "detection_method": "glcm_texture_analysis",
            "window_size": window_size,
        },
        "processing_time": processing_time,
        "anomalies": anomalies,
    }


def polarimetric_decomposition(
    multi_band: np.ndarray, src: rasterio.DatasetReader
) -> Dict[str, Any]:
    """Perform polarimetric decomposition (simplified Freeman-Durden)"""
    start_time = __import__("time").time()

    # Simulate polarimetric channels (HH, HV, VV)
    hh = multi_band[0]
    hv = multi_band[1] if multi_band.shape[0] > 1 else multi_band[0] * 0.5
    vv = multi_band[2] if multi_band.shape[0] > 2 else multi_band[0] * 0.7

    # Freeman-Durden decomposition components (simplified)
    # Surface scattering (even bounce)
    surface = np.abs(hh + vv) ** 2

    # Double-bounce scattering
    double_bounce = np.abs(hh - vv) ** 2

    # Volume scattering
    volume = 4 * np.abs(hv) ** 2

    # Total power
    total_power = surface + double_bounce + volume + 1e-10

    # Normalize components
    surface_norm = surface / total_power
    double_bounce_norm = double_bounce / total_power
    volume_norm = volume / total_power

    anomalies = []

    # Detect anomalies based on scattering mechanisms
    # High double-bounce can indicate structures/infrastructure
    high_double_bounce = double_bounce_norm > 0.6
    labeled = ndimage.label(high_double_bounce)[0]
    regions = ndimage.find_objects(labeled)

    for region in regions:
        if region is not None:
            y_coords, x_coords = region
            area = (y_coords.stop - y_coords.start) * (x_coords.stop - x_coords.start)

            if area > 150:
                y_center = (y_coords.start + y_coords.stop) / 2
                x_center = (x_coords.start + x_coords.stop) / 2
                lon, lat = rasterio.transform.xy(src.transform, y_center, x_center)

                db_value = np.mean(double_bounce_norm[y_coords, x_coords])
                confidence = min(0.9, 0.7 + (db_value - 0.6) * 0.5)

                severity = "high" if area > 2000 else "medium"

                anomalies.append(
                    {
                        "type": "scattering_change",
                        "severity": severity,
                        "location": [float(lat), float(lon)],
                        "area_m2": float(area * 0.1),
                        "confidence": float(confidence),
                        "description": "Polarimetric scattering anomaly (double-bounce dominant)",
                        "metadata": {"double_bounce_ratio": float(db_value)},
                    }
                )

    confidence_score = min(0.87, 0.6 + len(anomalies) * 0.08)
    overall_severity = (
        "high" if len(anomalies) > 5 else "medium" if len(anomalies) > 0 else "low"
    )

    processing_time = __import__("time").time() - start_time

    return {
        "confidence_score": float(confidence_score),
        "severity": overall_severity,
        "results": {
            "anomaly_count": len(anomalies),
            "mean_surface": float(np.mean(surface_norm)),
            "mean_double_bounce": float(np.mean(double_bounce_norm)),
            "mean_volume": float(np.mean(volume_norm)),
        },
        "metadata": {
            "detection_method": "polarimetric_decomposition",
            "decomposition_type": "freeman_durden_simplified",
        },
        "processing_time": processing_time,
        "anomalies": anomalies,
    }


def time_series_tracking(
    image: np.ndarray, src: rasterio.DatasetReader, satellite_image
) -> Dict[str, Any]:
    """Track temporal changes (requires multiple acquisitions - simplified single-image version)"""
    start_time = __import__("time").time()

    # In a full implementation, this would compare multiple time steps
    # For now, analyze temporal patterns within the image

    # Simulate temporal variance by analyzing spatial patterns
    window_size = 64
    height, width = image.shape
    temporal_variance_map = np.zeros_like(image)

    for i in range(0, height - window_size, window_size // 2):
        for j in range(0, width - window_size, window_size // 2):
            window = image[i : i + window_size, j : j + window_size]
            variance = np.var(window)
            temporal_variance_map[i : i + window_size, j : j + window_size] = variance

    # Detect high variance regions (potential changes)
    variance_threshold = np.percentile(temporal_variance_map, 90)
    high_variance = temporal_variance_map > variance_threshold

    anomalies = []
    labeled = ndimage.label(high_variance)[0]
    regions = ndimage.find_objects(labeled)

    for region in regions:
        if region is not None:
            y_coords, x_coords = region
            area = (y_coords.stop - y_coords.start) * (x_coords.stop - x_coords.start)

            if area > 250:
                y_center = (y_coords.start + y_coords.stop) / 2
                x_center = (x_coords.start + x_coords.stop) / 2
                lon, lat = rasterio.transform.xy(src.transform, y_center, x_center)

                variance_val = np.mean(temporal_variance_map[y_coords, x_coords])
                confidence = min(
                    0.85, 0.55 + (variance_val / variance_threshold) * 0.25
                )

                severity = "high" if area > 3000 else "medium"

                anomalies.append(
                    {
                        "type": "temporal_anomaly",
                        "severity": severity,
                        "location": [float(lat), float(lon)],
                        "area_m2": float(area * 0.1),
                        "confidence": float(confidence),
                        "description": "Temporal variation detected",
                        "metadata": {"variance": float(variance_val)},
                    }
                )

    confidence_score = min(0.8, 0.5 + len(anomalies) * 0.07)
    overall_severity = (
        "high" if len(anomalies) > 6 else "medium" if len(anomalies) > 0 else "low"
    )

    processing_time = __import__("time").time() - start_time

    return {
        "confidence_score": float(confidence_score),
        "severity": overall_severity,
        "results": {
            "anomaly_count": len(anomalies),
            "mean_temporal_variance": float(np.mean(temporal_variance_map)),
        },
        "metadata": {
            "detection_method": "time_series_tracking",
            "note": "Single acquisition - spatial variance used as proxy",
        },
        "processing_time": processing_time,
        "anomalies": anomalies,
    }


def insar_analysis(image: np.ndarray, src: rasterio.DatasetReader) -> Dict[str, Any]:
    """Interferometric SAR analysis (simplified - requires SAR complex data)"""
    start_time = __import__("time").time()

    # Simulate phase information from intensity
    simulated_phase = np.angle(image + 1j * np.gradient(image)[0])

    # Detect phase discontinuities
    phase_gradient = np.gradient(simulated_phase)
    phase_magnitude = np.sqrt(phase_gradient[0] ** 2 + phase_gradient[1] ** 2)

    # Threshold for detecting significant phase changes
    phase_threshold = np.percentile(phase_magnitude, 85)
    phase_anomalies = phase_magnitude > phase_threshold

    anomalies = []
    labeled = ndimage.label(phase_anomalies)[0]
    regions = ndimage.find_objects(labeled)

    for region in regions:
        if region is not None:
            y_coords, x_coords = region
            area = (y_coords.stop - y_coords.start) * (x_coords.stop - x_coords.start)

            if area > 100:
                y_center = (y_coords.start + y_coords.stop) / 2
                x_center = (x_coords.start + x_coords.stop) / 2
                lon, lat = rasterio.transform.xy(src.transform, y_center, x_center)

                phase_val = np.mean(phase_magnitude[y_coords, x_coords])
                confidence = min(0.88, 0.6 + (phase_val / phase_threshold) * 0.2)

                severity = (
                    "critical" if area > 4000 else "high" if area > 1500 else "medium"
                )

                anomalies.append(
                    {
                        "type": "phase_anomaly",
                        "severity": severity,
                        "location": [float(lat), float(lon)],
                        "area_m2": float(area * 0.1),
                        "confidence": float(confidence),
                        "description": "InSAR phase anomaly detected",
                        "metadata": {"phase_gradient": float(phase_val)},
                    }
                )

    confidence_score = min(0.85, 0.55 + len(anomalies) * 0.08)
    overall_severity = (
        "critical"
        if len(anomalies) > 8
        else "high" if len(anomalies) > 3 else "medium" if len(anomalies) > 0 else "low"
    )

    processing_time = __import__("time").time() - start_time

    return {
        "confidence_score": float(confidence_score),
        "severity": overall_severity,
        "results": {
            "anomaly_count": len(anomalies),
            "mean_phase_gradient": float(np.mean(phase_magnitude)),
        },
        "metadata": {
            "detection_method": "insar_analysis",
            "note": "Simplified implementation - requires complex SAR data for full InSAR",
        },
        "processing_time": processing_time,
        "anomalies": anomalies,
    }


def dinsar_analysis(image: np.ndarray, src: rasterio.DatasetReader) -> Dict[str, Any]:
    """Differential InSAR for displacement detection"""
    start_time = __import__("time").time()

    # Simulate differential phase (would require two SAR acquisitions)
    simulated_diff_phase = np.gradient(image)[0] - np.gradient(image)[1]

    # Detect significant displacements
    displacement_threshold = np.percentile(np.abs(simulated_diff_phase), 92)
    displacement_areas = np.abs(simulated_diff_phase) > displacement_threshold

    anomalies = []
    labeled = ndimage.label(displacement_areas)[0]
    regions = ndimage.find_objects(labeled)

    for region in regions:
        if region is not None:
            y_coords, x_coords = region
            area = (y_coords.stop - y_coords.start) * (x_coords.stop - x_coords.start)

            if area > 120:
                y_center = (y_coords.start + y_coords.stop) / 2
                x_center = (x_coords.start + x_coords.stop) / 2
                lon, lat = rasterio.transform.xy(src.transform, y_center, x_center)

                disp_val = np.mean(simulated_diff_phase[y_coords, x_coords])
                confidence = min(
                    0.9, 0.65 + abs(disp_val) / displacement_threshold * 0.2
                )

                severity = (
                    "critical"
                    if abs(disp_val) > displacement_threshold * 1.5
                    else "high" if area > 2000 else "medium"
                )

                anomalies.append(
                    {
                        "type": "displacement",
                        "severity": severity,
                        "location": [float(lat), float(lon)],
                        "area_m2": float(area * 0.1),
                        "confidence": float(confidence),
                        "description": f"Ground displacement detected ({abs(disp_val):.3f})",
                        "metadata": {"displacement_value": float(disp_val)},
                    }
                )

    confidence_score = min(0.88, 0.6 + len(anomalies) * 0.09)
    overall_severity = (
        "critical"
        if len(anomalies) > 5
        else "high" if len(anomalies) > 2 else "medium" if len(anomalies) > 0 else "low"
    )

    processing_time = __import__("time").time() - start_time

    return {
        "confidence_score": float(confidence_score),
        "severity": overall_severity,
        "results": {
            "anomaly_count": len(anomalies),
            "mean_displacement": float(np.mean(simulated_diff_phase)),
        },
        "metadata": {
            "detection_method": "dinsar_analysis",
            "note": "Requires two SAR acquisitions for full DInSAR",
        },
        "processing_time": processing_time,
        "anomalies": anomalies,
    }

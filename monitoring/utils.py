import os
import rasterio
from rasterio.enums import Resampling
from rasterio.warp import calculate_default_transform, reproject, transform_geom, transform_bounds
from rasterio.shutil import copy
from rasterio.io import MemoryFile
from rasterio.crs import CRS
import numpy as np
from typing import Dict, Any, List, Tuple
from skimage import filters, feature, morphology, segmentation
from scipy import ndimage
import json


def convert_to_cog(input_path: str, output_path: str | None = None) -> str:
    """
    Convert a GeoTIFF file to Cloud Optimized GeoTIFF (COG)
    
    Args:
        input_path: Path to input GeoTIFF file
        output_path: Optional output path. If None, will be auto-generated
        
    Returns:
        Path to the created COG file
    """
    if output_path is None:
        base, ext = os.path.splitext(input_path)
        output_path = f"{base}_cog.tif"
    
    with rasterio.open(input_path) as src:
        # Create COG profile
        profile = src.profile.copy()
        profile.update({
            'driver': 'GTiff',
            'tiled': True,
            'blockxsize': 512,
            'blockysize': 512,
            'compress': 'lzw',
            'interleave': 'band',
            'photometric': 'MINISBLACK',
        })
        
        # Add overviews
        with rasterio.open(
            output_path,
            'w',
            **profile
        ) as dst:
            # Copy data
            for i in range(1, src.count + 1):
                data = src.read(i)
                dst.write(data, i)
            
            # Build overviews
            overviews = [2, 4, 8, 16]
            dst.build_overviews(overviews, Resampling.average)
            dst.update_tags(ns='rio_overview', resampling='average')
    
    return output_path


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
                    bounds.top
                )
            else:
                # Already in WGS84
                minx, miny, maxx, maxy = bounds.left, bounds.bottom, bounds.right, bounds.top
            
            return {
                'minx': minx,
                'miny': miny,
                'maxx': maxx,
                'maxy': maxy
            }
    except Exception as e:
        import logging
        logger = logging.getLogger(__name__)
        logger.error(f"Error extracting bbox from {geotiff_path}: {str(e)}")
        return None


def run_pipeline_analysis(cog_path: str, satellite_image) -> Dict[str, Dict[str, Any]]:
    """
    Run comprehensive pipeline monitoring analysis on satellite imagery
    
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
                # RGB image
                red = src.read(1)
                green = src.read(2)
                blue = src.read(3)
                rgb_image = np.stack([red, green, blue], axis=0)
            elif src.count == 1:
                # Grayscale image
                rgb_image = np.stack([src.read(1)] * 3, axis=0)
            else:
                # Use first band as grayscale
                rgb_image = np.stack([src.read(1)] * 3, axis=0)
            
            # Normalize image
            image_2d = rgb_image[0] if rgb_image.shape[0] == 1 else np.mean(rgb_image, axis=0)
            image_normalized = (image_2d - image_2d.min()) / (image_2d.max() - image_2d.min() + 1e-8)
            
            # Run different analysis types
            results['oil_leak'] = detect_oil_leaks(image_normalized, src)
            results['integrity'] = detect_pipeline_integrity(image_normalized, src)
            results['thermal'] = detect_thermal_anomalies(image_normalized, src)
            results['environmental'] = assess_environmental_compliance(image_normalized, src)
            results['security'] = detect_security_threats(image_normalized, src)
            results['sar'] = analyze_sar_features(image_normalized, src) if satellite_image.image_type == 'sar' else {}
            results['vegetation'] = detect_vegetation_encroachment(image_normalized, src)
            results['corrosion'] = detect_corrosion(image_normalized, src)
            results['ground_subsidence'] = detect_ground_subsidence(image_normalized, src)
            
    except Exception as e:
        # Return empty results on error
        for analysis_type in [
            'oil_leak', 'integrity', 'thermal', 'environmental',
            'security', 'sar', 'vegetation', 'corrosion', 'ground_subsidence'
        ]:
            results[analysis_type] = {
                'confidence_score': 0.0,
                'severity': 'low',
                'results': {},
                'metadata': {'error': str(e)},
                'processing_time': 0,
                'anomalies': []
            }
    
    return results


def detect_oil_leaks(image: np.ndarray, src: rasterio.DatasetReader) -> Dict[str, Any]:
    """Detect oil leaks in pipeline areas"""
    start_time = __import__('time').time()
    
    # Use edge detection and thresholding to find dark spots (potential leaks)
    edges = feature.canny(image, sigma=1.0)
    
    # Apply morphological operations to identify potential leak areas
    kernel = morphology.disk(3)
    closed = morphology.closing(edges, kernel)
    
    # Find connected components (potential leaks)
    labeled = ndimage.label(closed)[0]
    regions = ndimage.find_objects(labeled)
    
    anomalies = []
    total_anomaly_area = 0
    
    for i, region in enumerate(regions):
        if region is not None:
            y_coords, x_coords = region
            area = (y_coords.stop - y_coords.start) * (x_coords.stop - x_coords.start)
            
            if area > 100:  # Minimum area threshold
                # Calculate centroid
                y_center = (y_coords.start + y_coords.stop) / 2
                x_center = (x_coords.start + x_coords.stop) / 2
                
                # Convert pixel coordinates to geographic coordinates
                lon, lat = rasterio.transform.xy(src.transform, y_center, x_center)
                
                # Calculate confidence based on area and intensity
                region_intensity = np.mean(image[y_coords, x_coords])
                confidence = min(0.95, 0.5 + (area / 1000) * 0.1 + (1 - region_intensity) * 0.3)
                
                # Determine severity
                if area > 10000:
                    severity = 'critical'
                elif area > 5000:
                    severity = 'high'
                elif area > 1000:
                    severity = 'medium'
                else:
                    severity = 'low'
                
                anomalies.append({
                    'type': 'oil_leak',
                    'severity': severity,
                    'location': [float(lat), float(lon)],
                    'area_m2': float(area * 0.1),  # Approximate conversion
                    'confidence': float(confidence),
                    'description': f'Potential oil leak detected (area: {area:.0f} pixels)',
                    'metadata': {'pixel_area': int(area)}
                })
                total_anomaly_area += area
    
    confidence_score = min(0.95, 0.3 + (len(anomalies) * 0.1) + (total_anomaly_area / 10000))
    overall_severity = 'critical' if len(anomalies) > 5 or total_anomaly_area > 50000 else \
                      'high' if len(anomalies) > 2 or total_anomaly_area > 20000 else \
                      'medium' if len(anomalies) > 0 else 'low'
    
    processing_time = __import__('time').time() - start_time
    
    return {
        'confidence_score': float(confidence_score),
        'severity': overall_severity,
        'results': {
            'anomaly_count': len(anomalies),
            'total_anomaly_area_pixels': int(total_anomaly_area),
        },
        'metadata': {
            'detection_method': 'edge_detection_morphology',
        },
        'processing_time': processing_time,
        'anomalies': anomalies
    }


def detect_pipeline_integrity(image: np.ndarray, src: rasterio.DatasetReader) -> Dict[str, Any]:
    """Detect pipeline integrity issues"""
    start_time = __import__('time').time()
    
    # Use texture analysis and structural patterns
    # Detect linear features (pipeline routes)
    edges = feature.canny(image, sigma=2.0)
    
    # Analyze continuity of linear features
    labeled = ndimage.label(edges)[0]
    regions = ndimage.find_objects(labeled)
    
    anomalies = []
    discontinuity_count = 0
    
    # Detect breaks or irregularities in pipeline structure
    for region in regions:
        if region is not None:
            y_coords, x_coords = region
            width = max(x_coords.stop - x_coords.start, y_coords.stop - y_coords.start)
            height = min(x_coords.stop - x_coords.start, y_coords.stop - y_coords.start)
            
            # Check for structural anomalies (irregular shapes)
            aspect_ratio = width / (height + 1e-8)
            if aspect_ratio > 10 or aspect_ratio < 0.1:  # Very elongated or very short
                discontinuity_count += 1
                
                y_center = (y_coords.start + y_coords.stop) / 2
                x_center = (x_coords.start + x_coords.stop) / 2
                lon, lat = rasterio.transform.xy(src.transform, y_center, x_center)
                
                severity = 'high' if discontinuity_count > 5 else 'medium' if discontinuity_count > 2 else 'low'
                
                anomalies.append({
                    'type': 'structural_damage',
                    'severity': severity,
                    'location': [float(lat), float(lon)],
                    'area_m2': None,
                    'confidence': float(0.6 + min(0.3, discontinuity_count * 0.05)),
                    'description': 'Potential pipeline integrity issue detected',
                    'metadata': {'aspect_ratio': float(aspect_ratio)}
                })
    
    confidence_score = min(0.9, 0.4 + (discontinuity_count * 0.1))
    overall_severity = 'high' if discontinuity_count > 5 else 'medium' if discontinuity_count > 0 else 'low'
    
    processing_time = __import__('time').time() - start_time
    
    return {
        'confidence_score': float(confidence_score),
        'severity': overall_severity,
        'results': {
            'discontinuity_count': discontinuity_count,
            'integrity_score': max(0.0, 1.0 - (discontinuity_count / 10.0)),
        },
        'metadata': {
            'detection_method': 'structural_analysis',
        },
        'processing_time': processing_time,
        'anomalies': anomalies
    }


def detect_thermal_anomalies(image: np.ndarray, src: rasterio.DatasetReader) -> Dict[str, Any]:
    """Detect thermal anomalies"""
    start_time = __import__('time').time()
    
    # Use statistical analysis to find thermal anomalies
    mean_temp = np.mean(image)
    std_temp = np.std(image)
    threshold_high = mean_temp + 2 * std_temp
    threshold_low = mean_temp - 2 * std_temp
    
    # Find hot and cold spots
    hot_spots = image > threshold_high
    cold_spots = image < threshold_low
    
    anomalies = []
    
    # Analyze hot spots
    labeled_hot = ndimage.label(hot_spots)[0]
    regions_hot = ndimage.find_objects(labeled_hot)
    
    for i, region in enumerate(regions_hot):
        if region is not None:
            y_coords, x_coords = region
            area = (y_coords.stop - y_coords.start) * (x_coords.stop - x_coords.start)
            
            if area > 50:
                y_center = (y_coords.start + y_coords.stop) / 2
                x_center = (x_coords.start + x_coords.stop) / 2
                lon, lat = rasterio.transform.xy(src.transform, y_center, x_center)
                
                intensity = np.mean(image[y_coords, x_coords])
                severity = 'critical' if intensity > threshold_high + std_temp else 'high' if intensity > threshold_high else 'medium'
                
                anomalies.append({
                    'type': 'thermal_anomaly',
                    'severity': severity,
                    'location': [float(lat), float(lon)],
                    'area_m2': float(area * 0.1),
                    'confidence': float(0.7 + min(0.2, (intensity - mean_temp) / std_temp * 0.1)),
                    'description': f'Thermal anomaly detected (intensity: {intensity:.3f})',
                    'metadata': {'intensity': float(intensity), 'threshold': float(threshold_high)}
                })
    
    confidence_score = min(0.95, 0.5 + (len(anomalies) * 0.1))
    overall_severity = 'critical' if len(anomalies) > 3 else 'high' if len(anomalies) > 1 else 'medium' if len(anomalies) > 0 else 'low'
    
    processing_time = __import__('time').time() - start_time
    
    return {
        'confidence_score': float(confidence_score),
        'severity': overall_severity,
        'results': {
            'anomaly_count': len(anomalies),
            'mean_temperature': float(mean_temp),
            'std_temperature': float(std_temp),
        },
        'metadata': {
            'detection_method': 'statistical_threshold',
        },
        'processing_time': processing_time,
        'anomalies': anomalies
    }


def assess_environmental_compliance(image: np.ndarray, src: rasterio.DatasetReader) -> Dict[str, Any]:
    """Assess environmental compliance"""
    start_time = __import__('time').time()
    
    # Analyze vegetation health, water quality indicators, etc.
    # For simplicity, using NDVI-like analysis if multiple bands available
    anomalies = []
    
    # Basic environmental indicators
    mean_intensity = np.mean(image)
    std_intensity = np.std(image)
    
    # Detect potential environmental issues
    if std_intensity > 0.3:  # High variability might indicate environmental stress
        anomalies.append({
            'type': 'vegetation_encroachment',
            'severity': 'medium',
            'location': [0.0, 0.0],  # General area
            'area_m2': None,
            'confidence': 0.6,
            'description': 'Potential environmental compliance issue',
            'metadata': {'variability': float(std_intensity)}
        })
    
    confidence_score = 0.5
    overall_severity = 'medium' if len(anomalies) > 0 else 'low'
    
    processing_time = __import__('time').time() - start_time
    
    return {
        'confidence_score': float(confidence_score),
        'severity': overall_severity,
        'results': {
            'compliance_score': float(1.0 - min(1.0, len(anomalies) * 0.2)),
            'environmental_indicators': {
                'mean_intensity': float(mean_intensity),
                'variability': float(std_intensity),
            }
        },
        'metadata': {
            'detection_method': 'environmental_indicators',
        },
        'processing_time': processing_time,
        'anomalies': anomalies
    }


def detect_security_threats(image: np.ndarray, src: rasterio.DatasetReader) -> Dict[str, Any]:
    """Detect security threats and unauthorized activity"""
    start_time = __import__('time').time()
    
    # Detect unusual patterns, vehicles, or structures
    # Use edge detection and pattern analysis
    edges = feature.canny(image, sigma=1.5)
    labeled = ndimage.label(edges)[0]
    regions = ndimage.find_objects(labeled)
    
    anomalies = []
    unusual_patterns = 0
    
    for region in regions:
        if region is not None:
            y_coords, x_coords = region
            area = (y_coords.stop - y_coords.start) * (x_coords.stop - x_coords.start)
            
            # Detect rectangular structures (potential unauthorized facilities)
            if 500 < area < 10000:
                y_center = (y_coords.start + y_coords.stop) / 2
                x_center = (x_coords.start + x_coords.stop) / 2
                lon, lat = rasterio.transform.xy(src.transform, y_center, x_center)
                
                unusual_patterns += 1
                severity = 'high' if unusual_patterns > 3 else 'medium'
                
                anomalies.append({
                    'type': 'unauthorized_activity',
                    'severity': severity,
                    'location': [float(lat), float(lon)],
                    'area_m2': float(area * 0.1),
                    'confidence': float(0.5 + min(0.3, unusual_patterns * 0.1)),
                    'description': 'Potential unauthorized activity detected',
                    'metadata': {'pattern_area': int(area)}
                })
    
    confidence_score = min(0.8, 0.4 + (len(anomalies) * 0.1))
    overall_severity = 'high' if len(anomalies) > 3 else 'medium' if len(anomalies) > 0 else 'low'
    
    processing_time = __import__('time').time() - start_time
    
    return {
        'confidence_score': float(confidence_score),
        'severity': overall_severity,
        'results': {
            'threat_count': len(anomalies),
            'security_score': max(0.0, 1.0 - (len(anomalies) / 10.0)),
        },
        'metadata': {
            'detection_method': 'pattern_analysis',
        },
        'processing_time': processing_time,
        'anomalies': anomalies
    }


def analyze_sar_features(image: np.ndarray, src: rasterio.DatasetReader) -> Dict[str, Any]:
    """Analyze SAR (Synthetic Aperture Radar) specific features"""
    start_time = __import__('time').time()
    
    # SAR-specific analysis (backscatter, coherence, etc.)
    anomalies = []
    
    # Analyze backscatter patterns
    mean_backscatter = np.mean(image)
    std_backscatter = np.std(image)
    
    # Detect anomalies in SAR data
    if std_backscatter > 0.2:
        anomalies.append({
            'type': 'thermal_anomaly',
            'severity': 'medium',
            'location': [0.0, 0.0],
            'area_m2': None,
            'confidence': 0.6,
            'description': 'SAR anomaly detected',
            'metadata': {'backscatter_variance': float(std_backscatter)}
        })
    
    confidence_score = 0.6
    overall_severity = 'medium' if len(anomalies) > 0 else 'low'
    
    processing_time = __import__('time').time() - start_time
    
    return {
        'confidence_score': float(confidence_score),
        'severity': overall_severity,
        'results': {
            'mean_backscatter': float(mean_backscatter),
            'backscatter_variance': float(std_backscatter),
        },
        'metadata': {
            'detection_method': 'sar_analysis',
        },
        'processing_time': processing_time,
        'anomalies': anomalies
    }


def detect_vegetation_encroachment(image: np.ndarray, src: rasterio.DatasetReader) -> Dict[str, Any]:
    """Detect vegetation encroachment near pipelines"""
    start_time = __import__('time').time()
    
    # Similar to environmental compliance but focused on vegetation
    anomalies = []
    
    # Use texture analysis to detect vegetation
    edges = feature.canny(image, sigma=2.0)
    labeled = ndimage.label(edges)[0]
    regions = ndimage.find_objects(labeled)
    
    vegetation_areas = 0
    
    for region in regions:
        if region is not None:
            y_coords, x_coords = region
            area = (y_coords.stop - y_coords.start) * (x_coords.stop - x_coords.start)
            
            if area > 1000:  # Large connected areas might indicate vegetation
                vegetation_areas += 1
                y_center = (y_coords.start + y_coords.stop) / 2
                x_center = (x_coords.start + x_coords.stop) / 2
                lon, lat = rasterio.transform.xy(src.transform, y_center, x_center)
                
                severity = 'high' if vegetation_areas > 5 else 'medium'
                
                anomalies.append({
                    'type': 'vegetation_encroachment',
                    'severity': severity,
                    'location': [float(lat), float(lon)],
                    'area_m2': float(area * 0.1),
                    'confidence': float(0.6 + min(0.2, vegetation_areas * 0.05)),
                    'description': 'Potential vegetation encroachment detected',
                    'metadata': {'vegetation_area': int(area)}
                })
    
    confidence_score = min(0.85, 0.5 + (len(anomalies) * 0.1))
    overall_severity = 'high' if len(anomalies) > 5 else 'medium' if len(anomalies) > 0 else 'low'
    
    processing_time = __import__('time').time() - start_time
    
    return {
        'confidence_score': float(confidence_score),
        'severity': overall_severity,
        'results': {
            'encroachment_areas': len(anomalies),
            'total_vegetation_area': sum(a.get('area_m2', 0) or 0 for a in anomalies),
        },
        'metadata': {
            'detection_method': 'texture_analysis',
        },
        'processing_time': processing_time,
        'anomalies': anomalies
    }


def detect_corrosion(image: np.ndarray, src: rasterio.DatasetReader) -> Dict[str, Any]:
    """Detect pipeline corrosion"""
    start_time = __import__('time').time()
    
    # Detect corrosion patterns (discoloration, texture changes)
    anomalies = []
    
    # Use edge and texture analysis
    edges = feature.canny(image, sigma=1.5)
    labeled = ndimage.label(edges)[0]
    regions = ndimage.find_objects(labeled)
    
    corrosion_spots = 0
    
    for region in regions:
        if region is not None:
            y_coords, x_coords = region
            area = (y_coords.stop - y_coords.start) * (x_coords.stop - x_coords.start)
            
            # Small irregular patches might indicate corrosion
            if 10 < area < 500:
                y_center = (y_coords.start + y_coords.stop) / 2
                x_center = (x_coords.start + x_coords.stop) / 2
                lon, lat = rasterio.transform.xy(src.transform, y_center, x_center)
                
                corrosion_spots += 1
                severity = 'high' if corrosion_spots > 10 else 'medium' if corrosion_spots > 5 else 'low'
                
                anomalies.append({
                    'type': 'corrosion',
                    'severity': severity,
                    'location': [float(lat), float(lon)],
                    'area_m2': float(area * 0.1),
                    'confidence': float(0.5 + min(0.3, corrosion_spots * 0.02)),
                    'description': 'Potential corrosion detected',
                    'metadata': {'spot_area': int(area)}
                })
    
    confidence_score = min(0.8, 0.4 + (len(anomalies) * 0.05))
    overall_severity = 'high' if len(anomalies) > 10 else 'medium' if len(anomalies) > 5 else 'low'
    
    processing_time = __import__('time').time() - start_time
    
    return {
        'confidence_score': float(confidence_score),
        'severity': overall_severity,
        'results': {
            'corrosion_spots': len(anomalies),
            'corrosion_index': min(1.0, len(anomalies) / 20.0),
        },
        'metadata': {
            'detection_method': 'texture_pattern_analysis',
        },
        'processing_time': processing_time,
        'anomalies': anomalies
    }


def detect_ground_subsidence(image: np.ndarray, src: rasterio.DatasetReader) -> Dict[str, Any]:
    """Detect ground subsidence"""
    start_time = __import__('time').time()
    
    # Detect elevation changes, depressions
    anomalies = []
    
    # Use morphological operations to detect depressions
    kernel = morphology.disk(5)
    closed = morphology.closing(image, kernel)
    opened = morphology.opening(image, kernel)
    
    # Find depressions (areas lower than surroundings)
    depression = closed - opened
    threshold = np.percentile(depression, 90)
    depressions = depression > threshold
    
    labeled = ndimage.label(depressions)[0]
    regions = ndimage.find_objects(labeled)
    
    subsidence_areas = 0
    
    for region in regions:
        if region is not None:
            y_coords, x_coords = region
            area = (y_coords.stop - y_coords.start) * (x_coords.stop - x_coords.start)
            
            if area > 500:
                y_center = (y_coords.start + y_coords.stop) / 2
                x_center = (x_coords.start + x_coords.stop) / 2
                lon, lat = rasterio.transform.xy(src.transform, y_center, x_center)
                
                subsidence_areas += 1
                severity = 'critical' if subsidence_areas > 3 or area > 10000 else 'high' if subsidence_areas > 1 else 'medium'
                
                anomalies.append({
                    'type': 'ground_subsidence',
                    'severity': severity,
                    'location': [float(lat), float(lon)],
                    'area_m2': float(area * 0.1),
                    'confidence': float(0.6 + min(0.3, subsidence_areas * 0.1)),
                    'description': f'Potential ground subsidence detected (area: {area:.0f} pixels)',
                    'metadata': {'subsidence_area': int(area)}
                })
    
    confidence_score = min(0.9, 0.5 + (len(anomalies) * 0.15))
    overall_severity = 'critical' if len(anomalies) > 3 else 'high' if len(anomalies) > 0 else 'low'
    
    processing_time = __import__('time').time() - start_time
    
    return {
        'confidence_score': float(confidence_score),
        'severity': overall_severity,
        'results': {
            'subsidence_areas': len(anomalies),
            'total_subsidence_area': sum(a.get('area_m2', 0) or 0 for a in anomalies),
        },
        'metadata': {
            'detection_method': 'morphological_depression_analysis',
        },
        'processing_time': processing_time,
        'anomalies': anomalies
    }


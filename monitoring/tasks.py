from celery import shared_task
from django.core.mail import send_mail
from django.conf import settings
from django.utils import timezone
import os
import json
from typing import Dict, Any
from .models import SatelliteImage, Analysis, Anomaly, Notification
from .utils import convert_to_cog, extract_geotiff_bbox, run_pipeline_analysis
from pathlib import Path


@shared_task(bind=True, max_retries=3)
def convert_to_cog_task(self, image_id: str):
    try:
        image = SatelliteImage.objects.get(id=image_id)
        image.conversion_status = "processing"
        image.save()

        # Now returns Path
        cog_path = convert_to_cog(image.original_tiff.path)

        # SAFE: Use Path methods
        media_root = Path(settings.MEDIA_ROOT)
        try:
            relative_path = cog_path.relative_to(media_root)
            image.cog_tiff.name = (
                relative_path.as_posix()
            )  # e.g., "images/image_cog.tif"
        except ValueError:
            # Fallback: if not under MEDIA_ROOT
            image.cog_tiff.name = cog_path.name

        image.is_cog_converted = True
        image.conversion_status = "completed"

        # extract_geotiff_bbox expects str
        bbox = extract_geotiff_bbox(str(cog_path))
        if bbox:
            image.bbox_minx = bbox["minx"]
            image.bbox_miny = bbox["miny"]
            image.bbox_maxx = bbox["maxx"]
            image.bbox_maxy = bbox["maxy"]

        image.save()
        return {"status": "success", "image_id": str(image_id)}

    except SatelliteImage.DoesNotExist:
        return {"status": "error", "message": "Image not found"}
    except Exception as exc:
        try:
            image = SatelliteImage.objects.get(id=image_id)
            image.conversion_status = "failed"
            image.save()
        except:
            pass
        raise self.retry(exc=exc, countdown=60)


@shared_task(bind=True, max_retries=3)
def run_analysis_task(self, image_id: str):
    """Run comprehensive SAR-based pipeline monitoring analysis"""
    try:
        image = SatelliteImage.objects.get(id=image_id)
        if not image.is_cog_converted:
            return {
                "status": "error",
                "message": "Image must be converted to COG first",
            }

        start_time = timezone.now()

        # Run analysis
        results = run_pipeline_analysis(image.cog_tiff.path, image)

        # Create Analysis objects for each analysis type
        analysis_objects = []
        for analysis_type, result_data in results.items():
            # Skip empty results
            if not result_data or result_data.get("confidence_score", 0) == 0:
                continue

            analysis = Analysis.objects.create(
                user=image.user,
                satellite_image=image,
                pipeline=image.pipeline,
                analysis_type=analysis_type,
                status="completed",
                confidence_score=result_data.get("confidence_score"),
                severity=result_data.get("severity"),
                results_json=result_data.get("results", {}),
                metadata=result_data.get("metadata", {}),
                processing_time_seconds=result_data.get("processing_time", 0),
            )
            analysis_objects.append(analysis)

            # Create anomalies if detected
            if result_data.get("anomalies"):
                for anomaly_data in result_data["anomalies"]:
                    anomaly = Anomaly.objects.create(
                        analysis=analysis,
                        user=image.user,
                        anomaly_type=anomaly_data["type"],
                        severity=anomaly_data["severity"],
                        location_lat=anomaly_data["location"][0],
                        location_lon=anomaly_data["location"][1],
                        area_m2=anomaly_data.get("area_m2"),
                        description=anomaly_data.get("description", ""),
                        confidence_score=anomaly_data["confidence"],
                        metadata=anomaly_data.get("metadata", {}),
                    )

                    # Create notification for critical and high severity anomalies
                    if anomaly.severity in ["critical", "high"]:
                        Notification.objects.create(
                            user=image.user,
                            anomaly=anomaly,
                            notification_type="both",
                            title=f"{anomaly.get_anomaly_type_display()} Detected",
                            message=(
                                f"A {anomaly.get_severity_display().lower()} severity "
                                f"{anomaly.get_anomaly_type_display()} has been detected "
                                f"at location ({anomaly.location_lat:.6f}, {anomaly.location_lon:.6f}). "
                                f"Analysis type: {analysis.get_analysis_type_display()}."
                            ),
                            metadata={
                                "anomaly_id": str(anomaly.id),
                                "analysis_id": str(analysis.id),
                                "analysis_type": analysis_type,
                            },
                        )

        # Send email notification if critical anomalies found
        critical_anomalies = Anomaly.objects.filter(
            analysis__in=analysis_objects, severity="critical"
        )
        if critical_anomalies.exists():
            send_analysis_notification_email.delay(
                str(image.user.id), len(critical_anomalies)
            )

        processing_time = (timezone.now() - start_time).total_seconds()

        return {
            "status": "success",
            "image_id": str(image_id),
            "analyses_created": len(analysis_objects),
            "total_anomalies": sum(a.anomalies.count() for a in analysis_objects),
            "processing_time": processing_time,
        }

    except SatelliteImage.DoesNotExist:
        return {"status": "error", "message": "Image not found"}
    except Exception as exc:
        # Update analysis status to failed
        Analysis.objects.filter(
            satellite_image_id=image_id, status="processing"
        ).update(status="failed", error_message=str(exc))
        raise self.retry(exc=exc, countdown=120)


@shared_task
def send_analysis_notification_email(user_id: str, anomaly_count: int):
    """Send email notification about analysis results"""
    try:
        from django.contrib.auth.models import User

        user = User.objects.get(id=user_id)

        subject = f"Critical Anomalies Detected - {anomaly_count} Alert(s)"
        message = (
            f"Dear {user.username},\n\n"
            f"Your SAR-based pipeline monitoring analysis has completed and detected "
            f"{anomaly_count} critical anomaly/anomalies.\n\n"
            f"The analysis includes advanced techniques such as:\n"
            f"- InSAR and DInSAR for displacement detection\n"
            f"- Polarimetric decomposition for scattering analysis\n"
            f"- GLCM texture analysis\n"
            f"- Coherence change detection\n"
            f"- Deep learning-based anomaly detection\n\n"
            f"Please log in to your dashboard to review the details.\n\n"
            f"Best regards,\n"
            f"Pipeline Monitoring System"
        )

        send_mail(
            subject,
            message,
            settings.DEFAULT_FROM_EMAIL,
            [user.email],
            fail_silently=False,
        )
        return {"status": "success", "user_id": user_id}
    except Exception as exc:
        return {"status": "error", "message": str(exc)}


@shared_task
def send_push_notification(user_id: str, notification_id: str):
    """Send push notification to user devices"""
    try:
        from django.contrib.auth.models import User
        from .models import UserDevice
        from django.conf import settings

        notification = Notification.objects.get(id=notification_id)
        devices = UserDevice.objects.filter(user_id=user_id, is_active=True)

        # In a real implementation, you would use Firebase Cloud Messaging
        # or similar service here
        # For now, we'll just mark it as sent
        notification.is_sent = True
        notification.sent_at = timezone.now()
        notification.save()

        return {"status": "success", "devices_notified": devices.count()}
    except Exception as exc:
        return {"status": "error", "message": str(exc)}


@shared_task
def batch_analyze_images(image_ids: list):
    """Run analysis on multiple images in batch"""
    results = []
    for image_id in image_ids:
        try:
            result = run_analysis_task.delay(image_id)
            results.append(
                {"image_id": image_id, "task_id": result.id, "status": "queued"}
            )
        except Exception as e:
            results.append({"image_id": image_id, "status": "error", "message": str(e)})

    return {"total_images": len(image_ids), "results": results}


@shared_task
def cleanup_old_notifications(days: int = 30):
    """Clean up old read notifications"""
    try:
        cutoff_date = timezone.now() - timezone.timedelta(days=days)
        deleted_count = Notification.objects.filter(
            is_read=True, read_at__lt=cutoff_date
        ).delete()[0]

        return {
            "status": "success",
            "deleted_count": deleted_count,
            "cutoff_date": cutoff_date.isoformat(),
        }
    except Exception as exc:
        return {"status": "error", "message": str(exc)}


# Add this task to your existing tasks.py file

@shared_task(bind=True, max_retries=3)
def run_object_identification_task(self, image_id: str):
    """
    Run automated object identification on a satellite image
    This creates MappedObject instances for detected objects
    """
    try:
        from .models import SatelliteImage, MappedObject, LegendCategory, Analysis
        from .utils import run_object_identification_analysis
        from django.core.files.base import ContentFile
        import json
        import uuid
        
        image = SatelliteImage.objects.get(id=image_id)
        
        if not image.is_cog_converted:
            return {
                "status": "error",
                "message": "Image must be converted to COG first",
            }

        start_time = timezone.now()

        # Run object identification analysis
        # This should return detected objects with their geometries
        identification_results = run_object_identification_analysis(
            image.cog_tiff.path, 
            image
        )

        if not identification_results or "detected_objects" not in identification_results:
            return {
                "status": "success",
                "message": "No objects detected",
                "objects_created": 0,
            }

        objects_created = 0
        
        # Get or create default legend categories for different object types
        legend_categories = {}
        object_type_colors = {
            "oil_spill": ("#FF0000", "🛢️"),
            "encroachment": ("#FF6600", "⚠️"),
            "building": ("#666666", "🏢"),
            "vehicle": ("#0066FF", "🚗"),
            "infrastructure": ("#9933FF", "🏗️"),
            "vegetation": ("#00CC00", "🌳"),
            "water_body": ("#0099CC", "💧"),
            "unknown": ("#CCCCCC", "❓"),
        }

        for obj_type, (color, icon) in object_type_colors.items():
            category, created = LegendCategory.objects.get_or_create(
                user=image.user,
                name=f"AI Detected {obj_type.replace('_', ' ').title()}",
                defaults={
                    "color": color,
                    "icon": icon,
                    "category_type": obj_type,
                    "description": f"Automatically detected {obj_type.replace('_', ' ')} objects",
                }
            )
            legend_categories[obj_type] = category

        # Create MappedObject instances for each detected object
        for detected_obj in identification_results["detected_objects"]:
            try:
                obj_type = detected_obj.get("type", "unknown")
                geojson_data = detected_obj.get("geojson")
                confidence = detected_obj.get("confidence", 0.0)
                
                if not geojson_data:
                    continue

                # Create a unique name for the object
                obj_name = f"AI_{obj_type}_{uuid.uuid4().hex[:8]}"

                # Create GeoJSON file content
                geojson_content = json.dumps(geojson_data, indent=2)
                geojson_file = ContentFile(
                    geojson_content.encode('utf-8'),
                    name=f"{obj_name}.geojson"
                )

                # Create MappedObject
                mapped_obj = MappedObject.objects.create(
                    user=image.user,
                    satellite_image=image,
                    pipeline=image.pipeline,
                    name=obj_name,
                    description=f"Automatically identified {obj_type.replace('_', ' ')} with {confidence:.1%} confidence",
                    geojson_file=geojson_file,
                    legend_category=legend_categories.get(obj_type),
                    object_type=obj_type,
                    confidence_score=confidence,
                    identified_by="ai",
                    metadata={
                        "detection_method": "deep_learning",
                        "analysis_date": timezone.now().isoformat(),
                        "model_confidence": confidence,
                    }
                )

                objects_created += 1

            except Exception as e:
                import logging
                logger = logging.getLogger(__name__)
                logger.error(f"Error creating mapped object: {str(e)}")
                continue

        # Create or update the object_identification analysis record
        processing_time = (timezone.now() - start_time).total_seconds()
        
        analysis, created = Analysis.objects.get_or_create(
            user=image.user,
            satellite_image=image,
            pipeline=image.pipeline,
            analysis_type="object_identification",
            defaults={
                "status": "completed",
                "confidence_score": identification_results.get("average_confidence", 0.75),
                "severity": "low",
                "results_json": {
                    "objects_detected": objects_created,
                    "detection_summary": identification_results.get("summary", {}),
                },
                "metadata": {
                    "detection_method": "deep_learning",
                    "processing_time": processing_time,
                },
                "processing_time_seconds": processing_time,
            }
        )

        if not created:
            # Update existing analysis
            analysis.status = "completed"
            analysis.confidence_score = identification_results.get("average_confidence", 0.75)
            analysis.results_json = {
                "objects_detected": objects_created,
                "detection_summary": identification_results.get("summary", {}),
            }
            analysis.processing_time_seconds = processing_time
            analysis.save()

        return {
            "status": "success",
            "image_id": str(image_id),
            "objects_created": objects_created,
            "processing_time": processing_time,
        }

    except SatelliteImage.DoesNotExist:
        return {"status": "error", "message": "Image not found"}
    except Exception as exc:
        import logging
        logger = logging.getLogger(__name__)
        logger.error(f"Object identification failed for image {image_id}: {str(exc)}")
        raise self.retry(exc=exc, countdown=120)
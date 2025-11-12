from django.http import JsonResponse
from django.views.decorators.http import require_http_methods
from django.db import connection


@require_http_methods(["GET"])
def health_check(request):
    """
    Health check endpoint for Docker healthchecks.
    Checks database connectivity and returns service status.
    """
    try:
        # Check database connection
        with connection.cursor() as cursor:
            cursor.execute("SELECT 1")

        return JsonResponse({"status": "healthy", "database": "connected"}, status=200)
    except Exception as e:
        return JsonResponse(
            {"status": "unhealthy", "database": "disconnected", "error": str(e)},
            status=503,
        )

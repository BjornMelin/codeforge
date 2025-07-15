from fastapi import HTTPException

async def get_health_status():
    # Simulating critical service checks
    services_health = {
        'database': 'healthy',  # Simulate a database status check
        'cache': 'healthy'      # Simulate a cache status check
    }
    unhealthy_services = [service for service, status in services_health.items() if status != 'healthy']
    if unhealthy_services:
        raise HTTPException(status_code=500, detail={"status": "unhealthy", "services": unhealthy_services})
    return {"status": "healthy"}
"""
FastAPI application factory.
"""
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from loguru import logger

from config.settings import settings


def create_app() -> FastAPI:
    """Create and configure FastAPI application."""
    
    app = FastAPI(
        title=settings.APP_NAME,
        version=settings.VERSION,
        description="V3 Validated Trading Signals - Ghost Protocol",
        docs_url="/docs",
        redoc_url="/redoc",
    )
    
    # CORS middleware
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    
    # Include routers
    from api.routes import router as main_router
    from api.debug_routes import router as debug_router
    
    app.include_router(main_router, tags=["main"])
    app.include_router(debug_router, prefix="/debug", tags=["debug"])
    
    @app.on_event("startup")
    async def startup():
        logger.info(f"🚀 {settings.APP_NAME} starting - Version {settings.VERSION}")
        logger.info(f"V3 Min Confidence: {settings.V3_MIN_CONFIDENCE:.0%}")
        logger.info(f"V3 Default Hold: {settings.V3_DEFAULT_HOLD_HOURS}h")
        
        # Initialize database pool if configured
        if settings.DATABASE_URL:
            try:
                from data.database import Database
                await Database.get_pool()
                logger.info("✅ Database pool initialized")
            except Exception as e:
                logger.warning(f"Database pool initialization skipped: {e}")
    
    @app.on_event("shutdown")
    async def shutdown():
        # Close database pool
        try:
            from data.database import Database
            await Database.close_pool()
        except Exception:
            pass
        logger.info("Application shutdown complete")
    
    return app


# Create app instance
app = create_app()

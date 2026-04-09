from fastapi import APIRouter, Depends, UploadFile, File, Query, BackgroundTasks
from app.api import deps
from app.schemas.image import SearchResponse, SearchMatch, IndexResponse, StatsResponse
from app.services.search_service import SearchService
from app.services.image_service import IndexingService
from app.repository.image_repo import ImageRepository
from app.services.clip_service import CLIPService

router = APIRouter()

@router.get("/stats", response_model=StatsResponse)
async def get_stats(
    repo: ImageRepository = Depends(deps.get_image_repo)
):
    from app.core.config import settings
    count = await repo.get_total_count()
    return StatsResponse(
        total_images=count,
        project_name=settings.PROJECT_NAME,
        version=settings.VERSION
    )

@router.post("/search", response_model=SearchResponse)
async def search_image(
    file: UploadFile = File(...),
    color_weight: float = Query(0.5, ge=0.0, le=1.0),
    limit: int = Query(6, ge=1, le=20),
    repo: ImageRepository = Depends(deps.get_image_repo),
    clip: CLIPService = Depends(deps.get_clip_service)
):
    search_service = SearchService(repo, clip)
    contents = await file.read()
    results = await search_service.hybrid_search(contents, color_weight, limit)
    
    matches = [
        SearchMatch(
            filename=r[0],
            total_similarity=float(r[1]),
            pattern_score=float(r[2]),
            color_score=float(r[3])
        ) for r in results
    ]
    
    return SearchResponse(matches=matches, total_found=len(matches))

@router.post("/index", response_model=IndexResponse)
async def trigger_indexing(
    background_tasks: BackgroundTasks,
    repo: ImageRepository = Depends(deps.get_image_repo),
    clip: CLIPService = Depends(deps.get_clip_service)
):
    indexing_service = IndexingService(repo, clip)
    # We run actual indexing in background for large collections
    # For now, let's just return a message and run it.
    indexed_count = await indexing_service.synchronize_index()
    total_count = await repo.get_total_count()
    
    return IndexResponse(
        message="Indexing complete",
        indexed_count=indexed_count,
        total_in_db=total_count
    )

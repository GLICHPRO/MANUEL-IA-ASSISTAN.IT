"""
🐙 GitHub API Routes for GIDEON

Endpoints per integrare GitHub con GIDEON.
"""

from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any
from loguru import logger

from integrations.github_integration import github

router = APIRouter(prefix="/github", tags=["GitHub"])


# ==================== MODELS ====================

class CreateRepoRequest(BaseModel):
    name: str = Field(..., description="Nome del repository")
    description: str = Field("", description="Descrizione")
    private: bool = Field(False, description="Se privato")
    auto_init: bool = Field(True, description="Inizializza con README")


class CreateIssueRequest(BaseModel):
    owner: str = Field(..., description="Proprietario del repo")
    repo: str = Field(..., description="Nome del repository")
    title: str = Field(..., description="Titolo issue")
    body: str = Field("", description="Corpo issue")
    labels: List[str] = Field(default_factory=list, description="Labels")


class UpdateIssueRequest(BaseModel):
    title: Optional[str] = None
    body: Optional[str] = None
    state: Optional[str] = None  # open, closed


class CommentRequest(BaseModel):
    body: str = Field(..., description="Testo del commento")


class CreateGistRequest(BaseModel):
    filename: str = Field(..., description="Nome del file")
    content: str = Field(..., description="Contenuto del file")
    description: str = Field("", description="Descrizione gist")
    public: bool = Field(False, description="Se pubblico")


class CreatePRRequest(BaseModel):
    owner: str
    repo: str
    title: str
    head: str  # branch sorgente
    base: str  # branch destinazione (es: main)
    body: str = ""


# ==================== STATUS ====================

@router.get("/status")
async def github_status():
    """
    📊 Stato dell'integrazione GitHub
    
    Verifica se il token è configurato e funzionante.
    """
    status = await github.get_status()
    return status


@router.get("/rate-limit")
async def get_rate_limit():
    """
    ⏱️ Verifica rate limit GitHub API
    """
    return await github.get_rate_limit()


# ==================== USER ====================

@router.get("/user")
async def get_current_user():
    """
    👤 Ottieni info utente autenticato
    """
    if not github.is_configured:
        raise HTTPException(status_code=401, detail="GitHub token non configurato")
    return await github.get_user()


@router.get("/user/repos")
async def get_user_repos(
    per_page: int = Query(30, le=100),
    page: int = Query(1, ge=1)
):
    """
    📚 Lista repository dell'utente
    """
    if not github.is_configured:
        raise HTTPException(status_code=401, detail="GitHub token non configurato")
    return await github.get_user_repos(per_page, page)


@router.get("/user/orgs")
async def get_user_organizations():
    """
    🏢 Lista organizzazioni dell'utente
    """
    if not github.is_configured:
        raise HTTPException(status_code=401, detail="GitHub token non configurato")
    return await github.get_user_orgs()


# ==================== REPOSITORIES ====================

@router.get("/repos/{owner}/{repo}")
async def get_repository(owner: str, repo: str):
    """
    📦 Dettagli di un repository
    """
    return await github.get_repo(owner, repo)


@router.post("/repos/create")
async def create_repository(request: CreateRepoRequest):
    """
    ➕ Crea nuovo repository
    """
    if not github.is_configured:
        raise HTTPException(status_code=401, detail="GitHub token non configurato")
    
    result = await github.create_repo(
        name=request.name,
        description=request.description,
        private=request.private,
        auto_init=request.auto_init
    )
    
    if "error" in result:
        raise HTTPException(status_code=result.get("status", 400), detail=result["error"])
    
    return {"success": True, "repo": result}


@router.delete("/repos/{owner}/{repo}")
async def delete_repository(owner: str, repo: str):
    """
    🗑️ Elimina repository (richiede permesso delete_repo nel token)
    """
    if not github.is_configured:
        raise HTTPException(status_code=401, detail="GitHub token non configurato")
    
    result = await github.delete_repo(owner, repo)
    if "error" in result:
        raise HTTPException(status_code=result.get("status", 400), detail=result["error"])
    
    return {"success": True, "message": f"Repository {owner}/{repo} eliminato"}


@router.get("/repos/{owner}/{repo}/contents")
async def get_repo_contents(
    owner: str, 
    repo: str, 
    path: str = "",
    ref: str = None
):
    """
    📄 Contenuti di un path nel repository
    """
    return await github.get_repo_contents(owner, repo, path, ref)


@router.get("/repos/{owner}/{repo}/languages")
async def get_repo_languages(owner: str, repo: str):
    """
    💻 Linguaggi usati nel repository
    """
    return await github.get_repo_languages(owner, repo)


# ==================== ISSUES ====================

@router.get("/repos/{owner}/{repo}/issues")
async def list_issues(
    owner: str, 
    repo: str,
    state: str = Query("open", enum=["open", "closed", "all"]),
    per_page: int = Query(30, le=100)
):
    """
    📋 Lista issues di un repository
    """
    return await github.list_issues(owner, repo, state, per_page)


@router.post("/repos/{owner}/{repo}/issues")
async def create_issue(owner: str, repo: str, request: CreateIssueRequest):
    """
    ➕ Crea nuova issue
    """
    if not github.is_configured:
        raise HTTPException(status_code=401, detail="GitHub token non configurato")
    
    result = await github.create_issue(
        owner=owner,
        repo=repo,
        title=request.title,
        body=request.body,
        labels=request.labels
    )
    
    if "error" in result:
        raise HTTPException(status_code=result.get("status", 400), detail=result["error"])
    
    return {"success": True, "issue": result}


@router.patch("/repos/{owner}/{repo}/issues/{issue_number}")
async def update_issue(
    owner: str, 
    repo: str, 
    issue_number: int,
    request: UpdateIssueRequest
):
    """
    ✏️ Aggiorna issue esistente
    """
    if not github.is_configured:
        raise HTTPException(status_code=401, detail="GitHub token non configurato")
    
    result = await github.update_issue(
        owner, repo, issue_number,
        title=request.title,
        body=request.body,
        state=request.state
    )
    
    if "error" in result:
        raise HTTPException(status_code=result.get("status", 400), detail=result["error"])
    
    return {"success": True, "issue": result}


@router.post("/repos/{owner}/{repo}/issues/{issue_number}/comments")
async def comment_on_issue(
    owner: str, 
    repo: str, 
    issue_number: int,
    request: CommentRequest
):
    """
    💬 Aggiungi commento a issue
    """
    if not github.is_configured:
        raise HTTPException(status_code=401, detail="GitHub token non configurato")
    
    result = await github.comment_issue(owner, repo, issue_number, request.body)
    
    if "error" in result:
        raise HTTPException(status_code=result.get("status", 400), detail=result["error"])
    
    return {"success": True, "comment": result}


# ==================== COMMITS ====================

@router.get("/repos/{owner}/{repo}/commits")
async def list_commits(
    owner: str, 
    repo: str,
    per_page: int = Query(30, le=100),
    sha: str = None
):
    """
    📝 Lista commits di un repository
    """
    return await github.list_commits(owner, repo, per_page, sha)


@router.get("/repos/{owner}/{repo}/commits/{sha}")
async def get_commit(owner: str, repo: str, sha: str):
    """
    🔍 Dettagli di un commit specifico
    """
    return await github.get_commit(owner, repo, sha)


# ==================== PULL REQUESTS ====================

@router.get("/repos/{owner}/{repo}/pulls")
async def list_pull_requests(
    owner: str, 
    repo: str,
    state: str = Query("open", enum=["open", "closed", "all"]),
    per_page: int = Query(30, le=100)
):
    """
    🔀 Lista pull requests
    """
    return await github.list_pull_requests(owner, repo, state, per_page)


@router.post("/repos/{owner}/{repo}/pulls")
async def create_pull_request(owner: str, repo: str, request: CreatePRRequest):
    """
    ➕ Crea pull request
    """
    if not github.is_configured:
        raise HTTPException(status_code=401, detail="GitHub token non configurato")
    
    result = await github.create_pull_request(
        owner, repo,
        title=request.title,
        head=request.head,
        base=request.base,
        body=request.body
    )
    
    if "error" in result:
        raise HTTPException(status_code=result.get("status", 400), detail=result["error"])
    
    return {"success": True, "pull_request": result}


# ==================== SEARCH ====================

@router.get("/search/repos")
async def search_repositories(
    q: str = Query(..., description="Query di ricerca"),
    per_page: int = Query(10, le=100)
):
    """
    🔍 Cerca repository
    """
    return await github.search_repos(q, per_page)


@router.get("/search/code")
async def search_code(
    q: str = Query(..., description="Query di ricerca"),
    per_page: int = Query(10, le=100)
):
    """
    🔍 Cerca codice (richiede autenticazione)
    """
    if not github.is_configured:
        raise HTTPException(status_code=401, detail="GitHub token richiesto per cercare codice")
    return await github.search_code(q, per_page)


@router.get("/search/issues")
async def search_issues(
    q: str = Query(..., description="Query di ricerca"),
    per_page: int = Query(10, le=100)
):
    """
    🔍 Cerca issues e PR
    """
    return await github.search_issues(q, per_page)


# ==================== GISTS ====================

@router.get("/gists")
async def list_gists(per_page: int = Query(30, le=100)):
    """
    📝 Lista gist dell'utente
    """
    if not github.is_configured:
        raise HTTPException(status_code=401, detail="GitHub token non configurato")
    return await github.list_gists(per_page)


@router.post("/gists")
async def create_gist(request: CreateGistRequest):
    """
    ➕ Crea nuovo gist
    """
    if not github.is_configured:
        raise HTTPException(status_code=401, detail="GitHub token non configurato")
    
    files = {request.filename: {"content": request.content}}
    result = await github.create_gist(files, request.description, request.public)
    
    if "error" in result:
        raise HTTPException(status_code=result.get("status", 400), detail=result["error"])
    
    return {"success": True, "gist": result}


# ==================== NOTIFICATIONS ====================

@router.get("/notifications")
async def list_notifications(all_notifications: bool = False):
    """
    🔔 Lista notifiche GitHub
    """
    if not github.is_configured:
        raise HTTPException(status_code=401, detail="GitHub token non configurato")
    return await github.list_notifications(all_notifications)

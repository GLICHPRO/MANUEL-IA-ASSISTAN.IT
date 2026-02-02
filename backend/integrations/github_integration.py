"""
🐙 GitHub Integration for GIDEON

Integrazione completa con GitHub REST API.
Funziona con Personal Access Token (GRATUITO).

Funzionalità:
- Repository: list, create, delete, clone info
- Issues: create, list, update, comment
- Pull Requests: list, create, review
- Commits: list, view diff
- Search: code, repos, issues
- User: profile, orgs, activity
"""

import os
import httpx
from typing import Optional, Dict, List, Any
from datetime import datetime
from pathlib import Path
from loguru import logger

# Read token from .env
def _get_github_token() -> Optional[str]:
    """Legge il token GitHub dal file .env"""
    token = os.getenv("GITHUB_TOKEN")
    if token:
        return token
    
    # Try to read directly from .env file
    env_path = Path(__file__).parent.parent / ".env"
    if env_path.exists():
        with open(env_path, 'r', encoding='utf-8') as f:
            for line in f:
                if line.startswith("GITHUB_TOKEN="):
                    return line.split("=", 1)[1].strip()
    return None


class GitHubIntegration:
    """
    Client per GitHub REST API
    
    Rate Limits:
    - Con token: 5.000 richieste/ora
    - Senza token: 60 richieste/ora
    """
    
    BASE_URL = "https://api.github.com"
    
    def __init__(self, token: Optional[str] = None):
        self.token = token or _get_github_token()
        self.headers = {
            "Accept": "application/vnd.github+json",
            "X-GitHub-Api-Version": "2022-11-28",
            "User-Agent": "GIDEON-Assistant"
        }
        if self.token:
            self.headers["Authorization"] = f"Bearer {self.token}"
            logger.info("✅ GitHub: Token configurato")
        else:
            logger.warning("⚠️ GitHub: Nessun token configurato (limite 60 req/ora)")
    
    @property
    def is_configured(self) -> bool:
        return bool(self.token)
    
    async def _request(
        self, 
        method: str, 
        endpoint: str, 
        data: Dict = None,
        params: Dict = None
    ) -> Dict[str, Any]:
        """Esegue una richiesta all'API GitHub"""
        url = f"{self.BASE_URL}{endpoint}"
        
        async with httpx.AsyncClient(timeout=30) as client:
            try:
                response = await client.request(
                    method=method,
                    url=url,
                    headers=self.headers,
                    json=data,
                    params=params
                )
                
                # Check rate limit
                remaining = response.headers.get("X-RateLimit-Remaining", "?")
                logger.debug(f"GitHub API: {endpoint} - Rate limit remaining: {remaining}")
                
                if response.status_code == 401:
                    return {"error": "Token non valido o scaduto", "status": 401}
                elif response.status_code == 403:
                    return {"error": "Rate limit superato o permessi insufficienti", "status": 403}
                elif response.status_code == 404:
                    return {"error": "Risorsa non trovata", "status": 404}
                elif response.status_code >= 400:
                    return {"error": response.text, "status": response.status_code}
                
                return response.json() if response.text else {"success": True}
                
            except Exception as e:
                logger.error(f"GitHub API error: {e}")
                return {"error": str(e), "status": 500}
    
    # ==================== USER ====================
    
    async def get_user(self) -> Dict:
        """Ottieni info utente autenticato"""
        return await self._request("GET", "/user")
    
    async def get_user_repos(self, per_page: int = 30, page: int = 1) -> List[Dict]:
        """Lista repository dell'utente"""
        return await self._request(
            "GET", "/user/repos",
            params={"per_page": per_page, "page": page, "sort": "updated"}
        )
    
    async def get_user_orgs(self) -> List[Dict]:
        """Lista organizzazioni dell'utente"""
        return await self._request("GET", "/user/orgs")
    
    # ==================== REPOSITORIES ====================
    
    async def get_repo(self, owner: str, repo: str) -> Dict:
        """Ottieni dettagli repository"""
        return await self._request("GET", f"/repos/{owner}/{repo}")
    
    async def create_repo(
        self, 
        name: str, 
        description: str = "", 
        private: bool = False,
        auto_init: bool = True
    ) -> Dict:
        """Crea nuovo repository"""
        return await self._request("POST", "/user/repos", data={
            "name": name,
            "description": description,
            "private": private,
            "auto_init": auto_init
        })
    
    async def delete_repo(self, owner: str, repo: str) -> Dict:
        """Elimina repository (richiede permesso delete_repo)"""
        return await self._request("DELETE", f"/repos/{owner}/{repo}")
    
    async def get_repo_contents(
        self, 
        owner: str, 
        repo: str, 
        path: str = "",
        ref: str = None
    ) -> Dict:
        """Ottieni contenuti di un path nel repo"""
        params = {"ref": ref} if ref else None
        return await self._request(
            "GET", f"/repos/{owner}/{repo}/contents/{path}",
            params=params
        )
    
    async def get_repo_languages(self, owner: str, repo: str) -> Dict:
        """Ottieni linguaggi usati nel repo"""
        return await self._request("GET", f"/repos/{owner}/{repo}/languages")
    
    # ==================== ISSUES ====================
    
    async def list_issues(
        self, 
        owner: str, 
        repo: str, 
        state: str = "open",
        per_page: int = 30
    ) -> List[Dict]:
        """Lista issues di un repository"""
        return await self._request(
            "GET", f"/repos/{owner}/{repo}/issues",
            params={"state": state, "per_page": per_page}
        )
    
    async def create_issue(
        self, 
        owner: str, 
        repo: str, 
        title: str,
        body: str = "",
        labels: List[str] = None
    ) -> Dict:
        """Crea nuova issue"""
        data = {"title": title, "body": body}
        if labels:
            data["labels"] = labels
        return await self._request("POST", f"/repos/{owner}/{repo}/issues", data=data)
    
    async def update_issue(
        self, 
        owner: str, 
        repo: str, 
        issue_number: int,
        title: str = None,
        body: str = None,
        state: str = None
    ) -> Dict:
        """Aggiorna issue esistente"""
        data = {}
        if title:
            data["title"] = title
        if body:
            data["body"] = body
        if state:
            data["state"] = state
        return await self._request(
            "PATCH", f"/repos/{owner}/{repo}/issues/{issue_number}", 
            data=data
        )
    
    async def comment_issue(
        self, 
        owner: str, 
        repo: str, 
        issue_number: int,
        body: str
    ) -> Dict:
        """Aggiungi commento a issue"""
        return await self._request(
            "POST", f"/repos/{owner}/{repo}/issues/{issue_number}/comments",
            data={"body": body}
        )
    
    # ==================== COMMITS ====================
    
    async def list_commits(
        self, 
        owner: str, 
        repo: str, 
        per_page: int = 30,
        sha: str = None
    ) -> List[Dict]:
        """Lista commits di un repository"""
        params = {"per_page": per_page}
        if sha:
            params["sha"] = sha
        return await self._request(
            "GET", f"/repos/{owner}/{repo}/commits",
            params=params
        )
    
    async def get_commit(self, owner: str, repo: str, sha: str) -> Dict:
        """Ottieni dettagli di un commit"""
        return await self._request("GET", f"/repos/{owner}/{repo}/commits/{sha}")
    
    # ==================== PULL REQUESTS ====================
    
    async def list_pull_requests(
        self, 
        owner: str, 
        repo: str, 
        state: str = "open",
        per_page: int = 30
    ) -> List[Dict]:
        """Lista pull requests"""
        return await self._request(
            "GET", f"/repos/{owner}/{repo}/pulls",
            params={"state": state, "per_page": per_page}
        )
    
    async def create_pull_request(
        self, 
        owner: str, 
        repo: str, 
        title: str,
        head: str,
        base: str,
        body: str = ""
    ) -> Dict:
        """Crea pull request"""
        return await self._request(
            "POST", f"/repos/{owner}/{repo}/pulls",
            data={"title": title, "head": head, "base": base, "body": body}
        )
    
    # ==================== SEARCH ====================
    
    async def search_repos(self, query: str, per_page: int = 10) -> Dict:
        """Cerca repository"""
        return await self._request(
            "GET", "/search/repositories",
            params={"q": query, "per_page": per_page, "sort": "stars"}
        )
    
    async def search_code(self, query: str, per_page: int = 10) -> Dict:
        """Cerca codice (richiede autenticazione)"""
        return await self._request(
            "GET", "/search/code",
            params={"q": query, "per_page": per_page}
        )
    
    async def search_issues(self, query: str, per_page: int = 10) -> Dict:
        """Cerca issues e PR"""
        return await self._request(
            "GET", "/search/issues",
            params={"q": query, "per_page": per_page, "sort": "created"}
        )
    
    # ==================== GISTS ====================
    
    async def list_gists(self, per_page: int = 30) -> List[Dict]:
        """Lista gist dell'utente"""
        return await self._request(
            "GET", "/gists",
            params={"per_page": per_page}
        )
    
    async def create_gist(
        self, 
        files: Dict[str, Dict[str, str]], 
        description: str = "",
        public: bool = False
    ) -> Dict:
        """
        Crea un gist
        files format: {"filename.py": {"content": "code here"}}
        """
        return await self._request("POST", "/gists", data={
            "files": files,
            "description": description,
            "public": public
        })
    
    # ==================== NOTIFICATIONS ====================
    
    async def list_notifications(self, all_notifications: bool = False) -> List[Dict]:
        """Lista notifiche"""
        return await self._request(
            "GET", "/notifications",
            params={"all": str(all_notifications).lower()}
        )
    
    # ==================== RATE LIMIT ====================
    
    async def get_rate_limit(self) -> Dict:
        """Ottieni stato rate limit"""
        return await self._request("GET", "/rate_limit")
    
    # ==================== HELPER METHODS ====================
    
    async def get_status(self) -> Dict:
        """Ottieni stato dell'integrazione GitHub"""
        if not self.is_configured:
            return {
                "configured": False,
                "message": "Token GitHub non configurato. Aggiungi GITHUB_TOKEN al file .env"
            }
        
        # Test connection
        user = await self.get_user()
        if "error" in user:
            return {
                "configured": True,
                "connected": False,
                "error": user["error"]
            }
        
        rate_limit = await self.get_rate_limit()
        
        return {
            "configured": True,
            "connected": True,
            "user": user.get("login"),
            "name": user.get("name"),
            "repos_count": user.get("public_repos", 0) + user.get("total_private_repos", 0),
            "rate_limit": {
                "limit": rate_limit.get("rate", {}).get("limit", 5000),
                "remaining": rate_limit.get("rate", {}).get("remaining", 0),
                "reset": rate_limit.get("rate", {}).get("reset", 0)
            }
        }


# Singleton instance
github = GitHubIntegration()

"""
Social Media Analyzer per Gideon 2.0
Analizza i dati esportati da Instagram (metodo legale)

Come usare:
1. Vai su Instagram → Impostazioni → Privacy → Scarica i tuoi dati
2. Scegli formato JSON
3. Scarica e estrai lo ZIP
4. Carica i file followers.json e following.json
"""

import json
from pathlib import Path
from datetime import datetime
from typing import Optional
from loguru import logger


class InstagramAnalyzer:
    """
    Analizza i dati Instagram esportati dall'utente.
    Metodo 100% legale usando i dati che Instagram fornisce.
    """
    
    def __init__(self):
        self.followers: set = set()
        self.following: set = set()
        self.data_loaded = False
        self.load_timestamp: Optional[datetime] = None
        
    def load_from_export(self, export_folder: str) -> dict:
        """
        Carica i dati dall'export di Instagram.
        
        Args:
            export_folder: Percorso alla cartella con i file JSON esportati
            
        Returns:
            dict con statistiche di caricamento
        """
        folder = Path(export_folder)
        
        # Instagram salva i dati in strutture diverse a seconda della versione
        # Proviamo entrambe le strutture comuni
        
        followers_loaded = 0
        following_loaded = 0
        
        # Struttura 1: followers_and_following/followers_1.json
        followers_path_v1 = folder / "followers_and_following" / "followers_1.json"
        following_path_v1 = folder / "followers_and_following" / "following.json"
        
        # Struttura 2: connections/followers_and_following/followers_1.json
        followers_path_v2 = folder / "connections" / "followers_and_following" / "followers_1.json"
        following_path_v2 = folder / "connections" / "followers_and_following" / "following.json"
        
        # Struttura 3: file diretti
        followers_path_v3 = folder / "followers.json"
        following_path_v3 = folder / "following.json"
        
        # Prova a caricare followers
        for path in [followers_path_v1, followers_path_v2, followers_path_v3]:
            if path.exists():
                followers_loaded = self._load_followers_file(path)
                break
                
        # Prova a caricare following
        for path in [following_path_v1, following_path_v2, following_path_v3]:
            if path.exists():
                following_loaded = self._load_following_file(path)
                break
        
        if followers_loaded > 0 or following_loaded > 0:
            self.data_loaded = True
            self.load_timestamp = datetime.now()
            
        logger.info(f"📊 Caricati {followers_loaded} followers e {following_loaded} following")
        
        return {
            "success": self.data_loaded,
            "followers_count": len(self.followers),
            "following_count": len(self.following),
            "timestamp": self.load_timestamp.isoformat() if self.load_timestamp else None
        }
    
    def _load_followers_file(self, path: Path) -> int:
        """Carica file followers."""
        try:
            with open(path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                
            # Estrai username in base alla struttura
            if isinstance(data, list):
                for item in data:
                    if isinstance(item, dict):
                        # Struttura: [{"string_list_data": [{"value": "username"}]}]
                        if "string_list_data" in item:
                            for entry in item.get("string_list_data", []):
                                if "value" in entry:
                                    self.followers.add(entry["value"].lower())
                        # Struttura: [{"username": "..."}]
                        elif "username" in item:
                            self.followers.add(item["username"].lower())
                        elif "value" in item:
                            self.followers.add(item["value"].lower())
                            
            return len(self.followers)
        except Exception as e:
            logger.error(f"Errore caricamento followers: {e}")
            return 0
    
    def _load_following_file(self, path: Path) -> int:
        """Carica file following."""
        try:
            with open(path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            # Struttura con "relationships_following"
            if isinstance(data, dict) and "relationships_following" in data:
                data = data["relationships_following"]
                
            if isinstance(data, list):
                for item in data:
                    if isinstance(item, dict):
                        if "string_list_data" in item:
                            for entry in item.get("string_list_data", []):
                                if "value" in entry:
                                    self.following.add(entry["value"].lower())
                        elif "username" in item:
                            self.following.add(item["username"].lower())
                        elif "value" in item:
                            self.following.add(item["value"].lower())
                            
            return len(self.following)
        except Exception as e:
            logger.error(f"Errore caricamento following: {e}")
            return 0
    
    def load_from_json_strings(self, followers_json: str, following_json: str) -> dict:
        """
        Carica dati da stringhe JSON (per upload via API).
        """
        try:
            followers_data = json.loads(followers_json)
            following_data = json.loads(following_json)
            
            # Estrai usernames
            self.followers = self._extract_usernames(followers_data)
            self.following = self._extract_usernames(following_data)
            
            self.data_loaded = True
            self.load_timestamp = datetime.now()
            
            return {
                "success": True,
                "followers_count": len(self.followers),
                "following_count": len(self.following)
            }
        except Exception as e:
            logger.error(f"Errore parsing JSON: {e}")
            return {"success": False, "error": str(e)}
    
    def _extract_usernames(self, data) -> set:
        """Estrae usernames da varie strutture JSON di Instagram."""
        usernames = set()
        
        if isinstance(data, dict) and "relationships_following" in data:
            data = data["relationships_following"]
            
        if isinstance(data, list):
            for item in data:
                if isinstance(item, dict):
                    if "string_list_data" in item:
                        for entry in item.get("string_list_data", []):
                            if "value" in entry:
                                usernames.add(entry["value"].lower())
                    elif "username" in item:
                        usernames.add(item["username"].lower())
                    elif "value" in item:
                        usernames.add(item["value"].lower())
                elif isinstance(item, str):
                    usernames.add(item.lower())
                    
        return usernames

    # =========================================================================
    # ANALISI: Chi non ti segue indietro
    # =========================================================================
    
    def get_not_following_back(self) -> dict:
        """
        Trova gli utenti che segui ma che NON ti seguono.
        
        Returns:
            dict con lista utenti e statistiche
        """
        if not self.data_loaded:
            return {
                "success": False,
                "error": "Dati non caricati. Usa load_from_export() prima."
            }
        
        # Chi segui ma non ti segue
        not_following_back = self.following - self.followers
        
        # Chi ti segue ma non segui
        not_following = self.followers - self.following
        
        # Seguiti reciproci
        mutual = self.followers & self.following
        
        return {
            "success": True,
            "not_following_back": sorted(list(not_following_back)),
            "not_following_back_count": len(not_following_back),
            "fans": sorted(list(not_following)),  # Ti seguono ma non li segui
            "fans_count": len(not_following),
            "mutual_friends": len(mutual),
            "total_followers": len(self.followers),
            "total_following": len(self.following),
            "ratio": round(len(self.followers) / max(len(self.following), 1), 2)
        }
    
    def get_analysis_summary(self) -> dict:
        """
        Restituisce un riepilogo completo dell'analisi.
        """
        if not self.data_loaded:
            return {"success": False, "error": "Dati non caricati"}
        
        analysis = self.get_not_following_back()
        
        # Calcola percentuali
        follow_back_rate = 0
        if len(self.following) > 0:
            mutual = len(self.followers & self.following)
            follow_back_rate = round((mutual / len(self.following)) * 100, 1)
        
        return {
            "success": True,
            "summary": {
                "followers": len(self.followers),
                "following": len(self.following),
                "mutual": len(self.followers & self.following),
                "not_following_back": analysis["not_following_back_count"],
                "fans": analysis["fans_count"],
                "follow_back_rate": f"{follow_back_rate}%",
                "ratio": analysis["ratio"]
            },
            "details": analysis,
            "recommendations": self._get_recommendations(analysis)
        }
    
    def _get_recommendations(self, analysis: dict) -> list:
        """Genera raccomandazioni basate sull'analisi."""
        recommendations = []
        
        ratio = analysis.get("ratio", 1)
        not_following_back = analysis.get("not_following_back_count", 0)
        
        if ratio < 0.5:
            recommendations.append({
                "type": "warning",
                "message": f"Il tuo ratio follower/following è basso ({ratio}). "
                          f"Considera di smettere di seguire alcuni account che non ricambiano."
            })
            
        if not_following_back > 100:
            recommendations.append({
                "type": "info",
                "message": f"Hai {not_following_back} account che non ti seguono. "
                          f"Potresti fare una pulizia per migliorare il tuo feed."
            })
            
        if ratio > 2:
            recommendations.append({
                "type": "success",
                "message": f"Ottimo ratio ({ratio})! Hai più follower che following."
            })
            
        return recommendations


class SocialAnalyzerManager:
    """
    Manager per analisi social integrato in Gideon.
    """
    
    def __init__(self):
        self.instagram = InstagramAnalyzer()
        self.supported_platforms = ["instagram"]
        
    def get_status(self) -> dict:
        """Stato del modulo."""
        return {
            "instagram": {
                "data_loaded": self.instagram.data_loaded,
                "followers": len(self.instagram.followers) if self.instagram.data_loaded else 0,
                "following": len(self.instagram.following) if self.instagram.data_loaded else 0,
                "last_update": self.instagram.load_timestamp.isoformat() if self.instagram.load_timestamp else None
            },
            "note": "Per analizzare i dati, scarica il tuo export da Instagram (Impostazioni → Privacy → Scarica i tuoi dati)"
        }
    
    def analyze_instagram(self, export_path: str = None, followers_json: str = None, following_json: str = None) -> dict:
        """
        Analizza dati Instagram.
        
        Args:
            export_path: Percorso cartella export Instagram
            followers_json: JSON string con followers (alternativo)
            following_json: JSON string con following (alternativo)
        """
        if export_path:
            load_result = self.instagram.load_from_export(export_path)
        elif followers_json and following_json:
            load_result = self.instagram.load_from_json_strings(followers_json, following_json)
        else:
            return {"success": False, "error": "Fornisci export_path o JSON strings"}
        
        if not load_result.get("success"):
            return load_result
            
        return self.instagram.get_analysis_summary()


# Istanza globale
social_analyzer = SocialAnalyzerManager()

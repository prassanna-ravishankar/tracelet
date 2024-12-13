import os
from typing import Dict, Any, Optional
from git import Repo, InvalidGitRepositoryError
from ..core.interfaces import CollectorInterface

class GitCollector(CollectorInterface):
    """Collector for Git repository information"""
    
    def __init__(self, repo_path: Optional[str] = None):
        self.repo_path = repo_path or os.getcwd()
        self._repo = None
        
    def initialize(self):
        try:
            self._repo = Repo(self.repo_path, search_parent_directories=True)
        except InvalidGitRepositoryError:
            self._repo = None
            
    def collect(self) -> Dict[str, Any]:
        if not self._repo:
            return {
                "git_available": False,
                "error": "Not a git repository"
            }
            
        try:
            active_branch = self._repo.active_branch
            commit = self._repo.head.commit
            
            # Get the git diff
            diff = self._repo.git.diff(None)
            
            # Get remote information
            remotes = {}
            for remote in self._repo.remotes:
                remotes[remote.name] = [url for url in remote.urls]
                
            return {
                "git_available": True,
                "branch": active_branch.name,
                "commit_hash": commit.hexsha,
                "commit_message": commit.message.strip(),
                "commit_author": f"{commit.author.name} <{commit.author.email}>",
                "commit_datetime": commit.committed_datetime.isoformat(),
                "is_dirty": self._repo.is_dirty(),
                "untracked_files": self._repo.untracked_files,
                "diff": diff if diff else None,
                "remotes": remotes
            }
        except Exception as e:
            return {
                "git_available": True,
                "error": str(e)
            }
            
    def start(self):
        pass  # Not a continuous collector
        
    def stop(self):
        pass  # Not a continuous collector 
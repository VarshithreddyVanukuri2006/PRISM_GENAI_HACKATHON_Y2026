from pathlib import Path

SUPPORTED_EXTENSIONS = {
    ".py": "Python", ".js": "JavaScript", ".jsx": "JavaScript",
    ".mjs": "JavaScript", ".cjs": "JavaScript", ".ts": "TypeScript",
    ".tsx": "TypeScript", ".java": "Java", ".c": "C++", ".cc": "C++",
    ".cpp": "C++", ".cxx": "C++", ".h": "C++", ".hh": "C++",
    ".hpp": "C++", ".hxx": "C++",
}

IGNORED_DIRS = {
    ".git", ".hg", ".svn", ".idea", ".vscode", "node_modules",
    "__pycache__", ".venv", "venv", "env", "build", "dist", "target",
    "coverage", ".next", ".nuxt", "vendor", "out",
}
MAX_SOURCE_FILE_BYTES = 1_000_000
DEFAULT_INDEX_DIR = Path("data/indexes")

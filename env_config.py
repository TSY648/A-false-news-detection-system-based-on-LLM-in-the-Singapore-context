from pathlib import Path

from dotenv import load_dotenv


def load_project_env() -> None:
    """Load local .env files for the project without overriding existing env vars."""
    system_dir = Path(__file__).resolve().parent
    project_root = system_dir.parent

    candidates = [
        project_root / ".env",
        system_dir / ".env",
        project_root / "pinecone.env",
        system_dir / "pinecone.env",
    ]

    for path in candidates:
        if path.exists():
            load_dotenv(path, override=False)

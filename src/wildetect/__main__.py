from dotenv import load_dotenv

from .cli import ROOT_DIR, app

if __name__ == "__main__":
    load_dotenv(ROOT_DIR / ".env", override=True)
    app()

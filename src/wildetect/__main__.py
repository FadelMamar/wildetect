from dotenv import load_dotenv

from .cli import ROOT, app

if __name__ == "__main__":
    load_dotenv(ROOT / ".env", override=True)
    app()

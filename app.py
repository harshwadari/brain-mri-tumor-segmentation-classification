from pathlib import Path

import runpy


frontend_app = Path(__file__).resolve().parent / "FRONTEND" / "app.py"
runpy.run_path(str(frontend_app), run_name="__main__")

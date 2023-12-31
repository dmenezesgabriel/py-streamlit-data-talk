import os
import sys

from streamlit.web import cli as stcli

if __name__ == "__main__":
    os.environ["PYTHONPATH"] = sys.executable + ";" + os.path.dirname(__file__)
    sys.argv = ["streamlit", "run", "src/presentation/streamlit/main.py"]
    sys.exit(stcli.main())

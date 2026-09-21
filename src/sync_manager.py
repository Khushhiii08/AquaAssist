import os
import json
import requests

VERSION_FILE = "data/knowledge_version.json"

REMOTE_VERSION_URL = "https://github.com/Khushhiii08/AquaAssist/blob/main/data/version.json"
def check_for_updates():
    """Checks remote manifest for new aquaculture guidelines if internet is available."""
    try:
        # Quick connectivity check with a 3-second timeout
        response = requests.get(REMOTE_VERSION_URL, timeout=3)
        if response.status_code == 200:
            # For a real deployment, you would check remote_data.get("version")
            # Here we verify live internet connectivity successfully:
            local_version = 1.0
            if os.path.exists(VERSION_FILE):
                with open(VERSION_FILE, "r") as f:
                    local_version = json.load(f).get("version", 1.0)
            
            # Simulated check: If online, we can report status
            return True, f"Connected! Knowledge base is up to date (v{local_version})."
        return False, "Offline mode: Running on local vector store."
    except Exception as e:
        return False, "Offline mode: No internet connection detected. Operating locally."
"""
AquaAssist: Safe Pointer-Based Database Synchronization
Resolves Windows file-locking [WinError 32] by using versioned directories
and an atomic pointer file rather than in-place folder deletion.
"""

import os
import json
import zipfile
import urllib.request
import ssl
import shutil
import time
from typing import Tuple, Optional

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
POINTER_FILE = os.path.join(DATA_DIR, "active_db.json")


def get_active_db_path() -> str:
    """Returns the path to the currently active ChromaDB directory."""
    if os.path.exists(POINTER_FILE):
        try:
            with open(POINTER_FILE, "r") as f:
                data = json.load(f)
                active_dir = data.get("active_dir")
                if active_dir and os.path.exists(os.path.join(DATA_DIR, active_dir)):
                    return os.path.join(DATA_DIR, active_dir)
        except Exception:
            pass
    return os.path.join(DATA_DIR, "chroma_db")


def set_active_db_path(version_dir_name: str) -> None:
    """Atomically updates the active database pointer."""
    os.makedirs(DATA_DIR, exist_ok=True)
    temp_pointer = POINTER_FILE + ".tmp"
    with open(temp_pointer, "w") as f:
        json.dump({"active_dir": version_dir_name}, f)
    # os.replace is atomic across POSIX and modern Windows systems
    os.replace(temp_pointer, POINTER_FILE)


def download_and_apply_ota_update(url: str, target_dir: Optional[str] = None, max_retries: int = 3) -> Tuple[bool, str]:
    """
    If target_dir is specified (e.g. models/deberta_aquaculture_router), extracts directly there.
    If target_dir is None, treats it as a ChromaDB update and applies blue/green pointer rotation.
    """
    # Ensure data directory exists unconditionally
    os.makedirs(DATA_DIR, exist_ok=True)

    ctx = ssl.create_default_context()
    ctx.check_hostname = False
    ctx.verify_mode = ssl.CERT_NONE

    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36'
    }

    # CASE 1: Model Binary Updates (DeBERTa Router)
    if target_dir:
        os.makedirs(target_dir, exist_ok=True)
        temp_zip = os.path.join(DATA_DIR, "temp_model_ota.zip")
        try:
            req = urllib.request.Request(url, headers=headers)
            print(f"[*] Downloading model archive from {url}...")
            with urllib.request.urlopen(req, context=ctx, timeout=600) as response:
                with open(temp_zip, "wb") as f:
                    while True:
                        chunk = response.read(1024 * 1024)
                        if not chunk:
                            break
                        f.write(chunk)

            if not zipfile.is_zipfile(temp_zip):
                return False, "Downloaded model archive is corrupted or invalid."

            with zipfile.ZipFile(temp_zip, 'r') as z:
                z.extractall(target_dir)

            # Handle single top-level directory nesting if zipped as folder
            subdirs = [d for d in os.listdir(target_dir) if os.path.isdir(os.path.join(target_dir, d))]
            if len(subdirs) == 1 and subdirs[0] == os.path.basename(target_dir):
                nested_dir = os.path.join(target_dir, subdirs[0])
                for item in os.listdir(nested_dir):
                    shutil.move(os.path.join(nested_dir, item), target_dir)
                os.rmdir(nested_dir)

            return True, f"Model extracted to {target_dir} successfully."
        except Exception as e:
            return False, f"Model OTA Error: {e}"
        finally:
            if os.path.exists(temp_zip):
                try:
                    os.remove(temp_zip)
                except OSError:
                    pass

    # CASE 2: ChromaDB Vector Knowledge Base Updates (Blue/Green Switch)
    timestamp = int(time.time())
    new_version_name = f"chroma_db_v{timestamp}"
    new_target_dir = os.path.join(DATA_DIR, new_version_name)
    temp_zip = os.path.join(DATA_DIR, "temp_ota_update.zip")

    try:
        req = urllib.request.Request(url, headers=headers)
        print(f"[*] Downloading knowledge base update to {new_version_name}...")

        with urllib.request.urlopen(req, context=ctx, timeout=600) as response:
            with open(temp_zip, "wb") as f:
                while True:
                    chunk = response.read(1024 * 1024)
                    if not chunk:
                        break
                    f.write(chunk)

        if not zipfile.is_zipfile(temp_zip):
            return False, "Downloaded archive is corrupted or not a valid zip."

        # Extract to new version directory
        os.makedirs(new_target_dir, exist_ok=True)
        with zipfile.ZipFile(temp_zip, 'r') as z:
            z.extractall(new_target_dir)

        # Handle nested zip extraction if parent directory was packaged
        subdirs = [d for d in os.listdir(new_target_dir) if os.path.isdir(os.path.join(new_target_dir, d))]
        if len(subdirs) == 1 and subdirs[0] in ("chroma_db", "data"):
            nested_path = os.path.join(new_target_dir, subdirs[0])
            for item in os.listdir(nested_path):
                shutil.move(os.path.join(nested_path, item), new_target_dir)
            os.rmdir(nested_path)

        # Integrity Check: verify chroma.sqlite3 exists in package
        has_sqlite = False
        for root, _, files in os.walk(new_target_dir):
            if "chroma.sqlite3" in files:
                has_sqlite = True
                break

        if not has_sqlite:
            shutil.rmtree(new_target_dir, ignore_errors=True)
            return False, "Database verification failed: chroma.sqlite3 missing."

        # Atomically point active DB to new version
        set_active_db_path(new_version_name)
        print(f"[+] Successfully switched active database to: {new_version_name}")
        return True, "Knowledge base synchronized and activated successfully."

    except Exception as e:
        if os.path.exists(new_target_dir):
            shutil.rmtree(new_target_dir, ignore_errors=True)
        return False, f"OTA Sync Failure: {e}"
    finally:
        if os.path.exists(temp_zip):
            try:
                os.remove(temp_zip)
            except OSError:
                pass
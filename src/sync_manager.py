import os
import shutil
import requests
import zipfile
import tempfile
import urllib.request
import json
import ssl

def get_latest_github_release_url(repo_path="Khushhiii08/AquaAssist", asset_name="chroma_db.zip"):
    """
    Queries the GitHub API for the latest release and returns the download URL,
    with SSL bypass for local Mac/Docker development.
    """
    api_url = f"https://api.github.com/repos/{repo_path}/releases/latest"
    
    try:
        # Bypass strict Mac/Docker SSL Certificate verification
        ctx = ssl.create_default_context()
        ctx.check_hostname = False
        ctx.verify_mode = ssl.CERT_NONE

        req = urllib.request.Request(api_url, headers={'User-Agent': 'AquaAssist-Edge-Device'})
        
        with urllib.request.urlopen(req, context=ctx, timeout=10) as response:
            data = json.loads(response.read().decode())
            
            # Search the assets array for our zip file
            for asset in data.get('assets', []):
                if asset.get('name') == asset_name:
                    return asset.get('browser_download_url')
                    
        return None
    except Exception as e:
        # This will print the exact reason to your terminal if it fails again
        print(f"[!] GitHub API Fetch Error: {e}")
        return None

def download_and_apply_ota_update(url):
    try:
        # Bypass strict Mac SSL
        ctx = ssl.create_default_context()
        ctx.check_hostname = False
        ctx.verify_mode = ssl.CERT_NONE

        # --- NEW: Intercept the GitHub redirect to get the tag name ---
        release_tag = "Latest"
        try:
            # Pinging the web URL (not the API) to avoid the 403 Rate Limit
            tag_req = urllib.request.Request("https://github.com/Khushhiii08/AquaAssist/releases/latest", method="HEAD", headers={'User-Agent': 'AquaAssist'})
            tag_resp = urllib.request.urlopen(tag_req, context=ctx, timeout=10)
            release_tag = tag_resp.url.split('/')[-1]
        except Exception as e:
            print(f"[!] Tag fetch failed: {e}") # Silently fallback to "Latest"

        # --- Execute the actual ZIP download ---
        req = urllib.request.Request(url, headers={'User-Agent': 'AquaAssist'})
        
        with urllib.request.urlopen(req, context=ctx, timeout=45) as response:
            if response.status != 200:
                return False, f"Download failed with status: {response.status}"
                
            with open("temp_db.zip", "wb") as f:
                f.write(response.read())

        if not zipfile.is_zipfile("temp_db.zip"):
            return False, "Downloaded file is corrupted or not a valid ZIP."

        # Ensure target directory exists and is empty before extracting
        target_dir = "data/chroma_db"
        if os.path.exists(target_dir):
            shutil.rmtree(target_dir)
        os.makedirs(target_dir, exist_ok=True)

        # Extract the ZIP
        with zipfile.ZipFile("temp_db.zip", 'r') as zip_ref:
            zip_ref.extractall(target_dir)
            
        os.remove("temp_db.zip")
        
        # --- ANTI-NESTING FIX ---
        # If macOS zipped the folder instead of the files, flatten the structure
        nested_dir = os.path.join(target_dir, "chroma_db")
        if os.path.exists(nested_dir) and os.path.isdir(nested_dir):
            for item in os.listdir(nested_dir):
                shutil.move(os.path.join(nested_dir, item), target_dir)
            os.rmdir(nested_dir)
            
        # Clean up macOS hidden metadata folder if it exists
        macosx_dir = os.path.join(target_dir, "__MACOSX")
        if os.path.exists(macosx_dir):
            shutil.rmtree(macosx_dir)
        
        return True, f"Knowledge base updated successfully from GitHub! (Release: {release_tag})"
        
    except Exception as e:
        return False, f"OTA Sync Error: {e}"
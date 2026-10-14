import os
import shutil
import zipfile
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
        ctx = ssl.create_default_context()
        ctx.check_hostname = False
        ctx.verify_mode = ssl.CERT_NONE

        req = urllib.request.Request(api_url, headers={'User-Agent': 'AquaAssist-Edge-Device'})
        
        with urllib.request.urlopen(req, context=ctx, timeout=15) as response:
            data = json.loads(response.read().decode())
            for asset in data.get('assets', []):
                if asset.get('name') == asset_name:
                    return asset.get('browser_download_url')
                    
        return None
    except Exception as e:
        print(f"[!] GitHub API Fetch Error: {e}")
        return None

def download_and_apply_ota_update(url, target_dir="data/chroma_db"):
    """
    Downloads and extracts a ZIP archive from GitHub Releases to the target directory,
    with SSL bypass and safe chunking for large model binaries (e.g., 473MB NLI router).
    """
    try:
        ctx = ssl.create_default_context()
        ctx.check_hostname = False
        ctx.verify_mode = ssl.CERT_NONE

        release_tag = "Latest"
        try:
            tag_req = urllib.request.Request("https://github.com/Khushhiii08/AquaAssist/releases/latest", method="HEAD", headers={'User-Agent': 'AquaAssist'})
            tag_resp = urllib.request.urlopen(tag_req, context=ctx, timeout=15)
            release_tag = tag_resp.url.split('/')[-1]
        except Exception as e:
            print(f"[!] Tag fetch failed: {e}")

        req = urllib.request.Request(url, headers={'User-Agent': 'AquaAssist'})
        temp_zip = "temp_ota.zip"
        
        print(f"[*] Downloading {url} to {target_dir} (this may take a moment for large assets)...")
        
        # 300-second (5 min) timeout to accommodate large model weights safely
        with urllib.request.urlopen(req, context=ctx, timeout=300) as response:
            if response.status != 200:
                return False, f"Download failed with status: {response.status}"
                
            # Stream-read in chunks to prevent memory spikes
            with open(temp_zip, "wb") as f:
                while True:
                    chunk = response.read(1024 * 1024) # 1MB chunks
                    if not chunk:
                        break
                    f.write(chunk)

        if not zipfile.is_zipfile(temp_zip):
            return False, "Downloaded file is corrupted or not a valid ZIP."

        if os.path.exists(target_dir):
            shutil.rmtree(target_dir)
        os.makedirs(target_dir, exist_ok=True)

        print(f"[*] Extracting archive into {target_dir}...")
        with zipfile.ZipFile(temp_zip, 'r') as zip_ref:
            zip_ref.extractall(target_dir)
            
        os.remove(temp_zip)
        
        # --- ANTI-NESTING FIX ---
        base_folder_name = os.path.basename(target_dir)
        nested_dir = os.path.join(target_dir, base_folder_name)
        if os.path.exists(nested_dir) and os.path.isdir(nested_dir):
            for item in os.listdir(nested_dir):
                shutil.move(os.path.join(nested_dir, item), target_dir)
            os.rmdir(nested_dir)
            
        macosx_dir = os.path.join(target_dir, "__MACOSX")
        if os.path.exists(macosx_dir):
            shutil.rmtree(macosx_dir)
        
        return True, f"Updated {target_dir} successfully from GitHub! (Release: {release_tag})"
        
    except Exception as e:
        if os.path.exists("temp_ota.zip"):
            try:
                os.remove("temp_ota.zip")
            except OSError:
                pass
        return False, f"OTA Sync Error: {e}"
import os
import shutil
import requests
import zipfile
import tempfile

def download_and_apply_ota_update(download_url):
    """
    Simulates a production edge update by downloading a pre-computed 
    ChromaDB from a centralized cloud source and replacing the local DB.
    """
    target_db_path = os.path.join("./data", "chroma_db")
    
    try:
        # Create a temporary directory to handle the download safely
        with tempfile.TemporaryDirectory() as temp_dir:
            zip_path = os.path.join(temp_dir, "update.zip")
            
            # 1. Download the new database
            response = requests.get(download_url, stream=True)
            response.raise_for_status()
            with open(zip_path, "wb") as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
                    
            # 2. Extract to temp folder
            extract_path = os.path.join(temp_dir, "extracted")
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                zip_ref.extractall(extract_path)
                
            # 3. Safely swap the old DB with the new one
            if os.path.exists(target_db_path):
                shutil.rmtree(target_db_path)
            
            # Move the extracted chroma_db folder into the data directory
            # (Assumes the zip contains the chroma_db folder directly)
            extracted_db = os.path.join(extract_path, "chroma_db")
            shutil.move(extracted_db, target_db_path)
            
        return True, "✅ Knowledge base updated successfully from cloud!"
    except Exception as e:
        return False, f"Failed to apply OTA update. Error: {e}"
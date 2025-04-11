import os
import json
import time
import hashlib
from pathlib import Path
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.options import Options
from selenium.common.exceptions import StaleElementReferenceException, WebDriverException
from urllib.request import urlretrieve
from urllib.error import URLError, HTTPError
from concurrent.futures import ThreadPoolExecutor, as_completed

# --- CONFIGURATION ---
TIERS_PATH = "/Users/Fan/sol_materials/vibe_project/tiers0,1,2.json"
IMAGE_DIR = Path("/Users/Fan/sol_materials/vibe_project/vibeplane_scraper/data/images")
METADATA_PATH = Path("/Users/Fan/sol_materials/vibe_project/vibeplane_scraper/data/metadata/image_metadata.json")
FAILED_PATH = Path("/Users/Fan/sol_materials/vibe_project/vibeplane_scraper/data/metadata/failed_downloads.json")
IMAGES_PER_QUERY = 100
MAX_WORKERS = 10
SCROLL_PAUSE_TIME = 2

# --- UTILS ---
def generate_filename(url):
    return hashlib.md5(url.encode()).hexdigest() + ".jpg"

def download_image(entry):
    url = entry["image_url"]
    filepath = Path(entry["local_path"])
    try:
        urlretrieve(url, filepath)
        return {"status": "success", "entry": entry}
    except (HTTPError, URLError) as e:
        return {"status": "fail", "url": url, "error": str(e)}

def scroll_and_collect(driver):
    last_height = driver.execute_script("return document.body.scrollHeight")
    while True:
        driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
        time.sleep(SCROLL_PAUSE_TIME)
        new_height = driver.execute_script("return document.body.scrollHeight")
        if new_height == last_height:
            break
        last_height = new_height

# --- MAIN FUNCTION ---
def run_pinterest_scraper():
    chrome_options = Options()
    #chrome_options.add_argument("--headless=new")
    chrome_options.add_argument("--disable-gpu")
    chrome_options.add_argument("--window-size=1920,1080")
    chrome_options.add_argument("--log-level=3")
    driver = webdriver.Chrome(options=chrome_options)

    with open(TIERS_PATH, "r") as f:
        tiers = json.load(f)

    os.makedirs(METADATA_PATH.parent, exist_ok=True)
    os.makedirs(FAILED_PATH.parent, exist_ok=True)

    image_metadata = []
    failed_downloads = []
    if METADATA_PATH.exists():
        with open(METADATA_PATH, "r") as f:
            image_metadata = json.load(f)

    seen_urls = set(entry["image_url"] for entry in image_metadata)
    processed_queries = set(entry["tier2_query"] for entry in image_metadata)
    download_queue = []

    for tier0, tier1s in tiers.items():
        for tier1, tier2_list in tier1s.items():
            for tier2_query in tier2_list:
                if tier2_query in processed_queries:
                    continue

                print(f"\n🔍 Querying: {tier2_query}")
                try:
                    search_url = f"https://www.pinterest.com/search/pins/?q={tier2_query.replace(' ', '%20')}"
                    driver.get(search_url)
                    time.sleep(3)
                    scroll_and_collect(driver)

                    img_elements = driver.find_elements(By.TAG_NAME, "img")
                    count = 0
                    for img in img_elements:
                        try:
                            url = img.get_attribute("src")
                            if not url or url in seen_urls or "data:image" in url or "https://s.pinimg.com" in url:
                                continue

                            filename = generate_filename(url)
                            folder = IMAGE_DIR / tier0 / tier1
                            os.makedirs(folder, exist_ok=True)
                            path = folder / filename

                            download_queue.append({
                                "image_url": url,
                                "filename": filename,
                                "tier0": tier0,
                                "tier1": tier1,
                                "tier2_query": tier2_query,
                                "local_path": str(path)
                            })
                            seen_urls.add(url)
                            count += 1

                            if count >= IMAGES_PER_QUERY:
                                break

                        except StaleElementReferenceException:
                            continue
                        except Exception as e:
                            print(f"⚠️ Failed to queue image: {e}")
                            failed_downloads.append({"url": url, "error": str(e)})
                except WebDriverException as e:
                    print(f"❌ Failed on query '{tier2_query}': {e}")
                    failed_downloads.append({"query": tier2_query, "error": str(e)})

    print(f"\n🚀 Starting batch download of {len(download_queue)} images...")
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = [executor.submit(download_image, entry) for entry in download_queue]
        for future in as_completed(futures):
            result = future.result()
            if result["status"] == "success":
                image_metadata.append(result["entry"])
                print(f"✅ Downloaded: {result['entry']['filename']}")
            else:
                print(f"⚠️ Failed to download {result['url']}: {result['error']}")
                failed_downloads.append(result)

    with open(METADATA_PATH, "w") as f:
        json.dump(image_metadata, f, indent=2)
    with open(FAILED_PATH, "w") as f:
        json.dump(failed_downloads, f, indent=2)

    driver.quit()
    print(f"\n✅ DONE — {len(image_metadata)} images saved.")
    print(f"⚠️ {len(failed_downloads)} failed downloads (see failed_downloads.json)")

if __name__ == "__main__":
    run_pinterest_scraper()

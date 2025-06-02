import csv
import os
import time
import requests
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from selenium.common.exceptions import TimeoutException, NoSuchElementException
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from urllib.parse import urljoin, urlparse
import re

class PinterestScraper:
    def __init__(self, csv_file, images_per_prompt=10, base_folder="pinterest_images"):
        self.csv_file = csv_file
        self.images_per_prompt = images_per_prompt
        self.base_folder = base_folder
        self.driver = None
        
    def setup_driver(self):
        """Initialize Chrome WebDriver with appropriate options"""
        chrome_options = Options()
        chrome_options.add_argument("--no-sandbox")
        chrome_options.add_argument("--disable-dev-shm-usage")
        chrome_options.add_argument("--disable-blink-features=AutomationControlled")
        chrome_options.add_experimental_option("excludeSwitches", ["enable-automation"])
        chrome_options.add_experimental_option('useAutomationExtension', False)
        chrome_options.add_argument("--user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36")
        
        # Uncomment the next line if you want to run headless
        # chrome_options.add_argument("--headless")
        
        try:
            self.driver = webdriver.Chrome(options=chrome_options)
            self.driver.execute_script("Object.defineProperty(navigator, 'webdriver', {get: () => undefined})")
        except Exception as e:
            print(f"Error setting up Chrome driver: {e}")
            print("Make sure you have ChromeDriver installed and in your PATH")
            return False
        return True
    
    def read_csv_prompts(self):
        """Read prompts from CSV file"""
        prompts = []
        try:
            with open(self.csv_file, 'r', encoding='utf-8') as file:
                csv_reader = csv.reader(file)
                # Skip header if present
                header = next(csv_reader, None)
                for row in csv_reader:
                    if row:  # Skip empty rows
                        prompts.append(row[0].strip())  # Assuming prompts are in first column
        except FileNotFoundError:
            print(f"CSV file '{self.csv_file}' not found!")
            return []
        except Exception as e:
            print(f"Error reading CSV file: {e}")
            return []
        
        return prompts
    
    def remove_processed_line(self, processed_prompt):
        """Remove the processed prompt from CSV file"""
        try:
            # Read all lines from the CSV
            with open(self.csv_file, 'r', encoding='utf-8') as file:
                lines = file.readlines()
            
            # Filter out the processed prompt (keep header and other lines)
            updated_lines = []
            header_processed = False
            
            for line in lines:
                line_content = line.strip()
                if not line_content:  # Keep empty lines
                    updated_lines.append(line)
                    continue
                
                # Keep header (first non-empty line)
                if not header_processed and line_content.lower() in ['prompt', 'prompts', 'query', 'search']:
                    updated_lines.append(line)
                    header_processed = True
                    continue
                
                # Check if this line contains the processed prompt
                row_data = next(csv.reader([line_content]))
                if row_data and row_data[0].strip() != processed_prompt:
                    updated_lines.append(line)
            
            # Write the updated content back to the file
            with open(self.csv_file, 'w', encoding='utf-8', newline='') as file:
                file.writelines(updated_lines)
            
            print(f"Removed '{processed_prompt}' from CSV file")
            
        except Exception as e:
            print(f"Error removing line from CSV: {e}")
            print("Continuing with next prompt...")
    
    def create_folder(self, folder_name):
        """Create folder for storing images"""
        # Clean folder name (remove invalid characters)
        clean_name = re.sub(r'[<>:"/\\|?*]', '_', folder_name)
        folder_path = os.path.join(self.base_folder, clean_name)
        
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)
        
        return folder_path
    
    def search_pinterest(self, query):
        """Navigate to Pinterest search results"""
        search_url = f"https://www.pinterest.com/search/pins/?q={query.replace(' ', '%20')}"
        
        try:
            self.driver.get(search_url)
            time.sleep(3)  # Wait for page to load
            return True
        except Exception as e:
            print(f"Error navigating to Pinterest: {e}")
            return False
    
    def scroll_and_load_images(self):
        """Scroll down to load more images"""
        last_height = self.driver.execute_script("return document.body.scrollHeight")
        
        for _ in range(3):  # Scroll 3 times to load more content
            self.driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
            time.sleep(2)
            
            new_height = self.driver.execute_script("return document.body.scrollHeight")
            if new_height == last_height:
                break
            last_height = new_height
    
    def get_image_urls(self):
        """Extract image URLs from Pinterest page"""
        image_urls = []
        
        try:
            # Wait for images to load
            WebDriverWait(self.driver, 10).until(
                EC.presence_of_element_located((By.TAG_NAME, "img"))
            )
            
            # Scroll to load more images
            self.scroll_and_load_images()
            
            # Find all image elements
            img_elements = self.driver.find_elements(By.CSS_SELECTOR, "img[src*='pinimg.com']")
            
            for img in img_elements[:self.images_per_prompt]:
                src = img.get_attribute("src")
                if src and "pinimg.com" in src:
                    # Get higher resolution version if available
                    if "/236x/" in src:
                        src = src.replace("/236x/", "/736x/")
                    elif "/474x/" in src:
                        src = src.replace("/474x/", "/736x/")
                    
                    image_urls.append(src)
            
        except TimeoutException:
            print("Timeout waiting for images to load")
        except Exception as e:
            print(f"Error extracting image URLs: {e}")
        
        return list(set(image_urls))  # Remove duplicates
    
    def download_image(self, url, folder_path, filename):
        """Download a single image"""
        try:
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
            }
            
            response = requests.get(url, headers=headers, timeout=10)
            response.raise_for_status()
            
            file_path = os.path.join(folder_path, filename)
            
            with open(file_path, 'wb') as file:
                file.write(response.content)
            
            return True
            
        except Exception as e:
            print(f"Error downloading image {filename}: {e}")
            return False
    
    def scrape_prompt(self, prompt):
        """Scrape images for a single prompt"""
        print(f"\nProcessing prompt: '{prompt}'")
        
        # Create folder for this prompt
        folder_path = self.create_folder(prompt)
        
        # Search Pinterest for the prompt
        if not self.search_pinterest(prompt):
            print(f"Failed to search Pinterest for '{prompt}'")
            return
        
        # Get image URLs
        image_urls = self.get_image_urls()
        
        if not image_urls:
            print(f"No images found for '{prompt}'")
            return
        
        print(f"Found {len(image_urls)} images for '{prompt}'")
        
        # Download images
        downloaded = 0
        for i, url in enumerate(image_urls[:self.images_per_prompt]):
            # Extract file extension from URL
            parsed_url = urlparse(url)
            ext = '.jpg'  # Default extension
            if '.' in parsed_url.path:
                ext = '.' + parsed_url.path.split('.')[-1]
            
            filename = f"{prompt.replace(' ', '_')}_{i+1}{ext}"
            
            if self.download_image(url, folder_path, filename):
                downloaded += 1
                print(f"Downloaded: {filename}")
            
            time.sleep(1)  # Be respectful to the server
        
        print(f"Successfully downloaded {downloaded} images for '{prompt}'")
        
        # Remove the processed prompt from CSV file
        self.remove_processed_line(prompt)
    
    def run(self):
        """Main method to run the scraper"""
        print("Starting Pinterest Scraper...")
        
        # Setup driver
        if not self.setup_driver():
            return
        
        # Create base folder
        if not os.path.exists(self.base_folder):
            os.makedirs(self.base_folder)
        
        # Process prompts one by one (read fresh each time to handle deletions)
        try:
            while True:
                # Read current prompts from CSV
                prompts = self.read_csv_prompts()
                
                if not prompts:
                    print("No more prompts found in CSV file! All done.")
                    break
                
                # Process the first prompt
                current_prompt = prompts[0]
                print(f"\nRemaining prompts: {len(prompts)}")
                self.scrape_prompt(current_prompt)
                time.sleep(2)  # Pause between prompts
                
        except KeyboardInterrupt:
            print("\nScraping interrupted by user")
        except Exception as e:
            print(f"Unexpected error: {e}")
        finally:
            if self.driver:
                self.driver.quit()
                print("\nBrowser closed. Scraping completed!")

# Usage example
if __name__ == "__main__":
    # Configuration
    CSV_FILE = "prompts_remain.csv"  # Path to your CSV file
    IMAGES_PER_PROMPT = 10    # Number of images to download per prompt
    BASE_FOLDER = "pinterest_images"  # Base folder to store images
    
    # Create and run scraper
    scraper = PinterestScraper(
        csv_file=CSV_FILE,
        images_per_prompt=IMAGES_PER_PROMPT,
        base_folder=BASE_FOLDER
    )
    
    scraper.run()
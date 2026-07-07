# Price Comparison Tool

# Step 1: Import Required Libraries

import re                           # For cleaning text and extracting numbers from prices
import time                         # For adding delays between requests to avoid being blocked
import random
import logging                      # For tracking errors and debugging information
from datetime import datetime       # For timestamping saved files
from urllib.parse import quote      # For encoding special characters in URLs
from typing import Optional, Dict, List, Tuple  # For type hints (better code documentation)
import warnings                     # For suppressing unnecessary warning messages
import os
import pandas as pd                 # For data manipulation and analysis (DataFrames)
import numpy as np                  # For numerical operations and statistics
import matplotlib.pyplot as plt     # For creating charts and visualizations
from selenium import webdriver      # For automating web browser interaction
from selenium.webdriver.common.by import By  # For finding HTML elements
from selenium.webdriver.chrome.service import Service  # For managing Chrome driver
from selenium.webdriver.chrome.options import Options  # For Chrome browser settings
from selenium.webdriver.support.ui import WebDriverWait  # For waiting for elements to load
from selenium.webdriver.support import expected_conditions as EC  # For defining wait conditions
from webdriver_manager.chrome import ChromeDriverManager  # For automatic Chrome driver management

# Try to import fuzzy matching for better product comparison
try:
    from fuzzywuzzy import fuzz      # For comparing product names even if not exactly same
    FUZZY_AVAILABLE = True           # Set flag to True so other functions know fuzzy matching is available
    print("Fuzzy matching enabled")
except ImportError:                 # If fuzzywuzzy is not installed
    FUZZY_AVAILABLE = False         # Set flag to False - will use simpler matching instead
    print("Install fuzzywuzzy for better matching: pip install fuzzywuzzy")

# Suppress warning messages for cleaner output
warnings.filterwarnings('ignore')

print("Libraries imported successfully")

# Step 2: Configuration Setup

# Setup logging system to track what's happening
logging.basicConfig(
    level=logging.INFO,              # Show info, warnings, and errors
    format='%(asctime)s - %(levelname)s - %(message)s'  # Include timestamp with each log
)
logger = logging.getLogger(__name__)  # Create logger instance

# %%

MIN_LAPTOP_PRICE = 20000     # Minimum price for a laptop
MAX_LAPTOP_PRICE = 300000    # Maximum price for a laptop
MAX_RETRIES = 3              # How many times to retry if scraping fails
RETRY_DELAY = 5              # Seconds to wait between retries

# Keywords that indicate product is NOT a laptop
ACCESSORY_KEYWORDS = [
    'keyboard', 'mouse', 'usb', 'cable', 'adapter', 'bag', 'case',
    'screen', 'monitor', 'headphone', 'speaker', 'charger', 'stand'
]

print("Configuration loaded")

# Step 3: Utility Functions

def safe_find_element(driver, by: By, selector: str, timeout: int = 10) -> Optional:
    # Safely finds a single HTML element - returns None if not found instead of crashing.
    try:
        # Wait up to 'timeout' seconds for element to appear in the DOM
        element = WebDriverWait(driver, timeout).until(
            EC.presence_of_element_located((by, selector))    # Condition: element exists in DOM
        )
        return element         # Return the found element
    except Exception as e:     # If element not found or timeout occurs
        logger.debug(f"Element not found: {selector} - {e}")     # Log debug message
        return None      # Return None instead of crashing

def safe_find_elements(driver, by: By, selector: str, timeout: int = 10) -> List:
    # Safely finds multiple HTML elements - returns empty list if none found.
    try:
        # First wait for at least one element to exist in the DOM
        WebDriverWait(driver, timeout).until(
            EC.presence_of_element_located((by, selector))
        )
        # Then return all matching elements
        return driver.find_elements(by, selector)
    except:          # If no elements found or timeout occurs
        return []    # Return empty list instead of crashing


def validate_product_data(product_dict: Dict) -> bool:
    # Validates that a product dictionary contains valid laptop data before adding to dataset.
    if 'Product Name' not in product_dict or not product_dict['Product Name']:   # Check required fields exist and are not empty
        return False     # Missing or empty product name
    
    if 'Price_Num' not in product_dict or not product_dict['Price_Num']:
        return False     # Missing or zero/empty price
    
    # Validate price range - must be between MIN and MAX laptop prices
    if product_dict['Price_Num'] < MIN_LAPTOP_PRICE or product_dict['Price_Num'] > MAX_LAPTOP_PRICE:
        return False      # Price outside reasonable laptop range
    
    # Validate product name length (too short = probably not a laptop)
    if len(product_dict['Product Name']) < 5:
        return False     # Name too short to be a real laptop product
    
    # Check for accessory keywords in product name
    name_lower = product_dict['Product Name'].lower()         # Convert to lowercase for case-insensitive checking
    for keyword in ACCESSORY_KEYWORDS:                        # Loop through each accessory keyword
        if keyword in name_lower:                             # If keyword found in product name
            return False  # This is an accessory, not a laptop - reject it
    
    return True  # Product is a valid laptop


def retry_on_failure(func):
    # Decorator function that automatically retries a function if it fails.

    def wrapper(*args, **kwargs):     # Inner wrapper function that replaces the original
        for attempt in range(MAX_RETRIES):      # Loop up to MAX_RETRIES (3) times
            try:
                result = func(*args, **kwargs)      # Call the original function with all arguments
                # Check if result is valid (not empty DataFrame)
                if result is not None and (not isinstance(result, pd.DataFrame) or not result.empty):
                    return result   # Return successful result immediately
                
                logger.warning(f"Attempt {attempt + 1} failed for {func.__name__}, retrying...")     # Log retry
                time.sleep(RETRY_DELAY)       # Wait before retrying to avoid rate limiting
            except Exception as e:            # If exception occurs during function execution
                logger.error(f"Attempt {attempt + 1} error: {e}")       # Log the error
                if attempt == MAX_RETRIES - 1:  # Last attempt failed
                    raise             # Give up and raise the error
                time.sleep(RETRY_DELAY)            # Wait before retrying
        return None                        # Return None if all attempts fail
    return wrapper                         # Return the wrapper function to replace the original


def random_delay(min_seconds=1.5, max_seconds=4, reason=""):
    #Random delay to avoid bot detection
    delay = random.uniform(min_seconds, max_seconds)
    if reason:
        print(f"{reason} - Waiting {delay:.1f}s")
    else:
        print(f"Waiting {delay:.1f}s")
    time.sleep(delay)


def clean_price(price_text: str) -> Optional[float]:           # Extracts numeric price from messy text
    if not price_text or price_text == 'N/A':                  # Check for empty or placeholder values
        return None                                            # Can't extract price from invalid input
    
    # Remove currency symbols and commas
    cleaned = re.sub(r'[₹Rs,\s]', '', str(price_text))         # Replace all matched characters with empty string
    
    # Extract numbers using regex
    numbers = re.findall(r'\d+', cleaned)                      # \d+ matches one or more digits

    if numbers:                                                # If we found at least one number
        try:
            # Join all digit sequences together and convert to float
            return float(''.join(numbers))  # Join all digits and convert to float
        except:                             # If conversion fails
            return None
    return None                             # No numbers found in price text

print("Utility functions ready")


# Step 4: Amazon Scraper Class

class AmazonScraper:
    # Extracts laptop product information from Amazon India (amazon.in)
    def __init__(self):
        self.driver = None      # Initialize driver as None - will be set when scraping starts
    
    def setup_driver(self):
        options = Options()     # Create Chrome options object for browser configuration
        options.add_argument('--headless')  # Run without opening browser window
        options.add_argument('--disable-blink-features=AutomationControlled')  # Hide automation
        options.add_experimental_option("excludeSwitches", ["enable-automation"])  # Remove automation flag
        options.add_argument('--no-sandbox')  # Required for some systems
        options.add_argument('--disable-dev-shm-usage')  # Overcome limited resource issues
        options.add_argument('--window-size=1920,1080')  # Set screen size
        options.add_argument('--user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 Chrome/120.0.0.0')  # Fake browser identity
        
        service = Service(ChromeDriverManager().install())  # Auto-install Chrome driver
        driver = webdriver.Chrome(service=service, options=options)
        # Hide the webdriver property from JavaScript detection (makes bot detection harder)
        driver.execute_script("Object.defineProperty(navigator, 'webdriver', {get: () => undefined})")  # Hide webdriver property
        return driver     # Return configured driver
    
    def extract_product_name(self, product) -> str:
        # List of possible CSS selectors where product names might be (Amazon changes structure often)
        name_selectors = [
            'h2 a span',           # Common pattern
            'h2 a',                # Alternative pattern
            '.a-size-medium',      # Another class name
            '.a-size-base-plus',   # Another class name
            '.a-text-normal'       # Another class name
        ]
        
        for selector in name_selectors:                     # Try each selector until we find a name
            try:
                elements = product.find_elements(By.CSS_SELECTOR, selector)    # Find all elements matching selector
                for elem in elements:              # Check each element found
                    name = elem.text.strip()       # Get text content and remove extra whitespace
                    # Validate name: not empty, longer than 5 chars, not a store name
                    if name and len(name) > 5 and not name.startswith('Visit the'):
                        return name         # Return the first valid name found
            except:
                continue
        return "N/A"  # Return if no name found
    
    def extract_price(self, product) -> Optional[float]:
        # List of possible CSS selectors where prices might be found
        price_selectors = [
            'span.a-price-whole',      # Main price element
            '.a-price .a-offscreen',   # Alternative price location
            '.a-price .a-price-whole'  # Another pattern
        ]
        
        for selector in price_selectors:           # Try each selector
            try:
                elem = product.find_element(By.CSS_SELECTOR, selector)      # Find the price element
                price_text = elem.text.strip()             # Get price text and clean whitespace
                if price_text:                            # If we found some text
                    price_num = clean_price(price_text)   # Use clean_price function to extract numeric price
                    if price_num:                          # If extraction succeeded
                        return price_num                  # Return the numeric price
            except:
                continue                                 # If this selector fails, try the next one
        return None                                      # Return None if no price found
    
    def extract_rating(self, product) -> str:
        try:
            # Amazon stores ratings in span with class "a-icon-alt" as text like "4.2 out of 5 stars"
            elem = product.find_element(By.CSS_SELECTOR, "span.a-icon-alt")     
            rating_text = elem.get_attribute('textContent')                   # Get the rating text content
            rating_match = re.search(r'(\d+\.?\d*)', rating_text)  # Extract number like 4.2
            if rating_match:
                return rating_match.group(1)                      # Return just the number part
        except:
            pass                            # If rating element not found, continue to return N/A
        return "N/A"                        # Return N/A if rating extraction fails
    
    @retry_on_failure  # Automatically retry if scraping fails
    def scrape(self, query: str, limit: int = 10) -> pd.DataFrame:   # Main scraping method for Amazon.
        print(f"\nSearching Amazon for: {query}")                    # Show what we're searching for
        
        try:
            self.driver = self.setup_driver()                        # Create and configure Chrome browser
            # Build Amazon search URL with query + "laptop" to ensure laptop results
            url = f"https://www.amazon.in/s?k={query.replace(' ', '+')}+laptop"
            self.driver.get(url)                    # Navigate to the search results page
            
            # Wait for products to load
            WebDriverWait(self.driver, 10).until(
                EC.presence_of_element_located((By.CSS_SELECTOR, '[data-component-type="s-search-result"]'))
            )
            
            # Find product containers
            products = self.driver.find_elements(By.CSS_SELECTOR, '[data-component-type="s-search-result"]')
            if not products:
                products = self.driver.find_elements(By.CSS_SELECTOR, '.s-result-item')
            
            if not products:        # If still no products found
                print("No products found on Amazon")       # Inform user
                return pd.DataFrame()                     # Return empty DataFrame
            
            print(f"Found {len(products)} products")       # Show how many products were found
            
            product_data = []                        # List to store valid product dictionaries
            for i, product in enumerate(products[:limit]):
                if i > 0:
                   random_delay(1.5, 3.5, "Between products")        # Process only up to limit products
                try:
                    name = self.extract_product_name(product)        # Extract product name
                    if name == "N/A":                                 # Skip if no valid name found
                        continue
                    
                    price_num = self.extract_price(product)          # Extract product price
                    if not price_num:                                # Skip if no valid price found
                        continue
                    
                    rating = self.extract_rating(product)           # Extract rating 
                    
                    # Create product dictionary with both display and numeric price
                    product_dict = {
                        'Product Name': re.sub(r'\s+', ' ', name)[:80],  # Clean extra spaces, limit length
                        'Price(₹)': f"₹{price_num:,.0f}",  # Formatted for display
                        'Price_Num': price_num,  # Numeric for calculations
                        'Rating': rating,        # Rating string or "N/A"
                        'Platform': 'Amazon'     # Platform identifier
                    }
                    
                    # Validate before adding to dataset
                    if validate_product_data(product_dict):
                        product_data.append(product_dict)
                        print(f"({len(product_data)}/{limit}) {name[:40]}... - ₹{price_num:,.0f}")
                        
                except Exception as e:          # Handle errors for individual products without stopping
                    logger.debug(f"Error extracting product: {e}")       # Log for debugging
                    continue          # Skip this product and continue with next
            
            df = pd.DataFrame(product_data)             # Convert list of dictionaries to DataFrame
            print(f"Amazon: {len(df)} laptops extracted (target: {limit})")    # Show extraction summary
            return df         # Return the DataFrame
            
        except Exception as e:              # Handle major errors
            print(f"Amazon error: {e}")     # Show error to user
            return pd.DataFrame()           # Return empty DataFrame
        finally:
            if self.driver:
                self.driver.quit()  # Always close browser to free resources

print("Amazon scraper ready")

# Step 5: Flipkart Scraper Class

class FlipkartScraper:
    
    def __init__(self):
        self.driver = None     # Initialize driver as None - will be set when scraping starts
    
    def setup_driver(self):
        options = Options()     # Create Chrome options object
        options.add_argument('--headless')                # Run without visible browser window
        options.add_argument('--disable-blink-features=AutomationControlled')        # Hide automation signs
        options.add_experimental_option("excludeSwitches", ["enable-automation"])     # Remove automation flag
        options.add_argument('--no-sandbox')                  # Required for some systems
        options.add_argument('--disable-dev-shm-usage')       # Handle resource limitations
        options.add_argument('--window-size=1920,1080')        # Standard desktop resolution
        options.add_argument('--user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 Chrome/120.0.0.0')  # Real browser user agent
        
        service = Service(ChromeDriverManager().install())      # Auto-install ChromeDriver
        driver = webdriver.Chrome(service=service, options=options)      # Create Chrome driver
        driver.execute_script("Object.defineProperty(navigator, 'webdriver', {get: () => undefined})")   # Hide webdriver property from JavaScript detection
        return driver          # Return configured driver
    
    def close_login_popup(self):
        # Attempts to close the login/signup popup that appears on Flipkart.
        try:
            # Look for the close button 
            close_btn = WebDriverWait(self.driver, 5).until(
                EC.element_to_be_clickable((By.CSS_SELECTOR, "button._2KpZ6l"))
            )
            close_btn.click()          # Click the close button
            print("Closed login popup")     # Confirm popup was closed
             # Wait for popup to disappear
            WebDriverWait(self.driver, 3).until(
                EC.invisibility_of_element_located((By.CSS_SELECTOR, "button._2KpZ6l"))
            )                 
        except: 
            pass  # Popup not present, continue
    
    def extract_product_name(self, product) -> str:
        # List of possible CSS selectors for product names on Flipkart
        name_selectors = [
            "a._1fQZEK",      # Common pattern
            ".s1Q9rs",        # Another pattern
            "._4rR01T",       # Another pattern
            ".IRpwTa",        # Another pattern
            "a[title]"        # Title attribute
        ]
        
        for selector in name_selectors:               # Try each selector
            try:
                elements = product.find_elements(By.CSS_SELECTOR, selector)         # Find all matching elements
                for elem in elements:              # Check each element
                    name = elem.text.strip()       # Get text and clean whitespace
                    if name and len(name) > 10 and name != "Add to Compare":     # Validate name: not empty, longer than 10 chars, not a UI element text
                        return name               # Return first valid name found
            except:
                continue
        
        # Fallback: Try to find laptop name in the product's text content
        try:
            product_text = product.text               # Get all text from the product element
            lines = product_text.split('\n')
            for line in lines[:3]:                    # Check first 3 lines only
                line = line.strip()                   # Clean whitespace
                 # Check if line looks like a laptop name
                if len(line) > 15 and line not in ["Add to Compare", "View Details", "Buy Now"]:
                 # Check if it contains laptop-related keywords
                    laptop_keywords = ['inspiron', 'g15', 'xps', 'latitude', 'dell', 'core', 'i3', 'i5', 'i7', 'ryzen']
                    if any(keyword in line.lower() for keyword in laptop_keywords):
                        return line         # Return the line that contains laptop keywords
        except:
            pass         # If all fallbacks fail, continue to return default
        
        return "Unknown Laptop"                     # Return default if no name found
    
    def extract_price(self, product) -> Optional[float]:
         # Wait for price elements to be present
        try:
            WebDriverWait(self.driver, 5).until(
                EC.presence_of_element_located((By.CSS_SELECTOR, "div._30jeq3, div._25b18c"))
            )
        except:
            pass  # Continue with fallback methods if timeout



        # First try: Search entire product text for ₹ symbol
        try:
            product_text = product.text       
            pattern = r'[₹]([\d,]+(?:\.\d{2})?)'  # Regex to find ₹ followed by numbers
            matches = re.findall(pattern, product_text)
            for match in matches:
                price_num = clean_price(f"₹{match}")
                if price_num:
                    return price_num
        except:
            pass
        
        # Second try: Look in specific price elements
        price_selectors = ["div._30jeq3", "div._25b18c"]
        for selector in price_selectors:
            try:
                elem = product.find_element(By.CSS_SELECTOR, selector)
                price_text = elem.text.strip()
                if price_text:
                    price_num = clean_price(price_text)
                    if price_num:
                        # Fix common Flipkart price errors (extra zero)
                        if 100000 < price_num < 200000 and price_num % 1000 > 900:
                            return price_num / 10  # Remove the extra digit
                        return price_num
            except:
                continue
        return None
    
    def extract_rating(self, product) -> str:
        # Extracts rating from Flipkart product HTML element.
        rating_selectors = ["div._3LWZlK", "div.MKiFS6"]           # Flipkart rating classes
        for selector in rating_selectors:
            try:
                elem = product.find_element(By.CSS_SELECTOR, selector)       # Find rating element
                rating = elem.text.strip()                     # Get rating text
                if rating:
                    return rating
            except:
                continue          # Try next selector if this fails
        return "N/A"
    
    @retry_on_failure                   # Decorator - automatically retry if scraping fails
    def scrape(self, query: str, limit: int = 10) -> pd.DataFrame:
        
        print(f"\nSearching Flipkart for: {query}")
        
        try:
            self.driver = self.setup_driver()                     # Create and configure Chrome browser
            url = f"https://www.flipkart.com/search?q={quote(query)}+laptop"
            self.driver.get(url)                   # Navigate to search results
            # Wait for page to load instead of static sleep
            WebDriverWait(self.driver, 10).until(
                EC.presence_of_element_located((By.CSS_SELECTOR, "div[data-id], div._1AtVbE"))
            )
            
            self.close_login_popup()  # Remove popup if present
            
            # Dynamic scrolling with height checking
            last_height = self.driver.execute_script("return document.body.scrollHeight")
            scroll_attempts = 0
            max_scroll_attempts = 3
            
            while scroll_attempts < max_scroll_attempts:
                # Scroll down
                self.driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
                random_delay(1.5, 3, "Scrolling for products")  # Wait for content to load
                
                # Check if new content loaded
                new_height = self.driver.execute_script("return document.body.scrollHeight")
                if new_height == last_height:
                    break  # No more content to load
                last_height = new_height
                scroll_attempts += 1
            
            # Wait for products to be present before finding them
            try:
                WebDriverWait(self.driver, 10).until(
                    EC.presence_of_element_located((By.CSS_SELECTOR, "div[data-id]"))
                )
            except:
                print("Timeout waiting for products to load")
            

            # Find product containers
            products = self.driver.find_elements(By.CSS_SELECTOR, "div[data-id]")
            if not products:
                products = self.driver.find_elements(By.CSS_SELECTOR, "div._1AtVbE div")
            
            if not products:
                print("No products found on Flipkart")
                return pd.DataFrame()
            
            print(f"Found {len(products)} potential products")
            
            product_data = []
            seen_prices = set()  # Track prices to avoid duplicates
            laptop_keywords = ['inspiron', 'g15', 'xps', 'latitude', 'vostro', 'precision', 'alienware', 'notebook']
            
            for i, product in enumerate(products):
                if len(product_data) >= limit:  # Stop when we have enough
                    break
    
                if i > 0 and len(product_data) > 0:
                    random_delay(2, 5, "Between products")
                    
                try:
                    name = self.extract_product_name(product)        # Extract product name
                    
                    if name == "Unknown Laptop" or len(name) < 5:
                        continue
                    
                    price_num = self.extract_price(product)           # Extract product price
                    if not price_num:
                        continue
                    
                    # Avoid duplicate prices (often same product repeated)
                    if price_num in seen_prices:
                        continue
                    seen_prices.add(price_num)
                    
                    # Check if this is actually a laptop (not an accessory)
                    is_laptop = (MIN_LAPTOP_PRICE <= price_num <= MAX_LAPTOP_PRICE)
                    has_keyword = any(keyword in name.lower() for keyword in laptop_keywords)
                    
                    if is_laptop or has_keyword:
                        rating = self.extract_rating(product)          # Extract rating
                        
                        product_dict = {
                            'Product Name': re.sub(r'\s+', ' ', name)[:80],
                            'Price(₹)': f"₹{price_num:,.0f}",
                            'Price_Num': price_num,
                            'Rating': rating if rating else 'N/A',
                            'Platform': 'Flipkart'
                        }
                        # Validate before adding to dataset
                        if validate_product_data(product_dict):
                            product_data.append(product_dict)
                            print(f"({len(product_data)}/{limit}) {name[:40]}... - ₹{price_num:,.0f}")
                    
                except Exception as e:
                    logger.debug(f"Error extracting product: {e}")
                    continue         # Skip this product and continue
            
            df = pd.DataFrame(product_data)
            df = df.drop_duplicates(subset=['Price_Num'])  # Final duplicate removal
            
            print(f"Flipkart: {len(df)} laptops extracted (target: {limit})")        # Show summary
            return df
            
        except Exception as e:
            print(f"Flipkart error: {e}")       # Show error to user
            return pd.DataFrame()               # Return empty DataFrame
        finally:
            if self.driver:
                self.driver.quit()           # Always close browser to free resources

print("Flipkart scraper ready")


# Step 6: Product Matcher Class (Cross-platform comparison)

class ProductMatcher:
   # Class that matches the same laptop products across Amazon and Flipkart.
    @staticmethod
    def extract_model_number(product_name: str) -> Optional[str]:
       # Extracts laptop model number from product name

        # List of regex patterns that might indicate a model number
        patterns = [
            r'([A-Za-z]+)\s+(\d+)',      # Pattern: Word followed by number
            r'([A-Z]+-\d+)',              # Pattern: Uppercase letters + dash + number
            r'(\d+[A-Za-z]*\s*Gen\d*)',   # Pattern: Generation indicator
            r'([A-Z][A-Z0-9]+-\d+)'       # Pattern: Mixed letters/numbers with dash
        ]
        
        for pattern in patterns:          # Try each pattern
            match = re.search(pattern, product_name, re.IGNORECASE)        # Search with case-insensitive flag
            if match:
                return match.group(0).upper()            # Return matched model number in uppercase for consistency
        return None            # Return None if no model number found
    
    @staticmethod
    def calculate_similarity(name1: str, name2: str) -> float:
        # Calculates similarity between two product names (0.0 to 1.0).

        # Clean names (remove punctuation, lowercase)
        name1_clean = re.sub(r'[^\w\s]', '', name1.lower())     # Remove anything not alphanumeric or space
        name2_clean = re.sub(r'[^\w\s]', '', name2.lower())
        
        # Use fuzzy matching if library is available
        if FUZZY_AVAILABLE:
            # token_set_ratio ignores word order and matches even if words are missing
            return fuzz.token_set_ratio(name1_clean, name2_clean) / 100           # Convert percentage to decimal
        
        # Fallback to simple word overlap
        words1 = set(name1_clean.split())      # Convert to set of unique words
        words2 = set(name2_clean.split())
        
        if not words1 or not words2:          # If either set is empty
            return 0                    # No similarity
        
        # Jaccard similarity: intersection size divided by union size
        intersection = words1.intersection(words2)           # Words that appear in both
        union = words1.union(words2)              # All unique words from both
        
        return len(intersection) / len(union)      # Return ratio as similarity score
    
    def find_matching_products(self, amazon_df: pd.DataFrame, flipkart_df: pd.DataFrame, 
                              threshold: float = 0.6) -> pd.DataFrame:
        # Finds products that appear on both Amazon and Flipkart.
        matches = []            # List to store matching product information
        
        # For each Amazon product, find the best matching Flipkart product
        for _, amazon_row in amazon_df.iterrows():        # Iterate over each Amazon product
            best_match = None                # Track the best matching Flipkart product
            best_score = 0                   # Track the highest similarity score
            
            # Compare with each Flipkart product
            for _, flipkart_row in flipkart_df.iterrows():           # Calculate name similarity
                similarity = self.calculate_similarity(
                    amazon_row['Product Name'], 
                    flipkart_row['Product Name']
                )
                
                # Also check model numbers for higher confidence
                amazon_model = self.extract_model_number(amazon_row['Product Name'])
                flipkart_model = self.extract_model_number(flipkart_row['Product Name'])
                
                # If model numbers match, that's a very strong indicator of same product
                if amazon_model and flipkart_model and amazon_model == flipkart_model:
                    similarity = 0.9  # High confidence match

                # Update best match if this similarity is higher
                if similarity > best_score:
                    best_score = similarity
                    best_match = flipkart_row
            
            # If we found a good match (above threshold)
            if best_score >= threshold and best_match is not None:
                # Calculate absolute price difference
                price_diff = abs(amazon_row['Price_Num'] - best_match['Price_Num'])
                matches.append({
                    'Product': amazon_row['Product Name'][:60],        # Product name (truncated to 60 chars)
                    'Amazon_Price': amazon_row['Price_Num'],           # Amazon price (numeric)
                    'Flipkart_Price': best_match['Price_Num'],         # Flipkart price (numeric)
                    'Price_Difference': price_diff,                    # Absolute difference between prices
                    'Better_Platform': 'Amazon' if amazon_row['Price_Num'] < best_match['Price_Num'] else 'Flipkart',  # Which is cheaper
                    'Savings': price_diff,              # How much you save by buying from cheaper platform
                    'Match_Confidence': f"{best_score:.0%}"        # Format as percentage
                })
        
        return pd.DataFrame(matches)      # Return DataFrame of matched products

print("Product matcher ready")

# Step 7: Analytics and Visualisation Functions

def create_basic_chart(combined_df: pd.DataFrame, query: str, output_dir: str = ".") -> Optional[str]:
    # Creates a horizontal bar chart showing laptop prices (cheapest at top).
    if combined_df.empty:             # Check if there's data to plot
        return None
    
    plt.close('all')             # Close any existing plots to free memory
    
    # Prepare data for chart (top 15 laptops)
    chart_data = combined_df.head(15).copy()            # Work with copy to avoid modifying original

     # Create shortened names for display
    chart_data['Short_Name'] = chart_data['Product Name'].apply(
        lambda x: x[:35] + '...' if len(str(x)) > 38 else str(x)
    )
    
    # Create the chart with appropriate size based on number of products
    fig, ax = plt.subplots(figsize=(12, max(6, len(chart_data) * 0.4)))
    
    # Set up y-axis positions 
    y_pos = range(len(chart_data))
    colors = ['#FF9900' if p == 'Amazon' else '#2874F0' for p in chart_data['Platform']]  # Amazon orange, Flipkart blue
    # Create horizontal bar chart
    bars = ax.barh(y_pos, chart_data['Price_Num'], color=colors, alpha=0.8)
    
    # Labels and title
    ax.set_yticks(y_pos)
    ax.set_yticklabels(chart_data['Short_Name'], fontsize=9)     # Use shortened names
    ax.set_xlabel('Price (₹)', fontsize=12, fontweight='bold')   # X-axis label
    ax.set_title(f'Laptop Prices: {query.upper()} (Lowest to Highest)', 
                 fontsize=14, fontweight='bold')     # Title
    ax.invert_yaxis()  # Show cheapest at top
    
    # Add price labels on each bar
    for bar, price in zip(bars, chart_data['Price_Num']):
        ax.text(bar.get_width() + (price * 0.01), bar.get_y() + bar.get_height()/2,
               f'₹{price:,.0f}', va='center', fontsize=8)      # Label with formatted price
    
    # Add legend
    from matplotlib.patches import Patch             # Import for custom legend patches
    legend_elements = [
        Patch(facecolor='#FF9900', label='Amazon', alpha=0.8),
        Patch(facecolor='#2874F0', label='Flipkart', alpha=0.8)
    ]
    ax.legend(handles=legend_elements, loc='lower right')            # Place legend at bottom righ
    ax.grid(axis='x', alpha=0.3)         
    
    plt.tight_layout()             # Adjust layout to prevent label cutoff
    
    # Save the chart
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")             # Create unique timestamp
    filename = os.path.join(output_dir, f'price_comparison_{query.replace(" ", "_")}_{timestamp}.png')          # Safe filename
    plt.savefig(filename, dpi=150, bbox_inches='tight')
    plt.close()               # Close plot to free memory
    
    print(f"Basic chart saved: {filename}")       # Confirm save
    return filename             # Return filename for reference


def create_advanced_charts(amazon_df: pd.DataFrame, flipkart_df: pd.DataFrame, query: str, output_dir: str = ".") -> Optional[str]:
    
    if amazon_df.empty and flipkart_df.empty:
        return None
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # GRAPH 1: Price Distribution Histogram (shows how prices are distributed)
    if not amazon_df.empty:
        axes[0,0].hist(amazon_df['Price_Num'], bins=15, alpha=0.7, label='Amazon', color='#FF9900', edgecolor='black')   # Histogram with 15 bins
    if not flipkart_df.empty:
        axes[0,0].hist(flipkart_df['Price_Num'], bins=15, alpha=0.7, label='Flipkart', color='#2874F0', edgecolor='black')
    axes[0,0].set_xlabel('Price (₹)')        # X-axis label
    axes[0,0].set_ylabel('Frequency')        # Y-axis label (number of products)
    axes[0,0].set_title('Price Distribution by Platform')           # Subplot title
    axes[0,0].legend()        # Show legend
    axes[0,0].grid(True, alpha=0.3)
    
    # GRAPH 2: Box Plot for Price Variability
    price_data = []          # List to hold price data for box plot
    labels = []           # Corresponding platform labels

    if not amazon_df.empty:
        price_data.append(amazon_df['Price_Num'])       # Add Amazon prices
        labels.append('Amazon')
    if not flipkart_df.empty:
        price_data.append(flipkart_df['Price_Num'])     # Add Flipkart prices
        labels.append('Flipkart')
    
    if price_data:
        # Create box plot (shows min, Q1, median, Q3, max, outliers)
        bp = axes[0,1].boxplot(price_data, labels=labels, patch_artist=True)
        # Color the boxes appropriately
        for patch, color in zip(bp['boxes'], ['#FF9900', '#2874F0'][:len(price_data)]):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)
        axes[0,1].set_ylabel('Price (₹)')     # Y-axis label
        axes[0,1].set_title('Price Variability (Box Plot)')          # Title
        axes[0,1].grid(True, alpha=0.3)
    
     # GRAPH 3: Percentile Comparison
    if not amazon_df.empty and not flipkart_df.empty:       # Need both platforms for comparison
        percentiles = [10, 25, 50, 75, 90]                  # Percentile values to calculate
         # Calculate price at each percentile for each platform
        amazon_percentiles = [np.percentile(amazon_df['Price_Num'], p) for p in percentiles]
        flipkart_percentiles = [np.percentile(flipkart_df['Price_Num'], p) for p in percentiles]
        
        x = range(len(percentiles))     # X-axis positions
        axes[1,0].plot(x, amazon_percentiles, 'o-', label='Amazon', color='#FF9900', linewidth=2, markersize=8)
        axes[1,0].plot(x, flipkart_percentiles, 's-', label='Flipkart', color='#2874F0', linewidth=2, markersize=8)
        axes[1,0].set_xticks(x)     # Set x-tick positions
        axes[1,0].set_xticklabels([f'{p}th' for p in percentiles])
        axes[1,0].set_ylabel('Price (₹)')           # Y-axis label

        axes[1,0].set_title('Price Percentile Comparison')     # Title
        axes[1,0].legend()       # Legend
        axes[1,0].grid(True, alpha=0.3)     # Grid
    
    # GRAPH 4: Price Curve Comparison
    if len(amazon_df) > 3 and len(flipkart_df) > 3:            # Need enough points for meaningful curve
         # Sort prices from lowest to highest for each platform
        sorted_amazon = amazon_df.sort_values('Price_Num')['Price_Num'].values
        sorted_flipkart = flipkart_df.sort_values('Price_Num')['Price_Num'].values
        
        # Plot price curves (index is rank, value is price)
        axes[1,1].plot(range(len(sorted_amazon)), sorted_amazon, 'o-', label='Amazon', color='#FF9900', alpha=0.7, linewidth=2)
        axes[1,1].plot(range(len(sorted_flipkart)), sorted_flipkart, 's-', label='Flipkart', color='#2874F0', alpha=0.7, linewidth=2)
        axes[1,1].set_xlabel('Product Rank (by price)')          # X-axis: cheapest to most expensive
        axes[1,1].set_ylabel('Price (₹)')         # Y-axis: price
        axes[1,1].set_title('Price Curve Comparison')       # Title
        axes[1,1].legend()        # Legend
        axes[1,1].grid(True, alpha=0.3)      # Grid
    else:
         # Show message if insufficient data
        axes[1,1].text(0.5, 0.5, 'Insufficient data for price curve', 
                      ha='center', va='center', transform=axes[1,1].transAxes)
        axes[1,1].set_title('Price Curve Comparison')
    
    # Main title for entire figure
    plt.suptitle(f'Advanced Price Analysis - {query.upper()}', fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    # Save the chart
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = os.path.join(output_dir, f'advanced_analysis_{query.replace(" ", "_")}_{timestamp}.png')
    plt.savefig(filename, dpi=150, bbox_inches='tight')
    plt.close()        # Close to free memory
    
    print(f"Advanced charts saved: {filename}")        # Confirm save
    return filename

def create_match_comparison_chart(matches_df, query, output_dir: str = "."):          # Creates a bar chart comparing prices of matched products across platforms.
   
    if matches_df.empty:              # Check if there are matched products to plot
        print("No matched products for comparison chart")
        return

    plt.figure(figsize=(12,6))        # Create new figure

    # Matched products
    chart_df = matches_df

    products = chart_df['Product']                  # Product names for x-axis labels
    amazon_prices = chart_df['Amazon_Price']        # Amazon prices
    flipkart_prices = chart_df['Flipkart_Price']    # Flipkart prices

    x = range(len(products))          # X-axis positions
    width = 0.35                      # Width of each bar

    # Create grouped bar chart
    # Amazon bars on left half of each position
    plt.bar(
        [i - width/2 for i in x],         # Offset left by half width
        amazon_prices,
        width=width,
        label='Amazon'
    )

    # Flipkart bars on right half of each position
    plt.bar(
        [i + width/2 for i in x],       # Offset right by half width
        flipkart_prices,
        width=width,
        label='Flipkart'
    )

    # Labels
    plt.xticks(
    ticks=list(x),
    labels=products,
    rotation=15,
    ha='right'
    )

    # Prevent label shifting
    plt.xlim(-0.5, len(products)-0.5)         # Set x-axis range with padding

    plt.ylabel("Price (₹)")        # Y-axis label
    plt.xlabel("Matched Laptop Models")          # X-axis label
 
    plt.title(f"Same Laptop Price Comparison - {query}")          # Chart title
    plt.legend()             # Show legend
    plt.tight_layout()       # Adjust layout

    # Save chart with query in filename
    filename = os.path.join(output_dir, f"matched_price_comparison_{query}.png")   
    plt.savefig(filename)        # Save file
    plt.close()                  # Close figure
    print(f"\nMatch comparison chart saved: {filename}")         # Confirm save


def generate_summary_stats(amazon_df: pd.DataFrame, flipkart_df: pd.DataFrame):
    # Prints summary statistics for both platforms and overall best deals.
    print("Summary Statistics")
    
    # Platform-specific statistics
    for df, platform, color in [(amazon_df, 'AMAZON', '🟠'), (flipkart_df, 'FLIPKART', '🔵')]:
        if not df.empty:             # Only print if platform has data
            print(f"\n{color} {platform}:")
            print(f"Products found: {len(df)}")          # Count of products
            print(f"Price range: ₹{df['Price_Num'].min():,.0f} - ₹{df['Price_Num'].max():,.0f}")
            print(f"Average price: ₹{df['Price_Num'].mean():,.0f}")    # Mean
            print(f"Median price: ₹{df['Price_Num'].median():,.0f}")   # Median (50th percentile)
            print(f"Std deviation: ₹{df['Price_Num'].std():,.0f}")     # Standard deviation - how spread out prices are
            print(f"Price variability (IQR): ₹{df['Price_Num'].quantile(0.75) - df['Price_Num'].quantile(0.25):,.0f}")     
            # Interquartile Range (IQR) - difference between 75th and 25th percentiles
    
    # Combined analysis - find cheapest laptops overall
    all_products = pd.concat([amazon_df, flipkart_df], ignore_index=True)      # Combine both DataFrames
    if not all_products.empty:
        print("Best Deals (Top 5 Cheapest)")
        
        cheapest = all_products.nsmallest(5, 'Price_Num')           # Get 5 cheapest products
        for i, row in cheapest.iterrows():     
            # Choose icon based on platform
            platform_icon = "🟠" if row['Platform'] == 'Amazon' else "🔵"
            print(f"\n{i+1}. {platform_icon} {row['Platform']}")         # Rank with icon
            print(f"{row['Product Name'][:60]}...")         # Product name (truncated)
            print(f"₹{row['Price_Num']:,.0f}")
            print(f"⭐ Rating: {row['Rating']}")

def save_summary_statistics(amazon_df, flipkart_df, query, timestamp, output_dir="."):
    #  Saves summary statistics to a CSV file
    stats_data = []           # List to store statistics dictionaries

    # Amazon statistics
    if not amazon_df.empty:
        stats_data.append({
            'Platform': 'Amazon',
            'Products Found': len(amazon_df),
            'Minimum Price': amazon_df['Price_Num'].min(),
            'Maximum Price': amazon_df['Price_Num'].max(),
            'Average Price': round(amazon_df['Price_Num'].mean(), 2),            # Round to 2 decimals
            'Median Price': round(amazon_df['Price_Num'].median(), 2),
            'Standard Deviation': round(amazon_df['Price_Num'].std(), 2),
            'IQR': round(              # Interquartile Range
                amazon_df['Price_Num'].quantile(0.75) -
                amazon_df['Price_Num'].quantile(0.25), 2
            )
        })

    # Flipkart statistics
    if not flipkart_df.empty:
        stats_data.append({
            'Platform': 'Flipkart',
            'Products Found': len(flipkart_df),
            'Minimum Price': flipkart_df['Price_Num'].min(),
            'Maximum Price': flipkart_df['Price_Num'].max(),
            'Average Price': round(flipkart_df['Price_Num'].mean(), 2),
            'Median Price': round(flipkart_df['Price_Num'].median(), 2),
            'Standard Deviation': round(flipkart_df['Price_Num'].std(), 2),
            'IQR': round(
                flipkart_df['Price_Num'].quantile(0.75) -
                flipkart_df['Price_Num'].quantile(0.25), 2
            )
        })

    # Combined statistics
    combined_df = pd.concat([amazon_df, flipkart_df], ignore_index=True)

    if not combined_df.empty:

        cheapest = combined_df.loc[combined_df['Price_Num'].idxmin()]
        expensive = combined_df.loc[combined_df['Price_Num'].idxmax()]

        stats_data.append({
            'Platform': 'OVERALL',
            'Products Found': len(combined_df),
            'Minimum Price': combined_df['Price_Num'].min(),
            'Maximum Price': combined_df['Price_Num'].max(),
            'Average Price': round(combined_df['Price_Num'].mean(), 2),
            'Median Price': round(combined_df['Price_Num'].median(), 2),
            'Standard Deviation': round(combined_df['Price_Num'].std(), 2),
            'IQR': round(
                combined_df['Price_Num'].quantile(0.75) -
                combined_df['Price_Num'].quantile(0.25), 2
            )
        })

    # Convert to DataFrame
    stats_df = pd.DataFrame(stats_data)

    # File name
    import os
    stats_file = os.path.join(output_dir, f"summary_statistics_{query}_{timestamp}.csv")

    # Save CSV
    stats_df.to_csv(stats_file, index=False)            # index=False prevents adding row numbers

    print(f"\nSummary statistics saved to: {stats_file}")          # Confirm save


# Step 8: Reporting and Output Functions

def create_comparison_table(amazon_df: pd.DataFrame, flipkart_df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    #Creates a sorted comparison table combining products from both platforms.
    combined = pd.concat([amazon_df, flipkart_df], ignore_index=True)    # Combine both DataFrames into one 
    
    if combined.empty:              # Check if there's any data
        print("No data to compare")
        return pd.DataFrame(), pd.DataFrame()     # Return empty DataFrames
    
    # Sort by numeric price (cheapest first)
    combined = combined.sort_values('Price_Num').reset_index(drop=True)    # drop=True removes old index
    # Add rank column starting from 1
    combined.insert(0, 'Rank', range(1, len(combined) + 1))
    
    # Create DataFrame with selected columns
    display_df = combined[['Rank', 'Platform', 'Product Name', 'Price(₹)', 'Rating']].copy()
    # Rename columns for better readability
    display_df.columns = ['Rank', 'Platform', 'Product', 'Price', 'Rating']
    
    return display_df, combined       # Return both display and full DataFrames

def generate_final_report(amazon_df: pd.DataFrame, flipkart_df: pd.DataFrame, 
                          matches_df: pd.DataFrame, query: str):             #Generates final recommendations based on the comparison.
    
    print("Final Verdict and Recommendations")
    
    # 1. Average price comparison between platforms
    if not amazon_df.empty and not flipkart_df.empty:
        avg_amazon = amazon_df['Price_Num'].mean()        # Average Amazon price
        avg_flipkart = flipkart_df['Price_Num'].mean()    # Average Flipkart price
        
        print(f"\nPrice Comparison:")
        if avg_amazon < avg_flipkart:
            savings = avg_flipkart - avg_amazon           # How much cheaper Amazon is
            print(f"Amazon is cheaper on average by ₹{savings:,.0f}")
        elif avg_flipkart < avg_amazon:
            savings = avg_amazon - avg_flipkart           # How much cheaper Flipkart is
            print(f"Flipkart is cheaper on average by ₹{savings:,.0f}")
        else:
            print(f"Both platforms have similar average prices")
    
    # 2. Matched products analysis (same laptop on both sites)
    if not matches_df.empty:
        print(f"\nDirect Product Comparison:")
        print(f"Found {len(matches_df)} matching products across platforms")
        
        # Count which platform is cheaper for matched products
        amazon_cheaper_count = len(matches_df[matches_df['Better_Platform'] == 'Amazon'])
        flipkart_cheaper_count = len(matches_df[matches_df['Better_Platform'] == 'Flipkart'])
        total_matches = len(matches_df)
        
        if total_matches > 0:     # Avoid division by zero
            amazon_percentage = (amazon_cheaper_count / total_matches) * 100
            flipkart_percentage = (flipkart_cheaper_count / total_matches) * 100
            
            print(f"Amazon cheaper for {amazon_cheaper_count}/{total_matches} products ({amazon_percentage:.0f}%)")
            print(f"Flipkart cheaper for {flipkart_cheaper_count}/{total_matches} products ({flipkart_percentage:.0f}%)")
            
            avg_savings = matches_df['Savings'].mean()       # Average savings from buying cheaper platform
            print(f"Average savings on matched products: ₹{avg_savings:,.0f}")
            
            # Best matched deal (largest price difference)
            max_savings_row = matches_df.loc[matches_df['Savings'].idxmax()] if not matches_df.empty else None
            if max_savings_row is not None and max_savings_row['Savings'] > 0:
                print(f"\nBest Matched Deal:")
                print(f"Product: {max_savings_row['Product'][:50]}...")    # Truncated name
                print(f"{max_savings_row['Better_Platform']} saves ₹{max_savings_row['Savings']:,.0f}")
                print(f"(Amazon: ₹{max_savings_row['Amazon_Price']:,.0f} vs Flipkart: ₹{max_savings_row['Flipkart_Price']:,.0f})")
    
    # 3. Final recommendations
    print("Recommendations")
    
    if not amazon_df.empty and not flipkart_df.empty:
        # Find the single cheapest laptop overall
        all_products = pd.concat([amazon_df, flipkart_df], ignore_index=True)
        cheapest_overall = all_products.nsmallest(1, 'Price_Num').iloc[0]  # Get cheapest product
        print(f"\nBest overall deal: {cheapest_overall['Platform']}")
        print(f"{cheapest_overall['Product Name'][:60]}...")
        print(f"Price: {cheapest_overall['Price(₹)']}")

        # In matched products, recommend best savings on identical models
        if not matches_df.empty and not matches_df[matches_df['Savings'] > 0].empty:
            best_match = matches_df.loc[matches_df['Savings'].idxmax()]    # Largest saving
            print(f"\nBest platform-specific deal: Buy from {best_match['Better_Platform']}")
            print(f"Save ₹{best_match['Savings']:,.0f} compared to other platform")

print("Reporting functions ready")


# Step 9: Main Function

def main(query: str = "laptop", limit: int = 10, output_dir: str = "."):        #This is the entry point that gets called when the script runs.

    print("AMAZON vs FLIPKART LAPTOP PRICE COMPARISON")
    # Create timestamp for files
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Validate and clean parameters
    if not query or not isinstance(query, str):
        query = "laptop"
        print(f"Invalid query, using default: {query}")
    
    if not isinstance(limit, int) or limit < 1:
        limit = 10
        print(f"Invalid limit, using default: {limit}")
    
    if not isinstance(output_dir, str):
        output_dir = "."
    
    # Create output directory if it doesn't exist
    import os
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"\nSearching for: {query.upper()} laptops")
    print(f"Target: {limit} products per platform")
    print(f"Output directory: {output_dir}")
    
    print(f"\n Searching for: {query.upper()} laptops")
    
    # Scrape Amazon
    print("\n Amazon scraping in progress...")
    amazon_scraper = AmazonScraper()                            # Create Amazon scraper instance
    amazon_df = amazon_scraper.scrape(query, limit=limit)         
    # Random delay between platforms to avoid detection
    random_delay(3, 7, "Between Amazon and Flipkart")
    # Scrape Flipkart
    print("\n Flipkart scraping in progress...")
    flipkart_scraper = FlipkartScraper()                       # Create Flipkart scraper instance
    flipkart_df = flipkart_scraper.scrape(query, limit=limit)    
    

    # Generate summary statistics
    generate_summary_stats(amazon_df, flipkart_df)

    # Save statistics CSV
    save_summary_statistics(amazon_df, flipkart_df, query, timestamp, output_dir)

    # Find matching products across platforms
    matches_df = pd.DataFrame()                        # Initialize empty DataFrame for matches
    if not amazon_df.empty and not flipkart_df.empty:   
        print("Finding Matching Products...")
        
        matcher = ProductMatcher()                         # Create matcher instance
        matches_df = matcher.find_matching_products(amazon_df, flipkart_df, threshold=0.6)      # Find matches with 60% similarity threshold
        
        if not matches_df.empty:
            print(f"\n Found {len(matches_df)} matching products!")
            print("\n Matched Products Comparison:")
            # Select columns to display
            display_cols = ['Product', 'Amazon_Price', 'Flipkart_Price', 'Better_Platform', 'Savings']
            # Filter to only columns that exist in DataFrame
            available_cols = [col for col in display_cols if col in matches_df.columns]
            # Print with to_string to show all rows without truncation
            print(matches_df[available_cols].head(10).to_string(index=False))
            # Save matched products to separate CSV file
            matched_csv = os.path.join(output_dir, f'matched_products_{query}_{timestamp}.csv')
            matches_df.to_csv(matched_csv, index=False)
            print(f"\n Matched products saved to: {matched_csv}")
            
            # Create matched product comparison chart
            create_match_comparison_chart(matches_df, query, output_dir)
        else:
            print("\n No matching products found across platforms")
    
    # Create comparison table
    print("Complete Product List (Sorted by Price)")
    
    display_df, combined_df = create_comparison_table(amazon_df, flipkart_df)
    
    if not display_df.empty:
        # Print the comparison table to console
        print(display_df.to_string(index=False))
        
        # Save to CSV
        csv_file = os.path.join(output_dir, f'price_comparison_{query.replace(" ", "_")}_{timestamp}.csv')
        combined_df.to_csv(csv_file, index=False, encoding='utf-8-sig')               # utf-8-sig for Excel compatibility

        print(f"\n Data saved to: {csv_file}")
        
        create_basic_chart(combined_df, query, output_dir)
        create_advanced_charts(amazon_df, flipkart_df, query, output_dir)
        
        # Generate final report
        if not amazon_df.empty and not flipkart_df.empty:
            generate_final_report(amazon_df, flipkart_df, matches_df, query)
    
    else:
        # No valid products were found
        print("\n No valid laptop products found!")
        print("\n Possible reasons:")
        print("   • Website structure may have changed")
        print("   • No laptops found for your search term")
        print("   • Network connectivity issues")
    
    print("Analysis Complete! Check the generated CSV and PNG files.")

# Step 10: Run the Application
if __name__ == "__main__":
    import sys
    
    # Check if user provided command line arguments
    if len(sys.argv) > 1:
        query = sys.argv[1] if len(sys.argv) > 1 else "laptop"
        limit = int(sys.argv[2]) if len(sys.argv) > 2 else 10
        output_dir = sys.argv[3] if len(sys.argv) > 3 else "output"
        
        try:
            main(query=query, limit=limit, output_dir=output_dir)
        except KeyboardInterrupt:
            print("\nProcess interrupted by user")
        except Exception as e:
            print(f"\nUnexpected error: {e}")
            import traceback
            traceback.print_exc()
    else:
        # No arguments provided - use defaults
        print("Laptop Price Comparison Tool")
        print("\nYou can also run with command line arguments:")
        print("\nUsing default values...")
        
        try:
            main()  # Uses defaults: query="laptop", limit=10, output_dir="."
        except KeyboardInterrupt:
            print("\nProcess interrupted by user")
        except Exception as e:
            print(f"\nUnexpected error: {e}")
            import traceback
            traceback.print_exc()



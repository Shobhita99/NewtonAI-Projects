# PRICE COMPARISON TOOL

## 1. Description

This is a Python-based web scraping project that compares laptop prices from multiple e-commerce websites such as Amazon and Flipkart.

The tool extracts:
- Product Name
- Price
- Ratings
- Platform Information

It also generates:
- CSV output files
- PNG charts and visualizations
- Statistical analysis

All generated files are automatically saved inside the "output" folder.

## 2. Features
- Scrapes product data from Amazon
- Scrapes product data from Flipkart
- Compares prices across platforms
- Generates CSV reports
- Creates visualization graphs
- Performs statistical analysis
- Supports command-line product search

## 3. Installation

STEP 1: Open CMD / Terminal
- Navigate to the project folder: cd folder_name
- Example: cd Desktop/Price_Comparison_Tool

STEP 2: Install Required Libraries
pip install -r requirements.txt

## 4. How to run the project
- Run the following command in CMD: python price_comparison_tool.py "HP Pavilion"
- You can replace "HP Pavilion" with any laptop or product name.

Examples:
python price_comparison_tool.py "Dell Inspiron"
python price_comparison_tool.py "Lenovo IdeaPad"
python price_comparison_tool.py "ASUS Vivobook"

## 5. CSV contains
- Product names
- Prices
- Ratings
- Platform names
- Statistical calculations

## 6. PNG files Contain
- Price comparison charts
- Platform-wise graphs
- Price distribution visualization

## 7. Technologies used
- Python
- Selenium
- Pandas
- Matplotlib
- NumPy
- Web Scraping
- Fuzzy Matching

## 8. Workflow example
   cd Price_Comparison_Tool
   python price_comparison_tool.py "HP Pavilion"

## 9. Notes
- Internet connection is required.
- Google Chrome must be installed.
- ChromeDriver is automatically managed using webdriver-manager.
- Excessive scraping may temporarily trigger website restrictions.

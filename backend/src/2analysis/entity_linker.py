# src.analysis.entity_linker.py  

# Imports.
import json
import logging
import pandas as pd
from pathlib import Path
from typing import Dict, Set, List, Tuple
from datetime import datetime, timedelta
import yfinance as yf
from colorama import Fore, Style
import requests
from bs4 import BeautifulSoup
from collections import defaultdict
import os

# Local Imports.
from src.utils.path_config import DEBUG_DIR, ENTITY_CACHE_FILE, REFERENCES_DIR

# Initialize logger.
logger = logging.getLogger(__name__)

class EntityLinker:
    """Links entity mentions to known entities."""
    
    def __init__(self, cache_duration_days: int = 1):
        """Initialize the entity linker."""
        # Initialize cache file path
        self.cache_file = Path(os.path.join(REFERENCES_DIR, 'entity_cache.json'))
        self.cache_duration = timedelta(days=cache_duration_days)
        
        # Initialize common word tickers to filter out
        self.common_word_tickers = {
            'A', 'ALL', 'AM', 'AN', 'AND', 'ANY', 'ARE', 'AS', 'AT', 'BE', 'BY', 'CAN', 'DO', 'FOR', 'FROM', 'GO', 'HAS',
            'HAD', 'HE', 'HER', 'HERE', 'HIS', 'HOW', 'I', 'IF', 'IN', 'INTO', 'IS', 'IT', 'ITS', 'JOB', 'MAY', 'ME',
            'MOST', 'MY', 'NEW', 'NO', 'NOT', 'NOW', 'OF', 'ON', 'ONE', 'OR', 'OUR', 'OUT', 'OVER', 'SEE', 'SHE', 'SO',
            'SOME', 'THAN', 'THAT', 'THE', 'THEIR', 'THEM', 'THEN', 'THERE', 'THESE', 'THEY', 'THIS', 'TO', 'UP', 'US',
            'WAS', 'WE', 'WERE', 'WHAT', 'WHEN', 'WHO', 'WILL', 'WITH', 'YOU', 'YOUR'
        }
        
        # Initialize valid ETFs that should always be considered valid
        self.valid_etfs = {
            'SPY', 'QQQ', 'IWM', 'DIA', 'VTI', 'VOO', 'VEA', 'VWO', 'BND', 'AGG', 'TLT', 'IEF', 'SHY', 'LQD', 'HYG',
            'GLD', 'SLV', 'USO', 'UNG', 'DBC', 'VNQ', 'XLF', 'XLE', 'XLV', 'XLK', 'XLI', 'XLP', 'XLY', 'XLB', 'XLU',
            'XLRE', 'XLC', 'EEM', 'EFA', 'IEMG', 'IEFA', 'ARKK', 'ARKG', 'ARKW', 'ARKF', 'ARKQ', 'TQQQ', 'SQQQ', 'UVXY',
            'VXX', 'VIXY', 'SVXY', 'UVIX', 'SPXU', 'SPXS', 'SPXL', 'UPRO', 'TMF', 'TMV', 'TYD', 'TYO', 'UUP', 'UDN',
            'FXE', 'FXY', 'FXB', 'FXF', 'FXC', 'FXA'
        }
        
        # Initialize tracking
        self.match_history = []
        self.entities = {}
        
        # Initialize fallback entities
        self._fallback_entities = self._get_fallback_entities()
        
        # Load cached entities
        self._load_cached_entities()
        
        print(f"\n{Fore.CYAN}Initializing Entity Linker...{Style.RESET_ALL}")
        
        # Initialize confidence scores
        self.confidence_scores = defaultdict(list)
        
        print(f"{Fore.GREEN}✓ Entity Linker initialized successfully{Style.RESET_ALL}\n")
    
    def _load_cached_entities(self):
        """Load entity cache from file."""
        try:
            if self.cache_file.exists():
                with open(self.cache_file, 'r') as f:
                    cache = json.load(f)
                    
                # Verify cache structure
                if isinstance(cache, dict) and 'entities' in cache:
                    print(f"{Fore.GREEN}✓ Loaded {len(cache['entities'])} entities from cache{Style.RESET_ALL}")
                    self.entities = cache['entities']
                else:
                    print(f"{Fore.YELLOW}No valid entity cache found, starting fresh{Style.RESET_ALL}")
                    self.entities = {}
            
        except (json.JSONDecodeError, KeyError) as e:
            print(f"{Fore.RED}✗ Error loading entity cache: {str(e)}{Style.RESET_ALL}")
            self.entities = {}
    
    def _save_entity_cache(self):
        """Save entity cache to file."""
        try:
            # Prepare cache data with list conversion for JSON serialization
            cache_data = {
                'last_update': datetime.now().isoformat(),
                'entities': self.entities
            }
            
            # Ensure directory exists
            self.cache_file.parent.mkdir(parents=True, exist_ok=True)
            
            # Save cache
            with open(self.cache_file, 'w') as f:
                json.dump(cache_data, f, indent=2)
                
            print(f"{Fore.GREEN}✓ Saved {len(self.entities)} entities to cache{Style.RESET_ALL}")
            
        except Exception as e:
            print(f"{Fore.RED}✗ Error saving entity cache: {str(e)}{Style.RESET_ALL}")
    
    def _load_or_update_cache(self) -> Dict:
        """Load entity cache or update if expired."""
        try:
            if self.cache_file.exists():
                try:
                    with self.cache_file.open('r') as f:
                        cache = json.load(f)
                    
                    last_update = datetime.fromisoformat(cache['last_update'])
                    if datetime.now() - last_update < self.cache_duration:
                        print(f"{Fore.GREEN}✓ Loaded entity cache{Style.RESET_ALL}")
                        
                        # Convert lists back to sets for aliases
                        for ticker, data in cache['entities'].items():
                            if 'aliases' in data:
                                data['aliases'] = set(data['aliases'])
                        
                        return cache['entities']
                except (json.JSONDecodeError, KeyError) as e:
                    logger.warning(f"Error reading cache file: {str(e)}")
                    # Don't return here, continue to update cache
            
            print(f"{Fore.YELLOW}Updating entity cache...{Style.RESET_ALL}")
            entities = self._fetch_current_entities()
            
            # Prepare cache data with list conversion for JSON serialization
            cache_data = {
                'last_update': datetime.now().isoformat(),
                'entities': {
                    ticker: {
                        **entity_data,
                        'aliases': list(entity_data['aliases']) if isinstance(entity_data.get('aliases'), set) else []
                    }
                    for ticker, entity_data in entities.items()
                }
            }
            
            # Save to cache with error handling
            try:
                # Ensure parent directory exists
                self.cache_file.parent.mkdir(parents=True, exist_ok=True)
                
                # Write cache file
                with self.cache_file.open('w') as f:
                    json.dump(cache_data, f, indent=2)
                print(f"{Fore.GREEN}✓ Updated entity cache with {len(entities)} stocks{Style.RESET_ALL}")
            except Exception as e:
                logger.error(f"Error saving cache: {str(e)}")
            
            return entities
            
        except Exception as e:
            logger.error(f"Error loading/updating entity cache: {str(e)}")
            return self._fallback_entities.copy()
    
    def _fetch_current_entities(self) -> Dict:
        """Fetch current hot stocks and their related entities."""
        entities = {}
        
        try:
            # Get top movers from Yahoo Finance
            movers = self._get_market_movers()
            
            # Get company info for each ticker
            for ticker in movers:
                try:
                    stock = yf.Ticker(ticker)
                    info = stock.info
                    
                    if not info or 'longName' not in info:
                        continue
                    
                    # Extract and clean company name variations
                    company_name = info.get('longName', '').lower()
                    short_name = info.get('shortName', '').lower()
                    
                    # Create base aliases set
                    aliases = {
                        company_name,
                        short_name,
                        ticker.lower(),
                        company_name.split()[0]  # First word of company name
                    }
                    
                    # Add common variations
                    if 'inc.' in company_name:
                        aliases.add(company_name.replace('inc.', '').strip())
                    if 'corp.' in company_name:
                        aliases.add(company_name.replace('corp.', '').strip())
                    if 'corporation' in company_name:
                        aliases.add(company_name.replace('corporation', '').strip())
                    
                    # Remove empty strings and clean aliases
                    aliases = {alias.strip() for alias in aliases if alias and len(alias.strip()) > 1}
                    
                    entities[ticker] = {
                        'company_name': company_name,
                        'short_name': short_name,
                        'industry': info.get('industry', ''),
                        'sector': info.get('sector', ''),
                        'officers': [
                            officer.get('name', '').lower()
                            for officer in info.get('officers', [])
                            if officer.get('name')
                        ],
                        'products': self._extract_products(info.get('longBusinessSummary', '')),
                        'aliases': aliases,
                        'confidence_class': 'HIGH'  # Default for verified stocks
                    }
                except Exception as e:
                    logger.debug(f"Error fetching info for {ticker}: {str(e)}")
                    continue
            
            return entities
            
        except Exception as e:
            logger.error(f"Error fetching entities: {str(e)}")
            return self._fallback_entities
    
    def _get_market_movers(self) -> List[str]:
        """Get current market movers from Yahoo Finance."""
        movers = set()
        
        try:
            # Most active stocks
            url = "https://finance.yahoo.com/most-active"
            response = requests.get(url, headers={'User-Agent': 'Mozilla/5.0'})
            soup = BeautifulSoup(response.text, 'html.parser')
            
            # Extract tickers from the table
            for row in soup.select('tr'):
                ticker_cell = row.select_one('td:first-child')
                if ticker_cell and ticker_cell.text.strip().isalpha():
                    ticker = ticker_cell.text.strip().upper()
                    if len(ticker) <= 5:  # Only consider standard length tickers
                        movers.add(ticker)
            
            if not movers:  # If no tickers found, use fallback
                return list(self._fallback_entities.keys())
            
            return list(movers)[:50]  # Limit to top 50
            
        except Exception as e:
            logger.error(f"Error fetching market movers: {str(e)}")
            return list(self._fallback_entities.keys())
    
    def _extract_products(self, text: str) -> List[str]:
        """Extract potential product names from business summary."""
        if not text:
            return []
        
        # Split on common separators and filter
        words = text.lower().replace(',', ' ').replace(';', ' ').split()
        products = []
        
        # Look for product-like terms
        product_indicators = ['product', 'platform', 'service', 'technology', 'solution']
        
        current_product = []
        for word in words:
            if word in product_indicators:
                if current_product:
                    products.append(' '.join(current_product))
                current_product = []
            elif len(word) > 3 and word.isalpha():
                current_product.append(word)
            else:
                if current_product:
                    products.append(' '.join(current_product))
                current_product = []
        
        if current_product:
            products.append(' '.join(current_product))
        
        return list(set(products))
    
    def _get_fallback_entities(self) -> Dict:
        """Provide fallback entity data for key tech stocks."""
        return {
            'NVDA': {
                'company_name': 'nvidia corporation',
                'short_name': 'nvidia',
                'aliases': {'nvidia', 'jensen huang', 'cuda', 'geforce', 'rtx', 'gpu'},
                'products': ['cuda', 'geforce', 'rtx', 'gpu', 'tensor', 'drive'],
                'confidence_class': 'HIGH'
            },
            'AAPL': {
                'company_name': 'apple inc.',
                'short_name': 'apple',
                'aliases': {'apple', 'tim cook', 'iphone', 'macbook', 'ios'},
                'products': ['iphone', 'macbook', 'ipad', 'airpods', 'mac'],
                'confidence_class': 'HIGH'
            },
            'MSFT': {
                'company_name': 'microsoft corporation',
                'short_name': 'microsoft',
                'aliases': {'microsoft', 'satya nadella', 'windows', 'azure', 'xbox'},
                'products': ['windows', 'azure', 'office', 'xbox', 'teams'],
                'confidence_class': 'HIGH'
            },
            'GOOGL': {
                'company_name': 'alphabet inc.',
                'short_name': 'google',
                'aliases': {'google', 'alphabet', 'sundar pichai', 'android', 'chrome'},
                'products': ['search', 'android', 'chrome', 'gmail', 'maps'],
                'confidence_class': 'HIGH'
            },
            'META': {
                'company_name': 'meta platforms inc.',
                'short_name': 'meta',
                'aliases': {'meta', 'facebook', 'mark zuckerberg', 'instagram', 'whatsapp'},
                'products': ['facebook', 'instagram', 'whatsapp', 'oculus', 'messenger'],
                'confidence_class': 'HIGH'
            }
        }
    
    def validate_context(self, text: str, ticker: str) -> Tuple[bool, float]:
        """Validate if a ticker appears in a meaningful context with known entities.
        
        Args:
            text (str): The text to validate
            ticker (str): The ticker to validate
            
        Returns:
            Tuple[bool, float]: (is_valid, match_strength)
        """
        # Return false if text or ticker is empty
        if not text or not ticker:
            return False, 0.0
            
        # Convert text to lowercase and ticker to uppercase for comparison
        text = text.lower()
        ticker = ticker.upper()
        
        # Always accept valid ETFs
        if ticker in self.valid_etfs:
            self.match_history.append({
                'ticker': ticker,
                'text': text,
                'match_type': 'ETF',
                'match_strength': 1.0
            })
            return True, 1.0
            
        # Reject common word tickers unless they have strong context
        if ticker in self.common_word_tickers:
            # Get entity info from cache or fallback
            entity_info = self.entities.get(ticker) or self._fallback_entities.get(ticker, {})
            
            # Only accept if we have strong entity matches
            if not entity_info:
                return False, 0.0
                
            # Check for strong company name match
            company_name = entity_info.get('company_name', '').lower()
            if company_name and company_name in text:
                self.match_history.append({
                    'ticker': ticker,
                    'text': text,
                    'match_type': 'STRONG_COMPANY_NAME',
                    'match_strength': 1.0
                })
                return True, 1.0
                
            # Check for strong officer match
            officers = entity_info.get('officers', [])
            if any(officer.lower() in text for officer in officers):
                self.match_history.append({
                    'ticker': ticker,
                    'text': text, 
                    'match_type': 'STRONG_OFFICER',
                    'match_strength': 0.9
                })
                return True, 0.9
                
            # Reject common word ticker without strong context
            return False, 0.0
            
        # Get entity info for non-common word tickers
        entity_info = self.entities.get(ticker) or self._fallback_entities.get(ticker, {})
        if not entity_info:
            return False, 0.0
            
        match_strength = 0.0
        
        # Check company name match
        company_name = entity_info.get('company_name', '').lower()
        if company_name and company_name in text:
            match_strength = max(match_strength, 0.8)
            self.match_history.append({
                'ticker': ticker,
                'text': text,
                'match_type': 'COMPANY_NAME',
                'match_strength': 0.8
            })
            
        # Check officer matches
        officers = entity_info.get('officers', [])
        if any(officer.lower() in text for officer in officers):
            match_strength = max(match_strength, 0.7)
            self.match_history.append({
                'ticker': ticker,
                'text': text,
                'match_type': 'OFFICER',
                'match_strength': 0.7
            })
            
        # Check product matches
        products = entity_info.get('products', [])
        if any(product.lower() in text for product in products):
            match_strength = max(match_strength, 0.6)
            self.match_history.append({
                'ticker': ticker,
                'text': text,
                'match_type': 'PRODUCT',
                'match_strength': 0.6
            })
            
        # Check industry/sector matches
        industry = entity_info.get('industry', '').lower()
        sector = entity_info.get('sector', '').lower()
        if (industry and industry in text) or (sector and sector in text):
            match_strength = max(match_strength, 0.5)
            self.match_history.append({
                'ticker': ticker,
                'text': text,
                'match_type': 'INDUSTRY_SECTOR',
                'match_strength': 0.5
            })
            
        # Cap match strength at 1.0
        match_strength = min(match_strength, 1.0)
        
        return match_strength > 0.0, match_strength
    
    def save_debug_info(self):
        """Save entity matching debug information."""
        if not self.match_history:
            return
        
        # Convert sets to lists for JSON serialization
        serializable_history = []
        for match in self.match_history:
            serializable_match = match.copy()
            if isinstance(match.get('matches'), set):
                serializable_match['matches'] = list(match['matches'])
            serializable_history.append(serializable_match)
        
        debug_file = DEBUG_DIR / f"entity_matches_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(debug_file, 'w') as f:
            json.dump(serializable_history, f, indent=2)
        
        print(f"{Fore.GREEN}✓ Saved entity matching debug info to {debug_file}{Style.RESET_ALL}")
    
    def get_confidence_class(self, ticker: str) -> str:
        """Get the confidence class for a ticker."""
        if ticker in self.entities:
            return self.entities[ticker].get('confidence_class', 'LOW')
        elif ticker in self._fallback_entities:
            return self._fallback_entities[ticker].get('confidence_class', 'LOW')
        return 'LOW' 
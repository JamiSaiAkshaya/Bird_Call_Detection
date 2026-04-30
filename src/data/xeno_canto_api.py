"""
Professional Xeno-Canto API Client for Bird Sound Data Acquisition
High-performance implementation with caching, error handling, and rate limiting
"""

import os
import time
import json
import hashlib
import requests
from pathlib import Path
from typing import List, Dict, Optional, Tuple, Union
import pandas as pd
from tqdm.auto import tqdm
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading
from urllib.parse import urljoin

class XenoCantoAPI:
    """
    Professional Xeno-Canto API client with intelligent caching and error handling
    """
    
    def __init__(self, config: Dict):
        """Initialize API client with configuration"""
        self.base_url = config['xeno_canto']['base_url']
        self.timeout = config['xeno_canto']['timeout']
        self.max_retries = config['xeno_canto']['max_retries']
        self.cache_enabled = config['xeno_canto']['cache_enabled']
        self.cache_dir = Path(config['xeno_canto']['cache_dir'])
        self.rate_limit = config['xeno_canto']['rate_limit']
        self.max_concurrent = config['xeno_canto']['max_concurrent_downloads']
        
        # Create cache directory
        if self.cache_enabled:
            self.cache_dir.mkdir(parents=True, exist_ok=True)
            
        # Rate limiting
        self.last_request_time = 0
        self.request_lock = threading.Lock()
        
        # Session for connection pooling
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'BirdCallDetection/1.0 (Conservation Research)',
            'Accept': 'application/json',
        })
        
        self.logger = logging.getLogger(__name__)
        self.logger.info(f"Initialized Xeno-Canto API client with cache: {self.cache_enabled}")
    
    def __enter__(self):
        return self
        
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.session.close()
    
    def _rate_limit(self):
        """Implement rate limiting to be respectful to API"""
        with self.request_lock:
            current_time = time.time()
            time_since_last = current_time - self.last_request_time
            
            if time_since_last < self.rate_limit:
                sleep_time = self.rate_limit - time_since_last
                time.sleep(sleep_time)
            
            self.last_request_time = time.time()
    
    def _get_cache_key(self, query: str, page: int = 1) -> str:
        """Generate cache key for query"""
        cache_string = f"{query}_{page}"
        return hashlib.md5(cache_string.encode()).hexdigest()
    
    def _load_from_cache(self, cache_key: str) -> Optional[Dict]:
        """Load data from cache if available"""
        if not self.cache_enabled:
            return None
            
        cache_file = self.cache_dir / f"{cache_key}.json"
        if cache_file.exists():
            try:
                with open(cache_file, 'r', encoding='utf-8') as f:
                    return json.load(f)
            except Exception as e:
                self.logger.warning(f"Failed to load cache {cache_key}: {e}")
        return None
    
    def _save_to_cache(self, cache_key: str, data: Dict):
        """Save data to cache"""
        if not self.cache_enabled:
            return
            
        cache_file = self.cache_dir / f"{cache_key}.json"
        try:
            with open(cache_file, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
        except Exception as e:
            self.logger.warning(f"Failed to save cache {cache_key}: {e}")
    
    def search_recordings(self, query: str, max_results: int = 100) -> List[Dict]:
        """
        Search for bird recordings using Xeno-Canto API
        
        Args:
            query: Search query (e.g., 'sp:"Grus americana"' for Whooping Crane)
            max_results: Maximum number of recordings to retrieve
            
        Returns:
            List of recording metadata dictionaries
        """
        all_recordings = []
        page = 1
        
        self.logger.info(f"Searching for recordings: {query} (max: {max_results})")
        
        with tqdm(desc=f"Fetching {query}", unit="recordings") as pbar:
            while len(all_recordings) < max_results:
                cache_key = self._get_cache_key(query, page)
                
                # Try to load from cache first
                cached_data = self._load_from_cache(cache_key)
                if cached_data:
                    response_data = cached_data
                    self.logger.debug(f"Loaded page {page} from cache")
                else:
                    # Make API request
                    response_data = self._make_api_request(query, page)
                    if response_data:
                        self._save_to_cache(cache_key, response_data)
                
                if not response_data or not response_data.get('recordings'):
                    break
                
                recordings = response_data['recordings']
                remaining_slots = max_results - len(all_recordings)
                recordings_to_add = recordings[:remaining_slots]
                
                all_recordings.extend(recordings_to_add)
                pbar.update(len(recordings_to_add))
                
                # Check if we have more pages
                total_pages = int(response_data.get('numPages', 1))
                if page >= total_pages:
                    break
                    
                page += 1
        
        self.logger.info(f"Retrieved {len(all_recordings)} recordings for {query}")
        return all_recordings
    
    def _make_api_request(self, query: str, page: int = 1) -> Optional[Dict]:
        """Make actual API request with retries and exponential backoff"""
        params = {
            'query': query,
            'page': page
        }
        
        for attempt in range(self.max_retries):
            try:
                self._rate_limit()
                
                response = self.session.get(
                    self.base_url,
                    params=params,
                    timeout=self.timeout
                )
                response.raise_for_status()
                
                # Validate JSON response
                data = response.json()
                if 'recordings' not in data:
                    self.logger.warning(f"Invalid API response structure for query: {query}")
                    return None
                    
                return data
                
            except requests.exceptions.RequestException as e:
                self.logger.warning(f"API request failed (attempt {attempt + 1}/{self.max_retries}): {e}")
                if attempt < self.max_retries - 1:
                    sleep_time = 2 ** attempt  # Exponential backoff
                    time.sleep(sleep_time)
                else:
                    self.logger.error(f"Failed to retrieve data after {self.max_retries} attempts")
            
            except json.JSONDecodeError as e:
                self.logger.error(f"Failed to decode JSON response: {e}")
                break
        
        return None
    
    def download_audio_file(self, recording: Dict, download_dir: Path) -> Optional[Path]:
        """
        Download audio file for a recording with robust error handling
        
        Args:
            recording: Recording metadata dictionary
            download_dir: Directory to save audio files
            
        Returns:
            Path to downloaded file or None if failed
        """
        if 'file' not in recording or not recording['file']:
            self.logger.warning(f"No download URL for recording {recording.get('id')}")
            return None
            
        download_url = recording['file']
        file_id = recording['id']
        
        # Determine file extension from URL
        file_extension = download_url.split('.')[-1].lower()
        if file_extension not in ['mp3', 'wav', 'flac', 'ogg']:
            file_extension = 'mp3'  # Default to mp3
            
        filename = f"{file_id}.{file_extension}"
        file_path = download_dir / filename
        
        # Check if file already exists and is valid
        if file_path.exists() and file_path.stat().st_size > 0:
            self.logger.debug(f"File already exists: {filename}")
            return file_path
        
        # Create download directory
        download_dir.mkdir(parents=True, exist_ok=True)
        
        try:
            self._rate_limit()
            
            # Stream download for large files
            with self.session.get(download_url, timeout=self.timeout, stream=True) as response:
                response.raise_for_status()
                
                # Check content type
                content_type = response.headers.get('content-type', '')
                if 'audio' not in content_type and 'octet-stream' not in content_type:
                    self.logger.warning(f"Unexpected content type: {content_type} for {filename}")
                
                # Download with progress
                total_size = int(response.headers.get('content-length', 0))
                
                with open(file_path, 'wb') as f:
                    for chunk in response.iter_content(chunk_size=8192):
                        if chunk:
                            f.write(chunk)
            
            # Verify file was downloaded successfully
            if file_path.exists() and file_path.stat().st_size > 0:
                self.logger.debug(f"Downloaded: {filename} ({file_path.stat().st_size} bytes)")
                return file_path
            else:
                self.logger.error(f"Download failed - empty file: {filename}")
                file_path.unlink(missing_ok=True)
                return None
                
        except Exception as e:
            self.logger.error(f"Failed to download {filename}: {e}")
            # Clean up partial file
            file_path.unlink(missing_ok=True)
            return None
    
    def get_species_recordings(self, species_config: List[Dict], 
                             recordings_per_species: int = 50) -> pd.DataFrame:
        """
        Get recordings for multiple endangered species
        
        Args:
            species_config: List of species configuration dictionaries
            recordings_per_species: Number of recordings per species
            
        Returns:
            DataFrame with all recording metadata
        """
        all_recordings = []
        
        for species in species_config:
            scientific_name = species['scientific_name']
            common_name = species['common_name']
            priority = species['priority']
            class_id = species.get('class_id', 0)
            
            self.logger.info(f"Processing {common_name} ({scientific_name})")
            
            # Search for recordings
            query = f'sp:"{scientific_name}"'
            recordings = self.search_recordings(query, max_results=recordings_per_species)
            
            # Add metadata
            for recording in recordings:
                recording['target_species'] = scientific_name
                recording['common_name'] = common_name
                recording['priority'] = priority
                recording['class_id'] = class_id
                recording['label'] = 1  # Positive label for target species
                
            all_recordings.extend(recordings)
            
            # Add delay between species to be respectful to API
            time.sleep(1)
        
        df = pd.DataFrame(all_recordings)
        self.logger.info(f"Created dataset with {len(df)} recordings from {len(species_config)} species")
        
        return df

    def create_balanced_dataset(self, species_config: List[Dict], 
                              recordings_per_species: int = 100,
                              negative_samples_ratio: float = 1.0,
                              background_species: Optional[List[Dict]] = None) -> pd.DataFrame:
        """
        Create balanced dataset with positive and negative samples
        
        Args:
            species_config: Target species configuration
            recordings_per_species: Recordings per target species
            negative_samples_ratio: Ratio of negative to positive samples
            background_species: List of background species for negative samples
            
        Returns:
            Balanced DataFrame with labels
        """
        # Get positive samples (target species)
        positive_df = self.get_species_recordings(species_config, recordings_per_species)
        
        # Default background species if not provided
        if background_species is None:
            background_species = [
                {'scientific_name': 'Turdus migratorius', 'common_name': 'American Robin'},
                {'scientific_name': 'Corvus brachyrhynchos', 'common_name': 'American Crow'},
                {'scientific_name': 'Cardinalis cardinalis', 'common_name': 'Northern Cardinal'},
                {'scientific_name': 'Poecile carolinensis', 'common_name': 'Carolina Chickadee'},
                {'scientific_name': 'Sialia sialis', 'common_name': 'Eastern Bluebird'}
            ]
        
        # Get negative samples (background species)
        negative_recordings = []
        total_negative_needed = int(len(positive_df) * negative_samples_ratio)
        negative_per_species = max(1, total_negative_needed // len(background_species))
        
        for species in background_species:
            scientific_name = species['scientific_name']
            common_name = species['common_name']
            
            self.logger.info(f"Processing background species: {common_name}")
            
            query = f'sp:"{scientific_name}"'
            recordings = self.search_recordings(query, max_results=negative_per_species)
            
            for recording in recordings:
                recording['target_species'] = 'background'
                recording['common_name'] = common_name
                recording['priority'] = 'background'
                recording['class_id'] = -1  # Background class
                recording['label'] = 0  # Negative label
                
            negative_recordings.extend(recordings)
            
            # Limit total negative samples
            if len(negative_recordings) >= total_negative_needed:
                negative_recordings = negative_recordings[:total_negative_needed]
                break
                
            time.sleep(1)  # Rate limiting
        
        negative_df = pd.DataFrame(negative_recordings)
        
        # Combine datasets
        final_df = pd.concat([positive_df, negative_df], ignore_index=True)
        
        # Shuffle the dataset
        final_df = final_df.sample(frac=1, random_state=42).reset_index(drop=True)
        
        self.logger.info(f"Created balanced dataset: {len(positive_df)} positive, "
                        f"{len(negative_df)} negative samples")
        
        return final_df
    
    def batch_download_audio(self, recordings_df: pd.DataFrame, 
                           download_dir: Path, 
                           max_workers: int = None) -> pd.DataFrame:
        """
        Download multiple audio files concurrently
        
        Args:
            recordings_df: DataFrame with recording metadata
            download_dir: Directory to save audio files
            max_workers: Maximum number of concurrent downloads
            
        Returns:
            DataFrame with added 'local_path' column
        """
        if max_workers is None:
            max_workers = min(self.max_concurrent, len(recordings_df))
        
        download_dir.mkdir(parents=True, exist_ok=True)
        results_df = recordings_df.copy()
        results_df['local_path'] = None
        results_df['download_success'] = False
        
        def download_single(idx_row_tuple):
            idx, row = idx_row_tuple
            file_path = self.download_audio_file(row.to_dict(), download_dir)
            return idx, file_path
        
        self.logger.info(f"Starting concurrent download of {len(recordings_df)} files with {max_workers} workers")
        
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            # Submit all download tasks
            future_to_idx = {
                executor.submit(download_single, (idx, row)): idx
                for idx, row in recordings_df.iterrows()
            }
            
            # Progress bar for downloads
            with tqdm(total=len(future_to_idx), desc="Downloading audio files") as pbar:
                for future in as_completed(future_to_idx):
                    idx = future_to_idx[future]
                    try:
                        idx, file_path = future.result(timeout=60)
                        if file_path:
                            results_df.loc[idx, 'local_path'] = str(file_path)
                            results_df.loc[idx, 'download_success'] = True
                    except Exception as e:
                        self.logger.warning(f"Download failed for index {idx}: {e}")
                    finally:
                        pbar.update(1)
        
        successful_downloads = results_df['download_success'].sum()
        self.logger.info(f"Successfully downloaded {successful_downloads}/{len(recordings_df)} files")
        
        return results_df
    
    def get_recording_stats(self, recordings_df: pd.DataFrame) -> Dict:
        """Get statistics about the recordings dataset"""
        stats = {
            'total_recordings': len(recordings_df),
            'species_count': recordings_df['target_species'].nunique(),
            'positive_samples': len(recordings_df[recordings_df['label'] == 1]),
            'negative_samples': len(recordings_df[recordings_df['label'] == 0]),
            'countries': recordings_df['cnt'].nunique() if 'cnt' in recordings_df.columns else 0,
            'quality_distribution': recordings_df['q'].value_counts().to_dict() if 'q' in recordings_df.columns else {},
            'avg_length': recordings_df['length'].mean() if 'length' in recordings_df.columns else 0,
            'date_range': {
                'earliest': recordings_df['date'].min() if 'date' in recordings_df.columns else None,
                'latest': recordings_df['date'].max() if 'date' in recordings_df.columns else None
            }
        }
        
        return stats
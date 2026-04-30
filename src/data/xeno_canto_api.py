"""
Xeno-Canto API Client — ML Extension v2 (fixed v3)
Correct endpoint: https://www.xeno-canto.org/api/2/recordings
Query: scientific name only — NO q:A suffix — quality filtered locally
"""

import time, json, hashlib, requests, threading
from pathlib import Path
from typing import List, Dict, Optional
import pandas as pd
from tqdm.auto import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed
import logging

logger = logging.getLogger(__name__)

# Hard-coded correct URL — do not change
_XC_URL = "https://www.xeno-canto.org/api/2/recordings"
_QUALITY_RANK = {"A": 5, "B": 4, "C": 3, "D": 2, "E": 1, "no score": 0, "": 0}


class XenoCantoAPI:
    def __init__(self, config: Dict):
        xc = config["xeno_canto"]
        self.base_url    = _XC_URL          # always override whatever is in config
        self.timeout     = xc.get("timeout", 30)
        self.max_retries = xc.get("max_retries", 3)
        self.cache_on    = xc.get("cache_enabled", True)
        self.cache_dir   = Path(xc.get("cache_dir", "./data/cache"))
        self.rate_limit  = xc.get("rate_limit", 1.2)
        self.max_dl      = xc.get("max_concurrent_downloads", 3)
        self.min_quality = xc.get("min_quality", "D")
        self._lock       = threading.Lock()
        self._last_req   = 0.0

        if self.cache_on:
            self.cache_dir.mkdir(parents=True, exist_ok=True)

        self.session = requests.Session()
        self.session.headers.update({
            "User-Agent": (
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/124.0.0.0 Safari/537.36"
            ),
            "Accept": "application/json, */*",
            "Accept-Language": "en-US,en;q=0.9",
            "Referer": "https://www.xeno-canto.org/",
        })
        logger.info(f"XenoCantoAPI ready  url={self.base_url}")

    # ------------------------------------------------------------------ helpers
    def _throttle(self):
        with self._lock:
            wait = self.rate_limit - (time.time() - self._last_req)
            if wait > 0:
                time.sleep(wait)
            self._last_req = time.time()

    def _cache_key(self, q: str, page: int) -> str:
        return hashlib.md5(f"{q}_{page}".encode()).hexdigest()

    def _load_cache(self, key: str) -> Optional[Dict]:
        if not self.cache_on:
            return None
        f = self.cache_dir / f"{key}.json"
        if f.exists():
            try:
                return json.loads(f.read_text(encoding="utf-8"))
            except Exception:
                return None
        return None

    def _save_cache(self, key: str, data: Dict):
        if not self.cache_on:
            return
        try:
            (self.cache_dir / f"{key}.json").write_text(
                json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
        except Exception:
            pass

    def _quality_ok(self, rec: Dict) -> bool:
        rank = _QUALITY_RANK.get(str(rec.get("q", "")).strip(), 0)
        return rank >= _QUALITY_RANK.get(self.min_quality, 2)

    # ------------------------------------------------------------------ search
    def search_recordings(self, query: str, max_results: int = 500) -> List[Dict]:
        """
        Search recordings. Pass scientific name ONLY — e.g. 'Turdus migratorius'.
        Do NOT include q:A or any quality suffix — quality is filtered locally.
        """
        results, page = [], 1
        with tqdm(desc=f"Fetching: {query[:45]}", unit="rec") as pbar:
            while len(results) < max_results:
                key  = self._cache_key(query, page)
                data = self._load_cache(key) or self._api_get(query, page)
                if not data:
                    break
                self._save_cache(key, data)

                recs = data.get("recordings", [])
                if not recs:
                    break

                good = [r for r in recs if self._quality_ok(r)]
                take = good[: max_results - len(results)]
                results.extend(take)
                pbar.update(len(take))

                if page >= int(data.get("numPages", 1)):
                    break
                page += 1

        logger.info(f'Got {len(results)} recordings for "{query}"')
        return results

    def _api_get(self, query: str, page: int) -> Optional[Dict]:
        """
        Single API request. Query must be plain text — scientific name only.
        """
        params = {"query": query, "page": page}

        for attempt in range(self.max_retries):
            try:
                self._throttle()
                r = self.session.get(
                    self.base_url, params=params, timeout=self.timeout)

                if r.status_code == 404:
                    logger.error(
                        f"404 Not Found — URL was: {r.url}\n"
                        f"This usually means the API endpoint changed.\n"
                        f"Check https://xeno-canto.org/help/search for the latest API docs."
                    )
                    return None   # don't retry 404

                r.raise_for_status()
                data = r.json()
                if "recordings" in data:
                    return data
                logger.warning(f"Unexpected response keys: {list(data.keys())}")
                return None

            except requests.exceptions.HTTPError as e:
                logger.warning(f"HTTP error attempt {attempt+1}: {e}")
                if attempt < self.max_retries - 1:
                    time.sleep(2 ** attempt)
            except Exception as e:
                logger.warning(f"Request failed attempt {attempt+1}: {e}")
                if attempt < self.max_retries - 1:
                    time.sleep(2 ** attempt)

        return None

    # ------------------------------------------------------------------ download
    def download_audio_file(self, recording: Dict,
                            download_dir: Path) -> Optional[Path]:
        url = recording.get("file", "")
        if not url:
            return None
        if url.startswith("//"):
            url = "https:" + url

        fid = recording.get("id", "unknown")
        ext = url.rsplit(".", 1)[-1].lower()
        if ext not in ("mp3", "wav", "flac", "ogg"):
            ext = "mp3"

        dest = download_dir / f"{fid}.{ext}"
        if dest.exists() and dest.stat().st_size > 1024:
            return dest

        download_dir.mkdir(parents=True, exist_ok=True)
        try:
            self._throttle()
            with self.session.get(url, timeout=self.timeout, stream=True) as resp:
                resp.raise_for_status()
                with open(dest, "wb") as fh:
                    for chunk in resp.iter_content(8192):
                        if chunk:
                            fh.write(chunk)
            if dest.stat().st_size > 1024:
                return dest
            dest.unlink(missing_ok=True)
        except Exception as e:
            logger.debug(f"Download failed {fid}: {e}")
            dest.unlink(missing_ok=True)
        return None

    def batch_download_audio(self, df: pd.DataFrame,
                             download_dir: Path,
                             max_workers: int = None) -> pd.DataFrame:
        max_workers = max_workers or self.max_dl
        download_dir.mkdir(parents=True, exist_ok=True)
        result = df.copy()
        result["local_path"]       = None
        result["download_success"] = False

        def _dl(args):
            idx, row = args
            p = self.download_audio_file(row.to_dict(), download_dir)
            return idx, p

        with ThreadPoolExecutor(max_workers=max_workers) as ex:
            futures = {ex.submit(_dl, (i, r)): i for i, r in df.iterrows()}
            with tqdm(total=len(futures), desc="Downloading audio") as pbar:
                for fut in as_completed(futures):
                    try:
                        idx, path = fut.result(timeout=90)
                        if path:
                            result.at[idx, "local_path"]       = str(path)
                            result.at[idx, "download_success"] = True
                    except Exception as e:
                        logger.debug(f"Download task error: {e}")
                    pbar.update(1)

        ok = result["download_success"].sum()
        logger.info(f"Downloaded {ok}/{len(df)} files")
        return result

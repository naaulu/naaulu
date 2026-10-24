import atexit
import hashlib
import logging
import os
import threading
import time
import warnings
import zipfile

import bs4
import requests

warnings.filterwarnings("ignore", message="I/O operation on closed file")

import naaulu.network
import naaulu.config

import tqdm

import boto3
import botocore.config
import botocore.exceptions


logger = logging.getLogger(__name__)

session = None

MAX_CONCURRENT_DOWNLOADS = 8
_download_semaphore = threading.Semaphore(MAX_CONCURRENT_DOWNLOADS)

# Connections kept alive per host by runtime_session()
MAX_POOL_SIZE = 16

# Transient-error retries for API handshakes such as get_temp
TEMP_RETRIES = 3
TEMP_BACKOFF = (20.0, 60.0)


def _backoff_delay(response, attempt, backoff):
    """Delay before retrying, honouring Retry-After if sent."""
    if response is not None:
        value = response.headers.get("retry-after")
        if value:
            try:
                return min(float(value), 60.0)
            except ValueError:
                pass
    return backoff[min(attempt - 1, len(backoff) - 1)]


def _close_session():
    global session
    if session is not None:
        try:
            session.close()
        except Exception:
            pass
        session = None


atexit.register(_close_session)


def write_content(response, filename):

    with open(filename, "wb") as f:
        for chunk in response.iter_content(chunk_size=8192):
            f.write(chunk)


def _user_agent():
    """Identify ourselves to data providers."""
    try:
        from importlib.metadata import version
        package_version = version("naaulu")
    except Exception:
        package_version = "dev"
    return (
        f"naaulu/{package_version} "
        "(https://github.com/naaulu/naaulu; edouard@egouden.dev)"
    )


def runtime_session():

    if naaulu.network.session is None:
        logging.info("creating network session")
        session = requests.sessions.Session()
        session.headers["User-Agent"] = _user_agent()
        # pooled connections: radar fetches plus the download semaphore
        adapter = requests.adapters.HTTPAdapter(
            pool_connections=20, pool_maxsize=MAX_POOL_SIZE
        )
        session.mount("https://", adapter)
        session.mount("http://", adapter)
        naaulu.network.session = session

    logging.info("reusing network session")

    return naaulu.network.session


def fetch(url: str, filename: str = None, stream: bool = False) -> bytes:
    """Fetch a URL, optionally caching the content under filename.
    Returns the content bytes, or a streaming response if stream=True.
    """
    session = runtime_session()

    if filename:
        destination = naaulu.config.get_download_dir()
        filepath = os.path.join(destination, filename)
        if os.path.exists(filepath):
            logger.info(f"Using cached file: {filepath}")
            with open(filepath, "rb") as f:
                return f.read()

    with _download_semaphore:
        response = session.get(url, stream=stream)
        response.raise_for_status()

        if stream:
            return response

        content = response.content

    if filename:
        with open(filepath, "wb") as f:
            f.write(content)
        logger.info(f"Fetched and cached: {filepath}")

    return content


def download(url, filename=None):

    if filename is None:
        filename = os.path.basename(url)

    destination = naaulu.config.get_download_dir()
    filepath = os.path.join(destination, filename)

    if os.path.exists(filepath):
        return filepath

    with _download_semaphore:
        try:
            response = requests.get(url, stream=True, timeout=30)
        except requests.exceptions.Timeout:
            raise TimeoutError(f"Download timeout for {url}")
        if response.status_code == 403:
            raise PermissionError(f"Access denied (403) for {url}")
        elif response.status_code == 404:
            raise FileNotFoundError(f"Radar file not found (404): {url}")
        elif not response.ok:
            raise ConnectionError(f"Download failed: {response.status_code} {response.reason} for {url}")

        total_size = int(response.headers.get("content-length", 0))

        disable = not logging.getLogger().isEnabledFor(logging.DEBUG)
        with open(filepath, "wb") as f, tqdm.tqdm(
            total=total_size,
            unit="B",
            unit_scale=True,
            desc=filepath,
            ascii=True,
            colour="green",
            bar_format="{l_bar}{bar} | {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]",
            disable=disable,
        ) as bar:
            for chunk in response.iter_content(chunk_size=1024):
                f.write(chunk)
                bar.update(len(chunk))

    logger.info(f"Download completed successfully: {filepath}")
    return filepath


# ─── Ranged remote files ─────────────────────────────────────────────────────

# ranged access runs behind a small semaphore regardless of caller count
RANGE_SEMAPHORE = threading.Semaphore(4)
RANGE_RETRIES = 3
RANGE_BACKOFF = (1.0, 2.0)

_range_sizes = {}
_range_lock = threading.Lock()


class RangeReader:
    """Minimal seekable file-like backed by HTTP Range requests."""

    def __init__(self, url, chunk=1 << 20, session=None, retries=None, semaphore=None):
        self.url = url
        self.session = session or runtime_session()
        self.chunk = chunk
        self.retries = max(1, RANGE_RETRIES if retries is None else retries)
        self.semaphore = RANGE_SEMAPHORE if semaphore is None else semaphore
        self.size = self._content_length()
        self.pos = 0
        self._buffer = b""
        self._buffer_start = 0

    @staticmethod
    def _delay(response, attempt):
        """Backoff delay, honouring Retry-After if sent."""
        return _backoff_delay(response, attempt, RANGE_BACKOFF)

    def _request(self, method, headers=None, expect=None):
        """Perform one request, retrying transient failures.
        expect: (start, end) byte range that the response must match.
        """
        last = None
        response = None
        for attempt in range(self.retries):
            if attempt:
                delay = self._delay(response, attempt)
                logger.debug(
                    f"retry {attempt}/{self.retries - 1} {method} {self.url} "
                    f"in {delay:.1f}s after {last}"
                )
                time.sleep(delay)
            response = None
            try:
                with self.semaphore:
                    if method == "HEAD":
                        response = self.session.head(
                            self.url, headers=headers, timeout=60
                        )
                    else:
                        response = self.session.get(
                            self.url, headers=headers, timeout=300
                        )
            except (
                requests.ConnectionError,
                requests.exceptions.ChunkedEncodingError,
            ) as exc:
                last = exc
                continue

            if response.status_code == 404:
                raise FileNotFoundError(f"archive not found: {self.url}")
            if response.status_code in (429, 500, 502, 503, 504):
                last = requests.HTTPError(
                    f"HTTP {response.status_code} for {self.url}"
                )
                continue
            try:
                response.raise_for_status()
            except requests.HTTPError as exc:
                last = exc
                continue

            if expect is not None:
                start, end = expect
                wanted = f"bytes {start}-{end}/{self.size}"
                content_range = response.headers.get("content-range", "")
                body = response.content
                if (
                    response.status_code != 206
                    or content_range != wanted
                    or len(body) != end - start + 1
                ):
                    last = RuntimeError(
                        f"bad range response from {self.url}: "
                        f"{response.status_code} {content_range!r} "
                        f"{len(body)} bytes, wanted {wanted!r}"
                    )
                    continue
            return response

        raise last

    def _content_length(self):
        with _range_lock:
            cached = _range_sizes.get(self.url)
        if cached is not None:
            return cached

        response = self._request("HEAD")
        size = int(response.headers.get("content-length") or 0)
        if size <= 0:
            # no Content-Length: probe the total size with a 1-byte range
            response = self._request("GET", headers={"Range": "bytes=0-0"})
            content_range = response.headers.get("content-range", "")
            if "/" not in content_range:
                raise RuntimeError(
                    f"server does not support range requests: {self.url}"
                )
            size = int(content_range.rsplit("/", 1)[1])
        if size <= 0:
            raise RuntimeError(f"empty archive at {self.url}")

        with _range_lock:
            _range_sizes[self.url] = size
        return size

    def _fill(self):
        start = self.pos
        end = min(start + self.chunk, self.size) - 1
        response = self._request(
            "GET", headers={"Range": f"bytes={start}-{end}"}, expect=(start, end)
        )
        self._buffer = response.content
        self._buffer_start = start

    def read(self, size=-1):
        if size is None or size < 0:
            size = self.size - self.pos
        size = min(size, self.size - self.pos)
        if size <= 0:
            return b""
        parts = []
        remaining = size
        while remaining > 0:
            if not self._buffer_start <= self.pos < self._buffer_start + len(self._buffer):
                self._fill()
            offset = self.pos - self._buffer_start
            take = min(remaining, len(self._buffer) - offset)
            if take <= 0:
                raise RuntimeError(f"empty range response for {self.url}")
            parts.append(self._buffer[offset:offset + take])
            self.pos += take
            remaining -= take
        return b"".join(parts)

    def seek(self, offset, whence=0):
        if whence == 0:
            position = offset
        elif whence == 1:
            position = self.pos + offset
        elif whence == 2:
            position = self.size + offset
        else:
            raise ValueError(f"invalid whence ({whence})")
        self.pos = max(position, 0)
        return self.pos

    def tell(self):
        return self.pos

    def readable(self):
        return True

    def seekable(self):
        return True

    def close(self):
        pass

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


def open_remote(url):
    """File-like for a remote archive: local file if it exists, else ranged HTTP."""
    if os.path.exists(url):
        return open(url, "rb")
    return RangeReader(url)


def content_length(url):
    """Remote file size in bytes (HEAD, cached per url for this run)."""
    return RangeReader(url).size


def list_s3_objects(bucket: str, prefix: str, suffix: str = None, max_keys: int = 1000) -> list:
    """List object keys in a public S3 bucket under a prefix."""
    s3 = boto3.client("s3", config=botocore.config.Config(signature_version=botocore.UNSIGNED))
    response = s3.list_objects_v2(Bucket=bucket, Prefix=prefix, MaxKeys=max_keys)

    if "Contents" not in response:
        return []

    keys = [obj["Key"] for obj in response["Contents"]]
    if suffix:
        keys = [k for k in keys if k.endswith(suffix)]

    return keys


def download_s3_file(s3_url: str, dest_path: str = None) -> str:
    bucket, key = s3_url.replace("s3://", "").split("/", 1)

    if dest_path is None:
        dest_path = os.path.basename(key)

    destination = naaulu.config.get_download_dir()
    dest_path = os.path.join(destination, dest_path)

    s3_client = boto3.client(
        service_name='s3',
        config=botocore.config.Config(signature_version=botocore.UNSIGNED)
    )

    try:
        with _download_semaphore:
            response = s3_client.get_object(Bucket=bucket, Key=key)
            with open(dest_path, 'wb') as f:
                f.write(response['Body'].read())
        return dest_path
    except botocore.exceptions.ClientError as error:
        raise PermissionError(f"Failed to download {key} from {bucket}: {error}")
    

def api_get(url, headers=None, params=None, timeout=None):
    try:
        response = requests.get(url, headers=headers, params=params, timeout=timeout)
        response.raise_for_status()
        return response
    except requests.exceptions.HTTPError as e:
        logger.error(f"HTTP error {response.status_code} for URL: {response.url}")
        logger.debug(f"Response content: {response.text}")
        raise
    except requests.exceptions.RequestException as e:
        logger.error(f"Request failed: {e}")
        raise


def post_extract(url, json, filename):

    destination = naaulu.config.get_download_dir()
    filepath = f"{destination}/{filename}"
    if os.path.exists(filepath):
        return filepath

    digest = hashlib.md5(str(json).encode("utf-8")).hexdigest()
    zipname = f"{digest}.zip"
    zipname = f"{destination}/{zipname}"
    response = requests.post(url, json=json)
    logger.debug(f"HTTP error {response.status_code} for URL: {response.url}")
    logger.debug(f"Response content: {response.text}")
    naaulu.network.write_content(response, zipname)

    with zipfile.ZipFile(zipname) as myzip:
        myzip.extract(filename, path=destination)

    os.remove(zipname)

    return filepath


def get_temp(path, key, filename, field):
    """Resolve the temporary download URL behind `path` and fetch the file.

    Backs off when rate limited, honouring Retry-After; a file the provider
    does not have is a FileNotFoundError, as in download().
    """
    headers = {"Authorization": key}
    response = None
    for attempt in range(TEMP_RETRIES):
        if attempt:
            delay = _backoff_delay(response, attempt, TEMP_BACKOFF)
            logger.info(f"rate limited, retrying in {delay:.0f}s")
            time.sleep(delay)

        response = runtime_session().get(path, headers=headers)
        if response.status_code != 429:
            break

    if response.status_code == 404:
        raise FileNotFoundError(f"Not found: {path}")
    response.raise_for_status()

    download_url = response.json().get(field)

    if not download_url:
        raise ValueError(f"Field '{field}' not found in response")

    file_path = download(download_url, filename)

    return file_path


def extract_links(url):

    session = runtime_session()
    response = session.get(url)
    soup = bs4.BeautifulSoup(response.text, "lxml")
    links = soup.find_all("a", href=True)
    links = [link["href"] for link in links]

    return links


def extract_tables(url):
    session = naaulu.network.runtime_session()
    response = session.get(url)
    soup = bs4.BeautifulSoup(response.content, "lxml")

    tables = []
    for table in soup.find_all("table"):
        rows = []
        for tr in table.find_all("tr"):
            cells = tr.find_all(["td", "th"])
            row = [cell.get_text(strip=True) for cell in cells]
            if row:  # skip empty rows
                rows.append(row)
        if rows:
            tables.append(rows)

    return tables

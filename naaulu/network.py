import atexit
import logging
import os
import threading
import time
import warnings

import requests

warnings.filterwarnings("ignore", message="I/O operation on closed file")

import naaulu.network
import naaulu.config
import naaulu.errors

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
    """Close the shared session and clear the module-level handle."""
    global session
    if session is not None:
        try:
            session.close()
        except Exception:
            pass
        session = None


atexit.register(_close_session)

# public alias of _close_session
close_session = _close_session


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
    """Shared requests session with pooled connections, created on first use;
    raises OfflineError when networking is disabled."""

    if naaulu.config.no_network():
        raise naaulu.errors.OfflineError(
            "--no-network/NAAULU_NO_NETWORK: refusing to open a connection")

    if naaulu.network.session is None:
        logger.info("creating network session")
        session = requests.sessions.Session()
        session.headers["User-Agent"] = _user_agent()
        # pooled connections: radar fetches plus the download semaphore
        adapter = requests.adapters.HTTPAdapter(
            pool_connections=20, pool_maxsize=MAX_POOL_SIZE
        )
        session.mount("https://", adapter)
        session.mount("http://", adapter)
        naaulu.network.session = session

    logger.info("reusing network session")

    return naaulu.network.session


def fetch(url: str, filename: str = None, *, stream: bool = False) -> bytes:
    """Fetch a URL, optionally caching the content under filename.
    Returns the content bytes, or a streaming response if stream=True.
    """
    if filename:
        destination = naaulu.config.get_download_dir()
        filepath = os.path.join(destination, filename)
        if os.path.exists(filepath):
            logger.info(f"Using cached file: {filepath}")
            with open(filepath, "rb") as f:
                return f.read()

    session = runtime_session()

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
    """Download url into the download dir under filename (its basename when
    omitted), returning the local path; an existing file is reused.

    Raises TimeoutError, PermissionError (403), FileNotFoundError (404) or
    ConnectionError for other failures.
    """

    if filename is None:
        filename = os.path.basename(url)

    destination = naaulu.config.get_download_dir()
    filepath = os.path.join(destination, filename)

    if os.path.exists(filepath):
        return filepath

    with _download_semaphore:
        try:
            response = runtime_session().get(url, stream=True, timeout=30)
        except requests.exceptions.Timeout:
            raise TimeoutError(f"Download timeout for {url}")
        try:
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
        finally:
            # close on every path, including raises
            try:
                response.close()
            except Exception:
                pass

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
        """Open url with chunk size chunk (1 MiB), retry count and semaphore
        defaulting to the module-wide settings."""
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
        """Perform one request, retrying transient failures; expect is the
        (start, end) byte range the response must match."""
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
        """Total size in bytes from HEAD, else a 1-byte range probe; cached
        per url. Raises RuntimeError on unsupported ranges or an empty file."""
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
        """Buffer the next chunk-sized range request at the current position."""
        start = self.pos
        end = min(start + self.chunk, self.size) - 1
        response = self._request(
            "GET", headers={"Range": f"bytes={start}-{end}"}, expect=(start, end)
        )
        self._buffer = response.content
        self._buffer_start = start

    def read(self, size=-1):
        """Up to size bytes from the current position, advancing it; all
        remaining bytes when size is negative. b"" at the end."""
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
        """Set the position from offset (whence 0 absolute, 1 current, 2 end),
        clamped at 0; returns the new position. ValueError on a bad whence."""
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
        """Current position in bytes."""
        return self.pos

    def seekable(self):
        """Always True."""
        return True

    def close(self):
        """No-op: the reader holds no resources to release."""
        pass

    def __enter__(self):
        """Return self for use as a context manager."""
        return self

    def __exit__(self, *exc):
        """Return False: exceptions are not suppressed."""
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
    if naaulu.config.no_network():
        raise naaulu.errors.OfflineError(
            f"--no-network: not listing s3://{bucket}/{prefix}")
    s3 = boto3.client("s3", config=botocore.config.Config(signature_version=botocore.UNSIGNED))
    response = s3.list_objects_v2(Bucket=bucket, Prefix=prefix, MaxKeys=max_keys)

    if "Contents" not in response:
        return []

    keys = [obj["Key"] for obj in response["Contents"]]
    if suffix:
        keys = [k for k in keys if k.endswith(suffix)]

    return keys


def download_s3_file(s3_url: str, dest_path: str = None) -> str:
    """Download bucket/key from a public S3 url into the download dir and
    return the local path; dest_path defaults to the key's basename.

    Raises OfflineError when not cached and networking is disabled,
    PermissionError on an S3 client error.
    """
    bucket, key = s3_url.replace("s3://", "").split("/", 1)

    if dest_path is None:
        dest_path = os.path.basename(key)

    destination = naaulu.config.get_download_dir()
    dest_path = os.path.join(destination, dest_path)

    if naaulu.config.no_network():
        if os.path.exists(dest_path):
            return dest_path
        raise naaulu.errors.OfflineError(
            f"--no-network: not cached: {s3_url}")

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

import atexit
import hashlib
import logging
import os
import threading
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


def runtime_session():

    if naaulu.network.session is None:
        logging.info("creating network session")
        naaulu.network.session = requests.sessions.Session()

    logging.info("reusing network session")

    return naaulu.network.session


def fetch(url: str, filename: str = None, stream: bool = False) -> bytes:
    """
    Fetch content from a remote URL with optional local caching.

    Parameters:
        url (str): Remote URL to fetch
        filename (str): Optional filename to cache locally
        stream (bool): If True, returns a streaming response

    Returns:
        bytes: Content of the response (unless stream=True)
    """
    session = runtime_session()

    if filename:
        destination = naaulu.config.get_temp_dir()
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

    destination = naaulu.config.get_temp_dir()
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

        disable = not logging.getLogger().isEnabledFor(logging.INFO)
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


def list_s3_objects(bucket: str, prefix: str, suffix: str = None, max_keys: int = 1000) -> list:
    """
    List object keys in a public S3 bucket under a given prefix.

    Parameters:
        bucket (str): S3 bucket name
        prefix (str): Prefix path to search under
        suffix (str): Optional suffix filter (e.g. '.gz')
        max_keys (int): Maximum number of keys to return

    Returns:
        list: List of matching object keys
    """
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

    destination = naaulu.config.get_temp_dir()
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

    destination = naaulu.config.get_temp_dir()
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

    headers = {"Authorization": key}
    response = requests.get(path, headers=headers)
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

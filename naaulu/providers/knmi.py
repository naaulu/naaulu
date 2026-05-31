import naaulu.providers.opera as opera

# WMO WSI -> OPERA WIGOS ID for KNMI radars (some have different IDs in OPERA DB)
KNMI_WSI_MAP = {
    "0-20010-0-06356": "0-21010-0-06357",  # Herwijnen
}


def radar(time, duration, wsi):
    wsi = KNMI_WSI_MAP.get(wsi, wsi)
    return opera.radar(time=time, duration=duration, wsi=wsi)


def euradclim(time, duration=None, resolution=None):
    dataset_name = "RAD_OPERA_HOURLY_RAINFALL_ACCUMULATION_EURADCLIM"
    version = "2.0"

    timestamp = time.strftime("%Y%m%d%H%M")
    yearmonth = time.strftime("%Y%m")

    temp_dir = naaulu.config.get_temp_dir()

    file_name = f"RAD_OPERA_HOURLY_RAINFALL_ACCUMULATION_{timestamp}.h5"
    target_path = os.path.join(temp_dir, file_name)

    if not os.path.exists(target_path):

        base_url = "https://api.dataplatform.knmi.nl/open-data/v1"
        archive_name = f"{dataset_name}_{yearmonth}_0002.zip"
        url = f"{base_url}/datasets/{dataset_name}/versions/{version}/files/{archive_name}/url"
        archive_path = naaulu.network.get_temp(
            url, api_key, archive_name, "temporaryDownloadUrl"
        )

        # Extract matching file by filename only (flatten structure)
        with zipfile.ZipFile(archive_path, "r") as zip_ref:
            for entry in zip_ref.infolist():
                if os.path.basename(entry.filename) == file_name:
                    with zip_ref.open(entry) as source, open(
                        target_path, "wb"
                    ) as target:
                        target.write(source.read())
                    break
            else:
                raise FileNotFoundError(f"{file_name} not found in archive")
            
    with h5py.File(target_path, "r") as f:
        from naaulu.providers import opera
        crs, bounds, resolution = opera.extract_odim_georef(f)
        precip = f["dataset1/data1/data"][:]
        
    
    dataset["precipitation"] = dataset["precipitation"].where(
        dataset["precipitation"] != -9999000, numpy.nan
    )

    dataset["precipitation"].attrs = {}

    return dataset

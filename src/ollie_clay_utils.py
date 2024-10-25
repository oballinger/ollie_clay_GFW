import zarr
from dateutil.parser import parse
import pystac_client
import geopandas as gpd
import pandas as pd
from shapely import Point
from matplotlib import pyplot as plt
import stackstac
from rasterio.enums import Resampling
import torch
import yaml
from box import Box
from sklearn import decomposition
import numpy as np
import math
from torchvision.transforms import v2
import sys
sys.path.append("../..")
from src.model import ClayMAEModule
import torch

if torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")

print(f"Using device: {device}")

def load_detection(i, z):
    id = z['detect_id'][i].split('_')
    zs2 = z['tiles_s2'][i][:,:,:]
    zs1= z['tiles_s1'][i][:,:,:]
    detect_id = z['detect_id'][i]
    start = parse(id[4]).strftime('%Y-%m-%d')
    end = parse(id[5]).strftime('%Y-%m-%d')
    lon, lat = [float(x) for x in id[-1].split(';')[-2:]]
    return detect_id, zs1, zs2, start, end, lon, lat

def visualize_data(zs2):
    plt.imshow(zs2[:,:,0:3])
    plt.show()

def search_catalog(start, end, lon, lat):
    STAC_API = "https://earth-search.aws.element84.com/v1"
    COLLECTION = "sentinel-2-l2a"
    
    catalog = pystac_client.Client.open(STAC_API)
    search = catalog.search(
        collections=[COLLECTION],
        datetime=f"{start}/{end}",
        bbox=(lon - 1e-5, lat - 1e-5, lon + 1e-5, lat + 1e-5),
        max_items=1,
        query={"eo:cloud_cover": {"lt": 50}},
    )

    all_items = search.item_collection()
    items = []
    dates = []
    for item in all_items:
        if item.datetime.date() not in dates:
            items.append(item)
            dates.append(item.datetime.date())

    print(f"Found {len(items)} items")
    return items

def calculate_bounds(items, lon, lat, size=100, gsd=10):
    epsg = items[0].properties["proj:epsg"]
    
    poidf = gpd.GeoDataFrame(
        pd.DataFrame(),
        crs="EPSG:4326",
        geometry=[Point(lon, lat)],
    ).to_crs(epsg)

    coords = poidf.iloc[0].geometry.coords[0]
    
    bounds = (
        coords[0] - (size * gsd) // 2,
        coords[1] - (size * gsd) // 2,
        coords[0] + (size * gsd) // 2,
        coords[1] + (size * gsd) // 2,
    )
    
    return bounds, epsg

def retrieve_stack(items, bounds, epsg, gsd=10):
    stack = stackstac.stack(
        items,
        bounds=bounds,
        snap_bounds=False,
        epsg=epsg,
        resolution=gsd,
        dtype="float32",
        rescale=False,
        fill_value=0,
        assets=["blue", "green", "red", "nir"],
        resampling=Resampling.nearest,
    )
    return stack.compute()


def load_model():
    # Set device
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    ckpt = "https://huggingface.co/made-with-clay/Clay/resolve/main/clay-v1-base.ckpt"
    torch.set_default_device(device)

    # Load model
    model = ClayMAEModule.load_from_checkpoint(
        ckpt, metadata_path="../../configs/metadata.yaml", shuffle=False, mask_ratio=0
    )
    model.eval()
    model = model.to(device)
    return model, device

def apply_model(model, device, zs2, stack, lon, lat):
    
    def normalize_timestamp(date):
        week = date.isocalendar().week * 2 * np.pi / 52
        hour = date.hour * 2 * np.pi / 24
        return (math.sin(week), math.cos(week)), (math.sin(hour), math.cos(hour))

    def normalize_latlon(lat, lon):
        lat = lat * np.pi / 180
        lon = lon * np.pi / 180
        return (math.sin(lat), math.cos(lat)), (math.sin(lon), math.cos(lon))

    # Update stack with zs2 values
    stack.data[0, 3, :, :] = zs2[:, :, 3]
    stack.data[0, 2, :, :] = zs2[:, :, 0]
    stack.data[0, 1, :, :] = zs2[:, :, 1]
    stack.data[0, 0, :, :] = zs2[:, :, 2]


    # Extract mean, std, and wavelengths from metadata
    platform = "sentinel-2-l2a"
    metadata = Box(yaml.safe_load(open("../../configs/metadata.yaml")))
    mean, std, waves = [], [], []

    for band in stack.band:
        mean.append(metadata[platform].bands.mean[str(band.values)])
        std.append(metadata[platform].bands.std[str(band.values)])
        waves.append(metadata[platform].bands.wavelength[str(band.values)])

    transform = v2.Compose([v2.Normalize(mean=mean, std=std)])

    # Prepare embeddings
    datetimes = stack.time.values.astype("datetime64[s]").tolist()
    times = [normalize_timestamp(dat) for dat in datetimes]
    week_norm = [dat[0] for dat in times]
    hour_norm = [dat[1] for dat in times]

    latlons = [normalize_latlon(lat, lon)] * len(times)
    lat_norm = [dat[0] for dat in latlons]
    lon_norm = [dat[1] for dat in latlons]

    # Normalize pixels
    pixels = torch.from_numpy(stack.data.astype(np.float32))
    pixels = transform(pixels)

    # Prepare datacube
    datacube = {
        "platform": platform,
        "time": torch.tensor(
            np.hstack((week_norm, hour_norm)),
            dtype=torch.float32,
            device=device,
        ),
        "latlon": torch.tensor(
            np.hstack((lat_norm, lon_norm)), dtype=torch.float32, device=device
        ),
        "pixels": pixels.to(device),
        "gsd": torch.tensor(stack.gsd.values, device=device),
        "waves": torch.tensor(waves, device=device),
    }

    with torch.no_grad():
        unmsk_patch, unmsk_idx, msk_idx, msk_matrix = model.model.encoder(datacube)

    embeddings = unmsk_patch[:, 0, :].cpu().numpy()
    return embeddings

def write_embeddings(detect_id, s1_embeddings,s2_embeddings,dest):


def get_embeddings(z, i):
    detect_id, zs1, zs2, start, end, lon, lat = load_detection(i, z)
    #visualize_data(zs2)
    items = search_catalog(start, end, lon, lat)
    bounds, epsg = calculate_bounds(items, lon, lat)
    stack = retrieve_stack(items, bounds, epsg)
    embeddings = apply_model(model, device, zs2, stack, lon, lat)

    return embeddings

#parallel apply run_pipleine, append results to df
from joblib import Parallel, delayed
import multiprocessing

def run_pipeline(in_zarr, out_zarr):
    z = zarr.open(zarr_path, mode='r')
    model, device = load_model()
    num_cores = multiprocessing.cpu_count()

    return results

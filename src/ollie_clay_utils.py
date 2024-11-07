import time 
import stackstac
from rasterio.enums import Resampling
import sys
from joblib import Parallel, delayed
import multiprocessing
from tqdm import tqdm
import time
import zarr
from dateutil.parser import parse
import pystac_client
import geopandas as gpd
import pandas as pd
from shapely import Point
from matplotlib import pyplot as plt
from rasterio.enums import Resampling
import torch
import yaml
from box import Box
from sklearn import decomposition
import numpy as np
import math
from torchvision.transforms import v2
sys.path.append("../..")
from src.model import ClayMAEModule
import torch
import torch.nn as nn
import time
import xarray as xr
import os 
import ee

if torch.cuda.is_available():
    device = torch.device("cpu")
else:
    device = torch.device("cpu")
ee.Initialize()
#print(f"Using device: {device}")

def load_model():
    # Set device
    device = torch.device("cpu") if torch.cuda.is_available() else torch.device("cpu")
    ckpt = "https://huggingface.co/made-with-clay/Clay/resolve/main/clay-v1-base.ckpt"
    torch.set_default_device(device)

    # Load model
    model = ClayMAEModule.load_from_checkpoint(
        ckpt, metadata_path="configs/metadata.yaml", shuffle=False, mask_ratio=0
    )
    model.eval()
    for module in model.modules():
            if isinstance(module, nn.Transformer):
                module.encoder_layer.self_attn.batch_first = True
        
    model.to(device)
    return model, device

class Thumbnail:
    def __init__(self, detect_id, data, start, end, lon, lat, gsd=None, platform=None):
        self.detect_id = detect_id
        self.data = data
        self.start = start
        self.end = end
        self.lon = lon
        self.lat = lat
        self.platform = platform
        self.embedding = None
        self.gsd = gsd

    @staticmethod
    def load_detection(i, z):
        detect_id = z['detect_id'][i]
        id_parts = detect_id.split('_')
        start=parse(id_parts[4]).strftime('%Y-%m-%d')
        end=parse(id_parts[5]).strftime('%Y-%m-%d')
        #start = (parse(id_parts[4]) + pd.DateOffset(months=2)).strftime('%Y-%m-%d')
        #end = (parse(id_parts[5]) + pd.DateOffset(months=-2)).strftime('%Y-%m-%d')
        lon, lat = [float(x) for x in id_parts[-1].split(';')[-2:]]

        zs2 = z['tiles_s2'][i][:, :, :]
        zs1 = z['tiles_s1'][i][:, :, :]
       
        s1 = Thumbnail(detect_id, zs1, start, end, lon, lat, platform='sentinel-1-grd', gsd=20)
        s2 = Thumbnail(detect_id, zs2, start, end, lon, lat, platform='sentinel-2-l2a', gsd=10)

        return s1, s2

def to_xarray(thumbnail):
    # from the get_stats_df function
    s1_median = {'VH_min': 0.0004993189359083772,
    'VH_max': 0.13001450896263123,
    'VV_min': 0.00935002788901329,
    'VV_max': 1.0629861950874329}

    s2_median={'B2_min': 453.5,
    'B2_max': 1852.0,
    'B3_min': 466.5,
    'B3_max': 1937.1666666666665,
    'B4_min': 250.5,
    'B4_max': 1762.0,
    'B8_min': 139.575,
    'B8_max': 1607.1666666666665}

    data = rescale(thumbnail, s1_median) if thumbnail.platform == 'sentinel-1-grd' else rescale(thumbnail, s2_median)

    stack = xr.DataArray(data, dims=["y", "x", "band"], coords={"y": np.arange(thumbnail.data.shape[0]), "x": np.arange(thumbnail.data.shape[1])})
    if thumbnail.platform == "sentinel-1-grd":
        stack = stack.assign_coords({"band": ["vh", "vv"]})
    elif thumbnail.platform == "sentinel-2-l2a":
        stack = stack.assign_coords({"band": ["nir","blue", "green", "red"]})

    datetime_ns = pd.to_datetime(thumbnail.start).to_datetime64()

    #stack.attrs = stac_item.properties
    #add time as an index
    stack = stack.expand_dims(time=[datetime_ns])
    #reorder such that its time, band, y, x

    stack = stack.transpose("time", "band", "y", "x")
    stack = stack.assign_coords({"time": [datetime_ns]})
    stack = stack.assign_coords({"gsd": thumbnail.gsd})
    thumbnail.stack = stack

    return stack


def apply_model(model, device, thumbnail):

    def normalize_timestamp(date):
        week = date.isocalendar().week * 2 * np.pi / 52
        hour = date.hour * 2 * np.pi / 24
        return (math.sin(week), math.cos(week)), (math.sin(hour), math.cos(hour))

    def normalize_latlon(lat, lon):
        lat = lat * np.pi / 180
        lon = lon * np.pi / 180
        return (math.sin(lat), math.cos(lat)), (math.sin(lon), math.cos(lon))

    stack=thumbnail.stack

    # Extract mean, std, and wavelengths from metadata
    platform = thumbnail.platform.replace("grd",'rtc')

    metadata = Box(yaml.safe_load(open("configs/metadata.yaml")))
    mean, std, waves = [], [], []

    for band in stack.band:
        mean.append(metadata[platform].bands.mean[str(band.values)])
        std.append(metadata[platform].bands.std[str(band.values)])
        waves.append(metadata[platform].bands.wavelength[str(band.values)])

    transform = v2.Compose([v2.Normalize(mean=mean, std=std)])

    datetimes = stack.time.values.astype("datetime64[s]").tolist()
    #datetimes= [datetimes[0]]*len(datetimes)
    times = [normalize_timestamp(dat) for dat in datetimes]
    week_norm = [dat[0] for dat in times]
    hour_norm = [dat[1] for dat in times]

    latlons = [normalize_latlon(thumbnail.lat, thumbnail.lon)] * len(times)
    #latlons = [normalize_latlon(0,0)] * len(times)
    
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

def write_embeddings(detect_id, embeddings, out_dir):
    # Ensure the output directory exists
    os.makedirs(out_dir, exist_ok=True)

    embeddings_path = f"{out_dir}/{detect_id}.npz"

    # Save embeddings as a compressed .npz file
    np.savez_compressed(embeddings_path, s1=embeddings[0], s2=embeddings[1])

    return embeddings_path

def get_embeddings_single_process(z, indices, model, out_dir):

    for i in tqdm(indices, desc="Processing detections", unit="detection"):
        try:
            s2, s1 = Thumbnail.load_detection(i, z)
            detect_id = s2.detect_id
            embeddings = []

            for thumbnail in [s1, s2]:
                #stac_item = search_catalog(thumbnail)
                to_xarray(thumbnail)
                embedding = apply_model(model, device, thumbnail)
                embeddings.append(embedding)
            write_embeddings(detect_id, embeddings, out_dir)
        except Exception as e:
            print(f"Error processing detection {i}: {e}")

def write_zarr(embeddings_dir, in_zarr):
    s1 = []
    s2 = []
    detect_ids=[]
    print('Reading embeddings')
    for root, dirs, files in os.walk(embeddings_dir):
        for file in files:
            if file.endswith(".npz"):
                embeddings=np.load(os.path.join(root, file))
                s1.append(embeddings['s1'])
                s2.append(embeddings['s2'])
                detect_ids.append(file.split('/')[-1].replace('.npz',''))
    print(in_zarr)
    z = zarr.open(in_zarr, mode="r")
    out_zarr=in_zarr.replace('.zarr','_embeddings.zarr')
    z_out = zarr.open(out_zarr, mode="w")
    zarr.copy_all(z, z_out, log=sys.stdout)

    # Create the order_map and sort the detect_ids
    order_map = {value: i for i, value in enumerate(z_out['detect_id'][:])}
    sorted_detect_ids = sorted(detect_ids, key=lambda x: order_map[x])

    order_map={value: i for i, value in enumerate(z_out['detect_id'][:])}
    sorted_detect_ids=sorted(detect_ids, key=lambda x: order_map[x])
    sorted_s1=[s1[detect_ids.index(i)] for i in sorted_detect_ids]
    sorted_s2=[s2[detect_ids.index(i)] for i in sorted_detect_ids]

    z_out.create_dataset("s1_embeddings", data=sorted_s1)
    z_out.create_dataset("s2_embeddings", data=sorted_s2)

def run_pipeline(in_zarr, out_dir):
    print('Running pipeline')
    num_cores = multiprocessing.cpu_count()
    z = zarr.open(in_zarr, mode="r")
    model, _ = load_model()
    
    # Split data into batches
    indices = list(range(len(z['detect_id'])))
    batch_size = len(indices) // num_cores
    batches = [indices[i:i + batch_size] for i in range(0, len(indices), batch_size)]

    # Process batches in parallel
    Parallel(n_jobs=num_cores)(
        delayed(get_embeddings_single_process)(z, batch, model, out_dir) for batch in batches
    )

    write_zarr(out_dir, in_zarr)


def search_catalog(thumbnail):
    STAC_API = "https://earth-search.aws.element84.com/v1"
    catalog = pystac_client.Client.open(STAC_API)
    kwargs={'collections':[thumbnail.platform],
            'datetime':f"{thumbnail.start}/{thumbnail.end}",
            'bbox':(thumbnail.lon - 1e-5, thumbnail.lat - 1e-5, thumbnail.lon + 1e-5, thumbnail.lat + 1e-5),
            'max_items':1}
    if "2" in thumbnail.platform:
        kwargs['query']={"eo:cloud_cover": {"lt": 50}}
    search = catalog.search(**kwargs)
    stac_item = search.item_collection()[0]
    return stac_item

def calculate_bounds(thumbnail, stac_item, size=100):
    epsg = stac_item.properties["proj:epsg"]
    gsd=thumbnail.gsd
    poidf = gpd.GeoDataFrame(
        pd.DataFrame(),
        crs="EPSG:4326",
        geometry=[Point(thumbnail.lon, thumbnail.lat)],
    ).to_crs(epsg)

    coords = poidf.iloc[0].geometry.coords[0]
    
    bounds = (
        coords[0] - (size * gsd) // 2,
        coords[1] - (size * gsd) // 2,
        coords[0] + (size * gsd) // 2,
        coords[1] + (size * gsd) // 2,
    )
    
    return bounds, epsg

def stack_stac(stac_item, bounds, epsg, gsd):
    if stac_item.collection_id == "sentinel-1-grd":
        assets=["vv","vh"]
    elif stac_item.collection_id == "sentinel-2-l2a":
        assets=["blue", "green", "red", "nir"]

    stack = stackstac.stack(
        stac_item,
        bounds=bounds,
        snap_bounds=False,
        epsg=epsg,
        resolution=gsd,
        #dtype="float32",
        rescale=False,
        fill_value=0,
        assets=assets,
        resampling=Resampling.nearest,
    )
    stack = stack.compute()

    return stack


def get_s2_sr_cld_col(aoi, start_date, end_date):
    CLOUD_FILTER = 60

    # Import and filter S2 SR.
    s2_sr_col = (ee.ImageCollection('COPERNICUS/S2_SR')
        .filterBounds(aoi)
        .filterDate(start_date, end_date)
        .filter(ee.Filter.lte('CLOUDY_PIXEL_PERCENTAGE', CLOUD_FILTER)))

    # Import and filter s2cloudless.
    s2_cloudless_col = (ee.ImageCollection('COPERNICUS/S2_CLOUD_PROBABILITY')
        .filterBounds(aoi)
        .filterDate(start_date, end_date))

    # Join the filtered s2cloudless collection to the SR collection by the 'system:index' property.
    return ee.ImageCollection(ee.Join.saveFirst('s2cloudless').apply(**{
        'primary': s2_sr_col,
        'secondary': s2_cloudless_col,
        'condition': ee.Filter.equals(**{
            'leftField': 'system:index',
            'rightField': 'system:index'
        })
    }))


def get_s2_sr_cld_col(aoi, start_date, end_date):
    CLOUD_FILTER = 60
    s2_sr_col = (ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED')
        .filterBounds(aoi)
        .filterDate(start_date, end_date)
        .filter(ee.Filter.lte('CLOUDY_PIXEL_PERCENTAGE', CLOUD_FILTER)))

    # Import and filter s2cloudless.
    s2_cloudless_col = (ee.ImageCollection('COPERNICUS/S2_CLOUD_PROBABILITY')
        .filterBounds(aoi)
        .filterDate(start_date, end_date))

    # Join the filtered s2cloudless collection to the SR collection by the 'system:index' property.
    return ee.ImageCollection(ee.Join.saveFirst('s2cloudless').apply(**{
        'primary': s2_sr_col,
        'secondary': s2_cloudless_col,
        'condition': ee.Filter.equals(**{
            'leftField': 'system:index',
            'rightField': 'system:index'
        })
    }))

def stack_gee(thumbnail):

    lat, lon = thumbnail.lat, thumbnail.lon
    pt = ee.Geometry.Point(lon, lat)

    platform = thumbnail.platform
    start = thumbnail.start

    if platform == "sentinel-1-grd":
        bounds = pt.buffer(200)
        bands = ['VV', 'VH']
        collection = ee.ImageCollection("COPERNICUS/S1_GRD_FLOAT").filterBounds(bounds)\
            .filter(ee.Filter.eq('instrumentMode', 'IW'))\
            .filter(ee.Filter.listContains('transmitterReceiverPolarisation', 'VV'))\
            .filterDate(thumbnail.start, thumbnail.end)\
            .select(bands)
    
    elif platform == "sentinel-2-l2a":
        bounds = pt.buffer(100)
        bands = ['B2', 'B3', 'B4', 'B8']
        collection = get_s2_sr_cld_col(bounds, thumbnail.start, thumbnail.end)\
            .select(bands)  # Blue, Green, Red, and NIR for Sentinel-2

    image = collection.median().clip(bounds)
    image_max = image.reduceRegion(reducer=ee.Reducer.max(), geometry=bounds, scale=thumbnail.gsd).getInfo()
    image_min = image.reduceRegion(reducer=ee.Reducer.min(), geometry=bounds, scale=thumbnail.gsd).getInfo()

    # Combine min and max values into a single DataFrame
    data = {}
    for band in image_min.keys():
        data[f"{band}_min"] = [image_min[band]]
        data[f"{band}_max"] = [image_max[band]]

    df = pd.DataFrame(data)
    print(df)
    return df


def get_stats(i, z):
    # Load the detection thumbnail
    s2, s1 = Thumbnail.load_detection(i, z)
    
    # Apply stack_gee and collect results for s1 and s2
    s1_stats = stack_gee(s1)
    s2_stats = stack_gee(s2)
    
    return s1_stats, s2_stats

def get_stats_df(z, n=100):
    random_indices = np.random.choice(50000, n)

    s1_df=pd.DataFrame()
    s2_df=pd.DataFrame()

    for i in random_indices:
        try:
            s1_stats, s2_stats = get_stats(i, z)
            s1_df = pd.concat([s1_df, s1_stats])
            s2_df = pd.concat([s2_df, s2_stats])
        except:
            pass
    s1_median=s1_df.median(axis=0).to_dict()
    s2_median=s2_df.median(axis=0).to_dict()
    return s1_median, s2_median

def stretch(image, target_min=0, target_max=1):
    image=image.astype(np.float32)
    normalized=image/255
    stretched = normalized * (target_max - target_min) + target_min
    return stretched.astype(np.float32)

def rescale(thumbnail, minmax_df):
    if thumbnail.platform == 'sentinel-1-grd':
        vh=stretch(thumbnail.data[:,:,0], minmax_df['VH_min'], minmax_df['VH_max'])
        vv=stretch(thumbnail.data[:,:,1], minmax_df['VV_min'], minmax_df['VV_max'])
        data=np.stack([vh, vv], axis=-1)
    elif thumbnail.platform == 'sentinel-2-l2a':
        nir=stretch(thumbnail.data[:,:,0], minmax_df['B8_min'], minmax_df['B8_max'])
        blue=stretch(thumbnail.data[:,:,1], minmax_df['B2_min'], minmax_df['B2_max'])
        green=stretch(thumbnail.data[:,:,2], minmax_df['B3_min'], minmax_df['B3_max'])
        red=stretch(thumbnail.data[:,:,3], minmax_df['B4_min'], minmax_df['B4_max'])
        data=np.stack([nir, blue, green, red], axis=-1)

    return data.astype(np.float32)

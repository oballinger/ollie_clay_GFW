import time 
#import stackstac
from rasterio.enums import Resampling
import sys
from joblib import Parallel, delayed
import multiprocessing
from tqdm import tqdm
import time
import zarr
from dateutil.parser import parse
#import pystac_client
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
import torch
import torch.nn as nn
import time
import os 
import xarray as xr
import ee
from PIL import Image
import numpy as np
from google.cloud import bigquery
from glob import glob

#ee.Initialize()
sys.path.append("model")

from src.module import ClayMAEModule

#device = torch.device("cuda")
#device = torch.device("cpu")
#os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

def load_model(device):
    #device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    torch.cuda.empty_cache()

    # Path to the checkpoint
    ckpt = "clay-v1.5.ckpt"
    current_dir = os.getcwd()
    metadata_path = os.path.join(current_dir, "model/configs", "metadata.yaml")

    model = ClayMAEModule.load_from_checkpoint(
        ckpt, metadata_path=metadata_path, shuffle=False, mask_ratio=0
    ).to(device)
    model.eval()
    model.batch_first=True

    for module in model.modules():
        if isinstance(module, nn.Transformer):
            module.encoder_layer.self_attn.batch_first = True
        
    model.to(device)
    print("Loaded", ckpt)
    return model, device


class Thumbnail:
    """
    A class to represent a thumbnail image with associated metadata. Two instances of this class is created for each 
    detection, one for Sentinel-1 and one for Sentinel-2.
    Attributes:
    -----------
    detect_id : str
        The detection ID of the thumbnail.
    data : ndarray
        The image data of the thumbnail.
    start : str
        The start date of the detection period.
    end : str
        The end date of the detection period.
    lon : float
        The longitude coordinate of the detection.
    lat : float
        The latitude coordinate of the detection.
    platform : str, optional
        The platform from which the image was captured (default is None).
    embedding : None
        Placeholder for embedding data (default is None).
    gsd : int, optional
        The ground sample distance of the image (default is None).
    Methods:
    --------
    load_detection(i, z):
        Static method to load detection data and create Thumbnail instances for Sentinel-1 and Sentinel-2 platforms.
    """

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
        """
        Load detection data and create Thumbnail objects for Sentinel-1 and Sentinel-2.
        Args:
            i (int): Index of the detection in the dataset.
            z (dict): Dictionary containing detection data with keys 'detect_id', 'tiles_s1', and 'tiles_s2'.
        Returns:
            tuple: A tuple containing two Thumbnail objects:
                - s1 (Thumbnail): Thumbnail object for Sentinel-1 data.
                - s2 (Thumbnail): Thumbnail object for Sentinel-2 data.
        Raises:
            KeyError: If the required keys are not present in the dictionary `z`.
            ValueError: If the detection ID format is incorrect or cannot be parsed.
        """

        detect_id = z['detect_id'][i]
        id_parts = detect_id.split('_')
        start=parse(id_parts[4]).strftime('%Y-%m-%d')
        end=parse(id_parts[5]).strftime('%Y-%m-%d')
        #start = (parse(id_parts[4]) + pd.DateOffset(months=2)).strftime('%Y-%m-%d')
        #end = (parse(id_parts[5]) + pd.DateOffset(months=-2)).strftime('%Y-%m-%d')
        lon, lat = [float(x) for x in id_parts[-1].split(';')[-2:]]

        s2_tile = z['tiles_s2'][i][:, :, :]
        s1_tile = z['tiles_s1'][i][:, :, :]

        s1 = Thumbnail(detect_id, s1_tile, start, end, lon, lat, platform='sentinel-1-grd', gsd=20)
        s2 = Thumbnail(detect_id, s2_tile, start, end, lon, lat, platform='sentinel-2-l2a', gsd=10)

        return s1, s2
    
    def load_s2_png(file):
        rgb=np.array(Image.open(file))[:,:,::-1]
        nir=np.array(Image.open(file.replace('RGB','NIR')))
        data=np.dstack([nir,rgb])
        lon, lat = [float(x.split('_')[0]) for x in file.split(';')[-2:]]
        date=parse(file.split('_')[2]).strftime('%Y-%m-%d')
        detect_id=file.split('/')[-1].replace('.png','')
        thumbnail= Thumbnail(detect_id, data, date, date, lon, lat, platform='sentinel-2-l2a', gsd=10)
        to_xarray(thumbnail)
        return thumbnail
        

def to_xarray(thumbnail):
    """
    Convert a thumbnail object to an xarray DataArray.
    Parameters:
    thumbnail (object): An object containing the thumbnail data and metadata. 
                        It must have the following attributes:
                        - platform (str): The platform of the thumbnail, either 'sentinel-1-grd' or 'sentinel-2-l2a'.
                        - data (ndarray): The image data as a NumPy array.
                        - start (str): The start time of the thumbnail in a format that can be parsed by pandas.to_datetime.
                        - gsd (float): The ground sampling distance of the thumbnail.
    Returns:
    xarray.DataArray: A DataArray containing the rescaled image data with appropriate coordinates and attributes.
                      The dimensions are ordered as (time, band, y, x).
    """
    
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
    """
    Apply the pretrained clay model to a thumbnail image and return the embeddings.
    
    Parameters:
    model (torch.nn.Module): The model to be applied.
    device (torch.device): The device on which to perform computations.
    thumbnail (Thumbnail): The thumbnail image containing the stack and metadata.
    
    Returns:
    numpy.ndarray: The embeddings generated by the model.
    The function performs the following steps:
    1. Normalize the timestamp and latitude/longitude of the thumbnail.
    2. Extract mean, standard deviation, and wavelengths from metadata.
    3. Normalize the pixel values of the stack.
    4. Prepare a datacube with normalized time, latitude/longitude, pixel values, ground sample distance (gsd), and wavelengths.
    5. Apply the model's encoder to the datacube to obtain embeddings.
    
    Helper Functions:
    - normalize_timestamp(date): Normalize the timestamp to sine and cosine components of week and hour.
    - normalize_latlon(lat, lon): Normalize latitude and longitude to sine and cosine components.
    """

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
    current_dir = os.path.dirname(os.path.abspath(__file__))
    metadata_path = os.path.join(current_dir, "model/configs", "metadata.yaml")
    metadata = Box.from_yaml(filename=metadata_path)
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

def apply_model_batched(model, device, thumbnails):
    """
    Apply the pretrained clay model to a batch of thumbnail images and return the embeddings.

    Parameters:
    model (torch.nn.Module): The model to be applied.
    device (torch.device): The device on which to perform computations.
    thumbnails (list of Thumbnail): A list of thumbnail images containing stacks and metadata.

    Returns:
    list of numpy.ndarray: A list of embeddings generated by the model for each thumbnail.

    Helper Functions:
    - normalize_timestamp(date): Normalize the timestamp to sine and cosine components of week and hour.
    - normalize_latlon(lat, lon): Normalize latitude and longitude to sine and cosine components.
    """
    
    def normalize_timestamp(date):
        week = date.isocalendar().week * 2 * np.pi / 52
        hour = date.hour * 2 * np.pi / 24
        return (math.sin(week), math.cos(week)), (math.sin(hour), math.cos(hour))

    def normalize_latlon(lat, lon):
        lat = lat * np.pi / 180
        lon = lon * np.pi / 180
        return (math.sin(lat), math.cos(lat)), (math.sin(lon), math.cos(lon))

    current_dir = os.path.dirname(os.path.abspath(__file__))
    metadata_path = os.path.join(current_dir, "model/configs", "metadata.yaml")
    metadata = Box.from_yaml(filename=metadata_path)

    embeddings_list = []

    for thumbnail in thumbnails:
        stack = thumbnail.stack

        # Extract mean, std, and wavelengths from metadata
        platform = thumbnail.platform.replace("grd", "rtc")
        mean, std, waves = [], [], []

        for band in stack.band:
            mean.append(metadata[platform].bands.mean[str(band.values)])
            std.append(metadata[platform].bands.std[str(band.values)])
            waves.append(metadata[platform].bands.wavelength[str(band.values)])

        transform = v2.Compose([v2.Normalize(mean=mean, std=std)])

        datetimes = stack.time.values.astype("datetime64[s]").tolist()
        times = [normalize_timestamp(dat) for dat in datetimes]
        week_norm = [dat[0] for dat in times]
        hour_norm = [dat[1] for dat in times]

        latlons = [normalize_latlon(thumbnail.lat, thumbnail.lon)] * len(times)
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
        embeddings_list.append(embeddings)

    return embeddings_list


def write_embeddings(detect_id, embeddings, out_dir):
    """
    Save embeddings to a compressed .npz file in the specified output directory.
    Parameters:
    detect_id (str): Identifier for the detection, used to name the output file.
    embeddings (tuple): A tuple containing two arrays of embeddings (s1 and s2).
    out_dir (str): The directory where the output file will be saved.
    Returns:
    str: The path to the saved .npz file.
    """

    # Ensure the output directory exists
    os.makedirs(out_dir, exist_ok=True)

    embeddings_path = f"{out_dir}/{detect_id}.npz"

    # Save embeddings as a compressed .npz file
    np.savez_compressed(embeddings_path, s1=embeddings[0], s2=embeddings[1])

    return embeddings_path

def get_embeddings_single_process(z, indices, model, out_dir=None):
    """
    Processes a list of detection indices to generate embeddings using a specified model.
    Args:
        z (object): An object containing necessary data for loading detections.
        indices (list): A list of detection indices to process.
        model (object): The model to be used for generating embeddings.
        out_dir (str): The directory where the embeddings will be saved.
    Returns:
        None
    Raises:
        Exception: If there is an error processing a detection, it will be caught and printed.
    """
    embeddings_df=pd.DataFrame()
    for i in tqdm(indices, desc="Extracting Embeddings", unit="detection"):
        try:
            s2, s1 = Thumbnail.load_detection(i, z)
            detect_id = s2.detect_id
            embeddings = []       
            for thumbnail in [s1, s2]:
                #stac_item = search_catalog(thumbnail)
                to_xarray(thumbnail)
                embedding = apply_model(model, device, thumbnail)
                embeddings.append(embedding)
            if out_dir:
                write_embeddings(detect_id, embeddings, out_dir)
            else:
                row=pd.DataFrame({'detect_id':detect_id, 's1_embeddings':[embeddings[0].flatten()], 's2_embeddings':[embeddings[1].flatten()]})
                embeddings_df=pd.concat([embeddings_df, row])
        except Exception as e:
            print(f"Error processing detection {i}: {e}")
        
    return embeddings_df
        


def write_zarr(embeddings_dir, in_zarr):
    """
    Reads .npz files containing embeddings from a specified directory, sorts them according to the order of detect_ids 
    in an input Zarr file, and writes the sorted embeddings to a new Zarr file.
    Args:
        embeddings_dir (str): The directory containing the .npz files with embeddings.
        in_zarr (str): The path to the input Zarr file.
    Returns:
        None
    Raises:
        KeyError: If a detect_id from the .npz files is not found in the input Zarr file.
    """

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

def run_pipeline(input, out_dir=None, n_jobs=-1, verbose=True):
    """
    Run the data processing pipeline in parallel.
    This function processes data stored in a Zarr file or dataframe, splits the data into batches,
    and processes each batch in parallel using multiple CPU cores. The results are
    then written to the specified output directory.
    Parameters:
    in_zarr (str): Path to the input Zarr file.
    out_dir (str): Path to the output directory where results will be saved.
    Returns:
    None
    """
    
    #print('Running pipeline')
        #if input is str
    if isinstance(input, str):
        z = zarr.open(input, mode="r")
    else:
        z=input

    model, _ = load_model()
    
    # Split data into batches
    indices = list(range(len(z['detect_id'])))
    
    batch_size = len(indices) // n_jobs
    
    batches = [indices[i:i + batch_size] for i in range(0, len(indices), batch_size)]

   # Process batches in parallel and combine into one dataframe
   
    if n_jobs == 1:
        embdeddings_df=get_embeddings_single_process(z, indices, model, out_dir)
    else:
        results = Parallel(n_jobs=n_jobs)(
            delayed(get_embeddings_single_process)(z, batch, model, out_dir) for batch in batches
        )
        embdeddings_df=pd.concat(results)

    
    if out_dir:
        write_zarr(out_dir, input)
    else :
        return embdeddings_df


def get_s2_sr_cld_col(aoi, start_date, end_date):
    """
    Retrieves a Sentinel-2 Surface Reflectance (S2 SR) image collection filtered by area of interest (AOI), 
    date range, and cloud coverage, and joins it with the Sentinel-2 cloud probability collection.
    Args:
        aoi (ee.Geometry): The area of interest to filter the image collection.
        start_date (str): The start date for the image collection filter in 'YYYY-MM-DD' format.
        end_date (str): The end date for the image collection filter in 'YYYY-MM-DD' format.
    Returns:
        ee.ImageCollection: An Earth Engine ImageCollection with the S2 SR images joined with the 
                            corresponding cloud probability images from the S2 cloud probability collection.
    """
    
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



def stack_gee(thumbnail):
    """
    Processes satellite imagery from Google Earth Engine (GEE) based on the given thumbnail parameters. Used for 
    converting the thumbnail data from 0-255 to the sensor's native units.
    Args:
        thumbnail (object): An object containing the following attributes:
            - lat (float): Latitude of the point of interest.
            - lon (float): Longitude of the point of interest.
            - platform (str): The satellite platform, either "sentinel-1-grd" or "sentinel-2-l2a".
            - start (str): The start date for the image collection period.
            - end (str): The end date for the image collection period.
            - gsd (float): The ground sampling distance (resolution) for the image.
    Returns:
        pd.DataFrame: A DataFrame containing the minimum and maximum values for each band in the selected image collection.
    """

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
    """
    Computes statistics for detection thumbnails.
    Args:
        i (int): An identifier for the detection.
        z (int): A parameter used for loading the detection thumbnail.
    Returns:
        tuple: A tuple containing two elements:
            - s1_stats: Statistics for the first detection thumbnail.
            - s2_stats: Statistics for the second detection thumbnail.
    """

    # Load the detection thumbnail
    s2, s1 = Thumbnail.load_detection(i, z)
    
    # Apply stack_gee and collect results for s1 and s2
    s1_stats = stack_gee(s1)
    s2_stats = stack_gee(s2)
    
    return s1_stats, s2_stats

def get_stats_df(z, n=100):
    """
    Generate median statistics from random samples.

    This function generates random indices, retrieves statistics for each index,
    and calculates the median of these statistics. This is used later on convert 
    the thumbnails from 0-255 to the sensor's native units. Would not be necessary 
    if the data was already in the sensor's native units.

    Parameters:
    z (any): A parameter to be passed to the get_stats function.
    n (int, optional): The number of random samples to generate. Default is 100.

    Returns:
    tuple: A tuple containing two dictionaries:
        - s1_median (dict): Median statistics for the first set.
        - s2_median (dict): Median statistics for the second set.
    """
    ee.Initialize()
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
    """
    Stretches the pixel values of an image to a specified range, in this case from 
    0-255 to the range of the sensor. The image is first normalized to the range 0-1,
    then stretched to the specified target range.

    Parameters:
    image (numpy.ndarray): The input image to be stretched.
    target_min (float, optional): The minimum value of the target range. Default is 0.
    target_max (float, optional): The maximum value of the target range. Default is 1.
    Returns:
    numpy.ndarray: The image with pixel values stretched to the specified range.
    """

    image=image.astype(np.float32)
    normalized=image/255
    stretched = normalized * (target_max - target_min) + target_min
    return stretched.astype(np.float32)

def rescale(thumbnail, minmax_df):
    """
    Rescales the data in a thumbnail image based on the platform type and provided min-max values.
    Parameters:
    thumbnail (object): An object containing the image data and platform type. 
                        The data attribute is expected to be a 3D numpy array.
    minmax_df (DataFrame): A pandas DataFrame containing the min and max values for rescaling.
    Returns:
    numpy.ndarray: The rescaled image data as a 3D numpy array with dtype float32.
    Notes:
    - For 'sentinel-1-grd' platform, the data is expected to have two channels (VH and VV).
    - For 'sentinel-2-l2a' platform, the data is expected to have four channels (B8, B2, B3, B4).
    """

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












# ------------------------------------ old functions ------------------------------------
"""
These functions are not used in the current pipeline but are kept for reference
they allow for the retrieval of data from the STAC catalog and the stacking of the data
into a single xarray dataset. In case we want to move away from GEE.
"""

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
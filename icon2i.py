#!/bin/env python3
import os
import requests
import json
import numpy as np
import xarray as xr
from datetime import datetime, timedelta, timezone
from bs4 import BeautifulSoup
from collections import Counter
from pydrive2.auth import GoogleAuth
from pydrive2.drive import GoogleDrive

# ------------------- CONFIG -------------------
WORKDIR = os.getcwd()
VARIABLES = ['T_2M', 'RELHUM', 'TOT_PREC', 'CLCT', 'CLCL', 'CLCM', 'CLCH', 'U_10M', 'V_10M', 'VMAX_10M', 'LPI', 'CAPE_ML', 'CAPE_CON', 'UH_MAX', 'PMSL', 'HSURF', 'ASOB_S']
VENUES_PATH = f"{WORKDIR}/comuni_italia.json"

LAPSE_DRY = 0.0098
LAPSE_MOIST = 0.006
LAPSE_P = 0.12

SEASON_THRESHOLDS = {
    "winter": {"start_day": 1, "end_day": 80, "fog_rh": 97, "haze_rh": 90, "fog_wind": 3, "haze_wind": 6},
    "spring": {"start_day": 81, "end_day": 172, "fog_rh": 95, "haze_rh": 87, "fog_wind": 3.5, "haze_wind": 7},
    "summer": {"start_day": 173, "end_day": 263, "fog_rh": 93, "haze_rh": 83, "fog_wind": 4, "haze_wind": 8},
    "autumn": {"start_day": 264, "end_day": 365, "fog_rh": 96, "haze_rh": 88, "fog_wind": 3.2, "haze_wind": 6.5}
}

A_MAGNUS, B_MAGNUS = 17.625, 243.04
CET = timezone(timedelta(hours=1))
CEST = timezone(timedelta(hours=2))

# ------------------- FUNZIONI METEO -------------------
def utc_to_local(dt_utc):
    now_utc = datetime.now(timezone.utc)
    if 31 <= now_utc.month <= 10 or (now_utc.month == 3 and now_utc.day >= 25) or (now_utc.month == 10 and now_utc.day <= 25):
        return dt_utc.astimezone(CEST)
    return dt_utc.astimezone(CET)

def dew_point_celsius(t_c, rh_percent):
    alpha = np.log(rh_percent/100.0) + (A_MAGNUS * t_c) / (B_MAGNUS + t_c)
    return (B_MAGNUS * alpha) / (A_MAGNUS - alpha)

def get_run_datetime_now_utc():
    now = datetime.now(timezone.utc)
    if now.hour < 3: return (now - timedelta(days=1)).strftime("%Y%m%d"), "12"
    elif now.hour < 14: return now.strftime("%Y%m%d"), "00"
    return now.strftime("%Y%m%d"), "12"

def download_icon_data(run_date, run_hour):
    grib_dir = f"{WORKDIR}/grib_data/{run_date}{run_hour}"
    os.makedirs(grib_dir, exist_ok=True)
    
    existing_files = [f for f in os.listdir(grib_dir) if f.endswith('.grib')]
    if len(existing_files) >= len(VARIABLES):
        print(f"Tutti i GRIB già presenti in {grib_dir} ({len(existing_files)}/{len(VARIABLES)})")
        return True
    
    base_url = f'https://meteohub.agenziaitaliameteo.it/nwp/ICON-2I_SURFACE_PRESSURE_LEVELS/{run_date}{run_hour}/'
    if requests.head(base_url, timeout=10).status_code != 200:
        raise RuntimeError(f"Run {run_date}{run_hour} non disponibile")
    
    downloaded = 0
    for var in VARIABLES:
        local_path = f"{grib_dir}/{var}.grib"
        if os.path.isfile(local_path) and os.path.getsize(local_path) > 1000:
            print(f"Skip {var} (già presente, {os.path.getsize(local_path)//1000}KB)")
            continue
            
        var_url = f'{base_url}{var}/'
        try:
            r = requests.get(var_url, timeout=30)
            r.raise_for_status()
            soup = BeautifulSoup(r.text, 'html.parser')
            grib_files = [a.get('href') for a in soup.find_all('a') if a.get('href', '').endswith('.grib')]
            if grib_files:
                file_url = var_url + grib_files[0]
                with requests.get(file_url, stream=True, timeout=180) as resp:
                    resp.raise_for_status()
                    with open(local_path, 'wb') as f: 
                        for chunk in resp.iter_content(chunk_size=8192):
                            f.write(chunk)
                downloaded += 1
                print(f"Scaricato {var} ({os.path.getsize(local_path)//1000}KB)")
        except Exception as e:
            print(f"Skip {var}: {e}")
            if os.path.exists(local_path): os.remove(local_path)
    
    final_count = len([f for f in os.listdir(grib_dir) if f.endswith('.grib')])
    print(f"Download completato: {downloaded} nuovi file → TOTALE {final_count}/{len(VARIABLES)}")
    return final_count >= len(VARIABLES)

def kelvin_to_celsius(k): return k - 273.15
def mps_to_kmh(mps): return mps * 3.6

def wind_speed_direction(u, v):
    speed_ms = np.sqrt(u**2 + v**2)
    deg = (np.degrees(np.arctan2(-u, -v)) % 360)
    return speed_ms, deg

def wind_dir_to_cardinal(deg):
    return ['N','NE','E','SE','S','SW','W','NW'][int((deg + 22.5) % 360 // 45)]

def get_season_precise(dt_utc): 
    day_of_year = dt_utc.timetuple().tm_yday
    for season, thresh in SEASON_THRESHOLDS.items():
        if thresh["start_day"] <= day_of_year <= thresh["end_day"]:
            return season, thresh
    return "winter", SEASON_THRESHOLDS["winter"]

def is_night(dt_utc): 
    return dt_utc.hour >= 21 or dt_utc.hour <= 6

def altitude_correction(t2m, rh, hs_model, hs_station, pmsl):
    delta_z = hs_model - hs_station
    if delta_z >= 0: return t2m, pmsl
    w_moist = np.clip(rh / 100.0, 0.0, 1.0)
    lapse_t = LAPSE_DRY * (1.0 - w_moist) + LAPSE_MOIST * w_moist
    t_corr = t2m + lapse_t * delta_z
    p_corr = pmsl + (delta_z / 100.0) * LAPSE_P * 100
    return t_corr, p_corr

def calculate_solar_radiation(asob_s, clct, dt_utc, lat, lon):
    is_daytime = not is_night(dt_utc)
    if not is_daytime: return 0
    ghi = np.maximum(asob_s * (1.0 - clct/100.0), 0)
    return np.round(ghi).astype(int)

# ------------------- FUNZIONI DI CLASSIFICAZIONE -------------------
def classify_weather(t2m, rh2m, clct, clcl, clcm, clch, tp_rate, wind_kmh, lpi, cape, uh, season, season_thresh, timestep_hours=1):
    dew_point = dew_point_celsius(t2m, rh2m)
    is_snow = dew_point < 0.1
    low_cloud = clcl if np.isfinite(clcl) else clcm if np.isfinite(clcm) else 0
    drizzle_min, drizzle_max = 0.09, 0.3
    if timestep_hours == 1:
        prec_debole_min, prec_debole_max = 0.3, 2.0
        prec_moderata_min, prec_moderata_max = 2.0, 7.0
        prec_intensa_min = 7.0
    elif timestep_hours == 3:
        prec_debole_min, prec_debole_max = 0.3, 5.0
        prec_moderata_min, prec_moderata_max = 5.0, 20.0
        prec_intensa_min = 20.0
    else:
        prec_debole_min, prec_debole_max = 0.3, 10.0
        prec_moderata_min, prec_moderata_max = 10.0, 30.0
        prec_intensa_min = 30.0
    
    if tp_rate >= prec_intensa_min:
        prec_intensity = "INTENSA"
    elif tp_rate >= prec_moderata_min:
        prec_intensity = "MODERATA"
    elif tp_rate >= prec_debole_min:
        prec_intensity = "DEBOLE"
    else: prec_intensity = None
    
    if prec_intensity:
        octas = clct / 100.0 * 8
        if clch > 60 and low_cloud < 30 and octas > 5: cloud_state = "NUBI ALTE"
        elif octas <= 2: cloud_state = "SERENO"
        elif octas <= 4: cloud_state = "POCO NUVOLOSO"
        elif octas <= 6: cloud_state = "NUVOLOSO"
        else: cloud_state = "COPERTO"
        prec_type = "NEVE" if is_snow else "PIOGGIA"
        return f"{cloud_state} {prec_type} {prec_intensity}"
    
    is_drizzle = drizzle_min <= tp_rate <= drizzle_max and low_cloud >= 50
    t_cold = t2m < 12
    if is_drizzle:
        if (rh2m >= season_thresh["fog_rh"] and wind_kmh <= season_thresh["fog_wind"] and low_cloud >= 80 and t_cold):
            return "NEBBIA"
        if (rh2m >= season_thresh["haze_rh"] and wind_kmh <= season_thresh["haze_wind"] and low_cloud >= 50 and t_cold):
            return "FOSCHIA"
        octas = clct / 100.0 * 8
        if clch > 60 and low_cloud < 30 and octas > 5: cloud_state = "NUBI ALTE"
        elif octas <= 2: cloud_state = "SERENO"
        elif octas <= 4: cloud_state = "POCO NUVOLOSO"
        elif octas <= 6: cloud_state = "NUVOLOSO"
        else: cloud_state = "COPERTO"
        prec_type = "NEVE DEBOLE" if is_snow else "PIOGGERELLA"
        return f"{cloud_state} {prec_type}"
    
    if (rh2m >= season_thresh["fog_rh"] and wind_kmh <= season_thresh["fog_wind"] and low_cloud >= 80 and t_cold): return "NEBBIA"
    if (rh2m >= season_thresh["haze_rh"] and wind_kmh <= season_thresh["haze_wind"] and low_cloud >= 50 and t_cold): return "FOSCHIA"
    
    octas = clct / 100.0 * 8
    if clch > 60 and low_cloud < 30 and octas > 5: return "NUBI ALTE"
    elif octas <= 2: return "SERENO"
    elif octas <= 4: return "POCO NUVOLOSO"
    elif octas <= 6: return "NUVOLOSO"
    else: return "COPERTO"

# ------------------- FUNZIONI UTILI -------------------
def extract_variable(var, lat_idx, lon_idx, is_precip=False):
    if np.isscalar(var) or (hasattr(var, 'size') and var.size == 1): return np.array([float(var)])
    if var.ndim == 1: return var
    spatial_slice = var[..., lat_idx, lon_idx]
    if var.ndim == 2: return spatial_slice
    first_ring = []
    NY, NX = var.shape[-2:]
    for di in [-1,0,1]:
        for dj in [-1,0,1]:
            if di==0 and dj==0: continue
            ni,nj = lat_idx+di, lon_idx+dj
            if 0<=ni<NY and 0<=nj<NX: first_ring.append(var[...,ni,nj])
    if not first_ring: return spatial_slice
    first_ring_mean = np.stack(first_ring, axis=-1).mean(axis=-1)
    if not is_precip: return 0.5*spatial_slice + 0.5*first_ring_mean
    second_ring = []
    for di in [-2,-1,0,1,2]:
        for dj in [-2,-1,0,1,2]:
            if abs(di)<=1 and abs(dj)<=1: continue
            ni,nj = lat_idx+di, lon_idx+dj
            if 0<=ni<NY and 0<=nj<NX: second_ring.append(var[...,ni,nj])
    if not second_ring: return 0.6*spatial_slice + 0.4*first_ring_mean
    second_ring_mean = np.stack(second_ring, axis=-1).mean(axis=-1)
    return 0.5*spatial_slice + 0.3*first_ring_mean + 0.2*second_ring_mean

def build_latlon_from_hsurf(hsurf_da):
    if 'latitude' in hsurf_da.coords and 'longitude' in hsurf_da.coords:
        lat_grid, lon_grid = hsurf_da['latitude'].values, hsurf_da['longitude'].values
        if lat_grid.ndim == 1: lon_grid, lat_grid = np.meshgrid(lon_grid, lat_grid)
        return lat_grid, lon_grid
    nlat, nlon = hsurf_da.shape[-2:]
    return np.meshgrid(np.linspace(3.0,22.0,nlon), np.linspace(33.7,48.89,nlat))

def find_land_nearest(lat_grid, lon_grid, target_lat, target_lon, hsurf_grid):
    for radius in [0.1,0.2,0.5,1.0]:
        y,x = np.where((np.abs(lat_grid-target_lat)<=radius) & (np.abs(lon_grid-target_lon)<=radius) & (hsurf_grid>0))
        if len(y)>0:
            dists = np.sqrt((lat_grid[y,x]-target_lat)**2 + (lon_grid[y,x]-target_lon)**2)
            center_idx = np.argmin(dists)
            return y[center_idx], x[center_idx]
    dist = np.sqrt((lat_grid-target_lat)**2 + (lon_grid-target_lon)**2)
    center_y, center_x = np.unravel_index(np.argmin(dist), dist.shape)
    return center_y, center_x

def load_venues(file_path):
    if not os.path.exists(file_path): raise FileNotFoundError(f"{file_path} non trovato")
    with open(file_path, 'r', encoding='utf-8') as f: data = json.load(f)
    venues = {}
    if isinstance(data, dict):
        for city_name, coords in data.items():
            if isinstance(coords, list) and len(coords)>=3:
                venues[city_name] = {"lat":float(coords[0]),"lon":float(coords[1]),"elev":float(coords[2])}
    print(f"Caricate {len(venues)} città da {file_path}")
    return venues

# ------------------- FUNZIONE DI UPLOAD DRIVE -------------------
def upload_to_drive(local_dir):
    DRIVE_FOLDER_ID = os.environ.get("GDRIVE_ICON2I_ID")
    CREDS_JSON = os.environ.get("GOOGLE_CREDENTIALS_JSON")
    if not DRIVE_FOLDER_ID or not CREDS_JSON:
        raise RuntimeError("Variabili ambiente GDRIVE_ICON2I_ID o GOOGLE_CREDENTIALS_JSON mancanti")
    
    gauth = GoogleAuth()
    gauth.LoadCredentialsFile(CREDS_JSON)
    if gauth.credentials is None:
        gauth.LocalWebserverAuth()
    elif gauth.access_token_expired:
        gauth.Refresh()
    else:
        gauth.Authorize()
    gauth.SaveCredentialsFile(CREDS_JSON)
    drive = GoogleDrive(gauth)
    
    run_basename = os.path.basename(local_dir)
    folder_list = drive.ListFile({'q': f"'{DRIVE_FOLDER_ID}' in parents and title='{run_basename}' and trashed=false"}).GetList()
    if folder_list: folder_id = folder_list[0]['id']
    else:
        folder_metadata = {'title': run_basename, 'parents':[{'id':DRIVE_FOLDER_ID}], 'mimeType':'application/vnd.google-apps.folder'}
        folder = drive.CreateFile(folder_metadata)
        folder.Upload()
        folder_id = folder['id']
    
    for fname in os.listdir(local_dir):
        if not fname.endswith(".json"): continue
        fpath = os.path.join(local_dir, fname)
        gfile_list = drive.ListFile({'q': f"'{folder_id}' in parents and title='{fname}' and trashed=false"}).GetList()
        if gfile_list: continue
        gfile = drive.CreateFile({'title':fname,'parents':[{'id':folder_id}]})
        gfile.SetContentFile(fpath)
        gfile.Upload()
        print(f"Caricato {fname} su Drive")

# ------------------- PROCESS DATA -------------------
def process_data():
    RUN = os.getenv("RUN", "")
    if not RUN:
        run_date, run_hour = get_run_datetime_now_utc()
        RUN = run_date + run_hour
        if not download_icon_data(run_date, run_hour):
            raise RuntimeError(f"Run {RUN} non disponibile o download incompleto")
    
    grib_dir = f"{WORKDIR}/grib_data/{RUN}"
    venues = load_venues(VENUES_PATH)
    output_dir = os.path.join(WORKDIR, "json_output", RUN)
    os.makedirs(output_dir, exist_ok=True)

    # Caricamento dati GRIB in xarray Dataset
    ds_vars = {}
    for var in VARIABLES:
        grib_file = os.path.join(grib_dir, f"{var}.grib")
        if not os.path.exists(grib_file):
            print(f"File GRIB mancante: {grib_file}")
            continue
        ds_vars[var] = xr.open_dataset(grib_file, engine="cfgrib")

    # Assumiamo tutti i dataset abbiano le stesse coordinate
    lat_grid, lon_grid = build_latlon_from_hsurf(ds_vars['HSURF'])

    hsurf_grid = ds_vars['HSURF'].values

    for city_name, info in venues.items():
        lat, lon, elev = info["lat"], info["lon"], info["elev"]
        lat_idx, lon_idx = find_land_nearest(lat_grid, lon_grid, lat, lon, hsurf_grid)

        city_data = []
        times = ds_vars['T_2M']['time'].values
        for i, t in enumerate(times):
            dt_utc = np.datetime64(t).astype(datetime)
            dt_local = utc_to_local(dt_utc)

            t2m = kelvin_to_celsius(extract_variable(ds_vars['T_2M'].values[i], lat_idx, lon_idx))
            rh2m = extract_variable(ds_vars['RELHUM'].values[i], lat_idx, lon_idx)
            tp = extract_variable(ds_vars['TOT_PREC'].values[i], lat_idx, lon_idx, is_precip=True)
            clct = extract_variable(ds_vars['CLCT'].values[i], lat_idx, lon_idx)
            clcl = extract_variable(ds_vars['CLCL'].values[i], lat_idx, lon_idx)
            clcm = extract_variable(ds_vars['CLCM'].values[i], lat_idx, lon_idx)
            clch = extract_variable(ds_vars['CLCH'].values[i], lat_idx, lon_idx)
            u10 = extract_variable(ds_vars['U_10M'].values[i], lat_idx, lon_idx)
            v10 = extract_variable(ds_vars['V_10M'].values[i], lat_idx, lon_idx)
            pmsl = extract_variable(ds_vars['PMSL'].values[i], lat_idx, lon_idx)
            asob_s = extract_variable(ds_vars['ASOB_S'].values[i], lat_idx, lon_idx)

            t2m_corr, pmsl_corr = altitude_correction(t2m, rh2m, hsurf_grid[lat_idx, lon_idx], elev, pmsl)
            wind_speed_ms, wind_deg = wind_speed_direction(u10, v10)
            wind_kmh = mps_to_kmh(wind_speed_ms)
            wind_card = wind_dir_to_cardinal(wind_deg)
            solar_rad = calculate_solar_radiation(asob_s, clct, dt_utc, lat, lon)
            season, season_thresh = get_season_precise(dt_utc)

            weather_desc = classify_weather(
                t2m_corr, rh2m, clct, clcl, clcm, clch, tp, wind_kmh, None, None, None,
                season, season_thresh, timestep_hours=1
            )

            city_data.append({
                "datetime_utc": dt_utc.isoformat(),
                "datetime_local": dt_local.isoformat(),
                "t2m": float(np.round(t2m_corr, 1)),
                "rh2m": float(np.round(rh2m, 1)),
                "tp": float(np.round(tp, 2)),
                "wind_speed_kmh": float(np.round(wind_kmh, 1)),
                "wind_deg": float(np.round(wind_deg, 1)),
                "wind_card": wind_card,
                "pmsl": float(np.round(pmsl_corr, 1)),
                "clct": float(np.round(clct, 1)),
                "clcl": float(np.round(clcl, 1)),
                "clcm": float(np.round(clcm, 1)),
                "clch": float(np.round(clch, 1)),
                "solar_radiation": int(solar_rad),
                "weather": weather_desc
            })

        # Salvataggio JSON per la città
        city_json_file = os.path.join(output_dir, f"{city_name}.json")
        with open(city_json_file, "w", encoding="utf-8") as f:
            json.dump(city_data, f, ensure_ascii=False, indent=2)

        print(f"{city_name} salvata in {city_json_file}")

    # Upload su Google Drive
    upload_to_drive(output_dir)
    print(f"Tutti i dati caricati su Google Drive nella cartella {RUN}")

# ------------------- MAIN -------------------
if __name__ == "__main__":
    process_data()

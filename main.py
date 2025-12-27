#!/bin/env python3
import os
import requests
import json
import numpy as np
import xarray as xr
from datetime import datetime, timedelta, timezone
from bs4 import BeautifulSoup
from collections import Counter

WORKDIR = os.path.dirname(os.path.abspath(__file__))
VARIABLES = ['T_2M', 'RELHUM', 'TOT_PREC', 'CLCT', 'CLCL', 'CLCM', 'CLCH', 'U_10M', 'V_10M', 'VMAX_10M', 'LPI', 'CAPE_ML', 'CAPE_CON', 'UH_MAX', 'PMSL', 'HSURF', 'ASOB_S']
VENUES_PATH = f"{WORKDIR}/comuni_italia_all.json"

# Google Drive API
DRIVE_FOLDER_ID_ICON2I = os.getenv("DRIVE_FOLDER_ID_ICON2I", "")
GDRIVE_SERVICE_ACCOUNT_JSON = os.getenv("GDRIVE_SERVICE_ACCOUNT_JSON", "")

from google.oauth2 import service_account
from googleapiclient.discovery import build
from googleapiclient.http import MediaFileUpload

# Lapse rates interpolati
LAPSE_DRY = 0.0098   # °C/m (9.8 °C/km)
LAPSE_MOIST = 0.006  # °C/m (~6 °C/km)
LAPSE_P = 0.12       # hPa/100m (12 hPa/km → 0.12 hPa/m)

# SOGLIE STAGIONALI con PIOGGRELLA (mm/h orario)
SEASON_THRESHOLDS = {
    "winter": {"start_day": 1, "end_day": 80, "fog_rh": 97, "haze_rh": 90, "fog_wind": 3, "haze_wind": 6},
    "spring": {"start_day": 81, "end_day": 172, "fog_rh": 95, "haze_rh": 87, "fog_wind": 3.5, "haze_wind": 7},
    "summer": {"start_day": 173, "end_day": 263, "fog_rh": 93, "haze_rh": 83, "fog_wind": 4, "haze_wind": 8},
    "autumn": {"start_day": 264, "end_day": 365, "fog_rh": 96, "haze_rh": 88, "fog_wind": 3.2, "haze_wind": 6.5}
}

# Magnus coefficients per dew point
A_MAGNUS, B_MAGNUS = 17.625, 243.04

# ITALIA CET/CEST = UTC+1/UTC+2
CET = timezone(timedelta(hours=1))
CEST = timezone(timedelta(hours=2))

def utc_to_local(dt_utc):
    """Converte UTC → ora locale Italia (CET/CEST automatica)"""
    now_utc = datetime.now(timezone.utc)
    if 31 <= now_utc.month <= 10 or (now_utc.month == 3 and now_utc.day >= 25) or (now_utc.month == 10 and now_utc.day <= 25):
        return dt_utc.astimezone(CEST)
    return dt_utc.astimezone(CET)

def dew_point_celsius(t_c, rh_percent):
    """Calcola dew point con formula Magnus (precisa ±0.35°C)"""
    alpha = np.log(rh_percent/100.0) + (A_MAGNUS * t_c) / (B_MAGNUS + t_c)
    return (B_MAGNUS * alpha) / (A_MAGNUS - alpha)

def get_run_datetime_now_utc():
    """Determina run ICON-2I disponibile: 00/12 UTC (priorità: attuale → ieri)"""
    now = datetime.now(timezone.utc)
    if now.hour < 3: return (now - timedelta(days=1)).strftime("%Y%m%d"), "12"
    elif now.hour < 14: return now.strftime("%Y%m%d"), "00"
    return now.strftime("%Y%m%d"), "12"

def download_icon_data(run_date, run_hour):
    """Scarica GRIB essenziali ICON-2I dal run specificato (SALTA se già presenti)"""
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
    """Calcola modulo e direzione vento da componenti U/V (meteo convenzione)"""
    speed_ms = np.sqrt(u**2 + v**2)
    deg = (np.degrees(np.arctan2(-u, -v)) % 360)
    return speed_ms, deg

def wind_dir_to_cardinal(deg):
    """Converte gradi → N,NE,E,SE,S,SW,W,NW"""
    return ['N','NE','E','SE','S','SW','W','NW'][int((deg + 22.5) % 360 // 45)]

def get_season_precise(dt_utc): 
    """Classifica stagione PRECISA per giorno dell'anno"""
    day_of_year = dt_utc.timetuple().tm_yday
    for season, thresh in SEASON_THRESHOLDS.items():
        if thresh["start_day"] <= day_of_year <= thresh["end_day"]:
            return season, thresh
    return "winter", SEASON_THRESHOLDS["winter"]

def is_night(dt_utc): 
    """Notte meteorologica: 21:00-06:59 UTC"""
    return dt_utc.hour >= 21 or dt_utc.hour <= 6

def altitude_correction(t2m, rh, hs_model, hs_station, pmsl):
    """Correzione altimetrica con lapse rate interpolato tra secco e umido (SOLO se hs_model < hs_station)"""
    delta_z = hs_model - hs_station
    if delta_z >= 0:
        return t2m, pmsl  # Nessuna correzione se modello è più in alto o allo stesso livello
    
    w_moist = np.clip(rh / 100.0, 0.0, 1.0)
    lapse_t = LAPSE_DRY * (1.0 - w_moist) + LAPSE_MOIST * w_moist
    t_corr = t2m + lapse_t * delta_z
    p_corr = pmsl + (delta_z / 100.0) * LAPSE_P * 100
    return t_corr, p_corr

def calculate_solar_radiation(asob_s, clct, dt_utc, lat, lon):
    """Calcola GHI corretto: ASOB_S di notte = 0, daytime scalato per nuvolosità"""
    is_daytime = not is_night(dt_utc)
    if not is_daytime:
        return 0
    ghi = np.maximum(asob_s * (1.0 - clct/100.0), 0)
    return np.round(ghi).astype(int)

def classify_weather(t2m, rh2m, clct, clcl, clcm, clch, tp_rate, wind_kmh, lpi, cape, uh, season, season_thresh, timestep_hours=1):
    """NUVOLOSITÀ + PRECIPITAZIONE: PIOGGIA DEBOLE+ > NEBBIA/FOSCHIA > PIOGGERELLA con SOGLIE DEFINITE"""
    
    # PRIORITÀ 1: TEMPORALE (sempre massimo)
    if (lpi > 2.0 or uh > 50 or cape > 400) and tp_rate > 0.5 * timestep_hours: 
        return "TEMPORALE"
    
    dew_point = dew_point_celsius(t2m, rh2m)
    is_snow = dew_point < 0.1
    low_cloud = clcl if np.isfinite(clcl) else clcm if np.isfinite(clcm) else 0
    
    # SOGLIE ESATTAMENTE COME DEFINITE
    drizzle_min = 0.09
    drizzle_max = 0.3
    
    if timestep_hours == 1:
        prec_debole_min = 0.3
        prec_debole_max = 2.0
        prec_moderata_min = 2.0
        prec_moderata_max = 7.0
        prec_intensa_min = 7.0
    elif timestep_hours == 3:
        prec_debole_min = 0.3
        prec_debole_max = 5.0
        prec_moderata_min = 5.0
        prec_moderata_max = 20.0
        prec_intensa_min = 20.0
    else:  # 24h
        prec_debole_min = 0.3
        prec_debole_max = 10.0
        prec_moderata_min = 10.0
        prec_moderata_max = 30.0
        prec_intensa_min = 30.0
    
    # DETERMINA intensità precipitazione con intervalli precisi
    if tp_rate >= prec_intensa_min:
        prec_intensity = "INTENSA"
    elif tp_rate >= prec_moderata_min:
        prec_intensity = "MODERATA"
    elif tp_rate >= prec_debole_min:
        prec_intensity = "DEBOLE"
    else:
        prec_intensity = None
    
    # PRIORITÀ 2: PIOGGIA DEBOLE+ (SOVRASTA tutto)
    if prec_intensity:
        octas = clct / 100.0 * 8
        if clch > 60 and low_cloud < 30 and octas > 5: cloud_state = "NUBI ALTE"
        elif octas <= 2: cloud_state = "SERENO"
        elif octas <= 4: cloud_state = "POCO NUVOLOSO"
        elif octas <= 6: cloud_state = "NUVOLOSO"
        else: cloud_state = "COPERTO"
        
        prec_type = "NEVE" if is_snow else "PIOGGIA"
        return f"{cloud_state} {prec_type} {prec_intensity}"
    
    # PRIORITÀ 3: PIOGGERELLA (0.09-0.3 mm indipendentemente dal timestep)
    is_drizzle = drizzle_min <= tp_rate <= drizzle_max and low_cloud >= 50
    
    if is_drizzle:
        t_cold = t2m < 12
        
        # NEBBIA ha priorità su PIOGGERELLA
        if (rh2m >= season_thresh["fog_rh"] and wind_kmh <= season_thresh["fog_wind"] and 
            low_cloud >= 80 and t_cold): return "NEBBIA"
        
        # FOSCHIA ha priorità su PIOGGERELLA
        if (rh2m >= season_thresh["haze_rh"] and wind_kmh <= season_thresh["haze_wind"] and 
            low_cloud >= 50 and t_cold): return "FOSCHIA"
        
        # Altrimenti PIOGGERELLA con nuvolosità
        octas = clct / 100.0 * 8
        if clch > 60 and low_cloud < 30 and octas > 5: cloud_state = "NUBI ALTE"
        elif octas <= 2: cloud_state = "SERENO"
        elif octas <= 4: cloud_state = "POCO NUVOLOSO"
        elif octas <= 6: cloud_state = "NUVOLOSO"
        else: cloud_state = "COPERTO"
        
        prec_type = "NEVE DEBOLE" if is_snow else "PIOGGERELLA"
        return f"{cloud_state} {prec_type}"
    
    # PRIORITÀ 4: NEBBIA/FOSCHIA (solo se ZERO precipitazione)
    t_cold = t2m < 12
    if (rh2m >= season_thresh["fog_rh"] and wind_kmh <= season_thresh["fog_wind"] and 
        low_cloud >= 80 and t_cold): return "NEBBIA"
    if (rh2m >= season_thresh["haze_rh"] and wind_kmh <= season_thresh["haze_wind"] and 
        low_cloud >= 50 and t_cold): return "FOSCHIA"
    
    # PRIORITÀ 5: NUVOLOSITÀ PURA (zero precip)
    octas = clct / 100.0 * 8
    if clch > 60 and low_cloud < 30 and octas > 5: return "NUBI ALTE"
    elif octas <= 2: return "SERENO"
    elif octas <= 4: return "POCO NUVOLOSO"
    elif octas <= 6: return "NUVOLOSO"
    else: return "COPERTO"


def extract_variable(var, lat_idx, lon_idx, is_precip=False):
    """
    Estrae valore pesato ROBUSTO per tutti i tipi di array (scalare/1D/2D/3D)
    """
    # CASO 1: Scalare singolo
    if np.isscalar(var) or (hasattr(var, 'size') and var.size == 1):
        return np.array([float(var)])
    
    # CASO 2: Array 1D (solo tempo)
    if var.ndim == 1:
        return var
    
    # CASO 3: Array 2D/3D+ (spazio)
    # Prendi le ultime 2 dimensioni come griglia spaziale
    spatial_slice = var[..., lat_idx, lon_idx]
    
    # Se 2D, restituisci centro
    if var.ndim == 2:
        return spatial_slice
    
    # PRIMO ANELLO: 8 vicini (3x3 - centro)
    first_ring = []
    NY, NX = var.shape[-2:]
    for di in [-1, 0, 1]:
        for dj in [-1, 0, 1]:
            if di == 0 and dj == 0: continue
            ni, nj = lat_idx + di, lon_idx + dj
            if 0 <= ni < NY and 0 <= nj < NX:
                first_ring.append(var[..., ni, nj])
    
    if not first_ring:
        return spatial_slice  # Nessun vicino disponibile
    
    first_ring_mean = np.stack(first_ring, axis=-1).mean(axis=-1)
    
    if not is_precip:
        # TEMPERATURA: 50% centro + 50% primo anello
        return 0.5 * spatial_slice + 0.5 * first_ring_mean
    
    # PRECIP: SECONDO ANELLO (5x5 - 3x3 interno)
    second_ring = []
    for di in [-2, -1, 0, 1, 2]:
        for dj in [-2, -1, 0, 1, 2]:
            if abs(di) <= 1 and abs(dj) <= 1: continue
            ni, nj = lat_idx + di, lon_idx + dj
            if 0 <= ni < NY and 0 <= nj < NX:
                second_ring.append(var[..., ni, nj])
    
    if not second_ring:
        # Solo primo anello disponibile per precip
        return 0.6 * spatial_slice + 0.4 * first_ring_mean
    
    second_ring_mean = np.stack(second_ring, axis=-1).mean(axis=-1)
    # PRECIP: 50% centro + 30% primo + 20% secondo
    return 0.5 * spatial_slice + 0.3 * first_ring_mean + 0.2 * second_ring_mean

def build_latlon_from_hsurf(hsurf_da):
    """Ricostruisce griglia lat/lon da coordinate HSURF o dominio ICON-2I"""
    if 'latitude' in hsurf_da.coords and 'longitude' in hsurf_da.coords:
        lat_grid, lon_grid = hsurf_da['latitude'].values, hsurf_da['longitude'].values
        if lat_grid.ndim == 1: lon_grid, lat_grid = np.meshgrid(lon_grid, lat_grid)
        return lat_grid, lon_grid
    nlat, nlon = hsurf_da.shape[-2:]
    return np.meshgrid(np.linspace(3.0, 22.0, nlon), np.linspace(33.7, 48.89, nlat))

def find_land_nearest(lat_grid, lon_grid, target_lat, target_lon, hsurf_grid):
    """Trova punto più vicino TERRESTRE (HSURF > 0) con raggio crescente"""
    for radius in [0.1, 0.2, 0.5, 1.0]:
        y, x = np.where((np.abs(lat_grid - target_lat) <= radius) & 
                       (np.abs(lon_grid - target_lon) <= radius) & 
                       (hsurf_grid > 0))
        if len(y) > 0:
            dists = np.sqrt((lat_grid[y,x] - target_lat)**2 + (lon_grid[y,x] - target_lon)**2)
            center_idx = np.argmin(dists)
            center_y, center_x = y[center_idx], x[center_idx]
            return center_y, center_x
    
    dist = np.sqrt((lat_grid - target_lat)**2 + (lon_grid - target_lon)**2)
    center_y, center_x = np.unravel_index(np.argmin(dist), dist.shape)
    return center_y, center_x

def load_venues(file_path):
    """Carica coordinate/elevazione città da venues.json"""
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File {file_path} non trovato")
    
    with open(file_path, 'r', encoding='utf-8') as f: 
        data = json.load(f)
    
    venues = {}
    if isinstance(data, dict):
        for city_name, coords in data.items():
            if isinstance(coords, list) and len(coords) >= 3:
                venues[city_name] = {
                    "lat": float(coords[0]),
                    "lon": float(coords[1]),  
                    "elev": float(coords[2])
                }
    
    print(f"Caricate {len(venues)} città da {file_path}")
    return venues

def calculate_daily_summaries(hourly_data, clct_daily, clcl_daily, clcm_daily, clch_daily, tp_h_daily, lpi_daily, cape_daily, uh_daily, season, season_thresh):
    """Calcola riepiloghi GIORNALIERI: Tmin/Tmax + SOMMA PREC + nuvolosità + pioggia/neve"""
    daily_summaries = []
    days_data = {}
    
    for i, record in enumerate(hourly_data):
        day = record["d"]
        if day not in days_data:
            days_data[day] = []
        days_data[day].append((i, record))
    
    for day, day_records in days_data.items():
        day_indices = [idx for idx, _ in day_records]
        day_temps = np.array([record["t"] for _, record in day_records])
        
        t_min = round(np.min(day_temps), 1)
        t_max = round(np.max(day_temps), 1)
        tp_total_day = sum([record["p"] for _, record in day_records])  # mm/24h
        
        # Medie per classificazione
        t_mean = np.mean(day_temps)
        r_mean = np.mean([record["r"] for _, record in day_records])
        v_mean = np.mean([record["v"] for _, record in day_records])
        
        first_idx = day_indices[0]
        
        # STESSA LOGICA classify_weather() con parametri GIORNALIERI (timestep=24h)
        weather_day = classify_weather(t_mean, r_mean, clct_daily[first_idx], clcl_daily[first_idx], 
                                     clcm_daily[first_idx], clch_daily[first_idx], tp_total_day, 
                                     v_mean, lpi_daily[first_idx], cape_daily[first_idx], 
                                     uh_daily[first_idx], season, season_thresh, timestep_hours=24)
        
        daily_summaries.append({
            "d": day,
            "tmin": t_min,
            "tmax": t_max,
            "p": round(tp_total_day, 1),
            "w": weather_day
        })
    
    return daily_summaries

def get_drive_service():
    """Crea client Drive v3 usando il service account passato via env JSON."""
    if not GDRIVE_SERVICE_ACCOUNT_JSON or not DRIVE_FOLDER_ID_ICON2I:
        print("Drive API disabilitata (mancano variabili d'ambiente)")
        return None

    try:
        info = json.loads(GDRIVE_SERVICE_ACCOUNT_JSON)
        creds = service_account.Credentials.from_service_account_info(
            info,
            scopes=["https://www.googleapis.com/auth/drive.file"],
        )
        service = build("drive", "v3", credentials=creds)
        print("Drive API inizializzata correttamente")
        return service
    except Exception as e:
        print(f"Errore inizializzazione Drive API: {e}")
        return None

def upload_to_drive(service, local_path, city, run):
    """Carica un file su Drive/ICON-2I/<RUN>_<city>.json"""
    if service is None or not os.path.exists(local_path):
        return

    try:
        file_name = os.path.basename(local_path)
        drive_name = f"{run}_{file_name}"
        
        file_metadata = {
            "name": drive_name,
            "parents": [DRIVE_FOLDER_ID_ICON2I],
        }
        media = MediaFileUpload(local_path, mimetype="application/json", resumable=True)
        created = service.files().create(
            body=file_metadata,
            media_body=media,
            fields="id,name",
        ).execute()
        print(f"✅ Caricato su Drive: {drive_name} (id={created.get('id')})")
    except Exception as e:
        print(f"❌ Upload Drive fallito per {local_path}: {e}")


def process_data():
    """PIPELINE COMPLETA: GRIB → JSON ULTRA-COMPATTO per città in WORKDIR/yyyymmddHH"""
    RUN = os.getenv("RUN", "")
    if not RUN:
        run_date, run_hour = get_run_datetime_now_utc()
        RUN = run_date + run_hour
        if not download_icon_data(run_date, run_hour):
            raise RuntimeError("Download GRIB incompleto")
        with open("build.env", "w") as f: f.write(f"RUN={RUN}\n")
    else:
        run_date, run_hour = RUN[:8], RUN[8:]
        print(f"Usando RUN esistente: {RUN}")
    
    output_dir = f"{WORKDIR}/{RUN}"
    os.makedirs(output_dir, exist_ok=True)

    drive_service = get_drive_service()

    
    data = {}
    for var in VARIABLES:
        grib_path = f"{WORKDIR}/grib_data/{RUN}/{var}.grib"
        if os.path.exists(grib_path):
            data[var] = xr.open_dataset(grib_path, engine='cfgrib', backend_kwargs={"indexpath": ""})
        else:
            print(f"Manca {var}.grib")
    
    if 'HSURF' not in data:
        raise RuntimeError("HSURF.grib mancante")
    
    hsurf_da = list(data['HSURF'].data_vars.values())[0]
    lat_grid, lon_grid = build_latlon_from_hsurf(hsurf_da)
    
    # CORREZIONE HSURF ROBUSTA
    if hsurf_da.ndim == 3:
        hsurf_grid = hsurf_da.isel(time=0).values
    elif hsurf_da.ndim == 2:
        hsurf_grid = hsurf_da.values
    else:
        hsurf_grid = hsurf_da.values
        if hsurf_grid.size == 1:
            hsurf_grid = np.full((lat_grid.shape), hsurf_grid)
    
    venues = load_venues(VENUES_PATH)
    reference_dt = datetime.strptime(RUN, "%Y%m%d%H").replace(tzinfo=timezone.utc)
    season, season_thresh = get_season_precise(reference_dt)
    
    processed = 0
    
    for city, info in venues.items():
        try:
            center_y, center_x = find_land_nearest(lat_grid, lon_grid, info['lat'], info['lon'], hsurf_grid)
            
            hs_model = hsurf_grid[center_y, center_x]
            t2m_raw = kelvin_to_celsius(data['T_2M']['t2m'].values)
            rh_raw = data['RELHUM']['r'].values
            tp_cum = data['TOT_PREC']['tp'].values
            tp_rate = np.diff(tp_cum, axis=0, prepend=0)
            clct_raw = data['CLCT']['clct'].values
            clcl_raw = data['CLCL']['ccl'].values if 'CLCL' in data else np.full_like(clct_raw, np.nan)
            clcm_raw = data['CLCM']['ccl'].values if 'CLCM' in data else np.full_like(clct_raw, np.nan)
            clch_raw = data['CLCH']['ccl'].values if 'CLCH' in data else np.full_like(clct_raw, np.nan)
            u10_raw = data['U_10M']['u10'].values
            v10_raw = data['V_10M']['v10'].values
            vmax10_raw = data['VMAX_10M']['fg10'].values
            lpi_raw = data['LPI']['unknown'].values
            cape_raw = np.maximum(data['CAPE_ML']['cape_ml'].values, data['CAPE_CON']['cape_con'].values)
            uh_raw = data['UH_MAX']['unknown'].values
            pmsl_raw = data['PMSL']['pmsl'].values / 100.0
            asob_s_raw = data['ASOB_S']['msnswrf'].values if 'ASOB_S' in data else np.zeros_like(t2m_raw)
            
            t2m = extract_variable(t2m_raw, center_y, center_x, False)
            rh2m = np.clip(extract_variable(rh_raw, center_y, center_x, False), 0, 100)
            tp_h = extract_variable(tp_rate, center_y, center_x, True)
            clct = extract_variable(clct_raw, center_y, center_x, False)
            clcl = extract_variable(clcl_raw, center_y, center_x, False)
            clcm = extract_variable(clcm_raw, center_y, center_x, False)
            clch = extract_variable(clch_raw, center_y, center_x, False)
            pmsl_point = extract_variable(pmsl_raw, center_y, center_x, False)
            
            t2m_alt, pmsl_corr = altitude_correction(t2m, rh2m, hs_model, info['elev'], pmsl_point)
            t2m_final = t2m_alt
            
            u10 = extract_variable(u10_raw, center_y, center_x, False)
            v10 = extract_variable(v10_raw, center_y, center_x, False)
            spd10_ms, wd_deg = wind_speed_direction(u10, v10)
            spd10_kmh = mps_to_kmh(spd10_ms)
            wd_card = np.vectorize(wind_dir_to_cardinal)(wd_deg)
            vmax10_kmh = mps_to_kmh(extract_variable(vmax10_raw, center_y, center_x, False))
            lpi = extract_variable(lpi_raw, center_y, center_x, False)
            cape = extract_variable(cape_raw, center_y, center_x, False)
            uh = extract_variable(uh_raw, center_y, center_x, False)
            asob_s = extract_variable(asob_s_raw, center_y, center_x, False)
            
            hourly_data = []
            for i in range(len(t2m_final)):
                dt_utc = reference_dt + timedelta(hours=i)
                ghi_h = calculate_solar_radiation(asob_s[i], clct[i], dt_utc, info['lat'], info['lon'])
                weather_type = classify_weather(t2m_final[i], rh2m[i], clct[i], clcl[i], clcm[i], clch[i],
                                              tp_h[i], spd10_kmh[i], lpi[i], cape[i], uh[i], season, season_thresh, timestep_hours=1)
                dt_local = utc_to_local(dt_utc)
                
                hourly_data.append({
                    "d": dt_local.strftime("%Y%m%d"),
                    "h": dt_local.strftime("%H"),
                    "t": round(float(t2m_final[i]), 1),
                    "r": round(float(rh2m[i])),
                    "p": round(float(tp_h[i]), 1),
                    "pr": round(float(pmsl_corr[i])),
                    "v": round(float(spd10_kmh[i]), 1),
                    "vd": wd_card[i],
                    "vg": round(float(vmax10_kmh[i]), 1),
                    # "s": int(ghi_h),
                    "w": weather_type
                })
            
            # TRIORARIO
            data_triorario = []
            if len(hourly_data) >= 3:
                n_blocks = len(hourly_data) // 3
                for bs in range(0, n_blocks * 3, 3):
                    block = hourly_data[bs:bs+3]
                    block_tp_sum = sum([x["p"] for x in block])
                    block_t_mean = np.mean([x["t"] for x in block])
                    block_r_mean = np.mean([x["r"] for x in block])
                    block_v_mean = np.mean([x["v"] for x in block])
                    block_pr_mean = np.mean([x["pr"] for x in block])
                    
                    weather_trio = classify_weather(block_t_mean, block_r_mean, clct[bs], clcl[bs], clcm[bs], clch[bs],
                                                  block_tp_sum, block_v_mean, lpi[bs], cape[bs], uh[bs], 
                                                  season, season_thresh, timestep_hours=3)
                    
                    data_triorario.append({
                        "d": block[0]["d"],
                        "h": block[0]["h"],
                        "t": round(block_t_mean, 1),
                        "r": round(block_r_mean),
                        "p": round(block_tp_sum, 1),
                        "pr": round(block_pr_mean),
                        "v": round(block_v_mean, 1),
                        "vd": Counter([x["vd"] for x in block]).most_common(1)[0][0],
                        "vg": round(max([x["vg"] for x in block]), 1),
                        # "s": int(np.mean([x["s"] for x in block])),
                        "w": weather_trio
                    })
            
            # GIORNALIERO
            daily_summaries = calculate_daily_summaries(hourly_data, clct, clcl, clcm, clch, tp_h, lpi, cape, uh, season, season_thresh)
            
            city_data = {
                "r": RUN,
                "c": city,
                "x": info['lat'],
                "y": info['lon'],
                "z": info['elev'],
                "ORARIO": hourly_data,
                "TRIORARIO": data_triorario,
                "GIORNALIERO": daily_summaries
            }
            
            city_json_path = f"{output_dir}/{city}.json"
            with open(city_json_path, 'w', encoding='utf-8') as f: 
                json.dump(city_data, f, separators=(',',':'), ensure_ascii=False)
            # Upload su Drive subito dopo la scrittura
            drive_service = get_drive_service()
            upload_to_drive(drive_service, city_json_path, city, RUN)

            
            processed += 1
            if processed % 100 == 0:
                print(f"Processate {processed}/{len(venues)} città...")
                
        except Exception as e:
            print(f"Errore {city}: {e}")
            continue
    
    print(f"Salvati {processed}/{len(venues)} JSON ULTRA-COMPATTI in {output_dir}/")
    return output_dir

if __name__ == "__main__": 
    process_data()

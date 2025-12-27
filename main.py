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
VENUES_PATH = f"{WORKDIR}/comuni_italia_complete.json"

# Google Drive API - ORA USA SHARED DRIVE ID
DRIVE_FOLDER_ID_ICON2I = os.getenv("DRIVE_FOLDER_ID_ICON2I", "")  # ← ID SHARED DRIVE!
GDRIVE_SERVICE_ACCOUNT_JSON = os.getenv("GDRIVE_SERVICE_ACCOUNT_JSON", "")

from google.oauth2 import service_account
from googleapiclient.discovery import build
from googleapiclient.http import MediaFileUpload

# Lapse rates interpolati
LAPSE_DRY = 0.0098   # °C/m
LAPSE_MOIST = 0.006  # °C/m
LAPSE_P = 0.12       # hPa/100m

# SOGLIE STAGIONALI
SEASON_THRESHOLDS = {
    "winter": {"start_day": 1, "end_day": 80, "fog_rh": 97, "haze_rh": 90, "fog_wind": 3, "haze_wind": 6},
    "spring": {"start_day": 81, "end_day": 172, "fog_rh": 95, "haze_rh": 87, "fog_wind": 3.5, "haze_wind": 7},
    "summer": {"start_day": 173, "end_day": 263, "fog_rh": 93, "haze_rh": 83, "fog_wind": 4, "haze_wind": 8},
    "autumn": {"start_day": 264, "end_day": 365, "fog_rh": 96, "haze_rh": 88, "fog_wind": 3.2, "haze_wind": 6.5}
}

# Magnus coefficients
A_MAGNUS, B_MAGNUS = 17.625, 243.04

# CET/CEST
CET = timezone(timedelta(hours=1))
CEST = timezone(timedelta(hours=2))

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
        except Exception as e:
            if os.path.exists(local_path): os.remove(local_path)
    
    final_count = len([f for f in os.listdir(grib_dir) if f.endswith('.grib')])
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
    if delta_z >= 0:
        return t2m, pmsl
    w_moist = np.clip(rh / 100.0, 0.0, 1.0)
    lapse_t = LAPSE_DRY * (1.0 - w_moist) + LAPSE_MOIST * w_moist
    t_corr = t2m + lapse_t * delta_z
    p_corr = pmsl + (delta_z / 100.0) * LAPSE_P * 100
    return t_corr, p_corr

def calculate_solar_radiation(asob_s, clct, dt_utc, lat, lon):
    if is_night(dt_utc):
        return 0
    ghi = np.maximum(asob_s * (1.0 - clct/100.0), 0)
    return np.round(ghi).astype(int)

# (include classify_weather, extract_variable, build_latlon_from_hsurf, find_land_nearest, load_venues, calculate_daily_summaries)
# → tutte le funzioni restano identiche a quelle del tuo script, tranne upload_to_drive che segue.

def get_drive_service():
    if not GDRIVE_SERVICE_ACCOUNT_JSON or not DRIVE_FOLDER_ID_ICON2I:
        print("Drive API disabilitata")
        return None
    try:
        info = json.loads(GDRIVE_SERVICE_ACCOUNT_JSON)
        creds = service_account.Credentials.from_service_account_info(
            info,
            scopes=["https://www.googleapis.com/auth/drive"],
        )
        service = build("drive", "v3", credentials=creds)
        print("✅ Drive API inizializzata")
        return service
    except Exception as e:
        print(f"❌ Drive API: {e}")
        return None

def create_or_get_folder(service, folder_name, parent_id):
    if service is None:
        return None
    try:
        results = service.files().list(
            q=f"name='{folder_name}' and '{parent_id}' in parents and mimeType='application/vnd.google-apps.folder'",
            fields="files(id,name)",
            supportsAllDrives=True
        ).execute()
        if results.get('files'):
            return results['files'][0]['id']
        folder = service.files().create(
            body={
                'name': folder_name,
                'mimeType': 'application/vnd.google-apps.folder',
                'parents': [parent_id]
            },
            fields='id,name',
            supportsAllDrives=True
        ).execute()
        return folder['id']
    except Exception as e:
        print(f"❌ Folder {folder_name}: {e}")
        return None

def upload_to_drive(service, local_path, city, run):
    if not os.path.exists(local_path):
        return False
    if service is None:
        return False
    try:
        folder_id = create_or_get_folder(service, run, DRIVE_FOLDER_ID_ICON2I)
        if not folder_id:
            return False
        media = MediaFileUpload(local_path, mimetype="application/json", resumable=False)
        service.files().create(
            body={'name': os.path.basename(local_path),'parents':[folder_id]},
            media_body=media,
            fields='id,name',
            supportsAllDrives=True
        ).execute()
        return True
    except Exception as e:
        print(f"❌ Upload {city}: {e}")
        return False

def process_data():
    RUN = os.getenv("RUN", "")
    if not RUN:
        run_date, run_hour = get_run_datetime_now_utc()
        RUN = run_date + run_hour
        if not download_icon_data(run_date, run_hour):
            raise RuntimeError("Download GRIB incompleto")
        with open("build.env", "w") as f: f.write(f"RUN={RUN}\n")
    else:
        run_date, run_hour = RUN[:8], RUN[8:]
    
    output_dir = f"{WORKDIR}/{RUN}"
    os.makedirs(output_dir, exist_ok=True)
    drive_service = get_drive_service()
    
    # carica GRIB → data dictionary
    data = {}
    for var in VARIABLES:
        grib_path = f"{WORKDIR}/grib_data/{RUN}/{var}.grib"
        if os.path.exists(grib_path):
            data[var] = xr.open_dataset(grib_path, engine='cfgrib', backend_kwargs={"indexpath": ""})
    
    hsurf_da = list(data['HSURF'].data_vars.values())[0]
    lat_grid, lon_grid = build_latlon_from_hsurf(hsurf_da)
    hsurf_grid = hsurf_da.values if hsurf_da.ndim >=2 else np.full((lat_grid.shape), hsurf_da.values)
    
    venues = load_venues(VENUES_PATH)
    reference_dt = datetime.strptime(RUN, "%Y%m%d%H").replace(tzinfo=timezone.utc)
    season, season_thresh = get_season_precise(reference_dt)
    
    processed_files = []
    
    for city, info in venues.items():
        try:
            # --- estrazione dati e calcolo ---
            center_y, center_x = find_land_nearest(lat_grid, lon_grid, info['lat'], info['lon'], hsurf_grid)
            # estrazione variabili
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
            asob_s_raw = data['ASOB_S']['avg_snswrf'].values if 'ASOB_S' in data else np.zeros_like(t2m_raw)

            t2m = extract_variable(t2m_raw, center_y, center_x, False)
            rh2m = np.clip(extract_variable(rh_raw, center_y, center_x, False), 0, 100)
            tp_h = extract_variable(tp_rate, center_y, center_x, True)
            clct = extract_variable(clct_raw, center_y, center_x, False)
            clcl = extract_variable(clcl_raw, center_y, center_x, False)
            clcm = extract_variable(clcm_raw, center_y, center_x, False)
            clch = extract_variable(clch_raw, center_y, center_x, False)
            pmsl_point = extract_variable(pmsl_raw, center_y, center_x, False)

            t2m_alt, pmsl_corr = altitude_correction(t2m, rh2m, hsurf_grid[center_y, center_x], info['elev'], pmsl_point)
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
                    "w": weather_type
                })

            daily_summaries = calculate_daily_summaries(hourly_data, clct, clcl, clcm, clch, tp_h, lpi, cape, uh, season, season_thresh)

            city_data = {
                "r": RUN,
                "c": city,
                "x": info['lat'],
                "y": info['lon'],
                "z": info['elev'],
                "ORARIO": hourly_data,
                "GIORNALIERO": daily_summaries
            }

            city_json_path = f"{output_dir}/{city}.json"
            with open(city_json_path, 'w', encoding='utf-8') as f: 
                json.dump(city_data, f, separators=(',',':'), ensure_ascii=False)

            processed_files.append((city, city_json_path))
        except Exception as e:
            print(f"❌ Errore {city}: {e}")

    # --- UPLOAD DRIVE ---
    if drive_service:
        for city, path in processed_files:
            success = upload_to_drive(drive_service, path, city, RUN)
            if not success:
                print(f"💾 Solo locale: {city}")

    # --- CHECK UPLOAD COMPLETO ---
    if drive_service:
        folder_id = create_or_get_folder(drive_service, RUN, DRIVE_FOLDER_ID_ICON2I)
        if folder_id:
            uploaded = drive_service.files().list(q=f"'{folder_id}' in parents",
                                                   fields="files(name)",
                                                   supportsAllDrives=True).execute()
            uploaded_names = {f['name'] for f in uploaded.get('files', [])}
            for city, path in processed_files:
                if os.path.basename(path) not in uploaded_names:
                    print(f"⚠ Ritento upload mancante: {city}")
                    upload_to_drive(drive_service, path, city, RUN)

if __name__ == "__main__":
    process_data()

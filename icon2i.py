#!/bin/env python3
import os
import requests
import json
import numpy as np
import xarray as xr
from datetime import datetime, timedelta, timezone
from bs4 import BeautifulSoup
from collections import Counter

# ------------------- CONFIG -------------------
WORKDIR = os.getcwd()
# RIMOSSO 'ASOB_S' dalla lista
VARIABLES = ['T_2M', 'RELHUM', 'TOT_PREC', 'CLCT', 'CLCL', 'CLCM', 'CLCH', 'U_10M', 'V_10M', 'VMAX_10M', 'LPI', 'CAPE_ML', 'CAPE_CON', 'UH_MAX', 'PMSL', 'HSURF']
VENUES_PATH = f"{WORKDIR}/comuni_italia.json"

# Lapse rates interpolati
LAPSE_DRY = 0.0098   # °C/m (9.8 °C/km)
LAPSE_MOIST = 0.006  # °C/m (~6 °C/km)
LAPSE_P = 0.12       # hPa/100m

# SOGLIE STAGIONALI (per Nebbia/Foschia)
SEASON_THRESHOLDS = {
    "winter": {"start_day": 1, "end_day": 80, "fog_rh": 97, "haze_rh": 90, "fog_wind": 3, "haze_wind": 6},
    "spring": {"start_day": 81, "end_day": 172, "fog_rh": 95, "haze_rh": 87, "fog_wind": 3.5, "haze_wind": 7},
    "summer": {"start_day": 173, "end_day": 263, "fog_rh": 93, "haze_rh": 83, "fog_wind": 4, "haze_wind": 8},
    "autumn": {"start_day": 264, "end_day": 365, "fog_rh": 96, "haze_rh": 88, "fog_wind": 3.2, "haze_wind": 6.5}
}

# TIMEZONES
CET = timezone(timedelta(hours=1))
CEST = timezone(timedelta(hours=2))

# ------------------- FUNZIONI METEO -------------------
def utc_to_local(dt_utc):
    """Converte UTC -> Locale (gestisce cambio ora solare/legale approssimato)"""
    now_utc = datetime.now(timezone.utc)
    if 31 <= now_utc.month <= 10 or (now_utc.month == 3 and now_utc.day >= 25) or (now_utc.month == 10 and now_utc.day <= 25):
        return dt_utc.astimezone(CEST)
    return dt_utc.astimezone(CET)

def wet_bulb_celsius(t_c, rh_percent):
    """Calcolo wet-bulb approssimato (Stull)"""
    tw = t_c * np.arctan(0.151977 * np.sqrt(rh_percent + 8.313659)) \
         + np.arctan(t_c + rh_percent) - np.arctan(rh_percent - 1.676331) \
         + 0.00391838 * rh_percent**1.5 * np.arctan(0.023101 * rh_percent) \
         - 4.686035
    return tw

def get_run_datetime_now_utc():
    """Determina run ICON 00 o 12"""
    now = datetime.now(timezone.utc)
    if now.hour < 3: return (now - timedelta(days=1)).strftime("%Y%m%d"), "12"
    elif now.hour < 14: return now.strftime("%Y%m%d"), "00"
    return now.strftime("%Y%m%d"), "12"

def download_icon_data(run_date, run_hour):
    """Scarica i file GRIB necessari"""
    grib_dir = f"{WORKDIR}/grib_data/{run_date}{run_hour}"
    os.makedirs(grib_dir, exist_ok=True)
    
    base_url = f'https://meteohub.agenziaitaliameteo.it/nwp/ICON-2I_SURFACE_PRESSURE_LEVELS/{run_date}{run_hour}/'
    
    # Check disponibilità run
    try:
        if requests.head(base_url, timeout=10).status_code != 200:
            raise RuntimeError(f"Run {run_date}{run_hour} non disponibile")
    except:
        pass
    
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
                with requests.get(file_url, stream=True, timeout=120) as resp:
                    resp.raise_for_status()
                    with open(local_path, 'wb') as f: 
                        for chunk in resp.iter_content(chunk_size=8192):
                            f.write(chunk)
                downloaded += 1
                print(f"Scaricato {var}")
        except Exception as e:
            print(f"Errore download {var}: {e}")
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
    """Corregge T e P in base alla differenza di quota (senza toccare RH)"""
    delta_z = hs_model - hs_station
    if delta_z >= 0:
        return t2m, pmsl
    
    w_moist = np.clip(rh / 100.0, 0.0, 1.0)
    lapse_t = LAPSE_DRY * (1.0 - w_moist) + LAPSE_MOIST * w_moist
    t_corr = t2m + lapse_t * delta_z
    p_corr = pmsl + (delta_z / 100.0) * LAPSE_P * 100
    return t_corr, p_corr

# Funzione calculate_solar_radiation RIMOSSA

# ------------------- CLASSIFICAZIONE TEMPO -------------------
def classify_weather(t2m, rh2m, clct, clcl, clcm, clch, tp_rate, wind_kmh, lpi, cape, uh, season, season_thresh, timestep_hours=1):
    
    # 1. TEMPORALE (Priorità Massima)
    if (lpi > 2.0 or uh > 50 or cape > 400) and tp_rate > (0.5 * timestep_hours): 
        return "TEMPORALE"
    
    # Calcolo tipo precipitazione
    wet_bulb = wet_bulb_celsius(t2m, rh2m)
    is_snow = wet_bulb < 0.5 
    prec_type_base = "NEVE" if is_snow else "PIOGGIA"
    prec_type_weak = "NEVICATA DEBOLE" if is_snow else "PIOGGERELLA"
    
    # Parametri nubi
    low_cloud = clcl if np.isfinite(clcl) else (clcm if np.isfinite(clcm) else 0)
    octas = clct / 100.0 * 8
    
    # Definizione stato cielo
    if clch > 60 and low_cloud < 30 and octas > 5: cloud_state = "NUBI ALTE"
    elif octas <= 2: cloud_state = "SERENO"
    elif octas <= 4: cloud_state = "POCO NUVOLOSO"
    elif octas <= 6: cloud_state = "NUVOLOSO"
    else: cloud_state = "COPERTO"

    # Soglie intensità (scalate per timestep)
    if timestep_hours == 1:
        soglia_debole = 0.2
        soglia_mod = 2.0
        soglia_forte = 7.0
    elif timestep_hours == 3:
        soglia_debole = 0.6
        soglia_mod = 5.0
        soglia_forte = 20.0
    else: # 24h
        soglia_debole = 1.0
        soglia_mod = 10.0
        soglia_forte = 30.0

    # 2. LOGICA PRECIPITAZIONE
    if tp_rate > 0:
        is_very_weak = tp_rate <= soglia_debole
        
        if is_very_weak:
            # Controllo se è Nebbia/Foschia (priorità su pioggerella)
            t_cold = t2m < 12
            is_fog = (rh2m >= season_thresh["fog_rh"] and wind_kmh <= season_thresh["fog_wind"] and low_cloud >= 80 and t_cold)
            is_haze = (rh2m >= season_thresh["haze_rh"] and wind_kmh <= season_thresh["haze_wind"] and low_cloud >= 50 and t_cold)
            
            if is_fog: return "NEBBIA"
            if is_haze: return "FOSCHIA"
            
            if cloud_state == "SERENO": cloud_state = "POCO NUVOLOSO"
            return f"{cloud_state} {prec_type_weak}"
        else:
            if tp_rate >= soglia_forte: intensity = "INTENSA"
            elif tp_rate >= soglia_mod: intensity = "MODERATA"
            else: intensity = "DEBOLE"
            
            if cloud_state == "SERENO": cloud_state = "POCO NUVOLOSO"
            return f"{cloud_state} {prec_type_base} {intensity}"

    # 3. NESSUNA PRECIPITAZIONE (0 mm)
    t_cold = t2m < 12
    if (rh2m >= season_thresh["fog_rh"] and wind_kmh <= season_thresh["fog_wind"] and low_cloud >= 80 and t_cold): return "NEBBIA"
    if (rh2m >= season_thresh["haze_rh"] and wind_kmh <= season_thresh["haze_wind"] and low_cloud >= 50 and t_cold): return "FOSCHIA"
    
    return cloud_state

# ------------------- ESTRAZIONE DATI -------------------
def extract_variable(var, lat_idx, lon_idx, use_weighted_avg=False):
    if np.isscalar(var) or (hasattr(var, 'size') and var.size == 1):
        return np.array([float(var)])
    if var.ndim == 1:
        return var
    
    spatial_slice = var[..., lat_idx, lon_idx]
    if var.ndim == 2: return spatial_slice
    if not use_weighted_avg: return spatial_slice

    NY, NX = var.shape[-2:]
    first_ring = []
    for di in [-1, 0, 1]:
        for dj in [-1, 0, 1]:
            if di == 0 and dj == 0: continue
            ni, nj = lat_idx + di, lon_idx + dj
            if 0 <= ni < NY and 0 <= nj < NX:
                first_ring.append(var[..., ni, nj])
    
    if not first_ring: return spatial_slice
    first_ring_mean = np.stack(first_ring, axis=-1).mean(axis=-1)
    
    second_ring = []
    for di in [-2, -1, 0, 1, 2]:
        for dj in [-2, -1, 0, 1, 2]:
            if abs(di) <= 1 and abs(dj) <= 1: continue
            ni, nj = lat_idx + di, lon_idx + dj
            if 0 <= ni < NY and 0 <= nj < NX:
                second_ring.append(var[..., ni, nj])
                
    if not second_ring:
        return 0.6 * spatial_slice + 0.4 * first_ring_mean
        
    second_ring_mean = np.stack(second_ring, axis=-1).mean(axis=-1)
    return 0.5 * spatial_slice + 0.3 * first_ring_mean + 0.2 * second_ring_mean

def build_latlon_from_hsurf(hsurf_da):
    if 'latitude' in hsurf_da.coords and 'longitude' in hsurf_da.coords:
        lat_grid, lon_grid = hsurf_da['latitude'].values, hsurf_da['longitude'].values
        if lat_grid.ndim == 1: lon_grid, lat_grid = np.meshgrid(lon_grid, lat_grid)
        return lat_grid, lon_grid
    nlat, nlon = hsurf_da.shape[-2:]
    return np.meshgrid(np.linspace(3.0, 22.0, nlon), np.linspace(33.7, 48.89, nlat))

def find_land_nearest(lat_grid, lon_grid, target_lat, target_lon, hsurf_grid):
    for radius in [0.1, 0.2, 0.5]:
        mask = (np.abs(lat_grid - target_lat) <= radius) & \
               (np.abs(lon_grid - target_lon) <= radius) & \
               (hsurf_grid > 0)
        y, x = np.where(mask)
        if len(y) > 0:
            dists = (lat_grid[y,x] - target_lat)**2 + (lon_grid[y,x] - target_lon)**2
            idx = np.argmin(dists)
            return y[idx], x[idx]
    dist = (lat_grid - target_lat)**2 + (lon_grid - target_lon)**2
    return np.unravel_index(np.argmin(dist), dist.shape)

def load_venues(file_path):
    if not os.path.exists(file_path): raise FileNotFoundError(f"{file_path} mancante")
    with open(file_path, 'r', encoding='utf-8') as f: data = json.load(f)
    venues = {}
    if isinstance(data, dict):
        for city, coords in data.items():
            if isinstance(coords, list) and len(coords) >= 3:
                venues[city] = {"lat": float(coords[0]), "lon": float(coords[1]), "elev": float(coords[2])}
    return venues

def calculate_daily_summaries(hourly_data, clct, clcl, clcm, clch, lpi, cape, uh, season, season_thresh):
    days_map = {}
    for i, rec in enumerate(hourly_data):
        d = rec["d"]
        if d not in days_map: days_map[d] = []
        days_map[d].append((i, rec))
        
    ordered_days = sorted(days_map.keys())
    complete_days = ordered_days[:-1] if len(ordered_days) > 1 else ordered_days
    
    summaries = []
    for day in complete_days:
        items = days_map[day]
        indices = [x[0] for x in items]
        recs = [x[1] for x in items]
        
        t_vals = [r["t"] for r in recs]
        t_min, t_max = min(t_vals), max(t_vals)
        t_mean = np.mean(t_vals)
        r_mean = np.mean([r["r"] for r in recs])
        v_mean = np.mean([r["v"] for r in recs])
        tp_sum = sum([r["p"] for r in recs]) 
        
        clct_mean = np.mean(clct[indices])
        clcl_mean = np.mean(clcl[indices])
        clcm_mean = np.mean(clcm[indices])
        clch_mean = np.mean(clch[indices])
        
        lpi_max = np.max(lpi[indices])
        cape_max = np.max(cape[indices])
        uh_max = np.max(uh[indices])

        weather = classify_weather(
            t_mean, r_mean, clct_mean, clcl_mean, clcm_mean, clch_mean,
            tp_sum, v_mean, lpi_max, cape_max, uh_max,
            season, season_thresh, timestep_hours=24
        )
        
        summaries.append({
            "d": day,
            "tmin": round(t_min, 1),
            "tmax": round(t_max, 1),
            "p": round(tp_sum, 1),
            "w": weather
        })
    return summaries

# ------------------- PROCESS -------------------
def process_data():
    RUN = os.getenv("RUN", "")
    if not RUN:
        run_date, run_hour = get_run_datetime_now_utc()
        RUN = run_date + run_hour
        download_icon_data(run_date, run_hour)
    
    print(f"Elaborazione RUN: {RUN}")
    output_dir = f"{WORKDIR}/{RUN}"
    os.makedirs(output_dir, exist_ok=True)
    
    data = {}
    for var in VARIABLES:
        p = f"{WORKDIR}/grib_data/{RUN}/{var}.grib"
        if os.path.exists(p):
            data[var] = xr.open_dataset(p, engine='cfgrib', backend_kwargs={"indexpath": ""})
        else:
            print(f"Warning: {var} mancante")
            
    hsurf_da = list(data['HSURF'].data_vars.values())[0]
    lat_grid, lon_grid = build_latlon_from_hsurf(hsurf_da)
    hsurf_grid = hsurf_da.values if hsurf_da.ndim == 2 else hsurf_da.isel(time=0).values
    
    venues = load_venues(VENUES_PATH)
    ref_dt = datetime.strptime(RUN, "%Y%m%d%H").replace(tzinfo=timezone.utc)
    season, season_thresh = get_season_precise(ref_dt)
    
    cnt = 0
    for city, info in venues.items():
        try:
            cy, cx = find_land_nearest(lat_grid, lon_grid, info['lat'], info['lon'], hsurf_grid)
            
            # RAW
            t2m_raw = kelvin_to_celsius(data['T_2M']['t2m'].values)
            rh_raw = data['RELHUM']['r'].values
            u10_raw = data['U_10M']['u10'].values
            v10_raw = data['V_10M']['v10'].values
            vmax_raw = data['VMAX_10M']['fg10'].values
            pmsl_raw = data['PMSL']['pmsl'].values / 100.0
            lpi_raw = data['LPI']['unknown'].values
            cape_raw = np.maximum(data['CAPE_ML']['cape_ml'].values, data['CAPE_CON']['cape_con'].values)
            uh_raw = data['UH_MAX']['unknown'].values
            # ASOB RAW rimosso

            # WEIGHTED
            tp_cum = data['TOT_PREC']['tp'].values
            tp_rate_raw = np.diff(tp_cum, axis=0, prepend=0)
            clct_raw = data['CLCT']['clct'].values
            clcl_raw = data['CLCL']['ccl'].values if 'CLCL' in data else np.zeros_like(clct_raw)
            clcm_raw = data['CLCM']['ccl'].values if 'CLCM' in data else np.zeros_like(clct_raw)
            clch_raw = data['CLCH']['ccl'].values if 'CLCH' in data else np.zeros_like(clct_raw)

            # EXTRACT POINT (Nearest)
            t2m = extract_variable(t2m_raw, cy, cx, False)
            rh2m = np.clip(extract_variable(rh_raw, cy, cx, False), 0, 100)
            u10 = extract_variable(u10_raw, cy, cx, False)
            v10 = extract_variable(v10_raw, cy, cx, False)
            vmax = mps_to_kmh(extract_variable(vmax_raw, cy, cx, False))
            pmsl_pt = extract_variable(pmsl_raw, cy, cx, False)
            lpi = extract_variable(lpi_raw, cy, cx, False)
            cape = extract_variable(cape_raw, cy, cx, False)
            uh = extract_variable(uh_raw, cy, cx, False)
            # asob rimosso
            
            # EXTRACT AREA (Weighted)
            tp_h = extract_variable(tp_rate_raw, cy, cx, True)
            clct = extract_variable(clct_raw, cy, cx, True)
            clcl = extract_variable(clcl_raw, cy, cx, True)
            clcm = extract_variable(clcm_raw, cy, cx, True)
            clch = extract_variable(clch_raw, cy, cx, True)

            t2m_corr, pmsl_corr = altitude_correction(t2m, rh2m, hsurf_grid[cy, cx], info['elev'], pmsl_pt)
            spd_ms, dir_deg = wind_speed_direction(u10, v10)
            spd_kmh = mps_to_kmh(spd_ms)
            card_dir = np.vectorize(wind_dir_to_cardinal)(dir_deg)
            
            # ORARIO
            hourly = []
            for i in range(len(t2m_corr)):
                dt_utc = ref_dt + timedelta(hours=i)
                dt_loc = utc_to_local(dt_utc)
                # Calcolo GHI rimosso
                
                w_str = classify_weather(
                    t2m_corr[i], rh2m[i], clct[i], clcl[i], clcm[i], clch[i],
                    tp_h[i], spd_kmh[i], lpi[i], cape[i], uh[i],
                    season, season_thresh, timestep_hours=1
                )
                
                hourly.append({
                    "d": dt_loc.strftime("%Y%m%d"),
                    "h": dt_loc.strftime("%H"),
                    "t": round(float(t2m_corr[i]), 1),
                    "r": round(float(rh2m[i])),
                    "p": round(float(tp_h[i]), 1),
                    "pr": round(float(pmsl_corr[i])), # Salviamo P corretta
                    "v": round(float(spd_kmh[i]), 1),
                    "vd": str(card_dir[i]),
                    "vg": round(float(vmax[i]), 1),
                    "w": w_str
                })
            
            # TRIORARIO
            trio = []
            if len(hourly) >= 3:
                for b in range(0, (len(hourly)//3)*3, 3):
                    chk = hourly[b:b+3]
                    t_avg = np.mean([x["t"] for x in chk])
                    r_avg = np.mean([x["r"] for x in chk])
                    v_avg = np.mean([x["v"] for x in chk])
                    pr_avg = np.mean([x["pr"] for x in chk])
                    p_sum = sum([x["p"] for x in chk])
                    
                    idx_m = b + 1 if b+1 < len(clct) else b
                    w_3h = classify_weather(
                        t_avg, r_avg, clct[idx_m], clcl[idx_m], clcm[idx_m], clch[idx_m],
                        p_sum, v_avg, lpi[idx_m], cape[idx_m], uh[idx_m],
                        season, season_thresh, timestep_hours=3
                    )
                    
                    trio.append({
                        "d": chk[0]["d"],
                        "h": chk[0]["h"],
                        "t": round(t_avg, 1),
                        "r": round(r_avg),
                        "p": round(p_sum, 1),
                        "pr": round(pr_avg),
                        "v": round(v_avg, 1),
                        "vd": Counter([x["vd"] for x in chk]).most_common(1)[0][0],
                        "vg": round(max([x["vg"] for x in chk]), 1),
                        "w": w_3h
                    })

            # GIORNALIERO
            daily = calculate_daily_summaries(hourly, clct, clcl, clcm, clch, lpi, cape, uh, season, season_thresh)
            
            final_obj = {
                "r": RUN, "c": city, "x": info['lat'], "y": info['lon'], "z": info['elev'],
                "ORARIO": hourly, "TRIORARIO": trio, "GIORNALIERO": daily
            }
            
            safe_name = city.replace("'", " ")
            with open(f"{output_dir}/{safe_name}.json", 'w') as f:
                json.dump(final_obj, f, separators=(',', ':'), ensure_ascii=False)
            
            cnt += 1
            if cnt % 100 == 0: print(f"Fatti {cnt}")
            
        except Exception as e:
            print(f"Errore {city}: {e}")
            continue

    print(f"Finito. {cnt} città salvate in {output_dir}")

if __name__ == "__main__":
    process_data()

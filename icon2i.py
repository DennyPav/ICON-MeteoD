#!/bin/env python3
import os
import requests
import json
import numpy as np
import xarray as xr
from datetime import datetime, timedelta, timezone
from bs4 import BeautifulSoup
from collections import Counter
import math

# ------------------- CONFIG -------------------
WORKDIR = os.getcwd()
VARIABLES = ['T_2M', 'RELHUM', 'TOT_PREC', 'CLCT', 'CLCL', 'CLCM', 'CLCH', 'U_10M', 'V_10M', 'VMAX_10M', 'LPI', 'CAPE_ML', 'CAPE_CON', 'UH_MAX', 'PMSL', 'HSURF']
VENUES_PATH = f"{WORKDIR}/comuni_italia.json"

# Lapse rates
LAPSE_DRY = 0.0098
LAPSE_MOIST = 0.006
LAPSE_P = 0.12

# SOGLIE NEBBIA OTTIMIZZATE (con fog_max_t dinamico)
SEASON_THRESHOLDS = {
    "winter": {
        "start_day": 1, "end_day": 80, 
        "fog_rh": 96, "haze_rh": 85, 
        "fog_wind": 7.0, "haze_wind": 12.0,
        "fog_max_t": 15.0
    },
    "spring": {
        "start_day": 81, "end_day": 172, 
        "fog_rh": 97, "haze_rh": 85, 
        "fog_wind": 6.0, "haze_wind": 10.0,
        "fog_max_t": 20.0
    },
    "summer": {
        "start_day": 173, "end_day": 263, 
        "fog_rh": 98, "haze_rh": 90, 
        "fog_wind": 4.0, "haze_wind": 9.0,
        "fog_max_t": 26.0
    },
    "autumn": {
        "start_day": 264, "end_day": 365, 
        "fog_rh": 95, "haze_rh": 88, 
        "fog_wind": 7.0, "haze_wind": 11.0,
        "fog_max_t": 20.0
    }
}

CET = timezone(timedelta(hours=1))
CEST = timezone(timedelta(hours=2))

# ------------------- METEO CORE -------------------
def utc_to_local(dt_utc):
    now_utc = datetime.now(timezone.utc)
    if 31 <= now_utc.month <= 10 or (now_utc.month == 3 and now_utc.day >= 25) or (now_utc.month == 10 and now_utc.day <= 25):
        return dt_utc.astimezone(CEST)
    return dt_utc.astimezone(CET)

def wet_bulb_celsius(t_c, rh_percent):
    return t_c * np.arctan(0.151977 * np.sqrt(rh_percent + 8.313659)) + np.arctan(t_c + rh_percent) - np.arctan(rh_percent - 1.676331) + 0.00391838 * rh_percent**1.5 * np.arctan(0.023101 * rh_percent) - 4.686035

def get_run_datetime_now_utc():
    now = datetime.now(timezone.utc)
    if now.hour < 3: return (now - timedelta(days=1)).strftime("%Y%m%d"), "12"
    elif now.hour < 14: return now.strftime("%Y%m%d"), "00"
    return now.strftime("%Y%m%d"), "12"

def download_icon_data(run_date, run_hour):
    grib_dir = f"{WORKDIR}/grib_data/{run_date}{run_hour}"
    os.makedirs(grib_dir, exist_ok=True)
    base_url = f'https://meteohub.agenziaitaliameteo.it/nwp/ICON-2I_SURFACE_PRESSURE_LEVELS/{run_date}{run_hour}/'
    try:
        if requests.head(base_url, timeout=10).status_code != 200: raise RuntimeError("Run off")
    except: pass
    
    for var in VARIABLES:
        local_path = f"{grib_dir}/{var}.grib"
        if os.path.isfile(local_path) and os.path.getsize(local_path) > 1000: continue
        try:
            r = requests.get(f'{base_url}{var}/', timeout=30)
            soup = BeautifulSoup(r.text, 'html.parser')
            grib_files = [a.get('href') for a in soup.find_all('a') if a.get('href', '').endswith('.grib')]
            if grib_files:
                with requests.get(f'{base_url}{var}/{grib_files[0]}', stream=True, timeout=120) as resp:
                    resp.raise_for_status()
                    with open(local_path, 'wb') as f: 
                        for chunk in resp.iter_content(chunk_size=8192): f.write(chunk)
                print(f"DL {var}")
        except Exception: pass
    return len([f for f in os.listdir(grib_dir) if f.endswith('.grib')]) >= len(VARIABLES)

def kelvin_to_celsius(k): return k - 273.15
def mps_to_kmh(mps): return mps * 3.6
def wind_speed_direction(u, v): return np.sqrt(u**2 + v**2), (np.degrees(np.arctan2(-u, -v)) % 360)
def wind_dir_to_cardinal(deg): return ['N','NE','E','SE','S','SW','W','NW'][int((deg + 22.5) % 360 // 45)]
def get_season_precise(dt): 
    d = dt.timetuple().tm_yday
    for s, t in SEASON_THRESHOLDS.items():
        if t["start_day"] <= d <= t["end_day"]: return s, t
    return "winter", SEASON_THRESHOLDS["winter"]

def altitude_correction(t2m, rh, hs_model, hs_station, pmsl):
    dz = hs_model - hs_station
    if dz >= 0: return t2m, pmsl
    wm = np.clip(rh / 100.0, 0.0, 1.0)
    return t2m + (LAPSE_DRY * (1.0 - wm) + LAPSE_MOIST * wm) * dz, pmsl + (dz / 100.0) * LAPSE_P * 100

# ------------------- CLASSIFIERS -------------------

# 1. Classificatore ORARIO
def classify_weather_hourly(t2m, rh2m, clct, clcl, clcm, clch,
                     tp_rate, wind_kmh, lpi, cape, uh,
                     season, season_thresh):
    
    wet_bulb = wet_bulb_celsius(t2m, rh2m)
    is_snow = wet_bulb < 0.5
    prec_high = "NEVE" if is_snow else "PIOGGIA"
    prec_low  = "NEVISCHIO" if is_snow else "PIOGGERELLA"
    
    octas = clct / 100.0 * 8
    low = clcl if np.isfinite(clcl) else (clcm if np.isfinite(clcm) else 0)

    if clch > 60 and low < 30 and octas > 5: c_state = "NUBI ALTE"
    elif octas <= 2: c_state = "SERENO"
    elif octas <= 4: c_state = "POCO NUVOLOSO"
    elif octas <= 6: c_state = "NUVOLOSO"
    else: c_state = "COPERTO"

    conv_signal = ((cape >= 400 and lpi >= 1.5) or (uh >= 50) or (cape >= 800))
    rain_signal = tp_rate >= 0.3
    gust_signal = wind_kmh >= 35
    deep_clouds = clct >= 90 and (clcm >= 40 or clch >= 40)
    
    if conv_signal and (rain_signal or gust_signal) and deep_clouds:
        return "TEMPORALE"
    
    tp_rate = round(tp_rate, 1)
    
    # Recupero soglie
    fog_rh = season_thresh.get("fog_rh", 95)
    fog_wd = season_thresh.get("fog_wind", 8)
    fog_t  = season_thresh.get("fog_max_t", 18) # default dinamico
    haze_rh = season_thresh.get("haze_rh", 85)
    haze_wd = season_thresh.get("haze_wind", 12)

    if tp_rate >= 0.1:
        if c_state == "SERENO": c_state = "POCO NUVOLOSO"
        if tp_rate > 0.3:
            intent = "INTENSA" if tp_rate >= 7.0 else ("MODERATA" if tp_rate >= 2.0 else "DEBOLE")
            return f"{c_state} {prec_high} {intent}"
        elif math.isclose(tp_rate, 0.3, abs_tol=1e-3):
            return f"{c_state} {prec_low}"
        else:
            # 0.1 <= tp < 0.3: Nebbia/Foschia/Nevischio
            # Controllo T < soglia dinamica
            if t2m < fog_t and rh2m >= fog_rh and wind_kmh <= fog_wd and low >= 80: return "NEBBIA"
            elif t2m < fog_t and rh2m >= haze_rh and wind_kmh <= haze_wd and low >= 50: return "FOSCHIA"
            else: return f"{c_state} {prec_low}"
    else:
        # tp < 0.1
        if t2m < fog_t and rh2m >= fog_rh and wind_kmh <= fog_wd and low >= 80: return "NEBBIA"
        elif t2m < fog_t and rh2m >= haze_rh and wind_kmh <= haze_wd and low >= 50: return "FOSCHIA"
        else: return c_state


# 2. Classificatore TRIORARIO AGGREGATO
def classify_weather_3h_aggregated(t_avg, rh_avg, clct_avg, tp_sum, wind_avg, hourly_descriptions_list, season_thresh):
    
    wet_bulb = wet_bulb_celsius(t_avg, rh_avg)
    prec_type_high = "NEVE" if wet_bulb < 0.5 else "PIOGGIA"
    prec_type_low = "NEVISCHIO" if wet_bulb < 0.5 else "PIOGGERELLA"
    
    octas = clct_avg / 100.0 * 8
    if octas <= 2: cloud_state = "SERENO"
    elif octas <= 4: cloud_state = "POCO NUVOLOSO"
    elif octas <= 6: cloud_state = "NUVOLOSO"
    else: cloud_state = "COPERTO"

    # --- CASO 1: P > 0.9 mm ---
    if tp_sum > 0.9:
        if cloud_state == "SERENO": cloud_state = "POCO NUVOLOSO"
        if tp_sum >= 20.0: intensity = "INTENSA"
        elif tp_sum >= 5.0: intensity = "MODERATA"
        else: intensity = "DEBOLE"
        return f"{cloud_state} {prec_type_high} {intensity}"

    # --- CASO 2: 0.1 <= P <= 0.9 mm ---
    elif 0.1 <= tp_sum <= 0.9:
        has_rain_snow = any(("PIOGGIA" in w or "NEVE" in w) for w in hourly_descriptions_list)
        has_drizzle_sleet = any(("PIOGGERELLA" in w or "NEVISCHIO" in w) for w in hourly_descriptions_list)
        has_fog = any("NEBBIA" in w for w in hourly_descriptions_list)
        has_haze = any("FOSCHIA" in w for w in hourly_descriptions_list)
        
        if has_rain_snow:
            if cloud_state == "SERENO": cloud_state = "POCO NUVOLOSO"
            return f"{cloud_state} {prec_type_high} DEBOLE"
        elif has_drizzle_sleet:
            if cloud_state == "SERENO": cloud_state = "POCO NUVOLOSO"
            return f"{cloud_state} {prec_type_low}"
        elif has_fog: return "NEBBIA"
        elif has_haze: return "FOSCHIA"
        else: return cloud_state

    # --- CASO 3: P < 0.1 mm ---
    else:
        fog_rh = season_thresh.get("fog_rh", 95)
        fog_wd = season_thresh.get("fog_wind", 8)
        fog_t  = season_thresh.get("fog_max_t", 18)
        haze_rh = season_thresh.get("haze_rh", 85)
        haze_wd = season_thresh.get("haze_wind", 12)

        if t_avg < fog_t:
            if rh_avg >= fog_rh and wind_avg <= fog_wd: return "NEBBIA"
            if rh_avg >= haze_rh and wind_avg <= haze_wd: return "FOSCHIA"
            
        return cloud_state

# 3. Classificatore GIORNALIERO
def classify_daily_weather(recs, clct_avg, clcl_avg, clcm_avg, clch_avg, tp_tot, season, thresh):
    snow_hours = 0
    rain_hours = 0
    has_significant_snow_or_rain = False
    fog_hours = 0
    haze_hours = 0

    for r in recs:
        hour = int(r.get("h", 0))
        wtxt = r.get("w", "")
        if "PIOGGIA" in wtxt:
            has_significant_snow_or_rain = True
            rain_hours += 1
        elif "NEVE" in wtxt:
            has_significant_snow_or_rain = True
            snow_hours += 1
        elif 5 <= hour <= 22:
            if "NEBBIA" in wtxt: fog_hours += 1
            elif "FOSCHIA" in wtxt: haze_hours += 1

    is_snow_day = snow_hours > rain_hours
    octas = clct_avg / 100.0 * 8
    low = clcl_avg if np.isfinite(clcl_avg) else (clcm_avg if np.isfinite(clcm_avg) else 0)

    if clch_avg > 60 and low < 30 and octas > 5: c_state = "NUBI ALTE"
    elif octas <= 2: c_state = "SERENO"
    elif octas <= 4: c_state = "POCO NUVOLOSO"
    elif octas <= 6: c_state = "NUVOLOSO"
    else: c_state = "COPERTO"

    if not has_significant_snow_or_rain:
        total_fog_haze = fog_hours + haze_hours
        if total_fog_haze >= 9:
            return "NEBBIA" if fog_hours >= haze_hours else "FOSCHIA"
        else:
            return c_state

    prec_type = "NEVE" if is_snow_day else "PIOGGIA"
    if tp_tot >= 30.0: intensity = "INTENSA"
    elif tp_tot >= 10.0: intensity = "MODERATA"
    else: intensity = "DEBOLE"
    
    if c_state == "SERENO": c_state = "POCO NUVOLOSO"
    return f"{c_state} {prec_type} {intensity}"


# ------------------- DATA PROCESSING -------------------
def extract(var, y, x, weighted=False):
    if np.isscalar(var) or var.size == 1: return np.array([float(var)])
    if var.ndim == 1: return var
    sl = var[..., y, x]
    if var.ndim == 2 or not weighted: return sl
    
    ring1 = []
    NY, NX = var.shape[-2:]
    for di in [-1,0,1]:
        for dj in [-1,0,1]:
            if di==0 and dj==0: continue
            ni, nj = y+di, x+dj
            if 0<=ni<NY and 0<=nj<NX: ring1.append(var[..., ni, nj])
            
    if not ring1: return sl
    m1 = np.stack(ring1, axis=-1).mean(axis=-1)
    
    ring2 = []
    for di in [-2,-1,0,1,2]:
        for dj in [-2,-1,0,1,2]:
            if abs(di)<=1 and abs(dj)<=1: continue
            ni, nj = y+di, x+dj
            if 0<=ni<NY and 0<=nj<NX: ring2.append(var[..., ni, nj])
            
    if not ring2: return 0.6*sl + 0.4*m1
    m2 = np.stack(ring2, axis=-1).mean(axis=-1)
    return 0.5*sl + 0.3*m1 + 0.2*m2

def process_data():
    RUN = os.getenv("RUN", "")
    if not RUN:
        d, h = get_run_datetime_now_utc()
        RUN = d+h
        download_icon_data(d, h)
    
    print(f"Start {RUN}")
    out = f"{WORKDIR}/{RUN}"
    os.makedirs(out, exist_ok=True)
    
    D = {}
    for v in VARIABLES:
        p = f"{WORKDIR}/grib_data/{RUN}/{v}.grib"
        if os.path.exists(p): D[v] = xr.open_dataset(p, engine='cfgrib', backend_kwargs={"indexpath": ""})
            
    hs_da = list(D['HSURF'].data_vars.values())[0]
    latg, long = hs_da['latitude'].values, hs_da['longitude'].values
    if latg.ndim==1: long, latg = np.meshgrid(long, latg)
    hsg = hs_da.values if hs_da.ndim==2 else hs_da.isel(time=0).values
    
    venues = json.load(open(VENUES_PATH))
    ref = datetime.strptime(RUN, "%Y%m%d%H").replace(tzinfo=timezone.utc)
    seas, thr = get_season_precise(ref)
    
    for c, i in venues.items():
        if isinstance(i, list): ly, lx, le = i[0], i[1], i[2]
        else: ly, lx, le = i['lat'], i['lon'], i['elev']
        
        try:
            dist = (latg - ly)**2 + (long - lx)**2
            cy, cx = np.unravel_index(np.argmin(dist), dist.shape)
            for rad in [0.1, 0.2, 0.5]:
                mask = (np.abs(latg-ly)<=rad) & (np.abs(long-lx)<=rad) & (hsg>0)
                if np.any(mask):
                    ys, xs = np.where(mask)
                    idx = np.argmin((latg[ys,xs]-ly)**2 + (long[ys,xs]-lx)**2)
                    cy, cx = ys[idx], xs[idx]
                    break
            
            t2 = extract(kelvin_to_celsius(D['T_2M']['t2m'].values), cy, cx)
            rh = np.clip(extract(D['RELHUM']['r'].values, cy, cx), 0, 100)
            u, v = extract(D['U_10M']['u10'].values, cy, cx), extract(D['V_10M']['v10'].values, cy, cx)
            vm = mps_to_kmh(extract(D['VMAX_10M']['fg10'].values, cy, cx))
            pm = extract(D['PMSL']['pmsl'].values/100, cy, cx)
            lpi = extract(D['LPI']['unknown'].values, cy, cx)
            cape = extract(np.maximum(D['CAPE_ML']['cape_ml'].values, D['CAPE_CON']['cape_con'].values), cy, cx)
            uh = extract(D['UH_MAX']['unknown'].values, cy, cx)
            
            tp = extract(np.diff(D['TOT_PREC']['tp'].values, axis=0, prepend=0), cy, cx, True)
            ct = extract(D['CLCT']['clct'].values, cy, cx, True)
            cl = extract(D['CLCL']['ccl'].values if 'CLCL' in D else np.zeros_like(ct), cy, cx, True)
            cm = extract(D['CLCM']['ccl'].values if 'CLCM' in D else np.zeros_like(ct), cy, cx, True)
            ch = extract(D['CLCH']['ccl'].values if 'CLCH' in D else np.zeros_like(ct), cy, cx, True)
            
            tc, pc = altitude_correction(t2, rh, hsg[cy,cx], le, pm)
            ws, wd = wind_speed_direction(u, v)
            wk = mps_to_kmh(ws)
            wdirs = np.vectorize(wind_dir_to_cardinal)(wd)
            
            H, T, G = [], [], []
            
            # ORARIO
            for k in range(len(tc)):
                loc = utc_to_local(ref + timedelta(hours=k))
                wtxt = classify_weather_hourly(tc[k], rh[k], ct[k], cl[k], cm[k], ch[k], tp[k], wk[k], lpi[k], cape[k], uh[k], seas, thr)
                H.append({
                    "d": loc.strftime("%Y%m%d"), 
                    "h": loc.strftime("%H"), 
                    "t": round(float(tc[k]),1), 
                    "r": round(float(rh[k])), 
                    "p": round(float(tp[k]),1), 
                    "pr": round(float(pc[k])), 
                    "v": round(float(wk[k]),1), 
                    "vd": str(wdirs[k]), 
                    "vg": round(float(vm[k]),1), 
                    "w": wtxt
                })
                
            # TRIORARIO (Nuova logica prioritaria + Fix Float)
            num_blocks = len(tc) // 3
            for i in range(num_blocks):
                b = i * 3
                e = b + 3
                
                # CAST A FLOAT PER JSON
                t_avg = float(np.mean(tc[b:e]))
                rh_avg = float(np.mean(rh[b:e]))
                ct_avg = float(np.mean(ct[b:e]))
                wk_avg = float(np.mean(wk[b:e]))
                pr_avg = float(np.mean(pc[b:e]))
                tp_sum = float(np.sum(tp[b:e]))
                
                # Lista descrizioni orarie per il controllo prioritario
                chk = H[b:e]
                w_list = [x["w"] for x in chk]
                
                # --- NUOVA LOGICA DIREZIONE VENTO ---
                # 1. Contiamo le frequenze delle direzioni
                dirs_counter = Counter([x["vd"] for x in chk])
                most_common_dir, freq = dirs_counter.most_common(1)[0]
                
                # 2. Se la frequenza è 1 (significa che abbiamo 3 direzioni diverse, es: N, NW, W)
                #    allora prendiamo la direzione dell'ora con il vento più forte ("v")
                if freq == 1:
                    max_wind_item = max(chk, key=lambda item: item["v"])
                    selected_vd = max_wind_item["vd"]
                else:
                    # Altrimenti vince la maggioranza (2 su 3, o 3 su 3)
                    selected_vd = most_common_dir
                # ------------------------------------

                # Classificazione 3H
                w3 = classify_weather_3h_aggregated(t_avg, rh_avg, ct_avg, tp_sum, wk_avg, w_list, thr)
                
                T.append({
                    "d": chk[0]["d"], 
                    "h": chk[0]["h"], 
                    "t": round(t_avg, 1), 
                    "r": round(rh_avg), 
                    "p": round(tp_sum, 1), 
                    "pr": round(pr_avg), 
                    "v": round(wk_avg, 1), 
                    "vd": selected_vd, # Qui usiamo la variabile calcolata sopra
                    "vg": round(max(x["vg"] for x in chk), 1), 
                    "w": w3
                })

                
            days = {}
            for r in H: days.setdefault(r["d"], []).append(r)
            d_keys = sorted(days.keys())
            if len(d_keys)>1: d_keys = d_keys[:-1]
            
            for d in d_keys:
                recs = days[d]
                idxs = [i for i, x in enumerate(H) if x["d"] == d]
                if not idxs: continue
                
                ct_a = np.mean(ct[idxs])
                cl_a = np.mean(cl[idxs])
                cm_a = np.mean(cm[idxs])
                ch_a = np.mean(ch[idxs])
                tp_tot = sum(r["p"] for r in recs)
                
                wdaily = classify_daily_weather(recs, ct_a, cl_a, cm_a, ch_a, tp_tot, seas, thr)
                G.append({
                    "d": d, 
                    "tmin": round(min(r["t"] for r in recs),1), 
                    "tmax": round(max(r["t"] for r in recs),1), 
                    "p": round(tp_tot,1), 
                    "w": wdaily
                })
            
            safe_c = c.replace("'", " ")
            with open(f"{out}/{safe_c}.json", 'w') as f:
                json.dump({"r": RUN, "c": c, "x": ly, "y": lx, "z": le, "ORARIO": H, "TRIORARIO": T, "GIORNALIERO": G}, f, separators=(',', ':'), ensure_ascii=False)
                
        except Exception as e: print(f"Err {c}: {e}")

if __name__ == "__main__":
    process_data()

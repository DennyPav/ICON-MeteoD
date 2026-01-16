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
VARIABLES = ['T_2M', 'RELHUM', 'TOT_PREC', 'CLCT', 'CLCL', 'CLCM', 'CLCH', 'U_10M', 'V_10M', 'VMAX_10M', 'LPI', 'CAPE_ML', 'CAPE_CON', 'UH_MAX', 'PMSL', 'HSURF']
VENUES_PATH = f"{WORKDIR}/comuni_italia.json"

# Lapse rates
LAPSE_DRY = 0.0098
LAPSE_MOIST = 0.006
LAPSE_P = 0.12

# Soglie Nebbia
SEASON_THRESHOLDS = {
    "winter": {"start_day": 1, "end_day": 80, "fog_rh": 97, "haze_rh": 90, "fog_wind": 3, "haze_wind": 6},
    "spring": {"start_day": 81, "end_day": 172, "fog_rh": 95, "haze_rh": 87, "fog_wind": 3.5, "haze_wind": 7},
    "summer": {"start_day": 173, "end_day": 263, "fog_rh": 93, "haze_rh": 83, "fog_wind": 4, "haze_wind": 8},
    "autumn": {"start_day": 264, "end_day": 365, "fog_rh": 96, "haze_rh": 88, "fog_wind": 3.2, "haze_wind": 6.5}
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

# ------------------- CLASSIFIER -------------------
def classify_weather(t2m, rh2m, clct, clcl, clcm, clch,
                     tp_rate, wind_kmh, lpi, cape, uh,
                     season, season_thresh, timestep_hours=1):

    wet_bulb = wet_bulb_celsius(t2m, rh2m)
    is_snow = wet_bulb < 0.5
    prec_high = "NEVE" if is_snow else "PIOGGIA"
    prec_low = "NEVISCHIO" if is_snow else "PIOGGERELLA"

    octas = clct / 100.0 * 8
    low = clcl if np.isfinite(clcl) else (clcm if np.isfinite(clcm) else 0)

    # Stato cielo
    if clch > 60 and low < 30 and octas > 5:
        c_state = "NUBI ALTE"
    elif octas <= 2: c_state = "SERENO"
    elif octas <= 4: c_state = "POCO NUVOLOSO"
    elif octas <= 6: c_state = "NUVOLOSO"
    else: c_state = "COPERTO"

    # TEMPORALE
    conv_signal = ((cape >= 400 and lpi >= 1.5) or (uh >= 50) or (cape >= 800))
    rain_signal = tp_rate >= (0.3 * timestep_hours)
    gust_signal = wind_kmh >= 35
    deep_clouds = clct >= 90 and (clcm >= 40 or clch >= 40)
    if conv_signal and (rain_signal or gust_signal) and deep_clouds:
        return "TEMPORALE"

    # PRECIPITAZIONE
    if tp_rate >= 0.1:

        # Se lo stato del cielo è SERENO e c'è precipitazione → cambialo in POCO NUVOLOSO
        if c_state == "SERENO":
            c_state = "POCO NUVOLOSO"

        # tp_rate > 0.3 → intensità
        if tp_rate > 0.3:
            if timestep_hours == 1: s_mod, s_int = 2.0, 7.0
            elif timestep_hours == 3: s_mod, s_int = 5.0, 20.0
            else: s_mod, s_int = 10.0, 30.0
            intent = "INTENSA" if tp_rate >= s_int else ("MODERATA" if tp_rate >= s_mod else "DEBOLE")
            return f"{c_state} {prec_high} {intent}"

        # tp_rate == 0.3 → solo prec_low
        if tp_rate == 0.3:
            return f"{c_state} {prec_low}"

        # tp_rate < 0.3 → prec_low o nebbia/foschia
        if t2m < 12 and rh2m >= season_thresh["fog_rh"] and wind_kmh <= season_thresh["fog_wind"] and low >= 80:
            return "NEBBIA"
        if t2m < 12 and rh2m >= season_thresh["haze_rh"] and wind_kmh <= season_thresh["haze_wind"] and low >= 50:
            return "FOSCHIA"
        return f"{c_state} {prec_low}"
    
    # Default: solo cielo
    return c_state



# ------------------- CLASSIFICATORE TRIORARIO -------------------
def classify_triorario(H_chunk, season, season_thresh):
    """
    H_chunk: lista di 3 record orari {"t":..., "r":..., "p":..., "w":...}
    """
    has_prec_high = False
    has_prec_low = False
    has_fog = False
    has_haze = False
    p_sum = sum([x["p"] for x in H_chunk])
    t_avg = np.mean([x["t"] for x in H_chunk])

    # Analizza le ore
    for h in H_chunk:
        wtxt = h["w"]
        if "PIOGGIA" in wtxt or "NEVE" in wtxt:
            has_prec_high = True
        elif "PIOGGERELLA" in wtxt or "NEVISCHIO" in wtxt:
            has_prec_low = True
        elif "NEBBIA" in wtxt:
            has_fog = True
        elif "FOSCHIA" in wtxt:
            has_haze = True

    # Stato cielo medio
    octas = np.mean([h.get("clct",0) for h in H_chunk])/100*8
    low = np.mean([h.get("clcl",0) for h in H_chunk])
    if octas<=2: c_state="SERENO"
    elif octas<=4: c_state="POCO NUVOLOSO"
    elif octas<=6: c_state="NUVOLOSO"
    else: c_state="COPERTO"

    # LOGICA TRIORARIO AGGREGATA
    if p_sum > 0.9:
        return f"{c_state} {'NEVE' if t_avg<0.5 else 'PIOGGIA'} DEBOLE"
    elif 0.1 <= p_sum <= 0.9:
        if has_prec_high:
            return f"{c_state} {'NEVE' if t_avg<0.5 else 'PIOGGIA'} DEBOLE"
        elif has_prec_low:
            return f"{c_state} {'NEVISCHIO' if t_avg<0.5 else 'PIOGGERELLA'}"
        elif has_fog:
            return "NEBBIA"
        elif has_haze:
            return "FOSCHIA"
        else:
            return c_state
    else:
        if has_fog: return "NEBBIA"
        if has_haze: return "FOSCHIA"
        return c_state


def classify_daily_weather(recs, clct_avg, clcl_avg, clcm_avg, clch_avg, tp_tot, season, thresh):
    # Conta ore con neve/pioggia "significativa" (no pioggerella/nevischio)
    snow_hours = 0
    rain_hours = 0
    has_significant_snow_or_rain = False

    for r in recs:
        wtxt = r.get("w", "")
        # Considera solo stati con PIOGGIA/NEVE e ignora PIOGGERELLA/NEVISCHIO
        if "PIOGGIA" in wtxt or "NEVE" in wtxt:
            has_significant_snow_or_rain = True
            wb = wet_bulb_celsius(r["t"], r["r"])
            if wb < 0.5:
                snow_hours += 1
            else:
                rain_hours += 1

    is_snow_day = snow_hours > rain_hours

    # Copertura nuvolosa media
    octas = clct_avg / 100.0 * 8
    low = clcl_avg if np.isfinite(clcl_avg) else (clcm_avg if np.isfinite(clcm_avg) else 0)

    if clch_avg > 60 and low < 30 and octas > 5:
        c_state = "NUBI ALTE"
    elif octas <= 2:
        c_state = "SERENO"
    elif octas <= 4:
        c_state = "POCO NUVOLOSO"
    elif octas <= 6:
        c_state = "NUVOLOSO"
    else:
        c_state = "COPERTO"

    # Se **non** c'è stata almeno un'ora con pioggia/neve significativa,
    # NON aggiungere PIOGGIA/NEVE nel giornaliero
    if not has_significant_snow_or_rain:
        return c_state

    # Se c'è stata pioggia/neve significativa, allora usa ancora la logica
    # sull'intensità basata su tp_tot
    prec_type = "NEVE" if is_snow_day else "PIOGGIA"

    if tp_tot >= 30.0:
        intensity = "INTENSA"
    elif tp_tot >= 10.0:
        intensity = "MODERATA"
    else:
        intensity = "DEBOLE"

    if c_state == "SERENO":
        c_state = "POCO NUVOLOSO"

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
            pm = extract(D['PMSL']['prmsl'].values/100, cy, cx)
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
            
            for k in range(len(tc)):
                loc = utc_to_local(ref + timedelta(hours=k))
                wtxt = classify_weather(tc[k], rh[k], ct[k], cl[k], cm[k], ch[k], tp[k], wk[k], lpi[k], cape[k], uh[k], seas, thr, 1)
                H.append({"d": loc.strftime("%Y%m%d"), "h": loc.strftime("%H"), "t": round(float(tc[k]),1), "r": round(float(rh[k])), "p": round(float(tp[k]),1), "pr": round(float(pc[k])), "v": round(float(wk[k]),1), "vd": str(wdirs[k]), "vg": round(float(vm[k]),1), "w": wtxt})
                
            for b in range(0, (len(H)//3)*3, 3):
                chk = H[b:b+3]
                psum = sum(x["p"] for x in chk)
                w3 = classify_triorario(chk, seas, thr)
                T.append({"d": chk[0]["d"], "h": chk[0]["h"], "t": round(np.mean([x["t"] for x in chk]),1), "r": round(np.mean([x["r"] for x in chk])), "p": round(psum,1), "pr": round(np.mean([x["pr"] for x in chk])), "v": round(np.mean([x["v"] for x in chk]),1), "vd": Counter([x["vd"] for x in chk]).most_common(1)[0][0], "vg": round(max(x["vg"] for x in chk),1), "w": w3})
                
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
                G.append({"d": d, "tmin": round(min(r["t"] for r in recs),1), "tmax": round(max(r["t"] for r in recs),1), "p": round(tp_tot,1), "w": wdaily})
            
            # CORREZIONE QUI
            safe_c = c.replace("'", " ")
            with open(f"{out}/{safe_c}.json", 'w') as f:
                json.dump({"r": RUN, "c": c, "x": ly, "y": lx, "z": le, "ORARIO": H, "TRIORARIO": T, "GIORNALIERO": G}, f, separators=(',', ':'), ensure_ascii=False)
                
        except Exception as e: print(f"Err {c}: {e}")

if __name__ == "__main__":
    process_data()

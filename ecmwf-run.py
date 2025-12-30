#!/bin/env python3
import os
import json
import numpy as np
import xarray as xr
from datetime import datetime, timedelta, timezone
from ecmwf.opendata import Client
from collections import Counter

WORKDIR = os.getcwd()
VENUES_PATH = f"{WORKDIR}/comuni_italia.json"

# Lapse rates
LAPSE_DRY = 0.0098   # °C/m
LAPSE_MOIST = 0.006 # °C/m
LAPSE_P = 0.012      # hPa/100m

# SOGLIE STAGIONALI
SEASON_THRESHOLDS = {
    "winter": {"start_day": 1, "end_day": 80, "fog_rh": 97, "haze_rh": 90, "fog_wind": 5, "haze_wind": 15},
    "spring": {"start_day": 81, "end_day": 172, "fog_rh": 95, "haze_rh": 87, "fog_wind": 7, "haze_wind": 20},
    "summer": {"start_day": 173, "end_day": 263, "fog_rh": 93, "haze_rh": 83, "fog_wind": 10, "haze_wind": 25},
    "autumn": {"start_day": 264, "end_day": 365, "fog_rh": 96, "haze_rh": 88, "fog_wind": 6, "haze_wind": 18}
}

CET = timezone(timedelta(hours=1))
CEST = timezone(timedelta(hours=2))

def utc_to_local(dt_utc):
    """UTC → CET/CEST Italia"""
    m, d = dt_utc.month, dt_utc.day
    if (m > 3 and m < 10) or (m == 3 and d >= 25) or (m == 10 and d <= 25):
        return dt_utc.astimezone(CEST)
    return dt_utc.astimezone(CET)

def wet_bulb_celsius(t_c, rh_percent):
    """
    Calcolo wet-bulb approssimato solo da temperatura [°C] e umidità [%]
    """
    tw = t_c * np.arctan(0.151977 * np.sqrt(rh_percent + 8.313659)) \
         + np.arctan(t_c + rh_percent) - np.arctan(rh_percent - 1.676331) \
         + 0.00391838 * rh_percent**1.5 * np.arctan(0.023101 * rh_percent) \
         - 4.686035
    return tw

def get_run_datetime_now_utc():
    now = datetime.now(timezone.utc)
    if now.hour < 8:
        return (now - timedelta(days=1)).strftime("%Y%m%d"), "00"
    elif now.hour < 20:
        return now.strftime("%Y%m%d"), "00"
    return now.strftime("%Y%m%d"), "12"

def download_ecmwf_data(run_date, run_hour):
    """Download ECMWF con MU CAPE + OROGRAPHY Z"""
    grib_dir = f"{WORKDIR}/grib_ecmwf/{run_date}{run_hour}"
    os.makedirs(grib_dir, exist_ok=True)
    
    grib_file = f"{grib_dir}/ecmwf_main.grib"
    wind_file = f"{grib_dir}/ecmwf_wind.grib"
    orog_file = f"{grib_dir}/ecmwf_orog.grib"
    
    steps = list(range(0, 144, 3))
    client = Client(source="ecmwf", model="ifs", resol="0p25")
    
    # MAIN: parametri superficiali + mucape
    if not os.path.exists(grib_file) or os.path.getsize(grib_file) < 30_000_000:
        print(f"📥 ECMWF main + MU CAPE {run_date}{run_hour}")
        client.retrieve(
            date=run_date, time=int(run_hour), stream="oper", type="fc", step=steps,
            param=["2t", "2d", "tcc", "msl", "tp", "mucape"],
            target=grib_file
        )
    
    # WIND: 10u/10v superficiali
    if not os.path.exists(wind_file) or os.path.getsize(wind_file) < 5_000_000:
        print(f"📥 ECMWF wind {run_date}{run_hour}")
        client.retrieve(
            date=run_date, time=int(run_hour), stream="oper", type="fc", step=steps,
            param=["10u", "10v"],
            target=wind_file
        )
    
    # ✅ OROGRAPHY: Z superficiale (orografia TERRENO reale!)
    if not os.path.exists(orog_file) or os.path.getsize(orog_file) < 1_000:
        print(f"📥 ECMWF orog Z {run_date}{run_hour}")
        client.retrieve(
            date=run_date, time=int(run_hour), stream="oper", type="fc", step=[0],
            param="z",                    # ✅ Surface orography (geopotential @ surface)
            target=orog_file              # ✅ No levelist!
        )
    
    return grib_file, wind_file, orog_file

def kelvin_to_celsius(k): return k - 273.15
def mps_to_kmh(mps): return mps * 3.6

def relative_humidity(t2m_k, td2m_k):
    t_c, td_c = kelvin_to_celsius(t2m_k), kelvin_to_celsius(td2m_k)
    es = 6.112 * np.exp((17.67 * t_c)/(t_c+243.5))
    e = 6.112 * np.exp((17.67 * td_c)/(td_c+243.5))
    return np.clip(100*e/es, 0, 100)

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

G = 9.80665        # m s-2
RD = 287.05        # J kg-1 K-1
def altitude_correction(t2m, rh, z_model, z_station, pmsl):
    delta_z = z_model - z_station
    # Peso umidità (0–1)
    w_moist = np.clip(rh / 100.0, 0.0, 1.0)
    # Lapse rate termico (K/m)
    lapse_t = LAPSE_DRY * (1.0 - w_moist) + LAPSE_MOIST * w_moist
    # Temperatura corretta
    t_corr = t2m + lapse_t * delta_z
    # Temperatura media dello strato (in Kelvin)
    T_mean = t_corr + 273.15
    # Pressione alla quota della stazione (da pmsl)
    p_corr = pmsl * np.exp(-G * z_station / (RD * T_mean))
    return t_corr, p_corr

def classify_weather(t2m, rh2m, clct, tp_rate, wind_kmh, mucape, season_thresh, timestep_hours=3):
    """Classificazione ECMWF con MU CAPE"""
    
    # 🔥 TEMPORALE: MU CAPE + Precipitazioni
    if mucape > 400 and tp_rate > 0.5 * timestep_hours:
        return "TEMPORALE"
    
    # SOGLIE PRECIP TRIORARIA
    drizzle_min, drizzle_max = 0.09, 0.3
    if timestep_hours == 3:
        prec_debole_min, prec_moderata_min, prec_intensa_min = 0.3, 5.0, 20.0
    else:  # 24h
        prec_debole_min, prec_moderata_min, prec_intensa_min = 0.3, 10.0, 30.0
    
    # PRECIPITAZIONE (esclusi temporali)
    if tp_rate >= prec_intensa_min:
        prec_intensity = "INTENSA"
    elif tp_rate >= prec_moderata_min:
        prec_intensity = "MODERATA"
    elif tp_rate >= prec_debole_min:
        prec_intensity = "DEBOLE"
    else:
        prec_intensity = None
    
    if prec_intensity:
        octas = clct / 100.0 * 8
        if octas <= 2: cloud_state = "SERENO"
        elif octas <= 4: cloud_state = "POCO NUVOLOSO"
        elif octas <= 6: cloud_state = "NUVOLOSO"
        else: cloud_state = "COPERTO"
        wet_bulb = wet_bulb_celsius(t2m, rh2m)
        prec_type = "NEVE" if wet_bulb < 0.1 else "PIOGGIA"
        return f"{cloud_state} {prec_type} {prec_intensity}"
    
    # PIOGGERELLA/NEBBIA/FOSCHIA
    if drizzle_min <= tp_rate <= drizzle_max:
        if rh2m >= season_thresh["fog_rh"] and wind_kmh <= season_thresh["fog_wind"]:
            return "NEBBIA"
        if rh2m >= season_thresh["haze_rh"] and wind_kmh <= season_thresh["haze_wind"]:
            return "FOSCHIA"
        return "COPERTO PIOGGERELLA"
    
    # NEBBIA/FOSCHIA
    if rh2m >= season_thresh["fog_rh"] and wind_kmh <= season_thresh["fog_wind"]:
        return "NEBBIA"
    if rh2m >= season_thresh["haze_rh"] and wind_kmh <= season_thresh["haze_wind"]:
        return "FOSCHIA"
    
    # NUVOLOSITÀ PURA
    octas = clct / 100.0 * 8
    if octas <= 2: return "SERENO"
    elif octas <= 4: return "POCO NUVOLOSO"
    elif octas <= 6: return "NUVOLOSO"
    return "COPERTO"

def load_venues(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    venues = {c: {"lat": float(v[0]), "lon": float(v[1]), "elev": float(v[2])} 
              for c, v in data.items()}
    print(f"Caricate {len(venues)} città")
    return venues

def calculate_daily_summaries(trihourly_data, clct_daily, tp_rate_daily, mucape_daily, season_thresh):
    """Riepiloghi GIORNALIERI con MU CAPE"""
    daily_summaries = []
    days_data = {}
    
    for i, record in enumerate(trihourly_data):
        day = record["d"]
        if day not in days_data:
            days_data[day] = []
        days_data[day].append((i, record))
    
    for day, day_records in days_data.items():
        day_temps = np.array([record["t"] for _, record in day_records])
        t_min = round(np.min(day_temps), 1)
        t_max = round(np.max(day_temps), 1)
        tp_total = sum([record["p"] for _, record in day_records])
        
        t_mean = np.mean(day_temps)
        r_mean = np.mean([record["r"] for _, record in day_records])
        v_mean = np.mean([record["v"] for _, record in day_records])
        first_idx = day_records[0][0]
        
        weather_day = classify_weather(t_mean, r_mean, clct_daily[first_idx], 
                                     tp_total, v_mean, mucape_daily[first_idx], 
                                     season_thresh, timestep_hours=24)
        
        daily_summaries.append({
            "d": day,
            "tmin": t_min,
            "tmax": t_max,
            "p": round(tp_total, 1),
            "w": weather_day
        })
    
    return daily_summaries

def process_ecmwf_data():
    run_date, run_hour = get_run_datetime_now_utc()
    RUN_DATE_TIME = f"{run_date}{run_hour}"
    RUN = f"{RUN_DATE_TIME}_ecmwf"
    
    print(f"🚀 ECMWF TRIORARIO + MU CAPE + OROGRAPHY Z {RUN}")
    grib_file, wind_file, orog_file = download_ecmwf_data(run_date, run_hour)
    
    # Caricamento dataset
    ds_main = xr.open_dataset(grib_file, engine="cfgrib")
    ds_wind = xr.open_dataset(wind_file, engine="cfgrib")
    ds_orog = xr.open_dataset(orog_file, engine="cfgrib")
    
    print("📊 Dataset caricati:")
    print(f"  Main: {list(ds_main.data_vars.keys())}")
    print(f"  Wind: {list(ds_wind.data_vars.keys())}")
    print(f"  Orog: {list(ds_orog.data_vars.keys())}")  # ✅ Deve mostrare 'z'
    
    venues = load_venues(VENUES_PATH)
    ref_dt = datetime.strptime(RUN_DATE_TIME, "%Y%m%d%H").replace(tzinfo=timezone.utc)
    season, season_thresh = get_season_precise(ref_dt)
    
    outdir = f"{WORKDIR}/{RUN}"
    os.makedirs(outdir, exist_ok=True)
    
    processed = 0
    for city, info in venues.items():
        try:
            lat_idx = np.abs(ds_main.latitude - info['lat']).argmin()
            lon_idx = np.abs(ds_main.longitude - info['lon']).argmin()
            
            # DATI TRIORARI ECMWF + MU CAPE ✅
            t2m_k = ds_main["t2m"].isel(latitude=lat_idx, longitude=lon_idx).values
            td2m_k = ds_main["d2m"].isel(latitude=lat_idx, longitude=lon_idx).values
            tcc = ds_main["tcc"].isel(latitude=lat_idx, longitude=lon_idx).values * 100
            msl = ds_main["msl"].isel(latitude=lat_idx, longitude=lon_idx).values / 100
            tp_cum = ds_main["tp"].isel(latitude=lat_idx, longitude=lon_idx).values
            mucape = ds_main["mucape"].isel(latitude=lat_idx, longitude=lon_idx).values
            
            u10 = ds_wind["u10"].isel(latitude=lat_idx, longitude=lon_idx).values
            v10 = ds_wind["v10"].isel(latitude=lat_idx, longitude=lon_idx).values
            z_model = ds_orog["z"].isel(latitude=lat_idx, longitude=lon_idx).values / 9.81  # ✅ Z surface orography
            
            # 🆕 PRINT ALTITUDINI OGNI CITTÀ (OROGRAFIA VERA!)
            z_model_scalar = float(z_model)
            print(f"📍 {city}: città {info['elev']:.0f}m, modello {z_model_scalar:.0f}m (Δ{z_model_scalar-info['elev']:+.0f}m)")
            
            # CALCOLI TRIORARI
            rh2m = relative_humidity(t2m_k, td2m_k)
            t2m_c = kelvin_to_celsius(t2m_k)
            t2m_corr, pmsl_corr = altitude_correction(t2m_c, rh2m, z_model, info['elev'], msl)
            
            # 🆕 PRINT COMPLETO con T e P prima/dopo correzione (PRIMO TIMESTEP)
            t_mod_uncorr = round(float(t2m_c[0]), 1)
            p_mod_uncorr = round(float(msl[0]), 1)
            t_mod_corr = round(float(t2m_corr[0]), 1)
            p_mod_corr = round(float(pmsl_corr[0]), 1)
            
            print(f"   T {t_mod_uncorr}→{t_mod_corr}°C | P {p_mod_uncorr}→{p_mod_corr}hPa")
            
            spd_ms, wd_deg = wind_speed_direction(u10, v10)
            spd_kmh = mps_to_kmh(spd_ms)
            tp_rate = np.diff(tp_cum, prepend=tp_cum[0]) * 1000  # mm/3h
            
            # TRIORARIO CON ORA LOCALE + MU CAPE
            trihourly_data = []
            for i in range(len(t2m_corr)):
                dt_utc = ref_dt + timedelta(hours=i*3)
                dt_local = utc_to_local(dt_utc)
                weather = classify_weather(t2m_corr[i], rh2m[i], tcc[i], 
                                         tp_rate[i], spd_kmh[i], mucape[i], 
                                         season_thresh, timestep_hours=3)
                
                trihourly_data.append({
                    "d": dt_local.strftime("%Y%m%d"),
                    "h": dt_local.strftime("%H"),
                    "t": round(float(t2m_corr[i]), 1),
                    "r": round(float(rh2m[i])),
                    "p": round(float(tp_rate[i]), 1),
                    "pr": round(float(pmsl_corr[i])),
                    "v": round(float(spd_kmh[i]), 1),
                    "vd": wind_dir_to_cardinal(wd_deg[i]),
                    "w": weather
                })
            
            # GIORNALIERO
            daily_summaries = calculate_daily_summaries(trihourly_data, tcc, tp_rate, mucape, season_thresh)
            
            # JSON FINALE
            city_data = {
                "r": RUN,
                "c": city,
                "x": info['lat'],
                "y": info['lon'],
                "z": info['elev'],
                "TRIORARIO": trihourly_data,
                "GIORNALIERO": daily_summaries
            }
            
            with open(f"{outdir}/{city}_ecmwf.json", 'w', encoding='utf-8') as f:
                json.dump(city_data, f, separators=(',',':'), ensure_ascii=False)
            
            processed += 1
            if processed % 50 == 0:
                print(f"✅ {processed}/{len(venues)} città")
                
        except Exception as e:
            print(f"❌ {city}: {e}")
            continue
    
    ds_main.close(); ds_wind.close(); ds_orog.close()
    print(f"🎉 COMPLETATO {RUN}: {processed}/{len(venues)} città → {outdir}/")
    return outdir

if __name__ == "__main__":
    process_ecmwf_data()

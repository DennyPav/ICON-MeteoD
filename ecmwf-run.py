#!/bin/env python3
import os, json, numpy as np, xarray as xr
from datetime import datetime, timedelta, timezone
from ecmwf.opendata import Client
from concurrent.futures import ThreadPoolExecutor, as_completed

WORKDIR = os.getcwd()
VENUES_PATH = f"{WORKDIR}/comuni_italia.json"

# Lapse rates
LAPSE_DRY = 0.0098
LAPSE_MOIST = 0.006
G = 9.80665
RD = 287.05

# Soglie stagionali
SEASON_THRESHOLDS = {
    "winter": {"start_day":1, "end_day":80, "fog_rh":97, "haze_rh":90, "fog_wind":5, "haze_wind":15},
    "spring": {"start_day":81, "end_day":172, "fog_rh":95, "haze_rh":87, "fog_wind":7, "haze_wind":20},
    "summer": {"start_day":173, "end_day":263, "fog_rh":93, "haze_rh":83, "fog_wind":10, "haze_wind":25},
    "autumn": {"start_day":264, "end_day":365, "fog_rh":96, "haze_rh":88, "fog_wind":6, "haze_wind":18}
}

CET = timezone(timedelta(hours=1))
CEST = timezone(timedelta(hours=2))

# --------------------------- FUNZIONI ---------------------------
def utc_to_local(dt_utc):
    m, d = dt_utc.month, dt_utc.day
    if (m > 3 and m < 10) or (m == 3 and d >= 25) or (m == 10 and d <= 25):
        return dt_utc.astimezone(CEST)
    return dt_utc.astimezone(CET)

def kelvin_to_celsius(k): return k-273.15
def mps_to_kmh(mps): return mps*3.6

def wet_bulb_celsius(t_c, rh_percent):
    tw = t_c * np.arctan(0.151977 * np.sqrt(rh_percent + 8.313659)) \
         + np.arctan(t_c + rh_percent) - np.arctan(rh_percent - 1.676331) \
         + 0.00391838 * rh_percent**1.5 * np.arctan(0.023101 * rh_percent) \
         - 4.686035
    return tw

def relative_humidity(t2m_k, td2m_k):
    t_c, td_c = kelvin_to_celsius(t2m_k), kelvin_to_celsius(td2m_k)
    es = 6.112 * np.exp((17.67*t_c)/(t_c+243.5))
    e = 6.112 * np.exp((17.67*td_c)/(td_c+243.5))
    return np.clip(100*e/es, 0, 100)

def wind_speed_direction(u,v):
    speed_ms = np.sqrt(u**2 + v**2)
    deg = (np.degrees(np.arctan2(-u,-v))%360)
    return speed_ms, deg

def wind_dir_to_cardinal(deg):
    return ['N','NE','E','SE','S','SW','W','NW'][int((deg+22.5)%360//45)]

def altitude_correction(t2m, rh, z_model, z_station, pmsl):
    delta_z = z_model-z_station
    w_moist = np.clip(rh/100.0,0,1)
    lapse_t = LAPSE_DRY*(1.0-w_moist)+LAPSE_MOIST*w_moist
    t_corr = t2m + lapse_t*delta_z
    T_mean = t_corr + 273.15
    p_corr = pmsl*np.exp(-G*z_station/(RD*T_mean))
    return t_corr, p_corr

def get_season_precise(dt_utc):
    day_of_year = dt_utc.timetuple().tm_yday
    for season, thresh in SEASON_THRESHOLDS.items():
        if thresh["start_day"]<=day_of_year<=thresh["end_day"]:
            return season, thresh
    return "winter", SEASON_THRESHOLDS["winter"]

def classify_weather(t2m, rh2m, clct, tp_rate, wind_kmh, mucape, season_thresh, timestep_hours=3):
    if mucape>400 and tp_rate>0.5*timestep_hours: return "TEMPORALE"
    drizzle_min, drizzle_max = 0.09, 0.3
    if timestep_hours==3: prec_debole_min, prec_moderata_min, prec_intensa_min=0.3,5.0,20.0
    else: prec_debole_min, prec_moderata_min, prec_intensa_min=0.3,10.0,30.0
    prec_intensity=None
    if tp_rate>=prec_intensa_min: prec_intensity="INTENSA"
    elif tp_rate>=prec_moderata_min: prec_intensity="MODERATA"
    elif tp_rate>=prec_debole_min: prec_intensity="DEBOLE"
    if prec_intensity:
        octas=clct/100.0*8
        cloud_state = "SERENO" if octas<=2 else "POCO NUVOLOSO" if octas<=4 else "NUVOLOSO" if octas<=6 else "COPERTO"
        wet_bulb=wet_bulb_celsius(t2m,rh2m)
        prec_type="NEVE" if wet_bulb<0.1 else "PIOGGIA"
        return f"{cloud_state} {prec_type} {prec_intensity}"
    if drizzle_min<=tp_rate<=drizzle_max:
        if rh2m>=season_thresh["fog_rh"] and wind_kmh<=season_thresh["fog_wind"]: return "NEBBIA"
        if rh2m>=season_thresh["haze_rh"] and wind_kmh<=season_thresh["haze_wind"]: return "FOSCHIA"
        return "COPERTO PIOGGERELLA"
    if rh2m>=season_thresh["fog_rh"] and wind_kmh<=season_thresh["fog_wind"]: return "NEBBIA"
    if rh2m>=season_thresh["haze_rh"] and wind_kmh<=season_thresh["haze_wind"]: return "FOSCHIA"
    octas=clct/100.0*8
    return "SERENO" if octas<=2 else "POCO NUVOLOSO" if octas<=4 else "NUVOLOSO" if octas<=6 else "COPERTO"

def load_venues(file_path):
    with open(file_path,'r',encoding='utf-8') as f:
        data=json.load(f)
    venues={c:{"lat":float(v[0]),"lon":float(v[1]),"elev":float(v[2])} for c,v in data.items()}
    print(f"Caricate {len(venues)} città")
    return venues

def calculate_daily_summaries(trihourly_data, clct_daily, tp_rate_daily, mucape_daily, season_thresh, timestep_hours):
    daily_summaries=[]
    days_data={}
    for i,record in enumerate(trihourly_data):
        day=record["d"]
        if day not in days_data: days_data[day]=[]
        days_data[day].append((i,record))
    for day,day_records in days_data.items():
        day_temps=np.array([record["t"] for _,record in day_records])
        t_min=round(np.min(day_temps),1)
        t_max=round(np.max(day_temps),1)
        tp_total=sum([record["p"] for _,record in day_records])
        t_mean=np.mean(day_temps)
        r_mean=np.mean([record["r"] for _,record in day_records])
        v_mean=np.mean([record.get("v",5.0) for _,record in day_records])
        first_idx=day_records[0][0]
        weather_day=classify_weather(t_mean,r_mean,clct_daily[first_idx],
                                     tp_total,v_mean,mucape_daily[first_idx],
                                     season_thresh,timestep_hours=timestep_hours)
        daily_summaries.append({"d":day,"tmin":t_min,"tmax":t_max,"p":round(tp_total,1),"w":weather_day})
    return daily_summaries

# --------------------------- DOWNLOAD ---------------------------
def download_ecmwf_triorario(run_date, run_hour):
    steps_tri = list(range(0, 144, 3))
    grib_dir = f"{WORKDIR}/grib_ecmwf/{run_date}{run_hour}"
    os.makedirs(grib_dir, exist_ok=True)
    main_file = f"{grib_dir}/ecmwf_main_tri.grib"
    wind_file = f"{grib_dir}/ecmwf_wind_tri.grib"
    orog_file = f"{grib_dir}/ecmwf_orog.grib"
    client = Client(source="ecmwf", model="ifs", resol="0p25")
    if not os.path.exists(main_file) or os.path.getsize(main_file)<30_000_000:
        client.retrieve(date=run_date, time=int(run_hour), stream="oper", type="fc",
                        step=steps_tri, param=["2t","2d","tcc","msl","tp","mucape"], target=main_file)
    if not os.path.exists(wind_file) or os.path.getsize(wind_file)<5_000_000:
        client.retrieve(date=run_date, time=int(run_hour), stream="oper", type="fc",
                        step=steps_tri, param=["10u","10v"], target=wind_file)
    if not os.path.exists(orog_file) or os.path.getsize(orog_file)<1_000:
        client.retrieve(date=run_date, time=int(run_hour), stream="oper", type="fc",
                        step=[0], param=["z"], target=orog_file)
    return main_file, wind_file, orog_file

def download_ecmwf_esaorario(run_date, run_hour):
    steps_esa = list(range(144, 331, 6)) if run_hour=="00" else list(range(144, 319, 6))
    grib_dir = f"{WORKDIR}/grib_ecmwf/{run_date}{run_hour}"
    os.makedirs(grib_dir, exist_ok=True)
    main_file_esa = f"{grib_dir}/ecmwf_main_esa.grib"
    orog_file = f"{grib_dir}/ecmwf_orog.grib"
    client = Client(source="ecmwf", model="ifs", resol="0p25")
    if not os.path.exists(main_file_esa) or os.path.getsize(main_file_esa)<30_000_000:
        client.retrieve(date=run_date, time=int(run_hour), stream="oper", type="fc",
                        step=steps_esa, param=["2t","2d","tcc","msl","tp","mucape"], target=main_file_esa)
    if not os.path.exists(orog_file) or os.path.getsize(orog_file)<1_000:
        client.retrieve(date=run_date, time=int(run_hour), stream="oper", type="fc",
                        step=[0], param=["z"], target=orog_file)
    return main_file_esa, orog_file

# --------------------------- GET RUN ---------------------------
def get_run_datetime_now_utc():
    now = datetime.now(timezone.utc)
    if now.hour < 8:
        return (now - timedelta(days=1)).strftime("%Y%m%d"), "00"
    elif now.hour < 20:
        return now.strftime("%Y%m%d"), "00"
    return now.strftime("%Y%m%d"), "12"

# --------------------------- MAIN ---------------------------
if __name__=="__main__":
    run_date, run_hour = get_run_datetime_now_utc()
    outdir=f"{WORKDIR}/json_ecmwf/{run_date}{run_hour}"
    os.makedirs(outdir,exist_ok=True)
    venues=load_venues(VENUES_PATH)

    # Download
    main_tri_file, wind_tri_file, orog_file = download_ecmwf_triorario(run_date, run_hour)
    main_esa_file, _ = download_ecmwf_esaorario(run_date, run_hour)

    # ------------------- OPEN DATASET CON CHUNK PER 12GB -------------------
    max_ram_gb = 12
    approx_bytes_per_cell = 8  # float64
    # stimiamo numero di celle massime: 12GB / 8bytes ≈ 1.5e9
    # lat/lon per chunk
    chunk_size_lat = 50
    chunk_size_lon = 50
    ds_main_tri = xr.open_dataset(main_tri_file, engine="cfgrib", chunks={"latitude":chunk_size_lat,"longitude":chunk_size_lon})
    ds_wind_tri = xr.open_dataset(wind_tri_file, engine="cfgrib", chunks={"latitude":chunk_size_lat,"longitude":chunk_size_lon})
    ds_main_esa = xr.open_dataset(main_esa_file, engine="cfgrib", chunks={"latitude":chunk_size_lat,"longitude":chunk_size_lon})
    ds_orog = xr.open_dataset(orog_file, engine="cfgrib", chunks={"latitude":chunk_size_lat,"longitude":chunk_size_lon})

    ref_dt=datetime.strptime(run_date+run_hour,"%Y%m%d%H").replace(tzinfo=timezone.utc)
    season, season_thresh = get_season_precise(ref_dt)

    # Parallel processing
    max_workers=8
    from functools import partial
    from concurrent.futures import ThreadPoolExecutor, as_completed

    process_func = partial(process_city, ds_main_tri=ds_main_tri, ds_wind_tri=ds_wind_tri,
                           ds_main_esa=ds_main_esa, ds_orog=ds_orog,
                           ref_dt=ref_dt, season_thresh=season_thresh, outdir=outdir)

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures=[executor.submit(process_func, city, info) for city,info in venues.items()]
        for f in as_completed(futures): pass

    ds_main_tri.close()
    ds_wind_tri.close()
    ds_main_esa.close()
    ds_orog.close()

    print("Elaborazione completata per tutte le città")

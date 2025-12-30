#!/bin/env python3
import os
from datetime import datetime, timedelta, timezone
from ecmwf.opendata import Client

WORKDIR = os.getcwd()

def get_run_datetime_now_utc():
    from datetime import datetime, timezone, timedelta
    now = datetime.now(timezone.utc)
    if now.hour < 8:
        return (now - timedelta(days=1)).strftime("%Y%m%d"), "00"
    elif now.hour < 20:
        return now.strftime("%Y%m%d"), "00"
    return now.strftime("%Y%m%d"), "12"

def download_ecmwf_triorario(run_date, run_hour):
    steps_tri = list(range(0, 144, 3))
    grib_dir = f"{WORKDIR}/grib_ecmwf/{run_date}{run_hour}"
    os.makedirs(grib_dir, exist_ok=True)
    main_file = f"{grib_dir}/ecmwf_main_tri.grib"
    wind_file = f"{grib_dir}/ecmwf_wind_tri.grib"
    orog_file = f"{grib_dir}/ecmwf_orog.grib"

    # Controllo esistenza file
    if os.path.exists(main_file) and os.path.getsize(main_file) > 30_000_000 \
       and os.path.exists(wind_file) and os.path.getsize(wind_file) > 5_000_000 \
       and os.path.exists(orog_file) and os.path.getsize(orog_file) > 1_000:
        print(f"GRIB triorario già presenti in {grib_dir}, salto download")
        return main_file, wind_file, orog_file

    client = Client(source="ecmwf", model="ifs", resol="0p25")
    print("⏬ Download ECMWF triorario...")
    client.retrieve(date=run_date, time=int(run_hour), stream="oper", type="fc",
                    step=steps_tri, param=["2t","2d","tcc","msl","tp","mucape"], target=main_file)
    client.retrieve(date=run_date, time=int(run_hour), stream="oper", type="fc",
                    step=steps_tri, param=["10u","10v"], target=wind_file)
    client.retrieve(date=run_date, time=int(run_hour), stream="oper", type="fc",
                    step=[0], param=["z"], target=orog_file)
    return main_file, wind_file, orog_file

def download_ecmwf_esaorario(run_date, run_hour):
    steps_esa = list(range(144, 331, 6)) if run_hour=="00" else list(range(144, 319, 6))
    grib_dir = f"{WORKDIR}/grib_ecmwf/{run_date}{run_hour}"
    os.makedirs(grib_dir, exist_ok=True)
    main_file_esa = f"{grib_dir}/ecmwf_main_esa.grib"
    orog_file = f"{grib_dir}/ecmwf_orog.grib"

    if os.path.exists(main_file_esa) and os.path.getsize(main_file_esa) > 30_000_000 \
       and os.path.exists(orog_file) and os.path.getsize(orog_file) > 1_000:
        print(f"GRIB esaorario già presenti in {grib_dir}, salto download")
        return main_file_esa, orog_file

    client = Client(source="ecmwf", model="ifs", resol="0p25")
    print("⏬ Download ECMWF esaorario...")
    client.retrieve(date=run_date, time=int(run_hour), stream="oper", type="fc",
                    step=steps_esa, param=["2t","2d","tcc","msl","tp","mucape"], target=main_file_esa)
    client.retrieve(date=run_date, time=int(run_hour), stream="oper", type="fc",
                    step=[0], param=["z"], target=orog_file)
    return main_file_esa, orog_file

if __name__=="__main__":
    run_date, run_hour = get_run_datetime_now_utc()
    print(f"RUN ECMWF {run_date}{run_hour}")
    download_ecmwf_triorario(run_date, run_hour)
    download_ecmwf_esaorario(run_date, run_hour)
    print("Download completato ✅")

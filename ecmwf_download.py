#!/bin/env python3
import os
from datetime import datetime, timedelta, timezone
from ecmwf.opendata import Client

WORKDIR = os.getcwd()

# =========================
# RUN TIME LOGIC
# =========================
def get_run_datetime_now_utc():
    now = datetime.now(timezone.utc)
    if now.hour < 8:
        return (now - timedelta(days=1)).strftime("%Y%m%d"), "00"
    elif now.hour < 20:
        return now.strftime("%Y%m%d"), "00"
    return now.strftime("%Y%m%d"), "12"

# =========================
# UTILS
# =========================
def file_ok(path, min_size):
    return os.path.exists(path) and os.path.getsize(path) > min_size

def safe_retrieve(client, target, min_size, **kwargs):
    if file_ok(target, min_size):
        print(f"✔ File già valido: {os.path.basename(target)}")
        return
    if os.path.exists(target):
        print(f"⚠ File incompleto, rimuovo: {os.path.basename(target)}")
        os.remove(target)

    print(f"⬇ Download {os.path.basename(target)}")
    client.retrieve(target=target, **kwargs)

    if not file_ok(target, min_size):
        raise RuntimeError(f"❌ Download fallito: {target}")

# =========================
# MAIN DOWNLOAD
# =========================
def main():
    run_date, run_hour = get_run_datetime_now_utc()
    RUN = f"{run_date}{run_hour}"

    grib_dir = f"{WORKDIR}/grib_ecmwf/{RUN}"
    os.makedirs(grib_dir, exist_ok=True)

    print(f"\n📥 ECMWF DOWNLOAD RUN {RUN}\n")

    client = Client(
        source="ecmwf",
        model="ifs",
        resol="0p25"
    )

    # ------------------------
    # STEP DEFINITIONS
    # ------------------------
    steps_tri = list(range(0, 144, 3))
    steps_esa = list(range(144, 331, 6)) if run_hour == "00" else list(range(144, 319, 6))

    # ------------------------
    # FILE PATHS
    # ------------------------
    main_tri = f"{grib_dir}/ecmwf_main_tri.grib"
    wind_tri = f"{grib_dir}/ecmwf_wind_tri.grib"
    main_esa = f"{grib_dir}/ecmwf_main_esa.grib"
    orog = f"{grib_dir}/ecmwf_orog.grib"

    # ------------------------
    # DOWNLOAD TRIORARIO
    # ------------------------
    safe_retrieve(
        client,
        target=main_tri,
        min_size=30_000_000,
        date=run_date,
        time=int(run_hour),
        stream="oper",
        type="fc",
        step=steps_tri,
        param=["2t", "2d", "tcc", "msl", "tp", "mucape"]
    )

    safe_retrieve(
        client,
        target=wind_tri,
        min_size=5_000_000,
        date=run_date,
        time=int(run_hour),
        stream="oper",
        type="fc",
        step=steps_tri,
        param=["10u", "10v"]
    )

    # ------------------------
    # DOWNLOAD ESAORARIO
    # ------------------------
    safe_retrieve(
        client,
        target=main_esa,
        min_size=30_000_000,
        date=run_date,
        time=int(run_hour),
        stream="oper",
        type="fc",
        step=steps_esa,
        param=["2t", "2d", "tcc", "msl", "tp", "mucape"]
    )

    # ------------------------
    # OROGRAFIA (UNA SOLA VOLTA)
    # ------------------------
    safe_retrieve(
        client,
        target=orog,
        min_size=1_000,
        date=run_date,
        time=int(run_hour),
        stream="oper",
        type="fc",
        step=[0],
        param=["z"]
    )

    print(f"\n✅ DOWNLOAD COMPLETATO — {RUN}")
    print(f"📂 {grib_dir}\n")

if __name__ == "__main__":
    main()

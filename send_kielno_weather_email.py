#!/usr/bin/env python3
from __future__ import annotations

import os
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from datetime import datetime
from typing import Dict, Tuple, List
from zoneinfo import ZoneInfo

import pandas as pd
import requests

# =========================
# KONFIG
# =========================

LOC_NAME = "Kielno (gm. Szemud)"
LOCATIONS: Dict[str, Tuple[float, float]] = {
    "Kielno (gm. Szemud)": (54.45278, 18.33750),
}

MODELS: Dict[str, str] = {
    "forecast_auto": "https://api.open-meteo.com/v1/forecast",
    "gfs": "https://api.open-meteo.com/v1/gfs",
    "dwd_icon": "https://api.open-meteo.com/v1/dwd-icon",
    "meteofrance": "https://api.open-meteo.com/v1/meteofrance",
    "jma": "https://api.open-meteo.com/v1/jma",
    "gem": "https://api.open-meteo.com/v1/gem",
}

DAYPARTS = {
    "noc": range(0, 6),
    "rano": range(6, 12),
    "poludnie": range(12, 18),
    "wieczor": range(18, 24),
}

DAYPART_LABELS = {
    "noc": "nocą",
    "rano": "rano",
    "poludnie": "w południe",
    "wieczor": "wieczorem",
}

SNOW_THRESHOLD_MM = 0.1

TZ = "Europe/Warsaw"


# =========================
# LOGIKA POGODY
# =========================

def is_fog(code: int) -> bool:
    return code in (45, 48)


def classify_precip(rain_mm: float, snow_mm: float) -> str:
    rain_mm = rain_mm or 0.0
    snow_mm = snow_mm or 0.0
    if rain_mm < 0.1 and snow_mm < 0.1:
        return "brak"
    if snow_mm >= 0.1 and rain_mm < 0.1:
        return "snieg"
    if rain_mm >= 0.1 and snow_mm < 0.1:
        return "deszcz"
    return "mieszane"


def fetch_hourly(
    lat: float,
    lon: float,
    days: int = 7,
    timezone: str = TZ,
    base_url: str = "https://api.open-meteo.com/v1/forecast",
) -> pd.DataFrame:
    params = {
        "latitude": lat,
        "longitude": lon,
        "timezone": timezone,
        "forecast_days": days,
        "hourly": ",".join([
            "temperature_2m",
            "precipitation",
            "snowfall",
            "windspeed_10m",
            "windgusts_10m",
            "weathercode",
            "visibility",
        ]),
    }
    r = requests.get(base_url, params=params, timeout=30)
    r.raise_for_status()
    j = r.json()

    if "hourly" not in j or "time" not in j["hourly"]:
        raise ValueError("Brak pola 'hourly' w odpowiedzi API")

    df = pd.DataFrame(j["hourly"])
    df["time"] = pd.to_datetime(df["time"])
    df["date"] = df["time"].dt.date.astype(str)
    df["hour"] = df["time"].dt.hour
    return df


def summarize_dayparts(df: pd.DataFrame, place: str, model: str) -> pd.DataFrame:
    rows = []
    for date, dfd in df.groupby("date"):
        for part, hours in DAYPARTS.items():
            sub = dfd[dfd["hour"].isin(hours)]
            if sub.empty:
                continue

            temp = sub["temperature_2m"].mean()
            wind = sub["windspeed_10m"].mean()
            gust = sub["windgusts_10m"].max()
            rain = sub["precipitation"].sum()
            snow = sub["snowfall"].sum()
            fog = any(is_fog(int(x)) for x in sub["weathercode"].fillna(0))
            vis_min = sub["visibility"].min()

            rows.append({
                "miejsce": place,
                "model": model,
                "data": date,
                "pora_dnia": part,
                "temp_C_srednia": float(temp) if pd.notna(temp) else None,
                "wiatr_km_h_sredni": float(wind) if pd.notna(wind) else None,
                "porywy_km_h_max": float(gust) if pd.notna(gust) else None,
                "opad_mm_suma": float(rain) if pd.notna(rain) else 0.0,
                "snieg_suma": float(snow) if pd.notna(snow) else 0.0,
                "typ_opadu": classify_precip(float(rain or 0.0), float(snow or 0.0)),
                "mgla": bool(fog),
                "widzialnosc_min_m": None if pd.isna(vis_min) else int(vis_min),
            })

    out = pd.DataFrame(rows)
    if out.empty:
        return out

    order = pd.Categorical(
        out["pora_dnia"],
        ["noc", "rano", "poludnie", "wieczor"],
        ordered=True
    )
    out = out.assign(pora_dnia=order).sort_values(
        ["miejsce", "model", "data", "pora_dnia"]
    )
    return out


def majority(series: pd.Series):
    s = series.dropna()
    if s.empty:
        return None
    return s.value_counts().index[0]


def average_across_models(df_models: pd.DataFrame) -> pd.DataFrame:
    if df_models.empty:
        return df_models

    keys = ["miejsce", "data", "pora_dnia"]

    numeric_cols = [
        "temp_C_srednia",
        "wiatr_km_h_sredni",
        "porywy_km_h_max",
        "opad_mm_suma",
        "snieg_suma",
    ]

    avg_num = df_models.groupby(keys, observed=False)[numeric_cols].mean().reset_index()

    extras = (
        df_models.groupby(keys, observed=False)
        .agg({
            "mgla": majority,
            "typ_opadu": majority,
            "widzialnosc_min_m": "min",
            "model": lambda s: ", ".join(sorted(set(s.dropna().astype(str)))),
        })
        .reset_index()
        .rename(columns={"model": "modele_uzyte"})
    )

    counts = (
        df_models.groupby(keys, observed=False)["model"]
        .nunique()
        .reset_index()
        .rename(columns={"model": "liczba_modeli"})
    )

    def snow_models_count(s: pd.Series) -> int:
        s = s.fillna(0.0)
        return int((s >= SNOW_THRESHOLD_MM).sum())

    def snow_models_pct(s: pd.Series) -> float:
        s = s.fillna(0.0)
        if len(s) == 0:
            return 0.0
        return round(100.0 * float((s >= SNOW_THRESHOLD_MM).mean()), 0)

    snow_stats = (
        df_models.groupby(keys, observed=False)["snieg_suma"]
        .agg([
            ("snieg_min_mm", "min"),
            ("snieg_max_mm", "max"),
            ("snieg_p90_mm", lambda x: float(x.fillna(0.0).quantile(0.9))),
            ("snieg_modele_ile", snow_models_count),
            ("snieg_modele_pct", snow_models_pct),
        ])
        .reset_index()
    )

    out = (
        avg_num
        .merge(extras, on=keys)
        .merge(counts, on=keys)
        .merge(snow_stats, on=keys)
    )

    for col in numeric_cols:
        out[col] = out[col].round(1)
    out["snieg_min_mm"] = out["snieg_min_mm"].round(1)
    out["snieg_max_mm"] = out["snieg_max_mm"].round(1)
    out["snieg_p90_mm"] = out["snieg_p90_mm"].round(1)

    order = pd.Categorical(
        out["pora_dnia"],
        ["noc", "rano", "poludnie", "wieczor"],
        ordered=True
    )
    out = out.assign(pora_dnia=order).sort_values(["miejsce", "data", "pora_dnia"])
    return out


# =========================
# PODSUMOWANIE DO MAILA
# =========================

def polish_date(date_str: str) -> str:
    dt = datetime.strptime(date_str, "%Y-%m-%d")
    weekdays = [
        "poniedziałek", "wtorek", "środa", "czwartek",
        "piątek", "sobota", "niedziela"
    ]
    months = [
        "", "stycznia", "lutego", "marca", "kwietnia", "maja", "czerwca",
        "lipca", "sierpnia", "września", "października", "listopada", "grudnia"
    ]
    return f"{weekdays[dt.weekday()]}, {dt.day} {months[dt.month]}"


def format_slot(row: pd.Series) -> str:
    return f"{polish_date(row['data'])} {DAYPART_LABELS.get(str(row['pora_dnia']), str(row['pora_dnia']))}"


def build_summary(df_loc: pd.DataFrame) -> str:
    if df_loc.empty:
        return "Brak danych dla Kielna."

    today = datetime.now(ZoneInfo(TZ)).date().isoformat()
    next_days = df_loc[df_loc["data"] >= today].copy()

    if next_days.empty:
        next_days = df_loc.copy()

    lines: List[str] = []
    lines.append("Najważniejsze dla Kielna na najbliższe dni:")

    # 1) Najmocniejszy wiatr
    wind_row = next_days.sort_values("porywy_km_h_max", ascending=False).iloc[0]
    lines.append(
        f"- Najsilniejszy wiatr: {format_slot(wind_row)} "
        f"– średni wiatr {wind_row['wiatr_km_h_sredni']:.1f} km/h, "
        f"porywy do {wind_row['porywy_km_h_max']:.1f} km/h."
    )

    # 2) Największy opad
    rain_candidates = next_days[next_days["opad_mm_suma"] >= 0.1]
    if not rain_candidates.empty:
        rain_row = rain_candidates.sort_values("opad_mm_suma", ascending=False).iloc[0]
        lines.append(
            f"- Największa szansa na mokrą pogodę: {format_slot(rain_row)} "
            f"– suma opadu około {rain_row['opad_mm_suma']:.1f} mm."
        )
    else:
        lines.append("- Większych opadów na razie nie widać.")

    # 3) Najcieplej / najchłodniej
    warm_row = next_days.sort_values("temp_C_srednia", ascending=False).iloc[0]
    cold_row = next_days.sort_values("temp_C_srednia", ascending=True).iloc[0]
    lines.append(
        f"- Najcieplej zapowiada się {format_slot(warm_row)} "
        f"– około {warm_row['temp_C_srednia']:.1f}°C."
    )
    lines.append(
        f"- Najchłodniej zapowiada się {format_slot(cold_row)} "
        f"– około {cold_row['temp_C_srednia']:.1f}°C."
    )

    # 4) Śnieg / mieszane
    snow_rows = next_days[next_days["snieg_modele_ile"] > 0]
    if not snow_rows.empty:
        snow_row = snow_rows.sort_values(
            ["snieg_modele_ile", "snieg_max_mm"], ascending=False
        ).iloc[0]
        lines.append(
            f"- Marginalny sygnał śniegu: {format_slot(snow_row)} "
            f"– widzi to {int(snow_row['snieg_modele_ile'])}/{int(snow_row['liczba_modeli'])} modeli."
        )
    else:
        lines.append("- Na ten moment modele praktycznie nie pokazują śniegu.")

    # 5) Słaba widzialność
    low_vis = next_days[next_days["widzialnosc_min_m"].notna()].sort_values("widzialnosc_min_m")
    if not low_vis.empty:
        vis_row = low_vis.iloc[0]
        if int(vis_row["widzialnosc_min_m"]) <= 3000:
            lines.append(
                f"- Uwaga na gorszą widzialność: {format_slot(vis_row)} "
                f"– minimum około {int(vis_row['widzialnosc_min_m'])} m."
            )

    # 6) Krótkie zdanie ogólne
    strong_wind_count = int((next_days["porywy_km_h_max"] >= 60).sum())
    rainy_count = int((next_days["opad_mm_suma"] >= 0.3).sum())

    if strong_wind_count >= 2:
        lines.append("- Ogólnie: najbardziej odczuwalny temat w prognozie to wiatr.")
    elif rainy_count >= 2:
        lines.append("- Ogólnie: prognoza wygląda na dość zmienną i okresami mokrą.")
    else:
        lines.append("- Ogólnie: bez pogodowego dramatu, ale warto obserwować wiatr i krótkie epizody opadów.")

    return "\n".join(lines)


def build_email_text(df_loc: pd.DataFrame) -> str:
    updated_at = datetime.now(ZoneInfo(TZ)).strftime("%Y-%m-%d %H:%M %Z")
    summary = build_summary(df_loc)

    preview = df_loc.copy()
    preview = preview[[
        "data", "pora_dnia", "temp_C_srednia", "wiatr_km_h_sredni",
        "porywy_km_h_max", "opad_mm_suma", "typ_opadu",
        "widzialnosc_min_m", "liczba_modeli"
    ]].head(12)

    table_txt = preview.to_string(index=False)

    return f"""Raport pogody dla: {LOC_NAME}
Aktualizacja: {updated_at}

{summary}

Najbliższe wpisy z raportu:
{table_txt}

Źródło: Open-Meteo, średnia z modeli: {", ".join(MODELS.keys())}
"""


# =========================
# SMTP
# =========================

def send_email(subject: str, body: str) -> None:
    smtp_host = os.environ["SMTP_HOST"]
    smtp_port = int(os.environ.get("SMTP_PORT", "587"))
    smtp_user = os.environ["SMTP_USER"]
    smtp_password = os.environ["SMTP_PASSWORD"]
    mail_from = os.environ.get("MAIL_FROM", smtp_user)
    mail_to = os.environ["MAIL_TO"]

    msg = MIMEMultipart()
    msg["From"] = mail_from
    msg["To"] = mail_to
    msg["Subject"] = subject
    msg.attach(MIMEText(body, "plain", "utf-8"))

    with smtplib.SMTP(smtp_host, smtp_port, timeout=30) as server:
        server.starttls()
        server.login(smtp_user, smtp_password)
        server.sendmail(mail_from, [x.strip() for x in mail_to.split(",")], msg.as_string())


# =========================
# MAIN
# =========================

def main() -> None:
    place, (lat, lon) = next(iter(LOCATIONS.items()))
    all_models_parts: List[pd.DataFrame] = []

    for model_name, base_url in MODELS.items():
        try:
            hourly = fetch_hourly(lat, lon, days=7, base_url=base_url)
            s = summarize_dayparts(hourly, place, model_name)
            if s is not None and not s.empty:
                all_models_parts.append(s)
            print(f"[OK] {place} / {model_name}")
        except Exception as e:
            print(f"[POMIJAM] {place} / {model_name}: {e}")

    if not all_models_parts:
        raise SystemExit("Nie udało się pobrać danych z żadnego modelu.")

    df_models = pd.concat(all_models_parts, ignore_index=True)
    df = average_across_models(df_models)
    df_loc = df[df["miejsce"] == LOC_NAME].copy()

    subject_date = datetime.now(ZoneInfo(TZ)).strftime("%Y-%m-%d")
    subject = f"Pogoda – Kielno – {subject_date}"
    body = build_email_text(df_loc)

    print(body)
    send_email(subject, body)
    print("[OK] Mail wysłany.")


if __name__ == "__main__":
    main()

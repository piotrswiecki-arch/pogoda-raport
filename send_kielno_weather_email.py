#!/usr/bin/env python3
from __future__ import annotations

import os
import smtplib
from datetime import datetime
from email.mime.application import MIMEApplication
from email.mime.image import MIMEImage
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from pathlib import Path
from typing import Dict, Tuple, List
from zoneinfo import ZoneInfo

import matplotlib.pyplot as plt
import pandas as pd
import requests

# =========================
# KONFIG
# =========================

TZ = "Europe/Warsaw"
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

DAYPART_ORDER = ["noc", "rano", "poludnie", "wieczor"]
DAYPART_LABELS = {
    "noc": "noc",
    "rano": "rano",
    "poludnie": "południe",
    "wieczor": "wieczór",
}

SNOW_THRESHOLD_MM = 0.1


# =========================
# DANE
# =========================

def is_fog(code: int) -> bool:
    return code in (45, 48)


def classify_precip(rain_mm: float, snow_mm: float) -> str:
    rain_mm = rain_mm or 0.0
    snow_mm = snow_mm or 0.0
    if rain_mm < 0.1 and snow_mm < 0.1:
        return "brak"
    if snow_mm >= 0.1 and rain_mm < 0.1:
        return "śnieg"
    if rain_mm >= 0.1 and snow_mm < 0.1:
        return "deszcz"
    return "mieszane"


def fetch_hourly(
    lat: float,
    lon: float,
    days: int = 3,
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

    out["pora_dnia"] = pd.Categorical(out["pora_dnia"], DAYPART_ORDER, ordered=True)
    out = out.sort_values(["miejsce", "model", "data", "pora_dnia"])
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

    snow_stats = (
        df_models.groupby(keys, observed=False)["snieg_suma"]
        .agg([
            ("snieg_min_mm", "min"),
            ("snieg_max_mm", "max"),
            ("snieg_modele_ile", snow_models_count),
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
    out["pora_dnia"] = pd.Categorical(out["pora_dnia"], DAYPART_ORDER, ordered=True)
    out = out.sort_values(["miejsce", "data", "pora_dnia"])
    return out


# =========================
# FILTR: dziś + jutro
# =========================

def filter_today_tomorrow(df: pd.DataFrame) -> pd.DataFrame:
    today = datetime.now(ZoneInfo(TZ)).date()
    tomorrow = today + pd.Timedelta(days=1)
    allowed = {today.isoformat(), tomorrow.isoformat()}
    out = df[df["data"].isin(allowed)].copy()
    out["pora_dnia"] = pd.Categorical(out["pora_dnia"], DAYPART_ORDER, ordered=True)
    out = out.sort_values(["data", "pora_dnia"])
    return out


# =========================
# PODSUMOWANIE
# =========================

def format_polish_date(date_str: str) -> str:
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


def build_summary(df: pd.DataFrame) -> str:
    if df.empty:
        return "Brak danych dla dzisiaj i jutra."

    lines: List[str] = []

    for date_value, day_df in df.groupby("data", observed=False):
        day_df = day_df.sort_values("pora_dnia")
        avg_temp = day_df["temp_C_srednia"].mean()
        max_gust = day_df["porywy_km_h_max"].max()
        total_rain = day_df["opad_mm_suma"].sum()
        min_vis = day_df["widzialnosc_min_m"].dropna().min() if day_df["widzialnosc_min_m"].notna().any() else None

        text = (
            f"<li><b>{format_polish_date(date_value)}</b>: "
            f"średnia temperatura około <b>{avg_temp:.1f}°C</b>, "
            f"maksymalne porywy do <b>{max_gust:.1f} km/h</b>, "
            f"suma opadu około <b>{total_rain:.1f} mm</b>"
        )

        if min_vis is not None and min_vis <= 3000:
            text += f", możliwe okresowo słabsze warunki widzialności (do ok. <b>{int(min_vis)} m</b>)"

        text += ".</li>"
        lines.append(text)

    top_wind = df.sort_values("porywy_km_h_max", ascending=False).iloc[0]
    top_rain = df.sort_values("opad_mm_suma", ascending=False).iloc[0]

    extra = (
        f"<p><b>Najważniejsze okno pogodowe:</b> "
        f"najmocniej może powiać <b>{format_polish_date(top_wind['data'])}</b> "
        f"w porze <b>{DAYPART_LABELS[str(top_wind['pora_dnia'])]}</b> "
        f"(porywy do <b>{top_wind['porywy_km_h_max']:.1f} km/h</b>).</p>"
    )

    if float(top_rain["opad_mm_suma"]) >= 0.1:
        extra += (
            f"<p><b>Największa szansa na opad:</b> "
            f"<b>{format_polish_date(top_rain['data'])}</b> "
            f"w porze <b>{DAYPART_LABELS[str(top_rain['pora_dnia'])]}</b> "
            f"(około <b>{top_rain['opad_mm_suma']:.1f} mm</b>).</p>"
        )

    return "<ul>" + "".join(lines) + "</ul>" + extra


# =========================
# WYKRESY
# =========================

def add_x_labels(df_plot: pd.DataFrame) -> pd.DataFrame:
    out = df_plot.copy()
    out["x_label"] = out.apply(
        lambda r: f"{r['data'][5:]}\n{DAYPART_LABELS[str(r['pora_dnia'])]}",
        axis=1
    )
    return out


def save_temp_chart(df: pd.DataFrame, path: str) -> None:
    d = add_x_labels(df)
    plt.figure(figsize=(10, 4.5))
    plt.plot(d["x_label"], d["temp_C_srednia"], marker="o")
    plt.title("Kielno – temperatura (dziś i jutro)")
    plt.ylabel("°C")
    plt.xticks(rotation=0)
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()


def save_wind_chart(df: pd.DataFrame, path: str) -> None:
    d = add_x_labels(df)
    plt.figure(figsize=(10, 4.5))
    plt.plot(d["x_label"], d["wiatr_km_h_sredni"], marker="o", label="wiatr średni")
    plt.plot(d["x_label"], d["porywy_km_h_max"], marker="o", label="porywy max")
    plt.title("Kielno – wiatr i porywy (dziś i jutro)")
    plt.ylabel("km/h")
    plt.xticks(rotation=0)
    plt.legend()
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()


def save_rain_chart(df: pd.DataFrame, path: str) -> None:
    d = add_x_labels(df)
    plt.figure(figsize=(10, 4.5))
    plt.bar(d["x_label"], d["opad_mm_suma"])
    plt.title("Kielno – opad (dziś i jutro)")
    plt.ylabel("mm")
    plt.xticks(rotation=0)
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()


# =========================
# TABELA HTML
# =========================

def build_table_html(df: pd.DataFrame) -> str:
    tbl = df.copy()
    tbl["pora_dnia"] = tbl["pora_dnia"].astype(str).map(DAYPART_LABELS)
    tbl["data"] = tbl["data"].map(format_polish_date)

    tbl = tbl[[
        "data",
        "pora_dnia",
        "temp_C_srednia",
        "wiatr_km_h_sredni",
        "porywy_km_h_max",
        "opad_mm_suma",
        "typ_opadu",
        "widzialnosc_min_m",
    ]]

    tbl = tbl.rename(columns={
        "data": "Dzień",
        "pora_dnia": "Pora dnia",
        "temp_C_srednia": "Temp. °C",
        "wiatr_km_h_sredni": "Wiatr km/h",
        "porywy_km_h_max": "Porywy km/h",
        "opad_mm_suma": "Opad mm",
        "typ_opadu": "Typ opadu",
        "widzialnosc_min_m": "Widzialność min m",
    })

    return tbl.to_html(index=False, border=0, justify="left")


# =========================
# MAIL
# =========================

def send_email(subject: str, html_body: str, attachments: List[Tuple[str, str]], csv_path: str | None = None) -> None:
    smtp_host = os.environ["SMTP_HOST"]
    smtp_port = int(os.environ.get("SMTP_PORT", "587"))
    smtp_user = os.environ["SMTP_USER"]
    smtp_password = os.environ["SMTP_PASSWORD"]

    mail_from = os.environ["MAIL_FROM"]
    mail_to = os.environ["MAIL_TO"]

    msg = MIMEMultipart("related")
    msg["Subject"] = subject
    msg["From"] = mail_from
    msg["To"] = mail_to

    alt = MIMEMultipart("alternative")
    msg.attach(alt)
    alt.attach(MIMEText("Raport HTML nie został wyświetlony.", "plain", "utf-8"))
    alt.attach(MIMEText(html_body, "html", "utf-8"))

    for cid, filepath in attachments:
        with open(filepath, "rb") as f:
            img = MIMEImage(f.read())
            img.add_header("Content-ID", f"<{cid}>")
            img.add_header("Content-Disposition", "inline", filename=Path(filepath).name)
            msg.attach(img)

    if csv_path:
        with open(csv_path, "rb") as f:
            part = MIMEApplication(f.read(), Name=Path(csv_path).name)
            part["Content-Disposition"] = f'attachment; filename="{Path(csv_path).name}"'
            msg.attach(part)

    with smtplib.SMTP(smtp_host, smtp_port, timeout=30) as server:
        server.starttls()
        server.login(smtp_user, smtp_password)
        server.sendmail(mail_from, [x.strip() for x in mail_to.split(",")], msg.as_string())


def build_html_email(df: pd.DataFrame, updated_at: str) -> str:
    summary_html = build_summary(df)
    table_html = build_table_html(df)

    return f"""
    <html>
      <body style="font-family: Arial, sans-serif; color: #222; line-height: 1.5;">
        <h2>Raport pogody – Kielno</h2>
        <p><b>Zakres:</b> dzisiaj i jutro<br>
           <b>Aktualizacja:</b> {updated_at}</p>

        {summary_html}

        <h3>Wykres temperatury</h3>
        <img src="cid:temp_chart" style="max-width: 100%; border: 1px solid #ddd; border-radius: 8px;">

        <h3>Wykres wiatru i porywów</h3>
        <img src="cid:wind_chart" style="max-width: 100%; border: 1px solid #ddd; border-radius: 8px;">

        <h3>Wykres opadu</h3>
        <img src="cid:rain_chart" style="max-width: 100%; border: 1px solid #ddd; border-radius: 8px;">

        <h3>Tabela szczegółowa</h3>
        {table_html}

        <p style="font-size: 12px; color: #666;">
          Dane: Open-Meteo, średnia z modeli: {", ".join(MODELS.keys())}
        </p>
      </body>
    </html>
    """


# =========================
# MAIN
# =========================

def main() -> None:
    place, (lat, lon) = next(iter(LOCATIONS.items()))
    all_models_parts: List[pd.DataFrame] = []

    for model_name, base_url in MODELS.items():
        try:
            hourly = fetch_hourly(lat, lon, days=3, base_url=base_url)
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
    df_mail = filter_today_tomorrow(df_loc)

    if df_mail.empty:
        raise SystemExit("Brak danych dla dziś i jutra.")

    out_dir = Path("mail_output")
    out_dir.mkdir(exist_ok=True)

    temp_chart = str(out_dir / "temp.png")
    wind_chart = str(out_dir / "wind.png")
    rain_chart = str(out_dir / "rain.png")
    csv_path = str(out_dir / "kielno_dzis_jutro.csv")

    save_temp_chart(df_mail, temp_chart)
    save_wind_chart(df_mail, wind_chart)
    save_rain_chart(df_mail, rain_chart)
    df_mail.to_csv(csv_path, index=False)

    updated_at = datetime.now(ZoneInfo(TZ)).strftime("%Y-%m-%d %H:%M %Z")
    subject = f"Pogoda Kielno – dziś i jutro – {datetime.now(ZoneInfo(TZ)).strftime('%Y-%m-%d')}"
    html_body = build_html_email(df_mail, updated_at)

    send_email(
        subject=subject,
        html_body=html_body,
        attachments=[
            ("temp_chart", temp_chart),
            ("wind_chart", wind_chart),
            ("rain_chart", rain_chart),
        ],
        csv_path=csv_path,
    )

    print("[OK] Mail wysłany.")


if __name__ == "__main__":
    main()

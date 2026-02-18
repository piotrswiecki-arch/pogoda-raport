#!/usr/bin/env python3
import webbrowser
from pathlib import Path
from typing import Dict, Tuple, List
from datetime import datetime
from zoneinfo import ZoneInfo

import pandas as pd
import plotly.express as px
import requests

# --- Lokacje ---
LOCATIONS: Dict[str, Tuple[float, float]] = {
    "Warszawa": (52.2297, 21.0122),
    "Kielno (gm. Szemud)": (54.45278, 18.33750),
    "Górowo Iławeckie": (54.28534, 20.49296),
}

# --- Modele / źródła prognozy (Open-Meteo) ---
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

SNOW_THRESHOLD_MM = 0.1  # kiedy uznajemy, że „model widzi śnieg” w danej porze dnia


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
    timezone: str = "Europe/Warsaw",
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

    order = pd.Categorical(out["pora_dnia"], ["noc", "rano", "poludnie", "wieczor"], ordered=True)
    out = out.assign(pora_dnia=order).sort_values(["miejsce", "model", "data", "pora_dnia"])
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

    # 1) Średnie liczbowe po modelach
    avg_num = df_models.groupby(keys, observed=False)[numeric_cols].mean().reset_index()

    # 2) Majority + min widzialność + lista modeli
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

    # 3) Liczba modeli
    counts = (
        df_models.groupby(keys, observed=False)["model"]
        .nunique()
        .reset_index()
        .rename(columns={"model": "liczba_modeli"})
    )

    # 4) Statystyki śniegu pomiędzy modelami (bardziej „szczegółowo”)
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

    # Zaokrąglenia
    for col in numeric_cols:
        out[col] = out[col].round(1)
    out["snieg_min_mm"] = out["snieg_min_mm"].round(1)
    out["snieg_max_mm"] = out["snieg_max_mm"].round(1)
    out["snieg_p90_mm"] = out["snieg_p90_mm"].round(1)

    # Porządek pór dnia
    order = pd.Categorical(out["pora_dnia"], ["noc", "rano", "poludnie", "wieczor"], ordered=True)
    out = out.assign(pora_dnia=order).sort_values(["miejsce", "data", "pora_dnia"])
    return out


def make_section(df_loc: pd.DataFrame, loc_name: str) -> str:
    fig_t = px.line(
        df_loc, x="data", y="temp_C_srednia", color="pora_dnia",
        markers=True, title=f"Temperatura (średnia z modeli) – {loc_name}"
    )
    fig_w = px.line(
        df_loc, x="data", y="wiatr_km_h_sredni", color="pora_dnia",
        markers=True, title=f"Wiatr (średnia z modeli) – {loc_name}"
    )
    fig_g = px.line(
        df_loc, x="data", y="porywy_km_h_max", color="pora_dnia",
        markers=True, title=f"Porywy (max, średnia z modeli) – {loc_name}"
    )
    fig_p = px.bar(
        df_loc, x="data", y="opad_mm_suma", color="pora_dnia",
        title=f"Opad (suma, średnia z modeli) – {loc_name}"
    )

    tbl = df_loc.copy()
    tbl["mgla"] = tbl["mgla"].map(lambda x: "TAK" if x else "")
    tbl["snieg_modele"] = tbl.apply(
        lambda r: f'{int(r["snieg_modele_ile"])}/{int(r["liczba_modeli"])} ({int(r["snieg_modele_pct"])}%)',
        axis=1
    )

    # Tabela: dodajemy szczegóły o śniegu
    tbl = tbl[[
        "data", "pora_dnia",
        "temp_C_srednia", "wiatr_km_h_sredni", "porywy_km_h_max",
        "opad_mm_suma",
        "snieg_suma", "snieg_modele", "snieg_min_mm", "snieg_p90_mm", "snieg_max_mm",
        "typ_opadu", "mgla", "widzialnosc_min_m",
        "liczba_modeli"
    ]]

    tbl = tbl.rename(columns={
        "snieg_suma": "snieg_mm_srednia",
        "snieg_min_mm": "snieg_mm_min",
        "snieg_p90_mm": "snieg_mm_p90",
        "snieg_max_mm": "snieg_mm_max",
    })

    table_html = tbl.to_html(index=False, classes="tbl", border=0)

    return f"""
    <section class="card" id="{loc_name}">
      <h2>{loc_name}</h2>
      <div class="grid">
        <div class="plot">{fig_t.to_html(full_html=False, include_plotlyjs=False)}</div>
        <div class="plot">{fig_w.to_html(full_html=False, include_plotlyjs=False)}</div>
        <div class="plot">{fig_g.to_html(full_html=False, include_plotlyjs=False)}</div>
        <div class="plot">{fig_p.to_html(full_html=False, include_plotlyjs=False)}</div>
      </div>
      <h3>Tabela 7 dni × 4 pory dnia (średnia z modeli + szczegóły śniegu)</h3>
      <div class="tablewrap">{table_html}</div>
      <p class="muted">
        „snieg_modele” = ile modeli przewiduje śnieg ≥ {SNOW_THRESHOLD_MM} mm w danej porze dnia.
        „p90” = poziom, którego 90% modeli nie przekracza (pomaga ocenić ryzyko „górki”).
      </p>
    </section>
    """


def main():
    updated_at = datetime.now(ZoneInfo("Europe/Warsaw"))
    updated_at_str = updated_at.strftime("%Y-%m-%d %H:%M %Z")

    all_models_parts: List[pd.DataFrame] = []

    # 1) Pobierz i policz pory dnia osobno dla każdego modelu
    for place, (lat, lon) in LOCATIONS.items():
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

    # 2) HTML
    sections = []
    nav_links = []
    for loc in LOCATIONS.keys():
        df_loc = df[df["miejsce"] == loc].copy()
        sections.append(make_section(df_loc, loc))
        nav_links.append(f'<a href="#{loc}">{loc}</a>')

    html = f"""<!doctype html>
<html lang="pl">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width,initial-scale=1" />
  <title>Raport pogody – 7 dni (średnia z modeli)</title>
  <script src="https://cdn.plot.ly/plotly-2.30.0.min.js"></script>
  <style>
    body {{ font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Arial, sans-serif; margin: 0; background: #f6f7fb; color: #111; }}
    header {{ position: sticky; top: 0; background: white; padding: 14px 18px; border-bottom: 1px solid #e7e7ee; z-index: 10; }}
    header h1 {{ margin: 0 0 6px 0; font-size: 18px; }}
    nav a {{ margin-right: 12px; text-decoration: none; padding: 6px 10px; border-radius: 10px; background: #f0f1f6; color: #111; display: inline-block; }}
    .wrap {{ max-width: 1250px; margin: 0 auto; padding: 14px; }}
    .card {{ background: white; border-radius: 16px; padding: 16px; margin: 14px 0; box-shadow: 0 6px 24px rgba(0,0,0,0.06); }}
    .grid {{ display: grid; grid-template-columns: 1fr; gap: 12px; }}
    @media (min-width: 980px) {{ .grid {{ grid-template-columns: 1fr 1fr; }} }}
    .plot {{ background: #fff; border: 1px solid #efeff6; border-radius: 14px; padding: 8px; }}
    .tablewrap {{ overflow-x: auto; border: 1px solid #efeff6; border-radius: 14px; }}
    table.tbl {{ border-collapse: collapse; width: 100%; font-size: 13px; }}
    table.tbl th, table.tbl td {{ padding: 9px 10px; border-bottom: 1px solid #efeff6; text-align: left; white-space: nowrap; }}
    table.tbl th {{ position: sticky; top: 0; background: #fafafe; }}
    .muted {{ color: #666; font-size: 12px; margin-top: 8px; }}
    footer {{ padding: 18px; color: #666; font-size: 12px; }}
    code {{ background: #f0f1f6; padding: 2px 6px; border-radius: 8px; }}
  </style>
</head>
<body>
  <header>
    <div class="wrap">
      <h1>Raport pogody (7 dni × 4 pory dnia) – średnia z modeli</h1>
      <nav>{"".join(nav_links)}</nav>
      <div class="muted">
        Ostatnia aktualizacja: <b>{updated_at_str}</b> · Modele próbowane: {", ".join(MODELS.keys())}
      </div>
    </div>
  </header>

  <main class="wrap">
    {"".join(sections)}
  </main>

  <footer class="wrap">
    Dane: Open-Meteo (wiele modeli); agregacja na pory dnia i uśrednianie w tym raporcie.
  </footer>
</body>
</html>
"""

    out = Path("raport_pogoda.html")
    out.write_text(html, encoding="utf-8")
    df.to_csv("pogoda_7dni_4pory.csv", index=False)

    print(f"Zapisano: {out.resolve()}")
    print("Zapisano: pogoda_7dni_4pory.csv")

    # Lokalnie spróbuj otworzyć w przeglądarce, ale na GitHub/CI po prostu to zignoruje.
    try:
        webbrowser.open(out.resolve().as_uri())
    except Exception:
        pass


if __name__ == "__main__":
    main()

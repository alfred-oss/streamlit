from pathlib import Path
import json
from urllib.parse import urlencode
from urllib.request import Request, urlopen

import numpy as np
import pandas as pd
import streamlit as st

import pydeck as pdk

st.set_page_config(page_title="Buy vs Rent Explorer", layout="wide")


DATASET_NAMES = [
    "merged",
    "merged_by_bed",
    "df",
    "own_town_bed",
    "town_med",
]


@st.cache_data(show_spinner=False)
def load_any_dataset(base_name: str) -> pd.DataFrame | None:
    candidates = [
        Path("data") / f"{base_name}.parquet",
        Path("data") / f"{base_name}.csv",
        Path("data") / f"{base_name}.xlsx",
        Path("data") / f"{base_name}.pkl",
        Path(f"{base_name}.parquet"),
        Path(f"{base_name}.csv"),
        Path(f"{base_name}.xlsx"),
        Path(f"{base_name}.pkl"),
    ]

    for path in candidates:
        if not path.exists():
            continue
        if path.suffix == ".parquet":
            return pd.read_parquet(path)
        if path.suffix == ".csv":
            return pd.read_csv(path)
        if path.suffix == ".xlsx":
            return pd.read_excel(path)
        if path.suffix == ".pkl":
            return pd.read_pickle(path)

    return None


@st.cache_data(show_spinner=False)
def load_all() -> dict[str, pd.DataFrame]:
    loaded = {}
    for name in DATASET_NAMES:
        df = load_any_dataset(name)
        if df is not None and not df.empty:
            loaded[name] = df.copy()
    return loaded


def normalize_town(s: pd.Series) -> pd.Series:
    return (
        s.astype(str)
        .str.strip()
        .str.replace(r"\s+", " ", regex=True)
        .str.lower()
    )


def extract_zip_from_text(s: pd.Series) -> pd.Series:
    raw = (
        s.astype(str)
        .str.replace(r"\.0$", "", regex=True)
        .str.extract(r"(\d{4,5})(?:-\d{4})?", expand=False)
    )
    return raw.where(raw.isna(), raw.str.zfill(5))


def build_full_address(df: pd.DataFrame) -> pd.Series:
    addr_col = next((c for c in df.columns if c.lower() in {"address_norm", "fulladdress", "address"}), None)
    town_col = next((c for c in df.columns if c.lower() == "town"), None)
    zip_col = next((c for c in df.columns if c.lower() in {"zip_code", "zip", "zipcode"}), None)

    if addr_col is None and town_col is None and zip_col is None:
        return pd.Series(index=df.index, dtype="object")

    address = df[addr_col].fillna("").astype(str).str.strip() if addr_col else pd.Series("", index=df.index)
    town = df[town_col].fillna("").astype(str).str.strip() if town_col else pd.Series("", index=df.index)

    if zip_col:
        zip_code = extract_zip_from_text(df[zip_col]).fillna("").astype(str).str.strip()
    elif addr_col:
        zip_code = extract_zip_from_text(df[addr_col]).fillna("").astype(str).str.strip()
    else:
        zip_code = pd.Series("", index=df.index)

    combined = pd.concat(
        [address, town, zip_code, pd.Series("USA", index=df.index)],
        axis=1,
    )

    return combined.apply(
        lambda row: ", ".join([part for part in row.tolist() if part and str(part).strip()]),
        axis=1,
    )


@st.cache_data(show_spinner=False)
def geocode_queries(queries: tuple[str, ...]) -> pd.DataFrame:
    if not queries:
        return pd.DataFrame(columns=["query", "lat", "lon"])

    rows = []
    for query in queries:
        lat = np.nan
        lon = np.nan
        try:
            params = urlencode({"q": query, "format": "json", "limit": 1})
            req = Request(
                f"https://nominatim.openstreetmap.org/search?{params}",
                headers={"User-Agent": "buy-rent-streamlit/1.0"},
            )
            with urlopen(req, timeout=12) as resp:
                data = json.loads(resp.read().decode("utf-8"))
            if data:
                lat = float(data[0].get("lat"))
                lon = float(data[0].get("lon"))
        except Exception:
            pass
        rows.append({"query": query, "lat": lat, "lon": lon})

    return pd.DataFrame(rows).dropna(subset=["lat", "lon"]).drop_duplicates(subset=["query"])


@st.cache_data(show_spinner=False)
def geocode_full_addresses(full_addresses: tuple[str, ...]) -> pd.DataFrame:
    geocoded = geocode_queries(full_addresses)
    if geocoded.empty:
        return pd.DataFrame(columns=["full_address", "lat", "lon"])

    return geocoded.rename(columns={"query": "full_address"})[["full_address", "lat", "lon"]]


def infer_town_coords(ownership_df: pd.DataFrame, rent_df: pd.DataFrame | None = None) -> pd.DataFrame:
    if ownership_df is None or ownership_df.empty:
        return pd.DataFrame(columns=["Town", "lat", "lon"])

    town_col = next((c for c in ownership_df.columns if c.lower() == "town"), None)
    zip_col = next((c for c in ownership_df.columns if c.lower() in {"zip_code", "zip", "zipcode"}), None)

    if town_col is None:
        return pd.DataFrame(columns=["Town", "lat", "lon"])

    if zip_col is not None:
        ref = ownership_df[[town_col, zip_col]].copy()
        ref.columns = ["Town", "ZIP"]
        ref["ZIP"] = extract_zip_from_text(ref["ZIP"])
    else:
        ref = pd.DataFrame({"Town": ownership_df[town_col], "ZIP": pd.NA})

    # Backup: add ZIP from each dataset directly (if present), then from full addresses.
    for df in [ownership_df, rent_df]:
        if df is None or df.empty:
            continue

        town_candidate = next((c for c in df.columns if c.lower() == "town"), None)
        zip_candidate = next((c for c in df.columns if c.lower() in {"zip_code", "zip", "zipcode"}), None)
        if zip_candidate is not None:
            extra_zip = pd.DataFrame({
                "Town": df[town_candidate] if town_candidate else "",
                "ZIP": extract_zip_from_text(df[zip_candidate]),
            })
            ref = pd.concat([ref, extra_zip], ignore_index=True)

        addr_col = next((c for c in df.columns if c.lower() in {"address_norm", "fulladdress", "address"}), None)
        if addr_col is None:
            continue

        extra_addr = pd.DataFrame({
            "Town": df[town_candidate] if town_candidate else "",
            "ZIP": extract_zip_from_text(df[addr_col]),
        })
        ref = pd.concat([ref, extra_addr], ignore_index=True)
    ref = ref.dropna(subset=["ZIP"])
    ref["Town"] = ref["Town"].fillna("").astype(str).str.strip()

    agg = pd.DataFrame(columns=["town_key", "Town", "lat", "lon"])
    if not ref.empty:
        try:
            import pgeocode

            nomi = pgeocode.Nominatim("us")
            coords = nomi.query_postal_code(ref["ZIP"].tolist())[["latitude", "longitude"]]
            ref = ref.reset_index(drop=True)
            ref["lat"] = coords["latitude"].values
            ref["lon"] = coords["longitude"].values
            ref = ref.dropna(subset=["lat", "lon"])
            if not ref.empty:
                ref = ref[ref["Town"] != ""]
                ref["town_key"] = normalize_town(ref["Town"])
                agg = ref.groupby("town_key", as_index=False).agg(
                    Town=("Town", "first"),
                    lat=("lat", "median"),
                    lon=("lon", "median"),
                )
        except Exception:
            pass

    # Online fallback by town string (works even if ZIP-based mapping is empty).
    known_keys = set(agg["town_key"]) if not agg.empty else set()
    raw_towns = pd.Series(dtype="object")
    if town_col in ownership_df.columns:
        raw_towns = pd.concat([raw_towns, ownership_df[town_col]], ignore_index=True)
    if rent_df is not None:
        rent_town_col = next((c for c in rent_df.columns if c.lower() == "town"), None)
        if rent_town_col:
            raw_towns = pd.concat([raw_towns, rent_df[rent_town_col]], ignore_index=True)

    missing = (
        pd.DataFrame({"Town": raw_towns.dropna().astype(str).str.strip().unique()})
        .query("Town != ''")
    )
    missing["town_key"] = normalize_town(missing["Town"])
    missing = missing[~missing["town_key"].isin(known_keys)]

    if not missing.empty:
        missing["query"] = missing["Town"].astype(str).str.strip() + ", USA"
        geocoded = geocode_queries(tuple(sorted(missing["query"].dropna().unique())))
        if not geocoded.empty:
            missing = missing.merge(geocoded, on="query", how="left")
            missing = missing.dropna(subset=["lat", "lon"])
            if not missing.empty:
                agg = pd.concat(
                    [agg, missing[["town_key", "Town", "lat", "lon"]]],
                    ignore_index=True,
                )

    return agg


def add_coords(df: pd.DataFrame, town_coord_ref: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()

    lat_col = next((c for c in out.columns if c.lower() in {"lat", "latitude"}), None)
    lon_col = next((c for c in out.columns if c.lower() in {"lon", "lng", "longitude"}), None)

    if lat_col and lon_col:
        out["lat"] = pd.to_numeric(out[lat_col], errors="coerce")
        out["lon"] = pd.to_numeric(out[lon_col], errors="coerce")
    else:
        out["lat"] = np.nan
        out["lon"] = np.nan

    town_col = next((c for c in out.columns if c.lower() == "town"), None)
    if town_col and not town_coord_ref.empty:
        out["town_key"] = normalize_town(out[town_col])
        out = out.merge(town_coord_ref[["town_key", "lat", "lon"]], on="town_key", how="left", suffixes=("", "_town"))
        out["lat"] = out["lat"].fillna(out["lat_town"])
        out["lon"] = out["lon"].fillna(out["lon_town"])
        out = out.drop(columns=["lat_town", "lon_town"], errors="ignore")

    missing_mask = out["lat"].isna() | out["lon"].isna()
    if not missing_mask.any():
        return out

    # ZIP fallback for unresolved rows (faster/more stable than full-address geocoding).
    zip_col = next((c for c in out.columns if c.lower() in {"zip_code", "zip", "zipcode"}), None)
    if zip_col is not None:
        zip_vals = extract_zip_from_text(out[zip_col])
        rows_need_zip = missing_mask & zip_vals.notna()
        if rows_need_zip.any():
            try:
                import pgeocode

                nomi = pgeocode.Nominatim("us")
                unique_zips = tuple(sorted(zip_vals[rows_need_zip].dropna().astype(str).unique()))
                zip_coords = nomi.query_postal_code(list(unique_zips))[["postal_code", "latitude", "longitude"]]
                zip_coords = zip_coords.rename(columns={"postal_code": "ZIP", "latitude": "lat_zip", "longitude": "lon_zip"})
                out["ZIP"] = zip_vals
                out = out.merge(zip_coords, on="ZIP", how="left")
                out["lat"] = out["lat"].fillna(out["lat_zip"])
                out["lon"] = out["lon"].fillna(out["lon_zip"])
                out = out.drop(columns=["ZIP", "lat_zip", "lon_zip"], errors="ignore")
            except Exception:
                pass

    missing_mask = out["lat"].isna() | out["lon"].isna()
    if not missing_mask.any():
        return out

    full_address = build_full_address(out)
    full_address = full_address.where(full_address.str.len() > 0, pd.NA)
    out["full_address"] = full_address

    to_geocode = tuple(sorted(out.loc[missing_mask & out["full_address"].notna(), "full_address"].dropna().astype(str).unique()))
    if not to_geocode:
        return out.drop(columns=["full_address"], errors="ignore")

    geocoded = geocode_full_addresses(to_geocode)
    if geocoded.empty:
        return out.drop(columns=["full_address"], errors="ignore")

    out = out.merge(geocoded, on="full_address", how="left", suffixes=("", "_geo"))
    out["lat"] = out["lat"].fillna(out["lat_geo"])
    out["lon"] = out["lon"].fillna(out["lon_geo"])

    return out.drop(columns=["full_address", "lat_geo", "lon_geo"], errors="ignore")


def make_map(df: pd.DataFrame, value_col: str, label_col: str, title: str, *, fixed_gap_colors: bool = False, marker_color: list[int] | None = None):
    map_df = df.dropna(subset=["lat", "lon", value_col]).copy()
    if map_df.empty:
        st.warning("No coordinates available for this map. Add ZIP codes in addresses or a ZIP column to infer locations.")
        return

    if marker_color is not None:
        def color_scale(v: float):
            return marker_color
    elif fixed_gap_colors:
        def color_scale(v: float):
            if v > 0:
                return [16, 185, 129, 180]  # green
            if v >= -200:
                return [132, 204, 22, 180]  # light green
            if v >= -400:
                return [245, 158, 11, 180]  # orange
            return [220, 38, 38, 180]  # red
    else:
        q = np.nanquantile(map_df[value_col], [0.1, 0.5, 0.9])
        lo, mid, hi = float(q[0]), float(q[1]), float(q[2])

        def color_scale(v: float):
            if v >= hi:
                return [16, 185, 129, 180]
            if v >= mid:
                return [132, 204, 22, 170]
            if v >= lo:
                return [245, 158, 11, 170]
            return [220, 38, 38, 180]

    map_df["color"] = map_df[value_col].apply(color_scale)
    radius_base = 5000 if map_df[label_col].nunique() < 200 else 2500

    st.subheader(title)
    if pdk is None:
        st.warning(
            "pydeck is not installed in this Python environment. "
            "Install it with `pip install pydeck` for interactive maps. "
            "Showing coordinate preview table instead."
        )
        preview_cols = [c for c in [label_col, value_col, "lat", "lon"] if c in map_df.columns]
        st.dataframe(
            map_df[preview_cols].sort_values(value_col, ascending=False),
            use_container_width=True,
            height=350,
        )
        return

    layer = pdk.Layer(
        "ScatterplotLayer",
        map_df,
        get_position="[lon, lat]",
        get_fill_color="color",
        get_radius=10,
        radius_units="pixels",
        radius_min_pixels=5,
        radius_max_pixels=12,
        pickable=True,
        stroked=True,
        filled=True,
    )

    view_state = pdk.ViewState(
        latitude=float(map_df["lat"].median()),
        longitude=float(map_df["lon"].median()),
        zoom=7,
        pitch=0,
    )

    tooltip = {
        "html": f"<b>{{{label_col}}}</b><br/>{value_col}: {{{value_col}}}",
        "style": {"backgroundColor": "#111827", "color": "white"},
    }
    st.pydeck_chart(pdk.Deck(layers=[layer], initial_view_state=view_state, tooltip=tooltip))


def choose_column(df: pd.DataFrame, options: list[str]) -> str | None:
    low_to_original = {c.lower(): c for c in df.columns}
    for opt in options:
        if opt.lower() in low_to_original:
            return low_to_original[opt.lower()]
    return None

def show_table(df: pd.DataFrame, *, sort_by: str | None = None, ascending: bool = False, height: int = 500):
    out = df.copy()
    if sort_by and sort_by in out.columns:
        out = out.sort_values(sort_by, ascending=ascending)

    out = out.reset_index(drop=True)

    # hide_index работает в новых версиях streamlit
    try:
        st.dataframe(out, use_container_width=True, height=height, hide_index=True)
    except TypeError:
        st.dataframe(out, use_container_width=True, height=height)

st.title("🏠 Buy vs Rent Interactive Explorer")
loaded = load_all()

if not loaded:
    st.error(
        "No datasets were found. Put csv/parquet/xlsx/pkl files into `data/` or project root using names: "
        + ", ".join(DATASET_NAMES)
    )
    st.stop()

st.caption("Main metric: gap = median_rent_per_bed - median_ownership_per_bed. Positive gap means buying is cheaper than renting per bedroom.")

ownership_base = loaded.get("df")
coords_seed_df = ownership_base if ownership_base is not None else loaded.get("merged")
town_coord_ref = infer_town_coords(coords_seed_df, loaded.get("merged")) if coords_seed_df is not None else pd.DataFrame()

# --- Section 1: main heatmap ---
st.header("1) Heatmap: where buying is better than renting")
main_source_name = st.selectbox(
    "Heatmap source",
    options=[n for n in ["merged", "merged_by_bed"] if n in loaded],
)
main_df = loaded[main_source_name].copy()

if "Bdrs" in main_df.columns:
    bdrs_values = sorted(main_df["Bdrs"].dropna().unique().tolist())
    selected_bdrs = st.multiselect("Bedroom filter (Bdrs)", bdrs_values, default=bdrs_values)
    main_df = main_df[main_df["Bdrs"].isin(selected_bdrs)]

main_df = add_coords(main_df, town_coord_ref)

gap_col = choose_column(main_df, ["gap"])
town_col = choose_column(main_df, ["Town"])
if not gap_col or not town_col:
    st.error("merged/merged_by_bed must include Town and gap columns.")
else:
    c1, c2 = st.columns([2, 1])
    with c1:
        total_points = len(main_df)
        mapped_points = int(main_df[["lat", "lon"]].notna().all(axis=1).sum())
        st.caption(f"Mapped towns: {mapped_points}/{total_points}")
        make_map(main_df, gap_col, town_col, "Town gap map", fixed_gap_colors=True)
    with c2:
        st.subheader("Town table")
        table_cols = [town_col]
        for col in ["Bdrs", "median_ownership_per_bed", "median_rent_per_bed", "gap"]:
            real_col = choose_column(main_df, [col])
            if real_col:
                table_cols.append(real_col)
        table_df = main_df[table_cols].copy()
        table_df = table_df.rename(columns={
            choose_column(table_df, ["median_ownership_per_bed"]) or "": "Median Ownership",
            choose_column(table_df, ["median_rent_per_bed"]) or "": "Median Rent",
            choose_column(table_df, ["gap"]) or "": "Difference",
        })
        sort_col = "Difference" if "Difference" in table_df.columns else gap_col
        show_table(table_df, sort_by=sort_col, ascending=False, height=500)

# --- Section 2: ownership map ---
st.header("2) Ownership map + address table")
own_df = loaded.get("df")
if own_df is not None:
    own_df = add_coords(own_df, town_coord_ref)
    own_val_col = choose_column(own_df, ["Monthly_ownership_per_bed", "monthly_ownership_per_bed"])
    own_addr_col = choose_column(own_df, ["Address_norm", "FullAddress", "address", "Address"])
    own_town_col = choose_column(own_df, ["Town"])
    own_bdrs_col = choose_column(own_df, ["Bdrs", "Beds", "Bedrooms"])

    if own_val_col and own_addr_col and own_town_col:
        own_df["map_label"] = (
            own_df[own_addr_col].fillna("").astype(str).str.strip()
            + ", "
            + own_df[own_town_col].fillna("").astype(str).str.strip()
            + ", USA"
        )
        c1, c2 = st.columns([2, 1])
        with c1:
            total_points = len(own_df)
            mapped_points = int(own_df[["lat", "lon"]].notna().all(axis=1).sum())
            st.caption(f"Mapped towns: {mapped_points}/{total_points}")
            make_map(own_df, own_val_col, "map_label", "Ownership per bed (address-level)", marker_color=[59, 130, 246, 170])
        with c2:
            st.subheader("Ownership table")
            cols = [c for c in [own_addr_col, own_town_col, own_bdrs_col, own_val_col] if c]
            table_df = own_df[cols].copy()
            table_df = table_df.rename(columns={
                own_addr_col: "Address",
                own_val_col: "Monthly Ownership",
            })
            sort_col = "Monthly Ownership" if "Monthly Ownership" in table_df.columns else own_val_col
            show_table(table_df, sort_by=sort_col, ascending=False, height=500)
    else:
        st.info("df must include address and monthly ownership per bed columns.")

st.divider()
st.markdown(
    """
**Heatmap color logic (gap):**
- Green: gap > 0
- Light green: gap from -200 to 0
- Orange: gap from -400 to -200
- Red: gap < -400
"""
)

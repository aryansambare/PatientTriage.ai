"""NHAMCS Emergency Department public-use file ingest.

NHAMCS is the CDC's National Hospital Ambulatory Medical Care Survey. The ED subfile
is a nationally representative sample of US emergency department visits, released as
a fixed-width ASCII file with a Stata dictionary describing the column positions. It
is fully public — no credentialing, no data use agreement — which is why the project
starts here while MIMIC-IV-ED access is being arranged.

What it gives us:
  * real triage vital signs, with real missingness patterns
  * a real assigned acuity (IMMEDR), on the same five-level, 1-is-most-acute scale
  * real outcomes: death in the department, admission, admission to critical care
  * left-without-being-seen, which is the trace a patient leaves when the wait beat them

What it does not give us:
  * free-text chief complaints (see `rules/rvc.py`)
  * anything resembling a timeline within the visit, so no waiting-room re-scoring
  * repeat visits by the same patient — each row is an independent visit

Survey design note: NHAMCS is a stratified multistage probability sample. PATWT is the
patient visit weight, CSTRATM the stratum and CPSUM the PSU. Those are carried through
untouched. They matter for any statement about national rates; they are deliberately
*not* used as training weights, because the model is being fitted to discriminate
within a department, not to reproduce national totals.
"""

from __future__ import annotations

import calendar
import io
import re
import zipfile
from dataclasses import dataclass
from datetime import date
from pathlib import Path

import pandas as pd

from patienttriage.data.quality import QualityReport, clean
from patienttriage.features.schema import TriageSnapshot

DATA_URL = "https://ftp.cdc.gov/pub/Health_Statistics/NCHS/Datasets/NHAMCS"
DICT_URL = "https://ftp.cdc.gov/pub/Health_Statistics/NCHS/dataset_documentation/nhamcs/stata"

_DCT_LINE = re.compile(r"^\s*(?:byte|int|long|float|double|str\d*)\s+(\w+)\s+(\d+)(?:-(\d+))?\s*$")

# Every negative value in NHAMCS is a sentinel: -9 blank, -8 unknown, -7 not applicable.
# All three mean "we do not know this", which is exactly what NaN means downstream.
SENTINEL_CEILING = 0

COLUMNS_USED: tuple[str, ...] = (
    "VMONTH", "VDAYR", "ARRTIME", "WAITTIME", "LOV",
    "AGE", "AGEDAYS", "SEX", "ARREMS",
    "TEMPF", "PULSE", "RESPR", "BPSYS", "BPDIAS", "POPCT", "PAINSCALE",
    "IMMEDR", "RFV1", "RFV2", "RFV3",
    "LWBS", "DOA", "DIEDED", "ADMITHOS", "OBSHOS", "TRANOTH", "ADMIT", "BOARDED",
    "PATWT", "CSTRATM", "CPSUM", "HOSPCODE", "PATCODE",
)


@dataclass(frozen=True)
class Column:
    name: str
    start: int  # zero-based, inclusive
    end: int  # zero-based, exclusive


def parse_dictionary(path: Path) -> list[Column]:
    """Column layout from the Stata `infix dictionary` file that ships with the data."""
    columns: list[Column] = []
    for line in path.read_text(encoding="latin-1").splitlines():
        match = _DCT_LINE.match(line)
        if match:
            name, start, end = match.groups()
            columns.append(Column(name, int(start) - 1, int(end) if end else int(start)))
    if not columns:
        raise ValueError(f"No column definitions found in {path}")
    return columns


def load_raw(year: int, data_dir: Path) -> pd.DataFrame:
    """The raw fixed-width file for one survey year, as strings.

    Read as strings and converted explicitly afterwards, so a sentinel is never
    silently coerced into a plausible-looking vital sign.
    """
    archive = data_dir / f"ed{year}.zip"
    dictionary = data_dir / f"eddict{year}.dct"
    if not archive.exists() or not dictionary.exists():
        raise FileNotFoundError(
            f"Missing NHAMCS {year}. Download with:\n"
            f"  curl -o {archive} {DATA_URL}/ed{year}.zip\n"
            f"  curl -o {dictionary} {DICT_URL}/eddict{year}.dct"
        )

    columns = parse_dictionary(dictionary)
    wanted = [c for c in columns if c.name in COLUMNS_USED]
    found = {c.name for c in wanted}
    if missing := set(COLUMNS_USED) - found:
        raise ValueError(f"NHAMCS {year} is missing expected columns: {sorted(missing)}")

    with zipfile.ZipFile(archive) as z:
        payload = z.read(z.namelist()[0])

    frame = pd.read_fwf(
        io.BytesIO(payload),
        colspecs=[(c.start, c.end) for c in wanted],
        names=[c.name for c in wanted],
        dtype=str,
    )
    frame["SURVEY_YEAR"] = year
    return frame


def _numeric(series: pd.Series) -> pd.Series:
    """To numbers, with every negative sentinel turned into a genuine missing value."""
    values = pd.to_numeric(series, errors="coerce")
    return values.where(values >= SENTINEL_CEILING)


def _age_years(row: pd.Series) -> float:
    """Age in years, using AGEDAYS for infants so 'under one' is not flattened to zero."""
    days = row["AGEDAYS"]
    if pd.notna(days):
        return float(days) / 365.25
    return float(row["AGE"]) if pd.notna(row["AGE"]) else 0.0


def _arrival_timestamp(year: int, month: float, weekday_code: float, arrtime: float) -> str:
    """An ISO timestamp carrying the real hour and weekday.

    NHAMCS deliberately does not release the date of a visit — that would help identify
    the hospital. It releases the month and the day of week, which is all the model
    actually uses. So we place the visit on the first day of that month falling on that
    weekday: the hour-of-day and day-of-week features are exact, and no false precision
    about the calendar date is introduced.
    """
    if pd.isna(month) or not 1 <= int(month) <= 12:
        return ""
    month_i = int(month)

    hour, minute = 12, 0
    if pd.notna(arrtime):
        clock = int(arrtime)
        hour, minute = min(clock // 100, 23), min(clock % 100, 59)

    day = 1
    if pd.notna(weekday_code) and 1 <= int(weekday_code) <= 7:
        # NHAMCS VDAYR: 1 = Sunday .. 7 = Saturday. Python: Monday = 0 .. Sunday = 6.
        target = (int(weekday_code) - 2) % 7
        for candidate in range(1, calendar.monthrange(year, month_i)[1] + 1):
            if date(year, month_i, candidate).weekday() == target:
                day = candidate
                break

    return f"{year:04d}-{month_i:02d}-{day:02d}T{hour:02d}:{minute:02d}:00"


def _arrival_mode(arrems: float) -> str:
    if pd.isna(arrems):
        return "unknown"
    return {1.0: "ambulance", 2.0: "walk_in"}.get(float(arrems), "unknown")


def _sex(code: float) -> str:
    if pd.isna(code):
        return "unknown"
    return {1.0: "female", 2.0: "male"}.get(float(code), "unknown")


def to_snapshots(
    raw: pd.DataFrame, report: QualityReport | None = None
) -> tuple[list[TriageSnapshot], pd.DataFrame]:
    """Split a raw NHAMCS frame into triage-time snapshots and their outcome labels.

    Pass a `QualityReport` to find out how many values were discarded as
    implausible rather than absorbed silently.
    """
    numeric_cols = [c for c in raw.columns if c not in {"ARRTIME", "SURVEY_YEAR"}]
    df = raw.copy()
    for column in numeric_cols:
        df[column] = _numeric(df[column])
    df["ARRTIME"] = _numeric(df["ARRTIME"])

    year = int(raw["SURVEY_YEAR"].iloc[0])
    snapshots: list[TriageSnapshot] = []

    for index, row in df.iterrows():
        codes = tuple(
            str(int(row[c])) for c in ("RFV1", "RFV2", "RFV3") if pd.notna(row[c])
        )
        temperature_f = row["TEMPF"]
        snapshots.append(
            TriageSnapshot(
                patient_id=f"nhamcs-{year}-{index}",
                arrived_at=_arrival_timestamp(year, row["VMONTH"], row["VDAYR"], row["ARRTIME"]),
                age_years=_age_years(row),
                sex=_sex(row["SEX"]),
                arrival_mode=_arrival_mode(row["ARREMS"]),
                heart_rate=clean("heart_rate", _opt(row["PULSE"]), report),
                systolic_bp=clean("systolic_bp", _opt(row["BPSYS"]), report),
                diastolic_bp=clean("diastolic_bp", _opt(row["BPDIAS"]), report),
                respiratory_rate=clean("respiratory_rate", _opt(row["RESPR"]), report),
                spo2=clean("spo2", _opt(row["POPCT"]), report),
                # TEMPF is Fahrenheit scaled by ten: 986 is 98.6F.
                temperature_c=clean(
                    "temperature_c",
                    None if pd.isna(temperature_f) else (temperature_f / 10 - 32) * 5 / 9,
                    report,
                ),
                pain_score=clean("pain_score", _opt(row["PAINSCALE"]), report),
                chief_complaint="",
                chief_complaint_codes=codes,
            )
        )

    labels = pd.DataFrame(
        {
            "acuity": df["IMMEDR"],
            "died_in_ed": ((df["DIEDED"] == 1) | (df["DOA"] == 1)).astype("int8"),
            # ADMIT is the type of unit admitted to; 1 is a critical care unit.
            "icu_admission": (df["ADMIT"] == 1).astype("int8"),
            "admitted": ((df["ADMITHOS"] == 1) | (df["OBSHOS"] == 1)).astype("int8"),
            "left_without_being_seen": (df["LWBS"] == 1).astype("int8"),
            "wait_minutes": df["WAITTIME"],
            "length_of_visit_minutes": df["LOV"],
            "patient_weight": df["PATWT"],
            "stratum": df["CSTRATM"],
            "psu": df["CPSUM"],
            "hospital": df["HOSPCODE"],
            "survey_year": year,
        }
    )
    # The composite the deterioration model is actually trying to catch: the patient
    # who died in the department or needed critical care. Admission alone is too broad
    # — most admissions are not emergencies of sequencing.
    labels["critical_outcome"] = (
        (labels["died_in_ed"] == 1) | (labels["icu_admission"] == 1)
    ).astype("int8")

    return snapshots, labels


def _opt(value: float) -> float | None:
    return None if pd.isna(value) else float(value)


def load_cohort(
    years: list[int], data_dir: Path, report: QualityReport | None = None
) -> tuple[list[TriageSnapshot], pd.DataFrame]:
    """Snapshots and labels across several survey years, concatenated in year order."""
    all_snapshots: list[TriageSnapshot] = []
    all_labels: list[pd.DataFrame] = []
    for year in sorted(years):
        snapshots, labels = to_snapshots(load_raw(year, data_dir), report)
        all_snapshots.extend(snapshots)
        all_labels.append(labels)
    return all_snapshots, pd.concat(all_labels, ignore_index=True)


def labelled_mask(labels: pd.DataFrame) -> pd.Series:
    """Rows with a usable acuity.

    IMMEDR is blank, 0 or 7 for roughly a third of visits — the department did not
    triage, did not use a five-level scale, or the field was not recorded. Those rows
    are dropped from acuity training and kept for outcome models, where the label is
    the outcome rather than the triage decision.
    """
    return labels["acuity"].between(1, 5)

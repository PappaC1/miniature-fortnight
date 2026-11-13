import io
import re
from datetime import datetime
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import streamlit as st
from pypdf import PdfReader


# ==========================================
# CONFIGURATION
# ==========================================

REQUIRED_FORM_COLUMNS = {
    "horse",
    "date",
    "course",
    "distance_m",
    "finish_position",
    "runners",
}

OPTIONAL_FORM_COLUMNS = {
    "going",
    "weight_kg",
    "draw",
    "jockey",
    "trainer",
    "market_odds",
    "won",
    "placed",
}


# ==========================================
# HISTORICAL FORM ENGINE
# ==========================================


class HistoricalFormEngine:
    """
    Calculates confidence and opportunity from a horse's historical runs.

    Expected input: one row per historical runner performance.
    """

    def __init__(
        self,
        recent_runs: int = 5,
        freshness_ideal_days: int = 35,
        freshness_max_days: int = 180,
    ):
        self.recent_runs = recent_runs
        self.freshness_ideal_days = freshness_ideal_days
        self.freshness_max_days = freshness_max_days

    def calculate_runner_profile(
        self,
        horse: str,
        course: str,
        distance_m: int,
        going: str,
        jockey: str,
        trainer: str,
        market_odds: Optional[float],
        form_history: pd.DataFrame,
    ) -> Dict[str, float]:
        history = form_history[
            form_history["horse"].str.casefold() == horse.casefold()
        ].copy()

        if history.empty:
            return self._no_history_profile(market_odds)

        history["date"] = pd.to_datetime(history["date"], errors="coerce")
        history = history.sort_values("date", ascending=False)

        recent = history.head(self.recent_runs)
        recent_form = self._recent_form_score(recent)
        consistency = self._consistency_score(history)
        course_distance = self._course_distance_score(history, course, distance_m)
        going_score = self._going_score(history, going)
        freshness = self._freshness_score(history)
        jockey_score = self._jockey_score(history, jockey)
        trainer_score = self._trainer_score(history, trainer)
        market_score = self._market_score(market_odds)

        confidence = (
            recent_form * 0.35
            + consistency * 0.20
            + jockey_score * 0.15
            + trainer_score * 0.15
            + freshness * 0.15
        )

        opportunity = (
            course_distance * 0.35
            + going_score * 0.20
            + market_score * 0.25
            + recent_form * 0.20
        )

        return {
            "recent_form": round(recent_form, 1),
            "consistency": round(consistency, 1),
            "course_distance": round(course_distance, 1),
            "going": round(going_score, 1),
            "freshness": round(freshness, 1),
            "jockey": round(jockey_score, 1),
            "trainer": round(trainer_score, 1),
            "market": round(market_score, 1),
            "confidence": round(confidence, 1),
            "opportunity": round(opportunity, 1),
            "historical_runs": int(len(history)),
        }

    def _no_history_profile(self, market_odds: Optional[float]) -> Dict[str, float]:
        market_score = self._market_score(market_odds)

        return {
            "recent_form": 65.0,
            "consistency": 65.0,
            "course_distance": 65.0,
            "going": 65.0,
            "freshness": 65.0,
            "jockey": 65.0,
            "trainer": 65.0,
            "market": market_score,
            "confidence": 65.0,
            "opportunity": market_score,
            "historical_runs": 0,
        }

    def _recent_form_score(self, recent: pd.DataFrame) -> float:
        if recent.empty:
            return 65.0

        scores = []

        for _, run in recent.iterrows():
            position = self._safe_number(run.get("finish_position"))
            runners = self._safe_number(run.get("runners"))

            if position is None or runners is None or runners < 2:
                continue

            # 1st = 100, last = 0, scaled by field size.
            percentile = 1 - ((position - 1) / max(runners - 1, 1))
            scores.append(percentile * 100)

        if not scores:
            return 65.0

        # Recent runs matter more than older ones.
        weights = np.linspace(1.0, 0.5, len(scores))
        return float(np.average(scores, weights=weights))

    def _consistency_score(self, history: pd.DataFrame) -> float:
        if history.empty:
            return 65.0

        positions = history["finish_position"].apply(self._safe_number).dropna()
        runners = history["runners"].apply(self._safe_number).dropna()

        if positions.empty or runners.empty:
            return 65.0

        valid = history[
            history["finish_position"].apply(self._safe_number).notna()
            & history["runners"].apply(self._safe_number).notna()
        ].copy()

        if valid.empty:
            return 65.0

        percentiles = []

        for _, run in valid.iterrows():
            position = self._safe_number(run["finish_position"])
            field_size = self._safe_number(run["runners"])

            if position is None or field_size is None or field_size < 2:
                continue

            percentiles.append(1 - ((position - 1) / max(field_size - 1, 1)))

        if not percentiles:
            return 65.0

        return float(np.mean(percentiles) * 100)

    def _course_distance_score(
        self, history: pd.DataFrame, course: str, distance_m: int
    ) -> float:
        if not course or not distance_m:
            return 65.0

        same_course = history[
            history["course"].astype(str).str.casefold() == course.casefold()
        ]
        same_distance = history[
            history["distance_m"].apply(self._safe_number) == distance_m
        ]

        course_score = self._result_score(same_course)
        distance_score = self._result_score(same_distance)

        return (course_score * 0.4) + (distance_score * 0.6)

    def _going_score(self, history: pd.DataFrame, going: str) -> float:
        if not going:
            return 65.0

        same_going = history[
            history.get("going", pd.Series(dtype=str))
            .astype(str)
            .str.casefold()
            == going.casefold()
        ]

        return self._result_score(same_going)

    def _freshness_score(self, history: pd.DataFrame) -> float:
        if history.empty:
            return 65.0

        last_run = history["date"].dropna().max()

        if pd.isna(last_run):
            return 65.0

        days_since_run = (datetime.now() - last_run.to_pydatetime()).days

        if days_since_run < 7:
            return 75.0
        if days_since_run <= self.freshness_ideal_days:
            return 95.0
        if days_since_run > self.freshness_max_days:
            return 55.0

        decline = (days_since_run - self.freshness_ideal_days) / (
            self.freshness_max_days - self.freshness_ideal_days
        )
        return float(95.0 - (decline * 40.0))

    def _jockey_score(self, history: pd.DataFrame, jockey: str) -> float:
        return self._participant_score(history, "jockey", jockey)

    def _trainer_score(self, history: pd.DataFrame, trainer: str) -> float:
        return self._participant_score(history, "trainer", trainer)

    def _participant_score(
        self, history: pd.DataFrame, column: str, participant: str
    ) -> float:
        if not participant or column not in history.columns:
            return 65.0

        participant_runs = history[
            history[column].astype(str).str.casefold() == participant.casefold()
        ]

        if participant_runs.empty:
            return 65.0

        return self._result_score(participant_runs)

    def _result_score(self, runs: pd.DataFrame) -> float:
        if runs.empty:
            return 65.0

        positions = runs["finish_position"].apply(self._safe_number).dropna()

        if positions.empty:
            return 65.0

        wins = (positions == 1).mean()
        places = (positions <= 3).mean()

        return float((wins * 65) + (places * 35))

    def _market_score(self, market_odds: Optional[float]) -> float:
        if not market_odds or market_odds <= 1:
            return 70.0

        return float(max(50.0, min(95.0, 100.0 - (market_odds * 2.0))))

    @staticmethod
    def _safe_number(value) -> Optional[float]:
        try:
            if pd.isna(value):
                return None
            return float(value)
        except (TypeError, ValueError):
            return None


# ==========================================
# UHIS AND ODDS ENGINES
# ==========================================


class UHISEngine:
    def calculate(self, metrics: Dict[str, float]) -> float:
        if not metrics:
            return 0.0
        return round(sum(metrics.values()) / len(metrics), 2)


class ProbabilityEngine:
    def calculate(self, uhis: float, total_race_uhis: float) -> float:
        if total_race_uhis <= 0 or uhis <= 0:
            return 0.0
        return round((uhis / total_race_uhis) * 100, 2)


class FairOddsEngine:
    def calculate(self, probability: float) -> float:
        if probability <= 0:
            return 0.0
        return round(100 / probability, 2)


class OverlayEngine:
    def calculate(self, fair_odds: float, market_odds: Optional[float]) -> float:
        if not fair_odds or not market_odds or fair_odds <= 0 or market_odds <= 0:
            return 0.0
        return round(((market_odds - fair_odds) / fair_odds) * 100, 2)


class EESEngine:
    def __init__(
        self,
        uhis_weight: float = 0.55,
        confidence_weight: float = 0.25,
        opportunity_weight: float = 0.20,
    ):
        total = uhis_weight + confidence_weight + opportunity_weight
        self.uhis_weight = uhis_weight / total
        self.confidence_weight = confidence_weight / total
        self.opportunity_weight = opportunity_weight / total

    def calculate(
        self, uhis: float, confidence: float, opportunity: float
    ) -> float:
        score = (
            uhis * self.uhis_weight
            + confidence * self.confidence_weight
            + opportunity * self.opportunity_weight
        )
        return round(score, 2)


class RecommendationEngine:
    def __init__(
        self,
        best_bet_ees: float = 88.0,
        value_overlay: float = 20.0,
        place_bet_ees: float = 80.0,
    ):
        self.best_bet_ees = best_bet_ees
        self.value_overlay = value_overlay
        self.place_bet_ees = place_bet_ees

    def generate(self, ees: float, overlay: float) -> str:
        if ees >= self.best_bet_ees and overlay >= self.value_overlay:
            return "BEST VALUE BET"
        if ees >= self.best_bet_ees:
            return "BEST BET"
        if overlay >= self.value_overlay:
            return "VALUE BET"
        if ees >= self.place_bet_ees:
            return "PLACE BET"
        return "WATCH"


# ==========================================
# RACECARD PARSER
# ==========================================


class RacecardParser:
    RACE_HEADER_PATTERN = re.compile(
        r"(?i)\bRACE\s+(?P<race_number>\d{1,2})\b(?P<description>[^\n]*)"
    )

    DISTANCE_PATTERN = re.compile(r"(?i)(?P<distance>\d{3,4})\s*m\b")

    RUNNER_PATTERN = re.compile(
        r"^\s*"
        r"(?P<number>\d{1,2})\s+"
        r"(?:\((?P<draw>\d{1,2})\)\s+)?"
        r"(?P<horse>[A-Za-z][A-Za-z\s'\-\.]{1,35}?)\s+"
        r"(?P<weight>\d{1,2}(?:\.\d)?)\s+"
        r"(?P<remainder>.+?)\s*$"
    )

    ODDS_PATTERN = re.compile(
        r"^(?P<value>\d+\s*/\s*\d+|\d+(?:\.\d+)?)(?:\s*[Ff])?$"
    )

    def parse(self, text: str) -> Dict[str, List[dict]]:
        races: Dict[str, List[dict]] = {}
        headers = list(self.RACE_HEADER_PATTERN.finditer(text))

        if not headers:
            return races

        for index, header in enumerate(headers):
            race_number = header.group("race_number")
            description = header.group("description")

            distance_match = self.DISTANCE_PATTERN.search(description)
            distance = (
                int(distance_match.group("distance")) if distance_match else 0
            )

            start = header.end()
            end = (
                headers[index + 1].start()
                if index + 1 < len(headers)
                else len(text)
            )

            runners = self._parse_runners(text[start:end], distance)

            if runners:
                title = f"Race {race_number}"
                if distance:
                    title += f" - {distance}m"
                races[title] = runners

        return races

    def _parse_runners(self, race_block: str, distance: int) -> List[dict]:
        runners = []
        seen = set()

        for line in race_block.splitlines():
            match = self.RUNNER_PATTERN.match(line)

            if not match:
                continue

            horse = re.sub(r"\s+", " ", match.group("horse")).strip().title()

            if horse in seen or len(horse) < 3:
                continue

            draw = int(match.group("draw")) if match.group("draw") else 0
            weight = float(match.group("weight"))
            jockey, trainer, market_odds = self._split_details(
                match.group("remainder")
            )

            runners.append(
                {
                    "horse": horse,
                    "draw": draw,
                    "weight": weight,
                    "jockey": jockey,
                    "trainer": trainer,
                    "market_odds": market_odds,
                    "distance": distance,
                }
            )
            seen.add(horse)

        return runners

    def _split_details(
        self, remainder: str
    ) -> Tuple[str, str, Optional[float]]:
        tokens = remainder.split()
        odds = None

        if tokens and self.ODDS_PATTERN.match(tokens[-1]):
            odds = self._parse_odds(tokens[-1])
            tokens = tokens[:-1]

        if len(tokens) >= 4:
            jockey = " ".join(tokens[:-2])
            trainer = " ".join(tokens[-2:])
        elif len(tokens) == 3:
            jockey = " ".join(tokens[:2])
            trainer = tokens[-1]
        elif len(tokens) == 2:
            jockey, trainer = tokens
        elif len(tokens) == 1:
            jockey, trainer = tokens[0], ""
        else:
            jockey, trainer = "", ""

        return jockey.strip(), trainer.strip(), odds

    @staticmethod
    def _parse_odds(odds_text: str) -> Optional[float]:
        try:
            if "/" in odds_text:
                numerator, denominator = odds_text.split("/")
                return round(
                    (float(numerator) / float(denominator)) + 1.0, 2
                )

            value = float(odds_text)
            return round(value if value > 1 else value + 1, 2)
        except (ValueError, ZeroDivisionError):
            return None


# ==========================================
# ANALYSIS PIPELINE
# ==========================================


def validate_form_file(form_data: pd.DataFrame) -> List[str]:
    columns = {column.strip().lower() for column in form_data.columns}
    missing = REQUIRED_FORM_COLUMNS - columns
    return sorted(missing)


def prepare_form_data(form_data: pd.DataFrame) -> pd.DataFrame:
    form_data = form_data.copy()
    form_data.columns = [column.strip().lower() for column in form_data.columns]

    for column in ("horse", "course", "going", "jockey", "trainer"):
        if column in form_data.columns:
            form_data[column] = form_data[column].astype(str).str.strip()

    for column in ("distance_m", "finish_position", "runners", "market_odds"):
        if column in form_data.columns:
            form_data[column] = pd.to_numeric(
                form_data[column], errors="coerce"
            )

    if "date" in form_data.columns:
        form_data["date"] = pd.to_datetime(form_data["date"], errors="coerce")

    if "won" not in form_data.columns and "finish_position" in form_data.columns:
        form_data["won"] = (
            form_data["finish_position"].apply(lambda x: x == 1)
        ).astype(int)

    if "placed" not in form_data.columns and "finish_position" in form_data.columns:
        form_data["placed"] = (
            form_data["finish_position"].apply(lambda x: 1 if x <= 3 else 0)
        ).astype(int)

    return form_data


def analyze_meeting(
    extracted_text: str,
    form_data: pd.DataFrame,
    normalise_probabilities: bool = True,
) -> Dict[str, pd.DataFrame]:
    parser = RacecardParser()
    parsed_races = parser.parse(extracted_text)

    if not parsed_races:
        return {}

    form_engine = HistoricalFormEngine()
    uhis_engine = UHISEngine()
    probability_engine = ProbabilityEngine()
    fair_odds_engine = FairOddsEngine()
    overlay_engine = OverlayEngine()
    ees_engine = EESEngine()
    recommendation_engine = RecommendationEngine()

    analyzed_races = {}

    for race_name, runners in parsed_races.items():
        rows = []

        for runner in runners:
            profile = form_engine.calculate_runner_profile(
                horse=runner["horse"],
                course=st.session_state.get("race_course", ""),
                distance_m=runner["distance"],
                going=st.session_state.get("race_going", ""),
                jockey=runner["jockey"],
                trainer=runner["trainer"],
                market_odds=runner["market_odds"],
                form_history=form_data,
            )

            metrics = {
                "recent_form": profile["recent_form"],
                "consistency": profile["consistency"],
                "course_distance": profile["course_distance"],
                "going": profile["going"],
                "freshness": profile["freshness"],
                "jockey": profile["jockey"],
                "trainer": profile["trainer"],
                "market": profile["market"],
            }

            uhis = uhis_engine.calculate(metrics)

            rows.append(
                {
                    "Horse": runner["horse"],
                    "Draw": runner["draw"] or "N/A",
                    "Weight (kg)": runner["weight"],
                    "Jockey": runner["jockey"] or "N/A",
                    "Trainer": runner["trainer"] or "N/A",
                    "Market odds": runner["market_odds"] or "N/A",
                    "Historical runs": profile["historical_runs"],
                    "Recent form": profile["recent_form"],
                    "Consistency": profile["consistency"],
                    "Course/distance": profile["course_distance"],
                    "Going suitability": profile["going"],
                    "Freshness": profile["freshness"],
                    "Jockey rating": profile["jockey"],
                    "Trainer rating": profile["trainer"],
                    "UHIS": uhis,
                    "Confidence": profile["confidence"],
                    "Opportunity": profile["opportunity"],
                }
            )

        dataframe = pd.DataFrame(rows)

        if dataframe.empty:
            continue

        total_uhis = dataframe["UHIS"].sum()

        dataframe["Win probability %"] = dataframe["UHIS"].apply(
            lambda value: probability_engine.calculate(
                value, total_uhis
            )
        )

        if normalise_probabilities:
            probability_total = dataframe["Win probability %"].sum()

            if probability_total > 0:
                dataframe["Win probability %"] = (
                    dataframe["Win probability %"] / probability_total * 100
                ).round(2)

        dataframe["Fair odds"] = dataframe["Win probability %"].apply(
            fair_odds_engine.calculate
        )

        dataframe["Overlay %"] = dataframe.apply(
            lambda row: overlay_engine.calculate(
                row["Fair odds"],
                row["Market odds"] if isinstance(row["Market odds"], float) else None,
            ),
            axis=1,
        )

        dataframe["EES"] = dataframe.apply(
            lambda row: ees_engine.calculate(
                row["UHIS"], row["Confidence"], row["Opportunity"]
            ),
            axis=1,
        )

        dataframe["Recommendation"] = dataframe.apply(
            lambda row: recommendation_engine.generate(
                row["EES"], row["Overlay %"]
            ),
            axis=1,
        )

        analyzed_races[race_name] = dataframe.sort_values(
            "EES", ascending=False
        ).reset_index(drop=True)

    return analyzed_races


def extract_text_from_pdf(uploaded_file) -> str:
    reader = PdfReader(io.BytesIO(uploaded_file.getvalue()))
    return "\n".join(page.extract_text() or "" for page in reader.pages)


# ==========================================
# STREAMLIT APP
# ==========================================


st.set_page_config(
    page_title="Horse Racing Prediction Assistant",
    page_icon="🐎",
    layout="wide",
)

st.title("🐎 Horse Racing Prediction Assistant")
st.caption(
    "Upload a racecard PDF and a historical form CSV. The app calculates form-based "
    "confidence and opportunity, UHIS, fair odds, market overlay and recommendations."
)

st.info(
    "Upload historical results for the horses in today’s card. More historical rows "
    "produce more reliable confidence and opportunity ratings."
)

race_course = st.text_input(
    "Today’s racecourse",
    value=st.session_state.get("race_course", ""),
    placeholder="e.g. Greyville",
)

race_going = st.text_input(
    "Going / track condition",
    value=st.session_state.get("race_going", ""),
    placeholder="e.g. Good, Good to Soft, Soft",
)

st.session_state["race_course"] = race_course
st.session_state["race_going"] = race_going

col1, col2 = st.columns(2)

with col1:
    racecard_file = st.file_uploader(
        "Upload racecard PDF",
        type=["pdf"],
    )

with col2:
    form_file = st.file_uploader(
        "Upload historical form CSV",
        type=["csv"],
    )

normalise = st.checkbox(
    "Normalise probabilities so each race totals 100%",
    value=True,
)

if racecard_file and form_file:
    try:
        form_data = pd.read_csv(form_file)
        missing_columns = validate_form_file(form_data)

        if missing_columns:
            st.error(
                "The form CSV is missing required columns: "
                + ", ".join(missing_columns)
            )
        else:
            form_data = prepare_form_data(form_data)

            with st.spinner("Analysing racecard and historical form..."):
                extracted_text = extract_text_from_pdf(racecard_file)
                races = analyze_meeting(
                    extracted_text,
                    form_data,
                    normalise_probabilities=normalise,
                )

            if not races:
                st.warning(
                    "No runners could be extracted from the racecard PDF. "
                    "Use the manual text option below instead."
                )
            else:
                st.success(
                    f"Analysed {len(races)} race(s) using "
                    f"{len(form_data)} historical run(s)."
                )

                for race_name, dataframe in races.items():
                    with st.expander(race_name, expanded=True):
                        st.dataframe(dataframe, use_container_width=True)

                        csv_buffer = io.StringIO()
                        dataframe.to_csv(csv_buffer, index=False)

                        st.download_button(
                            "Download race analysis CSV",
                            data=csv_buffer.getvalue(),
                            file_name=f"{race_name.replace(' ', '_')}.csv",
                            mime="text/csv",
                        )

    except Exception as error:
        st.error(f"Could not process the uploaded files: {error}")

st.divider()
st.subheader("Manual racecard text")

manual_text = st.text_area(
    "Paste racecard text",
    height=220,
    placeholder=(
        "RACE 1 - 1200M\n"
        "1 (3) CAPE CRUSADER 60.0 R Fourie J Snaith 4/1\n"
        "2 (5) HIGHVELD STAR 58.5 G Lerena M de Kock 5.50"
    ),
)

if st.button("Analyse pasted racecard"):
    if not manual_text.strip():
        st.warning("Please paste racecard text first.")
    elif form_file is None:
        st.warning("Please upload a historical form CSV first.")
    else:
        try:
            form_data = pd.read_csv(form_file)
            missing_columns = validate_form_file(form_data)

            if missing_columns:
                st.error(
                    "The form CSV is missing required columns: "
                    + ", ".join(missing_columns)
                )
            else:
                form_data = prepare_form_data(form_data)
                races = analyze_meeting(
                    manual_text,
                    form_data,
                    normalise_probabilities=normalise,
                )

                if not races:
                    st.warning("No runners could be parsed from the text.")
                else:
                    for race_name, dataframe in races.items():
                        st.subheader(race_name)
                        st.dataframe(dataframe, use_container_width=True)

        except Exception as error:
            st.error(f"Could not analyse the pasted racecard: {error}")

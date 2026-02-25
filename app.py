from __future__ import annotations

import ast
import csv
import json
import os
import shutil
import subprocess
import urllib.error
import urllib.request
from datetime import datetime, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

from dotenv import load_dotenv

from bokeh.layouts import column as bokeh_column, row as bokeh_row
from bokeh.embed import components
from bokeh.models import Button, ColumnDataSource, CustomJS, DaysTicker, HoverTool, InlineStyleSheet, LinearAxis, Range1d, Spacer
from bokeh.plotting import figure
from bokeh.resources import CDN
from flask import Flask, redirect, render_template, request, send_from_directory, url_for

load_dotenv()

CSV_COLUMNS = [
	"Christian Weight",
	"Krysty Weight",
	"Christian Target",
	"Krysty Target",
	"Date",
]
NUTRITION_COLUMNS = [
	"Date",
	"Meal",
	"Calories",
]
NUTRITION_SUMMARY_COLUMNS = [
	"Date",
	"Person",
	"Description",
	"Protein (g)",
	"Carbohydrates (g)",
	"Fat (g)",
	"Sodium (mg)",
	"Fiber (g)",
	"Vitamin A (mcg)",
	"Vitamin B6 (mg)",
	"Vitamin B12 (mcg)",
	"Vitamin C (mg)",
	"Vitamin D (mcg)",
	"Calcium (mg)",
	"Potassium (mg)",
	"Magnesium (mg)",
	"Iron (mg)",
	"Zinc (mg)",
	"Omega-3 (mg)",
]
NUTRITION_TARGETS = {
	"christian": 1200,
	"krysty": 1400,
}
NUTRITION_JSON_KEYS = {
	"Protein (g)": "protein_g",
	"Carbohydrates (g)": "carbohydrates_g",
	"Fat (g)": "fat_g",
	"Sodium (mg)": "sodium_mg",
	"Fiber (g)": "fiber_g",
	"Vitamin A (mcg)": "vitamin_a_mcg",
	"Vitamin B6 (mg)": "vitamin_b6_mg",
	"Vitamin B12 (mcg)": "vitamin_b12_mcg",
	"Vitamin C (mg)": "vitamin_c_mg",
	"Vitamin D (mcg)": "vitamin_d_mcg",
	"Calcium (mg)": "calcium_mg",
	"Potassium (mg)": "potassium_mg",
	"Magnesium (mg)": "magnesium_mg",
	"Iron (mg)": "iron_mg",
	"Zinc (mg)": "zinc_mg",
	"Omega-3 (mg)": "omega_3_mg",
}

DEFAULT_DATA_DIR = (
	"/data"
	if os.getenv("FLY_APP_NAME")
	else str(Path(__file__).resolve().parent)
)
DATA_DIR = Path(os.getenv("DATA_DIR", DEFAULT_DATA_DIR))
EASTERN_TZ = ZoneInfo("America/New_York")
SYNC_METADATA_PATH = Path(__file__).resolve().parent / ".fly_data_sync.json"
LOCAL_SYNC_FROM_FLY = os.getenv("LOCAL_SYNC_FROM_FLY", "1") == "1"
FLY_APP_NAME = os.getenv("FLY_APP_NAME", "fittinthispizza")
FLY_VOLUME_PATH = os.getenv("FLY_VOLUME_PATH", "/data")
GROQ_API_URL = "https://api.groq.com/openai/v1/chat/completions"
GROQ_MODEL = os.getenv("GROQ_MODEL", "llama-3.1-8b-instant")
PERSON_PROFILES = {
	"christian": {"age": 42, "height": "6'0\""},
	"krysty": {"age": 41, "height": "5'4\""},
}


def _build_data_sync_note() -> str:
	if os.getenv("FLY_APP_NAME"):
		return "Fly app · live data volume: /data"

	if not SYNC_METADATA_PATH.exists():
		return "Local app · no Fly sync metadata yet"

	try:
		with SYNC_METADATA_PATH.open(encoding="utf-8") as handle:
			metadata = json.load(handle)
	except (OSError, json.JSONDecodeError):
		return "Local app · sync metadata unavailable"

	pulled_at = metadata.get("pulled_at_utc", "unknown time")
	volume = metadata.get("source_volume", "/data")
	machine = metadata.get("source_machine_id")
	if machine:
		return f"Local app · pulled {pulled_at} from {volume} (machine {machine})"
	return f"Local app · pulled {pulled_at} from {volume}"


def _maybe_sync_from_fly() -> None:
	if not LOCAL_SYNC_FROM_FLY:
		return

	if os.getenv("FLY_REGION"):
		return

	files = (
		"fitness_data.csv",
		"fitness_data_prototype.csv",
		"nutrition_christian.csv",
		"nutrition_krysty.csv",
		"nutrition.csv",
	)
	synced = False
	for filename in files:
		command = f"cat {FLY_VOLUME_PATH}/{filename}"
		result = subprocess.run(
			["fly", "ssh", "console", "-C", command, "-a", FLY_APP_NAME],
			capture_output=True,
			text=True,
		)
		if result.returncode != 0:
			continue
		(DATA_DIR / filename).write_text(result.stdout, encoding="utf-8")
		synced = True

	if not synced:
		return

	metadata = {
		"pulled_at_utc": datetime.utcnow().isoformat(timespec="seconds") + "Z",
		"source_volume": FLY_VOLUME_PATH,
	}
	SYNC_METADATA_PATH.write_text(json.dumps(metadata, indent=2), encoding="utf-8")


def _read_rows(data_path: Path) -> list[dict[str, str]]:
	if not data_path.exists():
		return []

	with data_path.open(newline="", encoding="utf-8") as handle:
		reader = csv.DictReader(handle)
		rows = [row for row in reader]

	changed = False
	for row in rows:
		date_value = row.get("Date", "")
		if "/" in date_value:
			try:
				parsed = datetime.strptime(date_value, "%m/%d/%Y")
			except ValueError:
				continue
			row["Date"] = parsed.strftime("%m-%d-%Y")
			changed = True

	if changed:
		_write_rows(data_path, rows)

	return rows


def _write_rows(data_path: Path, rows: list[dict[str, str]]) -> None:
	with data_path.open("w", newline="", encoding="utf-8") as handle:
		writer = csv.DictWriter(handle, fieldnames=CSV_COLUMNS)
		writer.writeheader()
		writer.writerows(rows)


def _date_key(row: dict[str, str]) -> datetime:
	return datetime.strptime(row["Date"], "%m-%d-%Y")


def _parse_float(value: str) -> float | None:
	try:
		return float(value)
	except (TypeError, ValueError):
		return None


def _ensure_seed_data_file(data_path: Path, seed_filename: str) -> None:
	if data_path.exists():
		return
	seed_path = Path(__file__).resolve().parent / seed_filename
	if seed_path.exists():
		shutil.copyfile(seed_path, data_path)
		return
	_write_rows(data_path, [])


def _ensure_nutrition_file(data_path: Path) -> None:
	if data_path.exists():
		return
	_write_nutrition_rows(data_path, [])


def _nutrition_data_path(person: str) -> Path:
	return DATA_DIR / f"nutrition_{person}.csv"


def _nutrition_now() -> datetime:
	return datetime.now(EASTERN_TZ)


def _is_mobile_request() -> bool:
	user_agent = request.headers.get("User-Agent", "").lower()
	return any(token in user_agent for token in ("iphone", "android", "mobile", "ipad"))


def _call_groq(messages: list[dict[str, str]]) -> str:
	api_key = os.getenv("GROQ_API_KEY", "")
	if os.getenv("GROQ_DEBUG", "0") == "1":
		print(
			f"GROQ_DEBUG key_loaded={bool(api_key)} length={len(api_key)}",
			flush=True,
		)
	if not api_key:
		raise RuntimeError("Missing GROQ_API_KEY")

	payload = {
		"model": GROQ_MODEL,
		"messages": messages,
		"temperature": 0.1,
		"max_tokens": 900,
	}
	data = json.dumps(payload).encode("utf-8")
	request = urllib.request.Request(
		GROQ_API_URL,
		data=data,
		headers={
			"Authorization": f"Bearer {api_key}",
			"Content-Type": "application/json",
			"Accept": "application/json",
			"User-Agent": "FittinThisPizza/1.0",
		},
		method="POST",
	)
	try:
		with urllib.request.urlopen(request, timeout=20) as response:
			result = json.loads(response.read().decode("utf-8"))
			return result["choices"][0]["message"]["content"].strip()
	except urllib.error.HTTPError as exc:
		body = ""
		try:
			body = exc.read().decode("utf-8")
		except (OSError, UnicodeDecodeError):
			body = ""
		raise RuntimeError(
			f"Groq HTTP {exc.code}: {body or exc.reason}"
		) from exc


def _nutrition_summary_path() -> Path:
	return DATA_DIR / "nutrition.csv"


def _ensure_nutrition_summary_file(data_path: Path) -> None:
	if data_path.exists():
		return
	with data_path.open("w", newline="", encoding="utf-8") as handle:
		writer = csv.DictWriter(handle, fieldnames=NUTRITION_SUMMARY_COLUMNS)
		writer.writeheader()


def _read_nutrition_summary_rows(data_path: Path) -> list[dict[str, str]]:
	if not data_path.exists():
		return []

	with data_path.open(newline="", encoding="utf-8") as handle:
		reader = csv.DictReader(handle)
		return [row for row in reader]


def _write_nutrition_summary_rows(data_path: Path, rows: list[dict[str, str]]) -> None:
	with data_path.open("w", newline="", encoding="utf-8") as handle:
		writer = csv.DictWriter(handle, fieldnames=NUTRITION_SUMMARY_COLUMNS)
		writer.writeheader()
		writer.writerows(rows)


def _load_fixed_prompt() -> str:
	prompt_path = Path(__file__).resolve().parent / "fixed_prompt.txt"
	try:
		return prompt_path.read_text(encoding="utf-8").strip()
	except OSError:
		return ""


def _load_daily_values() -> dict[str, dict[str, float]]:
	values_path = Path(__file__).resolve().parent / "daily_values.csv"
	if not values_path.exists():
		return {}

	values: dict[str, dict[str, float]] = {}
	with values_path.open(newline="", encoding="utf-8") as handle:
		reader = csv.DictReader(handle)
		for row in reader:
			person = (row.get("Person") or "").strip().lower()
			if not person:
				continue
			values[person] = {
				key: float(value) if value not in (None, "") else 0.0
				for key, value in row.items()
				if key and key != "Person"
			}
	return values


def _get_daily_values(person: str) -> dict[str, float]:
	return _load_daily_values().get(person, {})


def _get_target_for_person(person: str) -> float:
	daily_values = _get_daily_values(person)
	energy = daily_values.get("Energy")
	if energy:
		return float(energy)
	return float(NUTRITION_TARGETS.get(person, 0))


def _apply_supplements(person: str, nutrients: dict[str, float]) -> dict[str, float]:
	adjusted = dict(nutrients)
	if person == "christian":
		supplements = {
			"Vitamin A (mcg)": 1050.0,
			"Vitamin B6 (mg)": 2.0,
			"Vitamin B12 (mcg)": 6.0,
			"Vitamin C (mg)": 90.0,
			"Vitamin D (mcg)": 10.0,
			"Calcium (mg)": 200.0,
			"Potassium (mg)": 80.0,
			"Magnesium (mg)": 100.0,
			"Iron (mg)": 18.0,
			"Zinc (mg)": 11.0,
			"Omega-3 (mg)": 1250.0,
		}
	elif person == "krysty":
		supplements = {
			"Vitamin A (mcg)": 1500.0,
			"Vitamin C (mg)": 100.0,
			"Vitamin D (mcg)": 87.5,
			"Calcium (mg)": 600.0,
			"Magnesium (mg)": 300.0,
			"Zinc (mg)": 32.5,
			"Sodium (mg)": 15.0,
			"Omega-3 (mg)": 1250.0,
		}
	else:
		supplements = {}

	for key, value in supplements.items():
		adjusted[key] = adjusted.get(key, 0.0) + value
	return adjusted


def _extract_json_payload(text: str) -> dict[str, object]:
	start = text.find("{")
	end = text.rfind("}")
	raw_payload = None
	if start != -1 and end != -1 and end > start:
		raw_payload = text[start : end + 1]

	for payload_text in [raw_payload, text]:
		if not payload_text:
			continue
		try:
			parsed = json.loads(payload_text)
			if isinstance(parsed, dict):
				return parsed
		except json.JSONDecodeError:
			pass
		try:
			parsed = ast.literal_eval(payload_text)
			if isinstance(parsed, dict):
				return parsed
		except (ValueError, SyntaxError):
			pass

	raise ValueError("No JSON or Python dict payload found in response.")


def _limit_words(text: str, max_words: int) -> str:
	words = text.split()
	if len(words) <= max_words:
		return text
	return " ".join(words[:max_words])


def _coerce_number(value: object) -> float | None:
	if value is None:
		return None
	if isinstance(value, (int, float)):
		return float(value)
	if isinstance(value, str):
		clean = value.replace(",", "").strip()
		try:
			return float(clean)
		except ValueError:
			return None
	return None


def _parse_nutrition_payload(text: str) -> tuple[str | None, dict[str, float] | None, str | None]:
	try:
		payload = _extract_json_payload(text)
	except (ValueError, json.JSONDecodeError) as exc:
		return None, None, f"Invalid JSON payload: {exc}"

	description = payload.get("description") if isinstance(payload, dict) else None
	if isinstance(payload, dict) and description is None:
		description = payload.get("summary")
	if not isinstance(description, str):
		return None, None, "Missing description in payload."

	nutrients = None
	if isinstance(payload, dict):
		nutrients = payload.get("nutrients") or payload.get("nutrition")
	if not isinstance(nutrients, dict):
		return None, None, "Missing nutrients map in payload."

	nutrient_values: dict[str, float] = {}
	for column, key in NUTRITION_JSON_KEYS.items():
		value = nutrients.get(key)
		nutrient_values[column] = _coerce_number(value) or 0.0

	return _limit_words(description, 40), nutrient_values, None


def _build_nutrition_messages(
	person: str,
	entry_date: str,
	entries: list[dict[str, str]],
	total_calories: int,
	fixed_prompt: str,
) -> list[dict[str, str]]:
	lines = []
	for entry in entries:
		meal = entry.get("Meal", "").strip()
		calories = entry.get("Calories", "").strip()
		if not meal and not calories:
			continue
		lines.append(f"- {meal}: {calories} calories")

	keys_hint = ", ".join(NUTRITION_JSON_KEYS.values())
	format_hint = (
		"Return only valid JSON (no markdown, no code fences, no commentary). "
		"Do not return Python code. Schema: {\"description\": \"...\", "
		"\"nutrients\": {" + keys_hint + "}}."
	)
	user_content = (
		f"Name: {person.title()}\n"
		f"Date: {entry_date}\n"
		f"Total calories: {total_calories}\n"
		"Entries:\n"
		+ "\n".join(lines)
		+ "\n\n"
		+ format_hint
	)

	return [
		{"role": "system", "content": fixed_prompt},
		{
			"role": "system",
			"content": "Output JSON only. Do not wrap in code fences or add extra text or code.",
		},
		{"role": "user", "content": user_content},
	]


def _generate_nutrition_summary(
	person: str,
	entry_date: str,
	entries: list[dict[str, str]],
	total_calories: int,
	fixed_prompt: str,
) -> tuple[dict[str, str] | None, str | None]:
	if not entries:
		return None, "No entries found for that date."
	if not fixed_prompt:
		return None, "Missing fixed_prompt.txt."

	try:
		response = _call_groq(
			_build_nutrition_messages(person, entry_date, entries, total_calories, fixed_prompt)
		)
		print("Groq response:\n" + response, flush=True)
		description, nutrient_values, error = _parse_nutrition_payload(response)
		if nutrient_values is not None:
			nutrient_values = _apply_supplements(person, nutrient_values)
		if nutrient_values is not None:
			print(list(nutrient_values.values()), flush=True)
		if error or nutrient_values is None or description is None:
			return None, error or "Unable to parse nutrition summary."
		row: dict[str, str] = {
			"Date": entry_date,
			"Person": person,
			"Description": description,
		}
		for column in NUTRITION_JSON_KEYS:
			row[column] = f"{nutrient_values.get(column, 0.0):.2f}"
		return row, None
	except (RuntimeError, urllib.error.URLError, KeyError, json.JSONDecodeError) as exc:
		return None, f"Groq error: {exc}"


def _summary_key(row: dict[str, str]) -> tuple[str, str]:
	return row.get("Person", "").lower(), row.get("Date", "")


def _ensure_nutrition_outputs(
	person: str,
	all_entries: list[dict[str, str]],
	today: str,
) -> dict[tuple[str, str], dict[str, str]]:
	summary_path = _nutrition_summary_path()
	summary_rows = _read_nutrition_summary_rows(summary_path)
	summary_map = {_summary_key(row): row for row in summary_rows}
	fixed_prompt = _load_fixed_prompt()
	target = _get_target_for_person(person)

	entries_by_date: dict[str, list[dict[str, str]]] = {}
	for row in all_entries:
		date_value = row.get("Date", "")
		if not date_value:
			continue
		entries_by_date.setdefault(date_value, []).append(row)

	updated = False
	for date_value, entries in entries_by_date.items():
		total_calories = 0
		for row in entries:
			try:
				total_calories += int(row.get("Calories", "0"))
			except ValueError:
				continue
		should_generate = date_value != today or (target and total_calories >= target)
		key = (person, date_value)
		if should_generate and key not in summary_map:
			row, error = _generate_nutrition_summary(
				person,
				date_value,
				entries,
				total_calories,
				fixed_prompt,
			)
			if row:
				summary_rows.append(row)
				summary_map[key] = row
				updated = True
			elif error:
				print(f"Nutrition summary error for {person} {date_value}: {error}", flush=True)

	if updated:
		_write_nutrition_summary_rows(summary_path, summary_rows)

	return summary_map


def _read_nutrition_rows(data_path: Path) -> list[dict[str, str]]:
	if not data_path.exists():
		return []

	with data_path.open(newline="", encoding="utf-8") as handle:
		reader = csv.DictReader(handle)
		return [row for row in reader]


def _write_nutrition_rows(data_path: Path, rows: list[dict[str, str]]) -> None:
	with data_path.open("w", newline="", encoding="utf-8") as handle:
		writer = csv.DictWriter(handle, fieldnames=NUTRITION_COLUMNS)
		writer.writeheader()
		writer.writerows(rows)


def _add_nutrition_entry(
	data_path: Path,
	meal: str,
	calories: int,
	entry_date: str,
) -> None:
	rows = _read_nutrition_rows(data_path)
	rows.append(
		{
			"Date": entry_date,
			"Meal": meal,
			"Calories": str(calories),
		}
	)
	_write_nutrition_rows(data_path, rows)


def _summary_value(row: dict[str, str], column: str) -> float:
	value = row.get(column, "")
	parsed = _parse_float(value)
	return parsed if parsed is not None else 0.0


def _build_day_logs(
	person: str,
	all_entries: list[dict[str, str]],
	today: str,
	summary_map: dict[tuple[str, str], dict[str, str]],
) -> list[dict[str, object]]:
	entries_by_date: dict[str, list[dict[str, str]]] = {}
	for row in all_entries:
		date_value = row.get("Date", "")
		if not date_value:
			continue
		entries_by_date.setdefault(date_value, []).append(row)

	all_dates = sorted(entries_by_date.keys(), reverse=True)
	if today not in all_dates:
		all_dates.insert(0, today)

	logs: list[dict[str, object]] = []
	target = _get_target_for_person(person)
	macro_columns = {
		"Protein (g)": "protein",
		"Carbohydrates (g)": "carbs",
		"Fat (g)": "fat",
	}
	other_columns = [
		column
		for column in NUTRITION_SUMMARY_COLUMNS
		if column not in {"Date", "Person", "Description", *macro_columns.keys()}
	]

	for date_value in all_dates:
		entries = entries_by_date.get(date_value, [])
		total_calories = 0
		for row in entries:
			try:
				total_calories += int(row.get("Calories", "0"))
			except ValueError:
				continue
		running = 0
		entry_rows: list[dict[str, object]] = []
		for row in entries:
			calories_value = 0
			try:
				calories_value = int(row.get("Calories", "0"))
			except ValueError:
				calories_value = 0
			is_late = target > 0 and running >= target
			running += calories_value
			entry_rows.append(
				{
					**row,
					"late_entry": is_late,
				}
			)

		summary_row = summary_map.get((person, date_value))
		macros: dict[str, float] | None = None
		other_nutrients: list[dict[str, object]] | None = None
		description = None
		if summary_row:
			description = summary_row.get("Description")
			macros = {
				label: _summary_value(summary_row, column)
				for column, label in macro_columns.items()
			}
			other_nutrients = [
				{
					"label": column,
					"value": _summary_value(summary_row, column),
				}
				for column in other_columns
			]

		logs.append(
			{
				"date": date_value,
				"entries": entry_rows,
				"total_calories": total_calories,
				"target": target,
				"description": description,
				"macros": macros,
				"other_nutrients": other_nutrients,
				"chart_id": date_value.replace("-", ""),
			}
		)

	return logs


def _render_nutrition_page(
	person: str,
	today: str,
	day_logs: list[dict[str, object]],
	is_mobile: bool,
) -> str:
	return render_template(
		"nutrition.html",
		title="Fittin' this Whole Pizza in my Mouth",
		active_person=person,
		today=today,
		day_logs=day_logs,
		is_mobile=is_mobile,
		profile=PERSON_PROFILES.get(person, {}),
		daily_values=_get_daily_values(person),
	)


def _upsert_weight(
	rows: list[dict[str, str]],
	date_value: datetime,
	person: str,
	weight_value: float,
) -> list[dict[str, str]]:
	date_str = date_value.strftime("%m-%d-%Y")
	person_key = f"{person} Weight"

	for row in rows:
		if row["Date"] == date_str:
			row[person_key] = f"{weight_value}"
			return sorted(rows, key=_date_key)

	base_row = rows[-1] if rows else {column: "0" for column in CSV_COLUMNS}
	new_row = {column: base_row.get(column, "0") for column in CSV_COLUMNS}
	new_row["Date"] = date_str
	new_row[person_key] = f"{weight_value}"
	rows.append(new_row)
	return sorted(rows, key=_date_key)


def _delete_date(rows: list[dict[str, str]], date_value: datetime) -> list[dict[str, str]]:
	date_str = date_value.strftime("%m-%d-%Y")
	return [row for row in rows if row.get("Date") != date_str]


def _handle_post(data_path: Path) -> bool:
	action = request.form.get("action", "add").strip()
	date_input = request.form.get("date", "").strip()
	rows = _read_rows(data_path)

	if action == "delete" and date_input:
		try:
			date_value = datetime.strptime(date_input, "%Y-%m-%d")
		except ValueError:
			return False
		rows = _delete_date(rows, date_value)
		_write_rows(data_path, rows)
		return True

	if action in {"add", "edit"}:
		person = request.form.get("person", "").strip()
		weight_input = request.form.get("weight", "").strip()

		if person in {"Christian", "Krysty"} and date_input and weight_input:
			try:
				date_value = datetime.strptime(date_input, "%Y-%m-%d")
				weight_value = float(weight_input)
			except ValueError:
				return False
			rows = _upsert_weight(rows, date_value, person, weight_value)
			_write_rows(data_path, rows)
			return True
		return False

	if action == "edit_row" and date_input:
		try:
			date_value = datetime.strptime(date_input, "%Y-%m-%d")
		except ValueError:
			return False
		updated = {
			"Christian Weight": request.form.get("christian_weight", "").strip(),
			"Krysty Weight": request.form.get("krysty_weight", "").strip(),
		}

		rows = _read_rows(data_path)
		date_str = date_value.strftime("%m-%d-%Y")
		for row in rows:
			if row.get("Date") == date_str:
				for key, value in updated.items():
					if value == "":
						row[key] = ""
						continue
					try:
						float(value)
					except ValueError:
						row[key] = ""
						continue
					row[key] = value
				break
		_write_rows(data_path, sorted(rows, key=_date_key))
		return True

	return False


def _render_progress(
	data_path: Path,
	show_range_buttons: bool = False,
	page_class: str = "",
	is_mobile: bool = False,
) -> str:
	rows = _read_rows(data_path)
	rows_asc = sorted(rows, key=_date_key)
	rows_desc = sorted(rows, key=_date_key, reverse=True)
	if not rows_asc:
		return render_template(
			"progress.html",
			title="Fittin' this Whole Pizza in my Mouth",
			script="",
			div="",
			bokeh_js=CDN.js_files,
			bokeh_css=CDN.css_files,
			rows=[],
			page_class=page_class,
			range_controls=None,
			is_mobile=is_mobile,
		)

	return _render_progress_rows(
		rows_asc,
		rows_desc,
		show_range_buttons,
		page_class,
		is_mobile,
	)


def _bootstrap_nutrition_outputs() -> None:
	for person in ("christian", "krysty"):
		data_path = _nutrition_data_path(person)
		all_entries = _read_nutrition_rows(data_path)
		today = _nutrition_now().strftime("%Y-%m-%d")
		_ensure_nutrition_outputs(person, all_entries, today)


def _render_progress_rows(
	rows_asc: list[dict[str, str]],
	rows_desc: list[dict[str, str]],
	show_range_buttons: bool = False,
	page_class: str = "",
	is_mobile: bool = False,
) -> str:
	dates: list[datetime] = []
	christian_weights: list[float | None] = []
	krysty_weights: list[float | None] = []
	christian_targets: list[float | None] = []
	krysty_targets: list[float | None] = []
	christian_colors: list[str] = []
	krysty_colors: list[str] = []

	rows_with_iso: list[dict[str, str]] = []
	for row in rows_desc:
		row_copy = dict(row)
		row_copy["DateISO"] = datetime.strptime(row["Date"], "%m-%d-%Y").strftime("%Y-%m-%d")
		rows_with_iso.append(row_copy)
	for row in rows_asc:
		christian_target = _parse_float(row["Christian Target"])
		krysty_target = _parse_float(row["Krysty Target"])
		if christian_target is None or krysty_target is None:
			continue

		dates.append(datetime.strptime(row["Date"], "%m-%d-%Y"))
		christian_weight = _parse_float(row["Christian Weight"])
		krysty_weight = _parse_float(row["Krysty Weight"])

		christian_weights.append(christian_weight)
		krysty_weights.append(krysty_weight)
		christian_targets.append(christian_target)
		krysty_targets.append(krysty_target)
		christian_colors.append(
			"#ef4444"
			if christian_weight is not None and christian_weight > christian_target
			else "#22c55e"
			if christian_weight is not None
			else "#94a3b8"
		)
		krysty_colors.append(
			"#ef4444"
			if krysty_weight is not None and krysty_weight > krysty_target
			else "#22c55e"
			if krysty_weight is not None
			else "#94a3b8"
		)

	weight_indices = [
		idx
		for idx, (c_weight, k_weight) in enumerate(zip(christian_weights, krysty_weights))
		if c_weight is not None or k_weight is not None
	]
	if weight_indices:
		end_idx = weight_indices[-1]
		start_idx = weight_indices[-7] if len(weight_indices) >= 7 else weight_indices[0]
		first_weight_idx = weight_indices[0]
		latest_start = dates[start_idx]
		latest_end = dates[end_idx] + timedelta(days=1)
		latest_weight_date = dates[end_idx]
	else:
		first_weight_idx = 0
		latest_end = dates[-1] + timedelta(days=1)
		latest_start = dates[max(len(dates) - 7, 0)]
		latest_weight_date = dates[-1]
	window_indices = [
		idx for idx, date_value in enumerate(dates) if latest_start <= date_value <= latest_end
	]
	if not window_indices:
		window_indices = list(range(len(dates)))

	def build_segments(
		xs: list[datetime],
		ys: list[float | None],
		targets: list[float | None],
		above_color: str,
		below_color: str,
	) -> tuple[list[datetime], list[datetime], list[float], list[float], list[str]]:
		x0: list[datetime] = []
		x1: list[datetime] = []
		y0: list[float] = []
		y1: list[float] = []
		colors: list[str] = []

		for i in range(len(xs) - 1):
			y0_val = ys[i]
			y1_val = ys[i + 1]
			t0_val = targets[i]
			t1_val = targets[i + 1]
			if y0_val is None or y1_val is None or t0_val is None or t1_val is None:
				continue

			d0_val = y0_val - t0_val
			d1_val = y1_val - t1_val
			segment_above = d0_val > 0 and d1_val > 0

			if d0_val == 0 or d1_val == 0 or (d0_val > 0) == (d1_val > 0):
				x0.append(xs[i])
				x1.append(xs[i + 1])
				y0.append(y0_val)
				y1.append(y1_val)
				colors.append(above_color if segment_above else below_color)
				continue

			fraction = d0_val / (d0_val - d1_val)
			cross_x = xs[i] + (xs[i + 1] - xs[i]) * fraction
			cross_y = y0_val + (y1_val - y0_val) * fraction

			x0.append(xs[i])
			x1.append(cross_x)
			y0.append(y0_val)
			y1.append(cross_y)
			colors.append(above_color if d0_val > 0 else below_color)

			x0.append(cross_x)
			x1.append(xs[i + 1])
			y0.append(cross_y)
			y1.append(y1_val)
			colors.append(above_color if d1_val > 0 else below_color)

		return x0, x1, y0, y1, colors

	def build_area_segments(
		xs: list[datetime],
		ys: list[float | None],
		targets: list[float | None],
		above_color: str,
		below_color: str,
	) -> tuple[list[list[datetime]], list[list[float]], list[str]]:
		area_xs: list[list[datetime]] = []
		area_ys: list[list[float]] = []
		colors: list[str] = []

		for i in range(len(xs) - 1):
			y0_val = ys[i]
			y1_val = ys[i + 1]
			t0_val = targets[i]
			t1_val = targets[i + 1]
			if y0_val is None or y1_val is None or t0_val is None or t1_val is None:
				continue

			d0_val = y0_val - t0_val
			d1_val = y1_val - t1_val
			segment_above = d0_val > 0 and d1_val > 0

			if d0_val == 0 or d1_val == 0 or (d0_val > 0) == (d1_val > 0):
				area_xs.append([xs[i], xs[i + 1], xs[i + 1], xs[i]])
				area_ys.append([y0_val, y1_val, t1_val, t0_val])
				colors.append(above_color if segment_above else below_color)
				continue

			fraction = d0_val / (d0_val - d1_val)
			cross_x = xs[i] + (xs[i + 1] - xs[i]) * fraction
			cross_y = y0_val + (y1_val - y0_val) * fraction
			cross_t = t0_val + (t1_val - t0_val) * fraction

			area_xs.append([xs[i], cross_x, cross_x, xs[i]])
			area_ys.append([y0_val, cross_y, cross_t, t0_val])
			colors.append(above_color if d0_val > 0 else below_color)

			area_xs.append([cross_x, xs[i + 1], xs[i + 1], cross_x])
			area_ys.append([cross_y, y1_val, t1_val, cross_t])
			colors.append(above_color if d1_val > 0 else below_color)

		return area_xs, area_ys, colors

	(
		christian_x0,
		christian_x1,
		christian_y0,
		christian_y1,
		christian_line_colors,
	) = build_segments(dates, christian_weights, christian_targets, "#166534", "#166534")
	(
		krysty_x0,
		krysty_x1,
		krysty_y0,
		krysty_y1,
		krysty_line_colors,
	) = build_segments(dates, krysty_weights, krysty_targets, "#f97316", "#f97316")
	(
		christian_area_xs,
		christian_area_ys,
		christian_area_colors,
	) = build_area_segments(dates, christian_weights, christian_targets, "#ef4444", "#22c55e")
	(
		krysty_area_xs,
		krysty_area_ys,
		krysty_area_colors,
	) = build_area_segments(dates, krysty_weights, krysty_targets, "#ef4444", "#22c55e")

	plot_height = 240 if (page_class == "tracker-page" and is_mobile) else 720
	min_height = 220 if (page_class == "tracker-page" and is_mobile) else 360
	plot_aspect = 24 / 9 if (page_class == "tracker-page" and is_mobile) else 16 / 9
	plot = figure(
		x_axis_type="datetime",
		sizing_mode="stretch_width",
		aspect_ratio=plot_aspect,
		min_height=min_height,
		height=plot_height,
		toolbar_location="above",
		x_range=(latest_start, latest_end),
	)
	plot_name = f"{page_class or 'progress'}-plot"
	plot.name = plot_name
	christian_window = [
		christian_targets[idx] for idx in window_indices if christian_targets[idx] is not None
	] + [
		christian_weights[idx] for idx in window_indices if christian_weights[idx] is not None
	]
	if not christian_window:
		christian_window = [0.0, 1.0]
	christian_min = min(christian_window)
	christian_max = max(christian_window)
	christian_span = christian_max - christian_min or 1.0
	plot.y_range = Range1d(christian_min - (christian_span * 0.6), christian_max)
	plot.xaxis.axis_label = "Date"
	plot.xaxis.formatter.days = "%m/%d"
	plot.xaxis.formatter.months = "%m/%d"
	plot.xaxis.formatter.hours = "%m/%d"
	plot.xaxis.formatter.minutes = "%m/%d"
	plot.xaxis.formatter.seconds = "%m/%d"
	plot.xaxis.ticker = DaysTicker(days=list(range(1, 32)))
	plot.xaxis.minor_tick_line_color = None
	plot.yaxis.axis_label = "Christian Weight (lbs)"
	plot.yaxis.axis_label_text_font_style = "bold"
	plot.yaxis.axis_label_text_color = "#007ba7"
	plot.yaxis.major_label_text_color = "#007ba7"
	plot.yaxis.major_label_orientation = 1.5708
	plot.yaxis.minor_tick_line_color = None
	if page_class == "tracker-page" and is_mobile:
		plot.xaxis.major_label_text_font_size = "8pt"
		plot.xaxis.major_label_orientation = 0.785398
		plot.yaxis.major_label_text_font_size = "8pt"
		plot.yaxis.major_label_orientation = 0.785398

	krysty_window = [
		krysty_targets[idx] for idx in window_indices if krysty_targets[idx] is not None
	] + [
		krysty_weights[idx] for idx in window_indices if krysty_weights[idx] is not None
	]
	if not krysty_window:
		krysty_window = [0.0, 1.0]
	krysty_min = min(krysty_window)
	krysty_max = max(krysty_window)
	krysty_span = krysty_max - krysty_min or 1.0
	plot.extra_y_ranges = {
		"krysty": Range1d(krysty_min, krysty_max + (krysty_span * 0.6))
	}
	right_axis = LinearAxis(y_range_name="krysty", axis_label="Krysty Weight (lbs)")
	right_axis.axis_label_text_font_style = "bold"
	right_axis.axis_label_text_color = "#6366f1"
	right_axis.major_label_text_color = "#6366f1"
	right_axis.major_label_orientation = 1.5708
	right_axis.minor_tick_line_color = None
	if page_class == "tracker-page" and is_mobile:
		right_axis.major_label_text_font_size = "8pt"
		right_axis.major_label_orientation = 0.785398
	plot.add_layout(right_axis, "right")
	christian_source = ColumnDataSource(
		{
			"x": dates,
			"y": christian_weights,
			"color": christian_colors,
			"target": christian_targets,
			"diff": [
				(weight - target) if weight is not None and target is not None else None
				for weight, target in zip(christian_weights, christian_targets)
			],
			"diff_color": [
				"#ef4444" if diff is not None and diff > 0 else "#22c55e"
				if diff is not None
				else "#94a3b8"
				for diff in [
					(weight - target) if weight is not None and target is not None else None
					for weight, target in zip(christian_weights, christian_targets)
				]
			],
		}
	)
	krysty_source = ColumnDataSource(
		{
			"x": dates,
			"y": krysty_weights,
			"color": krysty_colors,
			"target": krysty_targets,
			"diff": [
				(weight - target) if weight is not None and target is not None else None
				for weight, target in zip(krysty_weights, krysty_targets)
			],
			"diff_color": [
				"#ef4444" if diff is not None and diff > 0 else "#22c55e"
				if diff is not None
				else "#94a3b8"
				for diff in [
					(weight - target) if weight is not None and target is not None else None
					for weight, target in zip(krysty_weights, krysty_targets)
				]
			],
		}
	)
	christian_source.name = f"{plot_name}-christian"
	krysty_source.name = f"{plot_name}-krysty"

	christian_points = plot.scatter(
		"x",
		"y",
		source=christian_source,
		size=7,
		color="color",
	)
	plot.segment(
		christian_x0,
		christian_y0,
		christian_x1,
		christian_y1,
		line_width=2,
		color=christian_line_colors,
	)
	krysty_points = plot.scatter(
		"x",
		"y",
		source=krysty_source,
		size=7,
		color="color",
		y_range_name="krysty",
		marker="diamond",
	)
	plot.add_tools(
		HoverTool(
			renderers=[christian_points],
			tooltips="""
			<div>
			  <div><strong>Christian</strong></div>
			  <div><span>Weight:</span> <span>@y{0.0}</span></div>
			  <div style="text-align: right;">
			    <span style="color: @diff_color; font-weight: 600;">@diff{+0.0}</span>
			  </div>
			</div>
			""",
			mode="vline",
			attachment="left",
		),
		HoverTool(
			renderers=[krysty_points],
			tooltips="""
			<div>
			  <div><strong>Krysty</strong></div>
			  <div><span>Weight:</span> <span>@y{0.0}</span></div>
			  <div style="text-align: right;">
			    <span style="color: @diff_color; font-weight: 600;">@diff{+0.0}</span>
			  </div>
			</div>
			""",
			mode="vline",
			attachment="right",
		),
	)
	plot.segment(
		krysty_x0,
		krysty_y0,
		krysty_x1,
		krysty_y1,
		line_width=2,
		color=krysty_line_colors,
		y_range_name="krysty",
	)
	plot.patches(
		christian_area_xs,
		christian_area_ys,
		fill_color=christian_area_colors,
		fill_alpha=0.12,
		line_alpha=0,
	)
	plot.patches(
		krysty_area_xs,
		krysty_area_ys,
		fill_color=krysty_area_colors,
		fill_alpha=0.12,
		line_alpha=0,
		y_range_name="krysty",
	)
	plot.scatter(
		[dates[-1]],
		[christian_targets[-1]],
		size=7,
		color="#111827",
		legend_label="Christian Weight",
		alpha=1,
	)
	plot.scatter(
		[dates[-1]],
		[krysty_targets[-1]],
		size=7,
		color="#111827",
		legend_label="Krysty Weight",
		y_range_name="krysty",
		marker="diamond",
		alpha=1,
	)
	plot.line(
		dates,
		christian_targets,
		line_width=2,
		color="#007ba7",
		line_dash="dashed",
		legend_label="Christian Target",
	)
	plot.line(
		dates,
		krysty_targets,
		line_width=2,
		color="#6366f1",
		line_dash="dashed",
		legend_label="Krysty Target",
		y_range_name="krysty",
	)

	plot.legend.location = "top_right"
	plot.legend.label_text_font_size = "9pt"
	plot.legend.click_policy = "hide"
	plot.legend.visible = False

	plot_view = plot
	range_controls = None
	if show_range_buttons:
		days_in_ms = 24 * 60 * 60 * 1000
		latest_end_ms = int((latest_weight_date + timedelta(days=1)).timestamp() * 1000)
		max_start_ms = int(dates[first_weight_idx].timestamp() * 1000)
		max_end_ms = latest_end_ms

		if is_mobile:
			range_controls = {
				"plot_name": plot_name,
				"christian_source": christian_source.name,
				"krysty_source": krysty_source.name,
				"max_start_ms": max_start_ms,
				"max_end_ms": max_end_ms,
				"days_ms": days_in_ms,
			}
		else:
			def window_callback(window_days: int) -> CustomJS:
				return CustomJS(
					args={
						"x_range": plot.x_range,
						"y_range": plot.y_range,
						"krysty_range": plot.extra_y_ranges["krysty"],
						"christian_source": christian_source,
						"krysty_source": krysty_source,
						"latest_end_ms": latest_end_ms,
						"max_start_ms": max_start_ms,
						"days_ms": window_days * days_in_ms,
					},
					code="""
const start = Math.max(max_start_ms, latest_end_ms - days_ms);
x_range.start = start;
x_range.end = latest_end_ms;

const updateAxisRanges = (startMs, endMs) => {
	const cData = christian_source.data;
	const kData = krysty_source.data;
	const cX = cData.x || [];
	const cY = cData.y || [];
	const cTarget = cData.target || [];
	const kY = kData.y || [];
	const kTarget = kData.target || [];

	const cWindow = [];
	const kWindow = [];

	for (let i = 0; i < cX.length; i += 1) {
		const xVal = cX[i];
		if (xVal == null || xVal < startMs || xVal > endMs) {
			continue;
		}

		const cTargetVal = cTarget[i];
		const cWeightVal = cY[i];
		if (cTargetVal != null) cWindow.push(cTargetVal);
		if (cWeightVal != null) cWindow.push(cWeightVal);

		const kTargetVal = kTarget[i];
		const kWeightVal = kY[i];
		if (kTargetVal != null) kWindow.push(kTargetVal);
		if (kWeightVal != null) kWindow.push(kWeightVal);
	}

	const cValues = cWindow.length ? cWindow : [0, 1];
	const cMin = Math.min(...cValues);
	const cMax = Math.max(...cValues);
	const cSpan = (cMax - cMin) || 1;
	y_range.start = cMin - (cSpan * 0.6);
	y_range.end = cMax;

	const kValues = kWindow.length ? kWindow : [0, 1];
	const kMin = Math.min(...kValues);
	const kMax = Math.max(...kValues);
	const kSpan = (kMax - kMin) || 1;
	krysty_range.start = kMin;
	krysty_range.end = kMax + (kSpan * 0.6);
};

updateAxisRanges(start, latest_end_ms);
x_range.change.emit();
y_range.change.emit();
krysty_range.change.emit();
""",
				)

			button_width = 90
			max_width = 80
			one_week = Button(label="1 Week", button_type="warning", width=button_width)
			two_weeks = Button(label="2 Weeks", button_type="warning", width=button_width)
			one_month = Button(label="1 Month", button_type="warning", width=button_width)
			max_range = Button(label="Max", button_type="warning", width=max_width)

			for control_button in (one_week, two_weeks, one_month, max_range):
				control_button.styles = {
					"--inverted-color": "#111827",
					"color": "#111827",
				}
				control_button.stylesheets = [
					InlineStyleSheet(css=".bk-btn{font-weight:700 !important;}")
				]

			one_week.js_on_click(window_callback(7))
			two_weeks.js_on_click(window_callback(14))
			one_month.js_on_click(window_callback(30))
			max_range.js_on_click(
				CustomJS(
					args={
						"x_range": plot.x_range,
						"y_range": plot.y_range,
						"krysty_range": plot.extra_y_ranges["krysty"],
						"christian_source": christian_source,
						"krysty_source": krysty_source,
						"max_start_ms": max_start_ms,
						"max_end_ms": max_end_ms,
					},
					code="""
x_range.start = max_start_ms;
x_range.end = max_end_ms;

const updateAxisRanges = (startMs, endMs) => {
	const cData = christian_source.data;
	const kData = krysty_source.data;
	const cX = cData.x || [];
	const cY = cData.y || [];
	const cTarget = cData.target || [];
	const kY = kData.y || [];
	const kTarget = kData.target || [];

	const cWindow = [];
	const kWindow = [];

	for (let i = 0; i < cX.length; i += 1) {
		const xVal = cX[i];
		if (xVal == null || xVal < startMs || xVal > endMs) {
			continue;
		}

		const cTargetVal = cTarget[i];
		const cWeightVal = cY[i];
		if (cTargetVal != null) cWindow.push(cTargetVal);
		if (cWeightVal != null) cWindow.push(cWeightVal);

		const kTargetVal = kTarget[i];
		const kWeightVal = kY[i];
		if (kTargetVal != null) kWindow.push(kTargetVal);
		if (kWeightVal != null) kWindow.push(kWeightVal);
	}

	const cValues = cWindow.length ? cWindow : [0, 1];
	const cMin = Math.min(...cValues);
	const cMax = Math.max(...cValues);
	const cSpan = (cMax - cMin) || 1;
	y_range.start = cMin - (cSpan * 0.6);
	y_range.end = cMax;

	const kValues = kWindow.length ? kWindow : [0, 1];
	const kMin = Math.min(...kValues);
	const kMax = Math.max(...kValues);
	const kSpan = (kMax - kMin) || 1;
	krysty_range.start = kMin;
	krysty_range.end = kMax + (kSpan * 0.6);
};

updateAxisRanges(max_start_ms, max_end_ms);
x_range.change.emit();
y_range.change.emit();
krysty_range.change.emit();
""",
				)
			)

			controls = bokeh_row(
				Spacer(sizing_mode="stretch_width"),
				max_range,
				one_month,
				two_weeks,
				one_week,
				spacing=2,
				align="end",
				sizing_mode="stretch_width",
			)
			controls_card = bokeh_column(
				controls,
				sizing_mode="stretch_width",
				css_classes=["controls-card"],
			)
			plot_view = bokeh_column(plot, controls_card, sizing_mode="stretch_width")

	script, div = components(plot_view)
	return render_template(
		"progress.html",
		title="Fittin' this Whole Pizza in my Mouth",
		script=script,
		div=div,
		bokeh_js=CDN.js_files,
		bokeh_css=CDN.css_files,
		rows=rows_with_iso,
		page_class=page_class,
		range_controls=range_controls,
		is_mobile=is_mobile,
	)


def create_app() -> Flask:
	app = Flask(__name__)
	DATA_DIR.mkdir(parents=True, exist_ok=True)
	_maybe_sync_from_fly()
	_ensure_seed_data_file(DATA_DIR / "fitness_data.csv", "fitness_data.csv")
	_ensure_seed_data_file(DATA_DIR / "fitness_data_prototype.csv", "fitness_data_prototype.csv")
	_ensure_nutrition_file(_nutrition_data_path("christian"))
	_ensure_nutrition_file(_nutrition_data_path("krysty"))
	_ensure_nutrition_summary_file(_nutrition_summary_path())

	@app.context_processor
	def inject_data_dir() -> dict[str, str]:
		return {
			"data_sync_note": _build_data_sync_note(),
		}

	@app.get("/")
	def index() -> str:
		return redirect(url_for("tracker"))

	@app.get("/logo")
	def logo() -> str:
		return send_from_directory(Path(__file__).resolve().parent, "fitness.jpg")

	@app.route("/progress", methods=["GET", "POST"])
	def progress() -> str:
		data_path = DATA_DIR / "fitness_data_prototype.csv"
		if request.method == "POST":
			_handle_post(data_path)
			return redirect(url_for("progress"))
		return _render_progress(
			data_path,
			show_range_buttons=True,
			page_class="prototype-page",
			is_mobile=_is_mobile_request(),
		)

	@app.route("/tracker", methods=["GET", "POST"])
	def tracker() -> str:
		data_path = DATA_DIR / "fitness_data.csv"
		if request.method == "POST":
			_handle_post(data_path)
			return redirect(url_for("tracker"))
		return _render_progress(
			data_path,
			show_range_buttons=True,
			page_class="tracker-page",
			is_mobile=_is_mobile_request(),
		)

	@app.route("/nutrition", methods=["GET", "POST"])
	def nutrition() -> str:
		person = request.args.get("person", "krysty").strip().lower()
		if person not in {"christian", "krysty"}:
			person = "christian"
		today = _nutrition_now().strftime("%Y-%m-%d")

		data_path = _nutrition_data_path(person)
		all_entries = _read_nutrition_rows(data_path)
		if request.method == "POST":
			meal = request.form.get("meal", "").strip()
			calories_input = request.form.get("calories", "").strip()
			entry_date_input = request.form.get("entry_date", "").strip()
			entry_date = today
			if entry_date_input:
				try:
					entry_date = datetime.strptime(entry_date_input, "%Y-%m-%d").strftime("%Y-%m-%d")
				except ValueError:
					entry_date = today
			if meal and calories_input:
				try:
					calories = int(calories_input)
				except ValueError:
					calories = -1
				if calories >= 0:
					_add_nutrition_entry(data_path, meal, calories, entry_date)
			return redirect(url_for("nutrition", person=person))

		summary_rows = _read_nutrition_summary_rows(_nutrition_summary_path())
		summary_map = {_summary_key(row): row for row in summary_rows}
		day_logs = _build_day_logs(person, all_entries, today, summary_map)
		return _render_nutrition_page(
			person,
			today,
			day_logs,
			_is_mobile_request(),
		)

	@app.post("/nutrition/run-report")
	def run_nutrition_report() -> str:
		person = request.args.get("person", "krysty").strip().lower()
		if person not in {"christian", "krysty"}:
			person = "christian"
		today = _nutrition_now().strftime("%Y-%m-%d")
		entry_date_input = request.form.get("entry_date", "").strip()
		entry_date = entry_date_input or today
		try:
			entry_date = datetime.strptime(entry_date, "%Y-%m-%d").strftime("%Y-%m-%d")
		except ValueError:
			entry_date = today

		data_path = _nutrition_data_path(person)
		all_entries = _read_nutrition_rows(data_path)
		day_entries = [row for row in all_entries if row.get("Date", "") == entry_date]
		total_calories = 0
		for row in day_entries:
			try:
				total_calories += int(row.get("Calories", "0"))
			except ValueError:
				continue

		row, error = _generate_nutrition_summary(
			person,
			entry_date,
			day_entries,
			total_calories,
			_load_fixed_prompt(),
		)
		if row:
			summary_path = _nutrition_summary_path()
			summary_rows = _read_nutrition_summary_rows(summary_path)
			summary_rows = [
				existing for existing in summary_rows if _summary_key(existing) != (person, entry_date)
			]
			summary_rows.append(row)
			_write_nutrition_summary_rows(summary_path, summary_rows)
		elif error:
			print(f"Nutrition summary error for {person} {entry_date}: {error}", flush=True)

		return redirect(url_for("nutrition", person=person))

	return app


if __name__ == "__main__":
	create_app().run(debug=True)

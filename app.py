from __future__ import annotations

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
DEFAULT_NUTRITION_PROMPT = (
	"Keep it encouraging and concise. Mention protein, veggies, and hydration. "
	"Offer one small improvement idea for tomorrow."
)


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
		"temperature": 0.2,
		"max_tokens": 300,
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


def _summarize_nutrition(
	person: str,
	entry_date: str,
	entries: list[dict[str, str]],
	total_calories: int,
	prompt_text: str,
	current_weight: float | None,
	weekly_target_loss: float | None,
) -> tuple[str | None, str | None]:
	if not entries:
		return None, "No entries found for that date."

	lines = []
	for entry in entries:
		meal = entry.get("Meal", "").strip()
		calories = entry.get("Calories", "").strip()
		if not meal and not calories:
			continue
		lines.append(f"- {meal}: {calories} calories")

	weight_note = "unknown"
	if current_weight is not None:
		weight_note = f"{current_weight:.1f} lb"
	weekly_target_note = "unknown"
	if weekly_target_loss is not None:
		weekly_target_note = f"{weekly_target_loss:.1f} lb/week"

	prompt = (
		"You are a helpful nutrition assistant. Summarize the day in 3-5 sentences. "
		"Mention total calories and any patterns. If calories seem low or high, "
		"say so gently. No medical advice."
	)
	user_content = (
		f"Person: {person}\n"
		f"User prompt: {prompt_text}\n"
		f"Date: {entry_date}\n"
		f"Current weight: {weight_note}\n"
		f"Weekly weight loss target: {weekly_target_note}\n"
		f"Total calories: {total_calories}\n"
		"Entries:\n"
		+ "\n".join(lines)
	)
	try:
		summary = _call_groq(
			[
				{"role": "system", "content": prompt},
				{"role": "user", "content": user_content},
			]
		)
		context_lines = [
			f"Prompt: {prompt_text}",
			f"Current weight: {weight_note}",
			f"Weekly weight loss target: {weekly_target_note}",
			"Log:",
			*lines,
		]
		summary = "\n".join(context_lines) + "\n\nSummary:\n" + summary
		return summary, None
	except (RuntimeError, urllib.error.URLError, KeyError, json.JSONDecodeError) as exc:
		return None, f"Groq error: {exc}"


def _get_tracker_metrics(person: str) -> tuple[float | None, float | None]:
	data_path = DATA_DIR / "fitness_data.csv"
	rows = _read_rows(data_path)
	if not rows:
		return None, None
	rows_sorted = sorted(rows, key=_date_key)
	person_title = person.title()
	weight_key = f"{person_title} Weight"
	target_key = f"{person_title} Target"

	current_weight: float | None = None
	for row in reversed(rows_sorted):
		value = _parse_float(row.get(weight_key, ""))
		if value is not None:
			current_weight = value
			break

	latest_target_date: datetime | None = None
	latest_target_value: float | None = None
	for row in reversed(rows_sorted):
		value = _parse_float(row.get(target_key, ""))
		if value is None:
			continue
		latest_target_value = value
		latest_target_date = _date_key(row)
		break

	weekly_target_loss: float | None = None
	if latest_target_date and latest_target_value is not None:
		cutoff = latest_target_date - timedelta(days=7)
		previous_target_value: float | None = None
		for row in reversed(rows_sorted):
			row_date = _date_key(row)
			if row_date > cutoff:
				continue
			value = _parse_float(row.get(target_key, ""))
			if value is None:
				continue
			previous_target_value = value
			break
		if previous_target_value is not None:
			weekly_target_loss = previous_target_value - latest_target_value

	return current_weight, weekly_target_loss


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


def _render_nutrition_page(
	person: str,
	today: str,
	all_entries: list[dict[str, str]],
	summary_text: str | None,
	summary_error: str | None,
	summary_date: str,
	available_dates: list[str] | None,
	summary_generated_at: str | None,
	summary_source: str | None,
	prompt_text: str,
) -> str:
	today_entries = [
		row
		for row in all_entries
		if row.get("Date", "") == today
	]
	previous_by_date: dict[str, list[dict[str, str]]] = {}
	for row in all_entries:
		date_value = row.get("Date", "")
		if not date_value or date_value == today:
			continue
		previous_by_date.setdefault(date_value, []).append(row)

	previous_days: list[dict[str, object]] = []
	for date_value in sorted(previous_by_date.keys(), reverse=True):
		entries = previous_by_date[date_value]
		day_total = 0
		for row in entries:
			try:
				day_total += int(row.get("Calories", "0"))
			except ValueError:
				continue
		previous_days.append(
			{
				"date": date_value,
				"entries": entries,
				"total_calories": day_total,
			}
		)
	total_calories = 0
	for row in today_entries:
		try:
			total_calories += int(row.get("Calories", "0"))
		except ValueError:
			continue

	if available_dates is None:
		available_dates = sorted(
			{
				row.get("Date", "")
				for row in all_entries
				if row.get("Date", "") and row.get("Date", "") != today
			},
			reverse=True,
		)
		if summary_date not in available_dates:
			available_dates.insert(0, summary_date)

	return render_template(
		"nutrition.html",
		title="Nutrition",
		active_person=person,
		today=today,
		today_entries=today_entries,
		previous_days=previous_days,
		total_calories=total_calories,
		summary_text=summary_text,
		summary_error=summary_error,
		summary_date=summary_date,
		available_dates=available_dates,
		summary_generated_at=summary_generated_at,
		summary_source=summary_source,
		prompt_text=prompt_text,
		profile=PERSON_PROFILES.get(person, {}),
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
	plot.yaxis.axis_label_text_color = "#166534"
	plot.yaxis.major_label_text_color = "#166534"
	plot.yaxis.major_label_orientation = 1.5708
	plot.yaxis.minor_tick_line_color = None

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
	right_axis.axis_label_text_color = "#f97316"
	right_axis.major_label_text_color = "#f97316"
	right_axis.major_label_orientation = 1.5708
	right_axis.minor_tick_line_color = None
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
			renderers=[christian_points, krysty_points],
			tooltips="""
			<div>
			  <div><span>Weight:</span> <span>@y{0.0}</span></div>
			  <div style=\"text-align: right;\">
			    <span style=\"color: @diff_color; font-weight: 600;\">@diff{+0.0}</span>
			  </div>
			</div>
			""",
			mode="vline",
		)
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
		color="#166534",
		line_dash="dashed",
		legend_label="Christian Target",
	)
	plot.line(
		dates,
		krysty_targets,
		line_width=2,
		color="#f97316",
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
		person = request.args.get("person", "christian").strip().lower()
		if person not in {"christian", "krysty"}:
			person = "christian"
		today = _nutrition_now().strftime("%Y-%m-%d")
		default_prompt = DEFAULT_NUTRITION_PROMPT

		data_path = _nutrition_data_path(person)
		all_entries = _read_nutrition_rows(data_path)
		previous_dates = sorted(
			{
				row.get("Date", "")
				for row in all_entries
				if row.get("Date", "") and row.get("Date", "") != today
			},
			reverse=True,
		)
		if request.method == "POST":
			action = request.form.get("action", "add").strip().lower()
			if action == "summary":
				summary_date = request.form.get("summary_date", "").strip() or today
				prompt_text = request.form.get("prompt_text", "").strip() or default_prompt
				try:
					summary_date = datetime.strptime(summary_date, "%Y-%m-%d").strftime("%Y-%m-%d")
				except ValueError:
					summary_date = today

				if not previous_dates:
					return _render_nutrition_page(
						person,
						today,
						all_entries,
						None,
						"No previous days with entries yet.",
						today,
						previous_dates,
						datetime.now(EASTERN_TZ).strftime("%Y-%m-%d %H:%M:%S"),
						"manual",
						prompt_text,
					)
				if summary_date not in previous_dates:
					summary_date = previous_dates[0]
				entries_for_date = [
					row
					for row in all_entries
					if row.get("Date", "") == summary_date
				]
				summary_total = 0
				for row in entries_for_date:
					try:
						summary_total += int(row.get("Calories", "0"))
					except ValueError:
						continue
				current_weight, weekly_target_loss = _get_tracker_metrics(person)
				summary_text, summary_error = _summarize_nutrition(
					person,
					summary_date,
					entries_for_date,
					summary_total,
					prompt_text,
					current_weight,
					weekly_target_loss,
				)

				return _render_nutrition_page(
					person,
					today,
					all_entries,
					summary_text,
					summary_error,
					summary_date,
					previous_dates,
					datetime.now(EASTERN_TZ).strftime("%Y-%m-%d %H:%M:%S"),
					"manual",
					prompt_text,
				)

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

		summary_text = None
		summary_error = None
		summary_date = previous_dates[0] if previous_dates else today
		summary_generated_at = None
		summary_source = None
		prompt_text = default_prompt
		if previous_dates:
			entries_for_date = [
				row
				for row in all_entries
				if row.get("Date", "") == summary_date
			]
			summary_total = 0
			for row in entries_for_date:
				try:
					summary_total += int(row.get("Calories", "0"))
				except ValueError:
					continue
			current_weight, weekly_target_loss = _get_tracker_metrics(person)
			summary_text, summary_error = _summarize_nutrition(
				person,
				summary_date,
				entries_for_date,
				summary_total,
				prompt_text,
				current_weight,
				weekly_target_loss,
			)
			summary_generated_at = datetime.now(EASTERN_TZ).strftime("%Y-%m-%d %H:%M:%S")
			summary_source = "auto"
		else:
			summary_error = "No previous days with entries yet."
			summary_generated_at = datetime.now(EASTERN_TZ).strftime("%Y-%m-%d %H:%M:%S")
			summary_source = "auto"

		return _render_nutrition_page(
			person,
			today,
			all_entries,
			summary_text,
			summary_error,
			summary_date,
			previous_dates,
			summary_generated_at,
			summary_source,
			prompt_text,
		)

	return app


if __name__ == "__main__":
	create_app().run(debug=True)

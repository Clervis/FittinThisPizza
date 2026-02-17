from __future__ import annotations

import csv
import os
import shutil
from datetime import datetime, timedelta
from pathlib import Path

from bokeh.embed import components
from bokeh.models import ColumnDataSource, DaysTicker, HoverTool, LinearAxis, Range1d
from bokeh.plotting import figure
from bokeh.resources import CDN
from flask import Flask, redirect, render_template, request, send_from_directory, url_for

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

DATA_DIR = Path(os.getenv("DATA_DIR", "/data"))


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


def _nutrition_data_path(person: str) -> Path:
	return DATA_DIR / f"nutrition_{person}.csv"


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


def _add_nutrition_entry(data_path: Path, meal: str, calories: int) -> None:
	rows = _read_nutrition_rows(data_path)
	rows.append(
		{
			"Date": datetime.now().strftime("%Y-%m-%d"),
			"Meal": meal,
			"Calories": str(calories),
		}
	)
	_write_nutrition_rows(data_path, rows)


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


def _render_progress(data_path: Path) -> str:
	rows = _read_rows(data_path)
	rows_asc = sorted(rows, key=_date_key)
	rows_desc = sorted(rows, key=_date_key, reverse=True)
	if not rows_asc:
		return render_template(
			"progress.html",
			title="Fit'ness Whole Pizza in my Mouth",
			script="",
			div="",
			bokeh_js=CDN.js_files,
			bokeh_css=CDN.css_files,
			rows=[],
		)

	return _render_progress_rows(rows_asc, rows_desc)


def _render_progress_rows(
	rows_asc: list[dict[str, str]],
	rows_desc: list[dict[str, str]],
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
		latest_start = dates[start_idx]
		latest_end = dates[end_idx] + timedelta(days=1)
	else:
		latest_end = dates[-1] + timedelta(days=1)
		latest_start = dates[max(len(dates) - 7, 0)]
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

	plot = figure(
		x_axis_type="datetime",
		sizing_mode="stretch_width",
		aspect_ratio=16 / 9,
		min_height=360,
		height=840,
		toolbar_location="above",
		x_range=(latest_start, latest_end),
	)
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

	script, div = components(plot)
	return render_template(
		"progress.html",
		title="Fit'ness Whole Pizza in my Mouth",
		script=script,
		div=div,
		bokeh_js=CDN.js_files,
		bokeh_css=CDN.css_files,
		rows=rows_with_iso,
	)


def create_app() -> Flask:
	app = Flask(__name__)
	DATA_DIR.mkdir(parents=True, exist_ok=True)
	_ensure_seed_data_file(DATA_DIR / "fitness_data.csv", "fitness_data.csv")
	_ensure_seed_data_file(DATA_DIR / "fitness_data_prototype.csv", "fitness_data_prototype.csv")

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
		return _render_progress(data_path)

	@app.route("/tracker", methods=["GET", "POST"])
	def tracker() -> str:
		data_path = DATA_DIR / "fitness_data.csv"
		if request.method == "POST":
			_handle_post(data_path)
			return redirect(url_for("tracker"))
		return _render_progress(data_path)

	@app.route("/nutrition", methods=["GET", "POST"])
	def nutrition() -> str:
		person = request.args.get("person", "christian").strip().lower()
		if person not in {"christian", "krysty"}:
			person = "christian"

		data_path = _nutrition_data_path(person)
		if request.method == "POST":
			meal = request.form.get("meal", "").strip()
			calories_input = request.form.get("calories", "").strip()
			if meal and calories_input:
				try:
					calories = int(calories_input)
				except ValueError:
					calories = -1
				if calories >= 0:
					_add_nutrition_entry(data_path, meal, calories)
			return redirect(url_for("nutrition", person=person))

		today = datetime.now().strftime("%Y-%m-%d")
		today_entries = [
			row
			for row in _read_nutrition_rows(data_path)
			if row.get("Date", "") == today
		]
		total_calories = 0
		for row in today_entries:
			try:
				total_calories += int(row.get("Calories", "0"))
			except ValueError:
				continue

		return render_template(
			"nutrition.html",
			title="Nutrition",
			active_person=person,
			today=today,
			entries=today_entries,
			total_calories=total_calories,
		)

	return app


if __name__ == "__main__":
	create_app().run(debug=True)

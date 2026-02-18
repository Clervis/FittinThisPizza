# 🏋️‍♂️🍕 Fit’ness Pizza  
### *“I’m into fitness… fit’n this pizza into my face.”*  
A fitness, nutrition, and weight‑tracking app built for Christian & Krysty’s Carnival 2027 journey.

---

## 📌 Overview

**Fit’ness Pizza** is a lightweight Flask-based web application designed to help Christian and Krysty track their:

- Daily weight  
- Target weight trajectory  
- Nutrition and calorie intake  
- Progress toward Carnival 2027 goals  

The app blends simple data tracking with playful branding — because getting in shape doesn’t have to feel like punishment.

---

## 🎯 Features

### **✔ Weight Tracking**
- CSV‑based storage for simplicity and portability  
- Daily entries for Christian and Krysty  
- Jittered sample data for demos  
- Automatic target projections through **December 31, 2026**

### **✔ Target Projections**
- **Christian**
  - Loses **2/7 lbs/day** until **May 17, 2026**
  - Loses **1/7 lbs/day** from **May 18 → Dec 31, 2026**
- **Krysty**
  - Loses **18/318 lbs/day** through **Dec 31, 2026**

### **✔ Nutrition Logging**
- Daily calorie intake  
- Meal breakdowns  
- Future expansion for macros

### **✔ Flask Web Interface**
- Clean, simple UI  
- Routes for viewing data, adding entries, and visualizing progress  
- CSV ingestion and display

---

## 🔄 Local <-> Fly Data Sync

Use these helper scripts from the repo root to test locally with current Fly data.

- Pull latest data from Fly to local CSV files:
  - `./scripts/sync_from_fly.sh`
- Push local CSV files to Fly (creates remote backups first):
  - `./scripts/sync_to_fly.sh`
  - Non-interactive: `./scripts/sync_to_fly.sh --yes`

Optional environment variables:

- `APP_NAME` (default: `fittinthispizza`)
- `REMOTE_DIR` (default: `/data`)
- `FLY_MACHINE_ID` (target a specific machine)

---

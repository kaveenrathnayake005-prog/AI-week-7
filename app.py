"""
Sri Lanka Road Accident Risk Predictor - Web Version
Flask backend serving the AI model via REST API
"""

from flask import Flask, render_template, request, jsonify
import pandas as pd
import numpy as np
import math
import random
import threading
from datetime import datetime, timedelta
from collections import deque
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

app = Flask(__name__)

# ══════════════════════════════════════════════════════════════
#  DATA
# ══════════════════════════════════════════════════════════════
ALL_DISTRICTS = {
    "Colombo":      (0.16, 6.9271,  79.8612, "Western"),
    "Gampaha":      (0.11, 7.0873,  79.9993, "Western"),
    "Kalutara":     (0.06, 6.5854,  79.9607, "Western"),
    "Kandy":        (0.09, 7.2906,  80.6337, "Central"),
    "Matale":       (0.04, 7.4675,  80.6234, "Central"),
    "Nuwara Eliya": (0.03, 6.9497,  80.7891, "Central"),
    "Galle":        (0.06, 6.0535,  80.2210, "Southern"),
    "Matara":       (0.05, 5.9549,  80.5550, "Southern"),
    "Hambantota":   (0.03, 6.1241,  81.1185, "Southern"),
    "Jaffna":       (0.04, 9.6615,  80.0255, "Northern"),
    "Kilinochchi":  (0.02, 9.3803,  80.4037, "Northern"),
    "Mannar":       (0.02, 8.9810,  79.9044, "Northern"),
    "Vavuniya":     (0.02, 8.7514,  80.4971, "Northern"),
    "Mullaitivu":   (0.01, 9.2671,  80.8128, "Northern"),
    "Batticaloa":   (0.03, 7.7102,  81.6924, "Eastern"),
    "Ampara":       (0.03, 7.2913,  81.6722, "Eastern"),
    "Trincomalee":  (0.03, 8.5874,  81.2152, "Eastern"),
    "Kurunegala":   (0.07, 7.4818,  80.3609, "North Western"),
    "Puttalam":     (0.03, 8.0362,  79.8283, "North Western"),
    "Anuradhapura": (0.04, 8.3114,  80.4037, "North Central"),
    "Polonnaruwa":  (0.03, 7.9403,  81.0188, "North Central"),
    "Badulla":      (0.04, 6.9934,  81.0550, "Uva"),
    "Monaragala":   (0.02, 6.8728,  81.3507, "Uva"),
    "Ratnapura":    (0.05, 6.6828,  80.3992, "Sabaragamuwa"),
    "Kegalle":      (0.03, 7.2513,  80.3464, "Sabaragamuwa"),
}

DISTRICT_ROUTES = {
    "Colombo":      ["Gampaha","Kalutara","Kegalle","Ratnapura"],
    "Gampaha":      ["Colombo","Kurunegala","Kegalle"],
    "Kalutara":     ["Colombo","Ratnapura","Galle"],
    "Kandy":        ["Matale","Nuwara Eliya","Kegalle","Kurunegala","Badulla"],
    "Matale":       ["Kandy","Kurunegala","Anuradhapura","Trincomalee"],
    "Nuwara Eliya": ["Kandy","Badulla","Ratnapura"],
    "Galle":        ["Kalutara","Matara","Ratnapura"],
    "Matara":       ["Galle","Hambantota","Monaragala"],
    "Hambantota":   ["Matara","Monaragala","Ampara"],
    "Jaffna":       ["Kilinochchi","Mannar"],
    "Kilinochchi":  ["Jaffna","Mannar","Mullaitivu","Vavuniya"],
    "Mannar":       ["Jaffna","Kilinochchi","Vavuniya","Puttalam"],
    "Vavuniya":     ["Kilinochchi","Mannar","Mullaitivu","Anuradhapura"],
    "Mullaitivu":   ["Kilinochchi","Vavuniya","Trincomalee"],
    "Batticaloa":   ["Ampara","Trincomalee","Polonnaruwa"],
    "Ampara":       ["Batticaloa","Hambantota","Monaragala","Polonnaruwa"],
    "Trincomalee":  ["Mullaitivu","Batticaloa","Polonnaruwa","Anuradhapura","Matale"],
    "Kurunegala":   ["Gampaha","Kandy","Matale","Puttalam","Anuradhapura","Kegalle"],
    "Puttalam":     ["Kurunegala","Mannar","Anuradhapura"],
    "Anuradhapura": ["Puttalam","Vavuniya","Matale","Trincomalee","Polonnaruwa","Kurunegala"],
    "Polonnaruwa":  ["Anuradhapura","Trincomalee","Batticaloa","Ampara"],
    "Badulla":      ["Kandy","Nuwara Eliya","Monaragala","Ampara"],
    "Monaragala":   ["Badulla","Matara","Hambantota","Ampara"],
    "Ratnapura":    ["Colombo","Kalutara","Galle","Nuwara Eliya","Kegalle"],
    "Kegalle":      ["Colombo","Gampaha","Kandy","Kurunegala","Ratnapura"],
}

ROAD_TYPES       = {"Expressway":3,"Highway":2,"Urban Road":1,"Rural Road":2,"Mountain Road":3,"Coastal Road":2}
WEATHERS         = {"Clear":0,"Rainy":2,"Foggy":3,"Stormy":4,"Misty":2,"Windy":1}
VEHICLES         = {"Car":1,"Bus":2,"Truck":3,"Motorcycle":2,"Three-Wheeler":2,"Van":1,"Lorry":3,"Bicycle":2,"Tractor":2,"SUV":1}
TIMES            = {"Early Morning (4-6)":2,"Morning Rush (6-9)":3,"Midday (9-15)":1,"Evening Rush (15-19)":3,"Night (19-22)":2,"Late Night (22-4)":3}
ROAD_CONDITIONS  = {"Dry":0,"Wet":2,"Flooded":4,"Potholes":3,"Under Construction":3,"Ice/Oil":4}
LIGHT_CONDITIONS = {"Daylight":0,"Dawn/Dusk":1,"Street Lit":1,"Dark":3,"No Lighting":4}
DAYS             = ["Weekday","Weekend","Public Holiday"]
DIST_NAMES       = list(ALL_DISTRICTS.keys())

# ══════════════════════════════════════════════════════════════
#  MODEL (trained once at startup)
# ══════════════════════════════════════════════════════════════
MODEL = None
LES   = None
TLE   = None
ACC   = 0.0
FEAT_COLS = []

def find_route(origin, dest):
    if origin == dest: return [origin]
    queue = deque([[origin]]); visited = {origin}
    while queue:
        path = queue.popleft()
        for nb in DISTRICT_ROUTES.get(path[-1], []):
            if nb == dest: return path + [nb]
            if nb not in visited:
                visited.add(nb); queue.append(path + [nb])
    return [origin, dest]

def haversine_km(d1, d2):
    la1,lo1 = ALL_DISTRICTS[d1][1], ALL_DISTRICTS[d1][2]
    la2,lo2 = ALL_DISTRICTS[d2][1], ALL_DISTRICTS[d2][2]
    R = 6371; dlat = math.radians(la2-la1); dlon = math.radians(lo2-lo1)
    a = math.sin(dlat/2)**2 + math.cos(math.radians(la1))*math.cos(math.radians(la2))*math.sin(dlon/2)**2
    return R * 2 * math.asin(math.sqrt(a)) * 1.45

def generate_data(n=3000):
    random.seed(42); np.random.seed(42)
    dist_weights = [ALL_DISTRICTS[d][0] for d in DIST_NAMES]
    rows = []; base = datetime(2023,1,1)
    for _ in range(n):
        d  = random.choices(DIST_NAMES, weights=dist_weights)[0]
        rt = random.choice(list(ROAD_TYPES.keys()))
        w  = random.choices(list(WEATHERS.keys()),         weights=[0.45,0.22,0.10,0.08,0.05,0.10])[0]
        v  = random.choices(list(VEHICLES.keys()),         weights=[0.20,0.08,0.10,0.05,0.18,0.10,0.08,0.04,0.09,0.08])[0]
        t  = random.choices(list(TIMES.keys()),            weights=[0.08,0.18,0.28,0.22,0.14,0.10])[0]
        rc = random.choices(list(ROAD_CONDITIONS.keys()),  weights=[0.40,0.25,0.05,0.10,0.10,0.10])[0]
        lc = random.choices(list(LIGHT_CONDITIONS.keys()), weights=[0.45,0.08,0.15,0.18,0.14])[0]
        dy = random.choices(DAYS, weights=[0.65,0.25,0.10])[0]
        sp = int(np.clip(np.random.normal(68,28),5,180))
        ag = int(np.clip(np.random.normal(32,12),18,75))
        ex = int(np.clip(np.random.normal(8,6),0,40))
        mo = (base+timedelta(days=random.randint(0,364))).month
        sc = TIMES[t]+WEATHERS[w]+ROAD_TYPES[rt]+VEHICLES[v]+ROAD_CONDITIONS[rc]+LIGHT_CONDITIONS[lc]
        if sp>120: sc+=4
        elif sp>90: sc+=3
        elif sp>70: sc+=2
        elif sp>50: sc+=1
        if ag<22 or ag>65: sc+=2
        elif ag<25 or ag>60: sc+=1
        if ex<2: sc+=2
        elif ex<5: sc+=1
        if dy=="Public Holiday": sc+=1
        if mo in [4,12,1]: sc+=1
        if mo in [5,6,10,11]: sc+=1
        if d in ["Colombo","Gampaha","Kandy","Kurunegala"]: sc+=1
        risk = "High" if sc>=14 else ("Medium" if sc>=8 else "Low")
        rows.append([d,rt,w,v,t,rc,lc,sp,ag,ex,dy,mo,risk])
    return pd.DataFrame(rows, columns=["district","road_type","weather","vehicle_type",
                                        "time_of_day","road_condition","light_condition",
                                        "speed_kmh","driver_age","experience_years",
                                        "day_of_week","month","risk_level"])

def train_model(df):
    global MODEL, LES, TLE, ACC, FEAT_COLS
    les = {}; dfe = df.copy()
    for c in ["district","road_type","weather","vehicle_type","time_of_day",
              "road_condition","light_condition","day_of_week"]:
        le = LabelEncoder(); dfe[c] = le.fit_transform(df[c]); les[c] = le
    tle = LabelEncoder(); dfe["risk_level"] = tle.fit_transform(df["risk_level"])
    X = dfe.drop("risk_level", axis=1); y = dfe["risk_level"]
    Xtr,Xte,ytr,yte = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    mdl = RandomForestClassifier(n_estimators=200, max_depth=12, min_samples_split=5,
                                  random_state=42, class_weight="balanced", n_jobs=-1)
    mdl.fit(Xtr, ytr)
    MODEL = mdl; LES = les; TLE = tle
    ACC = accuracy_score(yte, mdl.predict(Xte))
    FEAT_COLS = X.columns.tolist()
    print(f"✅ Model trained — Accuracy: {ACC*100:.1f}%")

# Load pre-trained model (instant startup)
import pickle, os
_pkl = os.path.join(os.path.dirname(__file__), 'model.pkl')
if os.path.exists(_pkl):
    print("⚡ Loading pre-trained model...")
    _data = pickle.load(open(_pkl,'rb'))
    MODEL = _data['model']; LES = _data['les']; TLE = _data['tle']
    ACC   = _data['acc'];  FEAT_COLS = _data['feat_cols']
    print(f"✅ Model loaded — Accuracy: {ACC*100:.1f}%")
else:
    print("🔄 Training model (first run)...")
    df = generate_data(3000)
    train_model(df)

# ══════════════════════════════════════════════════════════════
#  ROUTES
# ══════════════════════════════════════════════════════════════
@app.route("/")
def index():
    return render_template("index.html",
        districts=DIST_NAMES,
        weathers=list(WEATHERS.keys()),
        vehicles=list(VEHICLES.keys()),
        times=list(TIMES.keys()),
        road_conditions=list(ROAD_CONDITIONS.keys()),
        light_conditions=list(LIGHT_CONDITIONS.keys()),
        days=DAYS,
        accuracy=round(ACC*100, 1)
    )

@app.route("/predict", methods=["POST"])
def predict():
    data = request.json
    origin  = data.get("origin")
    dest    = data.get("dest")
    weather = data.get("weather", "Clear")
    vehicle = data.get("vehicle", "Car")
    time_od = data.get("time_of_day", "Midday (9-15)")
    road_c  = data.get("road_condition", "Dry")
    light_c = data.get("light_condition", "Daylight")
    day     = data.get("day", "Weekday")
    speed   = int(data.get("speed", 70))
    age     = int(data.get("age", 32))
    exp     = int(data.get("experience", 8))
    month   = int(data.get("month", 6))

    if not origin or not dest:
        return jsonify({"error": "Origin and destination required"}), 400
    if origin == dest:
        return jsonify({"error": "Origin and destination cannot be the same"}), 400

    route    = find_route(origin, dest)
    segments = []
    total_km = 0.0
    risk_scores = {"Low":0,"Medium":1,"High":2}

    for i in range(len(route)-1):
        d1, d2 = route[i], route[i+1]
        km = haversine_km(d1, d2); total_km += km
        risk, conf, prob, rt = predict_district(d1, weather, vehicle, time_od, road_c, light_c, day, speed, age, exp, month)
        segments.append({"from":d1,"to":d2,"km":round(km,1),"risk":risk,"conf":round(conf,1),"prob":prob,"road_type":rt,"province":ALL_DISTRICTS[d1][3]})

    risk_d, conf_d, prob_d, rt_d = predict_district(dest, weather, vehicle, time_od, road_c, light_c, day, speed, age, exp, month)
    segments.append({"from":dest,"to":None,"km":0,"risk":risk_d,"conf":round(conf_d,1),"prob":prob_d,"road_type":rt_d,"province":ALL_DISTRICTS[dest][3]})

    total_w   = sum(max(s["km"],1) for s in segments)
    score_sum = sum(risk_scores[s["risk"]]*max(s["km"],1) for s in segments)
    avg_score = score_sum / total_w
    overall   = "High" if avg_score>=1.5 else ("Medium" if avg_score>=0.6 else "Low")
    avg_conf  = sum(s["conf"] for s in segments) / len(segments)
    gauge     = int(avg_score/2*100)

    # Smart tips
    tips = []
    if weather in ["Rainy","Stormy","Foggy","Misty"]: tips.append("🌧️ Reduce speed by 30% in wet/foggy conditions.")
    if time_od in ["Late Night (22-4)","Early Morning (4-6)"]: tips.append("🌙 Night driving fatigue risk — take breaks every hour.")
    if vehicle in ["Motorcycle","Bicycle","Three-Wheeler"]: tips.append("🏍️ Two/three-wheelers are 3× more vulnerable — wear full gear.")
    if speed > 90: tips.append(f"⚡ Speed {speed} km/h is too high — target under 80 km/h on Sri Lankan roads.")
    if any(s["risk"]=="High" for s in segments): tips.append("🚨 HIGH RISK segments — share your location before departing.")
    if any(s["road_type"]=="Mountain Road" for s in segments): tips.append("⛰️ Mountain roads ahead — use lower gears on descent.")
    if not tips: tips.append("✅ Conditions look reasonable. Maintain speed limits and stay alert.")

    return jsonify({
        "route": route,
        "segments": segments,
        "overall": overall,
        "avg_conf": round(avg_conf,1),
        "total_km": round(total_km,1),
        "gauge": gauge,
        "tips": tips,
        "accuracy": round(ACC*100,1)
    })

def predict_district(district, weather, vehicle, time_od, road_c, light_c, day, speed, age, exp, month):
    province = ALL_DISTRICTS[district][3]
    if province == "Western": rt = "Urban Road"
    elif district == "Nuwara Eliya" or "Mountain" in district: rt = "Mountain Road"
    elif district in ["Galle","Matara","Trincomalee"]: rt = "Coastal Road"
    elif district in ["Colombo","Kandy"]: rt = "Highway"
    else: rt = "Rural Road"

    sample = pd.DataFrame([{
        "district":        LES["district"].transform([district])[0],
        "road_type":       LES["road_type"].transform([rt])[0],
        "weather":         LES["weather"].transform([weather])[0],
        "vehicle_type":    LES["vehicle_type"].transform([vehicle])[0],
        "time_of_day":     LES["time_of_day"].transform([time_od])[0],
        "road_condition":  LES["road_condition"].transform([road_c])[0],
        "light_condition": LES["light_condition"].transform([light_c])[0],
        "speed_kmh": speed, "driver_age": age, "experience_years": exp,
        "day_of_week": LES["day_of_week"].transform([day])[0],
        "month": month,
    }])[FEAT_COLS]

    proba = MODEL.predict_proba(sample)[0]
    pred  = MODEL.predict(sample)[0]
    risk  = TLE.inverse_transform([pred])[0]
    conf  = float(max(proba))*100
    prob  = {TLE.classes_[i]: round(float(proba[i]),3) for i in range(len(TLE.classes_))}
    return risk, conf, prob, rt

@app.route("/districts")
def districts():
    return jsonify({"districts": DIST_NAMES, "data": {k: {"province": v[3], "lat": v[1], "lon": v[2]} for k,v in ALL_DISTRICTS.items()}})

if __name__ == "__main__":
    app.run(debug=False, host="0.0.0.0", port=5000)

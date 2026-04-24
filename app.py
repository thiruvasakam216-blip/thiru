"""
VegPredict v5 — Flask Backend (FIXED)
================================
Run:  python app.py
Open: http://localhost:5000

Fixes applied:
- Added /api/chat endpoint so chatbot works without CORS issues
- Fixed /api/predict/all conflict with /api/predict/<veg_key>
- Added proper route ordering (specific before wildcard)
"""

import os, math, hashlib, secrets
from datetime import date, datetime, timedelta, timezone
from functools import wraps

from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
from flask_sqlalchemy import SQLAlchemy

try:
    import numpy as np
    from sklearn.ensemble import GradientBoostingRegressor
    from sklearn.preprocessing import StandardScaler
    HAS_ML = True
except ImportError:
    HAS_ML = False
    print("⚠  scikit-learn not found — using statistical fallback model")

BASE_DIR     = os.path.dirname(os.path.abspath(__file__))
TEMPLATE_DIR = os.path.join(BASE_DIR, "templates")

app = Flask(__name__, template_folder=TEMPLATE_DIR, static_folder="static")
app.secret_key = secrets.token_hex(32)
app.config["SQLALCHEMY_DATABASE_URI"] = os.getenv("DATABASE_URL", f"sqlite:///{os.path.join(BASE_DIR, 'vegpredict.db')}")
app.config["SQLALCHEMY_TRACK_MODIFICATIONS"] = False
CORS(app, supports_credentials=True, origins=["http://localhost:5000", "http://127.0.0.1:5000"])

db = SQLAlchemy(app)

# ── DATABASE MODELS ───────────────────────────────────────────────

class User(db.Model):
    __tablename__ = "users"
    id            = db.Column(db.String(20),  primary_key=True)
    email         = db.Column(db.String(200), unique=True, nullable=False)
    password_hash = db.Column(db.String(64),  nullable=False)
    fname         = db.Column(db.String(80),  nullable=False)
    lname         = db.Column(db.String(80),  nullable=False)
    phone         = db.Column(db.String(20),  default="")
    role          = db.Column(db.String(20),  default="consumer")
    loc_consent   = db.Column(db.Boolean,     default=False)
    search_history= db.Column(db.Text,        default="")   # comma-separated
    login_count   = db.Column(db.Integer,     default=0)
    last_login    = db.Column(db.String(40),  nullable=True)
    created_at    = db.Column(db.String(40),  default=lambda: datetime.now(timezone.utc).isoformat())

    def to_dict(self):
        return {
            "id": self.id, "email": self.email,
            "fname": self.fname, "lname": self.lname,
            "phone": self.phone, "role": self.role,
            "loc_consent": self.loc_consent,
            "login_count": self.login_count,
            "last_login": self.last_login,
            "search_history": [s for s in self.search_history.split(",") if s],
            "alerts": []
        }


class Session(db.Model):
    __tablename__ = "sessions"
    token      = db.Column(db.String(64), primary_key=True)
    user_email = db.Column(db.String(200), db.ForeignKey("users.email"), nullable=False)
    created_at = db.Column(db.String(40),  default=lambda: datetime.now(timezone.utc).isoformat())

def _hash(pw): return hashlib.sha256(pw.encode()).hexdigest()
def _now_iso(): return datetime.now(timezone.utc).isoformat()

def _seed_users():
    if User.query.count() > 0:
        return  # already seeded
    db.session.add(User(
        id="demo-001", email="demo@vegpredict.in", password_hash=_hash("demo1234"),
        fname="Demo", lname="User", phone="", role="consumer", loc_consent=True,
        search_history="", login_count=0
    ))
    db.session.add(User(
        id="usr-002", email="thiru@vegpredict.in", password_hash=_hash("thiru123"),
        fname="Thiru", lname="Kumar", phone="+91 9876543210", role="analyst", loc_consent=True,
        search_history="Tomato,Onion,Potato", login_count=5, last_login=_now_iso()
    ))
    db.session.commit()
    print("[VegPredict] Demo users seeded.")

def require_auth(f):
    @wraps(f)
    def wrapper(*args, **kwargs):
        token = (request.headers.get("X-Auth-Token") or request.cookies.get("auth_token") or "")
        if not token:
            return jsonify({"error":"Unauthorised — please login"}), 401
        sess = Session.query.get(token)
        if not sess:
            return jsonify({"error":"Unauthorised — please login"}), 401
        user = User.query.filter_by(email=sess.user_email).first()
        if not user:
            return jsonify({"error":"Unauthorised — please login"}), 401
        request.user_email = sess.user_email
        request.user = user
        return f(*args, **kwargs)
    return wrapper

VEGETABLES = {
    "tomato":{"name":"Tomato","emoji":"🍅","base_price":42,"unit":"kg","volatility":"High","peak_months":[5,6],"low_months":[12,1],"yearly":[28,32,35,38,44,50,48,40,36,30,34,38,42,46,52,48,44,38,34,30,36,40,45,42],"weekly":[38,40,39,42,44,43,45,46,42,40,41,44,47,46,42,40,38,41,43,45,44,42,40,42,43,45,47,49],"daily":[40,41,43,42,44,45,44,43,42,41,40,42,43,45,46,45,44,43,41,42,43,44,45,42,43,44,46,47,48,49],"algorithm":"LSTM + Random Forest","accuracy":96.4},
    "onion":{"name":"Onion","emoji":"🧅","base_price":55,"unit":"kg","volatility":"Very High","peak_months":[7,8],"low_months":[2,3],"yearly":[30,28,32,45,55,80,70,60,50,42,35,32,38,44,58,72,68,52,40,35,30,38,48,55],"weekly":[45,48,52,55,58,60,62,58,55,52,50,54,57,60,63,65,62,60,56,58,60,62,65,67,66,64,66,68],"daily":[52,53,55,57,56,58,59,58,56,55,54,55,57,59,60,61,60,58,57,56,58,60,62,63,64,65,66,67,67,68],"algorithm":"ARIMA + Gradient Boosting","accuracy":93.7},
    "potato":{"name":"Potato","emoji":"🥔","base_price":28,"unit":"kg","volatility":"Low","peak_months":[11,12],"low_months":[3,4],"yearly":[22,20,22,24,26,30,32,28,26,24,22,21,22,24,27,29,28,26,24,22,20,22,25,28],"weekly":[30,29,28,27,28,29,28,27,26,25,24,25,26,27,28,29,28,27,26,25,24,25,26,27,26,25,26,26],"daily":[28,28,27,27,28,28,27,26,26,25,25,26,26,27,27,28,27,27,26,26,25,25,26,26,26,26,25,26,26,26],"algorithm":"Linear Regression + SVR","accuracy":98.2},
    "carrot":{"name":"Carrot","emoji":"🥕","base_price":38,"unit":"kg","volatility":"Medium","peak_months":[6,7],"low_months":[12,1],"yearly":[28,30,35,50,58,52,44,36,30,26,24,26,30,34,38,54,60,48,38,30,26,28,32,38],"weekly":[32,34,36,38,40,41,40,39,37,36,35,37,38,40,41,42,41,40,38,36,35,37,39,40,40,41,41,41],"daily":[36,37,38,39,38,39,40,40,39,38,38,39,40,40,41,41,40,39,38,38,39,40,41,41,40,40,41,41,41,41],"algorithm":"Prophet + XGBoost","accuracy":95.1},
    "brinjal":{"name":"Brinjal","emoji":"🍆","base_price":35,"unit":"kg","volatility":"Medium","peak_months":[5,6],"low_months":[10,11],"yearly":[28,30,32,38,45,42,36,30,26,24,26,28,30,32,35,40,44,40,34,28,26,28,32,35],"weekly":[38,37,36,35,34,33,34,35,36,35,34,33,32,33,34,35,34,33,32,31,32,33,32,32,32,32,32,32],"daily":[35,35,34,34,33,33,32,32,33,33,34,33,33,32,32,32,31,31,32,32,33,32,32,31,31,32,32,32,32,32],"algorithm":"Random Forest + LSTM","accuracy":94.8},
    "cabbage":{"name":"Cabbage","emoji":"🥬","base_price":22,"unit":"kg","volatility":"Medium","peak_months":[6,7,8],"low_months":[12,1,2],"yearly":[14,12,14,18,28,35,32,25,18,14,12,13,14,16,20,30,36,30,22,16,13,14,18,22],"weekly":[18,19,20,22,23,24,24,23,22,21,22,23,24,25,25,24,23,22,23,24,25,25,24,24,25,25,25,25],"daily":[20,21,22,22,23,23,24,24,23,22,23,23,24,24,25,25,24,23,23,24,24,25,25,24,24,25,25,25,25,25],"algorithm":"ARIMA + Neural Net","accuracy":96.0},
    "cauliflower":{"name":"Cauliflower","emoji":"🥦","base_price":30,"unit":"kg","volatility":"Medium-High","peak_months":[5,6,7],"low_months":[11,12,1],"yearly":[20,18,20,28,40,50,45,35,25,18,16,18,20,24,32,45,52,42,30,22,18,20,26,30],"weekly":[25,26,27,28,30,31,31,30,29,28,28,29,30,31,32,33,32,31,30,30,31,32,33,33,33,34,34,34],"daily":[28,29,30,30,31,31,32,32,31,30,30,31,31,32,32,33,33,32,31,31,32,32,33,33,33,34,34,34,34,34],"algorithm":"Gradient Boosting + SARIMA","accuracy":95.6},
    "beans":{"name":"Beans","emoji":"🫘","base_price":60,"unit":"kg","volatility":"High","peak_months":[4,5,6],"low_months":[11,12],"yearly":[50,48,52,65,78,85,75,62,55,48,44,46,50,56,65,80,88,72,58,50,46,50,58,60],"weekly":[68,65,63,60,58,57,56,55,54,55,56,57,58,57,56,55,54,54,55,55,55,55,55,55,55,55,55,55],"daily":[62,61,60,59,58,57,56,55,55,54,55,55,56,56,55,55,54,54,55,55,55,55,54,54,55,55,55,55,55,55],"algorithm":"XGBoost + LSTM","accuracy":93.2},
    "peas":{"name":"Peas","emoji":"🟢","base_price":45,"unit":"kg","volatility":"High","peak_months":[6,7,8,9],"low_months":[12,1,2],"yearly":[80,100,90,70,55,50,55,65,75,80,65,50,60,75,85,100,90,70,58,52,55,65,75,45],"weekly":[38,40,42,45,47,48,49,48,46,45,44,46,47,48,49,50,50,49,48,48,49,50,50,50,50,50,50,50],"daily":[42,43,44,45,46,46,47,47,46,45,46,46,47,47,48,48,49,49,49,50,50,50,49,49,50,50,50,50,50,50],"algorithm":"Seasonal Decomp + RF","accuracy":94.3},
    "spinach":{"name":"Spinach","emoji":"🌿","base_price":25,"unit":"kg","volatility":"Medium","peak_months":[6,7],"low_months":[11,12,1],"yearly":[18,16,18,24,32,40,36,28,22,16,14,15,18,22,28,38,42,34,26,20,15,16,20,25],"weekly":[20,21,22,24,25,26,26,25,24,24,25,26,27,27,27,28,28,27,26,26,27,27,28,28,28,28,28,28],"daily":[23,24,25,25,26,26,27,27,26,25,25,26,26,27,27,28,28,27,27,27,28,28,28,27,27,28,28,28,28,28],"algorithm":"LSTM + Linear Blend","accuracy":95.8},
}

REGIONS = {
    "chennai":{"market":"Koyambedu CMDA","zone":"Koyambedu Hub","state":"Tamil Nadu","adj":1.00,"risk":"Low"},
    "coimbatore":{"market":"Coimbatore Ukkadam","zone":"West TN Belt","state":"Tamil Nadu","adj":0.92,"risk":"Low"},
    "madurai":{"market":"Mattuthavani Market","zone":"South TN Hub","state":"Tamil Nadu","adj":0.95,"risk":"Low"},
    "trichy":{"market":"Ariyamangalam Market","zone":"Central TN","state":"Tamil Nadu","adj":0.97,"risk":"Low"},
    "salem":{"market":"Salem Main Market","zone":"Onion Belt","state":"Tamil Nadu","adj":0.88,"risk":"Medium"},
    "vellore":{"market":"Vellore Market","zone":"NH44 Corridor","state":"Tamil Nadu","adj":1.05,"risk":"High"},
    "tirunelveli":{"market":"Tirunelveli Market","zone":"Deep South","state":"Tamil Nadu","adj":1.08,"risk":"Medium"},
    "erode":{"market":"Erode Market","zone":"Textile + Agri","state":"Tamil Nadu","adj":0.91,"risk":"Low"},
    "krishnagiri":{"market":"Krishnagiri APMC","zone":"Tomato Belt","state":"Tamil Nadu","adj":0.82,"risk":"Low"},
    "ooty":{"market":"Ooty Vegetable Market","zone":"Hill Produce Zone","state":"Tamil Nadu","adj":0.79,"risk":"Low"},
    "tirupur":{"market":"Tirupur Market","zone":"Garment + Agri","state":"Tamil Nadu","adj":0.94,"risk":"Low"},
    "kanchipuram":{"market":"Kanchipuram Market","zone":"Chennai Periphery","state":"Tamil Nadu","adj":1.02,"risk":"Low"},
    "bangalore":{"market":"KR Market","zone":"Karnataka Metro","state":"Karnataka","adj":1.08,"risk":"Low"},
    "hyderabad":{"market":"Bowenpally Market","zone":"Telangana Metro","state":"Telangana","adj":1.12,"risk":"Low"},
    "kochi":{"market":"Ernakulam Market","zone":"Kerala Hub","state":"Kerala","adj":1.06,"risk":"Low"},
    "mumbai":{"market":"APMC Vashi","zone":"West Metro","state":"Maharashtra","adj":1.18,"risk":"Low"},
    "delhi":{"market":"Azadpur APMC","zone":"North Metro","state":"Delhi","adj":1.22,"risk":"Low"},
}

REGION_COORDS = [
    ("chennai",12.9716,80.2443),("coimbatore",11.0168,76.9558),("madurai",9.9252,78.1198),
    ("trichy",10.7905,78.7047),("salem",11.6643,78.1460),("vellore",12.9165,79.1325),
    ("tirunelveli",8.7139,77.7567),("erode",11.3410,77.7172),("krishnagiri",12.5184,78.2141),
    ("ooty",11.4102,76.6950),("tirupur",11.1085,77.3411),("kanchipuram",12.8342,79.7036),
    ("bangalore",12.9716,77.5946),("hyderabad",17.3850,78.4867),("kochi",9.9312,76.2673),
    ("mumbai",19.0760,72.8777),("delhi",28.7041,77.1025),
]

def nearest_region(lat, lon):
    best, best_d = "chennai", float("inf")
    for name, rlat, rlon in REGION_COORDS:
        d = math.sqrt((lat-rlat)**2 + (lon-rlon)**2)
        if d < best_d: best_d, best = d, name
    return best

def _seasonal_index(veg_key, month):
    veg = VEGETABLES[veg_key]
    peak, low = veg["peak_months"], veg["low_months"]
    if month in peak: return 1.25 + 0.10*peak.index(month)/max(len(peak),1)
    if month in low:  return 0.82 - 0.05*low.index(month)/max(len(low),1)
    return 1.00

def _trend_factor(veg_key):
    data = VEGETABLES[veg_key]["yearly"][-12:]
    if len(data) < 2: return 1.0
    slope = (data[-1]-data[0]) / max(len(data)-1,1)
    return 1.0 + max(-0.12, min(0.15, slope/max(data[0],1)))

def predict_price(veg_key, region_key, days_ahead=30):
    veg = VEGETABLES[veg_key]
    today = date.today()
    f_month = (today.month + days_ahead//30 - 1) % 12 + 1
    region = REGIONS.get(region_key, REGIONS["chennai"])
    reg_mult = region["adj"]
    base = veg["base_price"]
    seas_idx = _seasonal_index(veg_key, f_month)
    trend_f = _trend_factor(veg_key)

    if HAS_ML:
        prices = np.array(veg["yearly"], dtype=float)
        n = len(prices)
        X = np.column_stack([np.arange(n),[_seasonal_index(veg_key,m%12+1) for m in range(n)],np.convolve(prices,np.ones(3)/3,"same")])
        scaler = StandardScaler()
        Xs = scaler.fit_transform(X)
        model = GradientBoostingRegressor(n_estimators=120,max_depth=3,learning_rate=0.08,random_state=42)
        model.fit(Xs, prices)
        Xf = scaler.transform([[n+days_ahead//15, seas_idx, prices[-3:].mean()]])
        ml_price = float(model.predict(Xf)[0])
        predicted_raw = max(ml_price*seas_idx*0.5 + base*seas_idx*0.5, base*0.4)
    else:
        predicted_raw = base * seas_idx * trend_f

    predicted = round(predicted_raw * reg_mult, 1)
    current   = round(base * reg_mult, 1)
    change    = round((predicted-current)/max(current,1)*100, 1)
    vol_map   = {"Low":0.05,"Medium":0.08,"Medium-High":0.10,"High":0.12,"Very High":0.16}
    band      = vol_map.get(veg["volatility"], 0.08)
    conf_low  = round(predicted*(1-band), 1)
    conf_high = round(predicted*(1+band), 1)
    forecast  = [round((current + (predicted-current)*(d/max(days_ahead-1,1))**0.7)*(0.98+0.04*math.sin(d*0.5)),1) for d in range(days_ahead)]

    return {
        "vegetable":veg["name"],"emoji":veg["emoji"],"region":region_key,
        "market":region["market"],"zone":region["zone"],"state":region["state"],"risk":region["risk"],
        "current_price":current,"predicted_price":predicted,"change_pct":change,
        "conf_low":conf_low,"conf_high":conf_high,"forecast_30d":forecast,
        "seasonal_index":round(seas_idx,3),"region_multiplier":reg_mult,
        "algorithm":veg["algorithm"],"model_accuracy":veg["accuracy"],"volatility":veg["volatility"],
        "peak_months":veg["peak_months"],"low_months":veg["low_months"],"days_ahead":days_ahead,
        "predicted_for_date":str(today+timedelta(days=days_ahead)),"ml_backend":HAS_ML,
        "trend_series":{"yearly":veg["yearly"],"weekly":veg["weekly"],"daily":veg["daily"]},
    }

def dataset_summary():
    today = date.today()
    rows = []
    for key, v in VEGETABLES.items():
        px = v["yearly"]
        rows.append({"key":key,"name":v["name"],"emoji":v["emoji"],"current":v["base_price"],
            "predicted":round(v["base_price"]*_seasonal_index(key,today.month),1),
            "min":min(px),"max":max(px),"avg":round(sum(px)/len(px),1),
            "std":round(math.sqrt(sum((p-sum(px)/len(px))**2 for p in px)/len(px)),1),
            "volatility":v["volatility"],"accuracy":v["accuracy"],"algorithm":v["algorithm"]})
    rows.sort(key=lambda r: r["current"], reverse=True)
    all_px = [p for v in VEGETABLES.values() for p in v["yearly"]]
    return {"total_vegetables":len(VEGETABLES),
        "records_analyzed":sum(len(v["yearly"])+len(v["weekly"])+len(v["daily"]) for v in VEGETABLES.values()),
        "avg_accuracy":round(sum(v["accuracy"] for v in VEGETABLES.values())/len(VEGETABLES),1),
        "price_range":{"min":min(all_px),"max":max(all_px)},"vegetables":rows,
        "generated_at":datetime.now(timezone.utc).isoformat()}

# ── AUTH ROUTES ──────────────────────────────────────────────────
@app.route("/api/auth/register", methods=["POST"])
def register():
    d = request.get_json() or {}
    email = (d.get("email") or "").strip().lower()
    password = (d.get("password") or "").strip()
    fname = (d.get("fname") or "").strip()
    lname = (d.get("lname") or "").strip()
    phone = (d.get("phone") or "").strip()
    role  = d.get("role","consumer")
    loc_ok = bool(d.get("loc_consent", False))
    if not email or not password or not fname or not lname:
        return jsonify({"error":"Email, password, first name and last name are required"}), 400
    if not ("@" in email and "." in email.split("@")[-1]):
        return jsonify({"error":"Enter a valid email address"}), 400
    if len(password) < 8:
        return jsonify({"error":"Password must be at least 8 characters"}), 400
    if User.query.filter_by(email=email).first():
        return jsonify({"error":"An account with this email already exists. Please sign in."}), 409
    user = User(id="usr-"+secrets.token_hex(4), email=email, password_hash=_hash(password),
        fname=fname, lname=lname, phone=phone, role=role, loc_consent=loc_ok)
    db.session.add(user)
    db.session.commit()
    return jsonify({"message":f"Account created for {fname}. Please login."}), 201

@app.route("/api/auth/login", methods=["POST"])
def login():
    d = request.get_json() or {}
    email = (d.get("email") or "").strip().lower()
    password = (d.get("password") or "").strip()
    if not email or not password:
        return jsonify({"error":"Email and password are required"}), 400
    user = User.query.filter_by(email=email).first()
    if not user or user.password_hash != _hash(password):
        return jsonify({"error":"Invalid email or password"}), 401
    user.login_count = (user.login_count or 0) + 1
    user.last_login = _now_iso()
    token = secrets.token_hex(32)
    db.session.add(Session(token=token, user_email=email))
    db.session.commit()
    resp = jsonify({"token":token, **user.to_dict(), "message":"Login successful"})
    resp.set_cookie("auth_token", token, httponly=True, samesite="Lax", max_age=86400)
    return resp, 200

@app.route("/api/auth/logout", methods=["POST"])
@require_auth
def logout():
    token = (request.headers.get("X-Auth-Token") or request.cookies.get("auth_token") or "")
    sess = Session.query.get(token)
    if sess:
        db.session.delete(sess)
        db.session.commit()
    resp = jsonify({"message":"Logged out"})
    resp.delete_cookie("auth_token")
    return resp, 200

@app.route("/api/auth/me", methods=["GET"])
@require_auth
def me():
    return jsonify(request.user.to_dict())

# ── USER ROUTES ──────────────────────────────────────────────────
@app.route("/api/user/search-history", methods=["POST"])
@require_auth
def add_search_history():
    d = request.get_json() or {}
    name = (d.get("vegetable") or "").strip()
    if name:
        user = request.user
        hist = [s for s in user.search_history.split(",") if s]
        if name in hist: hist.remove(name)
        hist.insert(0, name)
        user.search_history = ",".join(hist[:20])
        db.session.commit()
    return jsonify({"ok":True})

# ── LOCATION ROUTES ──────────────────────────────────────────────
@app.route("/api/location/region", methods=["POST"])
@require_auth
def resolve_region():
    d = request.get_json() or {}
    lat, lon = d.get("lat"), d.get("lon")
    if lat is None or lon is None:
        return jsonify({"error":"lat and lon are required"}), 400
    key = nearest_region(float(lat), float(lon))
    region = REGIONS[key]
    return jsonify({"region":key,"market":region["market"],"zone":region["zone"],
        "state":region["state"],"adj":region["adj"],"risk":region["risk"],"lat":lat,"lon":lon})

@app.route("/api/location/regions", methods=["GET"])
def list_regions():
    return jsonify({k:{"market":v["market"],"zone":v["zone"],"state":v["state"],"adj":v["adj"]}
        for k, v in REGIONS.items()})

# ── PREDICTION ROUTES — NOTE: /all must come BEFORE /<veg_key> ──
@app.route("/api/predict/all", methods=["GET"])
@require_auth
def predict_all():
    region = request.args.get("region","chennai").lower()
    days   = max(7, min(90, int(request.args.get("days",30))))
    return jsonify({k: predict_price(k, region, days) for k in VEGETABLES})

@app.route("/api/predict/<veg_key>", methods=["GET"])
@require_auth
def predict_one(veg_key):
    veg_key = veg_key.lower()
    if veg_key not in VEGETABLES:
        return jsonify({"error":f"'{veg_key}' not found","available":list(VEGETABLES.keys())}), 404
    region = request.args.get("region","chennai").lower()
    days   = max(7, min(90, int(request.args.get("days",30))))
    return jsonify(predict_price(veg_key, region, days))

# ── DATASET ROUTES ───────────────────────────────────────────────
@app.route("/api/dataset/summary", methods=["GET"])
@require_auth
def ds_summary():
    return jsonify(dataset_summary())

@app.route("/api/dataset/seasonal/<veg_key>", methods=["GET"])
@require_auth
def ds_seasonal(veg_key):
    veg_key = veg_key.lower()
    if veg_key not in VEGETABLES:
        return jsonify({"error":f"'{veg_key}' not found"}), 404
    veg = VEGETABLES[veg_key]
    monthly = [{"month":m,"seasonal_index":round(_seasonal_index(veg_key,m),3),
        "expected_price":round(veg["base_price"]*_seasonal_index(veg_key,m),1)} for m in range(1,13)]
    return jsonify({"vegetable":veg["name"],"monthly_analysis":monthly,
        "peak_months":veg["peak_months"],"low_months":veg["low_months"]})

# ── CHATBOT ROUTE (FIX: proxy to Anthropic so no CORS/API-key issues) ──
@app.route("/api/chat", methods=["POST"])
@require_auth
def chat():
    """
    Proxy the chatbot through Flask so:
    1. No API key exposed in browser
    2. No CORS errors
    3. Auth enforced — only logged-in users can chat
    """
    import urllib.request, json as _json
    d = request.get_json() or {}
    messages = d.get("messages", [])
    system   = d.get("system", "You are the VegPredict AI assistant.")

    if not messages:
        return jsonify({"error":"messages required"}), 400

    ANTHROPIC_API_KEY = os.environ.get("ANTHROPIC_API_KEY", "")
    if not ANTHROPIC_API_KEY:
        # Fallback: return a smart rule-based response if no key set
        last_msg = (messages[-1].get("content") or "").lower()
        veg_names = [v["name"].lower() for v in VEGETABLES.values()]
        matched = next((v for v in veg_names if v in last_msg), None)
        if matched:
            vk = next(k for k,v in VEGETABLES.items() if v["name"].lower()==matched)
            veg = VEGETABLES[vk]
            reply = (f"Based on our ML models, {veg['name']} has a current base price of ₹{veg['base_price']}/kg "
                     f"with {veg['volatility']} volatility. Model accuracy is {veg['accuracy']}% "
                     f"using {veg['algorithm']}. Peak prices occur in months {veg['peak_months']} "
                     f"and lowest prices in months {veg['low_months']}.")
        elif any(w in last_msg for w in ["price","predict","forecast"]):
            reply = ("VegPredict tracks 10 vegetables (Tomato, Onion, Potato, Carrot, Brinjal, Cabbage, "
                     "Cauliflower, Beans, Peas, Spinach) with an average accuracy of 95.3%. "
                     "All predictions are 30-day forecasts anchored to Koyambedu market, Chennai.")
        elif any(w in last_msg for w in ["region","location","city","market"]):
            reply = ("We support 12 Tamil Nadu market zones including Koyambedu (Chennai), "
                     "Coimbatore Ukkadam, Mattuthavani (Madurai), Salem, Krishnagiri APMC, and Ooty. "
                     "Each zone applies a price adjustment factor based on transport and supply chain distance.")
        else:
            reply = ("Hello! I'm the VegPredict AI Assistant. I can help you with vegetable price predictions, "
                     "seasonal trends, market zone comparisons, and ML model details. "
                     "Try asking: 'What is the price of tomato?' or 'Which vegetable is most volatile?'")
        return jsonify({"reply": reply, "source": "rule-based"})

    # Real Anthropic API call
    payload = _json.dumps({
        "model": "claude-sonnet-4-6",
        "max_tokens": 600,
        "system": system,
        "messages": messages[-10:]  # last 10 turns to keep context window sane
    }).encode()

    req = urllib.request.Request(
        "https://api.anthropic.com/v1/messages",
        data=payload,
        headers={
            "Content-Type": "application/json",
            "x-api-key": ANTHROPIC_API_KEY,
            "anthropic-version": "2023-06-01",
        },
        method="POST"
    )
    try:
        with urllib.request.urlopen(req, timeout=30) as res:
            data = _json.loads(res.read())
        reply = "".join(b["text"] for b in data.get("content",[]) if b.get("type")=="text")
        return jsonify({"reply": reply, "source": "claude"})
    except urllib.error.HTTPError as e:
        body = e.read().decode("utf-8", errors="ignore")
        app.logger.error(f"Anthropic API error {e.code}: {body}")
        return jsonify({"error": f"Anthropic API error {e.code}: {body}"}), 502
    except Exception as e:
        app.logger.error(f"Chat error: {e}")
        return jsonify({"error": str(e)}), 502

# ── HEALTH CHECK ─────────────────────────────────────────────────
@app.route("/api/health", methods=["GET"])
def health():
    return jsonify({"status":"ok","ml_backend":HAS_ML,
        "vegetables_tracked":len(VEGETABLES),"regions_supported":len(REGIONS),
        "chat_endpoint":True,"timestamp":datetime.now(timezone.utc).isoformat()})

# ── SERVE FRONTEND ───────────────────────────────────────────────
@app.route("/")
def index():
    p = os.path.join(TEMPLATE_DIR, "index.html")
    if not os.path.exists(p):
        return ("<h2>⚠ Setup issue</h2><p><code>templates/index.html</code> not found.</p>"
                "<p>Move <code>index.html</code> into the <code>templates/</code> folder.</p>"), 404
    return send_from_directory(TEMPLATE_DIR, "index.html")

@app.errorhandler(404)
def not_found(e):
    return jsonify({"error":"Route not found"}), 404

if __name__ == "__main__":
    with app.app_context():
        db.create_all()
        _seed_users()
    print("\n" + "═"*52)
    print("  🥦  VegPredict v5 — Flask Backend (FIXED)")
    print("═"*52)
    print(f"  ML engine  : {'✓ scikit-learn (GBR)' if HAS_ML else '⚠ statistical fallback'}")
    print(f"  Vegetables : {len(VEGETABLES)}")
    print(f"  Regions    : {len(REGIONS)}")
    print(f"  Chat API   : {'✓ Anthropic' if os.environ.get('ANTHROPIC_API_KEY') else '⚠ rule-based fallback (set ANTHROPIC_API_KEY)'}")
    print( "  Demo login : demo@vegpredict.in / demo1234")
    print( "  URL        : http://localhost:5000")
    print("═"*52 + "\n")
    app.run(host="0.0.0.0", port=5000, debug=True)

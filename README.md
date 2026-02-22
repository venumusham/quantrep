# QuantRep — Options Backtesting Platform
**Real NSE Data via OpenChart · Black-Scholes Pricing · Free Deployment**

🌐 Frontend → **Netlify** | ⚙️ Backend → **Render**

---

## Repo Structure
```
quantrep/
├── frontend/
│   └── index.html        ← Deployed to Netlify
├── backend/
│   ├── app.py            ← Flask API (OpenChart + Black-Scholes)
│   └── requirements.txt
├── render.yaml           ← Render auto-reads this
├── netlify.toml          ← Netlify auto-reads this
└── .gitignore
```

---

## 🚀 Deploy in 4 Steps

### Step 1 — Push files to GitHub

```bash
git clone https://github.com/venumusham/quantrep.git
cd quantrep
# copy all downloaded files into this folder
git add .
git commit -m "Initial QuantRep deploy"
git push origin main
```

### Step 2 — Deploy Backend to Render

1. Go to https://render.com → Sign up with GitHub
2. New + → Web Service → select `venumusham/quantrep`
3. Render detects `render.yaml` automatically → Apply
4. Note your URL: `https://quantrep-api.onrender.com`

### Step 3 — Update API URL

In `frontend/index.html` replace `quantrep-api` with your actual Render URL, then push.

### Step 4 — Deploy Frontend to Netlify

1. Go to https://netlify.com → Add new site → Import from Git
2. Select `venumusham/quantrep`
3. Netlify reads `netlify.toml` → publish dir = `frontend`
4. Deploy ✅

---

## Local Development

```bash
pip install openchart flask flask-cors numpy gunicorn
cd backend && python app.py
# open frontend/index.html in browser
```

## Keep Render Awake (Free)
Add UptimeRobot monitor on `/api/health` every 14 min → stays warm 24/7.

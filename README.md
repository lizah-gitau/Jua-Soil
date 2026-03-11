# 🌱 Jua Soil — Smallholder Soil Health Advisor

> **Jua Soil tells farmers exactly what their soil needs and what affordable local product to buy in Swahili or English using AI and satellite soil data, in under 30 seconds.**

---

## 🌍 The Problem It Solves

Kenya has over 7 million smallholder farmers. Most of them buy fertilizer based on guesswork because soil testing labs are expensive (KSh 3,500 per sample), far away, and take days to return results. Every season, farmers spend thousands of shillings on products that don't fix their specific soil problem, and harvests suffer.

The same challenge exists across Africa and the developing world: farmers lack affordable, immediate, and actionable soil health guidance in their own language.

Existing digital tools either show raw data that farmers can't interpret, give generic advice without naming specific locally available products, require complex onboarding, or stop at a recommendation without bridging all the way to a purchase decision.

Jua Soil closes all of those gaps in one 30-second interaction.

---

## ✅ What It Does

A farmer opens the Jua Soil web app on any phone or browser: no download, no installation, no registration required. They share their GPS location or type their farm coordinates. Within 30 seconds, they receive a three-part soil health report:

**Part 1 — Soil Summary:** What is healthy and what is deficient in their soil, written in plain language using comparisons a farmer would recognise ("your soil has low nitrogen, which is like not having enough food for your maize to grow strong").

**Part 2 — What To Do:** The name of a specific, locally available fertilizer or soil amendment, how much to use per acre, the approximate price in Kenyan Shillings, and how to apply it in one sentence.

**Part 3 — Best Time To Apply:** Based on the live weather forecast for that exact location, whether to apply now or wait, and why.

Farmers can also upload a photo of their crop and receive a fourth panel with a visual diagnosis — what GPT-4o sees in the image, what stress symptoms it identifies, and what those symptoms likely mean for soil health.

Finally, a "Send to My Phone" button delivers the core recommendation as an SMS so the farmer has it in their pocket even after closing the browser.

---

## 🎯 Target User

**Primary:** Grace Wanjiru, 44 years old, smallholder maize and bean farmer in Meru County, Kenya. She farms 2 acres, owns a basic Android smartphone, uses WhatsApp daily, and has never done a soil test because the nearest lab is a full day's travel away and costs more than she can justify.

**Secondary:** Agronomy students, NGO field officers, and county agricultural extension workers who use the app to advise multiple farmers at once.

---

## 🗺️ Geographic Coverage

Jua Soil is built specifically for Kenyan farmers, with Swahili language support, locally available Kenyan product recommendations, and prices in Kenyan Shillings. Every design decision was made with Grace Wanjiru in mind.

However, because the underlying data sources have broad geographic coverage, the app is functional well beyond Kenya's borders:

- **iSDAsoil Africa API** covers all of sub-Saharan Africa at 30-metre resolution, meaning any farmer in Ghana, Tanzania, Uganda, Ethiopia, or across the continent can receive a meaningful soil report.
- **SoilGrids (global fallback)** provides soil data for any GPS coordinate on Earth, so the app works in South Asia, Latin America, and beyond.
- **OpenWeather API** works for any location in the world.

Scaling Jua Soil to serve farmers in other countries is primarily a content challenge like updating the product recommendations and pricing for local markets is not an engineering one. The architecture is already global.

---

## 🔧 Microsoft Technologies Used

- **Azure AI Foundry + Model Router** — GPT-4o deployment for soil report generation and crop photo vision analysis; Model Router handles intelligent routing between GPT-4o and lighter models for simpler tasks
- **Microsoft Agent Framework + Agent HQ** — Multi-step AI agent that orchestrates three tools in sequence: soil data retrieval, weather data retrieval, and report generation; Agent HQ provides real-time monitoring of tool calls and agent reasoning
- **Azure MCP Server** — Manages the connections to iSDAsoil and OpenWeather as registered data source connectors
- **Azure App Service** — Hosts the Python Flask backend server with automatic GitHub deployment pipeline
- **Azure Static Web Apps** — Hosts the mobile-first frontend on a globally distributed CDN, ensuring fast load times on 3G connections in rural Kenya
- **Azure AI Content Safety** — Reviews all AI-generated reports before delivery to ensure no harmful or misleading content reaches a farmer
- **Azure Monitor + Application Insights** — Live telemetry dashboard showing every API call, response time, and error rate in the running application
- **Africa's Talking** — SMS delivery of the core fertilizer recommendation to the farmer's phone

---

## 🌐 Free Data Sources

- **iSDAsoil Africa API** (api.isda-africa.com) — Soil nutrient data at 30-metre resolution across Africa: nitrogen, phosphorus, potassium, pH, organic carbon, and soil texture
- **SoilGrids API** (rest.isric.org) — Global soil data fallback for locations outside iSDAsoil coverage
- **OpenWeather API** — Current weather conditions and 7-day forecast by GPS coordinates
- **OpenStreetMap Nominatim** — Free geocoding, converts a town or village name into GPS coordinates with no API key required

---

## 🏗️ Architecture Overview

```
Farmer's Phone (Browser)
        ↓
Azure Static Web Apps (Frontend — HTML/CSS/JS)
        ↓
Azure App Service (Flask Backend)
        ↓
Microsoft Agent Framework
    ├── Tool 1: iSDAsoil API (via Azure MCP Server) → Soil nutrients
    ├── Tool 2: OpenWeather API (via Azure MCP Server) → Weather forecast
    └── Tool 3: GPT-4o (via Azure AI Foundry) → Plain-language report
        ↓
Azure AI Content Safety → Safety check
        ↓
Report delivered to farmer's browser
        ↓ (optional)
Africa's Talking SMS → Farmer's phone
```

---

## 🚀 How to Run Locally

**Prerequisites:** Python 3.11+, a free Azure account, a free OpenWeather API key, and a free iSDAsoil API key.

**Step 1 — Clone the repository:**
```bash
git clone https://github.com/lizah-gitau/jua-soil.git
cd jua-soil
```

**Step 2 — Install dependencies:**
```bash
pip install -r backend/requirements.txt
```

**Step 3 — Create your .env file** in the project root:
```
AZURE_OPENAI_ENDPOINT=https://your-project.openai.azure.com/
AZURE_OPENAI_KEY=your-key-here
AZURE_OPENAI_DEPLOYMENT=gpt-4o
OPENWEATHER_API_KEY=your-openweather-key
ISDA_API_KEY=your-isda-key
```

**Step 4 — Start the backend:**
```bash
python backend/app.py
```

**Step 5 — Open the frontend:** Open `frontend/index.html` in your browser. Test with Nakuru, Kenya coordinates: latitude `-0.3031`, longitude `36.0800`.

---

## 🌐 Try It Live

**Live App:** https://green-hill-020c2df0f.2.azurestaticapps.net

**Test coordinates to try:**
- Nakuru, Kenya: `-0.3031, 36.0800`
- Kisumu, Kenya: `-0.1022, 34.7617`
- Meru, Kenya: `0.0467, 37.6494`
- Eldoret, Kenya: `0.5200, 35.2698`
- Kitale, Kenya: `1.0154, 35.0062`

---

## 📁 Project Structure

```
jua-soil/
├── backend/
│   ├── app.py          # Flask web server — API routes and request handling
│   ├── agent.py        # AI agent — soil fetcher, weather fetcher, report generator
│   └── requirements.txt
├── frontend/
│   └── index.html      # Complete mobile-first single-page app
├── tests/
│   └── test_apis.py    # API connection tests for iSDAsoil and OpenWeather
└── README.md
```

---

## 🏆 Hackathon

**Event:** Microsoft AI Dev Days Hackathon
**Track:** Environment & Sustainability
**Target Region:** Kenya (scalable to broader Africa and globally)

---

## 📋 The One-Sentence Standard

*Jua Soil is done when a person with no farming or tech background can open a link, enter a Kenyan GPS coordinate or upload a crop photo, and receive a clear, specific, Swahili or English recommendation telling them exactly what to buy and why in under a minute.*

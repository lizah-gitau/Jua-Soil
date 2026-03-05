import os
import requests
import json
from openai import AzureOpenAI
from datetime import datetime
from dotenv import load_dotenv

from azure.ai.inference.tracing import AIInferenceInstrumentor
from azure.monitor.opentelemetry import configure_azure_monitor

load_dotenv()

# ── Set up tracing to Azure AI Foundry ───────────────────────────────
# This connects your agent's activity logs to the Foundry tracing dashboard.
# Every time your agent makes a tool call — fetching soil data, fetching
# weather, calling GPT-4o — a record of that call will appear in real time
# on the Tracing page in your Foundry project. Think of this as switching
# on the security cameras that watch your agent work.
# The connection string tells Azure Monitor exactly which Application Insights
# resource to send the data to — the one we created called
# JuaSoilProject-insights.
app_insights_connection = os.getenv("APPLICATIONINSIGHTS_CONNECTION_STRING")
if app_insights_connection:
    configure_azure_monitor(connection_string=app_insights_connection)
    AIInferenceInstrumentor().instrument()
    print("✓ Tracing enabled — agent activity will appear in Foundry dashboard")
else:
    print("⚠ Tracing not configured — APPLICATIONINSIGHTS_CONNECTION_STRING not found in .env")


# ── Connect to Azure AI Foundry ───────────────────────────────────────
# os.getenv() is how we safely read credentials without hardcoding them.
# Instead of writing the actual key in the code, we ask the environment
# "what is the value stored under this label?" and it looks it up in .env.
# This is the professional way to handle credentials in any real application.
client = AzureOpenAI(
    azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
    api_key=os.getenv("AZURE_OPENAI_KEY"),
    api_version="2024-12-01-preview"
)

# We read the deployment name from .env as well so it's easy to change
# in one place if you ever need to swap models in the future.
DEPLOYMENT = os.getenv("AZURE_OPENAI_DEPLOYMENT", "gpt-4o")


# ── TOOL 1: Fetch soil data ───────────────────────────────────────────
# This function takes GPS coordinates and returns soil nutrient data.
# The "-> dict" part just means "this function returns a dictionary (a
# collection of key-value pairs)" — like a little data package.
def get_soil_data(lat: float, lon: float) -> dict:
    """
    Fetches soil nutrient data for any location on Earth.

    The strategy is to try iSDAsoil first because it has higher resolution
    (30 metres) and is specifically calibrated for African soils, making it
    more accurate for Kenyan and African farmers. If iSDAsoil doesn't cover
    the location (i.e. the farm is outside Africa), we automatically fall
    back to SoilGrids, which covers the entire planet at 250-metre resolution.

    Think of it like having two maps — a detailed hand-drawn map of your
    neighbourhood (iSDAsoil for Africa) and a standard atlas that covers
    the whole world (SoilGrids). You use the detailed neighbourhood map
    when you can, and reach for the atlas when the farm is somewhere the
    neighbourhood map doesn't cover.
    """

    # ── Attempt 1: Try iSDAsoil first (best for African locations) ────
    email = os.getenv("ISDA_EMAIL")
    password = os.getenv("ISDA_PASSWORD")

    try:
        # Log in to get a fresh access token for this request
        login_response = requests.post(
            "https://api.isda-africa.com/login",
            data={"username": email, "password": password},
            timeout=10
        )
        login_response.raise_for_status()
        token = login_response.json().get("access_token")

        url = (
            f"https://api.isda-africa.com/isdasoil/v2/soilproperty"
            f"?lat={lat}&lon={lon}"
            f"&property=nitrogen_total"
            f"&property=phosphorus_extractable"
            f"&property=potassium_extractable"
            f"&property=ph"
            f"&property=carbon_organic"
            f"&depth=0-20"
        )

        headers = {"Authorization": f"Bearer {token}"}
        r = requests.get(url, headers=headers, timeout=10)
        r.raise_for_status()
        data = r.json()

        # We check for the "property" key because iSDAsoil always includes
        # it when real soil data exists for a location. If the farm is
        # outside Africa, iSDAsoil might respond with 200 OK but return
        # empty property data — this check catches that situation and
        # triggers the SoilGrids fallback rather than passing empty data
        # to GPT-4o.
        if "property" in data and data["property"]:
            print(" ℹ Soil data source: iSDAsoil (Africa, high resolution)")
            data["_source"] = "iSDAsoil"
            return data
        else:
            print(" ℹ iSDAsoil returned no data for this location — trying SoilGrids...")

    except Exception as e:
        print(f" ⚠ iSDAsoil unavailable ({e}) — trying SoilGrids...")

    # ── Attempt 2: SoilGrids global fallback ──────────────────────────
    # SoilGrids requires NO API key — it's completely free and open access.
    # It covers the entire planet at 250-metre resolution, which is slightly
    # less detailed than iSDAsoil's 30-metre African coverage but perfectly
    # sufficient for practical farming advice anywhere in the world.
    try:
        print(" ℹ Fetching from SoilGrids (global coverage, no API key needed)...")

        soilgrids_url = (
            f"https://rest.isric.org/soilgrids/v2.0/properties/query"
            f"?lon={lon}&lat={lat}"
            f"&property=nitrogen"
            f"&property=phh2o"
            f"&property=soc"
            f"&property=clay"
            f"&depth=0-5cm"
            f"&depth=5-15cm"
            f"&value=mean"
        )

        # SoilGrids is a public API but it can be slow sometimes — we give
        # it 20 seconds before giving up, which is more generous than the
        # 10 seconds we give iSDAsoil.
        r = requests.get(soilgrids_url, timeout=20)
        r.raise_for_status()
        raw_data = r.json()

        # ── NEW: Pass the raw SoilGrids response through our normaliser ──
        # This is the key change from the previous version. Instead of
        # sending SoilGrids' complex technical format directly to GPT-4o,
        # we first translate it into the same clean, clearly labelled
        # structure that iSDAsoil produces. GPT-4o then receives
        # consistently readable data regardless of which source was used,
        # which produces equally strong advice for farms outside Africa.
        normalised = normalise_soilgrids_data(raw_data)
        print(" ℹ Soil data source: SoilGrids (global, normalised)")
        return normalised

    except Exception as e:
        return {"error": f"Could not retrieve soil data from any source: {str(e)}"}


# ── NEW FUNCTION: SoilGrids Data Normaliser ───────────────────────────
# This function is the translator between SoilGrids' raw technical format
# and the clean, human-readable structure that GPT-4o works best with.
#
# To understand why this is needed, think of it this way. iSDAsoil hands
# you a report that says "Nitrogen: Low. pH: 6.2. Organic Carbon: Low."
# SoilGrids hands you a report that says "nitrogen cg/kg at 0-5cm depth,
# mean value: 142" — the same information, but buried inside a complex
# nested structure with unusual units that require mathematical conversion.
# This function does that conversion and translation automatically so
# GPT-4o always receives clear, labelled, interpreted soil data.
def normalise_soilgrids_data(raw: dict) -> dict:
    """
    Translates SoilGrids API response into the same clean structure
    that iSDAsoil produces, so GPT-4o receives consistently formatted
    soil data regardless of which database was used.

    SoilGrids has three quirks we handle here:
      1. Nitrogen is in cg/kg — we divide by 100 to get g/kg
      2. pH is stored multiplied by 10 — we divide by 10 to get real pH
      3. Data comes in two separate depth layers — we average them together
         to approximate iSDAsoil's single 0-20cm measurement
    """

    # This helper function digs into SoilGrids' deeply nested structure
    # to extract the actual numeric value for a given property and depth.
    # SoilGrids stores values inside layers > depths > values — imagine
    # a filing cabinet with three levels of drawers. This function opens
    # all three drawers and retrieves what's inside, returning None if
    # any drawer is missing rather than crashing.
    def extract_value(properties, property_name, depth_label):
        try:
            for layer in properties.get("layers", []):
                if layer.get("name") == property_name:
                    for depth in layer.get("depths", []):
                        if depth.get("label") == depth_label:
                            return depth.get("values", {}).get("mean")
        except Exception:
            pass
        return None

    props = raw.get("properties", {})

    # Extract raw values from both depth layers (0-5cm and 5-15cm).
    # We'll average them to approximate iSDAsoil's 0-20cm depth range,
    # which covers the root zone where nutrients matter most for crops.
    nitrogen_0_5  = extract_value(props, "nitrogen", "0-5cm")
    nitrogen_5_15 = extract_value(props, "nitrogen", "5-15cm")
    ph_0_5        = extract_value(props, "phh2o",    "0-5cm")
    ph_5_15       = extract_value(props, "phh2o",    "5-15cm")
    soc_0_5       = extract_value(props, "soc",      "0-5cm")
    soc_5_15      = extract_value(props, "soc",      "5-15cm")
    clay_0_5      = extract_value(props, "clay",     "0-5cm")
    clay_5_15     = extract_value(props, "clay",     "5-15cm")

    # Average the two depth layers for each property.
    # If one layer returned None (missing data), we fall back to just
    # the other layer rather than failing entirely. The "or 0" at the
    # end is a last resort to avoid crashing on completely missing data.
    def avg(a, b):
        if a is not None and b is not None:
            return round((a + b) / 2, 2)
        return a or b or 0

    raw_nitrogen = avg(nitrogen_0_5, nitrogen_5_15)
    raw_ph       = avg(ph_0_5, ph_5_15)
    raw_soc      = avg(soc_0_5, soc_5_15)
    raw_clay     = avg(clay_0_5, clay_5_15)

    # Apply the unit conversions that SoilGrids requires.
    # Nitrogen: cg/kg divided by 100 gives g/kg  (e.g. 142 becomes 1.42)
    # pH: the stored value divided by 10 gives real pH (e.g. 65 becomes 6.5)
    # SOC (soil organic carbon): dg/kg divided by 10 gives g/kg
    # Clay: already in g/kg, no conversion needed
    nitrogen_gkg = round(raw_nitrogen / 100, 3) if raw_nitrogen else 0
    ph_real      = round(raw_ph / 10, 1)        if raw_ph      else 0
    soc_gkg      = round(raw_soc / 10, 2)       if raw_soc     else 0
    clay_gkg     = raw_clay

    # Interpret the numeric values into plain English categories.
    # These thresholds come from standard agronomic benchmarks used
    # by soil scientists worldwide. Giving GPT-4o "Low (deficient)"
    # rather than just "1.2" means it can write better farmer-facing
    # language without needing to calculate what the number means.
    def nitrogen_level(n):
        if n < 1.0: return "Low (deficient — crops likely to show yellowing and poor growth)"
        if n < 2.0: return "Medium (adequate for most crops)"
        return "High (good nitrogen availability)"

    def ph_level(p):
        if p < 5.5: return f"{p} (Acidic — may limit nutrient availability, consider liming)"
        if p < 6.5: return f"{p} (Slightly acidic — suitable for most crops)"
        if p < 7.5: return f"{p} (Neutral — ideal for most crops)"
        return f"{p} (Alkaline — may limit iron and zinc availability)"

    def soc_level(s):
        if s < 10: return "Low (poor organic matter — soil will benefit from compost or organic amendments)"
        if s < 20: return "Medium (moderate organic matter)"
        return "High (good organic matter content)"

    def clay_level(c):
        if c < 180: return "Sandy or loamy (drains quickly, may need more frequent watering)"
        if c < 350: return "Loamy (good structure, holds moisture and nutrients well)"
        return "Clay-heavy (drains slowly, risk of waterlogging in heavy rain)"

    # Build the normalised output in a format that mirrors iSDAsoil's
    # clean, clearly labelled structure. GPT-4o will read this like a
    # well-organised brief rather than a raw data dump, which is what
    # produces the confident, specific soil summaries we want.
    return {
        "_source": "SoilGrids (global, normalised)",
        "location_note": "Global soil data from the SoilGrids international database",
        "soil_properties": {
            "nitrogen_total": {
                "value_g_per_kg": nitrogen_gkg,
                "interpretation": nitrogen_level(nitrogen_gkg)
            },
            "ph": {
                "value": ph_real,
                "interpretation": ph_level(ph_real)
            },
            "organic_carbon": {
                "value_g_per_kg": soc_gkg,
                "interpretation": soc_level(soc_gkg)
            },
            "clay_content": {
                "value_g_per_kg": clay_gkg,
                "interpretation": clay_level(clay_gkg)
            }
        },
        "depth_sampled": "0-15cm average (approximates root zone)",
        "data_quality_note": (
            "SoilGrids provides global coverage at 250m resolution. "
            "Values are means averaged from two depth layers (0-5cm and 5-15cm). "
            "For African farms, iSDAsoil provides higher precision at 30m resolution."
        )
    }


# ── TOOL 2: Fetch weather data from OpenWeather ───────────────────────
# This function takes GPS coordinates and returns current weather
# plus a short forecast — which tells us whether it's a good time
# to apply fertilizer or whether rain is coming that might wash it away.
def get_weather_data(lat: float, lon: float) -> dict:
    """Fetches current weather and short forecast for a given location."""

    key = os.getenv("OPENWEATHER_API_KEY")
    url = (
        f"https://api.openweathermap.org/data/2.5/forecast"
        f"?lat={lat}&lon={lon}&appid={key}&units=metric"
    )

    try:
        r = requests.get(url, timeout=10)
        r.raise_for_status()
        data = r.json()

        # The OpenWeather forecast returns data in 3-hour intervals.
        # We only take the first 8 entries (covering the next 24 hours)
        # to keep the data package lean — we don't need a full week's
        # forecast for a same-day planting decision.
        return {
            "current": data["list"][0],
            "forecast_24hrs": data["list"][:8],
            "city": data["city"]["name"]
        }

    except Exception as e:
        return {"error": str(e)}


# ── TOOL 3: Generate the soil report using GPT-4o ────────────────────
# This is the heart of Jua Soil. It takes the soil and weather data
# we've collected and asks GPT-4o to write a plain-language report
# that a farmer can immediately understand and act on.
def generate_report(soil_data: dict, weather_data: dict, language: str = "en", lat: float = 0, lon: float = 0) -> str:
    """Uses GPT-4o to write a plain-language soil health report."""

    # We tell the AI what language to write in based on the farmer's choice.
    # The system prompt is like a job description — it sets the AI's identity
    # and the rules it must follow for this entire conversation.
    if language == "sw":
        lang_instruction = "Write entirely in natural Swahili for Kenyan farmers. Use everyday farming language, not technical jargon or dictionary translations."
    else:
        lang_instruction = "Write in clear, simple English that a farmer with secondary school education can easily understand."

    # The system prompt now describes Jua Soil as a global advisor
    # rather than Kenya-only, since the app now serves farmers anywhere
    # on Earth. Kenya remains the primary focus, but the AI is no longer
    # constrained to thinking only about Kenyan context when a farm
    # outside Africa is being analysed.
    system_prompt = (
        f"You are Jua Soil, a friendly and practical agricultural advisor. "
        f"Your primary focus is smallholder farmers in Kenya and Africa, "
        f"but you serve farmers anywhere in the world. "
        f"Always tailor your advice to the specific location of the farm "
        f"based on the coordinates provided. {lang_instruction}"
    )

    # Today's date helps GPT-4o give seasonally relevant advice —
    # for example, knowing it's March (long rains season in Kenya) helps
    # it say "apply now before the rains arrive" rather than giving
    # generic advice that ignores the current season.
    today = datetime.now().strftime("%B %d, %Y")

    # The user prompt is the actual task we're giving the AI.
    # Notice how specific the instructions are — we're not just saying
    # "write a report." We're telling it exactly what three sections to
    # write, what language to use, what good phrasing looks like, and
    # what the maximum length is. This specificity is what separates
    # a useful AI output from a vague, unhelpful one.
    user_prompt = f"""
Today is {today}.
The farm is located at coordinates: latitude {lat}, longitude {lon}.

Here is the soil data for this farm
(data source: {soil_data.get('_source', 'satellite database')}):
{json.dumps(soil_data, indent=2)}

Here is the current weather and forecast:
{json.dumps(weather_data, indent=2)}

Please write a soil health report with exactly three sections.

SECTION 1 - SOIL SUMMARY:
In 3 to 4 sentences, explain what is healthy and what is deficient
in plain language a farmer can understand. Do not use chemical symbols
or units like g/kg. Use comparisons a farmer would recognise.
Good example: 'Your soil has low nitrogen, which is like not having
enough food for your maize to grow strong and tall.'

SECTION 2 - WHAT TO DO:
First, use the coordinates provided to identify what country or region
this farm is in. Then name ONE specific fertilizer or soil amendment
that is commonly available in that specific country or region.
Use a real local brand name or the most commonly used generic name
that farmers can find at their nearest agricultural supply store.
Tell the farmer: the product name, how much to use per acre or
hectare (use whichever measurement unit is standard in that country),
the approximate price in the LOCAL CURRENCY of that country,
and how to apply it in one simple sentence.
If you genuinely cannot determine the country from the coordinates,
recommend a universally available product like NPK fertilizer
with general guidance.

SECTION 3 - BEST TIME TO APPLY:
Based on the weather forecast provided, tell the farmer in one
sentence whether to apply now or wait, and why.

Keep the entire report under 250 words.
"""

    response = client.chat.completions.create(
        model=DEPLOYMENT,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ],
        max_tokens=500,
        # temperature controls how creative vs consistent the AI is.
        # 0.3 means "be consistent and factual" rather than creative —
        # we want the same quality of practical advice every single time,
        # not poetic variation. Think of 0 as fully robotic and 1 as fully creative.
        temperature=0.3
    )

    return response.choices[0].message.content


# ── THE MASTER AGENT FUNCTION ─────────────────────────────────────────
# This is the function that the rest of the app will call.
# Give it a GPS coordinate and a language preference, and it runs all
# three tools in the correct sequence and returns a complete report.
# This is the "personal assistant" behaviour we talked about earlier —
# one instruction triggers multiple coordinated actions automatically.
def run_jua_soil_agent(lat: float, lon: float, language: str = "en") -> dict:
    """
    The main agent function. Give it GPS coordinates and a language,
    and it returns a complete soil health report by running all three
    tools in sequence.
    """

    print(f"Agent starting for lat={lat}, lon={lon}, language={language}")

    # Step 1: Fetch soil data — the get_soil_data function itself will
    # print which source it ends up using (iSDAsoil or SoilGrids),
    # so we keep this message generic rather than assuming a specific source.
    print(" → Fetching soil data...")
    soil = get_soil_data(lat, lon)

    # Step 2: Fetch weather data for the same location
    print(" → Fetching weather data from OpenWeather...")
    weather = get_weather_data(lat, lon)

    # Step 3: Generate the AI report using both data sources.
    # We pass lat and lon through to the report generator so GPT-4o
    # knows which country the farm is in and can give locally relevant
    # product recommendations and pricing.
    print(" → Generating AI report with GPT-4o...")
    report = generate_report(soil, weather, language, lat, lon)

    print(" ✓ Agent complete.")

    # We return everything — the report plus the raw data —
    # because the backend server will want to send the report to the farmer
    # and might also want to log the raw data for debugging purposes.
    return {
        "report": report,
        "soil_raw": soil,
        "weather_raw": weather,
        "language": language
    }


# ── QUICK TEST ────────────────────────────────────────────────────────
# This block only runs when you call this file directly from the terminal
# using "python backend/agent.py". When other files import this file,
# this block is skipped automatically — Python checks __name__ to decide.
# Think of it as a "run me only if I'm the main event, not a supporting act."
if __name__ == "__main__":
    print("Testing Jua Soil agent...\n")

    # Test 1: African farm near Nakuru, Kenya — should use iSDAsoil
    print("=== TEST 1: Nakuru, Kenya (expects iSDAsoil) ===")
    result = run_jua_soil_agent(-0.3031, 36.0800, "en")
    print("\n=== ENGLISH REPORT ===")
    print(result["report"])

    print("\n=== SWAHILI REPORT ===")
    result_sw = run_jua_soil_agent(-0.3031, 36.0800, "sw")
    print(result_sw["report"])

    # Test 2: Farm in New Delhi, India — should fall back to SoilGrids,
    # run through the normaliser, and produce a confident specific report
    # with Indian fertilizer recommendations priced in Indian Rupees
    print("\n=== TEST 2: New Delhi, India (expects SoilGrids + normalisation) ===")
    result_india = run_jua_soil_agent(28.6139, 77.2090, "en")
    print(result_india["report"])
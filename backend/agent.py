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
# resource to send the data to — the one we just created called
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


# ── TOOL 1: Fetch soil data from iSDAsoil ────────────────────────────
# This function takes GPS coordinates and returns soil nutrient data.
# The "-> dict" part just means "this function returns a dictionary (a
# collection of key-value pairs)" — like a little data package.
def get_soil_data(lat: float, lon: float) -> dict:
    """Fetches soil nutrient data for a given GPS location from iSDAsoil."""

    # First we need to log in to iSDAsoil to get a token —
    # just like how you need to sign in before using certain websites.
    # We load the email and password from .env, never from the code itself.
    email = os.getenv("ISDA_EMAIL")
    password = os.getenv("ISDA_PASSWORD")

    try:
        # Step 1: Log in and get our temporary access token
        login_response = requests.post(
            "https://api.isda-africa.com/login",
            data={"username": email, "password": password},
            timeout=10
        )
        login_response.raise_for_status()
        token = login_response.json().get("access_token")

        # Step 2: Use that token to request soil data for our coordinates.
        # We're asking for the five most important soil properties for farming:
        # nitrogen (plant food), phosphorus (root development), potassium
        # (disease resistance), pH (soil acidity), and organic carbon
        # (soil health indicator). The depth 0-20cm is the root zone.
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
        return r.json()

    except Exception as e:
        # If anything goes wrong — network error, wrong credentials, etc. —
        # we return a dictionary with an "error" key rather than crashing.
        # This means the agent can handle the failure gracefully rather than
        # stopping completely.
        return {"error": str(e)}


#  TOOL 2: Fetch weather data from OpenWeather 
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
def generate_report(soil_data: dict, weather_data: dict, language: str = "en") -> str:
    """Uses GPT-4o to write a plain-language soil health report."""

    # We tell the AI what language to write in based on the farmer's choice.
    # The system prompt is like a job description — it sets the AI's identity
    # and the rules it must follow for this entire conversation.
    if language == "sw":
        lang_instruction = "Write entirely in natural Swahili for Kenyan farmers. Use everyday farming language, not technical jargon or dictionary translations."
    else:
        lang_instruction = "Write in clear, simple English that a farmer with secondary school education can easily understand."

    system_prompt = f"You are Jua Soil, a friendly and practical agricultural advisor for Kenyan smallholder farmers. {lang_instruction}"

    # Today's date helps GPT-4o give seasonally relevant advice —
    # for example, knowing it's March (long rains season) helps it say
    # "apply now before the rains arrive" rather than giving generic advice.
    today = datetime.now().strftime("%B %d, %Y")

    # The user prompt is the actual task we're giving the AI.
    # Notice how specific the instructions are — we're not just saying
    # "write a report." We're telling it exactly what three sections to write,
    # what language to use in each, what examples of good phrasing look like,
    # and what the maximum length is. This specificity is what separates
    # a useful AI output from a vague, unhelpful one.
    user_prompt = f"""
Today is {today}.

Here is the soil data for this farm:
{json.dumps(soil_data, indent=2)}

Here is the current weather and forecast:
{json.dumps(weather_data, indent=2)}

Please write a soil health report with exactly three sections.
Each section must start with its label on its own line.

SECTION 1 - SOIL SUMMARY:
In 3 to 4 sentences, explain what is healthy and what is deficient
in plain language a farmer can understand. Do not use chemical symbols
or units like g/kg. Use comparisons a farmer would recognise.
Good example: 'Your soil has low nitrogen, which is like not having
enough food for your maize to grow strong and tall.'

SECTION 2 - WHAT TO DO:
Name ONE specific fertilizer available in Kenya by its real brand name
that farmers find at agro-vets, such as: CAN (Calcium Ammonium Nitrate),
DAP (Di-Ammonium Phosphate), Mavuno Planting, NPK 17:17:17, or Urea.
Tell the farmer: the product name, how much to use per acre,
the approximate price in Kenyan Shillings, and how to apply it
in one simple sentence.

SECTION 3 - BEST TIME TO APPLY:
Based on the weather forecast, tell the farmer in one sentence
whether to apply now or wait, and why.

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

    # Step 1: Fetch soil data
    print(" → Fetching soil data from iSDAsoil...")
    soil = get_soil_data(lat, lon)

    # Step 2: Fetch weather data
    print(" → Fetching weather data from OpenWeather...")
    weather = get_weather_data(lat, lon)

    # Step 3: Generate the AI report using both data sources
    print(" → Generating AI report with GPT-4o...")
    report = generate_report(soil, weather, language)

    print(" ✓ Agent complete.")

    # We return everything — the report, plus the raw data —
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
    print("Testing Jua Soil agent with Nakuru farm coordinates...\n")

    # Test with a farm near Nakuru, Kenya
    result = run_jua_soil_agent(-0.3031, 36.0800, "en")
    print("\n=== ENGLISH REPORT ===")
    print(result["report"])

    print("\n=== SWAHILI REPORT ===")
    result_sw = run_jua_soil_agent(-0.3031, 36.0800, "sw")
    print(result_sw["report"])
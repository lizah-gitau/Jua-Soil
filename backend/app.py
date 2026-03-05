from flask import Flask, request, jsonify
from flask_cors import CORS
import os
from dotenv import load_dotenv
from agent import run_jua_soil_agent

# Load all credentials from our .env file before anything else runs
load_dotenv()

# Create the Flask application — think of this as opening the restaurant
# for business. The __name__ argument tells Flask where to find resources
# relative to this file.
app = Flask(__name__)

# CORS stands for Cross-Origin Resource Sharing. Without this, browsers
# enforce a security rule that prevents a webpage hosted on one address
# (your Azure Static Web App) from making requests to a server on a
# different address (your Azure App Service backend). Think of CORS as
# the permission slip that says "yes, requests from other addresses
# are welcome here." Without it, every request from your frontend
# would be blocked by the browser before it even reached Flask.
CORS(app)


# ── ROUTE 1: Health Check ─────────────────────────────────────────────
# This is the simplest possible route — it just confirms the server
# is running. Azure App Service will ping this route automatically
# every few minutes to verify your app is healthy. If it stops
# responding, Azure will restart your app automatically.
@app.route('/health', methods=['GET'])
def health_check():
    return jsonify({
        "status": "ok",
        "app": "Jua Soil",
        "version": "1.0.0"
    })


# ── ROUTE 2: Analyse Soil ─────────────────────────────────────────────
# This is the main route that powers the entire farmer experience.
# It receives the farmer's GPS coordinates and language preference,
# passes them to the agent, and returns the finished report.
# The @app.route decorator is what tells Flask "when a POST request
# arrives at /api/analyse, run the function below it."
@app.route('/api/analyse', methods=['POST'])
def analyse():
    try:
        # request.get_json() reads the data the farmer's browser sent.
        # It's like opening the envelope that arrived at your door.
        data = request.get_json()

        # Extract the three values we need from the request.
        # The float() conversion turns the text "-0.3031" into the
        # actual number -0.3031 that our agent's functions expect.
        lat = float(data.get('latitude'))
        lon = float(data.get('longitude'))

        # If no language is specified, default to English.
        # This means if someone forgets to include the language field,
        # the app doesn't crash — it just chooses English gracefully.
        language = data.get('language', 'en')

        # Global coordinate validation — confirms these are valid Earth
        # coordinates so we don't waste an API call on nonsense input.
        # Valid latitude runs from -90 (South Pole) to 90 (North Pole).
        # Valid longitude runs from -180 (far west Pacific) to 180 (far east).
        # Any real GPS coordinate on Earth falls within these ranges,
        # which means this app now works for farmers anywhere in the world —
        # not just Kenya. iSDAsoil handles African locations with high
        # precision, and SoilGrids handles everywhere else as a fallback.
        if not (-90 <= lat <= 90 and -180 <= lon <= 180):
            return jsonify({
                "error": "Please enter valid GPS coordinates.",
                "hint": "Latitude must be between -90 and 90, longitude between -180 and 180"
            }), 400

        # This is the one line that does all the real work —
        # calling your agent, which fetches soil data, fetches weather,
        # and generates the AI report. Everything else in this function
        # is just receiving the request and packaging the response.
        result = run_jua_soil_agent(lat, lon, language)

        # If the agent returned an error rather than a report,
        # we catch that and send a clear error message to the farmer
        # rather than sending them a confusing raw error object.
        if result.get('error'):
            return jsonify({
                'error': result['error']
            }), 500

        # Send the finished report back to the farmer's browser.
        # jsonify() converts our Python dictionary into JSON format,
        # which is the standard language that browsers and servers
        # use to exchange data.
        return jsonify({
            "success": True,
            "report": result["report"],
            "language": language
        })

    except ValueError:
        # This catches cases where latitude or longitude couldn't be
        # converted to a number — for example if someone typed "Nakuru"
        # in the latitude field instead of a number.
        return jsonify({
            'error': 'Invalid coordinates. Please enter numbers.'
        }), 400

    except Exception as e:
        # This catches any other unexpected error and returns a
        # human-readable message rather than crashing the server.
        return jsonify({
            'error': f'Something went wrong: {str(e)}'
        }), 500


# ── ROUTE 3: Analyse Crop Photo ───────────────────────────────────────
# This route handles the optional photo analysis feature.
# The farmer's browser converts their photo into a base64 string
# (essentially turning the image into a very long text string)
# and sends it here. We pass it directly to GPT-4o's vision capability.
@app.route('/api/analyse-photo', methods=['POST'])
def analyse_photo():
    try:
        import base64
        from openai import AzureOpenAI

        data = request.get_json()

        # The photo arrives as a base64-encoded string.
        # Base64 is just a way of representing binary image data
        # as plain text so it can travel safely inside a JSON request.
        photo_b64 = data.get('photo')
        lat = float(data.get('latitude', -0.3031))
        lon = float(data.get('longitude', 36.0800))
        language = data.get('language', 'en')

        # Create a fresh connection to GPT-4o for the vision call.
        # We use the same credentials from .env as in agent.py.
        ai = AzureOpenAI(
            azure_endpoint=os.getenv('AZURE_OPENAI_ENDPOINT'),
            api_key=os.getenv('AZURE_OPENAI_KEY'),
            api_version="2024-12-01-preview"
        )

        lang_note = (
            'Respond in natural Swahili.' if language == 'sw'
            else 'Respond in clear, simple English.'
        )

        # We send GPT-4o two things in the same message — a text prompt
        # explaining what we want, and the actual image. GPT-4o's multimodal
        # capability means it can read both text and images simultaneously
        # and reason about them together in one response.
        response = ai.chat.completions.create(
            model=os.getenv('AZURE_OPENAI_DEPLOYMENT', 'gpt-4o'),
            messages=[{
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": (
                            f"You are Jua Soil, an agricultural advisor. {lang_note} "
                            "Look at this crop photo. In 2-3 sentences, describe: "
                            "1) What crop this appears to be. "
                            "2) Any visible signs of stress, disease, or nutrient deficiency. "
                            "3) What this might mean for the farmer in plain language. "
                            "Be practical and specific."
                        )
                    },
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{photo_b64}"
                        }
                    }
                ]
            }],
            max_tokens=300
        )

        return jsonify({
            "success": True,
            "photo_analysis": response.choices[0].message.content
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 500


# ── START THE SERVER ──────────────────────────────────────────────────
# This block only runs when you start the server directly from the
# terminal with "python backend/app.py". When Azure App Service runs
# your app in production, it uses gunicorn instead, which imports
# this file rather than running it directly — so this block is skipped
# in production and only used for local development testing.
if __name__ == '__main__':
    port = int(os.getenv('PORT', 5000))
    print(f"Jua Soil backend starting on port {port}")
    print(f"Health check available at: http://localhost:{port}/health")
    app.run(host='0.0.0.0', port=port, debug=True)
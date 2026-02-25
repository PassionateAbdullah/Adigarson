import os
import json
import re
from typing import Dict, Any, Tuple
from dotenv import load_dotenv
import google.generativeai as genai

load_dotenv()

def run_digaxy_ai(user_input: str, session_state: Dict[str, Any]) -> Tuple[str, Dict[str, Any]]:
    """
    SINGLE EXECUTABLE FUNCTION: 
    Handles data extraction, distance estimation, cost calculation, 
    and UI formatting entirely through the LLM prompt.
    """
    
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        return "System Error: API Key missing.", session_state
    
    genai.configure(api_key=api_key)
    model = genai.GenerativeModel("gemini-2.5-flash")

    # The "Brain" of the operation. This prompt handles EVERYTHING.
    master_prompt = f"""
    You are the Digaxy AI Moving Assistant. You are a professional dispatcher.

    ### 1. KNOWLEDGE BASE (Rates):
    - Pickup: $42.92 base | $1.62/min labor
    - Van: $77.00 base | $2.02/min labor
    - Minibox: $144.51 base | $2.30/min labor
    - Bigbox: $230.00 base | $4.99/min labor
    - Distance Surcharge: $0.80 per KM

    ### 2. CORE LOGIC:
    - DATA EXTRACTION: Extract [service_type, item_description, pickup_location, dropoff_location, vehicle_type] from user input.
    - DISTANCE CALCULATION: If you have both locations, use your internal geographical data to estimate the REAL-WORLD driving distance (KM) between them.
    - LABOR ESTIMATION: Estimate labor time (minutes) based on the items and locations (minimum 30 mins). base it on the complexity of the move (e.g., more items or longer distance = more time).
    - MATH: Total = Base + (Estimated_KM * 0.80) + (Estimated_Mins * Labor_Rate).

    ### 3. VISUAL STYLE (MUST MATCH THIS):
    - Use 📍 for route, 📐 for distance, 🚐 for vehicle, 📦 for items.
    - Use header: ✅ **Your Estimate is Ready!**
    - Show Cost Breakdown: Base, Distance cost, Labor time ($cost).
    - Final Total: 💰 **TOTAL: $XX.XX** (including taxes).
    - Ask: "Would you like to **proceed with booking?** (Say 'yes' or 'no')".
    - after getting yes it will write this message : "Great! Please complete your booking here: http://localhost:3000/book"

    ### 4. CURRENT SESSION DATA:
    {json.dumps(session_state)}

    ### 5. NEW USER INPUT:
    "{user_input}"

    ### 6. OUTPUT FORMAT (STRICT JSON ONLY):
    {{
        "message": "The text response formatted with markdown and emojis",
        "updated_state": {{
            "fields": {{ "service_type": "...", "vehicle_type": "...", "pickup_location": "...", "dropoff_location": "...", "item_description": "..." }},
            "calculation": {{ "distance_km": 0, "labor_mins": 0, "total_cost": 0 }},
            "status": "collecting | confirming | booked"
        }}
    }}
    """

    try:
        response = model.generate_content(master_prompt)
        
        # Extract the JSON block from the LLM response
        clean_text = re.search(r'\{.*\}', response.text, re.DOTALL).group()
        data = json.loads(clean_text)
        
        # If the user confirmed, append the booking link to the message
        if data["updated_state"]["status"] == "booked":
            data["message"] += "\n\n🔗 **Complete your booking here:**\nhttp://localhost:3000/book"

        return data["message"], data["updated_state"]

    except Exception as e:
        print(f"Error logic: {e}")
        return "I'm sorry, I'm having trouble processing your request. Please try again.", session_state

# --- TEST RUNNER (How your Backend calls it) ---

if __name__ == "__main__":
    # INITIAL STATE (This would be stored in your Database/Redis)
    session = {
        "fields": {}, 
        "calculation": None, 
        "status": "collecting"
    }
    
    print("🚀 Digaxy AI Engine Online")
    
    while True:
        user_in = input("\nYou: ")
        if user_in.lower() in ["exit", "quit"]: break
        
        # The single call
        reply, session = run_digaxy_ai(user_in, session)
        
        print(f"\nAssistant: {reply}")
        # print(f"\n[DEBUG STATE]: {json.dumps(session['fields'], indent=2)}")
import os
import json
import re
from typing import Dict, Any, Tuple
from dotenv import load_dotenv
import google.generativeai as genai

load_dotenv()


def run_digaxy_ai(user_input: str, session_state: Dict[str, Any]) -> Tuple[str, Dict[str, Any]]:
    """
    Conversational moving estimator using Gemini.
    LLM handles extraction, reasoning, and cost estimation.
    session_state only maintains conversation memory.
    """
    
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        return "System Error: Gemini API key missing.", session_state
    
    genai.configure(api_key=api_key)
    model = genai.GenerativeModel("gemini-2.5-flash")

    master_prompt = f"""You are a moving dispatcher assistant. Help users get accurate moving estimates with CORRECT distances.

### SESSION STATE
{json.dumps(session_state)}

### PRICING
Van: $77 base + $2.02/min (furniture, 5-15 boxes)
Minibox: $144.51 base + $2.30/min (20-40 boxes)
Bigbox: $230 base + $4.99/min (40+ boxes)
Pickup: $42.92 base + $1.62/min (single item)
Distance: $0.80/km

### KNOWN DISTANCES (driving km)
Lynbrook, NY ↔ Madison, NJ: ~90 km
Lynbrook, NY ↔ Newark, NJ: ~45 km
Manhattan, NY ↔ Boston, MA: ~350 km
Manhattan, NY ↔ Philadelphia, PA: ~150 km
Manhattan, NY ↔ Washington, DC: ~360 km
Manhattan, NY ↔ Los Angeles, CA: ~4500 km

IMPORTANT: Use these distances. If route not listed, estimate realistic driving distance (NOT 0 km unless same location). Add 20-30% to straight line distance for road routes.

### REQUIRED FIELDS (in order)
1. item_description
2. pickup_location
3. dropoff_location
4. vehicle_type (USER CHOOSES, then validate & recommend if needed)

### WORKFLOW
- Extract new info from user message
- Identify missing fields
- Ask only for next missing field (never multiple at once)
- When items + locations complete → show 4 vehicle options, ask user to choose
- When user picks vehicle → check if it matches load
  - If good fit → proceed to estimate
  - If poor fit → recommend better option, ask for confirmation
- All 4 fields confirmed → CALCULATE DISTANCE, then show estimate
- User says yes/confirm/book/ok → status="booked", show booking link

### VEHICLE RECOMMENDATION RULES
Single item + pickup → Pickup
Furniture + 5-15 boxes → Van
20-40 boxes → Minibox
40+ boxes or full house → Bigbox

### DISTANCE CALCULATION
CRITICAL: Do NOT use 0 km unless pickup and dropoff are the same location.
- Extract pickup_location and dropoff_location
- Check against known distances list
- If not in list, estimate realistic driving distance
- For NY/NJ routes: ~50-60 miles ≈ 80-95 km
- Always provide realistic highway distance.

### COST FORMULA (with CORRECT distance)
total = base + (distance_km × 0.80) + (labor_mins × labor_rate)
labor_mins = 30 min minimum + (items_complexity × time_per_item)

### OUTPUT FORMAT

When asking for info: Ask naturally for next missing field.

When showing vehicles (after items + locations known):
"Which vehicle works best for your move?
1. **Pickup** ($42.92)
2. **Van** ($77)
3. **Minibox** ($144.51)
4. **Bigbox** ($230)"

When recommending vehicle:
"The **[VEHICLE]** would be the right choice for [ITEMS]. Proceed? (yes/no)"

When estimate ready (with CORRECT distance_km):
Certainly! Here are the details for your moving estimate:
*   **Items:** [item_description]
*   **Pickup:** [pickup_location]
*   **Dropoff:** [dropoff_location]
*   **Vehicle:** [vehicle_type]
*   **Estimated Distance:** [distance_km] km
*   **Estimated Labor:** [labor_mins] minutes
*   **Total Estimated Cost:** $[total_cost]

Do you want to process the booking? (yes/no)

When booking confirmed:
Great! Complete your booking here: http://localhost:3000/book

### USER MESSAGE
"{user_input}"

### RESPOND WITH STRICT JSON
{{
  "message": "your response",
  "updated_state": {{
    "fields": {{
      "item_description": null,
      "pickup_location": null,
      "dropoff_location": null,
      "vehicle_type": null,
      "service_type": null
    }},
    "calculation": {{
      "distance_km": 0,
      "labor_mins": 0,
      "total_cost": 0
    }},
    "status": "collecting | confirming | booked"
  }}
}}
"""

    try:
        response = model.generate_content(master_prompt)
        
        # Extract the JSON block from the LLM response safely
        json_match = re.search(r'\{[\s\S]*\}', response.text)
        if not json_match:
            return "Error: Invalid response format. Please try again.", session_state
        
        clean_text = json_match.group()
        data = json.loads(clean_text)
        
        return data["message"], data["updated_state"]

    except json.JSONDecodeError as je:
        print(f"JSON Error: {je}")
        return "Error parsing response. Please try again.", session_state
    
    except Exception as e:
        error_msg = str(e)
        if "429" in error_msg or "quota" in error_msg.lower():
            return "⚠️ API Quota Exceeded: Free tier limit reached. Try again tomorrow or upgrade at https://ai.google.dev", session_state
        print(f"Error: {type(e).__name__}: {e}")
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
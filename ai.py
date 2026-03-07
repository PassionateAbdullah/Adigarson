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

    master_prompt = f"""
You are **Digaxy AI**, a professional moving dispatcher chatbot.

Your ONLY job: Help users get moving estimates through natural conversation.
You MUST be conversational, smart, and dynamic.

--------------------------------------------------
### KNOWLEDGE BASE

**Van:** $77/base | $2.02/min labor (Furniture, 5-15 boxes)
**Minibox:** $144.51/base | $2.30/min labor (20-40 boxes, small apartment)
**Bigbox:** $230/base | $4.99/min labor (40+ boxes, entire house)
**Pickup:** $42.92/base | $1.62/min labor (Just item pickup)

Distance: $0.80 per KM

--------------------------------------------------
### CURRENT SESSION DATA

{json.dumps(session_state)}

--------------------------------------------------
### CONVERSATION RULES

**REQUIRED FIELDS FOR ESTIMATE:**
- item_description (What are you moving?)
- pickup_location (Where from?)
- dropoff_location (Where to?)
- vehicle_type (What vehicle? Auto-recommend from items)

**YOUR CONVERSATION FLOW:**

1. **ANALYZE SESSION STATE** - Check which fields are filled vs empty
2. **EXTRACT DATA** - Parse user message for any new information
3. **UPDATE STATE** - Save new data to session_state
4. **CHECK COMPLETENESS** - Do we have all 4 fields?

**IF COMPLETE:** Show estimate with Cost Breakdown
**IF INCOMPLETE:** Ask for THE NEXT missing field CONVERSATIONALLY

**NEVER:**
- Show ALL options upfront (e.g., "Van, Minibox, or Bigbox?")
- Show estimate button when fields are missing
- Ask for multiple things at once
- Ignore session_state values

**IF USER SAYS YES/CONFIRM/BOOK:**
- Status → "booked"
- Message → Booking link ONLY ONCE

--------------------------------------------------
### DECISION LOGIC

Pseudo-code:
```
missing_fields = []
if not session['fields']['item_description']: missing_fields.append('items')
if not session['fields']['pickup_location']: missing_fields.append('pickup')
if not session['fields']['dropoff_location']: missing_fields.append('dropoff')

if missing_fields is empty:
    → SHOW ESTIMATE & ask "proceed with booking?"
else:
    → Ask for missing_fields[0] conversationally
```

--------------------------------------------------
### VEHICLE AUTO-RECOMMENDATION

Only AFTER you know what they're moving:

User says: "couch and 10 boxes" → You recommend: "A Van is perfect for that!"
User says: "I'm moving my entire apartment, about 50 boxes" → You recommend: "You'll need a Bigbox"

NEVER ask which vehicle - RECOMMEND based on items.

--------------------------------------------------
### ESTIMATION FORMULA

Once all fields exist:

labor_mins = estimated based on items + distance (min 30)
total = base_price + (distance_km * 0.80) + (labor_mins * labor_rate)

Example:
- Van from NYC to Boston (350km), couch + 10 boxes
- Base: $77
- Distance: 350 * 0.80 = $280
- Labor: 75 mins * $2.02 = $151.50
- Total: $77 + $280 + $151.50 = $508.50

--------------------------------------------------
### EXAMPLE CONVERSATION

User: "I have a couch and 10 boxes from New York"
You: "Great! A Van would be perfect for your couch and 10 boxes. Where are we delivering to?"
[Updates: item_description="couch, 10 boxes", pickup_location="New York", vehicle_type="Van"]

User: "Boston"
You: ✅ **Your Estimate is Ready!**
[Shows full estimate]

User: "yes"
You: "Great! Complete booking here: http://localhost:3000/book"
[Status → "booked"]

--------------------------------------------------
### USER INPUT

"{user_input}"

--------------------------------------------------
### OUTPUT (STRICT JSON)

{{
  "message": "Your response - be conversational and natural",
  "updated_state": {{
    "fields": {{
      "item_description": "null or extracted items",
      "pickup_location": "null or extracted location",
      "dropoff_location": "null or extracted location",
      "vehicle_type": "null or recommended vehicle",
      "service_type": "null or service"
    }},
    "calculation": {{
      "distance_km": 0,
      "labor_mins": 0,
      "total_cost": 0
    }},
    "status": "collecting or confirming or booked"
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
        
        # If the user confirmed, append the booking link to the message
        if data["updated_state"]["status"] == "booked":
            data["message"] += "\n\n🔗 **Complete your booking here:**\nhttp://localhost:3000/book"

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
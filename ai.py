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
    You are **Digaxy AI**, a professional moving dispatcher assistant.

    Your job is to help users get a **moving estimate** through a natural conversation.

    Be conversational, accurate, and dynamic.

    Always return **STRICT JSON ONLY**.

    --------------------------------------------------

    ### CURRENT SESSION STATE

    {json.dumps(session_state)}

    --------------------------------------------------

    ### PRICING KNOWLEDGE

    Van
    Base: $77
    Labor: $2.02/min
    Best for: furniture, 5–15 boxes

    Minibox
    Base: $144.51
    Labor: $2.30/min
    Best for: 20–40 boxes, small apartment

    Bigbox
    Base: $230
    Labor: $4.99/min
    Best for: 40+ boxes, full house

    Pickup
    Base: $42.92
    Labor: $1.62/min
    Best for: single item pickup

    Distance surcharge:
    $0.80 per km

    --------------------------------------------------

    ### REQUIRED FIELDS

    item_description  
    pickup_location  
    dropoff_location  
    vehicle_type  

    --------------------------------------------------

    ### CONVERSATION LOGIC

    1. Read the current session state.
    2. Extract any new information from the user message.
    3. Update the session fields if new data appears.
    4. Determine which required fields are missing.

    Field priority:

    1. item_description
    2. pickup_location
    3. dropoff_location
    4. vehicle_type (USER CHOOSES, not automatic)

    **If item_description, pickup_location, dropoff_location are all filled BUT vehicle_type is missing:**
    → Present vehicle options and ask user to choose (see VEHICLE SELECTION FLOW)

    **If user provides a vehicle choice:**
    → Validate if it matches their load
    → If not ideal, recommend better option
    → Ask for confirmation
    
    **When all 4 fields are confirmed:**
    → Calculate estimate and show results

    If a field is missing → ask **only for the next missing field**.

    Never ask multiple questions at once.

    Never overwrite existing values unless the user explicitly changes them.

    --------------------------------------------------

    ### VEHICLE SELECTION FLOW

    **Step 1: Present Options (after items + locations known)**
    
    Ask user to choose from vehicle options:
    
    "Which vehicle works best for your move?
    
    1. **Pickup** ($42.92) - Single item pickup
    2. **Van** ($77) - Furniture, 5-15 boxes
    3. **Minibox** ($144.51) - 20-40 boxes, small apartment
    4. **Bigbox** ($230) - 40+ boxes, entire house"
    
    Do NOT automatically select. Wait for user response.

    **Step 2: Validate User's Choice**
    
    Once user picks a vehicle, check if it matches their load:
    
    Rules:
    - Single item → Pickup is appropriate
    - Furniture + 5-15 boxes → Van is appropriate
    - Light household (20-40 boxes) → Minibox is appropriate
    - Full household (40+ boxes) → Bigbox is appropriate
    
    **Step 3: If Choice is Not Ideal, Recommend**
    
    If user picks wrong vehicle, respond:
    
    "The **[RECOMMENDED_VEHICLE]** would be the right choice for your load. Would you like to proceed with it? (yes/no)"
    
    Example:
    User picks: "Pickup"
    Their items: "my household"
    You say: "The **Bigbox** would be the right choice for your entire household. Would you like to proceed with it? (yes/no)"
    
    **Step 4: After Confirmation, Show Estimate**
    
    Only show the estimate after vehicle is confirmed/selected.

    --------------------------------------------------

    ### ESTIMATE CALCULATION

    **Only calculate estimate after ALL 4 fields are confirmed:**
    - item_description ✓
    - pickup_location ✓
    - dropoff_location ✓
    - vehicle_type ✓ (user chose or confirmed recommendation)

    labor_mins = estimated from items (minimum 30 minutes)

    Total Cost:

    total = base_price + (distance_km * 0.80) + (labor_mins * labor_rate)

    Always round to 2 decimals.

    Distance calculation:
    - If pickup and dropoff are same location: 0 km (local service)
    - Otherwise: Use actual driving distance between locations

    --------------------------------------------------

    ### BOOKING LOGIC

    **When user confirms booking by saying:**
    yes / confirm / book / proceed / ok
    
    Then:
    status = "booked"
    
    Response: "Great! Complete your booking here: http://localhost:3000/book"
    
    **When user agrees to recommended vehicle:**
    Update vehicle_type to the recommended option
    Then proceed with estimate

    --------------------------------------------------

    ### EXAMPLE CONVERSATION FLOW
    
    **Message 1:**
    User: "I need to shift my household from Madison, New Jersey, USA to Lynbrook, New York 11563, USA"
    Bot: Extracts items (household), pickup (Madison), dropoff (Lynbrook)
    Response: "Which vehicle works best for your move?
    
    1. **Pickup** ($42.92) - Single item pickup
    2. **Van** ($77) - Furniture, 5-15 boxes
    3. **Minibox** ($144.51) - 20-40 boxes, small apartment
    4. **Bigbox** ($230) - 40+ boxes, entire house"
    
    **Message 2:**
    User: "Van"
    Bot: Checks - "household" needs Bigbox, not Van. Recommend upgrade.
    Response: "The **Bigbox** would be the right choice for shifting your entire household. Would you like to proceed with it? (yes/no)"
    
    **Message 3:**
    User: "Yes"
    Bot: Updates vehicle_type to Bigbox, calculates estimate
    Response: Shows full estimate with details
    [Status: "confirming"]
    
    **Message 4:**
    User: "yes"
    Bot: [Status: "booked"]
    Response: "Great! Complete your booking here: http://localhost:3000/book"

    --------------------------------------------------

    ### USER MESSAGE

    "{user_input}"

    --------------------------------------------------

    ### OUTPUT FORMAT (STRICT JSON)

    **When presenting vehicle options:**
    
    "Which vehicle works best for your move?
    
    1. **Pickup** ($42.92) - Single item pickup
    2. **Van** ($77) - Furniture, 5-15 boxes
    3. **Minibox** ($144.51) - 20-40 boxes, small apartment
    4. **Bigbox** ($230) - 40+ boxes, entire house"
    
    [Status: "collecting", vehicle_type: null]

    **When recommending based on load:**
    
    "The **[VEHICLE]** would be the right choice for [ITEMS]. Would you like to proceed with it? (yes/no)"
    
    [Status: "collecting", vehicle_type: null or user's choice]

    **When estimate is ready:**

    Certainly! Here are the details for your moving estimate:

    *   **Items:** [item_description]
    *   **Pickup:** [pickup_location]
    *   **Dropoff:** [dropoff_location]
    *   **Vehicle:** [vehicle_type]
    *   **Estimated Distance:** [distance_km] km
    *   **Estimated Labor:** [labor_mins] minutes
    *   **Total Estimated Cost:** $[total_cost]

    Do you want to process the booking? (yes/no)
    
    [Status: "confirming", all fields filled]

    **When user confirms booking:**

    Great! Complete your booking here: http://localhost:3000/book
    
    [Status: "booked"]

    **When collecting information:**
    
    Ask naturally for the next missing field (items, pickup, dropoff).

    --------------------------------------------------

    ### JSON RESPONSE STRUCTURE

    {{
    "message": "formatted response above",

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
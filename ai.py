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
            You are the **Digaxy AI Moving Assistant**, a professional logistics dispatcher.

            Your job is to help users estimate moving costs conversationally.

            Users may provide information gradually across multiple messages.
            You MUST use the existing session_state to remember previous information.

            --------------------------------------------------

            ### KNOWLEDGE BASE (Rates)

            Pickup: $42.92 base | $1.62/min labor  
            Van: $77.00 base | $2.02/min labor  
            Minibox: $144.51 base | $2.30/min labor  
            Bigbox: $230.00 base | $4.99/min labor  

            Distance surcharge: $0.80 per KM

            --------------------------------------------------

            ### CURRENT SESSION STATE

            {json.dumps(session_state)}

            --------------------------------------------------

            ### VEHICLE SELECTION GUIDE

            Recommend vehicle based on items:
            
            - **Van**: Couch, bed, dining table, cabinet, office desk, 20-30 boxes
            - **Minibox**: Multiple sofas, 2-3 beds, full apartment furniture, 40-60 boxes
            - **Bigbox**: Large house move, multiple rooms, piano, 80+ boxes, all furniture combined

            --------------------------------------------------

            ### CORE LOGIC

            1. DATA EXTRACTION
            Extract these fields from the user message if present:

            service_type
            vehicle_type
            pickup_location
            dropoff_location
            item_description

            2. STATE MERGE
            Merge extracted values with the existing session_state.
            Do NOT remove existing values unless user changes them.

            3. REQUIRED FIELDS FOR ESTIMATE

            vehicle_type  
            pickup_location  
            dropoff_location  
            item_description

            4. CONVERSATION FLOW - SMART VEHICLE RECOMMENDATION

            PRIORITY ORDER OF QUESTIONS:
            
            First → Ask what items/furniture they're moving (if missing)
            Second → Ask pickup/dropoff locations (if missing)
            Third → Based on items described, RECOMMEND the best vehicle (do NOT show all 3 options)
            
            SMART VEHICLE SELECTION:
            - Analyze the item_description the user provides
            - Based on the type and quantity of items, recommend ONE vehicle (not options)
            - Example: If user says "couch and 10 boxes" → Recommend Van
            - Only ask for confirmation if they want a different size

            5. IF ALL FIELDS EXIST

            Calculate estimate:

            Distance (KM) between locations  
            Labor time (minimum 30 mins) based on item complexity

            6. COST FORMULA

            Total = Base + (Distance_KM × 0.80) + (Labor_Mins × Labor_Rate)

            Round money values to 2 decimals.

            --------------------------------------------------

            ### RESPONSE FORMAT

            When asking about items (first priority):
            📦 What furniture and items are you planning to move? (e.g., couch, bed, dining table, number of boxes)

            When recommending vehicle (after items are described):
            🚐 Based on what you're moving, I recommend a **[Vehicle Name]**. Does that work for you?

            When estimate is ready:

            ✅ **Your Estimate is Ready!**

            📍 Route: Pickup → Dropoff  
            📐 Distance: XX km  
            🚐 Vehicle: TYPE  
            📦 Items: DESCRIPTION  

            Cost Breakdown

            Base price: $XX  
            Distance cost: $XX  
            Labor (XX mins): $XX  

            💰 **TOTAL: $XX.XX**

            This estimate is approximate and may vary after final inspection.

            Would you like to **proceed with booking?** (yes/no)

            If user confirms booking, respond:

            "Great! Please complete your booking here: http://localhost:3000/book"

            --------------------------------------------------

            ### USER MESSAGE

            "{user_input}"

            --------------------------------------------------

            ### OUTPUT FORMAT (STRICT JSON ONLY)

            {{
            "message": "assistant response",
            "updated_state": {{
                "fields": {{
                "service_type": "",
                "vehicle_type": "",
                "pickup_location": "",
                "dropoff_location": "",
                "item_description": ""
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
import os
import json
import re
from typing import Dict, Any, Optional
from dotenv import load_dotenv

import google.generativeai as genai

# Load environment variables from .env file
load_dotenv()

BOOKING_URL = "http://localhost:3000/book"


def load_dataset(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


class ConversationManager:
    """Manages multi-turn conversation state and flow."""
    
    def __init__(self):
        self.state = "greeting"
        self.data = {
            "service_type": None,
            "item_description": None,
            "pickup_location": None,
            "dropoff_location": None,
            "vehicle_type": None,
            "distance_km": None,
            "estimated_cost": None,
        }
    
    def reset(self):
        self.__init__()
    
    def update_state(self, new_state: str):
        self.state = new_state
    
    def set_data(self, key: str, value: Any):
        self.data[key] = value
    
    def get_data(self, key: str) -> Optional[Any]:
        return self.data.get(key)
    
    def is_complete(self) -> bool:
        required = ["pickup_location", "dropoff_location", "vehicle_type", "item_description"]
        return all(self.data.get(key) for key in required)


def estimate_distance_and_cost(
    pickup: str,
    dropoff: str,
    vehicle_type: str,
    item_description: str,
) -> Dict[str, Any]:
    """
    Use LLM to estimate distance and calculate cost based on base fare, distance, and labor time.
    """
    vehicle_rates = {
        "pickup": {"base": 42.92, "labor_per_min": 1.62},
        "van": {"base": 77.00, "labor_per_min": 2.02},
        "minibox": {"base": 144.51, "labor_per_min": 2.30},
        "bigbox": {"base": 230.00, "labor_per_min": 4.99},
    }
    
    vehicle_lower = vehicle_type.lower()
    if vehicle_lower not in vehicle_rates:
        vehicle_lower = "van"
    
    base_rate = vehicle_rates[vehicle_lower]["base"]
    labor_rate = vehicle_rates[vehicle_lower]["labor_per_min"]
    
    try:
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            raise RuntimeError("GEMINI_API_KEY not set")
        
        genai.configure(api_key=api_key)
        
        system_prompt = f"""You are a cost estimation assistant. Estimate:
1. Distance in km between: {pickup} and {dropoff}
2. Labor time in minutes for: {item_description}
3. Calculate: Base ${base_rate:.2f} + Distance Cost + Labor Cost (${labor_rate:.2f}/min)

Distance cost: ~$0.8 per km
Labor: estimate 20-120 minutes based on item complexity

Respond ONLY with JSON:
{{"distance_km": 10, "labor_minutes": 45, "distance_cost": 8.0, "labor_cost": 73.0, "total_cost": 159.0}}"""

        model_obj = genai.GenerativeModel("gemini-2.5-flash")
        response = model_obj.generate_content(system_prompt)
        
        text = (response.text or "").strip()
        
        # Extract JSON
        json_match = re.search(r'\{[^{}]*\}', text)
        if json_match:
            result = json.loads(json_match.group())
            return {
                "success": True,
                "distance_km": float(result.get("distance_km", 10)),
                "estimated_labor_minutes": int(result.get("labor_minutes", 45)),
                "distance_cost": float(result.get("distance_cost", 8.0)),
                "labor_cost": float(result.get("labor_cost", 73.0)),
                "total_cost": float(result.get("total_cost", base_rate + 81.0)),
                "reasoning": "Calculated estimate",
            }
    except Exception as e:
        pass
    
    # Smart fallback based on item type and vehicle
    item_lower = item_description.lower()
    
    if "apartment" in item_lower or "studio" in item_lower:
        distance = 12
        labor = 60
    elif "furniture" in item_lower:
        distance = 10
        labor = 45
    elif "box" in item_lower or "small" in item_lower:
        distance = 8
        labor = 25
    else:
        distance = 10
        labor = 40
    
    distance_cost = distance * 0.8
    labor_cost = labor * labor_rate
    total = base_rate + distance_cost + labor_cost
    
    return {
        "success": True,
        "distance_km": distance,
        "estimated_labor_minutes": labor,
        "distance_cost": distance_cost,
        "labor_cost": labor_cost,
        "total_cost": total,
        "reasoning": "Estimated based on typical moves"
    }




def get_next_response(
    user_input: str,
    conversation: ConversationManager,
    dataset: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Generate next response based on conversation state.
    Strict structured flow - no jumping between states.
    """
    user_lower = user_input.lower().strip()
    
    # State: GREETING
    if conversation.state == "greeting":
        if any(word in user_lower for word in ["hi", "hello", "hey", "start", "help", "begin"]):
            conversation.update_state("service_type")
            return {
                "reply": "👋 Hi there! I'm Digaxy Assistant. Let me help you with your moving or delivery needs.\n\nWhat service do you need?\n\n1. **Home Move** - Moving from one home to another\n2. **Apartment Move** - Apartment to apartment\n3. **Office Move** - Commercial/office moves\n4. **Furniture Delivery** - Deliver furniture items\n5. **Donation Pickup** - Schedule a donation pickup\n\nJust tell me your choice!",
                "follow_up": [],
                "state": conversation.state,
            }
        else:
            return {
                "reply": "👋 Welcome to Digaxy! Please say 'hello' or 'hi' to get started!",
                "follow_up": [],
                "state": conversation.state,
            }
    
    # State: SERVICE_TYPE
    elif conversation.state == "service_type":
        service_keywords = {
            "home": "Home Move",
            "apartment": "Apartment Move",
            "office": "Office Move",
            "furniture": "Furniture Delivery",
            "donation": "Donation Pickup",
        }
        
        service_found = None
        for keyword, service_name in service_keywords.items():
            if keyword in user_lower:
                service_found = service_name
                conversation.set_data("service_type", service_name)
                break
        
        if service_found:
            conversation.update_state("item_details")
            return {
                "reply": f"✓ Selected: **{service_found}**\n\nGreat! Now tell me **what items are you moving or what do you need delivered?**\n\nFor example: furniture, boxes, appliances, office equipment, electronics, etc.",
                "follow_up": [],
                "state": conversation.state,
            }
        else:
            return {
                "reply": "I didn't understand. Please choose one:\n1. Home Move\n2. Apartment Move\n3. Office Move\n4. Furniture Delivery\n5. Donation Pickup",
                "follow_up": [],
                "state": conversation.state,
            }
    
    # State: ITEM_DETAILS
    elif conversation.state == "item_details":
        if len(user_input) > 3:
            conversation.set_data("item_description", user_input)
            conversation.update_state("pickup_location")
            return {
                "reply": f"✓ Items: **{user_input}**\n\nPerfect! Now, **what's your pickup location?** (Please provide city name and specific address)",
                "follow_up": [],
                "state": conversation.state,
            }
        else:
            return {
                "reply": "Please provide more details about what you're moving. (At least 4-5 words)",
                "follow_up": [],
                "state": conversation.state,
            }
    
    # State: PICKUP_LOCATION
    elif conversation.state == "pickup_location":
        if len(user_input) > 5:
            conversation.set_data("pickup_location", user_input)
            conversation.update_state("dropoff_location")
            return {
                "reply": f"✓ Pickup: **{user_input}**\n\nGot it! Now, **what's your dropoff/destination location?** (Please provide city name and specific address)",
                "follow_up": [],
                "state": conversation.state,
            }
        else:
            return {
                "reply": "Please provide a valid pickup location with city and address.",
                "follow_up": [],
                "state": conversation.state,
            }
    
    # State: DROPOFF_LOCATION
    elif conversation.state == "dropoff_location":
        if len(user_input) > 5:
            conversation.set_data("dropoff_location", user_input)
            conversation.update_state("vehicle_type")
            return {
                "reply": f"✓ Dropoff: **{user_input}**\n\nExcellent! Now **choose a vehicle** for your move:\n\n• **Pickup** - $42.92 base (light items, small boxes)\n• **Van** - $77 base (furniture, moderate loads)\n• **Minibox** - $144.51 base (apartment moves)\n• **Bigbox** - $230 base (full house/office)\n\nWhich one works for you?",
                "follow_up": [],
                "state": conversation.state,
            }
        else:
            return {
                "reply": "Please provide a valid dropoff location with city and address.",
                "follow_up": [],
                "state": conversation.state,
            }
    
    # State: VEHICLE_TYPE
    elif conversation.state == "vehicle_type":
        vehicle_options = ["pickup", "van", "minibox", "bigbox"]
        vehicle_found = None
        for vehicle in vehicle_options:
            if vehicle in user_lower:
                vehicle_found = vehicle.capitalize()
                conversation.set_data("vehicle_type", vehicle_found)
                break
        
        if vehicle_found:
            # Immediately calculate estimate
            try:
                estimation = estimate_distance_and_cost(
                    pickup=conversation.get_data("pickup_location"),
                    dropoff=conversation.get_data("dropoff_location"),
                    vehicle_type=vehicle_found,
                    item_description=conversation.get_data("item_description"),
                )
                
                conversation.set_data("distance_km", estimation.get("distance_km"))
                conversation.set_data("estimated_cost", estimation.get("total_cost"))
                conversation.update_state("confirm_estimate")
                
                distance = estimation.get("distance_km", 0)
                labor_mins = estimation.get("estimated_labor_minutes", 0)
                distance_cost = estimation.get("distance_cost", 0)
                labor_cost = estimation.get("labor_cost", 0)
                total_cost = estimation.get("total_cost", 0)
                
                estimate_text = f"""✅ **Your Estimate is Ready!**

📍 **Route:** {conversation.get_data('pickup_location')} 
➜ {conversation.get_data('dropoff_location')}

📏 **Distance:** ~{distance:.1f} km
🚐 **Vehicle:** {vehicle_found}
📦 **Items:** {conversation.get_data('item_description')}

**Cost Breakdown:**
• Base fare: ${77.00 if 'Van' in vehicle_found else (42.92 if 'Pickup' in vehicle_found else (144.51 if 'Minibox' in vehicle_found else 230.00)):.2f}
• Distance cost: ${distance_cost:.2f}
• Labor time: ~{labor_mins} min (${labor_cost:.2f})

💰 **TOTAL: ${total_cost:.2f}** (including taxes)

*Note: This is a non-binding estimate. Final price may vary.*

Would you like to **proceed with booking?** (Say 'yes' or 'no')"""
                
                return {
                    "reply": estimate_text,
                    "follow_up": ["Say 'yes' to book or 'no' to cancel"],
                    "state": conversation.state,
                }
            except Exception as e:
                conversation.update_state("confirm_estimate")
                return {
                    "reply": "Got your details! Let me calculate...\n\n💰 **Estimated Cost: $150-300** (based on typical moves)\n\nWould you like to proceed with booking?",
                    "follow_up": ["Say 'yes' to continue or 'no' to cancel"],
                    "state": conversation.state,
                }
        else:
            return {
                "reply": "Please choose one: **Pickup**, **Van**, **Minibox**, or **Bigbox**",
                "follow_up": [],
                "state": conversation.state,
            }
    
    # State: CONFIRM_ESTIMATE
    elif conversation.state == "confirm_estimate":
        if any(word in user_lower for word in ["yes", "confirm", "book", "proceed", "ok", "yeah", "sure", "let's go"]):
            conversation.update_state("booking")
            total_cost = conversation.get_data("estimated_cost")
            return {
                "reply": f"""✅ **Booking Confirmed!**

Thank you for choosing Digaxy! Your booking details are ready:

📋 **Summary:**
• Service: {conversation.get_data('service_type')}
• Items: {conversation.get_data('item_description')}
• From: {conversation.get_data('pickup_location')}
• To: {conversation.get_data('dropoff_location')}
• Vehicle: {conversation.get_data('vehicle_type')}
• Estimated Cost: ${total_cost:.2f}

🔗 **Complete your booking here:**
{BOOKING_URL}

We'll be in touch shortly to confirm your pickup time. Have a great day! 🚚✨""",
                "follow_up": [],
                "state": conversation.state,
            }
        elif any(word in user_lower for word in ["no", "cancel", "back", "restart", "different"]):
            conversation.reset()
            conversation.update_state("greeting")
            return {
                "reply": "No problem! Let's start fresh. 👋 What can I help you with today?",
                "follow_up": [],
                "state": conversation.state,
            }
        else:
            return {
                "reply": f"Would you like to book this service for ${conversation.get_data('estimated_cost'):.2f}? (Say 'yes' to book or 'no' to cancel)",
                "follow_up": [],
                "state": conversation.state,
            }
    
    # State: BOOKING (final state)
    elif conversation.state == "booking":
        return {
            "reply": "Your booking is being processed. You should receive a confirmation shortly!",
            "follow_up": [],
            "state": conversation.state,
        }
    
    # Default
    else:
        return {
            "reply": "I'm not sure what you mean. Could you rephrase that?",
            "follow_up": [],
            "state": conversation.state,
        }
    


def run_chatbot(dataset_path: str = "data.json"):
    """
    Run interactive chatbot with structured conversation flow.
    Guides user through service selection → items → locations → vehicle → estimate → booking.
    """
    print("\n" + "="*75)
    print("🤖 DIGAXY ASSISTANT - Your Moving & Delivery Expert")
    print("Type 'exit' to leave | 'restart' to start over")
    print("="*75 + "\n")
    
    dataset = load_dataset(dataset_path)
    conversation = ConversationManager()
    
    while True:
        try:
            user_input = input("You: ").strip()
            
            # Exit commands
            if user_input.lower() in ["exit", "quit"]:
                print("\nAssistant: Thanks for choosing Digaxy! Have a great day! 👋\n")
                break
            
            # Restart conversation
            if user_input.lower() == "restart":
                conversation.reset()
                print("\nAssistant: Let's start fresh! 👋 What can I help you with?\n")
                continue
            
            # Skip empty
            if not user_input:
                continue
            
            # Get response based on state
            result = get_next_response(user_input, conversation, dataset)
            print(f"\nAssistant: {result['reply']}\n")
            
            if result.get("follow_up"):
                print("💡 Hint:")
                for hint in result["follow_up"]:
                    print(f"   {hint}")
                print()
        
        except KeyboardInterrupt:
            print("\n\nAssistant: Thanks for reaching out! Have a great day! 👋\n")
            break
        except EOFError:
            print("\n\nAssistant: Thanks for reaching out! Have a great day! 👋\n")
            break
        except Exception as e:
            # Don't print error, just continue
            continue


if __name__ == "__main__":
    run_chatbot()
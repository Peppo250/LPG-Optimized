# LPG Catering Intelligence System - Central Configuration

# Event types supported by the system
EVENT_TYPES = [
    "wedding",
    "corporate_lunch",
    "college_canteen",
    "birthday_party",
    "festival_event",
    "hospital_canteen",
    "school_canteen",
    "dhaba_daily"
]

# Menu profiles supported by the system
MENU_PROFILES = [
    "veg_simple",
    "veg_elaborate",
    "nonveg_simple",
    "nonveg_elaborate",
    "mixed_standard",
    "snacks_only",
    "biryani_special"
]

# Business size definitions
BUSINESS_SIZES = {
    "small":  {"hc": (50,  300),  "burners": (2, 4),  "exp": (1, 5)},
    "medium": {"hc": (200, 800),  "burners": (4, 8),  "exp": (3, 12)},
    "large":  {"hc": (500, 3000), "burners": (8, 20), "exp": (7, 25)},
}

# Gas consumption reference table (NRAI commercial benchmarks)
# Usable LPG (kg) per 100 guests
GAS_PER_100_GUESTS = {
    "veg_simple":       7.2,
    "veg_elaborate":   10.8,
    "nonveg_simple":    9.1,
    "nonveg_elaborate": 14.3,
    "mixed_standard":  11.5,
    "snacks_only":      4.8,
    "biryani_special":  8.6,
}

# Standard wastage rates by event type
WASTAGE_RATES = {
    "wedding":          0.12,
    "corporate_lunch":  0.07,
    "college_canteen":  0.09,
    "birthday_party":   0.14,
    "festival_event":   0.11,
    "hospital_canteen": 0.06,
    "school_canteen":   0.08,
    "dhaba_daily":      0.05,
}

# Delivery lead days by region/city
DELIVERY_LEAD_DAYS = {
    "Chennai":      1,
    "Coimbatore":   2,
    "Madurai":      2,
    "Trichy":       2,
    "default":      3,
}

# Cylinder specifications
COMMERCIAL_CYLINDER_KG = 17.5  # Usable LPG per 19-kg commercial cylinder
CYLINDER_PRICE_INR = 1910      # Standard commercial price (April 2024 Chennai IOC reference)

# Directories
MODEL_DIR = "models_final"
DATA_DIR = "data/final"

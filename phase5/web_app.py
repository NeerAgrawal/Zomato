import streamlit as st
import pandas as pd
import sys
import os
from pathlib import Path
from dotenv import load_dotenv

# Add parent directory to path to import from other phases
sys.path.insert(0, str(Path(__file__).parent.parent))

try:
    from phase3.data_integration import DataIntegrator
    from phase4.engine import GroqEngine
except ImportError as e:
    st.error(f"Error importing modules: {e}")
    st.info("Make sure you are running this from the correct directory layout.")
    st.stop()

# Configure page
st.set_page_config(
    page_title="Zomato AI Recommendations",
    page_icon="🍽️",
    layout="wide"
)

# Load environment variables
load_dotenv(Path(__file__).parent / '.env')
if not os.getenv('GROQ_API_KEY'):
    load_dotenv(Path(__file__).parent.parent / 'phase2' / '.env')

@st.cache_data
def load_data():
    """Load processed data from Phase 1"""
    possible_paths = [
        # 1. Deployment Lite File (Preferred for Cloud)
        Path(__file__).parent.parent / 'phase1' / 'data' / 'zomato_lite.pkl',
        # 2. Local Full Files
        Path(__file__).parent.parent / 'phase1' / 'data' / 'processed_zomato_data.pkl',
        Path(__file__).parent.parent / 'phase1' / 'data' / 'processed_zomato_data.csv'
    ]
    
    for data_path in possible_paths:
        if data_path.exists():
            if data_path.suffix == '.pkl':
                return pd.read_pickle(data_path)
            else:
                return pd.read_csv(data_path)
    
    raise FileNotFoundError("Processed data not found in Phase 1 directory")

# --- Phase 6 Aesthetics Integration ---
# Verified Unsplash URLs (Fixed broken links)
CUISINE_IMAGES = {
    'north indian': [
        'https://images.unsplash.com/photo-1585937421612-70a008356f36?w=800&q=80',
        'https://images.unsplash.com/photo-1596797038530-2c107229654b?w=800&q=80'
    ],
    'south indian': [
        'https://images.unsplash.com/photo-1610192244261-3f33de3f55e0?w=800&q=80',
        'https://images.unsplash.com/photo-1630384060421-14368b74f51e?w=800&q=80'
    ],
    'chinese': [
        'https://images.unsplash.com/photo-1525755662778-989d0524087e?w=800&q=80',
        'https://images.unsplash.com/photo-1541696490865-e6f72421bfcc?w=800&q=80'
    ],
    'pizza': [
        'https://images.unsplash.com/photo-1513104890138-7c749659a591?w=800&q=80',
        'https://images.unsplash.com/photo-1574071318508-1cdbab80d002?w=800&q=80'
    ],
    'burger': [
        'https://images.unsplash.com/photo-1568901346375-23c9450c58cd?w=800&q=80',
        'https://images.unsplash.com/photo-1550547660-d9450f859349?w=800&q=80',
        'https://images.unsplash.com/photo-1594212699903-ec8a3eca50f5?w=800&q=80'
    ],
    'biryani': [
        'https://images.unsplash.com/photo-1589302168068-964664d93dc0?w=800&q=80',
        'https://images.unsplash.com/photo-1642821373181-696a54913e93?w=800&q=80'
    ],
    'dessert': [
        'https://images.unsplash.com/photo-1563729768-3980346f028d?w=800&q=80',
        'https://images.unsplash.com/photo-1551024506-0bccd828d307?w=800&q=80',
        'https://images.unsplash.com/photo-1587314168485-3236d6710814?w=800&q=80'
    ],
    'cafe': [
        'https://images.unsplash.com/photo-1509042239860-f550ce710b93?w=800&q=80',
        'https://images.unsplash.com/photo-1559339352-11d035aa65de?w=800&q=80',
        'https://images.unsplash.com/photo-1445116572660-d38f22089581?w=800&q=80'
    ]
}

DEFAULT_IMAGES = [
    'https://images.unsplash.com/photo-1517248135467-4c7edcad34c4?w=800&q=80',
    'https://images.unsplash.com/photo-1552566626-52f8b828add9?w=800&q=80',
    'https://images.unsplash.com/photo-1559339352-11d035aa65de?w=800&q=80'
]

def get_restaurant_image(name, cuisines):
    """Deterministic image selection based on name hash"""
    def get_index(s, max_val):
        hash_val = 0
        for char in s:
            hash_val = ord(char) + ((hash_val << 5) - hash_val)
        return abs(hash_val) % max_val

    # Default logic
    image_url = DEFAULT_IMAGES[get_index(name, len(DEFAULT_IMAGES))]
    
    # Cuisine match logic
    if isinstance(cuisines, list) and len(cuisines) > 0:
        first_cuisine = cuisines[0].lower()
        for key, options in CUISINE_IMAGES.items():
            if key in first_cuisine:
                image_url = options[get_index(name, len(options))]
                break
                
    return image_url

def display_restaurant_card(restaurant, rank):
    """Display a single restaurant as a card with Phase 6 styling"""
    name = restaurant.get('name', 'Unknown')
    cuisines = restaurant.get('cuisines', [])
    if isinstance(cuisines, str):
        # Cleanup stringified list if needed
        cuisines = [c.strip() for c in cuisines.replace('[', '').replace(']', '').replace("'", "").split(',')]
        
    image_url = get_restaurant_image(name, cuisines)
    
    with st.container():
        # Custom CSS for card-like appearance
        st.markdown("""
        <style>
        div[data-testid="stVerticalBlock"] > div {
            border-radius: 10px;
        }
        </style>
        """, unsafe_allow_html=True)
        
        col1, col2 = st.columns([2, 5])
        
        with col1:
            st.image(image_url, use_container_width=True)
            st.caption(f"#{rank} Recommended")
            
        with col2:
            st.subheader(name)
            # Use columns for compact details
            d1, d2 = st.columns(2)
            d1.markdown(f"⭐ **{restaurant.get('rate', 'N/A')}** ({restaurant.get('votes', 0)} votes)")
            d2.markdown(f"💰 ₹{restaurant.get('price', 'N/A')} for two")
            
            st.markdown(f"📍 {restaurant.get('location', '')}")
            st.markdown(f"🍴 *{', '.join(cuisines[:4])}*")
            
            # Reason
            reason = restaurant.get('reason', '')
            if reason:
                st.info(f"**AI Reason:** {reason}")
                
        st.divider()

def main():
    # CSS Injection for Red Theme
    st.markdown("""
        <style>
        .stAppHeader {background-color: #CB202D;}
        h1 {color: #CB202D;}
        div.stButton > button:first-child {
            background-color: #CB202D;
            color: white;
            border-radius: 8px;
        }
        </style>
    """, unsafe_allow_html=True)

    st.title("Zomato - Restaurant Recommendation")
    st.markdown("### 🤖 Best-in-class features for your food cravings")
    
    # Sidebar
    st.sidebar.header("Filter Options")
    
    try:
        df = load_data()
        st.sidebar.success(f"Loaded {len(df)} restaurants")
    except FileNotFoundError as e:
        st.error(str(e))
        st.stop()
        
    # Get unique cities
    all_cities = sorted(df['city'].dropna().unique()) if 'city' in df.columns else []
    
    selected_city = st.sidebar.selectbox(
        "Select City",
        options=all_cities,
        index=all_cities.index('Banashankari') if 'Banashankari' in all_cities else 0
    )
    
    max_price = st.sidebar.slider(
        "Maximum Budget (for two)",
        min_value=100,
        max_value=5000,
        value=800,
        step=50
    )
    
    if st.sidebar.button("Find Best Restaurants", type="primary"):
        with st.spinner("AI is analyzing reviews and ratings..."):
            # 1. Integrate Data
            user_input = {'city': selected_city, 'price': max_price}
            integrator = DataIntegrator(df)
            filtered_df = integrator.integrate(user_input)
            
            if len(filtered_df) == 0:
                st.warning("No restaurants found matching your criteria. Try increasing your budget.")
            else:
                # 2. Generate Recommendations
                recommender = GroqEngine()
                
                # Check for API key warning using st.toast or warning
                if not os.getenv('GROQ_API_KEY'):
                    st.toast("Using fallback mode (Rule-based). Set GROQ_API_KEY for AI recommendations.", icon="ℹ️")
                
                recommendations = recommender.generate_recommendations(
                    filtered_df,
                    user_input,
                    top_k=10
                )
                
                # 3. Display
                st.subheader(f"Top 10 Picks in {selected_city}")
                
                for idx, row in recommendations.iterrows():
                    display_restaurant_card(row, idx + 1)

if __name__ == "__main__":
    main()

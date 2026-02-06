import streamlit as st
import pandas as pd
import sys
import os
from pathlib import Path
from dotenv import load_dotenv

# Add parent directory to path to import from other phases
# We need to go up one level to access other phase packages
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
# Also look in phase2 for .env if not found in phase3
if not os.getenv('GROQ_API_KEY'):
    load_dotenv(Path(__file__).parent.parent / 'phase2' / '.env')

@st.cache_data
def load_data():
    """Load processed data from Phase 1"""
    # Try different paths to locate the data file
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

def display_restaurant_card(restaurant, rank):
    """Display a single restaurant as a card"""
    with st.container():
        col1, col2 = st.columns([1, 4])
        
        with col1:
            st.metric("Rank",f"#{rank}")
            rate = restaurant.get('rate', 'N/A')
            st.metric("Rating", f"{rate}/5", delta=f"{restaurant.get('votes', 0)} votes")
            
        with col2:
            st.subheader(restaurant.get('name', 'Unknown'))
            st.caption(f"{restaurant.get('rest_type', 'Restaurant')} • {restaurant.get('location', '')}")
            
            # Price and Cuisines
            c1, c2 = st.columns(2)
            c1.markdown(f"**Cost for two:** ₹{restaurant.get('price', 'N/A')}")
            
            cuisines = restaurant.get('cuisines', [])
            if isinstance(cuisines, list):
                c2.markdown(f"**Cuisines:** {', '.join(cuisines[:5])}")
            else:
                c2.markdown(f"**Cuisines:** {str(cuisines)}")
            
            # Features badges
            features = []
            if restaurant.get('online_order'):
                features.append("🛵 Online Order")
            if restaurant.get('book_table'):
                features.append("📅 Table Booking")
            if features:
                st.markdown(" ".join([f"`{f}`" for f in features]))
                
            # Reason
            reason = restaurant.get('reason', '')
            if reason:
                st.info(f"**Why Recommended:** {reason}")
                
        st.divider()

def main():
    st.title("🍽️ Zomato AI Restaurant Recommender")
    st.markdown("Get personalized restaurant recommendations based on your preferences.")
    
    # Sidebar
    st.sidebar.header("Your Preferences")
    
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
    
    if st.sidebar.button("Find Restaurants", type="primary"):
        with st.spinner("Finding the best spots for you..."):
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
                    st.toast("⚠️ Using fallback mode (Rule-based). Set GROQ_API_KEY for AI recommendations.", icon="ℹ️")
                
                recommendations = recommender.generate_recommendations(
                    filtered_df,
                    user_input,
                    top_k=10
                )
                
                # 3. Display
                st.subheader(f"Top Recommendations in {selected_city}")
                st.markdown(f"Found **{len(filtered_df)}** valid options, showing top **10**.")
                
                for idx, row in recommendations.iterrows():
                    display_restaurant_card(row, idx + 1)

if __name__ == "__main__":
    main()

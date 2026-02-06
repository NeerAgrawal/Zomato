# Zomato AI Restaurant Recommendation Service - Architecture Document

## Table of Contents
1. [System Overview](#system-overview)
2. [Dataset Information](#dataset-information)
3. [Phase-wise Development Architecture](#phase-wise-development-architecture)
   - [STEP 1: Input the Zomato Data](#step-1-input-the-zomato-data)
   - [STEP 2: User Input](#step-2-user-input)
   - [STEP 3: Integrate](#step-3-integrate)
   - [STEP 4: Recommendation](#step-4-recommendation)
   - [STEP 5: Display to the User](#step-5-display-to-the-user)
   - [STEP 6: Frontend Application & API Integration](#step-6-frontend-application--api-integration)
4. [Technical Stack Recommendations](#technical-stack-recommendations)
5. [Data Flow Diagram](#data-flow-diagram)
6. [Recommendation Approach: Groq LLM](#recommendation-approach-groq-llm)

---

## System Overview

The Zomato AI Restaurant Recommendation Service is an intelligent system that recommends restaurants to users based on their city preference and budget constraints. The system processes restaurant data from a Hugging Face dataset and provides personalized recommendations using a **Groq LLM (Large Language Model)** for intelligent ranking and reasoning.

### Key Features
- **City-based Filtering**: Filter restaurants by user-specified city
- **Price-based Filtering**: Filter restaurants based on approximate cost for two people
- **LLM-powered Recommendations**: Use Groq model to analyze filtered data and generate ranked recommendations with natural-language reasoning
- **User-friendly Interface**: Display recommendations with relevant restaurant details

### Interface Strategy
- **Development Phase**: **CLI (Command Line Interface)** — Use CLI for development, testing, and validation.
- **Later Phase**: **WebUI** — Migrate to a web-based interface (e.g., Streamlit, Flask, or FastAPI) for end users.

### Input Parameters
- **City**: User-specified city name (e.g., "Banashankari", "Basavanagudi")
- **Price**: Budget range or maximum price for two people (e.g., 800, 1000)

---

## Dataset Information

### Source
- **Dataset**: `ManikaSaini/zomato-restaurant-recommendation`
- **Platform**: Hugging Face Datasets
- **Size**: ~51,700 rows
- **Format**: CSV (converted to Parquet)

### Dataset Schema
| Column Name | Data Type | Description | Example |
|------------|-----------|-------------|---------|
| `url` | string | Restaurant URL | "https://www.zomato.com/..." |
| `address` | string | Full address | "942, 21st Main Road..." |
| `name` | string | Restaurant name | "Jalsa" |
| `online_order` | string | Online ordering availability | "Yes"/"No" |
| `book_table` | string | Table booking availability | "Yes"/"No" |
| `rate` | string | Rating | "4.1/5" |
| `votes` | int64 | Number of votes | 775 |
| `phone` | string | Contact number | "080 42297555..." |
| `location` | string | Location/Area | "Banashankari" |
| `rest_type` | string | Restaurant type | "Casual Dining" |
| `dish_liked` | string | Popular dishes | "Pasta, Lunch Buffet..." |
| `cuisines` | string | Cuisine types | "North Indian, Mughlai..." |
| `approx_cost(for two people)` | string | Price range | "800" |
| `reviews_list` | string | Customer reviews | "[('Rated 4.0', '...')]" |
| `menu_item` | string | Menu items | "[]" |
| `listed_in(type)` | string | Listing category | "Buffet" |
| `listed_in(city)` | string | City listing | "Banashankari" |

### Key Fields for Recommendation
- **Location/City**: Primary filter (`location`, `listed_in(city)`)
- **Price**: Cost filter (`approx_cost(for two people)`)
- **Rating**: Quality indicator (`rate`, `votes`)
- **Cuisines**: Preference matching (`cuisines`)
- **Restaurant Type**: Category matching (`rest_type`)
- **Reviews**: Sentiment analysis (`reviews_list`)

---

## Phase-wise Development Architecture

### STEP 1: Input the Zomato Data

#### Objective
Load, validate, and preprocess the Zomato restaurant dataset from Hugging Face.

#### Architecture Components

**1.1 Data Loading Module**
- **Purpose**: Fetch dataset from Hugging Face
- **Technology**: `datasets` library (Hugging Face)
- **Process**:
  ```python
  from datasets import load_dataset
  dataset = load_dataset("ManikaSaini/zomato-restaurant-recommendation")
  ```

**1.2 Data Validation Module**
- **Purpose**: Validate data integrity and completeness
- **Checks**:
  - Missing values detection
  - Data type validation
  - Duplicate detection
  - Schema validation

**1.3 Data Preprocessing Module**
- **Purpose**: Clean and transform raw data
- **Operations**:
  - **Rate Normalization**: Convert "4.1/5" to float (4.1)
  - **Price Parsing**: Convert price strings to numeric values
  - **Location Standardization**: Normalize city/location names
  - **Text Cleaning**: Clean `cuisines`, `dish_liked`, `reviews_list`
  - **Missing Value Handling**: Impute or remove missing values
  - **Feature Extraction**: Extract numeric features from text fields

**1.4 Data Storage Module**
- **Purpose**: Store processed data for efficient access
- **Options**:
  - **Local Storage**: CSV/Parquet files
  - **Database**: SQLite/PostgreSQL (for larger datasets)
  - **In-Memory**: Pandas DataFrame (for development)

**1.5 Data Structure**
```python
{
    "restaurant_id": int,
    "name": str,
    "location": str,
    "city": str,
    "rate": float,
    "votes": int,
    "price": int,
    "cuisines": list[str],
    "rest_type": str,
    "online_order": bool,
    "book_table": bool,
    "dish_liked": list[str],
    "address": str,
    "phone": str,
    "reviews": list[dict],
    "menu_items": list[str]
}
```

#### Output
- Cleaned and structured dataset ready for querying
- Data validation report
- Preprocessing statistics

---

### STEP 2: User Input

#### Objective
Create an interface to capture user preferences (city and price).

#### Architecture Components

**2.1 Input Interface Module**
- **Purpose**: Collect user inputs
- **Input Fields**:
  - **City**: Text input or dropdown (with validation)
  - **Price**: Numeric input or range slider (with validation)

**2.2 Input Validation Module**
- **Purpose**: Validate user inputs
- **Validations**:
  - City exists in dataset
  - Price is positive numeric value
  - Price range is reasonable (e.g., 100-5000)
  - Handle case-insensitive city matching

**2.3 Input Processing Module**
- **Purpose**: Normalize and format user inputs
- **Operations**:
  - City name normalization (title case, trim spaces)
  - Price range conversion (single value or range)
  - Input sanitization

**2.4 User Interface Strategy**

**Phase 1 — Development: Command Line Interface (CLI)** *(Current)*
- Use CLI for development, testing, and validation.
- Simple, fast iteration without front-end overhead.
```python
def get_user_input():
    city = input("Enter city name: ")
    price = float(input("Enter maximum price for two: "))
    return {"city": city, "price": price}
```

**Phase 2 — Later: WebUI**
- Migrate to web interface for end users.
- **Option A**: Streamlit — widgets, city dropdown, price slider.
- **Option B**: Flask/FastAPI — HTML form, REST API (`POST /api/user-input`).
- **Option C**: Other web framework — card-based layout, responsive design.

**2.5 Input Data Structure**
```python
{
    "city": str,        # e.g., "Banashankari"
    "price": int,       # e.g., 800
    "price_range": {    # Optional: for range queries
        "min": int,
        "max": int
    }
}
```

#### Output
- Validated user input dictionary
- Error messages for invalid inputs

---

### STEP 3: Integrate

#### Objective
Integrate user inputs with the dataset to filter and prepare data for recommendation.

#### Architecture Components

**3.1 Data Filtering Module**
- **Purpose**: Filter restaurants based on user criteria
- **Filters**:
  - **City Filter**: Match `location` or `listed_in(city)` with user city
  - **Price Filter**: Filter by `approx_cost(for two people)` <= user price
  - **Optional Filters**: Rating threshold, restaurant type, cuisines

**3.2 Data Query Module**
- **Purpose**: Efficiently query filtered data
- **Implementation**:
  ```python
  def filter_restaurants(df, city, max_price):
      filtered = df[
          (df['location'].str.contains(city, case=False, na=False)) |
          (df['listed_in(city)'].str.contains(city, case=False, na=False))
      ]
      filtered = filtered[filtered['price'] <= max_price]
      return filtered
  ```

**3.3 Feature Engineering Module**
- **Purpose**: Create features for recommendation algorithm
- **Features**:
  - **Rating Score**: Normalized rating (0-1 scale)
  - **Popularity Score**: Based on votes (log-normalized)
  - **Price Score**: Inverse of price (lower price = higher score)
  - **Completeness Score**: Based on available data fields
  - **Text Features**: TF-IDF vectors from cuisines, dish_liked, reviews

**3.4 Data Integration Pipeline**
```
User Input → Validation → Filtering → Feature Engineering → Prepared Dataset
```

**3.5 Integration Logic**
```python
def integrate_user_input_with_data(user_input, dataset):
    # Step 1: Filter by city
    city_filtered = filter_by_city(dataset, user_input['city'])
    
    # Step 2: Filter by price
    price_filtered = filter_by_price(city_filtered, user_input['price'])
    
    # Step 3: Feature engineering
    enhanced_data = engineer_features(price_filtered)
    
    # Step 4: Prepare for recommendation
    return prepare_for_recommendation(enhanced_data)
```

#### Output
- Filtered dataset matching user criteria
- Enhanced features for recommendation algorithm
- Statistics: number of restaurants found, price distribution, etc.

---

### STEP 4: Recommendation

#### Objective
Generate intelligent restaurant recommendations using the **Groq LLM (Large Language Model)**. The LLM analyzes the filtered restaurant data and returns ranked recommendations with natural-language reasoning (e.g., why each restaurant was chosen).

#### Architecture Components

**4.1 Recommendation Engine: Groq LLM**
- **Purpose**: Core recommendation and ranking logic
- **Technology**: **Groq API** (LLM)
- **Approach**: Send filtered restaurant data + user context (city, price) to Groq; LLM returns ranked list and reasoning.

**4.2 Groq Integration Module**
- **Purpose**: Call Groq API with structured prompt and filtered data
- **Input to LLM**:
  - User context: city, max price
  - Filtered restaurant list (name, rate, votes, price, cuisines, rest_type, dish_liked, location, etc.)
  - Optional: sample of reviews or dish_liked for richer context
- **Output from LLM**: Ranked list of restaurant names/IDs and short reasoning per recommendation (or a single explanation block)

**4.3 Prompt Design**
- **System prompt**: Define role (e.g., “You are a restaurant recommendation assistant for Zomato data.”) and output format (e.g., JSON or numbered list with reason).
- **User prompt**: Include user city, max price, and a structured summary of each restaurant (name, rating, votes, price, cuisines, type, popular dishes).
- **Output format**: Request structured output (e.g., JSON with `rank`, `restaurant_name`, `reason`) so the app can parse and display consistently.

**4.4 Context Preparation for LLM**
- **Purpose**: Build a concise, token-efficient summary of filtered restaurants for the prompt
- **Operations**:
  - Limit number of restaurants sent (e.g., top 50 by rating/votes from filtered set) to stay within context window
  - Format each restaurant as a short line or key-value block (name, rate, votes, price, cuisines, rest_type, dish_liked)
  - Optionally truncate or summarize reviews if included

**4.5 Response Parsing Module**
- **Purpose**: Parse LLM response into a structured list
- **Process**: Extract ranked list and reasons; map back to full restaurant records (name, rate, price, location, etc.) for display
- **Fallback**: If LLM returns invalid/unparseable output, fall back to rule-based ranking (e.g., sort by rating, votes, price)

**4.6 Recommendation Pipeline**
```python
def generate_recommendations_with_groq(filtered_data, user_input, top_k=10):
    # Step 1: Prepare context for LLM (summarize filtered restaurants)
    context = prepare_llm_context(filtered_data, user_input)
    
    # Step 2: Build prompt (system + user with context)
    prompt = build_recommendation_prompt(context, user_input, top_k)
    
    # Step 3: Call Groq API
    response = groq_client.chat.completions.create(
        model="llama-3.1-70b-versatile",  # or chosen Groq model
        messages=[{"role": "user", "content": prompt}],
        temperature=0.3
    )
    
    # Step 4: Parse LLM response to get ranked list + reasons
    ranked_with_reasons = parse_llm_response(response.choices[0].message.content)
    
    # Step 5: Map back to full restaurant records and return
    recommendations = map_to_restaurant_records(ranked_with_reasons, filtered_data)
    return recommendations
```

**4.7 Groq API Configuration**
- **API Key**: Store Groq API key in environment variable (e.g., `GROQ_API_KEY`); never commit to repo
- **Model**: Use a Groq-supported model (e.g., `llama-3.1-70b-versatile` or `mixtral-8x7b`) for balance of speed and quality
- **Rate limits**: Handle rate limits and retries; optionally cache results for same (city, price) to reduce calls

**4.8 Fallback Strategy**
- If Groq API is unavailable or returns invalid output: use rule-based ranking (e.g., sort by rating ↓, votes ↓, price ↑) and return top K without LLM reasoning

#### Output
- Ranked list of restaurant recommendations (from Groq LLM)
- Per-restaurant or overall reasoning text (from LLM)
- Top K restaurants (default: 10), with full details for display

---

### STEP 5: Display to the User

#### Objective
Present recommendations to users in a clear and user-friendly format.

#### Architecture Components

**5.1 Display Format Module**
- **Purpose**: Format recommendation data for display
- **Phase 1 — Development**: **CLI (Console)** — Text-based display in terminal
- **Phase 2 — Later**: **WebUI** — Web interface (Streamlit, Flask/FastAPI, or other), HTML/CSS, or JSON API for clients

**5.2 Restaurant Card Component**
- **Purpose**: Display individual restaurant information
- **Information to Display**:
  - Restaurant name
  - Rating (with stars/visual indicator)
  - Price range
  - Location/Address
  - Cuisines
  - Restaurant type
  - Online order availability
  - Table booking availability
  - Popular dishes
  - Contact information
  - Recommendation score/reason

**5.3 Display Layout Strategy**

**Phase 1 — Development: CLI (Console)**
- Primary display during development.
```python
def display_recommendations(recommendations):
    for idx, restaurant in enumerate(recommendations, 1):
        print(f"\n{idx}. {restaurant['name']}")
        print(f"   Rating: {restaurant['rate']} ({restaurant['votes']} votes)")
        print(f"   Price: ₹{restaurant['price']} for two")
        print(f"   Location: {restaurant['location']}")
        print(f"   Cuisines: {', '.join(restaurant['cuisines'])}")
        print(f"   Type: {restaurant['rest_type']}")
        if restaurant.get('reason'):
            print(f"   Why recommended: {restaurant['reason']}")
```

**Phase 2 — Later: WebUI**
- **Option A**: Streamlit — interactive widgets, charts, restaurant cards
- **Option B**: Flask/FastAPI + HTML — card-based layout, responsive, filter/sort
- **Option C**: Other — map view, images, export (CSV/PDF)

**5.4 Visualization Components (Optional)**
- **Rating Distribution**: Bar chart of ratings
- **Price Distribution**: Histogram of prices
- **Cuisine Distribution**: Pie chart of cuisines
- **Location Map**: Map view of restaurant locations

**5.5 Display Features**
- **Sorting Options**: By rating, price, popularity
- **Filtering Options**: Additional filters (cuisine, type)
- **Pagination**: Display in pages if many results
- **Export Options**: Export to CSV, PDF

**5.6 Display Data Structure**
```python
{
    "total_results": int,
    "recommendations": [
        {
            "rank": int,
            "name": str,
            "rating": float,
            "votes": int,
            "price": int,
            "location": str,
            "cuisines": list[str],
            "rest_type": str,
            "online_order": bool,
            "book_table": bool,
            "popular_dishes": list[str],
            "address": str,
            "phone": str,
            "recommendation_score": float,
            "reason": str  # Why this restaurant was recommended
        }
    ],
    "statistics": {
        "avg_rating": float,
        "price_range": {"min": int, "max": int},
        "top_cuisines": list[str]
    }
}
```

#### Output
- Formatted recommendation display
- User-friendly interface
- Interactive elements (if web-based)

---

### STEP 6: Frontend Application & API Integration

#### Objective
Develop a decoupled web frontend and a robust backend API to separate the user interface from the logic. This enables scalability and allows multiple clients (web, mobile) to consume the recommendation service.

#### Architecture Components

**6.1 Backend API Layer**
- **Purpose**: Expose recommendation logic as a RESTful API
- **Technology**: **FastAPI** (recommended) or Flask
- **Endpoints**:
  - `POST /api/recommend`: Accepts `{city, price}`, returns JSON list of recommendations
  - `GET /api/health`: Health check
  - `GET /api/cities`: Returns list of available cities

**6.2 Frontend Application**
- **Purpose**: User-facing interface for inputs and results
- **Technology**: **React / Next.js** (Modern) or **HTML/CSS/Vanilla JS** (Simple)
- **Features**:
  - Input form (City search with autocomplete, Price slider)
  - Loading states (skeletons/spinners during LLM processing)
  - Responsive Restaurant Cards (Grid/List view)
  - Error handling toasts

#### Integration Details (Frontend ↔ Backend)

**Communication Protocol**: HTTP/JSON

**Request Flow**:
1. **User Action**: User selects "Bangalore" and inputs "2000" → Clicks "Find".
2. **Frontend**:
   ```javascript
   const payload = { city: "Bangalore", price: 2000 };
   const response = await fetch('http://localhost:8000/api/recommend', {
       method: 'POST',
       headers: { 'Content-Type': 'application/json' },
       body: JSON.stringify(payload)
   });
   ```
3. **Backend (API)**:
   - Receives request.
   - Calls `DataIntegrator` (Phase 3) to filter data.
   - Calls `GroqEngine` (Phase 4) to generate ranked list.
   - Returns structured JSON response.
4. **Frontend**: Parses JSON and renders `RestaurantCard` components.

**API Response Schema**:
```json
{
  "status": "success",
  "data": {
    "count": 10,
    "recommendations": [
      {
        "rank": 1,
        "name": "Truffles",
        "reason": "Known for great burgers...",
        "details": { ... }
      }
    ]
  }
}
```

#### Output
- Functional REST API serving logic from Phase 4
- Aesthetic Frontend Page consuming said API
- End-to-end decoupled flow

---

## Technical Stack Recommendations

### Data Processing & Recommendation
- **Python**: Core programming language
- **Pandas**: Data manipulation and analysis
- **NumPy**: Numerical computations (if needed for preprocessing)
- **Groq API**: LLM for recommendation (Groq client SDK / `requests`)
- **Environment**: Store `GROQ_API_KEY` in env; use a Groq-supported model (e.g., Llama, Mixtral)

### Data Loading
- **Hugging Face Datasets**: `datasets` library for loading dataset
- **Pandas**: CSV/Parquet file handling

### User Interface
- **Development (current)**: **CLI** — Command-line interface for development, testing, and validation
- **Later**: **WebUI** — Migrate to web interface:
  - **Option 1**: **Streamlit** — Quick to build, widgets, good for prototyping
  - **Option 2**: **Flask/FastAPI** — Custom web UI, REST API support
  - **Option 3**: Other front-end (React, etc.) consuming API

### Data Storage
- **Development**: Pandas DataFrame (in-memory)
- **Production**: SQLite (small scale) or PostgreSQL (large scale)
- **File Format**: Parquet (efficient) or CSV (readable)

### Visualization
- **Matplotlib/Seaborn**: Charts and graphs
- **Plotly**: Interactive visualizations (if using Streamlit)

### Development Tools
- **Jupyter Notebook**: For data exploration and prototyping
- **VS Code/PyCharm**: IDE for development
- **Git**: Version control

---

---

## System Architecture Diagram

This diagram represents the high-level system design, illustrating how the Frontend, Backend API, Core Services, and External Integrations interact.

```
┌──────────────────────────────────────────┐
│              Client Layer                │
│ ┌───────────────┐      ┌───────────────┐ │
│ │ CLI App       │      │ Web Frontend  │ │
│ │ (Phase 5)     │      │ (React/JS)    │ │
│ └───────┬───────┘      └───────┬───────┘ │
└─────────┼──────────────────────┼─────────┘
          │                      │
          ▼                      ▼
┌──────────────────────────────────────────┐     ┌──────────────────────┐
│       API & Orchestration Layer          │     │  External Services   │
│ ┌──────────────────────────────────────┐ │     │ ┌──────────────────┐ │
│ │           FastAPI Gateway            │ │     │ │    Groq Cloud    │ │
│ └───────────────────┬──────────────────┘ │     │ │       API        │ │
│                     ▼                    │     │ └─────────▲────────┘ │
│ ┌──────────────────────────────────────┐ │     │           │          │
│ │       GroqEngine Orchestrator        │─┼─────┼───────────┘          │
│ └───────────────────┬──────────────────┘ │     │                      │
└─────────────────────┼────────────────────┘     └──────────────────────┘
                      │
                      ▼
┌──────────────────────────────────────────┐     ┌──────────────────────┐
│           Core Logic Services            │     │   Data Persistence   │
│ ┌─────────────────┐  ┌─────────────────┐ │     │ ┌──────────────────┐ │
│ │ Data Integrator │--│ Feature Engineer│ │     │ │  Processed Data  │ │
│ └───────┬─────────┘  └─────────────────┘ │     │ │ (Parquet/Pickle) │ │
│         │                                │     │ └─────────▲────────┘ │
│         │           Reads                │     │           │          │
│         └────────────────────────────────┼─────┼───────────┘          │
└──────────────────────────────────────────┘     └──────────────────────┘
```

---

## Data Flow Diagram


```
┌─────────────────┐
│  Hugging Face   │
│     Dataset     │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  STEP 1: Data   │
│   Loading &     │
│  Preprocessing  │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  Processed      │
│  Dataset        │
└────────┬────────┘
         │
         ▼
┌─────────────────┐      ┌─────────────────┐
│  STEP 2: User   │─────▶│  STEP 3:       │
│     Input       │      │   Integration   │
│  (City, Price)  │      │   & Filtering  │
└─────────────────┘      └────────┬────────┘
                                  │
                                  ▼
                         ┌─────────────────┐
                         │  Filtered       │
                         │  Dataset        │
                         └────────┬────────┘
                                  │
                                  ▼
                         ┌─────────────────┐
                         │  STEP 4:        │
                         │  Recommendation │
                         │  (Groq LLM)     │
                         └────────┬────────┘
                                  │
                                  ▼
                         ┌─────────────────┐
                         │  Ranked         │
                         │  Recommendations│
                         └────────┬────────┘
                                  │
                                  ▼
                         ┌─────────────────┐
                         │  STEP 5:        │
                         │  Display to     │
                         │     User        │
                         └────────┬────────┘
                                  │
                                  ▼
                         ┌─────────────────┐
                         │  STEP 6:        │
                         │  Frontend App   │
                         │  & API Layer    │
                         └─────────────────┘
```

---

## Recommendation Approach: Groq LLM

### Primary Method: Groq LLM–Based Recommendation

Recommendations are generated by sending the **filtered restaurant list** and **user context (city, price)** to a **Groq LLM**. The model returns a **ranked list** and **reasoning** (why each restaurant was recommended).

#### Step 1: Prepare Context for LLM
- Filter restaurants by city and price (from STEP 3).
- Build a compact summary per restaurant: name, rate, votes, price, cuisines, rest_type, dish_liked, location.
- Optionally cap the list (e.g., top 30–50 by rating/votes) to fit context window and control cost.

#### Step 2: Build Prompt
- **System message**: Define role (e.g., Zomato recommendation assistant) and output format (e.g., JSON with rank, restaurant name, short reason).
- **User message**: Include user city, max price, and the summarized restaurant list. Ask for top K recommendations with a brief reason per pick.

#### Step 3: Call Groq API
- Use Groq client (or HTTP API) with a chosen model (e.g., `llama-3.1-70b-versatile`).
- Pass messages; use moderate temperature (e.g., 0.3) for stable ordering.
- Handle errors and rate limits; retry with backoff if needed.

#### Step 4: Parse Response
- Parse LLM output (JSON or structured text) to get ordered list and reasons.
- Map names/IDs back to full restaurant records for display.

#### Step 5: Return Recommendations
- Return ranked list with full details + `reason` field for display (CLI now, WebUI later).

### Fallback: Rule-Based Ranking

When Groq is unavailable or response is invalid, use a simple rule-based ranking:

```python
# Sort by: Rating (desc) → Votes (desc) → Price (asc)
recommendations = filtered_data.sort_values(
    by=['rate', 'votes', 'price'],
    ascending=[False, False, True]
).head(10)
```

### Groq Configuration Notes

- **API key**: Use environment variable `GROQ_API_KEY`.
- **Model**: Prefer Groq’s fast models (e.g., Llama 3.1 70B, Mixtral 8x7B) for low latency.
- **Token usage**: Keep prompt size bounded (summarize restaurants, limit count) to avoid excessive cost and context limits.

---

## Implementation Considerations

### Performance Optimization
- **Indexing**: Create indexes on frequently queried columns (city, price)
- **Caching**: Cache filtered results for repeated queries
- **Lazy Loading**: Load data only when needed
- **Batch Processing**: Process recommendations in batches

### Error Handling
- **Invalid City**: Suggest similar city names or show available cities
- **No Results**: Provide helpful message and suggest broader filters
- **Data Quality**: Handle missing values gracefully
- **Input Validation**: Validate all user inputs

### Scalability
- **Database**: Use database for large datasets
- **API**: RESTful API for multiple clients
- **Caching**: Cache recommendations for common queries
- **Async Processing**: Use async operations for better performance

### Testing Strategy
- **Unit Tests**: Test individual modules
- **Integration Tests**: Test data flow between modules
- **Edge Cases**: Test with empty results, invalid inputs
- **Performance Tests**: Test with large datasets

---

## Next Steps (Post-Implementation)

1. **Groq Integration**: Integrate Groq API; tune prompts and output format
2. **WebUI Migration**: Move from CLI to web interface (Streamlit, Flask, or FastAPI)
3. **Prompt Tuning**: Refine LLM prompts for better ranking and reasoning
4. **Caching**: Cache LLM responses for repeated (city, price) queries
5. **User Feedback**: Collect feedback to improve prompts and display
6. **Fallback Testing**: Ensure rule-based fallback works when Groq is unavailable

---

## Conclusion

This architecture provides a comprehensive, phase-wise approach to building the Zomato AI Restaurant Recommendation Service. Each phase is designed to be independent yet integrated, allowing for iterative development and testing. **Recommendations are powered by the Groq LLM**; the UI is **CLI during development**, with a **WebUI planned for later**.

**Key Success Factors**:
- Clean and well-preprocessed data
- Efficient filtering and querying (city, price)
- Groq LLM integration with clear prompts and parsing
- Fallback to rule-based ranking when LLM is unavailable
- CLI for development; WebUI for production use
- Robust error handling and API key management

**Estimated Development Time**:
- STEP 1: 2-3 days
- STEP 2: 1-2 days
- STEP 3: 2-3 days
- STEP 4: 3-5 days
- STEP 5: 2-3 days
- **Total**: 10-16 days (depending on complexity)

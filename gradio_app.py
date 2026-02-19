"""
Retail Demand Prediction - Gradio Interface
=============================================
Interactive web interface for demand prediction using the trained XGBoost model.
"""

import gradio as gr
import pandas as pd
import numpy as np
import xgboost as xgb
import joblib

# ============================================================
# LOAD TRAINED MODEL AND ARTIFACTS
# ============================================================

print("[INFO] Loading trained model and artifacts...")

# Load the trained XGBoost model
model = xgb.XGBRegressor()
model.load_model('demand_model.json')
print("[SUCCESS] Model loaded: demand_model.json")

# Load label encoders
label_encoders = joblib.load('label_encoders.pkl')
print("[SUCCESS] Label encoders loaded: label_encoders.pkl")

# Load feature columns
feature_cols = joblib.load('feature_columns.pkl')
print("[SUCCESS] Feature columns loaded: feature_columns.pkl")

# Category mappings from encoders
CATEGORIES = list(label_encoders['category'].classes_)
SEASONS = list(label_encoders['season'].classes_)
DAYS_OF_WEEK = list(label_encoders['day_of_week'].classes_)

print(f"[INFO] Categories: {CATEGORIES}")
print(f"[INFO] Seasons: {SEASONS}")
print(f"[INFO] Days: {DAYS_OF_WEEK}")


# ============================================================
# PREDICTION FUNCTION
# ============================================================

def predict_demand(
    price, cost_price, discount_percent, stock_available,
    competitor_price, rating, review_count, is_promoted,
    is_weekend, days_since_launch, category, season, day_of_week,
    year, month, day, is_holiday
):
    """
    Predict demand for given input parameters.
    """
    try:
        # Encode categorical variables
        category_encoded = label_encoders['category'].transform([category])[0]
        season_encoded = label_encoders['season'].transform([season])[0]
        day_of_week_encoded = label_encoders['day_of_week'].transform([day_of_week])[0]
        
        # Calculate derived features
        price_difference = price - competitor_price
        discount_value = price * (discount_percent / 100)
        price_ratio = price / competitor_price if competitor_price > 0 else 1
        profit_margin = (price - cost_price) / price if price > 0 else 0
        
        # Calculate week of year from month/day (approximation)
        week_of_year = int((month - 1) * 4.33 + (day // 7) + 1)
        week_of_year = min(max(week_of_year, 1), 52)
        
        # Create feature dictionary
        features = {
            'price': price,
            'cost_price': cost_price,
            'discount_percent': discount_percent,
            'stock_available': stock_available,
            'competitor_price': competitor_price,
            'rating': rating,
            'review_count': review_count,
            'is_promoted': 1 if is_promoted else 0,
            'is_weekend': 1 if is_weekend else 0,
            'days_since_launch': days_since_launch,
            'year': year,
            'month_num': month,
            'day': day,
            'week_of_year': week_of_year,
            'price_difference': price_difference,
            'discount_value': discount_value,
            'price_ratio': price_ratio,
            'profit_margin': profit_margin,
            'category_encoded': category_encoded,
            'season_encoded': season_encoded,
            'day_of_week_encoded': day_of_week_encoded,
            'is_holiday': 1 if is_holiday else 0
        }
        
        # Create DataFrame with correct column order
        input_df = pd.DataFrame([features])[feature_cols]
        
        # Make prediction
        prediction = model.predict(input_df)[0]
        prediction = max(0, round(prediction))  # Ensure non-negative integer
        
        # Calculate expected revenue
        expected_revenue = prediction * price
        
        # Generate insights
        insights = generate_insights(prediction, price, discount_percent, is_promoted, category, stock_available)
        
        return (
            f"🎯 Predicted Demand: {prediction:,} units",
            f"💰 Expected Revenue: ${expected_revenue:,.2f}",
            insights
        )
        
    except Exception as e:
        return (
            f"❌ Error: {str(e)}",
            "",
            "Please check your inputs and try again."
        )


def generate_insights(prediction, price, discount, is_promoted, category, stock):
    """
    Generate business insights based on prediction.
    """
    insights = []
    
    # Stock warning
    if prediction > stock:
        insights.append(f"⚠️ WARNING: Predicted demand ({prediction}) exceeds available stock ({stock}). Consider restocking!")
    else:
        stock_coverage = (stock / prediction * 100) if prediction > 0 else 100
        insights.append(f"✅ Stock coverage: {stock_coverage:.0f}% of predicted demand")
    
    # Promotion insight
    if is_promoted:
        insights.append("📢 Product is promoted - demand typically increases by 8-17%")
    else:
        insights.append("💡 TIP: Promotions can boost demand by 8-17% on average")
    
    # Discount insight
    if discount > 20:
        insights.append(f"🏷️ High discount ({discount}%) - Good for clearing inventory")
    elif discount < 5:
        insights.append("💎 Low discount - Consider premium positioning")
    
    # Category-specific insight
    category_tips = {
        'Electronics': "📱 Electronics: Focus on tech specs and warranty",
        'Fashion': "👗 Fashion: Seasonal trends impact demand significantly",
        'Home': "🏠 Home: Bundle with complementary products",
        'Sports': "⚽ Sports: Promotions work best here (+17% lift)",
        'Beauty': "💄 Beauty: Emphasize ratings and reviews"
    }
    if category in category_tips:
        insights.append(category_tips[category])
    
    return "\n".join(insights)


def batch_predict(file):
    """
    Process CSV file for batch predictions.
    """
    try:
        df = pd.read_csv(file.name)
        
        results = []
        for _, row in df.iterrows():
            # Encode categoricals
            cat_enc = label_encoders['category'].transform([row['category']])[0]
            sea_enc = label_encoders['season'].transform([row['season']])[0]
            dow_enc = label_encoders['day_of_week'].transform([row['day_of_week']])[0]
            
            # Derived features
            price_diff = row['price'] - row['competitor_price']
            disc_val = row['price'] * (row['discount_percent'] / 100)
            price_ratio = row['price'] / row['competitor_price']
            profit_margin = (row['price'] - row['cost_price']) / row['price']
            week = int(row.get('month', 1) * 4.33)
            
            features = {
                'price': row['price'],
                'cost_price': row['cost_price'],
                'discount_percent': row['discount_percent'],
                'stock_available': row['stock_available'],
                'competitor_price': row['competitor_price'],
                'rating': row['rating'],
                'review_count': row['review_count'],
                'is_promoted': row['is_promoted'],
                'is_weekend': row['is_weekend'],
                'days_since_launch': row['days_since_launch'],
                'year': row.get('year', 2024),
                'month_num': row.get('month', 1),
                'day': row.get('day', 1),
                'week_of_year': week,
                'price_difference': price_diff,
                'discount_value': disc_val,
                'price_ratio': price_ratio,
                'profit_margin': profit_margin,
                'category_encoded': cat_enc,
                'season_encoded': sea_enc,
                'day_of_week_encoded': dow_enc,
                'is_holiday': row.get('is_holiday', 0)
            }
            
            input_df = pd.DataFrame([features])[feature_cols]
            pred = max(0, round(model.predict(input_df)[0]))
            results.append(pred)
        
        df['predicted_units_sold'] = results
        df['predicted_revenue'] = df['predicted_units_sold'] * df['price']
        
        output_path = 'batch_predictions.csv'
        df.to_csv(output_path, index=False)
        
        summary = f"""
✅ Batch Prediction Complete!

📊 Processed: {len(df)} records
📈 Total Predicted Sales: {sum(results):,} units
💰 Total Predicted Revenue: ${df['predicted_revenue'].sum():,.2f}

📁 Results saved to: {output_path}
        """
        
        return summary, output_path
        
    except Exception as e:
        return f"❌ Error processing file: {str(e)}", None


# ============================================================
# GRADIO INTERFACE
# ============================================================

# Custom CSS for better styling
custom_css = """
.gradio-container {
    font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
}
.gr-button-primary {
    background-color: #2563eb !important;
}
"""

# Create the Gradio interface
with gr.Blocks(title="Retail Demand Prediction", css=custom_css, theme=gr.themes.Soft()) as demo:
    
    gr.Markdown("""
    # 🛒 Retail Demand Intelligence System
    ### Predict product demand using AI-powered XGBoost model
    
    **Model Performance:** R² = 0.9868 | **Cross-Validation:** Mean R² = 0.9752 (Std = 0.0019) ✓ Robust
    """)
    
    with gr.Tabs():
        
        # Tab 1: Single Prediction
        with gr.TabItem("📊 Single Prediction"):
            gr.Markdown("### Enter product details to predict demand")
            
            with gr.Row():
                with gr.Column():
                    gr.Markdown("#### 💵 Pricing")
                    price = gr.Number(label="Selling Price ($)", value=1000, minimum=0)
                    cost_price = gr.Number(label="Cost Price ($)", value=600, minimum=0)
                    competitor_price = gr.Number(label="Competitor Price ($)", value=1050, minimum=0)
                    discount_percent = gr.Slider(label="Discount (%)", minimum=0, maximum=50, value=10)
                
                with gr.Column():
                    gr.Markdown("#### 📦 Inventory & Product")
                    category = gr.Dropdown(label="Category", choices=CATEGORIES, value="Electronics")
                    stock_available = gr.Number(label="Stock Available", value=100, minimum=0)
                    days_since_launch = gr.Number(label="Days Since Launch", value=30, minimum=0)
                    rating = gr.Slider(label="Product Rating", minimum=1, maximum=5, step=0.1, value=4.0)
                    review_count = gr.Number(label="Review Count", value=50, minimum=0)
            
            with gr.Row():
                with gr.Column():
                    gr.Markdown("#### 📅 Timing")
                    season = gr.Dropdown(label="Season", choices=SEASONS, value="Winter")
                    day_of_week = gr.Dropdown(label="Day of Week", choices=DAYS_OF_WEEK, value="Saturday")
                    year = gr.Number(label="Year", value=2024, minimum=2020, maximum=2030)
                    month = gr.Slider(label="Month", minimum=1, maximum=12, value=12)
                    day = gr.Slider(label="Day", minimum=1, maximum=31, value=15)
                
                with gr.Column():
                    gr.Markdown("#### 🎯 Promotions & Flags")
                    is_promoted = gr.Checkbox(label="Is Promoted?", value=True)
                    is_weekend = gr.Checkbox(label="Is Weekend?", value=True)
                    is_holiday = gr.Checkbox(label="Is Holiday?", value=False)
            
            predict_btn = gr.Button("🔮 Predict Demand", variant="primary", size="lg")
            
            with gr.Row():
                demand_output = gr.Textbox(label="Predicted Demand", lines=1)
                revenue_output = gr.Textbox(label="Expected Revenue", lines=1)
            
            insights_output = gr.Textbox(label="💡 Business Insights", lines=5)
            
            predict_btn.click(
                fn=predict_demand,
                inputs=[
                    price, cost_price, discount_percent, stock_available,
                    competitor_price, rating, review_count, is_promoted,
                    is_weekend, days_since_launch, category, season, day_of_week,
                    year, month, day, is_holiday
                ],
                outputs=[demand_output, revenue_output, insights_output]
            )
        
        # Tab 2: Batch Prediction
        with gr.TabItem("📁 Batch Prediction"):
            gr.Markdown("""
            ### Upload CSV for batch predictions
            
            Your CSV should contain these columns:
            `price, cost_price, discount_percent, stock_available, competitor_price, 
            rating, review_count, is_promoted, is_weekend, days_since_launch, 
            category, season, day_of_week, year, month, day, is_holiday`
            """)
            
            file_input = gr.File(label="Upload CSV File", file_types=[".csv"])
            batch_btn = gr.Button("🚀 Run Batch Prediction", variant="primary")
            
            batch_output = gr.Textbox(label="Results Summary", lines=8)
            download_output = gr.File(label="Download Predictions")
            
            batch_btn.click(
                fn=batch_predict,
                inputs=[file_input],
                outputs=[batch_output, download_output]
            )
        
        # Tab 3: Price Elasticity Simulator
        with gr.TabItem("📉 Price Elasticity"):
            gr.Markdown("""
            ## Price Elasticity Simulator
            
            Simulate how price changes affect demand. The simulator updates ALL price-dependent features:
            - `price`, `price_difference`, `price_ratio`, `discount_value`, `profit_margin`
            """)
            
            with gr.Row():
                with gr.Column():
                    gr.Markdown("#### Base Product Settings")
                    pe_category = gr.Dropdown(label="Category", choices=CATEGORIES, value="Electronics")
                    pe_base_price = gr.Number(label="Base Price ($)", value=1000, minimum=0)
                    pe_cost_price = gr.Number(label="Cost Price ($)", value=600, minimum=0)
                    pe_competitor_price = gr.Number(label="Competitor Price ($)", value=1050, minimum=0)
                    pe_discount = gr.Slider(label="Discount (%)", minimum=0, maximum=50, value=10)
                    pe_stock = gr.Number(label="Stock Available", value=100, minimum=0)
                    pe_rating = gr.Slider(label="Rating", minimum=1, maximum=5, step=0.1, value=4.0)
                
                with gr.Column():
                    gr.Markdown("#### Price Change Simulation")
                    pe_change = gr.Slider(label="Price Change (%)", minimum=-20, maximum=20, value=5, step=1)
                    pe_is_promoted = gr.Checkbox(label="Is Promoted?", value=False)
            
            pe_btn = gr.Button("🔍 Simulate Price Impact", variant="primary")
            
            pe_output = gr.Textbox(label="Elasticity Analysis", lines=12)
            
            def simulate_elasticity(category, base_price, cost_price, competitor_price, 
                                   discount, stock, rating, price_change_pct, is_promoted):
                try:
                    # Encode category
                    cat_enc = label_encoders['category'].transform([category])[0]
                    sea_enc = label_encoders['season'].transform(['Winter'])[0]
                    dow_enc = label_encoders['day_of_week'].transform(['Saturday'])[0]
                    
                    def build_features(price):
                        return {
                            'price': price,
                            'cost_price': cost_price,
                            'discount_percent': discount,
                            'stock_available': stock,
                            'competitor_price': competitor_price,
                            'rating': rating,
                            'review_count': 50,
                            'is_promoted': 1 if is_promoted else 0,
                            'is_weekend': 1,
                            'days_since_launch': 30,
                            'year': 2024,
                            'month_num': 12,
                            'day': 15,
                            'week_of_year': 50,
                            'price_difference': price - competitor_price,
                            'discount_value': price * (discount / 100),
                            'price_ratio': price / competitor_price if competitor_price > 0 else 1,
                            'profit_margin': (price - cost_price) / price if price > 0 else 0,
                            'category_encoded': cat_enc,
                            'season_encoded': sea_enc,
                            'day_of_week_encoded': dow_enc,
                            'is_holiday': 0
                        }
                    
                    # Base prediction
                    base_features = pd.DataFrame([build_features(base_price)])[feature_cols]
                    base_demand = max(0, model.predict(base_features)[0])
                    
                    # New price prediction
                    new_price = base_price * (1 + price_change_pct / 100)
                    new_features = pd.DataFrame([build_features(new_price)])[feature_cols]
                    new_demand = max(0, model.predict(new_features)[0])
                    
                    # Calculate elasticity
                    pct_demand_change = ((new_demand - base_demand) / base_demand * 100) if base_demand > 0 else 0
                    elasticity = pct_demand_change / price_change_pct if price_change_pct != 0 else 0
                    
                    # Revenue comparison
                    base_revenue = base_demand * base_price
                    new_revenue = new_demand * new_price
                    revenue_change = ((new_revenue - base_revenue) / base_revenue * 100) if base_revenue > 0 else 0
                    
                    result = f"""
📊 PRICE ELASTICITY SIMULATION RESULTS
{'='*45}

📍 BASE SCENARIO:
   Price: ${base_price:,.2f}
   Predicted Demand: {base_demand:.0f} units
   Revenue: ${base_revenue:,.2f}

📍 NEW SCENARIO ({price_change_pct:+}% price change):
   Price: ${new_price:,.2f}
   Predicted Demand: {new_demand:.0f} units
   Revenue: ${new_revenue:,.2f}

📈 IMPACT ANALYSIS:
   Demand Change: {pct_demand_change:+.2f}%
   Revenue Change: {revenue_change:+.2f}%
   Price Elasticity: {abs(elasticity):.3f}

🎯 INTERPRETATION:
   {'INELASTIC (< 1): Price increase is profitable!' if abs(elasticity) < 1 else 'ELASTIC (> 1): Price sensitive - avoid increases'}
   
💡 RECOMMENDATION:
   {'✅ Safe to increase prices - demand is price insensitive' if abs(elasticity) < 0.5 else '⚠️ Moderate sensitivity - test carefully' if abs(elasticity) < 1 else '❌ High sensitivity - focus on volume not margins'}
"""
                    return result
                    
                except Exception as e:
                    return f"❌ Error: {str(e)}"
            
            pe_btn.click(
                fn=simulate_elasticity,
                inputs=[pe_category, pe_base_price, pe_cost_price, pe_competitor_price,
                        pe_discount, pe_stock, pe_rating, pe_change, pe_is_promoted],
                outputs=[pe_output]
            )
        
        # Tab 4: Model Info
        with gr.TabItem("ℹ️ Model Info"):
            gr.Markdown("""
            ## Model Details
            
            ### 🎯 XGBoost Regressor
            
            | Parameter | Value |
            |-----------|-------|
            | n_estimators | 200 |
            | max_depth | 6 |
            | learning_rate | 0.1 |
            | subsample | 0.8 |
            | colsample_bytree | 0.8 |
            | device | CUDA (GPU) |
            
            ### 📊 Performance Metrics
            
            | Metric | Train | Test |
            |--------|-------|------|
            | RMSE | 8.22 | 14.28 |
            | MAE | 5.31 | 7.02 |
            | R² Score | 0.9961 | 0.9868 |
            
            ### ✅ 5-Fold Cross-Validation (Model Stability)
            
            | Metric | Mean | Std Dev |
            |--------|------|---------|
            | R² Score | 0.9752 | **0.0019** |
            | RMSE | 20.66 | 0.58 |
            | MAE | 11.36 | 0.46 |
            
            **Status:** ✓ **ROBUST** — Std Dev (0.0019) < 0.01 indicates highly stable model
            
            ### 🔑 Top Demand Drivers
            1. **is_promoted** - Primary factor influencing sales
            2. **stock_available** - Secondary driver (r = 0.80)
            3. **rating** - Third most important predictor
            4. **price** - Weak predictor (r = 0.04), inelastic market
            
            ### 📈 Key Insights
            - **Seasonal Peak:** December (+21% vs July)
            - **Promotion Lift:** +8.4% overall, Sports +17.1%
            - **Price Elasticity:** Inelastic (0.06) - premium pricing opportunity
            - **Top Category:** Electronics (78.8% of revenue)
            
            ### 📉 Price Elasticity by Category
            
            | Category | Elasticity | Interpretation |
            |----------|------------|----------------|
            | Home | 0.090 | Very Inelastic |
            | Electronics | 0.065 | Very Inelastic |
            | Fashion | 0.059 | Very Inelastic |
            | Sports | 0.047 | Very Inelastic |
            | Beauty | 0.044 | Very Inelastic |
            
            *Note: Low elasticity indicates stock availability and promotions drive demand more than price.*
            """)

    gr.Markdown("""
    ---
    *Built with XGBoost & Gradio | Retail Demand Intelligence System*
    """)


# ============================================================
# LAUNCH APP
# ============================================================

if __name__ == "__main__":
    print("\n" + "=" * 50)
    print("🚀 Launching Gradio Interface...")
    print("=" * 50)
    demo.launch(share=False, server_name="0.0.0.0", server_port=7860)

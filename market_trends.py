# market_trends.py
import google.generativeai as genai
import streamlit as st
import os
from datetime import datetime
import json

class MarketTrendsAnalyzer:
    def __init__(self):
        """Initialize the Gemini AI client"""
        try:
            # Get API key from environment variable
            api_key = os.getenv('GEMINI_API_KEY')
            if not api_key:
                raise ValueError("GEMINI_API_KEY not found in environment variables")
            
            # Configure Gemini AI
            genai.configure(api_key=api_key)
            self.model = genai.GenerativeModel('gemini-pro')
            
        except Exception as e:
            st.error(f"❌ Error initializing Gemini AI: {str(e)}")
            self.model = None
    
    def get_market_trends(self, car_model, car_data=None):
        """Get market trends analysis for a specific car model"""
        if not self.model:
            return "❌ Gemini AI is not properly configured. Please check your API key."
        
        try:
            # Create a comprehensive prompt for market analysis
            current_year = datetime.now().year
            
            # Base prompt
            prompt = f"""
            As an automotive market analyst, provide a comprehensive market trends analysis for the {car_model}. 
            
            Please analyze the following aspects and provide insights in this exact format:
            
            📈 **MARKET POSITION & DEMAND**
            [Analyze current market position, demand trends, and popularity]
            
            💰 **PRICE TRENDS**
            [Discuss price movements, depreciation patterns, and value retention]
            
            🔄 **RESALE VALUE ANALYSIS**
            [Analyze resale value trends and factors affecting it]
            
            📊 **MARKET COMPARISON**
            [Compare with similar models in the same segment]
            
            🎯 **BUYING RECOMMENDATIONS**
            [Provide advice for potential buyers - best time to buy, what to look for]
            
            🔮 **FUTURE OUTLOOK**
            [Predict future trends and market expectations]
            
            ⚠️ **KEY FACTORS TO CONSIDER**
            [List important factors that affect this model's market value]
            """
            
            # Add car data if available
            if car_data:
                prompt += f"\n\nAdditional Context:\n"
                if isinstance(car_data, dict):
                    for key, value in car_data.items():
                        if value and str(value).lower() != 'n/a':
                            prompt += f"- {key}: {value}\n"
            
            prompt += f"\n\nCurrent Year: {current_year}\nLocation Context: Indian automotive market\n"
            prompt += "\nPlease provide practical, actionable insights based on current market conditions."
            
            # Generate response
            with st.spinner("🤖 AI is analyzing market trends..."):
                response = self.model.generate_content(prompt)
                return response.text
                
        except Exception as e:
            return f"❌ Error generating market trends: {str(e)}"
    
    def get_quick_insights(self, car_model):
        """Get quick market insights for sidebar or summary"""
        if not self.model:
            return {"error": "Gemini AI not configured"}
        
        try:
            prompt = f"""
            Provide 3 key market insights for {car_model} in exactly this JSON format:
            {{
                "trend": "Rising/Stable/Declining",
                "recommendation": "Buy/Hold/Sell",
                "key_point": "One sentence key insight"
            }}
            
            Respond only with valid JSON, no other text.
            """
            
            response = self.model.generate_content(prompt)
            try:
                return json.loads(response.text)
            except:
                return {
                    "trend": "Analysis Available",
                    "recommendation": "Check Details",
                    "key_point": "Detailed analysis available in market trends section"
                }
                
        except Exception as e:
            return {"error": str(e)}

def display_market_trends(car_model, car_data=None):
    """Display market trends analysis in Streamlit"""
    
    # Initialize analyzer
    analyzer = MarketTrendsAnalyzer()
    
    # Create tabs for different analyses
    tab1, tab2, tab3 = st.tabs(["📈 Market Analysis", "💡 Quick Insights", "📊 Data Summary"])
    
    with tab1:
        st.subheader(f"🚗 Market Trends Analysis: {car_model}")
        
        if st.button("🔍 Generate Market Analysis", type="primary", use_container_width=True):
            analysis = analyzer.get_market_trends(car_model, car_data)
            
            # Display analysis in a nice format
            st.markdown("""
            <div style="background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%); 
                        padding: 2rem; border-radius: 15px; margin: 1rem 0;">
            """, unsafe_allow_html=True)
            
            st.markdown(analysis)
            
            st.markdown("</div>", unsafe_allow_html=True)
            
            # Add disclaimer
            st.info("🤖 This analysis is generated by AI and should be used as a reference. Always consult with automotive experts for major purchase decisions.")
    
    with tab2:
        st.subheader("💡 Quick Market Insights")
        
        if st.button("⚡ Get Quick Insights", use_container_width=True):
            insights = analyzer.get_quick_insights(car_model)
            
            if "error" not in insights:
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.metric("📈 Market Trend", insights.get("trend", "N/A"))
                
                with col2:
                    st.metric("💰 Recommendation", insights.get("recommendation", "N/A"))
                
                with col3:
                    st.info(f"💡 **Key Insight**\n\n{insights.get('key_point', 'Analysis in progress...')}")
            else:
                st.error(f"Error: {insights['error']}")
    
    with tab3:
        st.subheader("📊 Current Data Summary")
        if car_data:
            # Display current car data in a nice format
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("**🚘 Vehicle Information**")
                for key, value in car_data.items():
                    if key in ['Model', 'Year', 'Fuel']:
                        st.write(f"• **{key}:** {value}")
            
            with col2:
                st.markdown("**⚙️ Technical Details**")
                for key, value in car_data.items():
                    if key in ['Transmission', 'Drive', 'KM_Driven']:
                        st.write(f"• **{key}:** {value}")
            
            # Price section
            price_keys = ['Average_Price_(₹)', 'Average Price (₹)', 'price', 'Price']
            price_value = None
            for key in price_keys:
                if key in car_data and car_data[key]:
                    price_value = car_data[key]
                    break
            
            if price_value:
                st.markdown(f"""
                <div style="background: linear-gradient(90deg, #56CCF2, #2F80ED); 
                           color: white; padding: 1rem; border-radius: 10px; 
                           text-align: center; margin: 1rem 0;">
                    <h3>💰 Current Market Price: ₹{price_value}</h3>
                </div>
                """, unsafe_allow_html=True)
        else:
            st.info("No specific car data available for detailed analysis.")

# Usage in your main app
def integrate_market_trends_button(car_model, car_data=None):
    """Function to call from your main app when market trends button is clicked"""
    
    # Store in session state to show market trends
    if 'show_market_trends' not in st.session_state:
        st.session_state.show_market_trends = False
    
    if st.button("📊 View Market Trends", use_container_width=True):
        st.session_state.show_market_trends = not st.session_state.show_market_trends
    
    # Display market trends if requested
    if st.session_state.show_market_trends:
        st.markdown("---")
        display_market_trends(car_model, car_data)
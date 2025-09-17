import streamlit as st
import pandas as pd
import plotly.express as px
from datetime import datetime
import re
import json
import google.generativeai as genai
from PIL import Image
import pytesseract
import numpy as np
#API KEY AIzaSyBFGyncp0__h4hUtsdWkDlrVbVOfEjh7-s
st.set_page_config(
    page_title="SmartCarbon AI",
    page_icon="🌱",
    layout="wide",
    initial_sidebar_state="expanded"
)
st.markdown("""
<style>
    .main-header {
        background: linear-gradient(90deg, #2E8B57, #32CD32);
        padding: 1rem;
        border-radius: 10px;
        color: white;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background: #f0f2f6;
        padding: 1rem;
        border-radius: 10px;
        border-left: 4px solid #2E8B57;
    }
    .insight-box {
        background: #000000;
        padding: 1rem;
        border-radius: 10px;
        border: 1px solid #2E8B57;
        margin: 1rem 0;
        color: #ffffff; /* Force text to black */
    }
</style>
""", unsafe_allow_html=True)


# Initialize session state
if 'processed_documents' not in st.session_state:
    st.session_state.processed_documents = []
if 'total_emissions' not in st.session_state:
    st.session_state.total_emissions = 0
if 'gemini_api_key' not in st.session_state:
    st.session_state.gemini_api_key = ""

# Hardcoded emission factors (kg CO2e per unit)
EMISSION_FACTORS = {
    'electricity': {'factor': 0.444, 'unit': 'kWh', 'description': 'Grid electricity (US average)'},
    'natural_gas': {'factor': 0.185, 'unit': 'kWh', 'description': 'Natural gas consumption'},
    'fuel': {'factor': 2.31, 'unit': 'liter', 'description': 'Gasoline/Diesel fuel'},
    'water': {'factor': 0.344, 'unit': 'm³', 'description': 'Water consumption'},
    'waste': {'factor': 0.42, 'unit': 'kg', 'description': 'General waste disposal'},
    'paper': {'factor': 1.84, 'unit': 'kg', 'description': 'Paper products'},
    'transport': {'factor': 0.12, 'unit': 'km', 'description': 'Vehicle transport'},
    'office_supplies': {'factor': 2.1, 'unit': '$', 'description': 'Office supplies (spend-based)'}
}

def setup_gemini_api(api_key):
    """Setup Gemini API with provided key"""
    try:
        genai.configure(api_key=api_key)
        return True
    except Exception as e:
        st.error(f"Error setting up Gemini API: {e}")
        return False

def extract_text_from_image(image):
    """Extract text from uploaded image using OCR"""
    try:
        # Convert to PIL Image if needed
        if hasattr(image, 'read'):
            image = Image.open(image)
        
        # Perform OCR
        text = pytesseract.image_to_string(image)
        return text
    except Exception as e:
        st.error(f"OCR Error: {e}")
        return ""

def extract_data_from_text(text, doc_type):
    """Extract relevant data from text based on document type"""
    data = {}
    
    # Common patterns for different data types
    amount_patterns = {
        'electricity': r'(\d+(?:\.\d+)?)\s*(?:kwh|kWh|KWH)',
        'natural_gas': r'(\d+(?:\.\d+)?)\s*(?:kwh|kWh|therms?)',
        'fuel': r'(\d+(?:\.\d+)?)\s*(?:liters?|gallons?|L|gal)',
        'water': r'(\d+(?:\.\d+)?)\s*(?:m³|cubic|gallons?)',
        'transport': r'(\d+(?:\.\d+)?)\s*(?:km|miles?|mi)'
    }
    
    cost_pattern = r'\$\s*(\d+(?:\.\d+)?)'
    date_pattern = r'(\d{1,2}[\/\-]\d{1,2}[\/\-]\d{2,4})'
    
    # Extract amount based on document type
    if doc_type in amount_patterns:
        amount_match = re.search(amount_patterns[doc_type], text, re.IGNORECASE)
        if amount_match:
            data['amount'] = float(amount_match.group(1))
    
    # Extract cost
    cost_match = re.search(cost_pattern, text)
    if cost_match:
        data['cost'] = float(cost_match.group(1))
    
    # Extract date
    date_match = re.search(date_pattern, text)
    if date_match:
        data['date'] = date_match.group(1)
    else:
        data['date'] = datetime.now().strftime('%m/%d/%Y')
    
    # If no amount found, estimate from cost for spend-based calculation
    if 'amount' not in data and 'cost' in data and doc_type == 'office_supplies':
        data['amount'] = data['cost']  # Dollar amount for spend-based
    
    return data

def calculate_emissions(amount, doc_type):
    """Calculate CO2 emissions based on amount and document type"""
    if doc_type in EMISSION_FACTORS:
        factor = EMISSION_FACTORS[doc_type]['factor']
        emissions = amount * factor
        return emissions
    return 0

def get_ai_insights(emissions_data, api_key):
    """Get AI insights using Gemini API"""
    if not api_key:
        return "Please configure Gemini API key in the sidebar to get AI insights."
    
    try:
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel('gemini-2.5-flash')
        
        # Prepare data summary for AI
        total_emissions = sum([doc['emissions'] for doc in emissions_data])
        doc_types = [doc['type'] for doc in emissions_data]
        
        prompt = f"""
        Analyze this carbon emissions data and provide actionable insights:
        
        Total Emissions: {total_emissions:.2f} kg CO2e
        Document Types: {', '.join(set(doc_types))}
        Number of Documents: {len(emissions_data)}
        
        Recent entries:
        {json.dumps(emissions_data[-3:], indent=2)}
        
        Please provide:
        1. Key emission hotspots and patterns
        2. Specific reduction recommendations
        3. Comparison to industry benchmarks
        4. Priority actions for next month
        
        Keep the response concise and actionable.
        """
        
        response = model.generate_content(prompt)
        return response.text
    
    except Exception as e:
        return f"Error generating AI insights: {str(e)}"

def main():
    # Header
    st.markdown("""
    <div class="main-header">
        <h1>🌱 SmartCarbon AI</h1>
        <p>Intelligent Carbon Accounting System</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Sidebar for API configuration
    with st.sidebar:
        st.header("⚙️ Configuration")
        api_key = st.text_input(
            "Gemini API Key",
            type="password",
            value=st.session_state.gemini_api_key,
            help="Enter your Google Gemini API key for AI insights"
        )
        if api_key:
            st.session_state.gemini_api_key = api_key
        
        st.markdown("---")
        st.header("📊 Quick Stats")
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Documents", len(st.session_state.processed_documents))
        with col2:
            st.metric("Total CO2e", f"{st.session_state.total_emissions:.1f} kg")
    
    # Main tabs
    tab1, tab2, tab3, tab4 = st.tabs(["📄 Document Upload", "📊 Dashboard", "🤖 AI Insights", "📈 Compliance"])
    
    with tab1:
        st.header("Document Processing")
        
        # Document type selection
        doc_type = st.selectbox(
            "Document Type",
            options=list(EMISSION_FACTORS.keys()),
            format_func=lambda x: f"{x.replace('_', ' ').title()} - {EMISSION_FACTORS[x]['description']}"
        )
        
        # File upload options
        upload_method = st.radio("Upload Method", ["Upload File", "Manual Entry"])
        
        if upload_method == "Upload File":
            uploaded_file = st.file_uploader(
                "Upload Document",
                type=['pdf', 'png', 'jpg', 'jpeg'],
                help="Upload invoices, bills, or receipts"
            )
            
            if uploaded_file is not None:
                if uploaded_file.type.startswith('image'):
                    # Display image
                    image = Image.open(uploaded_file)
                    st.image(image, caption="Uploaded Document", use_column_width=True)
                    
                    # Extract text using OCR
                    with st.spinner("Extracting text from image..."):
                        extracted_text = extract_text_from_image(uploaded_file)
                    
                    if extracted_text.strip():
                        st.text_area("Extracted Text", extracted_text, height=150)
                        
                        # Extract data
                        extracted_data = extract_data_from_text(extracted_text, doc_type)
                        
                        if extracted_data:
                            st.success("Data extracted successfully!")
                            
                            # Allow manual correction
                            col1, col2, col3 = st.columns(3)
                            with col1:
                                amount = st.number_input(
                                    f"Amount ({EMISSION_FACTORS[doc_type]['unit']})",
                                    value=extracted_data.get('amount', 0.0)
                                )
                            with col2:
                                cost = st.number_input(
                                    "Cost ($)",
                                    value=extracted_data.get('cost', 0.0)
                                )
                            with col3:
                                date = st.date_input(
                                    "Date",
                                    value=datetime.strptime(extracted_data.get('date', datetime.now().strftime('%m/%d/%Y')), '%m/%d/%Y').date()
                                )
                            
                            if st.button("Process Document"):
                                emissions = calculate_emissions(amount, doc_type)
                                
                                # Store processed document
                                doc_data = {
                                    'type': doc_type,
                                    'amount': amount,
                                    'cost': cost,
                                    'date': date.strftime('%Y-%m-%d'),
                                    'emissions': emissions,
                                    'unit': EMISSION_FACTORS[doc_type]['unit']
                                }
                                
                                st.session_state.processed_documents.append(doc_data)
                                st.session_state.total_emissions += emissions
                                
                                st.success(f"✅ Document processed! Emissions: {emissions:.2f} kg CO2e")
                        else:
                            st.warning("Could not extract relevant data. Please use manual entry.")
                    else:
                        st.warning("No text could be extracted from the image.")
        
        else:  # Manual Entry
            col1, col2, col3 = st.columns(3)
            with col1:
                amount = st.number_input(
                    f"Amount ({EMISSION_FACTORS[doc_type]['unit']})",
                    min_value=0.0,
                    value=0.0
                )
            with col2:
                cost = st.number_input("Cost ($)", min_value=0.0, value=0.0)
            with col3:
                date = st.date_input("Date", value=datetime.now().date())
            
            if st.button("Add Entry") and amount > 0:
                emissions = calculate_emissions(amount, doc_type)
                
                doc_data = {
                    'type': doc_type,
                    'amount': amount,
                    'cost': cost,
                    'date': date.strftime('%Y-%m-%d'),
                    'emissions': emissions,
                    'unit': EMISSION_FACTORS[doc_type]['unit']
                }
                
                st.session_state.processed_documents.append(doc_data)
                st.session_state.total_emissions += emissions
                
                st.success(f"✅ Entry added! Emissions: {emissions:.2f} kg CO2e")
    
    with tab2:
        st.header("Carbon Emissions Dashboard")
        
        if st.session_state.processed_documents:
            df = pd.DataFrame(st.session_state.processed_documents)
            df['date'] = pd.to_datetime(df['date'])
            
            # Summary metrics
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Total Emissions", f"{st.session_state.total_emissions:.2f} kg CO2e")
            with col2:
                avg_monthly = st.session_state.total_emissions / max(1, len(df.groupby(df['date'].dt.to_period('M'))))
                st.metric("Avg Monthly", f"{avg_monthly:.2f} kg CO2e")
            with col3:
                highest_day = df.groupby('date')['emissions'].sum().max()
                st.metric("Highest Day", f"{highest_day:.2f} kg CO2e")
            with col4:
                total_cost = df['cost'].sum()
                st.metric("Total Cost", f"${total_cost:.2f}")
            
            # Emissions by category
            category_emissions = df.groupby('type')['emissions'].sum().reset_index()
            fig_pie = px.pie(
                category_emissions,
                values='emissions',
                names='type',
                title="Emissions by Category"
            )
            st.plotly_chart(fig_pie, use_container_width=True)
            
            # Timeline chart
            daily_emissions = df.groupby('date')['emissions'].sum().reset_index()
            fig_line = px.line(
                daily_emissions,
                x='date',
                y='emissions',
                title="Daily Emissions Timeline",
                markers=True
            )
            st.plotly_chart(fig_line, use_container_width=True)
            
            # Recent documents table
            st.subheader("Recent Documents")
            st.dataframe(
                df.sort_values('date', ascending=False).head(10),
                use_container_width=True
            )
            
        else:
            st.info("No documents processed yet. Upload documents in the Document Upload tab.")
    
    with tab3:
        st.header("🤖 AI-Powered Insights")
        
        if st.session_state.processed_documents:
            if st.button("Generate AI Insights", type="primary"):
                with st.spinner("Analyzing your carbon data..."):
                    insights = get_ai_insights(st.session_state.processed_documents, st.session_state.gemini_api_key)
                
                st.markdown(f"""
                <div class="insight-box">
                <h3>🔍 AI Analysis Results</h3>
                {insights}
                </div>
                """, unsafe_allow_html=True)
            
            # Quick recommendations based on data
            st.subheader("Quick Recommendations")
            if st.session_state.processed_documents:
                df = pd.DataFrame(st.session_state.processed_documents)
                top_category = df.groupby('type')['emissions'].sum().idxmax()
                top_emissions = df.groupby('type')['emissions'].sum().max()
                
                recommendations = {
                    'electricity': "💡 Switch to renewable energy sources or improve energy efficiency",
                    'fuel': "🚗 Consider electric vehicles or optimize routes to reduce fuel consumption",
                    'natural_gas': "🔥 Improve building insulation or switch to heat pumps",
                    'office_supplies': "📋 Choose suppliers with lower carbon footprints or reduce paper usage",
                    'transport': "🚌 Promote public transport or remote work to reduce travel emissions"
                }
                
                st.info(f"🎯 **Top Priority**: {top_category.replace('_', ' ').title()} ({top_emissions:.1f} kg CO2e)")
                st.write(recommendations.get(top_category, "Focus on this category for maximum impact"))
        
        else:
            st.info("Process some documents first to get AI insights!")
    
    with tab4:
        st.header("📈 Compliance Dashboard")
        
        # Compliance scores (mock data based on completeness)
        if st.session_state.processed_documents:
            df = pd.DataFrame(st.session_state.processed_documents)
            unique_types = df['type'].nunique()
            total_docs = len(df)
            
            # Mock compliance scores
            ghg_score = min(100, (unique_types / len(EMISSION_FACTORS)) * 80 + (total_docs / 10) * 20)
            iso_score = min(100, ghg_score * 0.9)
            cdp_score = min(100, ghg_score * 0.7 if unique_types > 3 else 45)
            tcfd_score = min(100, ghg_score * 0.6 if total_docs > 5 else 30)
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.metric("GHG Protocol", f"{ghg_score:.0f}/100", "✅ Compliant" if ghg_score >= 70 else "⚠️ Needs Work")
                st.metric("ISO 14064", f"{iso_score:.0f}/100", "✅ Compliant" if iso_score >= 70 else "⚠️ Needs Work")
            
            with col2:
                st.metric("CDP", f"{cdp_score:.0f}/100", "✅ Compliant" if cdp_score >= 70 else "❌ Missing Scope 3")
                st.metric("TCFD", f"{tcfd_score:.0f}/100", "✅ Compliant" if tcfd_score >= 70 else "❌ Climate Risk Needed")
            
            # Compliance chart
            compliance_data = {
                'Framework': ['GHG Protocol', 'ISO 14064', 'CDP', 'TCFD'],
                'Score': [ghg_score, iso_score, cdp_score, tcfd_score]
            }
            
            fig_bar = px.bar(
                compliance_data,
                x='Framework',
                y='Score',
                title="Compliance Framework Scores",
                color='Score',
                color_continuous_scale='RdYlGn'
            )
            fig_bar.add_hline(y=70, line_dash="dash", line_color="red", annotation_text="Compliance Threshold")
            st.plotly_chart(fig_bar, use_container_width=True)
            
            # Action items
            st.subheader("Priority Actions")
            actions = []
            if cdp_score < 70:
                actions.append("📋 Implement comprehensive Scope 3 supply chain tracking")
            if tcfd_score < 70:
                actions.append("🌡️ Prepare climate risk assessment and scenario analysis")
            if ghg_score < 70:
                actions.append("📊 Increase data collection across all emission categories")
            
            if actions:
                for action in actions:
                    st.write(f"• {action}")
            else:
                st.success("🎉 Great job! You're meeting all major compliance requirements.")
        
        else:
            st.info("Upload documents to see compliance status.")

if __name__ == "__main__":
    main()
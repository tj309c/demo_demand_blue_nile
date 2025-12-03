import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import networkx as nx
from datetime import datetime, timedelta
from dateutil.relativedelta import relativedelta

# =============================================================================
# INTEGRATED BUSINESS PLANNING - PAGE CONFIGURATION
# =============================================================================
st.set_page_config(
    page_title="Blue Nile | Integrated Business Planning",
    layout="wide",
    page_icon="💎",
    initial_sidebar_state="expanded"
)

# =============================================================================
# DARK MATTER DESIGN SYSTEM
# =============================================================================
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&family=Roboto:wght@300;400;500;700&display=swap');

.stApp {
    background-color: #0b0c10 !important;
    color: #c5c6c7 !important;
    font-family: 'Inter', 'Roboto', sans-serif !important;
}

.block-container {
    padding-top: 1rem !important;
    padding-bottom: 0rem !important;
    padding-left: 1.5rem !important;
    padding-right: 1.5rem !important;
    max-width: 100% !important;
}

[data-testid="stSidebar"] {
    background-color: #1f2833 !important;
    border-right: 2px solid #45a29e !important;
}

[data-testid="stSidebar"] * {
    color: #c5c6c7 !important;
}

h1, h2, h3, h4, h5, h6 {
    color: #66fcf1 !important;
    font-weight: 600 !important;
    letter-spacing: 0.5px !important;
}

.stMarkdown p {
    color: #c5c6c7 !important;
}

[data-testid="stMetricValue"] {
    color: #66fcf1 !important;
    font-size: 32px !important;
    font-weight: 700 !important;
}

[data-testid="stMetricLabel"] {
    color: #c5c6c7 !important;
    text-transform: uppercase !important;
    letter-spacing: 1px !important;
    font-size: 11px !important;
}

.stButton > button {
    background: linear-gradient(135deg, #45a29e 0%, #66fcf1 100%) !important;
    color: #0b0c10 !important;
    border: none !important;
    font-weight: 600 !important;
    text-transform: uppercase !important;
    padding: 10px 24px !important;
    border-radius: 6px !important;
}

.stButton > button:hover {
    box-shadow: 0 0 20px rgba(102, 252, 241, 0.5) !important;
    transform: translateY(-2px) !important;
}

.stTabs [data-baseweb="tab"] {
    background-color: #1f2833 !important;
    color: #c5c6c7 !important;
    border: 1px solid #45a29e !important;
    border-radius: 8px 8px 0 0 !important;
    padding: 12px 24px !important;
}

.stTabs [aria-selected="true"] {
    background: linear-gradient(135deg, #45a29e 0%, #66fcf1 100%) !important;
    color: #0b0c10 !important;
    font-weight: 700 !important;
}

[data-testid="stSidebar"] .stSelectbox label, [data-testid="stSidebar"] .stRadio label {
    color: #66fcf1 !important;
    font-weight: 600 !important;
    font-size: 12px !important;
    text-transform: uppercase !important;
}

#MainMenu {visibility: hidden;}
footer {visibility: hidden;}
header {visibility: hidden;}

.stChatMessage {
    background-color: rgba(31, 40, 51, 0.6) !important;
    border: 1px solid #45a29e !important;
    border-radius: 8px !important;
}

::-webkit-scrollbar {width: 10px; height: 10px;}
::-webkit-scrollbar-track {background: #1f2833;}
::-webkit-scrollbar-thumb {background: #45a29e; border-radius: 5px;}
::-webkit-scrollbar-thumb:hover {background: #66fcf1;}
</style>
""", unsafe_allow_html=True)

# =============================================================================
# DATA STRUCTURES
# =============================================================================
DIAMOND_SUPPLIERS = {
    "Kiran Gems (Surat)": {"location": {"city": "Surat", "lat": 21.1702, "lon": 72.8311}, "lead_time_days": 2.0, "reliability_score": 9.1, "risk_score": 2, "inventory_value": 189000000},
    "Antwerp Diamond Bank": {"location": {"city": "Antwerp", "lat": 51.2194, "lon": 4.4025}, "lead_time_days": 3.0, "reliability_score": 9.8, "risk_score": 1, "inventory_value": 122400000},
    "Tel Aviv Diamond Exchange": {"location": {"city": "Tel Aviv", "lat": 32.0853, "lon": 34.7818}, "lead_time_days": 3.5, "reliability_score": 9.6, "risk_score": 3, "inventory_value": 84000000},
    "Mumbai Diamond Exchange": {"location": {"city": "Mumbai", "lat": 19.0760, "lon": 72.8777}, "lead_time_days": 2.5, "reliability_score": 8.9, "risk_score": 2, "inventory_value": 153600000},
}

SHOWROOM_LOCATIONS = [
    {"name": "Roosevelt Field", "lat": 40.7368, "lon": -73.6107, "revenue": 11016000},
    {"name": "Garden State Plaza", "lat": 40.9176, "lon": -74.0732, "revenue": 12512000},
    {"name": "Newport Beach", "lat": 33.6156, "lon": -117.8739, "revenue": 15088000},
    {"name": "Park Meadows", "lat": 39.5617, "lon": -104.8763, "revenue": 7117500},
]

# =============================================================================
# TIME SERIES FORECAST GENERATOR (18 MONTHS)
# =============================================================================
def generate_time_series_forecast():
    np.random.seed(42)
    base_date = datetime(2024, 1, 1)
    months = [(base_date + relativedelta(months=i)).strftime('%b-%y') for i in range(18)]
    
    # Historical actuals (12 months)
    base_demand = 12000
    actuals = []
    for i in range(12):
        month_idx = (i % 12) + 1
        # Engagement season (Nov-Feb): 40% higher
        if month_idx in [11, 12, 1, 2]:
            seasonal_factor = 1.4
        # Wedding season (May-Oct): 15% higher
        elif month_idx in [5, 6, 7, 8, 9, 10]:
            seasonal_factor = 1.15
        else:
            seasonal_factor = 0.9
        
        noise = np.random.normal(1.0, 0.08)
        actuals.append(int(base_demand * seasonal_factor * noise))
    
    # Forecast (6 months) - 3 scenarios
    forecast_months = 6
    statistical_forecast = []
    ml_forecast = []
    consensus_forecast = []
    confidence_upper = []
    confidence_lower = []
    
    for i in range(12, 18):
        month_idx = (i % 12) + 1
        if month_idx in [11, 12, 1, 2]:
            seasonal_factor = 1.4
        elif month_idx in [5, 6, 7, 8, 9, 10]:
            seasonal_factor = 1.15
        else:
            seasonal_factor = 0.9
        
        base_forecast = base_demand * seasonal_factor * 1.06  # 6% YoY growth
        
        statistical_forecast.append(int(base_forecast * 0.98))
        ml_forecast.append(int(base_forecast * 1.04))
        consensus_forecast.append(int(base_forecast * 1.01))
        confidence_upper.append(int(base_forecast * 1.15))
        confidence_lower.append(int(base_forecast * 0.85))
    
    return {
        'months': months,
        'actuals': actuals + [None]*6,
        'statistical': [None]*12 + statistical_forecast,
        'ml': [None]*12 + ml_forecast,
        'consensus': [None]*12 + consensus_forecast,
        'upper': [None]*12 + confidence_upper,
        'lower': [None]*12 + confidence_lower
    }

def calculate_forecast_kpis():
    mape = 8.2  # Mean Absolute Percentage Error
    bias = -1.5  # Negative = under-forecasting
    signal_strength = 87
    consensus_alignment = 94
    return {'mape': mape, 'bias': bias, 'signal': signal_strength, 'consensus': consensus_alignment}

def calculate_inventory_health_kpis():
    """Core inventory planning metrics - aligned to Manager, Inventory Planning JD"""
    return {
        'days_of_supply': 47,
        'days_of_supply_target': 45,
        'inventory_turns': 4.2,
        'inventory_turns_benchmark': 4.0,
        'service_level': 96.8,
        'service_level_target': 98.0,
        'working_capital': 549000000,  # $549M
        'working_capital_delta': -12000000,  # -$12M vs last month
        'shrinkage_rate': 0.31,
        'shrinkage_target': 0.50,
        'stock_availability': 94.2,
        'fill_rate': 97.1
    }

def generate_asset_jit_data():
    """Asset vs JIT SKU analysis - key concept from JD"""
    np.random.seed(42)

    # Generate SKU data with velocity and value characteristics
    sku_data = []
    shapes = ['Round', 'Princess', 'Oval', 'Cushion', 'Emerald', 'Pear', 'Marquise', 'Radiant']
    carats = ['0.5ct', '0.75ct', '1.0ct', '1.25ct', '1.5ct', '2.0ct', '2.5ct', '3.0ct']

    for i in range(150):
        shape = shapes[i % len(shapes)]
        carat = carats[i % len(carats)]

        # Higher carat = lower velocity, higher value
        carat_multiplier = 1 + (i % len(carats)) * 0.5
        velocity = max(5, int(np.random.exponential(50) / carat_multiplier))
        unit_value = int(2000 * carat_multiplier * np.random.uniform(0.8, 1.2))
        holding_cost = unit_value * 0.02  # 2% monthly holding cost
        lead_time = np.random.choice([2, 3, 5, 7, 14], p=[0.1, 0.3, 0.3, 0.2, 0.1])

        sku_data.append({
            'sku_id': f'BN-{shape[:3].upper()}-{carat.replace("ct","")}-{i:03d}',
            'shape': shape,
            'carat': carat,
            'monthly_velocity': velocity,
            'unit_value': unit_value,
            'holding_cost_monthly': holding_cost,
            'lead_time_days': lead_time,
            'current_strategy': 'Asset' if velocity > 30 else 'JIT'
        })

    return pd.DataFrame(sku_data)

def generate_demand_supply_flow():
    """Demand to Supply translation data - exact JD phrase"""
    return {
        'demand_forecast': 14200,
        'safety_stock_buffer': 2100,
        'net_requirements': 16300,
        'vendor_allocation': {
            'Kiran Gems (Surat)': {'allocation_pct': 45, 'units': 7335, 'capacity_util': 78},
            'Antwerp Diamond Bank': {'allocation_pct': 30, 'units': 4890, 'capacity_util': 65},
            'Mumbai Diamond Exchange': {'allocation_pct': 15, 'units': 2445, 'capacity_util': 82},
            'Tel Aviv Diamond Exchange': {'allocation_pct': 10, 'units': 1630, 'capacity_util': 54}
        },
        'po_summary': {
            'total_pos': 47,
            'this_week': 12,
            'pending_approval': 3,
            'in_transit': 28,
            'value_in_transit': 23400000
        }
    }

def generate_demand_drivers():
    np.random.seed(42)
    weeks = list(range(1, 13))
    promotion_impact = [5, 8, 15, 22, 12, 35, 28, 18, 25, 20, 15, 10]
    social_media_buzz = [45, 52, 58, 71, 85, 92, 78, 65, 72, 68, 55, 48]
    holiday_effect = [10, 8, 5, 3, 2, 5, 8, 12, 18, 25, 40, 55]
    return pd.DataFrame({'Week': weeks, 'Promotions': promotion_impact, 'Social': social_media_buzz, 'Holidays': holiday_effect})

def generate_attribute_heatmap():
    carat_sizes = ['0.5ct', '1.0ct', '1.5ct', '2.0ct+']
    price_tiers = ['Entry<$3K', 'Mid$3-7K', 'High$7-15K', 'Ultra>$15K']
    demand_matrix = np.array([[45, 72, 85, 62], [68, 91, 78, 55], [82, 94, 71, 48], [55, 68, 89, 92]])
    return carat_sizes, price_tiers, demand_matrix

def create_network_graph():
    G = nx.Graph()
    for supplier_name, supplier_data in DIAMOND_SUPPLIERS.items():
        G.add_node(supplier_name, type='supplier', inventory_value=supplier_data['inventory_value'],
                   lat=supplier_data['location']['lat'], lon=supplier_data['location']['lon'])
    G.add_node('NYC Hub', type='hub', lat=40.7580, lon=-73.9855)
    for showroom in SHOWROOM_LOCATIONS:
        G.add_node(showroom['name'], type='showroom', lat=showroom['lat'], lon=showroom['lon'], revenue=showroom['revenue'])
    for supplier_name in DIAMOND_SUPPLIERS.keys():
        G.add_edge(supplier_name, 'NYC Hub', weight=2.5)
    for showroom in SHOWROOM_LOCATIONS:
        G.add_edge('NYC Hub', showroom['name'], weight=2.3)
    
    pos_3d = {}
    for node in G.nodes():
        node_data = G.nodes[node]
        if 'lat' in node_data and 'lon' in node_data:
            z = 1.0 if node_data.get('type') == 'supplier' else (0.5 if node_data.get('type') == 'hub' else 0.0)
            pos_3d[node] = (node_data['lon'], node_data['lat'], z)
    
    edge_traces = []
    for edge in G.edges():
        x0, y0, z0 = pos_3d[edge[0]]
        x1, y1, z1 = pos_3d[edge[1]]
        edge_traces.append(go.Scatter3d(x=[x0, x1, None], y=[y0, y1, None], z=[z0, z1, None],
                                       mode='lines', line=dict(color='#45a29e', width=2), hoverinfo='none', showlegend=False))
    
    node_traces = []
    for node in [n for n in G.nodes() if G.nodes[n].get('type') == 'supplier']:
        x, y, z = pos_3d[node]
        node_traces.append(go.Scatter3d(x=[x], y=[y], z=[z], mode='markers+text',
                                       marker=dict(size=12, color='#00ff88', line=dict(color='white', width=2)),
                                       text=[node.split()[0]], textposition='top center', textfont=dict(size=9, color='white'),
                                       hovertemplate=f"<b>{node}</b><extra></extra>", showlegend=False))
    
    x, y, z = pos_3d['NYC Hub']
    node_traces.append(go.Scatter3d(x=[x], y=[y], z=[z], mode='markers+text',
                                   marker=dict(size=25, color='#66fcf1', symbol='diamond', line=dict(color='white', width=3)),
                                   text=['NYC HUB'], textposition='top center', textfont=dict(size=12, color='#66fcf1'),
                                   hovertemplate="<b>NYC Hub</b><extra></extra>", showlegend=False))
    
    for node in [n for n in G.nodes() if G.nodes[n].get('type') == 'showroom']:
        x, y, z = pos_3d[node]
        node_traces.append(go.Scatter3d(x=[x], y=[y], z=[z], mode='markers',
                                       marker=dict(size=10, color='#c5c6c7', symbol='square'),
                                       hovertemplate=f"<b>{node}</b><extra></extra>", showlegend=False))
    
    fig = go.Figure(data=edge_traces + node_traces)
    fig.update_layout(scene=dict(xaxis=dict(showbackground=False, showticklabels=False, showgrid=False, title=''),
                                 yaxis=dict(showbackground=False, showticklabels=False, showgrid=False, title=''),
                                 zaxis=dict(showbackground=False, showticklabels=False, showgrid=False, title=''),
                                 bgcolor='#0b0c10'),
                      paper_bgcolor='#0b0c10', plot_bgcolor='#0b0c10', showlegend=False,
                      height=550, margin=dict(l=0, r=0, t=0, b=0), hovermode='closest')
    return fig

# =============================================================================
# SIDEBAR
# =============================================================================
with st.sidebar:
    st.markdown("# 🤖 Blue Nile IBP")
    st.markdown("*Integrated Business Planning*")
    st.markdown("---")

    # AI Recommendations Panel (replaces chat)
    st.markdown("### 🧠 AI Recommendations")
    st.markdown("""
    <div style="background: #0b0c10; padding: 12px; border-radius: 8px; border-left: 3px solid #ff4444; margin-bottom: 10px;">
        <div style="color: #ff4444; font-size: 10px; font-weight: 600;">🚨 HIGH PRIORITY</div>
        <div style="color: #c5c6c7; font-size: 12px; margin-top: 5px;">Increase 1.5ct inventory by 25% - $2.3M revenue at risk</div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div style="background: #0b0c10; padding: 12px; border-radius: 8px; border-left: 3px solid #ffd700; margin-bottom: 10px;">
        <div style="color: #ffd700; font-size: 10px; font-weight: 600;">⚠️ REBALANCE</div>
        <div style="color: #c5c6c7; font-size: 12px; margin-top: 5px;">Move 45 units from Newport → Roosevelt (demand mismatch)</div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div style="background: #0b0c10; padding: 12px; border-radius: 8px; border-left: 3px solid #45a29e; margin-bottom: 10px;">
        <div style="color: #45a29e; font-size: 10px; font-weight: 600;">💡 OPTIMIZE</div>
        <div style="color: #c5c6c7; font-size: 12px; margin-top: 5px;">Shift 23 SKUs Asset→JIT to free $2.1M working capital</div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div style="background: #0b0c10; padding: 12px; border-radius: 8px; border-left: 3px solid #00ff88; margin-bottom: 10px;">
        <div style="color: #00ff88; font-size: 10px; font-weight: 600;">✅ ON TRACK</div>
        <div style="color: #c5c6c7; font-size: 12px; margin-top: 5px;">Forecast accuracy at 8.2% MAPE (target: <10%)</div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("### 🎛️ Forecast Scenario")
    forecast_scenario = st.radio("Select Model:", ["Statistical", "ML Model", "Consensus"], index=2)

    st.markdown("---")
    st.markdown("### 📊 Quick Stats")
    st.metric("Total SKUs", "2,847", delta="+124 this month")
    st.metric("Open POs", "47", delta="3 pending approval")
    st.metric("Forecast Cycle", "Week 49", delta="Next: Dec 9")

# =============================================================================
# MAIN APP
# =============================================================================
st.markdown("""
<div style="background: linear-gradient(135deg, #1f2833 0%, #0b0c10 100%); padding: 25px; border-radius: 12px; border: 2px solid #45a29e; margin-bottom: 20px;">
    <h1 style="margin: 0; font-size: 36px; color: #66fcf1;">💎 Blue Nile Integrated Business Planning</h1>
    <p style="margin: 5px 0 0 0; font-size: 14px; color: #c5c6c7;">Demand Planning | Inventory Strategy | Supply Chain Operations</p>
</div>
""", unsafe_allow_html=True)

# DEMAND PLANNING KPIs
kpis = calculate_forecast_kpis()
kpi_cols = st.columns(4)
with kpi_cols[0]:
    st.metric("Forecast Accuracy (MAPE)", f"{kpis['mape']:.1f}%", delta="-2.1pp vs LY", delta_color="inverse")
with kpi_cols[1]:
    st.metric("Forecast Bias", f"{kpis['bias']:.1f}%", delta="Under-forecast", delta_color="off")
with kpi_cols[2]:
    st.metric("Demand Signal Strength", f"{kpis['signal']}/100", delta="Strong", delta_color="normal")
with kpi_cols[3]:
    st.metric("Consensus Alignment", f"{kpis['consensus']}%", delta="High Agreement", delta_color="normal")

# INVENTORY HEALTH KPIs (Key metrics from Manager, Inventory Planning JD)
st.markdown("#### Inventory Health")
inv_kpis = calculate_inventory_health_kpis()
inv_cols = st.columns(5)
with inv_cols[0]:
    dos_delta = inv_kpis['days_of_supply'] - inv_kpis['days_of_supply_target']
    st.metric("Days of Supply", f"{inv_kpis['days_of_supply']} days",
              delta=f"+{dos_delta} vs target" if dos_delta > 0 else f"{dos_delta} vs target",
              delta_color="inverse" if dos_delta > 0 else "normal")
with inv_cols[1]:
    turns_delta = inv_kpis['inventory_turns'] - inv_kpis['inventory_turns_benchmark']
    st.metric("Inventory Turns", f"{inv_kpis['inventory_turns']}x",
              delta=f"+{turns_delta:.1f} vs benchmark", delta_color="normal")
with inv_cols[2]:
    svc_gap = inv_kpis['service_level'] - inv_kpis['service_level_target']
    st.metric("Service Level", f"{inv_kpis['service_level']}%",
              delta=f"{svc_gap:.1f}pp vs target", delta_color="inverse" if svc_gap < 0 else "normal")
with inv_cols[3]:
    wc_delta_m = inv_kpis['working_capital_delta'] / 1000000
    st.metric("Working Capital", f"${inv_kpis['working_capital']/1000000:.0f}M",
              delta=f"${wc_delta_m:.0f}M MoM", delta_color="inverse")
with inv_cols[4]:
    shrink_status = "Under target" if inv_kpis['shrinkage_rate'] < inv_kpis['shrinkage_target'] else "Above target"
    st.metric("Shrinkage Rate", f"{inv_kpis['shrinkage_rate']:.2f}%",
              delta=shrink_status, delta_color="normal" if inv_kpis['shrinkage_rate'] < inv_kpis['shrinkage_target'] else "inverse")

st.markdown("---")

# TABS (REDESIGNED ORDER - Aligned to VP presentation flow)
tab0, tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs([
    "📊 Executive Summary",
    "📈 Time Series Forecast",
    "🎯 Demand Drivers",
    "💎 Attribute Planning",
    "⚖️ Asset vs JIT",
    "🔄 Demand → Supply",
    "🕸️ Supply Network",
    "⚡ Control Tower"
])

# =============================================================================
# TAB 0: EXECUTIVE SUMMARY (VP-Level View)
# =============================================================================
with tab0:
    st.markdown("### 📊 Executive Summary - Strategic Overview")
    st.caption("One-page view for VP+ leadership | Updated: " + datetime.now().strftime("%B %d, %Y"))

    # Traffic Light Status Row
    st.markdown("#### Strategic Health Indicators")
    status_cols = st.columns(4)

    with status_cols[0]:
        st.markdown("""
        <div style="background: linear-gradient(135deg, #1f2833 0%, #0b0c10 100%); padding: 20px; border-radius: 10px; border-left: 5px solid #00ff88; text-align: center;">
            <div style="font-size: 40px;">🟢</div>
            <div style="color: #66fcf1; font-size: 14px; font-weight: 600; margin-top: 10px;">DEMAND FORECAST</div>
            <div style="color: #c5c6c7; font-size: 12px;">MAPE at 8.2% (target: <10%)</div>
        </div>
        """, unsafe_allow_html=True)

    with status_cols[1]:
        st.markdown("""
        <div style="background: linear-gradient(135deg, #1f2833 0%, #0b0c10 100%); padding: 20px; border-radius: 10px; border-left: 5px solid #ffd700; text-align: center;">
            <div style="font-size: 40px;">🟡</div>
            <div style="color: #66fcf1; font-size: 14px; font-weight: 600; margin-top: 10px;">SERVICE LEVEL</div>
            <div style="color: #c5c6c7; font-size: 12px;">96.8% (target: 98%) -1.2pp gap</div>
        </div>
        """, unsafe_allow_html=True)

    with status_cols[2]:
        st.markdown("""
        <div style="background: linear-gradient(135deg, #1f2833 0%, #0b0c10 100%); padding: 20px; border-radius: 10px; border-left: 5px solid #00ff88; text-align: center;">
            <div style="font-size: 40px;">🟢</div>
            <div style="color: #66fcf1; font-size: 14px; font-weight: 600; margin-top: 10px;">WORKING CAPITAL</div>
            <div style="color: #c5c6c7; font-size: 12px;">$549M (-$12M MoM improvement)</div>
        </div>
        """, unsafe_allow_html=True)

    with status_cols[3]:
        st.markdown("""
        <div style="background: linear-gradient(135deg, #1f2833 0%, #0b0c10 100%); padding: 20px; border-radius: 10px; border-left: 5px solid #00ff88; text-align: center;">
            <div style="font-size: 40px;">🟢</div>
            <div style="color: #66fcf1; font-size: 14px; font-weight: 600; margin-top: 10px;">SUPPLIER RISK</div>
            <div style="color: #c5c6c7; font-size: 12px;">Low concentration, diversified</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("---")

    # Top 3 Actions Required
    st.markdown("#### Top 3 Actions Required")
    action_cols = st.columns(3)

    with action_cols[0]:
        st.markdown("""
        <div style="background: #1f2833; padding: 15px; border-radius: 8px; border: 1px solid #ff4444;">
            <div style="color: #ff4444; font-weight: 600; font-size: 12px;">🚨 HIGH PRIORITY</div>
            <div style="color: #66fcf1; font-size: 16px; font-weight: 600; margin: 10px 0;">Increase 1.5ct Mid-Tier Inventory</div>
            <div style="color: #c5c6c7; font-size: 13px;">Demand score 94/100 but only 67% stock availability. Recommend +25% Antwerp orders.</div>
            <div style="color: #00ff88; font-size: 14px; font-weight: 600; margin-top: 10px;">💰 Impact: +$2.3M revenue opportunity</div>
        </div>
        """, unsafe_allow_html=True)

    with action_cols[1]:
        st.markdown("""
        <div style="background: #1f2833; padding: 15px; border-radius: 8px; border: 1px solid #ffd700;">
            <div style="color: #ffd700; font-weight: 600; font-size: 12px;">⚠️ MEDIUM PRIORITY</div>
            <div style="color: #66fcf1; font-size: 16px; font-weight: 600; margin: 10px 0;">Rebalance Newport Beach Inventory</div>
            <div style="color: #c5c6c7; font-size: 13px;">Showroom overstocked on 2.0ct+ (slow velocity). Transfer 45 units to Roosevelt Field.</div>
            <div style="color: #00ff88; font-size: 14px; font-weight: 600; margin-top: 10px;">💰 Impact: $890K working capital freed</div>
        </div>
        """, unsafe_allow_html=True)

    with action_cols[2]:
        st.markdown("""
        <div style="background: #1f2833; padding: 15px; border-radius: 8px; border: 1px solid #45a29e;">
            <div style="color: #45a29e; font-weight: 600; font-size: 12px;">📋 OPTIMIZATION</div>
            <div style="color: #66fcf1; font-size: 16px; font-weight: 600; margin: 10px 0;">Shift 23 SKUs from Asset → JIT</div>
            <div style="color: #c5c6c7; font-size: 13px;">Low-velocity 2.5ct+ items have 14-day lead time tolerance. Convert to made-to-order.</div>
            <div style="color: #00ff88; font-size: 14px; font-weight: 600; margin-top: 10px;">💰 Impact: $2.1M working capital freed</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("---")

    # AI Insight Summary
    st.markdown("#### AI Strategic Insight")
    st.markdown("""
    <div style="background: linear-gradient(135deg, #1f2833 0%, #0b0c10 100%); padding: 20px; border-radius: 10px; border: 2px solid #45a29e;">
        <div style="color: #66fcf1; font-size: 14px; line-height: 1.8;">
        <b>Summary:</b> Inventory health is strong with turns at 4.2x (above 4.0x benchmark). Primary gap is service level at 96.8% vs 98% target,
        driven by understocking in the high-demand 1.0-1.5ct mid-tier segment ($3-7K price point). Engagement season (Jan-Feb) will amplify this gap.
        <br><br>
        <b>Recommendation:</b> Execute the 3 actions above to capture $2.3M revenue opportunity while freeing $3.0M in working capital.
        Net impact: <span style="color: #00ff88; font-weight: 600;">+$5.3M value creation</span> with improved service levels heading into peak season.
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Key Metrics Trend (Mini sparklines)
    st.markdown("---")
    st.markdown("#### 12-Week Trend Summary")
    trend_cols = st.columns(4)

    # Generate mini trend data
    np.random.seed(42)
    weeks = list(range(1, 13))

    with trend_cols[0]:
        dos_trend = [45 + np.random.randint(-3, 4) for _ in weeks]
        fig_dos = go.Figure(go.Scatter(x=weeks, y=dos_trend, mode='lines', line=dict(color='#66fcf1', width=2), fill='tozeroy', fillcolor='rgba(102, 252, 241, 0.1)'))
        fig_dos.update_layout(height=100, margin=dict(l=0, r=0, t=0, b=0), paper_bgcolor='#0b0c10', plot_bgcolor='#0b0c10',
                             xaxis=dict(visible=False), yaxis=dict(visible=False))
        st.plotly_chart(fig_dos, use_container_width=True)
        st.caption("Days of Supply: 47 days")

    with trend_cols[1]:
        svc_trend = [96 + np.random.uniform(-1, 1.5) for _ in weeks]
        fig_svc = go.Figure(go.Scatter(x=weeks, y=svc_trend, mode='lines', line=dict(color='#ffd700', width=2), fill='tozeroy', fillcolor='rgba(255, 215, 0, 0.1)'))
        fig_svc.update_layout(height=100, margin=dict(l=0, r=0, t=0, b=0), paper_bgcolor='#0b0c10', plot_bgcolor='#0b0c10',
                             xaxis=dict(visible=False), yaxis=dict(visible=False))
        st.plotly_chart(fig_svc, use_container_width=True)
        st.caption("Service Level: 96.8%")

    with trend_cols[2]:
        turns_trend = [4.0 + np.random.uniform(-0.2, 0.3) for _ in weeks]
        fig_turns = go.Figure(go.Scatter(x=weeks, y=turns_trend, mode='lines', line=dict(color='#00ff88', width=2), fill='tozeroy', fillcolor='rgba(0, 255, 136, 0.1)'))
        fig_turns.update_layout(height=100, margin=dict(l=0, r=0, t=0, b=0), paper_bgcolor='#0b0c10', plot_bgcolor='#0b0c10',
                               xaxis=dict(visible=False), yaxis=dict(visible=False))
        st.plotly_chart(fig_turns, use_container_width=True)
        st.caption("Inventory Turns: 4.2x")

    with trend_cols[3]:
        wc_trend = [560 - i*1 + np.random.randint(-5, 5) for i in weeks]
        fig_wc = go.Figure(go.Scatter(x=weeks, y=wc_trend, mode='lines', line=dict(color='#45a29e', width=2), fill='tozeroy', fillcolor='rgba(69, 162, 158, 0.1)'))
        fig_wc.update_layout(height=100, margin=dict(l=0, r=0, t=0, b=0), paper_bgcolor='#0b0c10', plot_bgcolor='#0b0c10',
                            xaxis=dict(visible=False), yaxis=dict(visible=False))
        st.plotly_chart(fig_wc, use_container_width=True)
        st.caption("Working Capital: $549M")

with tab1:
    st.markdown("### 📈 Time Series Forecast - Historical vs Forecast (18 Months)")
    st.caption("12 months actuals + 6 months forecast | Engagement season drives 40% demand spike | 80% confidence interval")
    
    ts_data = generate_time_series_forecast()
    
    fig = go.Figure()
    
    # Historical actuals
    fig.add_trace(go.Scatter(x=ts_data['months'][:12], y=ts_data['actuals'][:12], mode='lines+markers',
                            name='Historical Actuals', line=dict(color='#66fcf1', width=4), marker=dict(size=8)))
    
    # Forecast scenario selection
    if forecast_scenario == "Statistical":
        forecast_values = ts_data['statistical'][12:]
        forecast_color = '#ffd700'
    elif forecast_scenario == "ML Model":
        forecast_values = ts_data['ml'][12:]
        forecast_color = '#ff8c00'
    else:  # Consensus
        forecast_values = ts_data['consensus'][12:]
        forecast_color = '#45a29e'
    
    fig.add_trace(go.Scatter(x=ts_data['months'][12:], y=forecast_values, mode='lines+markers',
                            name=f'{forecast_scenario} Forecast', line=dict(color=forecast_color, width=4, dash='dash'),
                            marker=dict(size=8)))
    
    # Confidence interval
    fig.add_trace(go.Scatter(x=ts_data['months'][12:] + ts_data['months'][12:][::-1],
                            y=ts_data['upper'][12:] + ts_data['lower'][12:][::-1],
                            fill='toself', fillcolor='rgba(69, 162, 158, 0.2)', line=dict(color='rgba(255,255,255,0)'),
                            name='80% Confidence', hoverinfo='skip', showlegend=True))
    
    # "Today" marker
    fig.add_vline(x=11.5, line=dict(color='#ff4444', width=2, dash='dot'), annotation_text="Today", annotation_position="top")
    
    # Annotations
    fig.add_annotation(x=11, y=ts_data['actuals'][11], text="Holiday Peak<br>+42% vs avg", showarrow=True,
                      arrowhead=2, arrowcolor='#ffd700', font=dict(size=11, color='#ffd700'),
                      bgcolor='#1f2833', bordercolor='#ffd700', borderwidth=2)
    
    fig.update_layout(height=500, paper_bgcolor='#0b0c10', plot_bgcolor='#1f2833',
                     xaxis=dict(title='Month', gridcolor='#45a29e', color='#c5c6c7', showgrid=True),
                     yaxis=dict(title='Demand (Units)', gridcolor='#45a29e', color='#c5c6c7', showgrid=True),
                     font=dict(color='#c5c6c7', size=12), hovermode='x unified',
                     legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1))
    st.plotly_chart(fig, use_container_width=True)
    
    # Scenario comparison
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown("**Statistical Model**")
        st.caption(f"Feb-25 Forecast: {ts_data['statistical'][13]:,} units")
    with col2:
        st.markdown("**ML Model**")
        st.caption(f"Feb-25 Forecast: {ts_data['ml'][13]:,} units (+4% vs Statistical)")
    with col3:
        st.markdown("**Consensus (Approved) ✓**")
        st.caption(f"Feb-25 Forecast: {ts_data['consensus'][13]:,} units (Balanced)")
    
    st.success("✅ **Key Insight:** Engagement season (Jan-Feb) drives 40% demand spike. ML model predicts 4% higher than statistical due to social media signals.")

with tab2:
    st.markdown("### 🎯 Demand Drivers & Causal Factors")
    st.caption("Promotion calendar, social media buzz, holiday effects | Waterfall analysis")
    
    df_drivers = generate_demand_drivers()
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df_drivers['Week'], y=df_drivers['Promotions'], mode='lines+markers',
                            name='Promotion Impact', line=dict(color='#ffd700', width=3), marker=dict(size=8)))
    fig.add_trace(go.Scatter(x=df_drivers['Week'], y=df_drivers['Social'], mode='lines+markers',
                            name='Social Media Buzz', line=dict(color='#66fcf1', width=3), marker=dict(size=8)))
    fig.add_trace(go.Scatter(x=df_drivers['Week'], y=df_drivers['Holidays'], mode='lines+markers',
                            name='Holiday Effect', line=dict(color='#00ff88', width=3), marker=dict(size=8)))
    
    fig.update_layout(height=400, paper_bgcolor='#0b0c10', plot_bgcolor='#1f2833',
                     xaxis=dict(title='Week', gridcolor='#45a29e', color='#c5c6c7'),
                     yaxis=dict(title='Impact Score', gridcolor='#45a29e', color='#c5c6c7'),
                     font=dict(color='#c5c6c7'), hovermode='x unified')
    st.plotly_chart(fig, use_container_width=True)
    
    st.warning("⚠️ **Alert:** Week 6 shows viral TikTok trend (92 social buzz score). Recommend 25% inventory increase for 1.5ct Oval cuts.")

with tab3:
    st.markdown("### 💎 Attribute-Based Planning - Demand Intensity Heatmap")
    st.caption("Red = High demand / Low supply | Plan by attributes (carat, price) not SKU")
    
    carat_sizes, price_tiers, demand_matrix = generate_attribute_heatmap()
    
    fig = go.Figure(data=go.Heatmap(z=demand_matrix, x=price_tiers, y=carat_sizes,
                                     colorscale=[[0, '#00ff88'], [0.5, '#ffd700'], [1, '#ff4444']],
                                     text=demand_matrix, texttemplate='%{text}', textfont=dict(size=14, color='#0b0c10')))
    fig.update_layout(height=400, paper_bgcolor='#0b0c10', plot_bgcolor='#0b0c10',
                     xaxis=dict(title='Price Tier', color='#c5c6c7'), yaxis=dict(title='Carat Size', color='#c5c6c7'),
                     font=dict(color='#c5c6c7'))
    st.plotly_chart(fig, use_container_width=True)
    
    st.warning("⚠️ **UNDER-INVENTORIED:** 1.5ct / Mid-Tier ($3-7K) shows demand score 94 (red hot). Recommend increasing Antwerp orders by 25%.")

# =============================================================================
# TAB 4: ASSET vs JIT ANALYSIS (Key concept from JD)
# =============================================================================
with tab4:
    st.markdown("### ⚖️ Asset vs JIT SKU Strategy")
    st.caption("Optimize working capital by balancing always-in-stock vs made-to-order inventory | Key metric from Manager, Inventory Planning role")

    # Get SKU data
    sku_df = generate_asset_jit_data()

    # Interactive threshold slider
    st.markdown("#### Optimization Threshold")
    velocity_threshold = st.slider(
        "Monthly Velocity Threshold (units/month)",
        min_value=10, max_value=60, value=30,
        help="SKUs above this threshold = Asset (always in stock). Below = JIT (made-to-order)."
    )

    # Recalculate based on slider
    sku_df['recommended_strategy'] = sku_df['monthly_velocity'].apply(lambda x: 'Asset' if x > velocity_threshold else 'JIT')
    sku_df['strategy_change'] = sku_df['current_strategy'] != sku_df['recommended_strategy']

    # Summary metrics
    asset_skus = len(sku_df[sku_df['recommended_strategy'] == 'Asset'])
    jit_skus = len(sku_df[sku_df['recommended_strategy'] == 'JIT'])
    changes_needed = sku_df['strategy_change'].sum()

    # Calculate working capital impact
    current_asset_wc = sku_df[sku_df['current_strategy'] == 'Asset']['unit_value'].sum() * 3  # 3 units avg on hand
    recommended_asset_wc = sku_df[sku_df['recommended_strategy'] == 'Asset']['unit_value'].sum() * 3
    wc_freed = current_asset_wc - recommended_asset_wc

    metric_cols = st.columns(4)
    with metric_cols[0]:
        st.metric("Asset SKUs", f"{asset_skus}", delta=f"Always in stock")
    with metric_cols[1]:
        st.metric("JIT SKUs", f"{jit_skus}", delta="Made-to-order")
    with metric_cols[2]:
        st.metric("Strategy Changes", f"{changes_needed}", delta="Recommended moves")
    with metric_cols[3]:
        st.metric("Working Capital Impact", f"${wc_freed/1000000:.1f}M", delta="Freed if optimized", delta_color="normal")

    st.markdown("---")

    # Scatter plot: Velocity vs Value with strategy coloring
    col1, col2 = st.columns([2, 1])

    with col1:
        fig = px.scatter(
            sku_df, x='monthly_velocity', y='unit_value',
            color='recommended_strategy',
            color_discrete_map={'Asset': '#00ff88', 'JIT': '#ffd700'},
            hover_data=['sku_id', 'shape', 'carat', 'lead_time_days'],
            title='SKU Portfolio: Velocity vs Value'
        )
        fig.add_vline(x=velocity_threshold, line_dash="dash", line_color="#ff4444",
                     annotation_text=f"Threshold: {velocity_threshold} units/mo")
        fig.update_layout(
            height=400, paper_bgcolor='#0b0c10', plot_bgcolor='#1f2833',
            xaxis=dict(title='Monthly Velocity (units)', gridcolor='#45a29e', color='#c5c6c7'),
            yaxis=dict(title='Unit Value ($)', gridcolor='#45a29e', color='#c5c6c7'),
            font=dict(color='#c5c6c7'),
            legend=dict(orientation="h", yanchor="bottom", y=1.02)
        )
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        # Pie chart of strategy mix
        strategy_counts = sku_df['recommended_strategy'].value_counts()
        fig_pie = go.Figure(data=[go.Pie(
            labels=strategy_counts.index,
            values=strategy_counts.values,
            hole=0.6,
            marker=dict(colors=['#00ff88', '#ffd700'])
        )])
        fig_pie.update_layout(
            height=300, paper_bgcolor='#0b0c10',
            font=dict(color='#c5c6c7'),
            showlegend=True,
            annotations=[dict(text='Strategy<br>Mix', x=0.5, y=0.5, font_size=14, showarrow=False, font_color='#66fcf1')]
        )
        st.plotly_chart(fig_pie, use_container_width=True)

    # Recommendation box
    if changes_needed > 0:
        st.info(f"""
        **💡 Optimization Recommendation:**
        Move **{changes_needed} SKUs** from Asset → JIT to free **${wc_freed/1000000:.1f}M** in working capital.

        Top candidates for JIT conversion:
        - 2.5ct+ diamonds (14-day lead time tolerance, low velocity)
        - Specialty shapes (Marquise, Pear) in larger sizes

        *This aligns with the "asset versus just-in-time SKU mix optimization" objective.*
        """)

# =============================================================================
# TAB 5: DEMAND TO SUPPLY TRANSLATION
# =============================================================================
with tab5:
    st.markdown("### 🔄 Demand to Supply Translation")
    st.caption("End-to-end flow from forecast to purchase orders | Key capability from Manager, Inventory Planning role")

    flow_data = generate_demand_supply_flow()

    # Flow visualization using Sankey diagram
    st.markdown("#### Planning Waterfall: Forecast → Orders")

    # Sankey data
    labels = ['Demand Forecast', 'Safety Stock', 'Net Requirements',
              'Kiran Gems (Surat)', 'Antwerp Diamond Bank', 'Mumbai Diamond Exchange', 'Tel Aviv Diamond Exchange',
              'Purchase Orders']

    source = [0, 1, 2, 2, 2, 2, 3, 4, 5, 6]  # From nodes
    target = [2, 2, 3, 4, 5, 6, 7, 7, 7, 7]  # To nodes
    values = [
        flow_data['demand_forecast'],
        flow_data['safety_stock_buffer'],
        flow_data['vendor_allocation']['Kiran Gems (Surat)']['units'],
        flow_data['vendor_allocation']['Antwerp Diamond Bank']['units'],
        flow_data['vendor_allocation']['Mumbai Diamond Exchange']['units'],
        flow_data['vendor_allocation']['Tel Aviv Diamond Exchange']['units'],
        flow_data['vendor_allocation']['Kiran Gems (Surat)']['units'],
        flow_data['vendor_allocation']['Antwerp Diamond Bank']['units'],
        flow_data['vendor_allocation']['Mumbai Diamond Exchange']['units'],
        flow_data['vendor_allocation']['Tel Aviv Diamond Exchange']['units']
    ]

    fig_sankey = go.Figure(data=[go.Sankey(
        node=dict(
            pad=20,
            thickness=20,
            line=dict(color='#45a29e', width=1),
            label=labels,
            color=['#66fcf1', '#ffd700', '#45a29e', '#00ff88', '#00ff88', '#00ff88', '#00ff88', '#ff8c00']
        ),
        link=dict(
            source=source,
            target=target,
            value=values,
            color='rgba(69, 162, 158, 0.4)'
        )
    )])
    fig_sankey.update_layout(
        height=400, paper_bgcolor='#0b0c10', font=dict(color='#c5c6c7', size=12)
    )
    st.plotly_chart(fig_sankey, use_container_width=True)

    st.markdown("---")

    # Vendor allocation details
    st.markdown("#### Vendor Allocation & Capacity")
    vendor_cols = st.columns(4)

    for idx, (vendor, data) in enumerate(flow_data['vendor_allocation'].items()):
        with vendor_cols[idx]:
            capacity_color = '#00ff88' if data['capacity_util'] < 70 else '#ffd700' if data['capacity_util'] < 85 else '#ff4444'
            st.markdown(f"""
            <div style="background: #1f2833; padding: 15px; border-radius: 8px; text-align: center;">
                <div style="color: #66fcf1; font-size: 14px; font-weight: 600;">{vendor.split('(')[0].strip()}</div>
                <div style="color: #c5c6c7; font-size: 12px; margin: 5px 0;">{data['allocation_pct']}% allocation</div>
                <div style="color: #66fcf1; font-size: 20px; font-weight: 700;">{data['units']:,}</div>
                <div style="color: #c5c6c7; font-size: 11px;">units this period</div>
                <div style="margin-top: 10px; background: #0b0c10; border-radius: 4px; height: 8px;">
                    <div style="background: {capacity_color}; width: {data['capacity_util']}%; height: 8px; border-radius: 4px;"></div>
                </div>
                <div style="color: {capacity_color}; font-size: 11px; margin-top: 5px;">{data['capacity_util']}% capacity</div>
            </div>
            """, unsafe_allow_html=True)

    st.markdown("---")

    # PO Summary
    st.markdown("#### Purchase Order Summary")
    po_cols = st.columns(5)
    po_data = flow_data['po_summary']

    with po_cols[0]:
        st.metric("Total Active POs", po_data['total_pos'])
    with po_cols[1]:
        st.metric("Created This Week", po_data['this_week'], delta="+3 vs last week")
    with po_cols[2]:
        st.metric("Pending Approval", po_data['pending_approval'], delta="Needs action", delta_color="inverse")
    with po_cols[3]:
        st.metric("In Transit", po_data['in_transit'])
    with po_cols[4]:
        st.metric("Value In Transit", f"${po_data['value_in_transit']/1000000:.1f}M")

    st.success("✅ **Translation Complete:** Demand forecast of 14,200 units + 2,100 safety stock buffer = 16,300 net requirements distributed across 4 vendors based on capacity and lead time.")

# =============================================================================
# TAB 6: SUPPLY NETWORK (moved from old tab4)
# =============================================================================
with tab6:
    st.markdown("### 🕸️ Enterprise Knowledge Graph - Supply Network")
    st.caption("4 Suppliers → NYC Hub → 4 Showrooms | 3D interactive visualization")

    fig = create_network_graph()
    st.plotly_chart(fig, use_container_width=True)

    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown("**🟢 Suppliers**")
        st.caption("India, Belgium, Israel, etc.")
    with col2:
        st.markdown("**💎 NYC Hub**")
        st.caption("Central fulfillment node")
    with col3:
        st.markdown("**🏪 Showrooms**")
        st.caption("18 retail locations")

# =============================================================================
# TAB 7: CONTROL TOWER WITH WHAT-IF SCENARIOS
# =============================================================================
with tab7:
    st.markdown("### ⚡ Control Tower - Global Logistics & Scenario Planning")
    st.caption("Real-time view of diamond flows + What-If analysis")

    # What-If Scenario Controls
    st.markdown("#### 🎯 What-If Scenario Simulator")
    scenario_cols = st.columns(3)

    with scenario_cols[0]:
        demand_shift = st.slider("Demand Shift", -20, 20, 0, format="%d%%",
                                help="Simulate demand increase/decrease")
    with scenario_cols[1]:
        supplier_disruption = st.selectbox("Supplier Disruption",
                                          ["None", "Kiran Gems (Surat)", "Antwerp Diamond Bank", "Mumbai Diamond Exchange", "Tel Aviv Diamond Exchange"],
                                          help="Simulate a supplier going offline")
    with scenario_cols[2]:
        lead_time_delay = st.slider("Lead Time Delay", 0, 10, 0, format="+%d days",
                                   help="Simulate customs/shipping delays")

    # Calculate scenario impact
    base_service_level = 96.8
    base_working_capital = 549

    scenario_service_impact = 0
    scenario_wc_impact = 0
    scenario_alerts = []

    if demand_shift != 0:
        scenario_service_impact -= abs(demand_shift) * 0.15  # Higher demand = lower service level
        scenario_wc_impact += demand_shift * 5  # More demand = more working capital needed
        if demand_shift > 0:
            scenario_alerts.append(f"📈 +{demand_shift}% demand requires +${demand_shift * 5}M inventory investment")

    if supplier_disruption != "None":
        scenario_service_impact -= 3.5  # Significant impact
        scenario_alerts.append(f"🚨 {supplier_disruption} offline: -3.5pp service level, activate backup suppliers")

    if lead_time_delay > 0:
        scenario_service_impact -= lead_time_delay * 0.5
        scenario_wc_impact += lead_time_delay * 2  # Need more safety stock
        scenario_alerts.append(f"⏱️ +{lead_time_delay} days delay requires +${lead_time_delay * 2}M safety stock")

    # Show scenario impact
    impact_cols = st.columns(3)
    with impact_cols[0]:
        new_service = base_service_level + scenario_service_impact
        delta_str = f"{scenario_service_impact:+.1f}pp" if scenario_service_impact != 0 else "No change"
        st.metric("Projected Service Level", f"{new_service:.1f}%", delta=delta_str,
                 delta_color="inverse" if scenario_service_impact < 0 else "normal")
    with impact_cols[1]:
        new_wc = base_working_capital + scenario_wc_impact
        delta_str = f"${scenario_wc_impact:+.0f}M" if scenario_wc_impact != 0 else "No change"
        st.metric("Projected Working Capital", f"${new_wc:.0f}M", delta=delta_str,
                 delta_color="inverse" if scenario_wc_impact > 0 else "normal")
    with impact_cols[2]:
        risk_level = "🟢 Low" if scenario_service_impact > -2 else "🟡 Medium" if scenario_service_impact > -5 else "🔴 High"
        st.metric("Risk Level", risk_level, delta="Scenario assessment")

    if scenario_alerts:
        for alert in scenario_alerts:
            st.warning(alert)

    st.markdown("---")

    # Original map
    fig = go.Figure(go.Scattergeo(
        locationmode='ISO-3',
        lon=[s['location']['lon'] for s in DIAMOND_SUPPLIERS.values()] + [-73.9855] + [l['lon'] for l in SHOWROOM_LOCATIONS],
        lat=[s['location']['lat'] for s in DIAMOND_SUPPLIERS.values()] + [40.7580] + [l['lat'] for l in SHOWROOM_LOCATIONS],
        mode='markers',
        marker=dict(size=[15]*len(DIAMOND_SUPPLIERS) + [25] + [10]*len(SHOWROOM_LOCATIONS),
                   color=['#00ff88']*len(DIAMOND_SUPPLIERS) + ['#66fcf1'] + ['#c5c6c7']*len(SHOWROOM_LOCATIONS),
                   line=dict(width=2, color='white')),
        text=list(DIAMOND_SUPPLIERS.keys()) + ['NYC Hub'] + [l['name'] for l in SHOWROOM_LOCATIONS]
    ))

    fig.update_geos(bgcolor='#0b0c10', landcolor='#1f2833', oceancolor='#0b0c10',
                    showcountries=True, countrycolor='#45a29e', projection_type='natural earth')
    fig.update_layout(height=400, paper_bgcolor='#0b0c10', geo=dict(bgcolor='#0b0c10'), margin=dict(l=0, r=0, t=0, b=0))
    st.plotly_chart(fig, use_container_width=True)

    metrics_cols = st.columns(3)
    with metrics_cols[0]:
        st.metric("Active Shipments", "47", delta="+12 today")
    with metrics_cols[1]:
        st.metric("Avg Transit Time", "2.8 days", delta="-0.3 days")
    with metrics_cols[2]:
        st.metric("Customs Clearance", "18 hrs", delta="On target")

st.markdown("---")
st.caption("Blue Nile Integrated Business Planning Platform | Demand • Inventory • Supply Chain | v3.0 - Executive Edition")

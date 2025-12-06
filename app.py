import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.cm as cm

# ========================
# Page Config
# ========================
st.set_page_config(page_title="⚡ Electricity Bill Predictor", page_icon="⚡", layout="centered")

# ========================
# Load Model
# ========================
@st.cache_resource
def load_model(path="electricity_model.pkl"):
    return joblib.load(path)

model = load_model()

# ========================
# Custom CSS
# ========================
st.markdown("""
    <style>
        .stButton button {
            background-color: #0099f7;
            color: white;
            border-radius: 8px;
        }
        .stButton button:hover {
            background-color: #007acc;
        }
        div[data-baseweb="number-input"] input {
            color: gray;
        }
    </style>
""", unsafe_allow_html=True)

# ========================
# Title & Instructions
# ========================
st.title("⚡ Smart Electricity Bill Predictor")
st.markdown("🔍 Enter your appliance usage below to predict your monthly bill and get insights!")

# ========================
# Helper Function: Convert decimal hours to H:M string
# ========================
def format_hours_decimal(hours):
    h = int(hours)
    m = int(round((hours - h) * 60))
    return f"{h}h {m}m"

# ========================
# User Inputs
# ========================
st.header("🛠 Appliance Usage")
col1, col2 = st.columns(2)

with col1:
    ac_count = st.number_input("Number of ACs", min_value=0, step=1, value=0)
    ac_hours = st.number_input("AC usage (hrs/day)", min_value=0.0, max_value=24.0, value=0.0, step=0.01, format="%.2f")
    fan_count = st.number_input("Number of Fans", min_value=0, step=1, value=0)
    fan_hours = st.number_input("Fan usage (hrs/day)", min_value=0.0, max_value=24.0, value=0.0, step=0.01, format="%.2f")
    geyser_count = st.number_input("Number of Geysers", min_value=0, step=1, value=0)
    geyser_hours = st.number_input("Geyser usage (hrs/day)", min_value=0.0, max_value=24.0, value=0.0, step=0.01, format="%.2f")

with col2:
    tv_count = st.number_input("Number of TVs", min_value=0, step=1, value=0)
    tv_hours = st.number_input("TV usage (hrs/day)", min_value=0.0, max_value=24.0, value=0.0, step=0.01, format="%.2f")
    fridge_count = st.number_input("Number of Fridges", min_value=0, step=1, value=0)
    fridge_hours = st.number_input("Fridge usage (hrs/day)", min_value=0.0, max_value=24.0, value=0.0, step=0.01, format="%.2f")

# ========================
# Custom Appliances
# ========================
st.subheader("➕ Additional Appliances")
if "custom_appliances" not in st.session_state:
    st.session_state.custom_appliances = []

if "new_appliance_name" not in st.session_state:
    st.session_state.new_appliance_name = ""
if "new_appliance_hours" not in st.session_state:
    st.session_state.new_appliance_hours = 0.0
if "new_appliance_power" not in st.session_state:
    st.session_state.new_appliance_power = 0.0

def add_appliance():
    if st.session_state.new_appliance_name.strip():
        hours = round(st.session_state.new_appliance_hours, 2)
        st.session_state.custom_appliances.append({
            "name": st.session_state.new_appliance_name.strip(),
            "hours": hours,
            "power": st.session_state.new_appliance_power
        })
        st.session_state.new_appliance_name = ""
        st.session_state.new_appliance_hours = 0.0
        st.session_state.new_appliance_power = 0.0
        st.success("Appliance added!")

with st.expander("Add new appliance"):
    st.text_input("Appliance Name", key="new_appliance_name", placeholder="Enter appliance name")
    st.number_input("Usage Hours per Day", min_value=0.0, max_value=24.0, value=0.0, step=0.01, format="%.2f", key="new_appliance_hours")
    st.number_input("Power Rating (kW)", min_value=0.0, max_value=10.0, value=0.0, step=0.1, key="new_appliance_power")
    st.button("➕ Add Appliance", on_click=add_appliance)

# Show added appliances
if st.session_state.custom_appliances:
    st.subheader("**Added Appliances:**")
    for i, a in enumerate(st.session_state.custom_appliances):
        container = st.container()
        cols = container.columns([4,1])
        cols[0].markdown(f"<span style='color:gray'>{a['name']} — {format_hours_decimal(a['hours'])} × {a['power']} kW</span>", unsafe_allow_html=True)
        if cols[1].button("❌", key=f"del_{i}"):
            st.session_state.custom_appliances.pop(i)
            st.experimental_rerun()

# ========================
# Rate & Previous Bill
# ========================
st.subheader("💰 Electricity Rate & Previous Bill")
rate_per_unit = st.number_input("Electricity Rate per Unit (₹)", min_value=0.0, value=0.0, step=0.1)
previous_bill = st.number_input("💸 Last Month's Bill (₹)", min_value=0.0, value=0.0, step=0.1)

# ========================
# Prepare Input for Model
# ========================
input_data = {}
for feature in model.feature_names_in_:
    if feature == "AC_Count": input_data[feature] = [ac_count]
    elif feature == "AC_Hours": input_data[feature] = [ac_hours]
    elif feature == "Fan_Count": input_data[feature] = [fan_count]
    elif feature == "Fan_Hours": input_data[feature] = [fan_hours]
    elif feature == "TV_Count": input_data[feature] = [tv_count]
    elif feature == "TV_Hours": input_data[feature] = [tv_hours]
    elif feature == "Fridge_Count": input_data[feature] = [fridge_count]
    elif feature == "Fridge_Hours": input_data[feature] = [fridge_hours]
    elif feature == "Geyser_Count": input_data[feature] = [geyser_count]
    elif feature == "Geyser_Hours": input_data[feature] = [geyser_hours]
    elif feature == "Rate_per_Unit": input_data[feature] = [rate_per_unit]
    else:
        input_data[feature] = [0]

input_df = pd.DataFrame(input_data)

# ========================
# History
# ========================      
if "history" not in st.session_state:
    st.session_state.history = pd.DataFrame(columns=list(input_df.columns)+["Other_Units","Predicted_Bill","Previous_Bill"])

# ========================
# Predict Button
# ========================
if st.button("🔮 Predict My Bill"):
    try:
        model_prediction = model.predict(input_df)[0]
    except:
        model_prediction = 0
        st.error("Prediction failed.")

    other_units = sum([a['hours']*a['power'] for a in st.session_state.custom_appliances])
    other_bill = other_units * 30 * rate_per_unit
    final_bill = round(max(model_prediction + other_bill, 0),2)
    st.success(f"💰 Estimated Monthly Bill: ₹{final_bill}")

    new_row = input_df.copy()
    new_row["Other_Units"] = other_units
    new_row["Predicted_Bill"] = final_bill
    new_row["Previous_Bill"] = previous_bill
    st.session_state.history = pd.concat([st.session_state.history, new_row], ignore_index=True)

    # ========================
    # Appliance Usage Chart
    # ========================
    st.subheader("📊 Appliance Usage Overview")
    appliances = ["AC","Fan","TV","Fridge","Geyser"] + [a['name'] for a in st.session_state.custom_appliances]
    total_hours = [
        ac_count*ac_hours, fan_count*fan_hours, tv_count*tv_hours,
        fridge_count*fridge_hours, geyser_count*geyser_hours
    ] + [a['hours']*a['power'] for a in st.session_state.custom_appliances]

    fig1, ax1 = plt.subplots(figsize=(7,4))
    ax1.barh(appliances, total_hours, color=cm.tab20.colors[:len(appliances)])
    ax1.set_xlabel("Total Energy Use (kWh per day)")
    ax1.set_title("Appliance Usage Overview")
    for i, v in enumerate(total_hours):
        ax1.text(v+0.5,i,f"{format_hours_decimal(v)}", va='center')
    st.pyplot(fig1)

    # Pie chart cost contribution
    st.subheader("🥧 Estimated Cost Contribution")
    pie_data = {
        "AC": ac_count*ac_hours*rate_per_unit,
        "Fan": fan_count*fan_hours*rate_per_unit,
        "TV": tv_count*tv_hours*rate_per_unit,
        "Fridge": fridge_count*fridge_hours*rate_per_unit,
        "Geyser": geyser_count*geyser_hours*rate_per_unit
    }
    pie_data.update({a['name']: a['hours']*a['power']*rate_per_unit for a in st.session_state.custom_appliances})
    pie_data = {k:v for k,v in pie_data.items() if v>0}

    if pie_data:
        fig2, ax2 = plt.subplots(figsize=(6,6))
        ax2.pie(list(pie_data.values()), labels=list(pie_data.keys()), autopct='%1.1f%%', colors=cm.tab20.colors[:len(pie_data)])
        ax2.set_title("Bill Contribution by Appliance")
        st.pyplot(fig2)

# ========================
# Insights & History
# ========================
if not st.session_state.history.empty:
    st.header("📈 Prediction Insights")
    history = st.session_state.history

    if len(history)>=2:
        diff = history.iloc[-1]["Predicted_Bill"]-history.iloc[-2]["Predicted_Bill"]
        emoji = "📈 Increased" if diff>0 else "📉 Decreased" if diff<0 else "➖ No Change"
        st.info(f"From last prediction: ₹{abs(diff):.2f} ({emoji})")

    # Enhanced Trend Plot
    st.subheader("📊 Bill Prediction Trend")
    fig3, ax3 = plt.subplots(figsize=(8,4))
    ax3.plot(history.index, history["Predicted_Bill"], marker='o', linestyle='-', color='blue', label="Predicted Bill")
    ax3.plot(history.index, history["Previous_Bill"], marker='x', linestyle='--', color='orange', label="Previous Bill")
    for x, y in zip(history.index, history["Predicted_Bill"]):
        ax3.text(x, y+0.5, f"₹{y:.0f}", ha='center', va='bottom', fontsize=9, color='blue')
    for x, y in zip(history.index, history["Previous_Bill"]):
        ax3.text(x, y+0.5, f"₹{y:.0f}", ha='center', va='bottom', fontsize=9, color='orange')
    ax3.set_xlabel("Prediction Number")
    ax3.set_ylabel("Bill Amount (₹)")
    ax3.set_title("Monthly Bill Trend")
    ax3.grid(True, linestyle='--', alpha=0.5)
    ax3.legend()
    ax3.set_xticks(history.index)
    st.pyplot(fig3)

    # Heatmap of appliance usage
    st.subheader("🔥 Heatmap of Appliance Usage")
    if history.shape[0]>1:
        available_cols = [c for c in ["AC_Hours","Fan_Hours","TV_Hours","Fridge_Hours","Geyser_Hours"] if c in history.columns]
        usage_data = history[available_cols].copy()
        for a in st.session_state.custom_appliances:
            usage_data[a['name']] = a['hours']
        fig4, ax4 = plt.subplots(figsize=(8,6))
        sns.heatmap(usage_data.T, annot=True, fmt=".2f", cmap="YlGnBu", cbar_kws={'label': 'Hours per Day'}, ax=ax4)
        ax4.set_xlabel("Prediction Number")
        ax4.set_ylabel("Appliance")
        ax4.set_title("Appliance Usage Heatmap (Hours/Day)")
        st.pyplot(fig4)
    else:
        st.info("Make at least 2 predictions to see heatmap.")

    st.subheader("📁 Full Prediction History")
    st.dataframe(history)
    st.download_button("⬇ Download as CSV", data=history.to_csv(index=False), file_name="prediction_history.csv", mime="text/csv")

# ========================
# Saving Tips
# ========================
st.header("💡 Save Electricity Tips")
tips = [
    "✔ Use LED bulbs instead of regular ones",
    "✔ Turn off appliances when not needed",
    "✔ Keep AC at 24°C for efficiency",
    "✔ Wash clothes in full loads",
    "✔ Clean fridge coils for better cooling",
    "✔ Use natural light in daytime",
    "✔ Unplug idle electronics"
]
for tip in tips:
    st.markdown(f"- {tip}")
 ###########
 #########
 #####
 #######
 ###
 ###
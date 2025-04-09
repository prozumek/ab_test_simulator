import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Set random seed for reproducibility
np.random.seed(42)

st.title("🧩 A/B Test Segment Simulator — Fixed Version")

st.markdown("""
This sandbox lets you explore the impact of segmentation precision and uplift assumptions
for both Age-based (Gen Z) and Behavioural segments.

""")

# --- INPUTS ---
st.sidebar.header("Simulation Settings")

# Toggle behavioural curves
show_behavioural = st.sidebar.checkbox("Show Behavioural Segments", value=True)

# Population splits
gen_z_base_rate = st.sidebar.slider("% of Gen Z in population", 0, 100, 50) / 100
casual_base_rate = st.sidebar.slider("% of Casuals in population", 0, 100, 70) / 100

# Uplifts: Gen Z
st.sidebar.markdown('<span style="color:blue">**Gen Z Uplifts**</span>', unsafe_allow_html=True)
gen_z_control = st.sidebar.number_input("Gen Z Control Conversion Rate (%)", value=10.0) / 100
gen_z_treatment = st.sidebar.number_input("Gen Z Treatment Conversion Rate (%)", value=16.0) / 100

st.sidebar.markdown('<span style="color:orange">**Non-Gen Z Uplifts**</span>', unsafe_allow_html=True)
non_gen_z_control = st.sidebar.number_input("Non-Gen Z Control Conversion Rate (%)", value=10.0) / 100
non_gen_z_treatment = st.sidebar.number_input("Non-Gen Z Treatment Conversion Rate (%)", value=5.0) / 100

# Uplifts: Behavioural
st.sidebar.markdown('<span style="color:green">**Behavioural Uplifts (Casual Users)**</span>', unsafe_allow_html=True)
casual_control = st.sidebar.number_input("Casuals Control Conversion Rate (%)", value=10.0) / 100
casual_treatment = st.sidebar.number_input("Casuals Treatment Conversion Rate (%)", value=12.0) / 100

st.sidebar.markdown('<span style="color:red">**Heavies Uplifts**</span>', unsafe_allow_html=True)
heavy_control = st.sidebar.number_input("Heavies Control Conversion Rate (%)", value=10.0) / 100
heavy_treatment = st.sidebar.number_input("Heavies Treatment Conversion Rate (%)", value=7.0) / 100

# Fixed settings
population_size = 100_000

# --- SIMULATION ---
users = pd.DataFrame({
    'user_id': np.arange(population_size),
    'true_gen_z': np.random.choice([1, 0], size=population_size, p=[gen_z_base_rate, 1 - gen_z_base_rate]),
    'true_casual': np.random.choice([1, 0], size=population_size, p=[casual_base_rate, 1 - casual_base_rate]),
    'group': np.random.choice(['control', 'treatment'], size=population_size)
})

# Precompute conversion probabilities
users['conversion_prob_gen_z'] = np.where(
    users['true_gen_z'] == 1,
    np.where(users['group'] == 'control', gen_z_control, gen_z_treatment),
    np.where(users['group'] == 'control', non_gen_z_control, non_gen_z_treatment)
)

users['conversion_prob_behaviour'] = np.where(
    users['true_casual'] == 1,
    np.where(users['group'] == 'control', casual_control, casual_treatment),
    np.where(users['group'] == 'control', heavy_control, heavy_treatment)
)

# Precompute conversions
users['converted_gen_z'] = np.random.binomial(1, users['conversion_prob_gen_z'])
users['converted_behaviour'] = np.random.binomial(1, users['conversion_prob_behaviour'])

# Segment simulation
def apply_precision_segment(users, precision, segment_size, label):
    segment = pd.Series(0, index=users.index)

    true_positive_idx = users[users[label] == 1].index
    false_positive_idx = users[users[label] == 0].index

    n_tp = int(segment_size * precision)
    n_fp = segment_size - n_tp

    n_tp = min(n_tp, len(true_positive_idx))
    n_fp = min(n_fp, len(false_positive_idx))

    if n_tp > 0:
        tp_idx = np.random.choice(true_positive_idx, size=n_tp, replace=False)
        segment.loc[tp_idx] = 1

    if n_fp > 0:
        fp_idx = np.random.choice(false_positive_idx, size=n_fp, replace=False)
        segment.loc[fp_idx] = 1

    return segment

# Metrics function
def calculate_metrics(data, segment_column, precision, converted_column):
    results = []
    for segment_value in [1, 0]:
        segment_data = data[data[segment_column] == segment_value]
        if segment_data.empty:
            continue

        control = segment_data[segment_data['group'] == 'control']
        treatment = segment_data[segment_data['group'] == 'treatment']

        conv_control = control[converted_column].mean() if len(control) > 0 else 0
        conv_treatment = treatment[converted_column].mean() if len(treatment) > 0 else 0

        lift = (conv_treatment / conv_control - 1) if conv_control > 0 else np.nan

        results.append({
            'Segment': segment_value,
            'Control Conversion': conv_control,
            'Treatment Conversion': conv_treatment,
            'Lift': lift,
            'Precision': precision
        })
    return results

# Collect results
def simulate_segment(label, precision_values, converted_column, base_rate):
    results = []
    segment_size = int(population_size * base_rate)
    for precision in precision_values:
        users['predicted_segment'] = apply_precision_segment(users, precision, segment_size, label)
        metrics = calculate_metrics(users, 'predicted_segment', precision, converted_column)

        for metric in metrics:
            metric['Strategy'] = 'Age-based Gen Z' if label == 'true_gen_z' else 'Behavioural'
            metric['Segment Name'] = 'Gen Z' if (label == 'true_gen_z' and metric['Segment'] == 1) else (
                'Non-Gen Z' if label == 'true_gen_z' else (
                    'Casuals' if metric['Segment'] == 1 else 'Heavies'
                )
            )
            results.append(metric)

    return pd.DataFrame(results)

# Precision ranges
gen_z_precisions = np.linspace(0.1, 1.0, 10)
behaviour_precisions = np.linspace(0.1, 1.0, 10)

# Run simulations
gen_z_results = simulate_segment('true_gen_z', gen_z_precisions, 'converted_gen_z', gen_z_base_rate)
behaviour_results = simulate_segment('true_casual', behaviour_precisions, 'converted_behaviour', casual_base_rate)

final_results = pd.concat([gen_z_results, behaviour_results], ignore_index=True)

# Format for display
final_results_display = final_results.copy()
for col in ['Control Conversion', 'Treatment Conversion', 'Lift']:
    final_results_display[col] = (final_results_display[col] * 100).round(1).astype(str) + '%'

# --- PLOT FIRST ---
st.subheader("📈 Segment Precision vs. Observed Lift")
fig, ax = plt.subplots(figsize=(10, 6))

# Gen Z curve (blue & orange)
colors = {'Gen Z': 'blue', 'Non-Gen Z': 'orange', 'Casuals': 'green', 'Heavies': 'red'}
for segment_name, data in gen_z_results.groupby('Segment Name'):
    ax.plot(data['Precision'], data['Lift'], marker='o', color=colors[segment_name], label=f"Age-Based – {segment_name}")

# Behavioural curve (green & red), toggleable
if show_behavioural:
    for segment_name, data in behaviour_results.groupby('Segment Name'):
        ax.plot(data['Precision'], data['Lift'], marker='s', linestyle='--', color=colors[segment_name], label=f"Behavioural – {segment_name}")

# Reference lines
ax.axhline(0, color='grey', linestyle='--')
ax.axvline(0.3, color='lightgrey', linestyle='--')
ax.axvline(0.85, color='lightgrey', linestyle='--')

ax.set_xlabel('Segment Precision')
ax.set_ylabel('Observed Lift')
ax.set_ylim(-1, 1)
ax.set_title('Effect of Segment Precision on Observed Lift')
ax.legend()

st.pyplot(fig)

# --- THEN TABLE ---
st.subheader("📊 Simulation Results Table")
st.dataframe(final_results_display)

# Optional export
st.download_button("💾 Download Results as CSV", final_results.to_csv(index=False).encode('utf-8'), "simulation_results.csv", "text/csv")

st.markdown("""
---
Made with 🐅 by your data team. Now even your boss can play safely 😉
""")

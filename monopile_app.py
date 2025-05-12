import streamlit as st
import numpy as np


# --- Helper Functions (Conceptual & Simplified) ---
def calculate_fixed_base_frequency(EI_eff, M_RNA, M_tower_eff, H_eff):
    """Conceptual fixed-base natural frequency for a monopile-tower system."""
    if H_eff == 0: return 0
    # Simplified cantilever beam formula (very rough approximation)
    # M_tower_eff is an effective mass of the tower, 0.23 or 0.243 factor often used
    try:
        freq = (1 / (2 * np.pi)) * np.sqrt((3 * EI_eff) / ((0.243 * M_tower_eff + M_RNA) * H_eff ** 3))
        return freq
    except ZeroDivisionError:
        return 0
    except Exception as e:
        st.error(f"Frequency calculation error: {e}")
        return 0


def calculate_loads(wind_speed, wave_height, current_speed, rotor_diameter, hub_height, water_depth):
    """Conceptual load calculation - highly simplified."""
    # Placeholder thrust force
    F_wind_thrust = 0.5 * 1.225 * (np.pi * (rotor_diameter / 2) ** 2) * wind_speed ** 2 * 0.8  # Ct ~0.8
    M_wind_mudline = F_wind_thrust * (hub_height + water_depth)

    # Placeholder wave force (Morison's equation would be complex)
    F_wave = 0.5 * 1025 * 1.0 * 5.0 * wave_height * current_speed ** 2  # Very rough
    M_wave_mudline = F_wave * water_depth * 0.6  # Acting at some point above mudline

    return F_wind_thrust, M_wind_mudline, F_wave, M_wave_mudline


def check_structural_capacity(M_total_uls, pile_diameter, pile_thickness, steel_yield_strength):
    """Conceptual check for monopile structural capacity (bending)."""
    if pile_diameter == 0 or pile_thickness == 0: return "N/A", 0, 0
    OuterRadius = pile_diameter / 2
    InnerRadius = OuterRadius - pile_thickness
    I_pile = (np.pi / 4) * (OuterRadius ** 4 - InnerRadius ** 4)  # Second moment of area
    Z_pile = I_pile / OuterRadius  # Section modulus
    if Z_pile == 0: return "N/A", 0, 0
    max_bending_stress = M_total_uls / Z_pile
    allowable_stress = steel_yield_strength / 1.1  # Basic safety factor
    status = "PASS" if max_bending_stress <= allowable_stress else "FAIL"
    return status, max_bending_stress, allowable_stress


def check_geotechnical_lateral_capacity(M_mudline_uls, H_mudline_uls, pile_diameter, embedment_length, soil_type):
    """Conceptual check for lateral geotechnical capacity."""
    # Highly simplified - actual methods involve p-y curves
    if pile_diameter == 0 or embedment_length == 0: return "N/A", 0
    # Arbitrary capacity based on dimensions and soil type
    if soil_type == "Sand":
        capacity_factor = 50e3
    elif soil_type == "Clay":
        capacity_factor = 30e3
    else:
        capacity_factor = 10e3

    # Simplified resistance moment (very conceptual)
    R_moment_capacity = capacity_factor * pile_diameter * embedment_length ** 2 / 6
    status_moment = "PASS" if M_mudline_uls <= R_moment_capacity else "FAIL (Moment)"

    # Simplified resistance shear (very conceptual)
    R_shear_capacity = capacity_factor * pile_diameter * embedment_length / 2
    status_shear = "PASS" if H_mudline_uls <= R_shear_capacity else "FAIL (Shear)"

    if "FAIL" in status_moment or "FAIL" in status_shear:
        final_status = "FAIL"
    else:
        final_status = "PASS"

    return final_status, R_moment_capacity, R_shear_capacity


def calculate_pile_head_stiffness(E_soil, pile_diameter, embedment_length, EI_pile):
    """Conceptual pile head stiffness (lateral and rotational)."""
    # Extremely simplified - e.g., using empirical relations or simplified beam on elastic foundation
    if pile_diameter == 0 or embedment_length == 0 or EI_pile == 0: return 0, 0

    # Placeholder formulas (these are not standard engineering formulas, just illustrative)
    # Actual calculations are complex (e.g., Poulos & Davis, API guidelines)
    k_L = E_soil * pile_diameter * (embedment_length / pile_diameter) ** 0.5  # Lateral stiffness
    k_R = E_soil * pile_diameter ** 3 * (embedment_length / pile_diameter) ** 0.2  # Rotational stiffness

    # Ensure stiffness is not excessively large if E_soil is very high or dimensions are large
    k_L = min(k_L, EI_pile / (embedment_length / 3) ** 2 * 10)  # Cap lateral stiffness
    k_R = min(k_R, EI_pile / (embedment_length / 3) * 10)  # Cap rotational stiffness

    return k_L, k_R


def calculate_system_natural_frequency(EI_eff_system, M_RNA, M_tower_eff, H_eff_system, k_L_foundation, k_R_foundation):
    """Conceptual system natural frequency including soil-structure interaction."""
    if H_eff_system == 0: return 0

    # Fixed base frequency (recalculate for effective system)
    f_fixed = calculate_fixed_base_frequency(EI_eff_system, M_RNA, M_tower_eff, H_eff_system)
    if f_fixed == 0: return 0

    # Simplistic reduction factor due to foundation flexibility
    # This is a highly conceptual representation of SSI effects
    # In reality, one would use more complex models (e.g. equivalent spring models, dynamic impedance)

    # Effective stiffness of the structure
    K_struct_lateral = 3 * EI_eff_system / H_eff_system ** 3
    K_struct_rotational = EI_eff_system / H_eff_system  # Simplified

    if k_L_foundation == 0 or k_R_foundation == 0:  # Avoid division by zero if foundation is rigid
        return f_fixed

    # Flexibility method (conceptual)
    flex_lateral = 1 / K_struct_lateral + 1 / k_L_foundation
    flex_rotational = 1 / K_struct_rotational + 1 / k_R_foundation

    if flex_lateral == 0 or flex_rotational == 0: return f_fixed

    K_sys_lateral_inv = flex_lateral
    K_sys_rotational_inv = flex_rotational

    # Reduction factor based on ratio of system stiffness to fixed-base stiffness (conceptual)
    # This is a placeholder for more rigorous SSI analysis
    reduction_factor_L = (1 / K_sys_lateral_inv) / K_struct_lateral if K_struct_lateral > 0 else 1
    reduction_factor_R = (1 / K_sys_rotational_inv) / K_struct_rotational if K_struct_rotational > 0 else 1

    # Average reduction factor (highly simplified)
    overall_reduction = np.sqrt(min(reduction_factor_L, reduction_factor_R, 1.0))

    f_system = f_fixed * overall_reduction
    return f_system


# --- Streamlit App UI ---
st.set_page_config(layout="wide")
st.title("🚢 Conceptual 10-Step Monopile Design for OWT")
st.markdown("""
    **Disclaimer:** This app is for illustrative and educational purposes only. 
    It is based on a conceptual adaptation of the 10-step design philosophy presented for *jacket structures* in Jalbi & Bhattacharya (2020) to *monopiles*. 
    **The calculations are highly simplified placeholders and should NOT be used for actual engineering design.**
    Consult relevant standards (e.g., DNVGL, IEC) and specialized literature (e.g., Arany et al., 2017 for monopiles)
    for actual design.
""")

# --- Sidebar for Inputs ---
st.sidebar.header("⚙️ Global Inputs")

# Step 1: Design Basis & Input Data
st.sidebar.subheader("1. Turbine & Site Data")
rated_power = st.sidebar.number_input("Rated Power (MW)", min_value=1.0, value=5.0, step=0.1)
rotor_diameter = st.sidebar.number_input("Rotor Diameter (m)", min_value=50.0, value=126.0, step=1.0)
hub_height_msl = st.sidebar.number_input("Hub Height above MSL (m)", min_value=50.0, value=90.0, step=1.0)
mass_RNA = st.sidebar.number_input("Rotor-Nacelle Assembly Mass (tonnes)", min_value=50.0, value=350.0,
                                   step=10.0) * 1000  # to kg
mass_tower = st.sidebar.number_input("Tower Mass (tonnes)", min_value=100.0, value=350.0, step=10.0) * 1000  # to kg

water_depth = st.sidebar.number_input("Water Depth (m)", min_value=10.0, value=30.0, step=1.0)
# Metocean
st.sidebar.markdown("**Metocean (ULS):**")
uls_wind_speed = st.sidebar.number_input("ULS Rated Wind Speed (m/s)", min_value=10.0, value=11.4,
                                         step=0.1)  # From paper example
uls_sig_wave_height = st.sidebar.number_input("ULS 50-yr Sig. Wave Height, Hs (m)", min_value=1.0, value=8.27, step=0.1)
uls_wave_period = st.sidebar.number_input("ULS 50-yr Wave Period, Tp (s)", min_value=1.0, value=10.97, step=0.1)
uls_current_speed = st.sidebar.number_input("ULS Current Speed (m/s)", min_value=0.0, value=1.5, step=0.1)

st.sidebar.markdown("**Geotechnical:**")
soil_type = st.sidebar.selectbox("Dominant Soil Type", ["Sand", "Clay", "Layered (Simplified)"])
E_soil_avg = st.sidebar.number_input("Average Soil Young's Modulus, Es (MPa)", min_value=5.0, value=20.0,
                                     step=1.0) * 1e6  # Pa
steel_yield_strength = st.sidebar.number_input("Steel Yield Strength, fy (MPa)", value=355.0) * 1e6  # Pa
E_steel = 210e9  # Pa (Steel Young's Modulus)

st.sidebar.markdown("---")
st.sidebar.subheader("Target Frequencies")
rotor_freq_1P_min = st.sidebar.number_input("Min Rotor Frequency (1P) (Hz)", value=0.115)  # 6.9rpm/60
rotor_freq_1P_max = st.sidebar.number_input("Max Rotor Frequency (1P) (Hz)", value=0.202)  # 12.1rpm/60
rotor_freq_3P_min = rotor_freq_1P_min * 3
rotor_freq_3P_max = rotor_freq_1P_max * 3
st.sidebar.markdown(f"*3P Range: {rotor_freq_3P_min:.3f} - {rotor_freq_3P_max:.3f} Hz*")

# --- Main Page with 10 Steps ---

# Step 2: Initial Monopile Dimensions
with st.expander("STEP 2: Initial Monopile Dimensions", expanded=True):
    st.markdown("Guess initial dimensions for the monopile.")
    col1, col2, col3 = st.columns(3)
    pile_diameter_init = col1.number_input("Initial Pile Diameter, D (m)", min_value=3.0, value=6.0, step=0.5)
    pile_thickness_init = col2.number_input("Initial Pile Wall Thickness, t (mm)", min_value=20.0, value=60.0,
                                            step=5.0) / 1000  # m
    embedment_length_init = col3.number_input("Initial Embedment Length, L_e (m)", min_value=10.0, value=30.0, step=1.0)

    # Derived properties for calculations
    L_pile_above_mudline = water_depth
    H_tower = hub_height_msl - L_pile_above_mudline  # Assuming TP is at MSL or negligible height for this
    H_effective_structure = H_tower + L_pile_above_mudline  # From mudline to RNA

    st.write(f"Tower height above mudline: {H_tower:.2f} m")
    st.write(f"Effective structure height (mudline to RNA): {H_effective_structure:.2f} m")

# Step 3: Estimate Fixed-Base Natural Frequency
with st.expander("STEP 3: Estimate Fixed-Base Natural Frequency (Tower + Monopile Above Mudline)"):
    st.markdown(
        "Estimate the fixed-base natural frequency of the tower and monopile section above mudline to guide initial sizing.")
    # Simplified effective EI for tower + monopile section above mudline
    I_pile_init = (np.pi / 64) * (pile_diameter_init ** 4 - (pile_diameter_init - 2 * pile_thickness_init) ** 4)
    EI_pile_init = E_steel * I_pile_init

    # Assuming tower EI is similar or can be roughly estimated (very simplified)
    # A more detailed approach would consider tower taper and varying thickness
    D_tower_avg = pile_diameter_init * 0.8  # Assuming tower base matches pile, tapers
    t_tower_avg = pile_thickness_init * 0.5
    I_tower_avg = (np.pi / 64) * (
                D_tower_avg ** 4 - (D_tower_avg - 2 * t_tower_avg) ** 4) if D_tower_avg > 2 * t_tower_avg else 0
    EI_tower_avg = E_steel * I_tower_avg

    # Weighted average EI (conceptual)
    EI_eff_fixed_base = (EI_pile_init * L_pile_above_mudline + EI_tower_avg * H_tower) / (
                L_pile_above_mudline + H_tower) if (L_pile_above_mudline + H_tower) > 0 else EI_pile_init

    st.write(f"Monopile $I_{{init}}$: {I_pile_init:.4f} m⁴, $EI_{{init}}$: {EI_pile_init / 1e9:.2f} GNm²")
    st.write(f"Approx. Tower $EI_{{avg}}$: {EI_tower_avg / 1e9:.2f} GNm²")
    st.write(f"Effective $EI_{{fixed\_base}}$ (Tower + Pile above mudline): {EI_eff_fixed_base / 1e9:.2f} GNm²")

    f_fixed_base_init = calculate_fixed_base_frequency(EI_eff_fixed_base, mass_RNA, mass_tower, H_effective_structure)
    st.metric("Initial Fixed-Base Natural Frequency (Tower + Pile above mudline)", f"{f_fixed_base_init:.3f} Hz")
    st.markdown(f"""
    * 1P Rotor Frequency Range: {rotor_freq_1P_min:.3f} - {rotor_freq_1P_max:.3f} Hz
    * 3P Rotor Frequency Range: {rotor_freq_3P_min:.3f} - {rotor_freq_3P_max:.3f} Hz
    * Target: Typically soft-stiff (between 1P and 3P) or stiff-stiff (above 3P). Avoid resonance.
    """)
    if f_fixed_base_init > 0 and (
            rotor_freq_1P_min < f_fixed_base_init < rotor_freq_1P_max or rotor_freq_3P_min < f_fixed_base_init < rotor_freq_3P_max):
        st.warning("Initial fixed-base frequency is within a rotor harmonic range! Consider adjusting dimensions.")

# Step 4: Calculate Loads (Wind, Wave, Current) for ULS
with st.expander("STEP 4: Calculate Loads on Structure and Foundation (ULS)"):
    st.markdown("Calculate environmental loads (wind, wave, current) for Ultimate Limit State (ULS). Simplified.")
    F_wind_uls, M_wind_uls, F_wave_uls, M_wave_uls = calculate_loads(
        uls_wind_speed, uls_sig_wave_height, uls_current_speed, rotor_diameter, H_tower, water_depth
    )
    H_total_uls_mudline = F_wind_uls + F_wave_uls  # Total horizontal force at mudline
    M_total_uls_mudline = M_wind_uls + M_wave_uls  # Total overturning moment at mudline

    st.write(f"ULS Wind Thrust: {F_wind_uls / 1e3:.2f} kN, Mudline Moment from Wind: {M_wind_uls / 1e6:.2f} MNm")
    st.write(
        f"ULS Wave/Current Force (conceptual): {F_wave_uls / 1e3:.2f} kN, Mudline Moment from Wave/Current: {M_wave_uls / 1e6:.2f} MNm")
    st.metric("Total ULS Horizontal Force at Mudline (H_uls)", f"{H_total_uls_mudline / 1e3:.2f} kN")
    st.metric("Total ULS Overturning Moment at Mudline (M_uls)", f"{M_total_uls_mudline / 1e6:.2f} MNm")
    st.caption("Note: Dynamic Amplification Factors (DAFs) should be applied. Not included in this simplification.")

# Step 5: Structural Capacity of Monopile (ULS)
current_D = pile_diameter_init
current_t = pile_thickness_init
current_L_e = embedment_length_init

with st.expander("STEP 5: Check Structural Capacity of Monopile (ULS)"):
    st.markdown("Check if the monopile section can withstand the ULS bending moments.")
    st.write("Revise D and t if necessary:")
    col1, col2 = st.columns(2)
    current_D_rev_str = col1.number_input("Revised Pile Diameter, D (m) for Step 5", min_value=current_D,
                                          value=current_D, step=0.1, key="D_rev_str")
    current_t_rev_str = col2.number_input("Revised Pile Wall Thickness, t (mm) for Step 5",
                                          min_value=pile_thickness_init * 1000, value=current_t * 1000, step=1.0,
                                          key="t_rev_str") / 1000

    status_struct, stress_calc, stress_allow = check_structural_capacity(M_total_uls_mudline, current_D_rev_str,
                                                                         current_t_rev_str, steel_yield_strength)
    st.metric("Structural Check (Bending)", status_struct)
    st.write(f"Calculated Max Bending Stress: {stress_calc / 1e6:.2f} MPa")
    st.write(f"Allowable Bending Stress (fy/1.1): {stress_allow / 1e6:.2f} MPa")
    if status_struct == "FAIL":
        st.warning("Structural capacity failed. Increase diameter/thickness.")
    else:
        st.success("Structural capacity sufficient for ULS bending.")
    # Update effective D and t for subsequent steps if revised here
    if current_D_rev_str != current_D or current_t_rev_str != current_t:
        current_D = current_D_rev_str
        current_t = current_t_rev_str
        st.info(
            f"Monopile D and t updated to {current_D:.2f}m and {current_t * 1000:.0f}mm for subsequent steps based on Step 5 revisions.")

# Step 6: Geotechnical Capacity of Foundation (ULS)
with st.expander("STEP 6: Estimate Geotechnical Capacity of Foundation (ULS)"):
    st.markdown(
        "Estimate the ultimate lateral and moment capacity of the monopile foundation. Determine required embedment length.")
    st.write("Revise Embedment Length if necessary (based on D, t from Step 5):")
    current_L_e_rev_geo = st.number_input("Revised Embedment Length, L_e (m) for Step 6", min_value=current_L_e,
                                          value=current_L_e, step=0.5, key="L_e_rev_geo")

    status_geo, M_cap_geo, H_cap_geo = check_geotechnical_lateral_capacity(M_total_uls_mudline, H_total_uls_mudline,
                                                                           current_D, current_L_e_rev_geo, soil_type)
    st.metric("Geotechnical Lateral/Moment Check (ULS)", status_geo)
    st.write(
        f"Calculated Geotechnical Moment Capacity (conceptual): {M_cap_geo / 1e6:.2f} MNm (vs. Applied: {M_total_uls_mudline / 1e6:.2f} MNm)")
    st.write(
        f"Calculated Geotechnical Shear Capacity (conceptual): {H_cap_geo / 1e3:.2f} kN (vs. Applied: {H_total_uls_mudline / 1e3:.2f} kN)")

    if status_geo == "FAIL":
        st.warning("Geotechnical capacity failed. Increase embedment length or reconsider diameter.")
    else:
        st.success("Geotechnical capacity sufficient for ULS.")
    if current_L_e_rev_geo != current_L_e:
        current_L_e = current_L_e_rev_geo
        st.info(
            f"Monopile Embedment Length updated to {current_L_e:.2f}m for subsequent steps based on Step 6 revisions.")

    st.caption(
        "Note: Axial capacity (for self-weight and vertical loads) should also be checked. Typically less critical for monopiles than lateral.")

# Step 7: Foundation Stiffness & Deformations (SLS)
with st.expander("STEP 7: Calculate Foundation Stiffness and Estimate Deformations (SLS)"):
    st.markdown(
        "Calculate the monopile head stiffness (lateral $k_L$, rotational $k_R$) and estimate mudline deformations under Serviceability Limit State (SLS) loads.")
    st.caption("SLS loads are typically smaller than ULS. For simplicity, we'll use a fraction of ULS loads.")

    sls_load_factor = st.slider("SLS Load Factor (as % of ULS loads)", 0.1, 1.0, 0.7, 0.05)
    M_sls_mudline = M_total_uls_mudline * sls_load_factor
    H_sls_mudline = H_total_uls_mudline * sls_load_factor

    # Pile properties based on potentially revised D, t
    I_pile_current = (np.pi / 64) * (current_D ** 4 - (current_D - 2 * current_t) ** 4)
    EI_pile_current = E_steel * I_pile_current

    k_L, k_R = calculate_pile_head_stiffness(E_soil_avg, current_D, current_L_e, EI_pile_current)
    st.write(
        f"Monopile section for stiffness: D={current_D:.2f}m, t={current_t * 1000:.0f}mm, Le={current_L_e:.2f}m, EI={EI_pile_current / 1e9:.2f} GNm²")
    st.metric("Lateral Foundation Stiffness, k_L (conceptual)", f"{k_L / 1e6:.2f} MN/m")
    st.metric("Rotational Foundation Stiffness, k_R (conceptual)", f"{k_R / 1e9:.2f} GNm/rad")

    if k_L > 0 and k_R > 0:
        mudline_deflection_lat = H_sls_mudline / k_L
        mudline_rotation = M_sls_mudline / k_R  # in radians

        # Tip deflection (very simplified - assumes rigid rotation contribution dominates)
        # RNA_tip_deflection_approx = mudline_deflection_lat + mudline_rotation * H_effective_structure # Simplified
        # More accurate would be: deflection_shear + deflection_bending_pile + deflection_bending_tower + rotation_foundation * H_eff
        # This is super simplified:
        RNA_tip_deflection_approx = (M_sls_mudline * H_effective_structure / k_R) + (
                    H_sls_mudline * H_effective_structure ** 2 / (
                        2 * EI_eff_fixed_base)) if EI_eff_fixed_base > 0 else 0

        st.metric("Est. Mudline Lateral Deflection (SLS)", f"{mudline_deflection_lat * 1000:.2f} mm")
        st.metric("Est. Mudline Rotation (SLS)", f"{mudline_rotation * 180 / np.pi:.4f} degrees")  # To degrees
        st.metric("Est. RNA Tip Deflection (SLS - very approx.)", f"{RNA_tip_deflection_approx:.3f} m")
        st.caption(
            "Deformations should be checked against manufacturer/project limits (e.g., <0.5 deg rotation, specific tip deflection).")
    else:
        st.warning("Cannot calculate deformations due to zero stiffness values.")

# Step 8: Calculate System Natural Frequency (SLS)
with st.expander("STEP 8: Calculate System Natural Frequency (Including SSI)"):
    st.markdown(
        "Calculate the natural frequency of the entire OWT system, including soil-structure interaction (SSI) effects from foundation stiffness.")

    # Effective EI of the whole system (tower + pile above mudline). Using fixed_base one for simplicity.
    EI_eff_system = EI_eff_fixed_base
    H_eff_system = H_effective_structure  # Total height from mudline to RNA

    f_system_ssi = calculate_system_natural_frequency(EI_eff_system, mass_RNA, mass_tower, H_eff_system, k_L, k_R)
    st.metric("System Natural Frequency (with SSI - conceptual)", f"{f_system_ssi:.3f} Hz")

    st.markdown(f"""
    * Reference Fixed-Base Frequency (Step 3): {f_fixed_base_init:.3f} Hz
    * 1P Rotor Frequency Range: {rotor_freq_1P_min:.3f} - {rotor_freq_1P_max:.3f} Hz
    * 3P Rotor Frequency Range: {rotor_freq_3P_min:.3f} - {rotor_freq_3P_max:.3f} Hz
    * Typical Wave Freq. Range (example): 0.05 - 0.25 Hz (not from input, just illustrative)
    """)

    freq_ok = True
    if f_system_ssi > 0:
        if rotor_freq_1P_min < f_system_ssi < rotor_freq_1P_max:
            st.warning(f"System frequency ({f_system_ssi:.3f} Hz) is within 1P range! Potential resonance.")
            freq_ok = False
        if rotor_freq_3P_min < f_system_ssi < rotor_freq_3P_max:
            st.warning(f"System frequency ({f_system_ssi:.3f} Hz) is within 3P range! Potential resonance.")
            freq_ok = False
        # Add check for wave frequency if available/relevant
        # Example: if 0.05 < f_system_ssi < 0.25:
        #     st.warning("System frequency might be close to typical wave energy peak!")
        #     freq_ok = False
        if freq_ok:
            st.success("System frequency appears to be outside critical rotor harmonics based on these inputs.")
    else:
        st.error("System frequency could not be calculated.")
    st.caption("If frequency is not acceptable, iterate on monopile dimensions (D, t, Le) or tower properties.")

# Step 9: Long-term Performance (Placeholder)
with st.expander("STEP 9: Check Long-Term Performance (e.g., Scour, Stiffness Degradation)"):
    st.markdown("""
    This step involves assessing changes over the lifetime of the structure. For monopiles, key aspects include:
    * **Scour:** Erosion of soil around the pile head, reducing effective embedment and stiffness. Requires scour protection or designing for scour depth.
    * **Soil Stiffness Degradation/Cyclic Effects:** Cyclic loading can alter soil stiffness, impacting natural frequency and long-term deformations.
    * **Material Degradation:** Corrosion of steel.

    **These are advanced checks requiring detailed analysis and are beyond this conceptual app.**
    A simple approach might be to re-evaluate with reduced soil stiffness or effective embedment.
    """)
    scour_depth_allowance = st.number_input("Scour Depth Allowance (m)", value=max(1.0, 0.1 * current_D), min_value=0.0,
                                            step=0.1)
    effective_embedment_scour = current_L_e - scour_depth_allowance
    st.write(
        f"Effective embedment length after scour: {effective_embedment_scour:.2f} m (if {scour_depth_allowance:.2f}m scour)")

    if effective_embedment_scour <= 0:
        st.error("Scour allowance exceeds embedment length!")
    else:
        # Recalculate stiffness with scoured embedment (conceptual)
        k_L_scour, k_R_scour = calculate_pile_head_stiffness(E_soil_avg, current_D, effective_embedment_scour,
                                                             EI_pile_current)
        f_system_ssi_scour = calculate_system_natural_frequency(EI_eff_system, mass_RNA, mass_tower, H_eff_system,
                                                                k_L_scour, k_R_scour)

        st.write(f"Conceptual k_L with scour: {k_L_scour / 1e6:.2f} MN/m (Original: {k_L / 1e6:.2f} MN/m)")
        st.write(f"Conceptual k_R with scour: {k_R_scour / 1e9:.2f} GNm/rad (Original: {k_R / 1e9:.2f} GNm/rad)")
        st.write(f"Conceptual System Freq. with scour: {f_system_ssi_scour:.3f} Hz (Original: {f_system_ssi:.3f} Hz)")
        if abs(f_system_ssi_scour - f_system_ssi) > 0.01 * f_system_ssi and f_system_ssi > 0:  # More than 1% change
            st.warning("Scour significantly impacts system frequency. Re-check resonance.")

# Step 10: Estimate Fatigue Life (Placeholder)
with st.expander("STEP 10: Estimate Fatigue Life (FLS)"):
    st.markdown("""
    Fatigue Limit State (FLS) is often the governing design driver for monopiles due to cyclic wind and wave loading.
    This involves:
    * Defining long-term load spectra (scatter diagrams for wind/wave).
    * Calculating stress ranges at critical locations (e.g., welds, mudline).
    * Using S-N curves (Stress-Number of cycles) to estimate fatigue damage.
    * Applying Miner's rule for cumulative damage.

    **Fatigue analysis is highly complex and requires specialized software and detailed input. It is beyond the scope of this conceptual app.**
    A very rough indicator might be to check stress ranges under operational conditions against fatigue endurance limits, but this is not a substitute for proper FLS analysis.
    """)
    st.info(
        "Fatigue life estimation requires detailed time-domain simulations or spectral methods and is not implemented here.")

st.markdown("---")
st.header("🏁 Final Conceptual Dimensions (Iterate if needed)")
st.write(f"Monopile Diameter (D): {current_D:.2f} m")
st.write(f"Monopile Wall Thickness (t): {current_t * 1000:.0f} mm")
st.write(f"Monopile Embedment Length (L_e): {current_L_e:.2f} m")
st.write(f"Resulting System Natural Frequency (conceptual): {f_system_ssi:.3f} Hz")

st.markdown("---")
st.markdown(
    "Reminder: This is a conceptual tool. Real design involves many iterations, more detailed calculations, adherence to codes (DNVGL-ST-0126, IEC 61400-3 etc.), and specialized software.")
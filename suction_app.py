import streamlit as st
import numpy as np
import matplotlib.pyplot as plt


# --- Helper Functions for Calculations ---
# These are highly simplified examples. Refer to the OWA document for detailed formulas.

def calculate_self_weight_penetration(caisson_weight_submerged, tip_resistance_initial, side_friction_initial):
    """
    Highly simplified SWP calculation.
    In reality, this involves iterative calculation of resistance vs. depth.
    """
    # This is a placeholder logic.
    if tip_resistance_initial + side_friction_initial > 0:
        swp = caisson_weight_submerged / (tip_resistance_initial + side_friction_initial)  # Simplistic
        return min(swp, 5)  # Assume max 5m SWP for this demo
    return 0.1  # Minimal penetration if no resistance data


def calculate_required_suction_clay(V_prime, D_o, D_i, h_penetration, s_u1, alpha_o, alpha_i, N_c_tip, s_u2, t_skirt,
                                    gamma_prime_soil):
    """
    Simplified calculation for required suction in clay during installation.
    Based on rearranging a simplified version of concepts related to Eq. 6-2 from OWA guidelines.
    V_prime: Effective vertical load (submerged weight of caisson + structure)
    s_u1: Average undrained strength over skirt depth
    s_u2: Undrained strength at skirt tip
    This function calculates suction 's' required for a given penetration 'h_penetration'.
    """
    A_caisson_internal = np.pi * (D_i ** 2) / 4

    # Resistance components (simplified)
    R_outside_friction = alpha_o * np.pi * D_o * h_penetration * s_u1
    R_inside_friction = alpha_i * np.pi * D_i * h_penetration * s_u1

    # Tip resistance (simplified, using Nc for bearing capacity factor at tip)
    # Effective area of the skirt tip (annulus)
    A_tip = np.pi * (D_o ** 2 - D_i ** 2) / 4
    # More accurately, it's (pi * D_avg * t_skirt) * Nc * su2
    # For simplicity, let's use a simplified version of tip resistance from general bearing capacity concepts
    # R_tip = A_tip * N_c_tip * s_u2 # This would be for bearing capacity

    # From Eq 6-2: V' + s * A_caisson = R_inside + R_outside + R_tip
    # R_tip for penetration resistance is more like (gamma_prime_soil*h + Nc*s_u2)*(pi*D_avg*t)
    # Let's use a simplified tip resistance term for penetration
    D_avg = (D_o + D_i) / 2
    R_tip = (gamma_prime_soil * h_penetration + N_c_tip * s_u2) * (np.pi * D_avg * t_skirt)

    if A_caisson_internal > 0:
        suction = (R_outside_friction + R_inside_friction + R_tip - V_prime) / A_caisson_internal
        return max(0, suction)  # Suction cannot be negative
    return 0


def calculate_uls_vertical_capacity_clay(D_o, h_embedment, s_u_base, N_c_factor, gamma_prime_soil, alpha_side,
                                         s_u_side_avg):
    """
    Simplified ULS vertical bearing capacity in clay (compression).
    Based on concepts from Appendix C (e.g., related to Eq. C.15 for base, C.4 for side).
    Assumes full contact and undrained conditions.
    Ignores shape, depth, inclination factors for simplicity in this demo.
    """
    # Base resistance (simplified from Terzaghi/Hansen)
    A_base = np.pi * (D_o ** 2) / 4  # Assuming D_o is effective diameter at base
    q_base_ultimate = N_c_factor * s_u_base + gamma_prime_soil * h_embedment  # Bearing capacity at base
    R_base = A_base * q_base_ultimate

    # Side friction (external)
    A_side_external = np.pi * D_o * h_embedment
    R_side_external = A_side_external * alpha_side * s_u_side_avg

    # Total vertical capacity (compression)
    V_ult_c = R_base + R_side_external
    return V_ult_c


# --- Streamlit App Layout ---
st.set_page_config(layout="wide", page_title="Suction Caisson Preliminary Design")

st.title("🌊 Offshore Suction Caisson Preliminary Design Tool")
st.markdown("""
    **Disclaimer:** This tool is for illustrative and educational purposes only, based on simplified interpretations of the
    *Offshore Wind Accelerator: Suction Installed Caisson Foundations for Offshore Wind: Design Guidelines*.
    It is **NOT** a substitute for detailed engineering design, analysis, and professional judgment.
    Always refer to the original OWA document and relevant design standards.
""")

# --- Sidebar for Inputs ---
with st.sidebar:
    st.header("⚙️ Input Parameters")

    st.subheader("1. Project & Site Information")
    project_name = st.text_input("Project Name", "Demo Project")
    water_depth = st.number_input("Water Depth (m, $h_w$)", min_value=5.0, value=30.0, step=1.0)

    st.subheader("2. Soil Properties (Simplified - Single Layer)")
    soil_type = st.selectbox("Dominant Soil Type", ["Clay", "Sand (Not fully implemented)"])

    if soil_type == "Clay":
        s_u_avg = st.number_input("Average Undrained Shear Strength over skirt (kPa, $s_{u1}$)", min_value=5.0,
                                  value=50.0, step=5.0)
        s_u_tip = st.number_input("Undrained Shear Strength at skirt tip (kPa, $s_{u2}$)", min_value=5.0, value=60.0,
                                  step=5.0)
        alpha_o = st.number_input("Adhesion Factor (Outer skirt, $\\alpha_o$)", min_value=0.1, max_value=1.0, value=0.5,
                                  step=0.1)
        alpha_i = st.number_input("Adhesion Factor (Inner skirt, $\\alpha_i$)", min_value=0.1, max_value=1.0, value=0.5,
                                  step=0.1)
        gamma_eff_soil = st.number_input("Effective Unit Weight of Soil (kN/m³ $\\gamma'$)", min_value=4.0, value=8.0,
                                         step=0.5)  # Submerged
        N_c_bearing = st.number_input("Bearing Capacity Factor $N_c$ (for ULS base)",
                                      value=5.14)  # For shallow foundations
        N_c_install_tip = st.number_input("Bearing Capacity Factor $N_c$ (for installation tip resistance)",
                                          value=9.0)  # Often higher for piles/caissons
    else:  # Sand (placeholders)
        st.info("Sand calculations are not fully implemented in this demo.")
        phi_prime = st.number_input("Angle of Friction (degrees, $\\phi'$)", value=30.0)
        gamma_eff_soil = st.number_input("Effective Unit Weight of Soil (kN/m³ $\\gamma'$)", value=10.0)
        K_sand = st.number_input("Coefficient of Lateral Earth Pressure (K)", value=0.8)
        delta_sand = st.number_input("Interface Friction Angle (degrees, $\\delta$)", value=25.0)

    st.subheader("3. Suction Caisson Geometry")
    D_o = st.number_input("Outer Diameter (m, $D_o$)", min_value=1.0, value=10.0, step=0.5)
    skirt_thickness = st.number_input("Skirt Wall Thickness (m, t)", min_value=0.02, value=0.05, step=0.01)
    D_i = D_o - 2 * skirt_thickness
    st.markdown(f"Inner Diameter ($D_i$): {D_i:.2f} m")
    L_skirt = st.number_input("Skirt Length/Embedment (m, L or h)", min_value=1.0, value=8.0, step=0.5)

    st.subheader("4. Loading (Simplified)")
    V_eff_structure = st.number_input("Effective Vertical Load from Structure (Submerged Weight, kN, $V'$)",
                                      min_value=100.0, value=5000.0, step=100.0)
    # For ULS, loads would be factored. For installation, it's the weight driving penetration.

    st.subheader("5. Safety Factors (LRFD)")
    gamma_f_load = st.number_input("Partial Safety Factor for Loads (ULS, $\\gamma_f$)", value=1.35)
    gamma_m_material_undrained = st.number_input("Partial Safety Factor for Material (Undrained ULS, $\\gamma_m$)",
                                                 value=1.25)

# --- Main Panel with Tabs ---
tab1, tab2, tab3 = st.tabs(
    ["📝 Inputs Summary", "🛠️ Installation Design (Simplified)", "🛡️ In-Service Design (ULS - Simplified)"])

with tab1:
    st.header("Input Parameters Summary")
    st.write(f"**Project:** {project_name}")
    st.write(f"**Water Depth:** {water_depth} m")
    st.write(f"**Dominant Soil Type:** {soil_type}")
    if soil_type == "Clay":
        st.write(f"  - Avg. Undrained Shear Strength ($s_{u1}$): {s_u_avg} kPa")
        st.write(f"  - Tip Undrained Shear Strength ($s_{u2}$): {s_u_tip} kPa")
        st.write(f"  - Adhesion Factor (Outer, $\\alpha_o$): {alpha_o}")
        st.write(f"  - Adhesion Factor (Inner, $\\alpha_i$): {alpha_i}")
    st.write(f"  - Effective Soil Unit Weight ($\\gamma'$): {gamma_eff_soil} kN/m³")
    st.write(f"**Caisson Geometry:**")
    st.write(f"  - Outer Diameter ($D_o$): {D_o} m")
    st.write(f"  - Inner Diameter ($D_i$): {D_i:.2f} m")
    st.write(f"  - Skirt Thickness (t): {skirt_thickness} m")
    st.write(f"  - Skirt Length/Embedment (L): {L_skirt} m")
    st.write(f"**Loading:**")
    st.write(f"  - Effective Vertical Load ($V'$): {V_eff_structure} kN")
    st.write(f"**Safety Factors (ULS):**")
    st.write(f"  - Load Factor ($\\gamma_f$): {gamma_f_load}")
    st.write(f"  - Material Factor (Undrained, $\\gamma_m$): {gamma_m_material_undrained}")

with tab2:
    st.header("Installation Design (Simplified)")

    if soil_type == "Clay":
        # Self-Weight Penetration (highly conceptual for demo)
        # For a real calculation, you'd need initial tip and side resistance values.
        swp_estimate = calculate_self_weight_penetration(V_eff_structure, s_u_tip * 0.5,
                                                         s_u_avg * 0.1)  # Dummy resistances
        st.subheader("Self-Weight Penetration (SWP)")
        st.write(f"Estimated SWP (Conceptual): {swp_estimate:.2f} m")
        st.markdown("> *Note: This SWP is a very rough estimate. Detailed analysis is required.*")

        st.subheader("Required Suction Pressure (Clay)")
        penetration_depths = np.linspace(max(0.1, swp_estimate), L_skirt, 20)  # Start from after SWP
        required_suctions = []

        if D_i <= 0:
            st.error("Inner diameter must be greater than 0. Check caisson dimensions.")
        else:
            for h_pen in penetration_depths:
                suction = calculate_required_suction_clay(
                    V_prime=V_eff_structure,
                    D_o=D_o, D_i=D_i,
                    h_penetration=h_pen,
                    s_u1=s_u_avg, alpha_o=alpha_o, alpha_i=alpha_i,
                    N_c_tip=N_c_install_tip,  # Using Nc for tip resistance during installation
                    s_u2=s_u_tip,
                    t_skirt=skirt_thickness,
                    gamma_prime_soil=gamma_eff_soil
                )
                required_suctions.append(suction)

            fig, ax = plt.subplots()
            ax.plot(required_suctions, penetration_depths)
            ax.set_xlabel("Required Suction (kPa)")
            ax.set_ylabel("Penetration Depth (m)")
            ax.set_title("Required Suction vs. Penetration Depth (Clay)")
            ax.invert_yaxis()  # Depth increases downwards
            ax.grid(True)
            st.pyplot(fig)

            st.markdown(f"**Max Required Suction (at L={L_skirt:.2f}m): {required_suctions[-1]:.2f} kPa**")

            # Cavitation Limit (Simplified)
            p_atm = 101.3  # kPa
            p_vapor = 2.3  # kPa (approx for water at 20C)
            gamma_water = 9.81  # kN/m3
            # Absolute pressure at caisson top inside: p_atm + gamma_water * water_depth - suction
            # Cavitation if this pressure drops to p_vapor
            s_cavitation_limit = p_atm + (gamma_water * water_depth) - p_vapor
            st.markdown(f"**Indicative Cavitation Limit (Suction): {s_cavitation_limit:.2f} kPa**")
            if required_suctions[-1] > s_cavitation_limit:
                st.warning("Required suction may exceed cavitation limit!")

            st.markdown("""
                **Notes for Installation Design:**
                - This calculation is simplified. Refer to OWA Section 6 for detailed methods (mechanism-based, CPT-based).
                - Factors like limits to suction (pump capacity, buckling, piping, plug uplift) are critical and need detailed assessment.
                - Layered soils significantly complicate installation analysis.
            """)
    else:
        st.info("Installation design for sand is not implemented in this demo. Refer to OWA Section 6.2.1.2 and 6.2.2.")

with tab3:
    st.header("In-Service Design (ULS - Simplified)")

    if soil_type == "Clay":
        st.subheader("Ultimate Vertical Compressive Capacity (Clay - Undrained)")

        # Characteristic resistance
        V_ult_c_char = calculate_uls_vertical_capacity_clay(
            D_o=D_o,
            h_embedment=L_skirt,
            s_u_base=s_u_tip,  # Assuming s_u at tip is representative for base capacity
            N_c_factor=N_c_bearing,
            gamma_prime_soil=gamma_eff_soil,
            alpha_side=alpha_o,  # Assuming outer friction dominates
            s_u_side_avg=s_u_avg
        )
        st.write(f"Characteristic Vertical Compressive Capacity ($V_{{ult,c,char}}$): {V_ult_c_char:.2f} kN")

        # Design resistance
        # Rd = Rk / gamma_m
        V_ult_c_design = V_ult_c_char / gamma_m_material_undrained
        st.write(
            f"Design Vertical Compressive Capacity ($V_{{ult,c,design}} = R_k / \\gamma_m$): {V_ult_c_design:.2f} kN")

        # Design load effect
        # Sd = Sk * gamma_f
        V_design_effect = V_eff_structure * gamma_f_load  # Assuming V_eff_structure is the characteristic permanent load
        st.write(f"Design Vertical Load Effect ($S_d = S_k \\times \\gamma_f$): {V_design_effect:.2f} kN")

        # Check ULS
        if V_design_effect <= V_ult_c_design:
            st.success(f"ULS Check for Vertical Compression: PASSED ($S_d \\leq R_d$)")
            st.metric(label="Demand/Capacity Ratio (DCR)", value=f"{V_design_effect / V_ult_c_design:.2f}", delta="OK",
                      delta_color="normal")
        else:
            st.error(f"ULS Check for Vertical Compression: FAILED ($S_d > R_d$)")
            st.metric(label="Demand/Capacity Ratio (DCR)", value=f"{V_design_effect / V_ult_c_design:.2f}",
                      delta="High", delta_color="inverse")

        st.markdown("""
            **Notes for In-Service Design (ULS):**
            - This is a highly simplified check for vertical compression in clay.
            - OWA Section 8 and Appendix C provide detailed methods for VHM capacity, drained/undrained analysis, layering, etc.
            - Horizontal, Moment, Tension capacities, and combined loading (VHM envelopes) are critical.
            - SLS (stiffness, deformation) and FLS also need to be assessed.
            - Cyclic loading effects can be significant.
        """)
    else:
        st.info("In-service design for sand is not implemented in this demo. Refer to OWA Section 8 and Appendix C.")

st.sidebar.markdown("---")
st.sidebar.info("Developed based on OWA Suction Caisson Design Guidelines (Feb 2019).")


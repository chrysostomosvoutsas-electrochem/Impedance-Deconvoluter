import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# -----------------------------------------------------------------------------
# PAGE SETUP & STYLESHEET
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="Impedance-Deconvoluter | Physical Electrochemistry Hub",
    page_icon="⚡",
    layout="wide"
)

st.markdown("""
<style>
    .metric-card {
        background-color: #1e2430;
        border-radius: 8px;
        padding: 12px;
        border: 1px solid #2d3748;
        margin-bottom: 10px;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }
    .stTabs [data-baseweb="tab"] {
        padding-top: 10px;
        padding-bottom: 10px;
        font-weight: 600;
    }
</style>
""", unsafe_allow_html=True)

st.title("⚡ Impedance-Deconvoluter: From Equivalent Circuits to Physical Electrochemistry")
st.caption("A rigorous educational and diagnostic suite grounded in Bard & Faulkner, Gileadi, and solid-state battery electrochemistry.")

# Top Navigation Tabs
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "🔬 1. Randles Circuit & Frequency Deconvoluter",
    "⏱️ 2. The Time-Domain vs. Frequency-Domain Bridge",
    "🌊 3. Diffusion & The Warburg Impedance (Zw)",
    "📐 4. Cell Geometry & Luggin Capillary Sandbox",
    "🔋 5. Solid-State Pellet & Composite Normalizer"
])

# =============================================================================
# TAB 1: THE RANDLES CIRCUIT & PHYSICAL DECONVOLUTION
# =============================================================================
with tab1:
    st.subheader("1. The Classical Randles Interface: Parallel Pathways & Semicircle Mechanics")
    st.markdown("""
    **Physical Principle (Bard & Faulkner §1.2.2 & §10.1; Gileadi §1.2.2 & §16.1):**  
    When electric charge enters an electrode-electrolyte interface, it can take only two parallel routes:
    1. **Non-Faradaic ($C_{dl}$):** Charging the double layer without crossing the boundary.
    2. **Faradaic ($R_{ct}$):** Transferring electrons across the interface to drive a chemical redox reaction.
    Because charge can take either path simultaneously, **$C_{dl}$ and $R_{ct}$ are placed in parallel**. 
    All ions moving through the bulk electrolyte must overcome its ohmic resistance, so the solution resistance **$R_s$ is in series** with the interface.
    """)

    col_t1_ctrl, col_t1_plots = st.columns([1, 2.2], gap="large")

    with col_t1_ctrl:
        st.markdown("#### Interface Circuit Parameters")
        rs_val = st.slider("Solution Resistance ($R_s$, $\\Omega$):", 1.0, 500.0, 50.0, 1.0)
        rct_val = st.slider("Charge-Transfer Resistance ($R_{ct}$, $\\Omega$):", 10.0, 2500.0, 450.0, 10.0)
        cdl_val_uf = st.slider("Double-Layer Capacitance ($C_{dl}$, $\\mu\\text{F}$):", 0.1, 100.0, 10.0, 0.5)
        cdl_val = cdl_val_uf * 1e-6

        use_cpe = st.checkbox("Model non-ideal surface roughness with a Constant Phase Element (CPE)?", value=False)
        if use_cpe:
            n_cpe = st.slider("CPE Exponent ($n$):", 0.50, 1.00, 0.85, 0.01,
                              help="n=1.0 is an ideal flat capacitor; n=0.5 corresponds to a porous transmission line or rough surface.")
        else:
            n_cpe = 1.0

        st.markdown("---")
        st.markdown("#### Frequency Probe Control")
        log_f_sel = st.slider("Interactive Probe Frequency $\\log_{10}(f)$ [Hz]:", -2.0, 6.0, 1.55, 0.05)
        f_probe = 10 ** log_f_sel
        w_probe = 2 * np.pi * f_probe

        # Time constant and characteristic frequency
        tau_val = rct_val * cdl_val
        w_max = 1.0 / tau_val
        f_max = w_max / (2 * np.pi)

        st.markdown(f"""
        <div class="metric-card">
            <b>Interfacial Time Constant ($\\tau = R_{{ct}}C_{{dl}}$):</b><br>
            <code>{tau_val*1e3:.3f} ms</code><br><br>
            <b>Peak Semicircle Frequency ($f_{{apex}} = 1/2\\pi\\tau$):</b><br>
            <code>{f_max:.2f} Hz</code> ($\\omega = {w_max:.1f}\\ \\text{{rad/s}}$)<br><br>
            <b>Apex Imaginary Impedance ($-Z''_{{max}}$):</b><br>
            <code>{rct_val/2.0:.1f} \\Omega</code>
        </div>
        """, unsafe_allow_html=True)

    with col_t1_plots:
        # Generate Frequency Sweep
        freqs = np.logspace(-2, 6, 400)
        omega = 2 * np.pi * freqs

        if not use_cpe:
            # Standard Randles: Z = Rs + Rct / (1 + j*w*Rct*Cdl)
            denom = 1 + (omega * rct_val * cdl_val) ** 2
            z_real = rs_val + (rct_val / denom)
            z_imag = (omega * rct_val**2 * cdl_val) / denom
        else:
            # CPE Impedance: Z_cpe = 1 / [Q * (j*w)^n]
            # Parallel: Y_parallel = 1/Rct + Q * (j*w)^n
            q_val = cdl_val  # effective magnitude
            y_parallel = (1.0 / rct_val) + q_val * (omega**n_cpe) * (np.cos(n_cpe * np.pi / 2) + 1j * np.sin(n_cpe * np.pi / 2))
            z_parallel = 1.0 / y_parallel
            z_real = rs_val + z_parallel.real
            z_imag = -z_parallel.imag

        # Calculate Single Probe Point
        if not use_cpe:
            d_p = 1 + (w_probe * rct_val * cdl_val) ** 2
            zp_real = rs_val + (rct_val / d_p)
            zp_imag = (w_probe * rct_val**2 * cdl_val) / d_p
        else:
            yp = (1.0 / rct_val) + q_val * (w_probe**n_cpe) * (np.cos(n_cpe * np.pi / 2) + 1j * np.sin(n_cpe * np.pi / 2))
            zp = 1.0 / yp
            zp_real = rs_val + zp.real
            zp_imag = -zp.imag

        # Semicircle Nyquist & Bode plots
        fig_t1 = make_subplots(
            rows=1, cols=2,
            subplot_titles=("Nyquist Plot ($Z'$ vs. $-Z''$)", "Bode Plot (Phase Angle $\\theta$ vs. Frequency)")
        )

        # 1. Nyquist
        fig_t1.add_trace(go.Scatter(
            x=z_real, y=z_imag, mode='lines',
            line=dict(color='#00d2ff', width=3),
            name="Impedance Arc"
        ), row=1, col=1)

        fig_t1.add_trace(go.Scatter(
            x=[zp_real], y=[zp_imag], mode='markers+text',
            marker=dict(size=14, color='#f1c40f', symbol='diamond'),
            text=[f"f = {f_probe:.1f} Hz"], textposition="top right",
            name="Selected Frequency"
        ), row=1, col=1)

        # 2. Bode Phase
        phase_deg = np.degrees(np.arctan2(z_imag, z_real - rs_val))
        probe_phase = np.degrees(np.arctan2(zp_imag, zp_real - rs_val))

        fig_t1.add_trace(go.Scatter(
            x=freqs, y=phase_deg, mode='lines',
            line=dict(color='#2ecc71', width=2.5),
            name="Phase Shift $\\theta$"
        ), row=1, col=2)

        fig_t1.add_trace(go.Scatter(
            x=[f_probe], y=[probe_phase], mode='markers',
            marker=dict(size=12, color='#f1c40f', symbol='diamond'),
            name="Probe Phase"
        ), row=1, col=2)

        fig_t1.update_xaxes(title_text="Real Impedance Z' (\\Omega)", row=1, col=1)
        fig_t1.update_yaxes(title_text="Negative Imaginary Impedance -Z'' (\\Omega)", scaleanchor="x", scaleratio=1, row=1, col=1)
        fig_t1.update_xaxes(title_text="Frequency (Hz)", type="log", row=1, col=2)
        fig_t1.update_yaxes(title_text="Phase Angle (degrees)", range=[0, 95], row=1, col=2)

        fig_t1.update_layout(
            template="plotly_dark",
            height=420,
            margin=dict(l=40, r=40, t=40, b=20),
            showlegend=False
        )
        st.plotly_chart(fig_t1, use_container_width=True)

        # Current Split Analysis at Selected Frequency
        # Z_c = 1 / (j*w*Cdl); Z_rct = Rct
        z_c_mag = 1.0 / (w_probe * cdl_val)
        # Fractional current through Cdl vs Rct:
        # I_c / I_total = Z_rct / (Z_c + Z_rct)
        i_c_frac = rct_val / np.sqrt(rct_val**2 + z_c_mag**2)
        i_r_frac = z_c_mag / np.sqrt(rct_val**2 + z_c_mag**2)

        st.markdown(f"""
        **Current Distribution at current frequency ($f = {f_probe:.2f}\\ \\text{{Hz}}$):**
        * **Capacitive Route ($C_{{dl}}$):** `{i_c_frac*100.0:.1f}%` of the AC current is charging/discharging the double layer.
        * **Faradaic Route ($R_{{ct}}$):** `{i_r_frac*100.0:.1f}%` of the AC current is transferring electrons across the interface.
        * **Physical Interpretation:** At high frequencies, the double layer acts as an electrical bypass (short), leaving only $R_s$. At low frequencies, the double layer blocks charge, forcing current across $R_{ct}$ ($Z' \\rightarrow R_s + R_{ct}$).
        """)

# =============================================================================
# TAB 2: TIME-DOMAIN VS FREQUENCY-DOMAIN BRIDGE
# =============================================================================
with tab2:
    st.subheader("2. The Unification Bridge: One Interface under Step, Ramp, and AC Perturbations")
    st.markdown("""
    **Pedagogical Truth (Bard & Faulkner §1.2.4 & Figures 1.2.6–1.2.11; Gileadi §4.2.2):**  
    Chronoamperometry, cyclic voltammetry background, and Nyquist semicircles are **not** different physical phenomena. 
    They are simply different mathematical boundary conditions applied to the **same $R_s - [C_{dl} \\parallel R_{ct}]$ interface**.
    """)

    col_t2_p, col_t2_g = st.columns([1, 2.2], gap="large")

    with col_t2_p:
        st.markdown("#### Universal Interface Parameters")
        u_rs = st.slider("Electrolyte Resistance ($R_s$, $\\Omega$):", 5.0, 200.0, 30.0, 5.0, key="u_rs")
        u_rct = st.slider("Charge-Transfer Resistance ($R_{ct}$, $\\Omega$):", 50.0, 1000.0, 350.0, 25.0, key="u_rct")
        u_cdl_uf = st.slider("Double-Layer Capacitance ($C_{dl}$, $\\mu\\text{F}$):", 1.0, 50.0, 15.0, 1.0, key="u_cdl")
        u_cdl = u_cdl_uf * 1e-6

        st.markdown("#### Excitation Settings")
        v_step_mv = st.slider("Potential Step Amplitude (mV):", 10, 250, 100, 10)
        scan_rate_vs = st.select_slider("CV Scan Rate ($v$, V/s):", options=[0.02, 0.05, 0.1, 0.5, 1.0, 5.0, 20.0], value=0.1)

    with col_t2_g:
        fig_bridge = make_subplots(
            rows=2, cols=2,
            subplot_titles=(
                "A. Potential Step: Chronoamperometry i(t)",
                "B. Linear Sweep: Non-Faradaic Background i(t)",
                "C. Cyclic Sweep: Hysteresis Window i(V)",
                "D. AC Small-Signal: Nyquist Semicircle Z(ω)"
            )
        )

        # 1. Potential Step i(t) = (E/Rs) * exp(-t / Rs*Cdl) + E / (Rs + Rct)
        tau_s = u_rs * u_cdl
        t_arr = np.linspace(0, 5 * tau_s, 300)
        v_step = v_step_mv * 1e-3
        i_step = (v_step / u_rs) * np.exp(-t_arr / tau_s) + (v_step / (u_rs + u_rct)) * (1 - np.exp(-t_arr / tau_s))

        fig_bridge.add_trace(go.Scatter(
            x=t_arr * 1e3, y=i_step * 1e3, mode='lines',
            line=dict(color='#e74c3c', width=2.5), name="Step Current"
        ), row=1, col=1)

        # 2. Linear Sweep i(t) = v*Cdl * [1 - exp(-t / Rs*Cdl)]
        t_ramp = np.linspace(0, 8 * tau_s, 300)
        i_ramp = scan_rate_vs * u_cdl * (1 - np.exp(-t_ramp / tau_s))
        fig_bridge.add_trace(go.Scatter(
            x=t_ramp * 1e3, y=i_ramp * 1e6, mode='lines',
            line=dict(color='#3498db', width=2.5), name="Ramp Charging"
        ), row=1, col=2)

        # 3. Cyclic Voltammetry Hysteresis Box: forward and reverse
        e_range = np.linspace(0, 0.8, 300)
        i_ss_pos = scan_rate_vs * u_cdl * 1e6
        i_ss_neg = -scan_rate_vs * u_cdl * 1e6
        fig_bridge.add_trace(go.Scatter(
            x=np.concatenate([e_range, e_range[::-1]]),
            y=np.concatenate([np.full_like(e_range, i_ss_pos), np.full_like(e_range, i_ss_neg)]),
            fill='toself', fillcolor='rgba(46, 204, 113, 0.25)',
            line=dict(color='#2ecc71', width=2), name="CV Hysteresis Box"
        ), row=2, col=1)

        # 4. Semicircle Nyquist
        w_bridge = np.logspace(-1, 6, 200)
        denom_b = 1 + (w_bridge * u_rct * u_cdl) ** 2
        zb_re = u_rs + (u_rct / denom_b)
        zb_im = (w_bridge * u_rct**2 * u_cdl) / denom_b
        fig_bridge.add_trace(go.Scatter(
            x=zb_re, y=zb_im, mode='lines',
            line=dict(color='#f1c40f', width=2.5), name="EIS Semicircle"
        ), row=2, col=2)

        fig_bridge.update_xaxes(title_text="Time t (ms)", row=1, col=1)
        fig_bridge.update_yaxes(title_text="Current i (mA)", row=1, col=1)
        fig_bridge.update_xaxes(title_text="Time t (ms)", row=1, col=2)
        fig_bridge.update_yaxes(title_text="Charging Current (\\mu A)", row=1, col=2)
        fig_bridge.update_xaxes(title_text="Applied Potential E (V)", row=2, col=1)
        fig_bridge.update_yaxes(title_text="Current (\\mu A)", row=2, col=1)
        fig_bridge.update_xaxes(title_text="Real Z' (\\Omega)", row=2, col=2)
        fig_bridge.update_yaxes(title_text="-Imag Z'' (\\Omega)", scaleanchor="x4", scaleratio=1, row=2, col=2)

        fig_bridge.update_layout(
            template="plotly_dark",
            height=520,
            margin=dict(l=30, r=30, t=30, b=20),
            showlegend=False
        )
        st.plotly_chart(fig_bridge, use_container_width=True)

        st.info(f"""
        **Diagnostic Summary:**
        * **Chronoamperometry time-to-decay (95% charge):** $3\\tau = 3 R_s C_{{dl}} = {3 * tau_s * 1e3:.2f}\\ \\text{{ms}}$.
        * **CV Non-Faradaic Plateau:** $i_c = v \\cdot C_{{dl}} = {scan_rate_vs * u_cdl * 1e6:.2f}\\ \\mu\\text{{A}}$.
        * **EIS Characteristic Frequency:** $f_{{apex}} = \\frac{{1}}{{2\\pi R_{{ct}} C_{{dl}}}} = {1.0 / (2 * np.pi * u_rct * u_cdl):.1f}\\ \\text{{Hz}}$.
        """)

# =============================================================================
# TAB 3: DIFFUSION & THE WARBURG IMPEDANCE
# =============================================================================
with tab3:
    st.subheader("3. Diffusion Dynamics: From Semi-Infinite 45° Warburg to Finite-Length Boundaries")
    st.markdown("""
    **Physical Principle (Bard & Faulkner §10.1.3 & §10.2; Gileadi §16.2):**  
    At low frequencies, concentration gradients develop as reactants are depleted at the electrode. 
    Solving Fick's second law for sinusoidal oscillations yields an impedance where the concentration wave lags potential by exactly **45°** ($Z_W = \\sigma_W \\omega^{-1/2} (1 - j)$).
    * If the diffusion layer can grow indefinitely into bulk liquid: **Infinite 45° tail**.
    * If diffusion is bounded by convective stirring or a porous membrane:
      * **Transmissive Boundary (Open Warburg / RDE):** The AC diffusion wave reaches a well-stirred reservoir, and the line **curls down into a low-frequency arc**.
      * **Reflective Boundary (Blocking Warburg / Intercalation):** Ions cannot leave the far boundary (e.g., solid host particles), and the line **bends upward into a 90° capacitive spike**.
    """)

    col_t3_c, col_t3_p = st.columns([1, 2.2], gap="large")

    with col_t3_c:
        st.markdown("#### Kinetic & Mass-Transfer Controls")
        w_rs = st.slider("Solution Resistance ($R_s$, $\\Omega$):", 5.0, 100.0, 20.0, 5.0, key="w_rs")
        w_rct = st.slider("Charge-Transfer Resistance ($R_{ct}$, $\\Omega$):", 20.0, 800.0, 150.0, 10.0, key="w_rct")
        w_cdl_uf = st.slider("Double-Layer Capacitance ($C_{dl}$, $\\mu\\text{F}$):", 1.0, 50.0, 10.0, 1.0, key="w_cdl")
        w_cdl = w_cdl_uf * 1e-6

        sigma_w = st.slider("Warburg Coefficient ($\\sigma_W$, $\\Omega\\cdot\\text{s}^{-1/2}$):", 10.0, 500.0, 80.0, 5.0)

        diffusion_mode = st.radio(
            "Boundary Condition:",
            ["Semi-Infinite Linear (Unstirred, unbounded)", "Transmissive / Open Boundary (Nernst layer δ, RDE)", "Reflective / Blocking Boundary (Solid-state intercalation)"]
        )

        if "Boundary" in diffusion_mode:
            delta_diff = st.slider("Diffusion Layer Thickness ($\\delta$, $\\mu\\text{m}$):", 5.0, 150.0, 35.0, 5.0)
            d_diff = 1e-5  # cm^2/s typical
            tau_diff = ((delta_diff * 1e-4) ** 2) / d_diff
            st.caption(f"Diffusion Relaxation Time $\\tau_d = \\delta^2 / D = {tau_diff:.3f}\\ \\text{{s}}$")

    with col_t3_p:
        w_freqs = np.logspace(-2, 5, 400)
        w_omega = 2 * np.pi * w_freqs

        # Calculate Warburg component
        if "Semi-Infinite" in diffusion_mode:
            z_w_real = sigma_w / np.sqrt(w_omega)
            z_w_imag = sigma_w / np.sqrt(w_omega)
        elif "Transmissive" in diffusion_mode:
            # Z_w = R_d * tanh( (j*w*tau_d)^0.5 ) / (j*w*tau_d)^0.5
            r_d = sigma_w * np.sqrt(2 * tau_diff)
            arg = np.sqrt(1j * w_omega * tau_diff)
            zw_trans = r_d * np.tanh(arg) / arg
            z_w_real = zw_trans.real
            z_w_imag = -zw_trans.imag
        else:  # Reflective
            # Z_w = R_d * coth( (j*w*tau_d)^0.5 ) / (j*w*tau_d)^0.5
            r_d = sigma_w * np.sqrt(2 * tau_diff)
            arg = np.sqrt(1j * w_omega * tau_diff)
            zw_refl = r_d / (arg * np.tanh(arg))
            z_w_real = zw_refl.real
            z_w_imag = -zw_refl.imag

        # Total Randles Impedance: Z = Rs + 1 / [ j*w*Cdl + 1 / (Rct + Zw) ]
        z_faradaic = (w_rct + z_w_real) - 1j * z_w_imag
        y_total = 1j * w_omega * w_cdl + (1.0 / z_faradaic)
        z_interface = 1.0 / y_total

        z_tot_real = w_rs + z_interface.real
        z_tot_imag = -z_interface.imag

        fig_w = go.Figure()
        fig_w.add_trace(go.Scatter(
            x=z_tot_real, y=z_tot_imag, mode='lines',
            line=dict(color='#00d2ff', width=3),
            name="Impedance Spectrum"
        ))

        fig_w.update_layout(
            title=f"Complex Plane Nyquist: {diffusion_mode}",
            xaxis_title="Real Impedance Z' (\\Omega)",
            yaxis_title="Negative Imaginary Impedance -Z'' (\\Omega)",
            yaxis=dict(scaleanchor="x", scaleratio=1),
            template="plotly_dark",
            height=460,
            margin=dict(l=40, r=40, t=50, b=30)
        )
        st.plotly_chart(fig_w, use_container_width=True)

# =============================================================================
# TAB 4: CELL GEOMETRY & LUGGIN CAPILLARY SANDBOX
# =============================================================================
with tab4:
    st.subheader("4. Cell Geometry & Uncompensated Resistance: Gileadi's Chapter 3 Formulations")
    st.markdown("""
    **Analytical Theory (Gileadi §3.2.3, Equations 3.3–3.8 & Figure 3.3; Bard & Faulkner §1.3.4):**  
    A Luggin capillary is designed to sample solution potential near a working electrode without drawing current. 
    However, the **uncompensated ohmic drop ($j R_u$)** depends fundamentally on the electrode's spatial geometry:
    * **Planar:** $j R_u = j \\cdot \\frac{d}{\\kappa}$ $\\longrightarrow$ **Linear growth** with distance $d$. Moving further away increases error indefinitely.
    * **Cylindrical:** $j R_u = \\frac{j \\cdot r_0}{\\kappa} \\ln\\left(1 + \\frac{d}{r_0}\\right)$ $\\longrightarrow$ **Logarithmic growth**. Thinner wire ($r_0$) drastically reduces ohmic drop.
    * **Spherical (UME / Drop):** $j R_u = \\frac{j \\cdot d}{\\kappa} \\left(\\frac{r_0}{r_0 + d}\\right) \\longrightarrow$ **Plateaus at a finite limit** $\\frac{j r_0}{\\kappa}$. Virtually all ohmic drop is confined within 5 radii ($d \\le 5 r_0$)!
    """)

    col_t4_c, col_t4_p = st.columns([1, 2.0], gap="large")

    with col_t4_c:
        st.markdown("#### Electrochemical Cell Conditions")
        kappa_cond = st.slider("Electrolyte Specific Conductivity ($\\kappa$, $\\text{mS/cm}$):", 0.5, 50.0, 10.0, 0.5)
        kappa = kappa_cond * 1e-3  # S/cm
        j_curr_ma = st.slider("Applied Current Density ($j$, $\\text{mA/cm}^2$):", 0.1, 20.0, 2.0, 0.1)
        j_curr = j_curr_ma * 1e-3  # A/cm^2

        r0_wire_mm = st.slider("Cylinder / Sphere Radius ($r_0$, mm):", 0.05, 5.0, 0.5, 0.05)
        r0 = r0_wire_mm * 1e-1  # cm

        st.markdown("#### Luggin Tip Position")
        d_probe_mm = st.slider("Luggin Tip Distance ($d$, mm):", 0.1, 15.0, 3.0, 0.1)
        d_probe = d_probe_mm * 1e-1  # cm

    with col_t4_p:
        d_range_cm = np.linspace(0.005, 1.5, 300)

        # 1. Planar
        ru_planar = d_range_cm / kappa
        j_ru_planar = (j_curr * ru_planar) * 1e3  # mV

        # 2. Cylindrical
        ru_cyl = (r0 / kappa) * np.log(1 + d_range_cm / r0)
        j_ru_cyl = (j_curr * ru_cyl) * 1e3

        # 3. Spherical
        ru_sph = (d_range_cm / kappa) * (r0 / (r0 + d_range_cm))
        j_ru_sph = (j_curr * ru_sph) * 1e3

        fig_geom = go.Figure()
        fig_geom.add_trace(go.Scatter(x=d_range_cm * 10, y=j_ru_planar, mode='lines', line=dict(color='#e74c3c', width=3), name="Planar Electrode"))
        fig_geom.add_trace(go.Scatter(x=d_range_cm * 10, y=j_ru_cyl, mode='lines', line=dict(color='#f39c12', width=3), name=f"Cylindrical Wire (r₀ = {r0_wire_mm:.2f} mm)"))
        fig_geom.add_trace(go.Scatter(x=d_range_cm * 10, y=j_ru_sph, mode='lines', line=dict(color='#2ecc71', width=3), name=f"Spherical Drop (r₀ = {r0_wire_mm:.2f} mm)"))

        # Mark selected point
        p_val_pl = (j_curr * (d_probe / kappa)) * 1e3
        p_val_cy = (j_curr * (r0 / kappa) * np.log(1 + d_probe / r0)) * 1e3
        p_val_sp = (j_curr * (d_probe / kappa) * (r0 / (r0 + d_probe))) * 1e3

        fig_geom.add_trace(go.Scatter(
            x=[d_probe_mm, d_probe_mm, d_probe_mm],
            y=[p_val_pl, p_val_cy, p_val_sp],
            mode='markers', marker=dict(size=10, color='#f1c40f', symbol='diamond'),
            name="Selected Position"
        ))

        fig_geom.update_layout(
            title="Uncompensated Ohmic Drop j·Ru vs. Luggin Distance d (Gileadi Fig. 3.3)",
            xaxis_title="Capillary Distance from Working Electrode d (mm)",
            yaxis_title="Uncompensated Potential Drop j·Ru (mV)",
            template="plotly_dark",
            height=440,
            legend=dict(x=0.02, y=0.98),
            margin=dict(l=40, r=40, t=50, b=20)
        )
        st.plotly_chart(fig_geom, use_container_width=True)

        st.markdown(f"""
        **Ohmic Drop Comparison at $d = {d_probe_mm:.1f}\\ \\text{{mm}}$:**
        * **Planar:** `{p_val_pl:.1f} mV` error included in measured overpotential.
        * **Cylindrical:** `{p_val_cy:.1f} mV` error.
        * **Spherical:** `{p_val_sp:.1f} mV` error *(stabilized at {((j_curr * r0 / kappa) * 1e3):.1f} mV asymptote)*.
        """)

# =============================================================================
# TAB 5: SOLID-STATE PELLET & COMPOSITE NORMALIZER
# =============================================================================
with tab5:
    st.subheader("5. Solid-State Pellet & Composite Electrolyte Normalizer")
    st.markdown("""
    **Direct Implementation of Master's Thesis Formulations (Sections 4.1.3 & 5.1.2, Equations 4.1 & 5.1–5.3):**  
    In ceramic pellets ($\text{Li}_{0.4}\text{La}_{0.6}\text{TiO}_3$, LLZO, etc.) pressed in Swagelok cells ($SS \\mid \\text{Pellet} \\mid SS$), 
    raw impedance underestimates conductivity because green/sintered pellets possess **residual microporosity ($P_{or} = \\rho_{exp} / \\rho_{th}$)**. 
    True crystallographic bulk conductivity requires normalization by theoretical single-crystal density:
    $$\\rho_{th} = \\frac{6 \\cdot M_{LLTO}}{N_A \\cdot V_{uc} \\cdot 10^{-24}}, \\quad \\sigma_{corrected} = \\frac{\\sigma_{raw}}{P_{or}}$$
    """)

    col_t5_in, col_t5_out = st.columns([1, 1], gap="large")

    with col_t5_in:
        st.markdown("#### Pellet Dimensions & Geometry")
        pellet_diam_mm = st.number_input("Pellet Diameter ($D$, mm):", value=13.0, step=0.5)
        pellet_thick_um = st.number_input("Pellet Thickness ($L$, $\\mu\\text{m}$):", value=1150.0, step=25.0)
        pellet_mass_mg = st.number_input("Pellet Mass ($m$, mg):", value=520.0, step=10.0)

        st.markdown("#### Crystallographic Unit Cell")
        v_uc_ang3 = st.number_input("XRD Unit Cell Volume ($V_{uc}$, $\\text{Å}^3$):", value=58.70, step=0.1,
                                    help="LLTO cubic unit cell volume from Thesis Section 5.1.1.")
        mw_material = st.number_input("Molecular Weight (g/mol):", value=207.72, step=0.5,
                                      help="Li0.4La0.6TiO3 molar mass.")

        st.markdown("#### Swagelok EIS Deconvolution Values")
        r_bulk = st.number_input("High-Frequency Bulk Resistance ($R_{bulk}$, $\\Omega$):", value=1500.0, step=50.0)
        r_gb = st.number_input("Grain Boundary Resistance ($R_{gb}$, $\\Omega$):", value=85000.0, step=1000.0)

    with col_t5_out:
        # Calculations
        area_cm2 = np.pi * ((pellet_diam_mm / 2.0) / 10.0) ** 2
        thick_cm = pellet_thick_um * 1e-4
        vol_pellet_cm3 = area_cm2 * thick_cm

        # Densities
        n_a = 6.02214e23
        rho_th = (6.0 * mw_material) / (n_a * (v_uc_ang3 * 1e-24))  # g/cm^3
        rho_exp = (pellet_mass_mg * 1e-3) / vol_pellet_cm3
        porosity_fraction = rho_exp / rho_th

        # Conductivities
        r_total = r_bulk + r_gb
        sigma_bulk_raw = thick_cm / (r_bulk * area_cm2)
        sigma_total_raw = thick_cm / (r_total * area_cm2)

        sigma_bulk_cor = sigma_bulk_raw / porosity_fraction
        sigma_total_cor = sigma_total_raw / porosity_fraction

        st.markdown("#### Density & Porosity Analysis")
        st.markdown(f"""
        <div class="metric-card">
            <b>Theoretical Single-Crystal Density ($\\rho_{{th}}$):</b> <code>{rho_th:.3f} g/cm³</code><br>
            <b>Measured Experimental Bulk Density ($\\rho_{{exp}}$):</b> <code>{rho_exp:.3f} g/cm³</code><br>
            <b>Pellet Relative Density Fraction ($P_{{or}}$):</b> <code>{porosity_fraction*100.0:.1f}%</code><br>
            <b>Estimated Residual Porosity:</b> <code>{(1.0 - porosity_fraction)*100.0:.1f}%</code>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("#### Conductivity Deconvolution")
        res_df = pd.DataFrame([
            {"Parameter": "Bulk Intra-granular (σ_bulk)", "Raw Measured": f"{sigma_bulk_raw:.2e} S/cm", "Porosity-Corrected": f"{sigma_bulk_cor:.2e} S/cm"},
            {"Parameter": "Total Grain+Boundary (σ_total)", "Raw Measured": f"{sigma_total_raw:.2e} S/cm", "Porosity-Corrected": f"{sigma_total_cor:.2e} S/cm"}
        ])
        st.dataframe(res_df, use_container_width=True, hide_index=True)

        st.success(f"**Correction Factor:** True conductivity is **{1.0/porosity_fraction:.2f}×** higher than raw measured resistance due to residual green pellet micropores!")

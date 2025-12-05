import streamlit as st
import requests, json, joblib
import lightkurve as lk
import numpy as np
import matplotlib.pyplot as plt
from astropy.constants import R_earth, R_sun, G, M_sun
import astropy.units as u
import os
from streamlit_oauth import OAuth2Component

# ========== PAGE CONFIG & STYLING ==========
st.set_page_config(page_title="Bunting — Exoplanet Detection", page_icon="🔭", layout="wide")

# Custom CSS for centering and bold text in login UI
st.markdown("""
    <style>
    .main {
        display: flex;
        flex-direction: column;
        align-items: center;
        justify-content: center;
        text-align: center;
    }
    .stButton>button {
        display: block;
        margin: 0 auto;
        font-weight: bold;
    }
    .stTextInput>div>div>input {
        text-align: center;
        font-weight: bold;
    }
    .login-container {
        display: flex;
        flex-direction: column;
        align-items: center;
        justify-content: center;
        width: 100%;
    }
    .google-button {
        display: flex;
        justify-content: center;
    }
    h3, h4 {
        font-weight: bold;
        text-align: center;
    }
    hr {
        width: 50%;
        margin: 20px auto;
    }
    .stImage > div {
        display: flex !important;
        justify-content: center !important;
    }
    .stRadio > div {
        display: flex;
        flex-direction: row;
        justify-content: center;
        flex-wrap: wrap;
        gap: 10px;
    }
    .stRadio > div > label {
        margin: 0 10px;
    }
    </style>
""", unsafe_allow_html=True)


# ========== FIREBASE AUTHENTICATION ==========

def sign_in(email: str, password: str):
    apikey = 'AIzaSyAqvXwzaDvA3F3xkhHzbAGWmswYu5NDAds'
    payload = json.dumps({'email': email, 'password': password, 'returnSecureToken': True})
    r = requests.post(
        'https://identitytoolkit.googleapis.com/v1/accounts:signInWithPassword',
        params={'key': apikey}, data=payload
    )
    resp = r.json()
    if 'error' in resp:
        return None, resp['error']['message']
    return resp.get('idToken'), None


def sign_up(email: str, password: str):
    apikey = 'AIzaSyAqvXwzaDvA3F3xkhHzbAGWmswYu5NDAds'
    details = {'email': email, 'password': password, 'returnSecureToken': True}
    r = requests.post(
        f'https://identitytoolkit.googleapis.com/v1/accounts:signUp?key={apikey}',
        data=details
    )
    resp = r.json()
    if 'error' in resp:
        return None, resp['error']['message']
    return resp.get('idToken'), None


# def sign_in_with_google(id_token: str):
#     apikey = 'AIzaSyAqvXwzaDvA3F3xkhHzbAGWmswYu5NDAds'
#     payload = {
#         'postBody': f'id_token={id_token}&providerId=google.com',
#         'requestUri': 'http://localhost:8501',
#         'returnSecureToken': True
#     }
#     r = requests.post(
#         f'https://identitytoolkit.googleapis.com/v1/accounts:signInWithIdp?key={apikey}',
#         json=payload
#     )
#     resp = r.json()
#     if 'error' in resp:
#         return None, resp['error']['message']
#     return resp.get('idToken'), None


# ========== HABITABILITY CALCULATION ==========

def calculate_habitability(depth, r_star_sun, t_eff, albedo, period):
    # Convert star radius to meters
    r_star = r_star_sun * R_sun

    # Calculate planet radius from transit depth
    r_planet = np.sqrt(depth) * r_star

    # Calculate semi-major axis using Kepler's Third Law
    # a^3 / P^2 = GM / (4π^2)
    star_mass = 1.0 * M_sun  # Assume star mass is 1 solar mass for simplicity
    period_sec = period * 86400 * u.s  # Convert period from days to seconds with unit
    a_cubed = (G * star_mass * period_sec ** 2) / (4 * np.pi ** 2)
    a = (a_cubed.value ** (1 / 3)) * u.m  # Take cube root and assign meter unit

    # Calculate equilibrium temperature
    t_eq = t_eff * np.sqrt(r_star / (2 * a)) * (1 - albedo) ** 0.25

    return {
        'radius_sun': r_planet.to(R_sun).value,  # Planet radius in solar radii
        'radius_earth': r_planet.to(R_earth).value,  # Planet radius in Earth radii
        'semi_major_sun': a.to(R_sun).value,  # Semi-major axis in solar radii
        'semi_major_au': a.to(u.au).value,  # Semi-major axis in AU
        'semi_major_earth': (a.to(u.m).value / R_earth.to(u.m).value),  # Semi-major_EDIT axis in Earth radii
        'equilibrium_temp': t_eq.value  # Equilibrium temperature in Kelvin
    }


# ========== ATMOSPHERIC ANALYSIS ==========

def analyze_atmosphere(radius_earth, t_eq):
    # Simplified atmospheric composition based on temperature and radius
    gases = ['H₂', 'He', 'CO₂', 'N₂', 'CH₄', 'H₂O']
    likelihoods = np.zeros(len(gases))

    # Adjust likelihoods based on temperature
    if t_eq < 150:  # Very cold
        likelihoods[4] = 0.8  # CH₄
        likelihoods[3] = 0.6  # N₂
        likelihoods[5] = 0.2  # H₂O (frozen)
    elif 150 <= t_eq < 300:  # Cold, potentially habitable
        likelihoods[5] = 0.7  # H₂O (if liquid water exists)
        likelihoods[2] = 0.6  # CO₂
        likelihoods[3] = 0.5  # N₂
        likelihoods[4] = 0.3  # CH₄
    elif 300 <= t_eq < 600:  # Warm
        likelihoods[2] = 0.7  # CO₂
        likelihoods[3] = 0.6  # N₂
        likelihoods[4] = 0.4  # CH₄
    else:  # Hot
        likelihoods[0] = 0.9  # H₂
        likelihoods[1] = 0.8  # He

    # Adjust based on planet size (larger planets retain lighter gases)
    if radius_earth > 5:  # Gas giant-like
        likelihoods[0] += 0.2  # H₂
        likelihoods[1] += 0.2  # He
    elif 1 < radius_earth <= 5:  # Earth-like
        likelihoods[2] += 0.1  # CO₂
        likelihoods[3] += 0.1  # N₂
        likelihoods[5] += 0.1  # H₂O

    # Normalize likelihoods to sum to 1
    likelihoods = likelihoods / np.sum(likelihoods) if np.sum(likelihoods) > 0 else np.ones(len(gases)) / len(gases)

    return gases, likelihoods


# ========== SESSION STATE ==========
if 'token' not in st.session_state:
    st.session_state.token = None
if 'page' not in st.session_state:
    st.session_state.page = 'login'

# ========== LOGIN ==========
if st.session_state.token is None:
    st.markdown('<div class="login-container">', unsafe_allow_html=True)

    # Create three columns with specific width ratios for better centering
    left_spacer, content, right_spacer = st.columns([1, 2, 1])

    # Place everything in the center column
    with content:
        # Center the image
        try:
            st.image('Images/logo_bird.png', width=60, use_container_width=False)
        except:
            pass

        # Add the text headers directly below the image in the same column
        st.markdown("### Welcome to **Bunting**", unsafe_allow_html=True)
        st.markdown("#### **Exoplanet Detection App Login**", unsafe_allow_html=True)

        # Add the form elements
        email = st.text_input("Email", key="login_email")
        pwd = st.text_input("Password", type="password", key="login_pwd")

        # Login and Sign Up buttons side by side
        col1, col2 = st.columns(2)
        with col1:
            if st.button("**Login**", key="login_button"):
                token, err = sign_in(email, pwd)
                if err:
                    st.error(err)
                else:
                    st.session_state.token = token
                    st.session_state.page = 'exo_search'
                    st.rerun()
        with col2:
            if st.button("**Sign Up**", key="signup_button"):
                token, err = sign_up(email, pwd)
                if err:
                    st.error(err)
                else:
                    st.success("Account created! Please log in.")

        # Separator
        st.markdown("<hr>", unsafe_allow_html=True)

        # ========== GOOGLE SSO INTEGRATION ==========
        # NOTE: You must create OAuth credentials in Google Cloud Console
        # and replace the placeholders below.

        # CLIENT_ID = st.secrets.get("GOOGLE_CLIENT_ID", os.environ.get("GOOGLE_CLIENT_ID"))
        # CLIENT_SECRET = st.secrets.get("GOOGLE_CLIENT_SECRET", os.environ.get("GOOGLE_CLIENT_SECRET"))
        # REDIRECT_URI = "http://localhost:8501"
        #
        # if CLIENT_ID == "YOUR_GOOGLE_CLIENT_ID_HERE":
        #     st.warning("⚠️ Google OAuth credentials not set in code.")
        # else:
        #     try:
        #         # OAuth endpoints for Google
        #         oauth2 = OAuth2Component(
        #             CLIENT_ID,
        #             CLIENT_SECRET,
        #             "https://accounts.google.com/o/oauth2/v2/auth",
        #             "https://oauth2.googleapis.com/token",
        #             "https://oauth2.googleapis.com/token",
        #             "https://oauth2.googleapis.com/revoke"
        #         )
        #
        #         # Render the Authorization Button
        #         result = oauth2.authorize_button(
        #             name="Sign in with Google",
        #             icon="https://www.google.com.tw/favicon.ico",
        #             redirect_uri=REDIRECT_URI,
        #             scope="openid email profile",
        #             key="google_oauth",
        #             extras_params={"prompt": "consent", "access_type": "offline"}
        #         )
        #
        #         if result and 'token' in result:
        #             id_token = result.get('token', {}).get('id_token')
        #             if id_token:
        #                 with st.spinner("Authenticating with Firebase..."):
        #                     token, err = sign_in_with_google(id_token)
        #                     if err:
        #                         st.error(f"Firebase Auth Failed: {err}")
        #                     else:
        #                         st.session_state.token = token
        #                         st.session_state.page = 'exo_search'
        #                         st.rerun()
        #     except Exception as e:
        #         st.error(f"OAuth Error: {e}")

    st.markdown('</div>', unsafe_allow_html=True)

else:
    # ========== SIDEBAR NAVIGATION ==========
    st.sidebar.title("Navigation")
    module = st.sidebar.radio("Select Module", ["Exoplanet Detection", "Habitability Analysis"])
    if st.sidebar.button("Logout"):
        st.session_state.token = None
        st.session_state.page = 'login'
        st.rerun()

    # Sync page state
    if module == "Exoplanet Detection":
        if not st.session_state.page.startswith('exo'):
            st.session_state.page = 'exo_search'
            st.rerun()
    else:
        if not st.session_state.page.startswith('hab'):
            st.session_state.page = 'hab_search'
            st.rerun()

    # ========== EXOPLANET DETECTION: SEARCH ==========
    if st.session_state.page == 'exo_search':
        st.header("Exoplanet Detection - Search")
        target = st.text_input("TIC ID", "TIC 145241359")
        mission = st.selectbox("Mission", ["All", "Kepler", "K2", "TESS"])
        if st.button("Search"):
            with st.spinner("Querying light curves…"):
                try:
                    res = (lk.search_lightcurve(target, cadence='long')
                           if mission == "All"
                           else lk.search_lightcurve(target, mission=mission, cadence='long'))
                    st.session_state.search = res
                    st.session_state.filtered = res
                    st.session_state.page = 'exo_results'
                    st.rerun()
                except Exception as e:
                    st.error(f"Search failed: {e}")

    # ========== EXOPLANET DETECTION: RESULTS ==========
    elif st.session_state.page == 'exo_results':
        st.header("Exoplanet Detection - Results")
        res = st.session_state.filtered
        st.write(f"Found **{len(res)}** entries")
        st.dataframe(res.table)
        idx = st.number_input("Pick an index", 0, len(res) - 1, 0)
        if st.button("Download & Analyze"):
            with st.spinner("Downloading…"):
                st.session_state.lc = res[idx].download()
                st.session_state.page = 'exo_analyze'
                st.rerun()

    # ========== EXOPLANET DETECTION: ANALYZE ==========
    elif st.session_state.page == 'exo_analyze':
        lc = st.session_state.lc
        st.header("Exoplanet Detection - Analysis")

        # Line Type Selection
        line_type = st.radio(
            "Select Line Type",
            ["Line", "Scatter", "Error Bar Plot"],
            horizontal=True
        )

        # Plot Type Selection
        plot_type = st.radio(
            "Select Plot Type",
            ["Raw", "Normalized", "Folded", "Flattened", "Binned", "BLS Periodogram"],
            horizontal=True
        )

        # BLS Parameters Input
        if plot_type == "BLS Periodogram":
            st.subheader("BLS Parameters")
            col1, col2, col3 = st.columns(3)
            with col1:
                min_period = st.number_input("Minimum Period (days)", value=0.5, step=0.1, min_value=0.1)
            with col2:
                max_period = st.number_input("Maximum Period (days)", value=20.0, step=0.1, min_value=min_period + 0.1)
            with col3:
                freq_factor = st.number_input("Frequency Factor", value=500, step=50, min_value=100)

        # Plotting Logic
        try:
            fig, ax = plt.subplots(figsize=(8, 3))  # Reduced size


            # Function to plot based on line type
            def plot_light_curve(data, ax, line_type):
                if line_type == "Line":
                    data.plot(ax=ax)
                elif line_type == "Scatter":
                    data.scatter(ax=ax)
                elif line_type == "Error Bar Plot":
                    # Use errorbar plot with flux errors if available
                    ax.errorbar(
                        data.time.value,
                        data.flux.value,
                        yerr=data.flux_err.value if hasattr(data, 'flux_err') else None,
                        fmt='o',
                        markersize=3,
                        capsize=2
                    )
                    ax.set_xlabel("Time (days)")
                    ax.set_ylabel("Flux")
                    ax.set_title(data.meta.get('title', ''))


            if plot_type == "Raw":
                plot_light_curve(lc, ax, line_type)
                st.pyplot(fig)

            elif plot_type == "Normalized":
                norm = lc.normalize()
                plot_light_curve(norm, ax, line_type)
                st.pyplot(fig)

            elif plot_type == "Folded":
                P = st.number_input("Fold period [d]", value=1.0, step=0.1, min_value=0.1)
                folded = lc.fold(period=P)
                plot_light_curve(folded, ax, line_type)
                st.pyplot(fig)

            elif plot_type == "Flattened":
                W = st.slider("Window length", 50, 500, 101)
                flat = lc.flatten(window_length=W)
                plot_light_curve(flat, ax, line_type)
                st.pyplot(fig)

            elif plot_type == "Binned":
                B = st.slider("Bins", 10, 500, 100)
                time_range = (lc.time[-1] - lc.time[0]).value  # Total time span in days
                time_bin_size = time_range / B  # Size of each bin in days
                b = lc.bin(time_bin_size=time_bin_size)
                plot_light_curve(b, ax, line_type)
                st.pyplot(fig)

            elif plot_type == "BLS Periodogram":
                periods = np.linspace(min_period, max_period, freq_factor)
                bls = lc.to_periodogram(method='bls', period=periods)
                st.session_state.bls = bls
                bls.plot(ax=ax)
                st.pyplot(fig)

                # Button to show BLS results
                if st.button("Show BLS Results"):
                    st.write(f"**Best Period**: {bls.period_at_max_power.value:.4f} days")
                    st.write(f"**Transit Time**: {bls.transit_time_at_max_power.value:.4f} days")
                    st.write(f"**Duration**: {bls.duration_at_max_power.value:.4f} days")

        except Exception as e:
            st.error(f"Plot error: {str(e)}")

        # BLS Phase Fold Button
        if 'bls' in st.session_state:
            if st.button("BLS Phase Fold"):
                try:
                    bls = st.session_state.bls
                    folded = lc.fold(period=bls.period_at_max_power,
                                     epoch_time=bls.transit_time_at_max_power)
                    fig, ax = plt.subplots(figsize=(8, 3))
                    plot_light_curve(folded, ax, line_type)
                    ax.set_xlim(-1, 1)
                    st.pyplot(fig)
                except Exception as e:
                    st.error(f"BLS Phase Fold error: {str(e)}")

        # ML Prediction Button
        if st.button("Predict with AI"):
            if 'bls' not in st.session_state:
                st.warning("Run BLS first!")
            else:
                with st.spinner("Running ML…"):
                    bls = st.session_state.bls
                    P = bls.period_at_max_power.value
                    t0 = bls.transit_time_at_max_power.value
                    dur = bls.duration_at_max_power.value

                    # simple transit depth from normalized folded light curve
                    folded_norm = lc.normalize().fold(period=P, epoch_time=t0)
                    depth_vals = 1 - folded_norm.flux.value
                    max_depth = np.nanmax(depth_vals)

                    # if depth exceeds threshold, classify as exoplanet
                    if max_depth > 0.05:
                        st.success(f"🔭 Likely an exoplanet (deep transit: {max_depth:.3f})")
                    else:
                        # build global & local arrays
                        f = lc.fold(period=P, epoch_time=t0)
                        g_vec = f.bin(time_bin_size=0.005).normalize().flux.value - 1
                        mask = (f.phase.value > -4 * dur / P) & (f.phase.value < 4 * dur / P)
                        l_vec = f[mask].bin(time_bin_size=0.0005).normalize().flux.value - 1

                        # subsample & pad/truncate
                        global_vec = np.resize(g_vec[::10], 17134)
                        local_vec = np.resize(l_vec, 2773)

                        # clean NaNs/Infs
                        G = np.nan_to_num(global_vec.reshape(1, -1), nan=-999, posinf=-999, neginf=-999)
                        L = np.nan_to_num(local_vec.reshape(1, -1), nan=-999, posinf=-999, neginf=-999)

                        # load RF models
                        g_model = joblib.load("RFM_G_model_pkl")
                        l_model = joblib.load("RFM_L_model_pkl")

                        # hard predictions
                        g_pred = g_model.predict(G)[0]
                        l_pred = l_model.predict(L)[0]

                        if g_pred == 1 or l_pred == 1:
                            st.success("🔭 Likely to be an exoplanet")
                        else:
                            st.error("❌ Not likely an exoplanet")

    # ========== HABITABILITY ANALYSIS ==========
    elif st.session_state.page == 'hab_search':
        st.header("Habitability Analysis - Search")
        target = st.text_input("Enter Target ID", "TIC 145241359")
        mission = st.selectbox("Mission", ["All", "Kepler", "K2", "TESS"])
        if st.button("Search"):
            with st.spinner("Searching…"):
                try:
                    res = (lk.search_lightcurve(target, cadence='long')
                           if mission == "All"
                           else lk.search_lightcurve(target, mission=mission, cadence='long'))
                    st.session_state.hab_search = res
                    st.session_state.hab_filtered = res
                    st.session_state.page = 'hab_results'
                    st.rerun()
                except Exception as e:
                    st.error(f"Search error: {e}")

    elif st.session_state.page == 'hab_results':
        st.header("Habitability Analysis - Results")
        res = st.session_state.hab_filtered
        st.write(f"Found **{len(res)}** entries")
        st.dataframe(res.table)
        idx = st.number_input("Pick an index", 0, len(res) - 1, 0)
        if st.button("Download Light Curve"):
            with st.spinner("Downloading…"):
                st.session_state.hab_lc = res[idx].download()
                st.session_state.page = 'hab_analysis'
                st.rerun()

    elif st.session_state.page == 'hab_analysis':
        lc = st.session_state.hab_lc
        st.header("Habitability Analysis - Calculation")

        # Period and Transit Epoch as text boxes
        _, col, _ = st.columns([1, 2, 1])
        with col:
            period_input = st.text_input("Enter Period (days)", value="18.0", key="period_input")
            epoch_input = st.text_input("Enter Transit Epoch (t0)", value="2.0", key="epoch_input")

        # Convert inputs to floats with error handling
        try:
            period = float(period_input)
            if period <= 0:
                st.error("Period must be greater than 0.")
                period = None
        except ValueError:
            st.error("Invalid input for Period. Please enter a valid number.")
            period = None

        try:
            epoch = float(epoch_input)
        except ValueError:
            st.error("Invalid input for Transit Epoch. Please enter a valid number.")
            epoch = None

        # Initialize folded light curve in session state
        if 'folded_lc' not in st.session_state:
            st.session_state.folded_lc = None

        # Phase Fold button
        if st.button("Phase Fold") and period is not None and epoch is not None:
            try:
                st.session_state.folded_lc = lc.fold(period=period, epoch_time=epoch)
            except Exception as e:
                st.error(f"Error folding light curve: {str(e)}")

        # Display the folded light curve with adjustable depth line and axis limits
        if st.session_state.folded_lc is not None:
            folded = st.session_state.folded_lc

            # Adjustable depth line using a slider with 6 decimal places
            depth_line = st.slider(
                "Adjust Transit Depth Line (Flux)",
                min_value=0.9,
                max_value=1.0,
                value=0.99,
                step=0.000001,
                format="%.6f",
                help="Adjust the horizontal line to match the transit depth (flux level during transit)."
            )

            # Adjustable Y limits
            _, col, _ = st.columns([1, 2, 1])
            with col:
                ymin_input = st.text_input("Y-axis Min", value="0.95", key="ymin_input")
                ymax_input = st.text_input("Y-axis Max", value="1.05", key="ymax_input")
            try:
                ymin = float(ymin_input)
                ymax = float(ymax_input)
                if ymin >= ymax:
                    st.error("Y-axis Min must be less than Y-axis Max.")
                    ymin, ymax = None, None
            except ValueError:
                st.error("Invalid Y-axis limits. Please enter valid numbers.")
                ymin, ymax = None, None

            # Adjustable X limits
            _, col, _ = st.columns([1, 2, 1])
            with col:
                xmin_input = st.text_input("X-axis Min", value="-0.5", key="xmin_input")
                xmax_input = st.text_input("X-axis Max", value="0.5", key="xmax_input")
            try:
                xmin = float(xmin_input)
                xmax = float(xmax_input)
                if xmin >= xmax:
                    st.error("X-axis Min must be less than X-axis Max.")
                    xmin, xmax = None, None
            except ValueError:
                st.error("Invalid X-axis limits. Please enter valid numbers.")
                xmin, xmax = None, None

            # Plot the folded light curve with the adjustable depth line
            if ymin is not None and ymax is not None and xmin is not None and xmax is not None:
                fig, ax = plt.subplots(figsize=(8, 3))
                folded.plot(ax=ax)
                ax.axhline(depth_line, color='r', linestyle='--', label='Transit Depth')
                ax.set_ylim(ymin, ymax)
                ax.set_xlim(xmin, xmax)
                ax.legend()
                st.pyplot(fig)
            else:
                st.warning("Please fix the axis limits to display the plot.")

        # Inputs for habitability calculation
        _, col, _ = st.columns([1, 2, 1])
        with col:
            star_radius_input = st.text_input("Star Radius (R_sun)", value="2.0", key="star_radius_input")
            star_temp_input = st.text_input("Star Temperature (K)", value="3000.0", key="star_temp_input")
        albedo = st.slider(
            "Planet Albedo",
            min_value=0.0,
            max_value=1.0,
            value=0.3,
            step=0.01,
            help="Adjust the albedo (reflectivity) of the planet (0.0 to 1.0)."
        )

        # Convert inputs to floats with error handling
        try:
            star_radius = float(star_radius_input)
            if star_radius <= 0:
                st.error("Star Radius must be greater than 0.")
                star_radius = None
        except ValueError:
            st.error("Invalid input for Star Radius. Please enter a valid number.")
            star_radius = None

        try:
            star_temp = float(star_temp_input)
            if star_temp <= 0:
                st.error("Star Temperature must be greater than 0.")
                star_temp = None
        except ValueError:
            st.error("Invalid input for Star Temperature. Please enter a valid number.")
            star_temp = None

        # Calculate habitability
        if st.button("Calculate") and period is not None and star_radius is not None and star_temp is not None:
            depth = 1 - depth_line  # Depth is 1 - flux level
            if depth <= 0:
                st.error("Transit depth must be greater than 0 (adjust the depth line to be less than 1).")
            else:
                try:
                    # Perform the habitability calculation
                    out = calculate_habitability(depth, star_radius, star_temp, albedo, period)

                    # Display Summary and Results side by side in two columns
                    col1, col2 = st.columns(2)

                    with col1:
                        st.subheader("Summary of Input Parameters")
                        st.write(f"**Period**: {period:.4f} days")
                        st.write(f"**Transit Epoch (t0)**: {epoch:.4f}")
                        st.write(f"**Transit Depth (1 - Axhline)**: {depth:.6f}")
                        st.write(f"**Star Radius**: {star_radius:.2f} R_sun")
                        st.write(f"**Star Temperature**: {star_temp:.2f} K")
                        st.write(f"**Planet Albedo**: {albedo:.2f}")

                    with col2:
                        st.subheader("Results")
                        st.write(f"**Planet Radius (Solar Radii)**: {out['radius_sun']:.6f} R_sun")
                        st.write(f"**Planet Radius (Earth Radii)**: {out['radius_earth']:.2f} R_earth")
                        st.write(f"**Orbit Semi-Major Axis (Solar Radii)**: {out['semi_major_sun']:.2f} R_sun")
                        st.write(f"**Orbit Semi-Major Axis (AU)**: {out['semi_major_au']:.2f} au")
                        st.write(f"**Orbit Semi-Major Axis (Earth Radii)**: {out['semi_major_earth']:.2f} R_earth")
                        st.write(f"**Equilibrium Temperature**: {out['equilibrium_temp']:.2f} K")

                    # Atmospheric Analysis
                    st.subheader("Atmospheric Analysis")
                    gases, likelihoods = analyze_atmosphere(out['radius_earth'], out['equilibrium_temp'])

                    # Create a professional stacked area plot
                    fig, ax = plt.subplots(figsize=(10, 5))
                    try:
                        plt.style.use('seaborn-v0_8')  # Use Seaborn-like style
                    except:
                        plt.style.use('ggplot')  # Fallback to ggplot if seaborn-v0_8 is unavailable

                    # Create a smooth x-axis for the area plot
                    x = np.linspace(0, 1, 100)
                    y_stack = np.zeros((len(gases), len(x)))
                    colors = plt.cm.viridis(np.linspace(0, 1, len(gases)))

                    # Stack the likelihoods
                    for i, likelihood in enumerate(likelihoods):
                        y_stack[i] = likelihood * np.ones_like(x)
                        if i > 0:
                            y_stack[i] += y_stack[i - 1]

                    # Plot the stacked areas
                    for i, (gas, color) in enumerate(zip(gases, colors)):
                        if i == 0:
                            ax.fill_between(x, 0, y_stack[i], label=gas, color=color, alpha=0.8)
                        else:
                            ax.fill_between(x, y_stack[i - 1], y_stack[i], label=gas, color=color, alpha=0.8)

                    # Add annotations for each gas
                    for i, (gas, likelihood) in enumerate(zip(gases, likelihoods)):
                        y_pos = y_stack[i, 50] - (y_stack[i, 50] - (y_stack[i - 1, 50] if i > 0 else 0)) / 2
                        ax.text(0.5, y_pos, f"{gas}: {likelihood:.2f}", ha='center', va='center', fontsize=10,
                                color='white', bbox=dict(facecolor='black', alpha=0.5))

                    # Customize the plot
                    ax.set_xlim(0, 1)
                    ax.set_ylim(0, 1)
                    ax.set_xlabel("Normalized Scale", fontsize=12)
                    ax.set_ylabel("Cumulative Likelihood", fontsize=12)
                    ax.set_title("Estimated Atmospheric Composition", fontsize=14, pad=20)
                    ax.legend(loc='upper right', bbox_to_anchor=(1.15, 1), title="Gases")
                    ax.grid(True, linestyle='--', alpha=0.7)
                    plt.tight_layout()
                    st.pyplot(fig)

                    # Atmospheric Summary
                    st.subheader("Atmospheric Summary")
                    dominant_gases = [gas for gas, likelihood in zip(gases, likelihoods) if likelihood > 0.2]
                    if dominant_gases:
                        st.write(
                            f"Based on the planet's equilibrium temperature ({out['equilibrium_temp']:.2f} K) and radius ({out['radius_earth']:.2f} R_earth), the atmosphere is likely to contain the following gases: {', '.join(dominant_gases)}.")
                        if out['equilibrium_temp'] > 600:
                            st.write(
                                "The high temperature suggests a hydrogen-helium dominated atmosphere, typical of gas giants.")
                        elif 150 <= out['equilibrium_temp'] < 300:
                            st.write(
                                "The temperature range suggests potential habitability with possible water vapor, if liquid water exists.")
                        elif out['equilibrium_temp'] < 150:
                            st.write(
                                "The extremely low temperature indicates a cold atmosphere, likely dominated by methane and nitrogen.")
                    else:
                        st.write("No dominant gases identified. The atmosphere may be thin or non-existent.")

                    st.subheader("Atmospheric Escape Analysis")

                    try:
                        R_planet = out['radius_earth'] * R_earth.value  # Planet radius in meters
                        M_planet = out['radius_earth'] ** 3 * 5.972e24  # Estimate mass (Earth-like density)
                        T_planet = out['equilibrium_temp']  # Planet temperature in K

                        dominant_gas = gases[np.argmax(likelihoods)]
                        mu_dict = {
                            'H₂': 2.0,
                            'He': 4.0,
                            'CO₂': 44.01,
                            'N₂': 28.0,
                            'CH₄': 16.0,
                            'H₂O': 18.0
                        }
                        mu = mu_dict.get(dominant_gas, 1.0)  # Default to atomic hydrogen

                        # Physical constants
                        k_B = 1.380649e-23  # Boltzmann constant in J/K
                        m_p = 1.6726219e-27  # Proton mass in kg

                        # Calculate thermal velocity
                        v_th = np.sqrt(2 * k_B * T_planet / (mu * m_p))

                        # Calculate escape velocity at the planet's surface
                        v_esc = np.sqrt(2 * G.value * M_planet / R_planet)

                        # Calculate Jeans escape parameter
                        lambda_J = (v_esc / v_th) ** 2

                        # Calculate scale height
                        H = k_B * T_planet * R_planet ** 2 / (G.value * M_planet * mu * m_p)

                        # Simplified Jeans escape flux (particles/m²/s)
                        # Formula from Catling & Kasting (2017)
                        n_exobase = 1e12  # Approximate exobase number density (m^-3)
                        F_jeans = n_exobase * v_th * (1 + lambda_J) * np.exp(-lambda_J) / (2 * np.sqrt(np.pi))

                        # Mass loss rate (kg/s)
                        m_dot = 4 * np.pi * R_planet ** 2 * F_jeans * mu * m_p

                        # Is the atmosphere stable?
                        atmosphere_stable = lambda_J > 15  # Rule of thumb: stable if λ > 15

                        # Calculate the Bondi radius (critical parameter for escape)
                        c_s = np.sqrt(k_B * T_planet / (mu * m_p))  # Sound speed
                        bondi_radius = (G.value * M_planet) / (2 * c_s ** 2)

                        # Display results
                        st.write(f"""
                        **Atmospheric Escape Results**:
                        - **Mass Loss Rate**: {m_dot:.2e} kg/s
                        - **Jeans Parameter (λ)**: {lambda_J:.1f} {'(Stable)' if atmosphere_stable else '(Unstable)'}
                        - **Escape Velocity**: {v_esc:.1f} m/s
                        - **Thermal Velocity**: {v_th:.1f} m/s
                        - **Scale Height**: {H / 1000:.1f} km
                        - **Bondi Radius**: {bondi_radius / R_earth.value:.1f} R_earth
                        - **Dominant Atmospheric Component**: {dominant_gas} (μ = {mu:.1f})
                        """)

                        # Create radial grid for visualization
                        r_grid = np.linspace(R_planet, 10 * R_planet, 100)

                        # Calculate velocity profile based on isothermal Parker wind solution
                        # This is a simplified analytical solution
                        u_parker = c_s * np.sqrt(2 * np.log(r_grid / R_planet))

                        # Simple visualization
                        fig, ax = plt.subplots(figsize=(8, 4))

                        # Plot escape velocity at different radii
                        v_esc_r = np.sqrt(2 * G.value * M_planet / r_grid)
                        ax.plot(r_grid / R_planet, v_esc_r, 'b-', label='Escape Velocity')

                        # Plot sound speed (constant for isothermal atmosphere)
                        ax.plot(r_grid / R_planet, np.ones_like(r_grid) * c_s, 'g--', label='Sound Speed')

                        # Plot simplified Parker wind solution
                        ax.plot(r_grid / R_planet, u_parker, 'r-', label='Wind Velocity')

                        ax.set_xlabel('Radius (Planet Radii)')
                        ax.set_ylabel('Velocity (m/s)')
                        ax.axvline(bondi_radius / R_planet, color='purple', linestyle='--', label='Bondi Radius')
                        ax.legend()
                        ax.set_title('Atmospheric Escape Profile')
                        st.pyplot(fig)

                    except Exception as e:
                        st.error(f"Atmospheric escape calculation failed: {str(e)}")
                        st.write("""
                        Note: This analysis uses a simplified Jeans escape model.
                        For more accurate results, specific atmospheric composition data is needed.
                        """)

                except Exception as e:
                    st.error(f"Calculation error: {str(e)}")

    # Always show Back in sidebar for detail pages
    if st.session_state.page in ['exo_results', 'exo_analyze', 'hab_results', 'hab_analysis']:
        if st.sidebar.button("Back"):
            prev = 'exo_search' if st.session_state.page.startswith('exo') else 'hab_search'
            st.session_state.page = prev
            st.rerun()
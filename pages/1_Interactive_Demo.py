import streamlit as st
import numpy as np
import plotly.graph_objects as go
from utils import ringing_filter as rf
from utils import ui_components as ui

st.title("💡 Interactive Demo")
if "demo_data_import_complete" not in st.session_state:
    st.session_state.demo_data_import_complete = False
if "demo_potential_run_complete" not in st.session_state:
    st.session_state.demo_potential_run_complete = False
if "demo_maxima_found" not in st.session_state:
    st.session_state.demo_maxima_found = False
if "demo_broad_run_complete" not in st.session_state:
    st.session_state.demo_broad_run_complete = False
if "demo_tolerances_set" not in st.session_state:
    st.session_state.demo_tolerances_set = False
if "demo_refined_run_complete" not in st.session_state:
    st.session_state.demo_refined_run_complete = False

ion_mobility_type = "DT"
sns_deep_palette = ['#4c72b0',
                    '#dd8452',
                    '#55a868',
                    '#c44e52',
                    '#8172b3',
                    '#937860',
                    '#da8bc3',
                    '#8c8c8c',
                    '#ccb974',
                    '#64b5cd']

if not st.session_state.demo_data_import_complete:
    # 1. Load the Data
    demo_df = rf.load_demo_dataset("data/demo_data.csv")

    # Validation
    if "m/z" not in demo_df.columns:
        st.error(f"Expected columns 'm/z', but found {list(demo_df.columns)}")
        st.stop()

    # Determine feature attributes and normalize intensity
    feature_attributes = ['m/z', "RT", ion_mobility_type, "CCS"]
    feature_attributes = [c for c in demo_df.columns if c in feature_attributes]
    feature_info = demo_df[feature_attributes].copy()

    sample_names = [c for c in demo_df.columns if c not in feature_attributes]
    peak_intensity_info = demo_df[sample_names].copy()
    feature_info["max_intensity"] = peak_intensity_info.max(axis=1) / peak_intensity_info.max().max()

    example_df = demo_df.head(n=20).copy()
    
    del demo_df
    feature_info.rename(columns={"max_intensity": "Area"}, inplace=True)

    # Store in session state
    st.session_state['feature_info'] = feature_info
    st.session_state['peak_intensity_info'] = peak_intensity_info
    st.session_state["example_dataset"] = example_df
    st.session_state['feature_attributes'] = feature_attributes
    st.session_state.demo_data_import_complete = True
else:
    feature_info = st.session_state['feature_info']
    peak_intensity_info = st.session_state['peak_intensity_info']
    example_df = st.session_state["example_dataset"]
    feature_attributes = st.session_state['feature_attributes']

# --- SECTION 1: RAW DATA ---
st.subheader("Raw data visualization")

axis_titles = {
    "m/z": "m/z",
    "RT": "retention time (min.)",
    "DT": "drift time (ms)",
    "CCS": "CCS (Å²)",
    "max_intensity": "maximum intensity (a.u.)",
    }

with st.expander("🔬 View Demo Feature Table"):
        st.dataframe(example_df, width="content")

col1, col2 = st.columns([1, 2.5])
with col1:
    st.markdown("This plot shows the raw Time-of-Flight data before artifact filtering.")
    x_axis = st.selectbox("X-axis", options=feature_attributes + ["max_intensity"], index=0, help="Select the x-axis for the raw data plot.")
    y_axis = st.selectbox("Y-axis", options=feature_attributes + ["max_intensity"], index=len(feature_attributes), help="Select the y-axis for the raw data plot.")

with col2:
    fig_raw = go.Figure()
    fig_raw.add_trace(go.Scatter(
        x=feature_info[x_axis.replace("max_intensity", "Area")], 
        y=feature_info[y_axis.replace("max_intensity", "Area")], 
        name="Raw", 
        mode="markers",
        marker=dict(size=4, color="rgba(100, 150, 255, 0.6)")
    ))
    fig_raw.update_xaxes(title_text=axis_titles[x_axis])
    fig_raw.update_yaxes(title_text=axis_titles[y_axis])

    fig_raw.update_layout(template="plotly_dark", height=300, margin=dict(t=5, b=20))
    st.plotly_chart(fig_raw, width="content", key="raw_plot")

st.divider()

# --- SECTION 2: ALGORITHM CONFIGURATION ---
st.subheader(r"Strict artifact detection to find $\Delta \sqrt{m/z}$ values")

st.write("Configure the parameters used for feature grouping.")

# Using columns to organize the 6 parameters into two blocks
col1, col2, col3 = st.columns(3)

with col1:
    rt_tol = st.slider("RT tolerance (min.)", 0.0, 0.1, 0.015, step=0.001, format="%.3f", help="Retention time tolerance in minutes.")
    im_tol = st.slider("DT tolerance (ms)", 0.0, 0.5, 0.1, step=0.05, format="%.2f", help="Drift time tolerance in milliseconds.")

with col2:
    min_intensity = st.select_slider("Minimum precursor intensity (a.u.)", options=[1e-5, 5e-5, 1e-4, 5e-4, 1e-3, 5e-3, 1e-2, 5e-2, 1e-1, 5e-1, 1], value=1e-2, format_func=lambda x: f"{x:.0e}".replace("-0", "-"), help="Minimum intensity of precursor.")
    max_rel_intensity = st.slider("Maximum relative intensity (%)", 0, 100, 10, step=1, format="%d%%", help="Maximum relative intensity of the artifact compared to the precursor.", key="max_rel_intensity") / 100
with col3:
    min_correlation = st.slider("Minimum correlation", 0.0, 1.0, 0.9, step=0.01, help="Minimum correlation between precursor and artifact across samples.")

# Centralized Run Button
if st.button("🚀 Find potential artifacts", type="primary", use_container_width=True):
    with st.spinner("Finding potential artifacts..."):
        # Run the algorithm
        potential_df = rf.mark_potential_artifacts(
            feature_info,
            peak_intensity_info.T,
            [],
            rt_tol,
            im_tol,
            ion_mobility_type,
            min_intensity,
            max_rel_intensity,
            min_correlation,
            )

        drmz_conv, drmz_maxima = rf.find_drmz_maxima(
            potential_df.loc[potential_df["artifact"] == "yes", "drmz"],
            fwhm=2e-5,
            min_height=5,
            min_separation=1.5E-4,
            )
        
        # Store in session state
        st.session_state['potential_df'] = potential_df
        st.session_state['drmz_conv'] = drmz_conv
        st.session_state['drmz_maxima'] = drmz_maxima
        st.session_state.demo_potential_run_complete = True
        st.session_state.demo_maxima_found = False
        st.session_state.demo_broad_run_complete = False
        st.session_state.demo_tolerances_set = False
        st.session_state.demo_refined_run_complete = False
# --- SECTION 3: RESULTS ---
if st.session_state.demo_potential_run_complete:
    plot_container = st.container()

    potential_df = st.session_state['potential_df']
    drmz_conv = st.session_state['drmz_conv']
    drmz_maxima = st.session_state['drmz_maxima']
    
    # --- SECTION 4: ALGORITHM CONFIGURATION ---
    st.write("Configure the parameters used for identifying the maxima.")

    # Using columns to organize the 6 parameters into two blocks
    col1, col2, col3 = st.columns(3)

    with col1:
        fwhm = st.slider("FWHM", 0.0, 10e-5, 2e-5, step=0.5e-5, format="%.1e", help="Full-width at half maximum for Gaussian convolution.")

    with col2:
        min_counts = st.slider("Minimum counts", 0, int(drmz_conv[:, 1].max().round()) + 1, 5, step=1, help="Minimum counts of maxima.")

    with col3:
        min_separation = st.slider("Minimum separation", 0.0, 10E-4, 1.5E-4, step=0.5E-4, format="%.1e", help="Minimum separation between maxima.")

    # Centralized Run Button
    if st.button("🚀 Find maxima", type="primary", use_container_width=True):
        with st.spinner("Finding maxima..."):
            drmz_conv, drmz_maxima = rf.find_drmz_maxima(
                potential_df.loc[potential_df["artifact"].isin(["no", "precursor"]) == False, "drmz"],
                fwhm=fwhm,
                min_height=min_counts,
                min_separation=min_separation,
                )
            
            potential_df, fit_parameters = rf.segment_artifacts(
                potential_df, drmz_maxima[:, 0],
                drmz_tol=min_separation / 2,
                confidence_interval=0.99,
                max_iter=100,
                )
            
            # Update in session state
            st.session_state['potential_df'] = potential_df
            st.session_state['drmz_conv'] = drmz_conv
            st.session_state['drmz_maxima'] = drmz_maxima
            st.session_state.demo_maxima_found = True
            st.session_state.demo_broad_run_complete = False
            st.session_state.demo_tolerances_set = False
            st.session_state.demo_refined_run_complete = False
            
        st.info(f"Maxima updated. Assigned {drmz_maxima.shape[0]} rings to the TOF-MS data.".replace("1 rings", "1 ring"))
    
    artifact_colors = sns_deep_palette * (1 + drmz_maxima.shape[0] // len(sns_deep_palette))
    with plot_container:
        fig = ui.plot_artifact_maxima(potential_df, drmz_conv, drmz_maxima, maxima_found=st.session_state.demo_maxima_found, artifact_colors=artifact_colors)
        st.plotly_chart(fig, width="content", key="maxima_plot")

else:
    st.info(r"Configuration set. Click the find $\Delta \sqrt{m/z}$ values.")

st.divider()

# --- SECTION 3: ALGORITHM CONFIGURATION ---
if st.session_state.demo_maxima_found:
    st.subheader(r"Broad artifact detection to refine all parameters")

    st.write("Reconfigure the parameters used for broader feature grouping.")

    col1, col2, col3 = st.columns(3)
    with col1:
        rt_tol_broad = st.slider("RT tolerance (min.)", 0.0, 0.1, 0.02, step=0.005, format="%.3f", help="Retention time tolerance in minutes.", key="rt_tol_broad")
        im_tol_broad = st.slider("DT tolerance (ms)", 0.0, 0.5, 0.2, step=0.1, format="%.2f", help="Drift time tolerance in milliseconds.", key="dt_tol_broad")
    with col2:
        min_intensity_broad = st.select_slider(
            "Minimum precursor intensity (a.u.)",
            options=np.append(np.ravel(np.outer(np.array([1.0, 2.5, 5, 7.5]), np.power(10.0, np.array([-5, -4, -3, -2, -1]))), order="F"), 1.0),
            value=1e-4, format_func=lambda x: f"{x:.1e}".replace("-0", "-").replace("+0", "+"),
            help="Minimum intensity of precursor.", key="min_intensity_broad")
        max_rel_intensity_broad = st.slider("Maximum relative intensity (%)", 0, 100, 10, step=1, format="%d%%", help="Maximum relative intensity of the artifact compared to the precursor.", key="max_rel_intensity_broad") / 100
    with col3:
        min_correlation_broad = st.slider("Minimum correlation",0.0, 1.0, 0.2, step=0.01, help="Minimum correlation between precursor and artifact across samples.", key="min_correlation_broad")
        drmz_tol_broad = st.slider(r"$\Delta \sqrt{m/z}$ tolerance", 0.0, 50e-5, 15e-5, step=2.5e-5, format="%.1e", help="Tolerance for Δ√(m/z) values.", key="drmz_tol_broad")
    
    if st.button("🚀 Run broad artifact detection", type="primary", use_container_width=True, key="broad_run_button"):
        with st.spinner("Running broad artifact detection..."):
            drmz_maxima = st.session_state['drmz_maxima']

            # Re-run the algorithm with broad parameters
            broad_df = rf.mark_artifacts(
                feature_info,
                drmz_maxima[:, 0],
                peak_intensity_info.T,
                rt_tol_broad,
                im_tol_broad,
                ion_mobility_type,
                min_intensity_broad,
                max_rel_intensity_broad,
                min_correlation_broad,
                drmz_tol=drmz_tol_broad,
                )
            
            # Update in session state
            st.session_state['broad_df'] = broad_df
            st.session_state.demo_broad_run_complete = True
            st.session_state.demo_tolerances_set = False
            st.session_state.demo_refined_run_complete = False
            st.success("Broad artifact detection run complete.")

if st.session_state.demo_broad_run_complete:
    main_col, col3 = st.columns([2, 1])
    with main_col:
        tol_plot_container = st.container()

    with col3:
        st.write("Reconfigure the parameters used for refined feature grouping.")
        st.markdown("----")
        rt_tol_refined = st.slider("RT tolerance (min.)", 0.0, 0.1, 0.015, step=0.001, format="%.3f", help="Retention time tolerance in minutes.", key="rt_tol_refined")
        dt_tol_refined = st.slider("DT tolerance (ms)", 0.0, 0.5, 0.1, step=0.01, format="%.2f", help="Drift time tolerance in milliseconds.", key="dt_tol_refined")
        drmz_tol_refined = st.slider(r"$\Delta \sqrt{m/z}$ tolerance", 0.0, 50e-5, 10e-5, step=.5e-5, format="%.1e", help="Tolerance for Δ√(m/z) values.", key="drmz_tol_refined")
        iso_offset_refined = st.slider(r"$\Delta \sqrt{m/z}$ offset for M+1", -20e-5, 20e-5, 5e-5, step=.5e-5, format="%.1e", help="Tolerance for Δ√(m/z) values.", key="iso_offset_refined")
        min_intensity_refined = st.select_slider(
            "Minimum precursor intensity (a.u.)",
            options=np.append(np.ravel(np.outer(np.array([1.0, 2.5, 5, 7.5]), np.power(10.0, np.array([-5, -4, -3, -2, -1]))), order="F"), 1.0),
            value=1e-3, format_func=lambda x: f"{x:.1e}".replace("-0", "-").replace("+0", "+"),
            help="Minimum intensity of precursor.", key="min_intensity_refined")
        max_rel_intensity_refined = st.slider("Maximum relative intensity (%)", 0, 100, 5, step=1, format="%d%%", help="Maximum relative intensity for artifact consideration.", key="max_rel_intensity_refined") / 100
        min_correlation_refined = st.slider("Minimum correlation", 0.0, 1.0, 0.5, step=0.01, help="Minimum correlation between precursor and artifact across samples.", key="min_correlation_refined")
        refined_tols = {
            "rt_tol": rt_tol_refined,
            "im_tol": dt_tol_refined,
            "drmz_tol": drmz_tol_refined,
            "iso_offset": iso_offset_refined,
            "min_intensity": min_intensity_refined,
            "max_rel_intensity": max_rel_intensity_refined,
            "min_correlation": min_correlation_refined,
        }
        if st.button("Check parameters", type="primary", use_container_width=True):
            st.session_state.demo_tolerances_set = True
            st.session_state['refined_tols'] = refined_tols
            st.session_state.demo_refined_run_complete = False
    
    if st.button("🚀 Run refined artifact detection", type="primary", use_container_width=True, disabled=not st.session_state.demo_tolerances_set):
        with st.spinner("Running artifact detection..."):
            drmz_maxima = st.session_state['drmz_maxima']

            # Re-run the algorithm with broad parameters
            refined_df = rf.mark_artifacts(
                feature_info,
                drmz_maxima[:, 0],
                peak_intensity_info.T,
                rt_tol_refined,
                dt_tol_refined,
                ion_mobility_type,
                min_intensity_refined,
                max_rel_intensity_refined,
                min_correlation_refined,
                drmz_tol=drmz_tol_refined,
                iso_offset=iso_offset_refined,
                )
            
            # Update in session state
            st.session_state['refined_df'] = refined_df
            st.session_state.demo_refined_run_complete = True

    with tol_plot_container:
        broad_df = st.session_state['broad_df']
        artifacts = [f"{i+1}th ring".replace("1th", "1st").replace("2th", "2nd").replace("3th", "3rd") for i in range(st.session_state['drmz_maxima'].shape[0])]
        artifact_colors = sns_deep_palette[1:len(artifacts)+1] + ["#FFFFFF"]
        if "M+1" in broad_df["artifact"].unique():
            artifacts += ["M+1"]
        
        fig = ui.plot_tolerance_refinement(
            broad_df,
            artifacts,
            artifact_colors,
            ion_mobility_type=ion_mobility_type,
            tolerances=st.session_state['refined_tols'] if st.session_state.demo_tolerances_set else None,
            tolerances_set=st.session_state.demo_tolerances_set,
        )
        st.plotly_chart(fig, width="content", key="tolerance_plot")

if st.session_state.demo_refined_run_complete == True:
    feature_attributes = st.session_state['feature_attributes']
    refined_df = st.session_state['refined_df']
    feature_info = st.session_state['feature_info']
    peak_intensity_info = st.session_state['peak_intensity_info']

    export_df = feature_info[feature_attributes].join(refined_df[["artifact", "precursor"]]).join(peak_intensity_info)
    st.success("Artifact detection complete! You can now view the filtered feature table below.")
    
    
    with st.expander("🔬 View Filtered Feature Table"):
        st.dataframe(export_df.head(50), width="content")
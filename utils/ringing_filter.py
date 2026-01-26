import time
import streamlit as st
from molmass import ELEMENTS
import pandas as pd
import numpy as np
import scipy as sc

@st.cache_data
def load_demo_dataset(file_path="demo_data.csv"):
    """Loads the CSV once and keeps it in memory."""
    df = pd.read_csv(file_path, index_col=0)
    return df

def segment_artefacts(df, drmz_maxima, drmz_tol=1.5E-4, confidence_interval=0.99, max_iter=100):
    df.loc[(df["artefact"] != "no") & (df["artefact"] != "precursor"), "artefact"] = "outlier"
    fit_parameters = {}
    print(f"\nring counts    drmz ddrmz std ppm std")
    #print(f"------------------------------------")
    n_artefacts = 0
    total_ppm = []
    for ring_number, drmz_maximum in enumerate(drmz_maxima):
        label = f"{ring_number + 1}th ring".replace("1th", "1st").replace("2th", "2nd").replace("3th", "3rd")
        df_selected = df.loc[(drmz_maximum - drmz_tol <= df["drmz"]) & (df["drmz"] <= drmz_maximum + drmz_tol), "drmz"]
        i = 0
        n_outliers = 0
        while i < max_iter:
            residuals = df_selected - df_selected.mean()
            outliers = df_selected.index[np.abs(residuals / np.std(residuals)) > sc.stats.norm.ppf(1 - (1 - confidence_interval) / 2)]
            n_outliers += len(outliers)
            if len(outliers) == 0: # no new outliers found
                i = max_iter # this is the final round
            else:
                df_selected.drop(index=outliers, inplace=True)
            i += 1
            if i == max_iter:
                print("Max iter reached. Probably not good.")
        df.loc[df_selected.index, "artefact"] = label
        drmz_mean = df_selected.mean()
        drmz_std = df_selected.std()
        fit_parameters[label] = {"mean": drmz_mean, "std": drmz_std}

        df_selected = df.loc[df_selected.index]
        ddmz = df_selected["m/z"] - (df_selected["rmz"] - df_selected["drmz"] + drmz_mean) ** 2
        ppm = ddmz / df_selected["m/z"] * 1E6
        
        print(f"{'r' + str(ring_number + 1):>3} {len(df_selected):>7} {drmz_mean:.5f} {drmz_std:.7f} {ppm.std():>7.2f}")# {n_outliers:>2}")
        n_artefacts += len(df_selected)
        total_ppm.append(ppm)

    print(f"total {n_artefacts:>5}                   {pd.concat(total_ppm).std():>7.2f}")# {n_outliers:>2}")
    print(f"from {df[df['artefact'] == 'precursor'].shape[0]} precursors")

    # do some checking of isotope numbers.
    if "Isotope Distribution" in df.columns:
        df["num_isotopes"] = df["Isotope Distribution"].apply(count_isotopes)
        isotopic_artefacts = 0
        for i in range(len(drmz_maxima)):
            temp_df = df[df["artefact"] == f"{i+1}th ring".replace("1th", "1st").replace("2th", "2nd").replace("3th", "3rd")]
            isotopic_artefacts += temp_df["num_isotopes"].sum()
    
        print(f"total isotopic artefacts: {isotopic_artefacts}")
    print("")
    return df, fit_parameters

def find_drmz_maxima(drmz_df, fwhm=2e-5, min_height=5, min_separation=1.5e-4):
    # create drmz axis
    drmzs = drmz_df.values
    drmz_step = fwhm / 5
    drmz_extend = fwhm * 3
    drmz_axis = np.arange(
        start=round(drmzs.min() / drmz_step) * drmz_step - drmz_extend,
        stop=round(drmzs.max() / drmz_step) * drmz_step + drmz_extend,
        step=drmz_step,
        )

    # convolution with gaussians
    sigma = fwhm / (2 * np.sqrt(2 * np.log(2)))
    drmz_axis_mat = np.tile(drmz_axis, (drmzs.shape[0], 1))
    drmz_conv = np.exp(-0.5 * ((drmz_axis_mat - drmzs[:, None])/sigma)**2)
    drmz_conv = drmz_conv.sum(axis=0)
    #drmz_conv /= drmz_conv.max()
    drmz_conv = np.vstack((drmz_axis, drmz_conv)).T

    # peak detection
    maxima, heights = sc.signal.find_peaks(
        x=drmz_conv[:, 1],
        height=min_height,
        distance=int(round(min_separation / drmz_step)),
        )
    
    # compile maxima
    drmz_maxima = np.vstack((drmz_conv[maxima, 0], heights["peak_heights"])).T
    return drmz_conv, drmz_maxima

def count_isotopes(isotope_distribution_str):
    if pd.isna(isotope_distribution_str):
        return 0
    isotopes = isotope_distribution_str.split("-")
    return len(isotopes) - 1  # Subtract 1 to exclude the monoisotopic peak


def compare_features(i, j, arrays, results, params, drmz, im_mode="DT", do_iso=False):
    """
    Optimized logic-only function. 
    Checks RT, DT/CCS, and Area 'gates' first, then evaluates mass rings.
    """
    # 1. Gate Checks
    if (abs(arrays['rt'][j] - arrays['rt'][i]) < params['rt_tol'] and 
        abs(arrays[im_mode.lower()][j] - arrays[im_mode.lower()][i]) < params['im_tol'] and 
        arrays['area'][j] / arrays['area'][i] < params['max_rel_area']):
        
        iso_off = params['iso_offset'] if do_iso else 0
        
        # 2. Loop through mass rings
        for ring_idx, f_mean in enumerate(params['fitted_means']):
            if abs(drmz - (f_mean + iso_off)) < params['drmz_tol'][ring_idx]:
                
                # 3. Correlation Check
                corr_val = 0.0
                if arrays['z_data'] is not None:
                    corr_val = np.dot(arrays['z_data'][i], arrays['z_data'][j]) / params['n_samples']
                
                if corr_val > params['min_corr']:
                    # 1. Determine the "Ring" label first
                    if len(params['fitted_means']) == 1 and params['fitted_means'][0] == 0:
                        label = "yes"
                    else:
                        # This identifies WHICH ring it belongs to, regardless of if it's an isotope or not
                        label = f"{ring_idx + 1}th ring".replace("1th", "1st").replace("2th", "2nd").replace("3th", "3rd")
                    
                    # 2. Assign labels to results
                    results['precursor'][j] = arrays['features'][i]
                    results['artefact'][j] = label # Even isotopes get "1st ring" etc. temporarily
                    results['artefact'][i] = "precursor"

                    if do_iso:
                        # This flag is what we use to distinguish Loop 2 results from Loop 1
                        results['dmz_iso'][j] = arrays['mz'][j] - (arrays['mz'][i] + params['c_spacing'])
                        
                        # Predict M+1 mz based on precursor and the ring offset
                        pred_mz_iso = (np.sqrt(arrays['mz'][i] + params['c_spacing']) + f_mean)**2
                        results['ddmz_iso'][j] = arrays['mz'][j] - pred_mz_iso
                        results['ppm_iso'][j] = (results['ddmz_iso'][j] / arrays['mz'][j]) * 1e6
                    
                    # Metrics Collection
                    results['drmz'][j] = arrays['rmz'][j] - arrays['rmz'][i]
                    results['dmz'][j] = arrays['mz'][j] - arrays['mz'][i]
                    
                    results['ddrmz'][j] = drmz - f_mean
                    results['drt'][j] = arrays['rt'][j] - arrays['rt'][i]
                    results['dccs'][j] = arrays['ccs'][j] - arrays['ccs'][i]
                    results['ddt'][j] = arrays['dt'][j] - arrays['dt'][i]
                    results['rel_area'][j] = arrays['area'][j] / arrays['area'][i]
                    results['corr'][j] = corr_val
                    
                    # PPM for calibrated path
                    ddmz = arrays['mz'][j] - (arrays['rmz'][i] + f_mean)**2
                    results['ppm'][j] = (ddmz / arrays['mz'][j]) * 1e6
                    
                    return True
    return False

def mark_artefacts(df, fitted_means, peak_data, rt_tol=0.02, im_tol=0.3, im_mode="DT",
                        min_area=10, max_rel_area=0.2, min_corr=0.2, drmz_tol=0.015, iso_offset=0):
    
    if "Feature" not in df.columns:
        df = df.rename_axis("Feature").reset_index()
    
    # --- Initialization ---
    ordered_features = df["Feature"].tolist()
    df = df.sort_values(["m/z"]).reset_index(drop=True)
    
    rmz_search_col = "rmz"
    df["rmz"] = np.sqrt(df["m/z"])

    C_SPACING = 1.00335
    if isinstance(drmz_tol, (float, int)):
        drmz_tol = np.full(len(fitted_means), drmz_tol)

    # --- Z-Score peak_data ---
    z_data, n_samples = None, 1
    if isinstance(peak_data, pd.DataFrame):
        subset = peak_data[df["Feature"]].values.astype(np.float32)
        stds = np.std(subset, axis=0)
        stds[stds == 0] = 1.0
        z_data = ((subset - np.mean(subset, axis=0)) / stds).T
        n_samples = subset.shape[0]

    # --- Build lookup arrays ---
    arrays = {
        'mz': df["m/z"].values, 'rmz': df["rmz"].values,
        'area': df["Area"].values, 'features': df["Feature"].values,
        'rt': df["RT"].values if "RT" in df.columns else np.zeros(len(df)),
        'ccs': df["CCS"].values if "CCS" in df.columns else np.zeros(len(df)),
        'dt': df["DT"].values if "DT" in df.columns else np.zeros(len(df)),
        'z_data': z_data, 'n_samples': n_samples
    }
    rmz_vals = df[rmz_search_col].values

    params = {
        'fitted_means': fitted_means, 'drmz_tol': drmz_tol, 'rt_tol': rt_tol, 
        'im_tol': im_tol, 'max_rel_area': max_rel_area, 'min_corr': min_corr,
        'n_samples': n_samples, 'iso_offset': iso_offset, 'c_spacing': C_SPACING
    }

    res_dict = {
        'artefact': np.array(["no"] * len(df), dtype='object'),
        'precursor': np.array([""] * len(df), dtype='object'),
        'drmz': np.zeros(len(df)), 'dmz': np.zeros(len(df)),
        'ddrmz': np.zeros(len(df)), 'drt': np.zeros(len(df)), 'dccs': np.zeros(len(df)),
        'ddt': np.zeros(len(df)), 'rel_area': np.zeros(len(df)), 'corr': np.zeros(len(df)),
        'dmz_iso': np.zeros(len(df)), 'ddmz_iso': np.zeros(len(df)), 
        'ppm': np.zeros(len(df)), 'ppm_iso': np.zeros(len(df))
    }

    # --- LOOP 1: Standard Artefacts ---
    t0 = time.time()
    for i in range(len(df) - 1):
        if arrays['area'][i] <= min_area or res_dict['artefact'][i] != "no":
            continue
        j_start = np.searchsorted(rmz_vals, rmz_vals[i] + (fitted_means[0] - drmz_tol[0]))
        j_end = np.searchsorted(rmz_vals, rmz_vals[i] + (fitted_means[-1] + drmz_tol[-1]), side='right')
        for j in range(j_start, j_end):
            if i == j or res_dict['artefact'][j] != "no": continue
            drmz = rmz_vals[j] - rmz_vals[i]
            compare_features(i, j, arrays, res_dict, params, drmz, im_mode=im_mode, do_iso=False)
    t1 = time.time()

    # --- LOOP 2: M+1 Isotope Artefacts ---
    precursor_idx = np.where(res_dict['artefact'] == "precursor")[0]
    for i in precursor_idx:
        rmz_iso = np.sqrt(arrays['mz'][i] + C_SPACING)
        j_start = np.searchsorted(rmz_vals, rmz_iso + (fitted_means[0] - drmz_tol[0] + iso_offset))
        j_end = np.searchsorted(rmz_vals, rmz_iso + (fitted_means[-1] + drmz_tol[-1] + iso_offset), side='right')
        for j in range(j_start, j_end):
            if i == j or res_dict['artefact'][j] != "no": continue
            drmz = rmz_vals[j] - rmz_iso
            compare_features(i, j, arrays, res_dict, params, drmz, im_mode=im_mode, do_iso=True)
    t2 = time.time()

    # Map results to DataFrame
    for key, val in res_dict.items(): df[key] = val
    
    # Identify which artefacts were created in Loop 2 (M+1 search)
    iso_mask = df["dmz_iso"] != 0.0
    iso_df = df[iso_mask].copy()

    # --- UPDATED PRINTING LOGIC ---
    print(f"Time taken to mark artefacts: {t1 - t0:.3f} s")
    print(f"Time taken to mark deviating isotopes: {t2 - t1:.3f} s")

    num_m1 = iso_df.shape[0]
    num_precursors = df[df['artefact'] == 'precursor'].shape[0]
    # Total artefacts excluding precursors
    num_artefacts = df[~df['artefact'].isin(['no', 'precursor'])].shape[0]

    print(f"{num_artefacts} artefacts (of which {num_m1} are M+1 isotopes) from {num_precursors} precursors")
    print(f"Artefacts are {round(num_artefacts / df.shape[0] * 100, 1)}% of total features ({df.shape[0]})")

    # Table 1: Standard Artefact Stats (Excluding M+1 from standard ring stats)
    print(f"\ncat    drmz counts ddrmz std ppm std")
    print(f"tot       - {df.shape[0]:>6}         -       -")
    print(f"pre       - {num_precursors:>6}         -       -")

    for i, mean in enumerate(fitted_means):
        label = f"{i+1}th ring".replace("1th", "1st").replace("2th", "2nd").replace("3th", "3rd")
        # EXCLUSION: Filter for the label BUT ensure it's NOT an M+1 artefact
        temp_df = df[(df["artefact"] == label) & (~iso_mask)]
        print(f"{'r' + str(i+1):>3} {mean:>7.5f} {temp_df.shape[0]:>6} {temp_df['drmz'].std():>8.7f} {temp_df['ppm'].std():>7.2f}")
    
    print(f"M+1       - {num_m1:>6}         - {iso_df['ppm_iso'].std():>7.2f}\n")

    # Table 2: Isotope-Specific Artefact Stats (Only M+1 loop results)
    print(f"cat    drmz counts ddrmz std ppm std")
    for i, mean in enumerate(fitted_means):
        label = f"{i+1}th ring".replace("1th", "1st").replace("2th", "2nd").replace("3th", "3rd")
        temp_iso_df = iso_df[iso_df["artefact"] == label]
        print(f"{'r' + str(i+1):>3} {mean:>7.5f} {temp_iso_df.shape[0]:>6} {temp_iso_df['drmz'].std():>8.7f} {temp_iso_df['ppm'].std():>7.2f}")

    if "Isotope Distribution" in df.columns:
        df["num_isotopes"] = df["Isotope Distribution"].apply(count_isotopes)
        # Sum only standard artefacts found in Loop 1
        std_artefact_df = df[(~df['artefact'].isin(['no', 'precursor'])) & (~iso_mask)]
        isotopic_artefacts = std_artefact_df["num_isotopes"].sum()
        print(f"total isotopic artefacts: {int(isotopic_artefacts)}\n")

    # Set label to M+1 for the returned dataframe
    df.loc[iso_mask, "artefact"] = "M+1"

    df = df.set_index("Feature")
    return df.loc[ordered_features]

import numpy as np
import pandas as pd
import time

def mark_potential_artefacts(df, peak_data, fitted_means=None, rt_tol=0.05, 
                                 im_tol=0.3, im_mode="DT", min_area=1e-4, max_rel_area=0.1, min_corr=0.9, 
                                 drmz_min=0.0001, drmz_max=0.02, iso_offset=0):
    
    if "Feature" not in df.columns:
        df = df.rename_axis("Feature").reset_index()
    
    # --- 1. Setup & Feature Ordering ---
    ordered_features = df["Feature"].tolist()
    df = df.sort_values(["m/z"]).reset_index(drop=True)
    
    rmz_search_col = "rmz"
    df["rmz"] = np.sqrt(df["m/z"])
    #for col in ["RT", "DT", "CCS"]:
    #    if col not in df.columns: df[col] = 0.0
    
    if fitted_means is None or len(fitted_means) == 0:
        actual_fitted_means = [0.0]
        actual_drmz_tol = [np.inf]
    else:
        actual_fitted_means = fitted_means
        actual_drmz_tol = np.full(len(fitted_means), np.inf)

    # --- 2. Z-Score Peak Data ---
    z_data, n_samples = None, 1
    if isinstance(peak_data, pd.DataFrame):
        subset = peak_data[df["Feature"]].values.astype(np.float32)
        stds = np.std(subset, axis=0)
        stds[stds == 0] = 1.0
        z_data = ((subset - np.mean(subset, axis=0)) / stds).T
        n_samples = subset.shape[0]

    # --- 3. Build NumPy Views & Containers ---
    arrays = {
        'mz': df["m/z"].values, 'rmz': df["rmz"].values,
        'area': df["Area"].values, 'features': df["Feature"].values,
        'rt': df["RT"].values if "RT" in df.columns else np.zeros(len(df)),
        "dt": df["DT"].values if "DT" in df.columns else np.zeros(len(df)),
        'ccs': df["CCS"].values if "CCS" in df.columns else np.zeros(len(df)),
        'z_data': z_data, 'n_samples': n_samples
    }
    rmz_vals = df[rmz_search_col].values

    params = {
        'fitted_means': actual_fitted_means, 'drmz_tol': actual_drmz_tol,
        'rt_tol': rt_tol, 'im_tol': im_tol, 'max_rel_area': max_rel_area, 
        'min_corr': min_corr, 'n_samples': n_samples, 'iso_offset': iso_offset
    }

    res_dict = {
        'artefact': np.array(["no"] * len(df), dtype='object'),
        'precursor': np.array([""] * len(df), dtype='object'),
        'drmz': np.zeros(len(df)), 'dmz': np.zeros(len(df)),
        'ddrmz': np.zeros(len(df)), 'drt': np.zeros(len(df)), 'dccs': np.zeros(len(df)),
        'ddt': np.zeros(len(df)), 'rel_area': np.zeros(len(df)), 'corr': np.zeros(len(df)),
        'ppm': np.zeros(len(df))
    }

    # --- 4. Main Search Loop ---
    t0 = time.time()
    n_features = len(df)
    for i in range(n_features - 1):
        if arrays['area'][i] <= min_area or res_dict['artefact'][i] != "no":
            continue
        
        j_start = np.searchsorted(rmz_vals, rmz_vals[i] + drmz_min)
        j_end = np.searchsorted(rmz_vals, rmz_vals[i] + drmz_max, side='right')
        
        for j in range(j_start, j_end):
            if i == j or res_dict['artefact'][j] != "no":
                continue
            drmz = rmz_vals[j] - rmz_vals[i]
            compare_features(i, j, arrays, res_dict, params, drmz, im_mode=im_mode, do_iso=False)
            
    t1 = time.time()

    # --- 5. Assembly & Reporting ---
    for key, val in res_dict.items():
        df[key] = val

    # --- ORIGINAL PRINTING LOGIC ---
    print(f"Time taken to mark potential artefacts: {t1 - t0:.3f} s")
    
    num_precursors = df[df['artefact'] == 'precursor'].shape[0]
    num_artefacts = df[~df['artefact'].isin(['no', 'precursor'])].shape[0]

    print(f"{num_artefacts} artefacts from {num_precursors} precursors")
    print(f"Artefacts are {round(num_artefacts / df.shape[0] * 100, 1)}% of total features ({df.shape[0]})")

    if "Isotope Distribution" in df.columns:
        df["num_isotopes"] = df["Isotope Distribution"].apply(count_isotopes)
        isotopic_artefacts = df[~df['artefact'].isin(['no', 'precursor'])]["num_isotopes"].sum()
        print(f"total isotopic artefacts: {int(isotopic_artefacts)}")

    df = df.set_index("Feature")
    return df.loc[ordered_features]

import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np

def plot_artifact_maxima(potential_df, drmz_conv, drmz_maxima, maxima_found, artifact_colors):
    fig_res = make_subplots(rows=1, cols=2, shared_yaxes=True, column_widths=[0.7, 0.3])
    
    # Plot the clean signal
    fig_res.add_trace(go.Scatter(
        x=potential_df[potential_df["artifact"] == "no"]["m/z"], 
        y=potential_df[potential_df["artifact"] == "no"]["drmz"], 
        name="clustered features", 
        mode="markers",
        marker=dict(size=4, color="#BBB7B7", opacity=0.6),
    ), row=1, col=1)

    fig_res.add_trace(go.Scatter(
        x=potential_df[potential_df["artifact"] == "precursor"]["m/z"], 
        y=potential_df[potential_df["artifact"] == "precursor"]["drmz"], 
        name="clustered features", 
        mode="markers",
        marker=dict(size=4, color=artifact_colors[0], opacity=0.6),
    ), row=1, col=1)

    if maxima_found:
        fig_res.add_trace(go.Scatter(
                x=potential_df[potential_df["artifact"] == "outlier"]["m/z"], 
                y=potential_df[potential_df["artifact"] == "outlier"]["drmz"], 
                name="clustered features", 
                mode="markers",
                marker=dict(size=4, color="#BBB7B7", opacity=0.6),
            ), row=1, col=1)
        
        for i in range(drmz_maxima.shape[0]):
            fig_res.add_trace(go.Scatter(
                x=potential_df[potential_df["artifact"] == f"{i+1}th ring".replace("1th", "1st").replace("2th", "2nd").replace("3th", "3rd")]["m/z"], 
                y=potential_df[potential_df["artifact"] == f"{i+1}th ring".replace("1th", "1st").replace("2th", "2nd").replace("3th", "3rd")]["drmz"], 
                name="clustered features", 
                mode="markers",
                marker=dict(size=4, color=artifact_colors[i+1], opacity=0.6),
            ), row=1, col=1)
    else:
        fig_res.add_trace(go.Scatter(
            x=potential_df[potential_df["artifact"] == "yes"]["m/z"], 
            y=potential_df[potential_df["artifact"] == "yes"]["drmz"], 
            name="clustered features", 
            mode="markers",
            marker=dict(size=4, color="#5F52DA", opacity=0.6),
        ), row=1, col=1)

    fig_res.add_trace(go.Scatter(
        x=np.concatenate([drmz_conv[:, 1], np.zeros(len(drmz_conv[:, 1]))]), # x1 then reversed x2
        y=np.concatenate([drmz_conv[:, 0], drmz_conv[:, 0][::-1]]),   # y then reversed y
        name="convolution",
        fill='toself',
        fillcolor="#5F52DA",
        line=dict(width=0.1, color="#5F52DA"),
    ), row=1, col=2)

    # RIGHT SUBPLOT (Conditional Update)
    if maxima_found:
        for i in range(drmz_maxima.shape[0]):
            fig_res.add_trace(
                go.Scatter(
                x=[drmz_maxima[i, 1]],
                y=[drmz_maxima[i, 0]],
                name="maxima",
                mode='markers',
                marker=dict(size=8, color=artifact_colors[i+1], symbol="circle-open-dot"),
            ), row=1, col=2)

    fig_res.update_xaxes(title_text="m/z", row=1, col=1)
    fig_res.update_yaxes(title_text="Δ√(m/z)", row=1, col=1)
    fig_res.update_xaxes(title_text="aligned counts", row=1, col=2)
    
    fig_res.update_layout(template="plotly_dark", showlegend=False, height=400, margin=dict(t=20, b=20))
    return fig_res

def plot_tolerance_refinement(broad_df, artifacts, artifact_colors, ion_mobility_type, tolerances, tolerances_set):

    if ion_mobility_type == "DT":
        im_value = "ddt"
        im_title = "drift time error"
        im_xlabel = r"ΔDT (ms)"
    elif ion_mobility_type == "CCS":
        im_value = "dccs"
        im_title = "CCS error"
        im_xlabel = r"ΔCCS (Å²)"

    panels = {
        "RT": {"value": "drt", "title": "retention time error", "xlabel": r"ΔRT (min.)"},
        ion_mobility_type: {"value": im_value, "title": im_title, "xlabel": im_xlabel},
        "drmz": {"value": "ddrmz", "title": r"Δ√(m/z) error", "xlabel": r"ΔΔ√(m/z)"},
        "relative area": {"value": "rel_area", "title": "relative area", "xlabel": "relative peak area of artifact to precursor"},
        "precursor area": {"value": "Area", "title": "precursor area", "xlabel": "precursor peak area (a.u.)"},
        "correlation": {"value": "corr", "title": "correlation", "xlabel": "correlation of artifact to precursor"},
        }
    if tolerances is not None:
        panels["RT"]["tol"] = tolerances["rt_tol"]
        panels[ion_mobility_type]["tol"] = tolerances["im_tol"]
        panels["drmz"]["tol"] = tolerances["drmz_tol"]
        panels["relative area"]["tol"] = tolerances["max_rel_intensity"]
        panels["precursor area"]["tol"] = tolerances["min_intensity"]
        panels["correlation"]["tol"] = tolerances["min_correlation"]
    
    panel_order = ["RT", ion_mobility_type, "drmz", "precursor area", "relative area", "correlation"]

    fig = make_subplots(
        rows=3, cols=2,
        subplot_titles=[panels[p]["title"] for p in panel_order],
        vertical_spacing=0.12,
        horizontal_spacing=0.08,
    )

    for i, p in enumerate(panel_order):
        r = (i // 2) + 1
        c = (i % 2) + 1
        
        # define bars
        if p in ["RT", "DT", "drmz"]:
            artifact_df = broad_df[~broad_df["artifact"].isin(["no", "precursor"])]
            all_data = artifact_df[panels[p]["value"]].dropna().round(5)
            min_step = min(all_data[all_data > 0].min(), -all_data[all_data < 0].max())
            if np.isnan(min_step): min_step = 0.0001
            n_bins = int(all_data.abs().max() / min_step) * 2 + 1
            bins = np.linspace(-n_bins / 2, n_bins / 2, n_bins + 1) * min_step
            x_labels = np.linspace(-(n_bins - 1) / 2, (n_bins - 1) / 2, n_bins) * min_step
            while len(x_labels) > 11:
                x_labels = np.linspace(-(n_bins - 1) / 2, (n_bins - 1) / 2, len(x_labels) - 2) * min_step
            if i == 0:
                x_labels = x_labels.round(4)
        else:
            n_bins = 50
            if p in ["precursor area"]:
                all_data = broad_df.loc[artifact_df["precursor"], panels[p]["value"]]
                bins = np.logspace(np.log10(all_data.min()*0.99999999), np.log10(all_data.max()*1.00000001), num=n_bins + 1)
            else:
                all_data = artifact_df[panels[p]["value"]].dropna().round(5)
                _, bins = np.histogram(all_data, bins=n_bins)

        # segment data into bars
        for j, artifact in enumerate(artifacts):
            if p in ["precursor area"]:
                data = broad_df.loc[artifact_df.loc[artifact_df["artifact"] == artifact, "precursor"], panels[p]["value"]]
            else:
                data = artifact_df.loc[artifact_df["artifact"] == artifact, panels[p]["value"]].dropna().round(5)
            counts, _ = np.histogram(data, bins=bins)

            fig.add_trace(
                go.Bar(
                    x=bins[:-1] + np.diff(bins) / 2,
                    y=counts / counts.max(),
                    base=j * 1.2,
                    width=np.diff(bins),
                    name=artifact.replace(" ring", ""),
                    marker_color=artifact_colors[j],
                    legendgroup=artifact.replace(" ring", ""), 
                    showlegend=(i == 0),
                ),
                row=r, col=c
            )
        
        if tolerances_set:
            # put up some shade
            shade_color = "LightSeaGreen"
            opacity=0.3
            if p in ["RT", ion_mobility_type]:
                fig.add_vrect(
                    x0=-panels[p]["tol"], x1=panels[p]["tol"],
                    fillcolor=shade_color,
                    opacity=opacity,
                    layer="below",
                    line_width=0,
                    row=r, col=c,
                )
            elif p in ["drmz"]:
                height = 1.2 * len(artifacts)
                if "M+1" in artifacts:
                    fig.add_shape(
                        type="rect",
                        x0=-panels[p]["tol"] + tolerances["iso_offset"], x1=panels[p]["tol"] + tolerances["iso_offset"],
                        y0=height - 1.2, y1=height,
                        fillcolor=shade_color,
                        opacity=opacity,
                        layer="below",
                        line_width=0,
                        row=r, col=c
                    )
                    height -= 1.2
                fig.add_shape(
                        type="rect",
                        x0=-panels[p]["tol"], x1=panels[p]["tol"],
                        y0=0, y1=height,
                        fillcolor=shade_color,
                        opacity=opacity,
                        layer="below",
                        line_width=0,
                        row=r, col=c
                    )
            elif p in ["relative area"]:
                fig.add_vrect(
                    x0=0, x1=panels[p]["tol"],
                    fillcolor=shade_color,
                    opacity=opacity,
                    layer="below",
                    line_width=0,
                    row=r, col=c,
                )
            elif p in ["precursor area", "correlation"]:
                fig.add_vrect(
                    x0=panels[p]["tol"], x1=1,
                    fillcolor=shade_color,
                    opacity=opacity,
                    layer="below",
                    line_width=0,
                    row=r, col=c,
                )
        
        # make pretty axes
        fig.update_xaxes(
            showline=True,
            mirror=True,
            linecolor='white',
            linewidth=0.1,
            title_text=panels[p]["xlabel"], 
            showgrid=True,
            row=r, col=c
        )

        if p == "drmz":
            fig.update_xaxes(
                tickformat=".0e",
                row=r, col=c
            )
        elif p == "precursor area":
            if "tol" in panels[p]:
                tol = panels[p]["tol"]
            else:
                tol = 1e+0
            fig.update_xaxes(
                type="log",
                tickformat=".0e",
                range=[np.log10(min(tol, all_data.min()) * 0.8), np.log10(1e+0 * 1.5)],
                showgrid=True,
                dtick=1,
                row=r, col=c)
        elif p == "relative area":
            fig.update_xaxes(
                tickformat=".0%",
                row=r, col=c
            )
        elif p == "correlation":
            fig.update_xaxes(
                range=[-0.05, 1.05],
                row=r, col=c
            )
        
        if c == 1:
            fig.update_yaxes(
                title_text="relative counts", 
                row=r, col=c
            )
        
        counts, _ = np.histogram(all_data, bins=bins)

    fig.update_yaxes(
        showline=True,
        mirror=True,
        linecolor='white',
        linewidth=0.1,
        showticklabels=False,
        showgrid=False,
        zeroline=False,
    )

    fig.update_annotations(yshift=5)
    fig.update_xaxes(title_standoff=5)

    fig.update_layout(
        template="plotly_dark",
        barmode="overlay",
        margin=dict(t=95, b=0, l=0, r=9),
        height=800,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            tracegroupgap=1,
            y=1.04,
            xanchor="left",
            x=0,
        ),
    )
    return fig
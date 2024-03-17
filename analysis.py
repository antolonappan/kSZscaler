import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import *
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from xgboost import XGBRegressor

plt.rcParams["text.usetex"] = True
plt.rcParams["ytick.minor.visible"] = True
plt.rcParams["xtick.minor.visible"] = True


def data_manupulation(df: pd.DataFrame) -> pd.DataFrame:
    dff = df.copy()
    dff = dff[["Yksz", "Vz", "Mstar", "M", "Vlos", "Ytsz"]]
    dff.Vz = dff.Vz.apply(lambda x: x / 1e5)
    dff.Mstar = dff.Mstar.apply(lambda x: np.log(x))
    q_list = [0, 0.5, 0.75, 1]
    labels = [1, 2, 3]
    dff["M_q"] = pd.qcut(dff["M"], q=q_list, labels=labels)
    return dff


def HaloMass_quartile_fit(df: pd.DataFrame, save: Optional[str] = None) -> None:
    df = data_manupulation(df)
    slope1, inter1 = np.polyfit(df.M[df.M_q == 1], df.Yksz[df.M_q == 1], 1)
    slope2, inter2 = np.polyfit(df.M[df.M_q == 2], df.Yksz[df.M_q == 2], 1)
    slope3, inter3 = np.polyfit(df.M[df.M_q == 3], df.Yksz[df.M_q == 3], 1)

    num_ticks = 6
    tick_positions = np.linspace(df.M.min(), df.M.max(), num_ticks)
    tick_labels = [f"${i}$" for i in np.round(np.log10(np.exp(tick_positions)), 1)]

    plt.figure(figsize=(6, 6))
    plt.scatter(
        df.M[df.M_q == 1],
        df.Yksz[df.M_q == 1],
        c="r",
        s=2,
        alpha=0.2,
        label="quartile $[0.0-0.50]$",
    )
    plt.scatter(
        df.M[df.M_q == 2],
        df.Yksz[df.M_q == 2],
        c="g",
        s=2,
        alpha=0.2,
        label="quartile $[0.5-0.75]$",
    )
    plt.scatter(
        df.M[df.M_q == 3],
        df.Yksz[df.M_q == 3],
        c="b",
        s=2,
        alpha=0.2,
        label="quartile $[0.75-1.0]$",
    )
    plt.plot(
        df.M,
        df.M * slope1 + inter1,
        color="r",
        label="$\\alpha_{kSZ}$" + f" = {slope1:.1f}",
    )
    plt.plot(
        df.M,
        df.M * slope2 + inter2,
        color="g",
        label="$\\alpha_{kSZ}$" + f" = {slope2:.1f}",
    )
    plt.plot(
        df.M,
        df.M * slope3 + inter3,
        color="b",
        label="$\\alpha_{kSZ}$" + f" = {slope3:.1f}",
    )
    plt.legend(fontsize=15)
    plt.xlabel("$\mathrm{log10}\; M_{500c} [M_\\odot]$", fontsize=15)
    plt.ylabel("$\ln Y_{kSZ} [Mpc^2]$", fontsize=15)
    plt.xticks(ticks=tick_positions, labels=tick_labels)

    plt.ylim(-25, None)
    if save:
        plt.savefig(save, bbox_inches="tight", dpi=300)


def V_losVSrec(
    df1: pd.DataFrame,
    df2: pd.DataFrame,
    df3: pd.DataFrame,
    z_arr: List,
    v_comp: str = "z",
    save: Optional[str] = None,
    ylim: Optional[Tuple] = None,
    xlim: Optional[Tuple] = None,
) -> None:
    fig, axs = plt.subplots(1, 3, figsize=(17, 4))
    plt.subplots_adjust(
        wspace=0.3, right=0.8
    )  # Adjust right margin to leave space for colorbar

    assert v_comp in ["z", "net"]
    
    for i, df in enumerate([df1, df2, df3]):
        if v_comp == "z":
            Vnet = df.Vz
            vnet = df.vz
        elif v_comp == "net":
            Vnet = df.Vnet
            vnet = df.vnet
        else:
            raise ValueError('v_comp must be either "z" or "net"')

        corr = np.corrcoef(Vnet / 1e5, vnet)[0, 1]
        print(f"Correlation at z={z_arr[i]:.2f} is {corr:.2f}")

        hb = axs[i].hexbin(
            Vnet / 1e5, vnet, gridsize=30, cmap="Spectral", bins="log", mincnt=1
        )
        if i == 1:
            axs[i].set_xlabel("$V_{rec}[km/s]$", fontsize=15)
        if i == 0:
            axs[i].set_ylabel("$V_{halo}[km/s]$", fontsize=15)
        if xlim:
            axs[i].set_xlim(xlim)
        axs[i].axline((0, 0), (1, 1), color="k", linestyle="--")
        if ylim:
            axs[i].set_ylim(ylim)
        axs[i].set_title(f"$z={z_arr[i]:.2f}$", fontsize=15)
        axs[i].tick_params(labelsize=15)

    # Create a colorbar outside the subplots
    cbar_ax = fig.add_axes([0.82, 0.15, 0.01, 0.7])  # Adjust these values as needed
    cbar = fig.colorbar(hb, cax=cbar_ax, label="$\\rm log_{10}(N)$")
    cbar.ax.tick_params(labelsize=15)
    cbar.set_label(label="$\\rm log_{10}(N)$", size=15)
    if save:
        plt.savefig(save, bbox_inches="tight", dpi=300)


def Yksz_Ytsz(df: pd.DataFrame, save: Optional[str] = None) -> None:
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(6, 13), sharex=True)
    plt.subplots_adjust(hspace=0.1)
    im = ax1.hexbin(
        np.log10(np.exp(df.M)), df.Yksz, gridsize=40, cmap="Blues", bins="log", mincnt=1
    )
    ax1.plot(
        np.log10(np.exp(df.M)),
        df.M * 1.20 - 53.5,
        lw=1.5,
        c="k",
        ls="--",
        label="$\\rm Best\;fit$",
    )
    fig.colorbar(im, ax=ax1, label="log10(N)")
    ax1.set_ylabel("$\ln Y_\mathrm{kSZ} [Mpc^2]$", fontsize=15)
    ax1.set_ylim(-22)
    ax1.legend(loc="lower right", fontsize=15)
    # ax2.scatter(np.log10(np.exp(df.M)),df.Ytsz,s=1,c='C9')
    im = ax2.hexbin(
        np.log10(np.exp(df.M)),
        df.Ytsz,
        gridsize=40,
        cmap="Oranges",
        bins="log",
        mincnt=1,
    )
    ax2.plot(
        np.log10(np.exp(df.M)),
        df.M * (1.88) - 73.67,
        lw=1.5,
        c="k",
        ls="--",
        label="$\\rm Best\;fit$",
    )
    ax2.set_ylabel("$\ln Y_\mathrm{tSZ} [Mpc^2]$", fontsize=15)
    fig.colorbar(im, ax=ax2, label="log10(N)")
    # ax3.scatter(np.log10(np.exp(df.M)),df.Yksz/df.Ytsz,s=1,c='C6')
    im = ax3.hexbin(
        np.log10(np.exp(df.M)),
        df.Yksz / df.Ytsz,
        gridsize=40,
        cmap="Greens",
        bins="log",
        mincnt=1,
    )
    fig.colorbar(im, ax=ax3, label="log10(N)")
    ax3.set_ylabel("$Y_\mathrm{tSZ}/Y_\mathrm{kSZ}$", fontsize=15)
    ax3.set_xlabel("$\mathrm{log10}\; M_{500c} [M_\\odot]$", fontsize=15)
    if save:
        fig.savefig(save, dpi=300, bbox_inches="tight")


def fit_regression_models(
    df: pd.DataFrame,
    features: List[str],
    target: str,
    df_test: Optional[pd.DataFrame] = None,
    verbose: Optional[bool] = False,
) -> Dict:
    if verbose:
        print(f"Features: {features}")
    X = df[features].values.reshape(-1, len(features))
    y = df[target].values

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=df.M_q, random_state=42
    )

    # Initialize regression models
    models = {
        "Random Forest Regression": RandomForestRegressor(
            max_depth=10,
            min_samples_leaf=4,
            min_samples_split=10,
            n_estimators=200,
            random_state=42,
        ),
        "XGB": XGBRegressor(n_estimators=100, learning_rate=0.1, random_state=42),
        "GradientBoost": GradientBoostingRegressor(random_state=42),
    }

    results = {}

    for model_name, model in models.items():
        if verbose:
            print(f"Running {model_name}...")
        # Fit the model
        # if model_name == "Perceptron":
        #     model.fit(X_train, y_train,)
        # else:
        model.fit(X_train, y_train, sample_weight=np.sqrt(X_train[:, 0]))

        # Make predictions on the test set
        y_pred = model.predict(X_test)
        y_pred_train = model.predict(X_train)

        # Evaluate the model
        mse_test = mean_squared_error(y_test, y_pred, squared=False)
        mse_train = mean_squared_error(y_train, y_pred_train, squared=False)
        # r2 = r2_score(y_test, y_pred)

        results[model_name] = {"MSE_test": mse_test, "MSE_train": mse_train}

        if df_test is not None:
            assert type(df_test) == list
            for i, df_ in enumerate(df_test):
                X_ = df_[features].values.reshape(-1, len(features))
                y_ = df_[target].values
                _, X_test_, _, y_test_ = train_test_split(
                    X_, y_, test_size=0.2, stratify=df_.M_q, random_state=42
                )
                y_pred_ = model.predict(X_test_)
                rmse = mean_squared_error(y_test_, y_pred_, squared=False)
                r2 = r2_score(y_test_, y_pred_)
                results[model_name][f"MSE_{i}"] = rmse
                # results[model_name][f"R-squared_{i}"] = r2

    return results


def get_results(
    df: pd.DataFrame, verbose: Optional[bool] = True
) -> Tuple[Dict, Dict, Dict, Dict]:
    results_M = fit_regression_models(df, ["M"], "Yksz", verbose=verbose)
    results_MM = fit_regression_models(df, ["M", "Mstar"], "Yksz", verbose=verbose)
    results_MVz = fit_regression_models(df, ["M", "Vz"], "Yksz", verbose=verbose)
    results_MMVz = fit_regression_models(
        df, ["M", "Mstar", "Vz"], "Yksz", verbose=verbose
    )
    return results_M, results_MM, results_MVz, results_MMVz


def dicresults(
    results_M: Dict, results_MM: Dict, results_MVz: Dict, results_MMVz: Dict
) -> Dict:
    res = {}
    for feature_set, result in zip(
        ["M", "[M,Mstar]", "[M, Vz]", "[M, Mstar, Vz]"],
        [results_M, results_MM, results_MVz, results_MMVz],
    ):
        res[feature_set] = result
    return res


def MSE_df_feature(
    df1: pd.DataFrame,
    df2: pd.DataFrame,
    df3: pd.DataFrame,
    z_arr: List[str],
    verbose: Optional[bool] = False,
    save: Optional[str] = None,
) -> None:
    df1 = data_manupulation(df1)
    df2 = data_manupulation(df2)
    df3 = data_manupulation(df3)
    mse_data = {}
    for df, z in zip([df1, df2, df3], z_arr):
        res = dicresults(*get_results(df, verbose=verbose))
        mse_data[z] = res

    positions = ["M", "[M,Mstar]", "[M, Vz]", "[M, Mstar, Vz]"]
    plabels = [
        "$M_{halo}$",
        "$[M_{halo},M_*]$",
        "$[M_{halo}, V_{rec}]$",
        "$[M_{halo}, M_*, V_{rec}]$",
    ]
    mname = ["$\\rm RF$", "$\\rm XGB$", "$\\rm GB$"]

    # Set up the figure
    fig, axs = plt.subplots(nrows=1, ncols=len(positions), figsize=(10, 3), sharey=True)

    # Define colors and markers for each model
    model_colors = {
        "Random Forest Regression": "red",
        "XGB": "green",
        "GradientBoost": "blue",
    }
    model_markers = {"Random Forest Regression": "o", "XGB": "^", "GradientBoost": "s"}

    # Define line styles for train and test
    line_styles = {"train": "-", "test": "--"}

    # Plotting the data
    for feature_index, feature_name in enumerate(positions):
        ax = axs[feature_index]
        for model_name in ["Random Forest Regression", "XGB", "GradientBoost"]:
            # Extract the MSE values for training and testing across all redshifts for the current model and feature set
            redshifts = []
            mse_train_values = []
            mse_test_values = []
            for redshift, features_data in mse_data.items():
                mse_train_values.append(
                    features_data[feature_name][model_name]["MSE_train"]
                )
                mse_test_values.append(
                    features_data[feature_name][model_name]["MSE_test"]
                )
                redshifts.append(float(redshift))

            # Sort the arrays by redshift
            sorted_indices = np.argsort(redshifts)
            redshifts = np.array(redshifts)[sorted_indices]
            mse_train_values = np.array(mse_train_values)[sorted_indices]
            mse_test_values = np.array(mse_test_values)[sorted_indices]

            # Plot training and testing MSE values
            ax.plot(
                redshifts,
                mse_train_values,
                marker=model_markers[model_name],
                color=model_colors[model_name],
                linestyle=line_styles["train"],
                linewidth=2,
            )
            ax.plot(
                redshifts,
                mse_test_values,
                marker=model_markers[model_name],
                color=model_colors[model_name],
                linestyle=line_styles["test"],
                linewidth=2,
            )

        # Set the title for each subplot
        ax.set_title(f"{plabels[feature_index]}", fontsize=15)

        plt.xticks(fontsize=15)

        # Remove the left spine (y-axis line) for subplots from the second to the last
        if feature_index > 0:
            # ax.spines['left'].set_visible(False)
            if feature_index < 3:
                pass
                # ax.spines['right'].set_visible(False)
            ax.tick_params(left=False)

        if feature_index == 0:
            ax.set_ylabel("$\\rm MSE$", fontsize=15)
            # ax.spines['right'].set_visible(False)
        ax.tick_params(labelsize=15)
        # Remove the x-axis label for all subplots
        ax.set_xlabel("")

    # Set a common x-axis label in the center of the figure
    fig.text(0.5, -0.01, "$\\rm Redshift$", ha="center", fontsize=15)

    # Create two legends
    legend_elements_1 = [
        plt.Line2D(
            [0],
            [0],
            color="black",
            linestyle=line_styles["train"],
            linewidth=2,
            label="$\\rm Training$",
        ),
        plt.Line2D(
            [0],
            [0],
            color="black",
            linestyle=line_styles["test"],
            linewidth=2,
            label="$\\rm Testing$",
        ),
    ]

    legend_elements_2 = [
        plt.Line2D(
            [0],
            [0],
            marker=model_markers[name],
            color=model_colors[name],
            linestyle="None",
            markersize=10,
            label=mname[i],
        )
        for i, name in enumerate(model_markers)
    ]

    # Place the first legend in the left-most subplot
    axs[0].legend(
        handles=legend_elements_1, loc="lower left", title="Data Type", fontsize=15
    )

    # Place the second legend in the right-most subplot
    axs[-1].legend(
        handles=legend_elements_2, loc="upper right", title="Models", fontsize=15
    )

    # Adjust layout
    plt.tight_layout()
    plt.subplots_adjust(wspace=0, hspace=0)
    # Show the plot
    if save:
        plt.savefig(save, dpi=300, bbox_inches="tight")


def MSE_cross_df(
    df_dict: Dict,
    z_dict: Dict,
    snaps: List[str],
    verbose: Optional[bool] = True,
    save: Optional[str] = None,
) -> None:

    assert len(snaps) == 5, "snaps must be a list of 5 snapshots"
    for i in snaps:
        assert i in df_dict.keys(), f"{i} not in Dfs"
    print("This function assumes first snapshot as the training dataset")

    df_train = data_manupulation(df_dict[snaps[0]])
    df_test = [data_manupulation(df_dict[snaps[i]]) for i in range(1, 5)]

    mse_data = {
        "[M]": fit_regression_models(df_train, ["M"], "Yksz", df_test, verbose=verbose),
        "[M,Mstar]": fit_regression_models(
            df_train, ["M", "Mstar"], "Yksz", df_test, verbose=verbose
        ),
        "[M,Vr]": fit_regression_models(
            df_train, ["M", "Vz"], "Yksz", df_test, verbose=verbose
        ),
        "[M,Mstar,Vr]": fit_regression_models(
            df_train, ["M", "Mstar", "Vz"], "Yksz", df_test, verbose=verbose
        ),
    }

    # Define the feature sets for the x-axis
    features = ["[M]", "[M,Mstar]", "[M,Vr]", "[M,Mstar,Vr]"]
    flabel = [
        "$[M_{halo}]$",
        "$[M_{halo},M_*]$",
        "$[M_{halo},V_{rec}]$",
        "$[M_{halo},M_*,V_{rec}]$",
    ]

    # Define correct colors for the redshifts
    redshift_colors_corrected = ["r", "b"]  # Corrected colors for redshifts

    # Create a figure with subplots
    fig, axes = plt.subplots(3, 1, figsize=(7, 10), sharex=True)

    # Adding space between subplots
    plt.subplots_adjust(hspace=0.25)

    # Set the titles for each model
    titles = ["Random Forest Regression", "XGB", "GradientBoost"]
    tlables = ["$\\rm RF$", "$\\rm XGB$", "$\\rm GB$"]

    # Iterate over each axis to plot the data
    pp = 0
    for ax, title in zip(axes, titles):
        ax.set_title(tlables[pp], fontsize=15)
        pp += 1
        # ax.grid(True,)  # Enable grid

        # Iterate over each feature set for the current model
        for i, (feature_set, results) in enumerate(mse_data.items()):
            # Extract the MSE values for each type
            mse_train = results[title]["MSE_train"]
            mse_test = results[title]["MSE_test"]
            mse_0 = results[title]["MSE_0"]
            mse_1 = results[title]["MSE_1"]
            mse_2 = results[title]["MSE_2"]
            mse_3 = results[title]["MSE_3"]

            # Define x position for each MSE type
            x_positions = [i - 0.15, i - 0.1, i, i + 0.1, i + 0.15]

            # Plot the train and test MSE
            ax.scatter(
                x_positions[2], mse_train, color="black", marker="s", s=100
            )  # facecolors='none'
            ax.scatter(x_positions[0], mse_test, color="C6", marker="^", s=100)
            # Plot the MSE_0 and MSE_1 with corrected redshift colors
            ax.scatter(x_positions[1], mse_0, color="C9", marker="X", s=100)
            ax.scatter(x_positions[2], mse_1, color="C4", marker="D", s=100)
            ax.scatter(x_positions[3], mse_2, color="C1", marker="P", s=100)
            ax.scatter(x_positions[4], mse_3, color="C7", marker="p", s=100)
            plt.xticks(fontsize=15)

        # Setting the x-axis labels only on the last subplot
        ax.set_ylim(0.7, 1.3)
        plt.xticks(fontsize=15)

        if ax is axes[-1]:
            ax.set_xticks(range(len(features)))
            ax.set_xticklabels(flabel, fontsize=15)
        else:

            ax.set_xticks([])  # Hide x-tick labels for the upper subplots

    # Set common y-label
    fig.text(-0.01, 0.5, "$ \\rm MSE$", va="center", rotation="vertical", fontsize=15)

    # Creating two legends with correct labels and colors
    labels_1 = [f"$z = {z_dict[snaps[0]]:.2f}$"]
    labels_2 = [f"$z = {z_dict[snaps[i]]:.2f}$" for i in range(1, 5)]

    # Legend elements for Train and Test
    legend_elements_1 = [
        plt.Line2D(
            [0], [0], color="black", marker="s", linestyle="None", markersize=10,
        ),
    ]  # markerfacecolor='none'
    # plt.Line2D([0], [0], color='black', marker='s', linestyle='None', markersize=10)]

    # Legend elements for Redshifts
    legend_elements_2 = [
        plt.Line2D([0], [0], color="C6", marker="^", linestyle="None", markersize=10),
        plt.Line2D([0], [0], color="C9", marker="X", linestyle="None", markersize=10),
        plt.Line2D([0], [0], color="C4", marker="D", linestyle="None", markersize=10),
        plt.Line2D([0], [0], color="C1", marker="P", linestyle="None", markersize=10),
        plt.Line2D([0], [0], color="C7", marker="p", linestyle="None", markersize=10),
    ]

    # Place the first legend on the first subplot and the second legend on the last subplot
    legend1 = axes[0].legend(
        legend_elements_1, labels_1, loc="upper right", title="Train", fontsize=15
    )
    legend2 = axes[0].legend(
        legend_elements_2, labels_2, ncol=2, loc="lower left", title="Test", fontsize=15
    )

    axes[0].add_artist(legend1)
    axes[0].add_artist(legend2)

    # axes[2].legend(legend_elements_3, labels_3, loc='lower left', title='Test',fontsize=15)

    # Adjust the title of the legends
    axes[0].get_legend().get_title().set_fontsize("large")
    # axes[1].get_legend().get_title().set_fontsize('large')
    # axes[2].get_legend().get_title().set_fontsize('large')

    plt.tight_layout()
    if save:
        plt.savefig(save, dpi=300, bbox_inches="tight")

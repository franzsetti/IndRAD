import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import numpy as np
import statsmodels.api as sm
import seaborn as sns
import pandas as pd

# plt.style.use('seaborn')

def plot_trajectory(fig_title, trajectory, trajectory_anomalies = None, colors={}, anomalies=None):
    fig, axs = plt.subplots(7, 3, figsize=(10,10))

    fig2 = None
    # axs2 = None
    # if trajectory_anomalies is not None:
    #     fig2, axs2 = plt.subplots(7, 3, figsize=(10,10))

    fig.suptitle(fig_title)
    col = 0
    feature_index = 0
    for feature in ['position', 'velocity', 'effort']:
        for i in range(7):
            if isinstance(trajectory, dict):
                for key, values in trajectory.items():
                    if isinstance(colors, dict) and key in colors.keys():
                        values.iloc[:,feature_index].plot(ax=axs[i, col], color=colors[key], label=key)
                    else:
                        values.iloc[:,feature_index].plot(ax=axs[i, col], label=key)
            else:
                nominal = pd.Series(trajectory.iloc[:,feature_index])
                # trajectory.iloc[:,feature_index].plot(ax=axs[i, col], color='black', legend=False)

                nominal.plot(ax=axs[i, col], color=['#cfcfcf'], legend=False)

                if trajectory_anomalies is not None and len(trajectory_anomalies) > 0:
                    anomaly = pd.Series(trajectory_anomalies.iloc[:,feature_index])
                    if feature=='effort' and i==1:
                        anomaly.plot(ax=axs[i, col], color=['red'], legend=False, style='o', markersize=1)

                    # key = f"{feature}_{i}"
                    # if key in anomalies.keys():
                    #     anomaly.plot(ax=axs2[i, col], color=['red'], legend=False, style='.')

                    # trajectory_anomalies.iloc[:,feature_index].plot(ax=axs[i, col], color='red', legend=False)
                    # trajectory_anomalies.iloc[:,feature_index].plot(ax=axs2[i, col], color='red', legend=False)

                    # key = f"{feature}_{i}"
                    # if key in anomalies.keys():
                    #     indexes = anomalies[key]
                    #     print(key)
                    #     print(indexes)
                    #     print(anomaly[indexes])

            axs[i, col].set_title(f'{feature} {i}')
            # if trajectory_anomalies is not None:
            #     axs2[i, col].set_title(f'{feature} {i}')

            feature_index += 1
        col += 1
    plt.show()


def plot_anomaly_score(t_name, anomaly_score, anomalies, threshold):
    # Plot anomaly score
    fig, ax = plt.subplots()
    plt.title(f"Anomaly score: {t_name}")
    ax.axhline(y=threshold, color='orange', linestyle='--')
    anomaly_score.plot(ax=ax, color=['#cfcfcf'], legend=False)
    if anomalies is not None:
        anomaly_score[anomalies].plot(ax=ax, color='r', legend=False, style='o', markersize=2)
    plt.show()

def plot_stats(trajectory, plot_type):
    fig, axs = plt.subplots(7, 3, figsize=(10,10))
    col = 0
    feature_index = 0

    for feature in ['position', 'velocity', 'effort']:
        for i in range(7):
            if plot_type == 'quantile':
                sm.qqplot(trajectory[:,feature_index], ax=axs[i, col], line ='45')
            elif plot_type == 'distribution':
                sns.kdeplot(data=trajectory[:,feature_index], ax=axs[i, col], fill=True, alpha=0.5)
            feature_index += 1
        col += 1
    plt.show()


def fit_pca(dataset, n_components=20, show_plot_variance=False):
    """
    """
    print(f"Apply PCA on trajectory {dataset.name} ({n_components} components)")
    pca = PCA(n_components=n_components)
    print("\t* Fit PCA")
    pca.fit(dataset.dataset_processed)

    if show_plot_variance:
        plt.figure()
        plt.plot(np.arange(len(pca.explained_variance_ratio_)), pca.explained_variance_ratio_, color='black')
        plt.show()

    return pca


def is_pos_def(A):
    if np.allclose(A, A.T):
        try:
            np.linalg.cholesky(A)
            return True
        except np.linalg.LinAlgError:
            return False
    else:
        return False


def cov_matrix(data):
    covariance_matrix = np.cov(data, rowvar=False)
    if is_pos_def(covariance_matrix):
        inv_covariance_matrix = np.linalg.inv(covariance_matrix)

        if is_pos_def(inv_covariance_matrix):
            return covariance_matrix, inv_covariance_matrix
        else:
            print("Error: Inverse of Covariance Matrix is not positive definite!")
    else:
        print("Error: Covariance Matrix is not positive definite!")


def mahalanobis_dist(inv_cov_matrix, mean_distr, data):
    inv_covariance_matrix = inv_cov_matrix
    vars_mean = mean_distr
    diff = data - vars_mean

    left = np.dot(diff, inv_covariance_matrix)
    mahal = np.dot(left, diff.T)
    return mahal.diagonal()


def md_threshold(dist, extreme=False):
    k = 3. if extreme else 2.
    threshold = np.mean(dist) * k
    return threshold
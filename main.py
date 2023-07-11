from Dataset import Dataset
from utils import *
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix
from sklearn.mixture import GaussianMixture
import os
from sklearn.svm import OneClassSVM
import pickle

WINDOW_SIZE = 2000
MOVING_AVG_STEP = 50
PCA_COMPONENTS = 21
GAUSSIAN_MIXTURE_COMPONENTS = 10
ANOMALY_METHOD = 'mahalanobis'  # quantile | mahalanobis | score | oneclass

scaler = StandardScaler()

print("Load train dataset nominal")
train_dataset_nominal = Dataset(name='train_dataset_nominal',
                                window_size=WINDOW_SIZE,
                                scaler=scaler)
train_dataset_nominal.load(path=Dataset.TRAIN_DATASET_PATH)
train_dataset_nominal.apply_moving_average(step=MOVING_AVG_STEP)
# plot_trajectory("Train trajectory", train_dataset_nominal.dataset)

train_dataset_nominal.add_sliding_window()
# Fit scaler with train dataset of nominal data
scaler.fit(train_dataset_nominal.dataset_processed)
print("Normalize train dataset nominal")
train_dataset_nominal.normalize()

# Fit PCA model
pca = fit_pca(dataset=train_dataset_nominal, n_components=PCA_COMPONENTS, show_plot_variance=False)
# Transform train dataset of nominal data with pca model
train_dataset_nominal.pca(model=pca)

gm = GaussianMixture(n_components=GAUSSIAN_MIXTURE_COMPONENTS, random_state=0).fit(train_dataset_nominal.dataset_processed)
covariance_matrix, inverse_covariance_matrix = cov_matrix(train_dataset_nominal.dataset_processed)

# Inverse PCA on train dataset of nominal data
train_pca_errors = train_dataset_nominal.pca_inverse(model=pca)

print("Fit One Class SVM")
# Save to file in the current working directory
filename = os.path.join(os.path.dirname(os.path.realpath(__file__)), "oneclasssvm.pkl")

if True:
    # Load from file
    with open(filename, 'rb') as file:
        one_class_model = pickle.load(file)
if False:
    one_class_model = OneClassSVM(gamma=0.001, nu=0.03, kernel='rbf').fit(train_pca_errors)

    with open(filename, 'wb') as file:
        pickle.dump(one_class_model, file)

# CALCOLO THRESHOLD CON QUANTILE
# print("Training errors distribution & quantile")
# plot_stats(train_pca_errors, 'distribution')
# plt.show()
# plot_stats(train_pca_errors, 'quantile')
# plt.show()

if ANOMALY_METHOD in ['quantile']:
    q1, q2 = np.quantile(train_pca_errors, [0.05, 0.95], axis=0)
    pc_iqr = q2 - q1
    pc_lower_bound = q1 - (5 * pc_iqr)
    pc_upper_bound = q2 + (5 * pc_iqr)

    print(f"* Lower Bound: {pc_lower_bound}")
    print(f"* Upper Bound: {pc_upper_bound}")

test_dataset_nominal = {}
test_dataset_nominal_pca = {}
test_dataset_nominal_pca_errors = []

# Load test dataset of nominal trajectories
print("Load Test Dataset of nominal trajectories")
for t_name in ['trajectory_6', 'trajectory_8', 'trajectory_13bis']:
    print(f"\t* Loading Trajectory: {t_name}")
    dataset = Dataset(name=t_name,
                      window_size=WINDOW_SIZE,
                      scaler=scaler)
    dataset.load(Dataset.TEST_NOMINAL_DATASET_PATH)
    dataset.apply_moving_average(step=MOVING_AVG_STEP)
    # plot_trajectory(f"Test nominal trajectory: {t_name}", dataset.dataset, None)

    # add sliding window
    dataset.add_sliding_window()
    # normalize dataset
    dataset.normalize()
    # Plot scaled dataset
    """
    plot_trajectory("Dataset preprocessing", {'original': dataset.dataset,
                                              'normalized': Dataset.dataset_remove_sliding_window(
                                                  dataset=dataset.dataset_processed, window_size=WINDOW_SIZE)}, None,
                    {'original': 'black', 'normalized': 'green'})
    """
    # pca transform
    dataset.pca(model=pca)
    test_dataset_nominal_pca[t_name] = dataset.dataset_processed.copy()

    # pca inverse transform
    test_dataset_nominal_pca_errors.append(dataset.pca_inverse(model=pca))

    # test_dataset_nominal_pca_errors.append(pca_error.flatten().reshape(pca_error.shape[0]*WINDOW_SIZE, 21))
    # scaler inverse transform
    dataset.normalize_inverse()
    # remove sliding window
    dataset.remove_sliding_window()

    """
    plot_trajectory("Reconstruction of test nominal trajectory", {'original': dataset.dataset, 'reconstruct':
        dataset.dataset_processed}, None, {'original': 'black', 'reconstruct': 'orange'})

    plt.show()
    """
    test_dataset_nominal[t_name] = dataset

pca_errors = np.vstack(test_dataset_nominal_pca_errors)
# threshold = np.max(np.abs(pca_errors), axis=0)

if ANOMALY_METHOD == 'mahalanobis':
    print("* Calculating mahalanobis distance threshold")
    test_dataset_nominal_pca_full = pd.concat(test_dataset_nominal_pca.values())
    mahalanobis_distance_nominal = []
    # for t in test_dataset_nominal_pca.values():
    for i in range(GAUSSIAN_MIXTURE_COMPONENTS):
        mahalanobis_distance_nominal.append(
            mahalanobis_dist(inverse_covariance_matrix, gm.means_[i], test_dataset_nominal_pca_full))
    mahalanobis_distance_nominal = np.median(np.array(mahalanobis_distance_nominal).T)
    print(f"Mahalanobis distance nominal: {mahalanobis_distance_nominal}")
if ANOMALY_METHOD in ['score', 'mahalanobis']:
    threshold = np.median(np.sum(np.abs(pca_errors), axis=1), axis=0) * 5
else:
    threshold = np.median(np.abs(pca_errors), axis=0)
print(f"* Threshold: {threshold}")

# Load test dataset of nominal trajectories and corrupt
print("Load Test dataset of nominal trajectories and corrupt")
for t_name, t_name_nominal in {'trajectory_13bis': 'trajectory_13bis', 'trajectory_8': 'trajectory_8', 'trajectory_6': 'trajectory_6'}.items():
    # plot_trajectory(f"Test nominal trajectory: {t_name_nominal}", test_dataset_nominal[t_name_nominal].dataset, None)

    print(f"\t* Loading Trajectory: {t_name}")
    dataset = Dataset(name=t_name,
                      dataset=test_dataset_nominal[t_name_nominal].dataset,
                      window_size=WINDOW_SIZE,
                      scaler=scaler)
    # dataset.corrupt(corruption_type='step')
    # dataset.corrupt(corruption_type='freeze_zero')
    # dataset.corrupt(corruption_type='freeze_last_value')
    dataset.corrupt(corruption_type='spike')

    # plot_trajectory(f"Test trajectory with anomalies: {t_name}", dataset.dataset, None)

    dataset.add_sliding_window()
    dataset.normalize()
    dataset.pca(model=pca)

    if ANOMALY_METHOD == 'mahalanobis':
        mahalanobis_distance = []
        for i in range(GAUSSIAN_MIXTURE_COMPONENTS):
            mahalanobis_distance.append(mahalanobis_dist(inverse_covariance_matrix, gm.means_[i], dataset.dataset_processed))
        mahalanobis_distance = np.min(np.array(mahalanobis_distance).T, axis=1)

    errors = dataset.pca_inverse(model=pca)

    trajectory = dataset.dataset

    # 1. anomalies with quantiles
    if ANOMALY_METHOD == 'quantile':
        anomalies = ((errors > pc_upper_bound) | (errors < pc_lower_bound)) == True
        anomalies = pd.DataFrame(anomalies, columns=Dataset.DEFAULT_COLUMNS)
        trajectory_anomalies = dataset.dataset[anomalies]
    # 2. anomalies with median threshold
    elif ANOMALY_METHOD == 'median':
        anomalies = (np.abs(errors) > threshold) == True
        anomalies = pd.DataFrame(anomalies, columns=Dataset.DEFAULT_COLUMNS)
        trajectory_anomalies = dataset.dataset[anomalies]
    # 3. anomalies with anomaly score and threshold
    elif ANOMALY_METHOD == 'score':
        anomaly_score = pd.DataFrame(np.sum(np.abs(errors), axis=1), columns=['score'])
        anomalies = (anomaly_score > threshold) == True
        # non_anomalies = (anomaly_score <= threshold) != True

        trajectory_anomalies = dataset.dataset[anomalies.to_numpy()]

        plot_anomaly_score(t_name, anomaly_score, anomalies, threshold)
    # 4. anomalies with mahalanobis distance
    elif ANOMALY_METHOD == 'mahalanobis':
        mahalanobis_anomaly_score = pd.DataFrame(mahalanobis_distance, columns=['score'])
        anom_index = np.where((mahalanobis_anomaly_score > mahalanobis_distance_nominal) == True)
        anomalies_list = []

        plot_anomaly_score(t_name, mahalanobis_anomaly_score, None, mahalanobis_distance_nominal)

        anomaly_score = pd.DataFrame(np.sum(np.abs(errors), axis=1), columns=['score'])
        anomalies = ((anomaly_score > threshold) == True)
        non_anomalies = (anomaly_score <= threshold) != True

        if len(anom_index[0]) > 0:
            trajectory_anomalies = pd.concat([dataset.dataset[anomalies.to_numpy()], dataset.dataset[non_anomalies.to_numpy()]])

            for index in anom_index[0]:
                start = index*WINDOW_SIZE
                end = index*WINDOW_SIZE+WINDOW_SIZE
                window_anomalies = trajectory_anomalies.iloc[index*WINDOW_SIZE:index*WINDOW_SIZE+WINDOW_SIZE, :]
                anomalies_list.append(window_anomalies)
            for index in range(len(mahalanobis_anomaly_score)):
                if index not in anom_index[0]:
                    anomalies.iloc[index*WINDOW_SIZE:index*WINDOW_SIZE+WINDOW_SIZE, :] = False
                    non_anomalies.iloc[index*WINDOW_SIZE:index*WINDOW_SIZE+WINDOW_SIZE, :] = True
            trajectory_anomalies = pd.concat(anomalies_list, axis=0)
        else:
            for index in range(len(mahalanobis_anomaly_score)):
                if index not in anom_index[0]:
                    anomalies.iloc[index*WINDOW_SIZE:index*WINDOW_SIZE+WINDOW_SIZE, :] = False
                    non_anomalies.iloc[index*WINDOW_SIZE:index*WINDOW_SIZE+WINDOW_SIZE, :] = True
            trajectory_anomalies = pd.DataFrame([], columns=Dataset.DEFAULT_COLUMNS)
    # 5. anomalies with one class svm
    elif ANOMALY_METHOD == 'oneclass':
        one_class_predict = one_class_model.predict(errors)
        anomalies = np.where(one_class_predict == -1)
        trajectory_anomalies = dataset.dataset.loc[anomalies[0]]
        # plot_anomaly_score(t_name, anomaly_score, anomalies, threshold)
    # scaler inverse transform
    dataset.normalize_inverse()
    # remove sliding window from dataset_processed
    dataset.remove_sliding_window()

    plot_trajectory(f"Test corrupted trajectory with Anomalies: {t_name}", trajectory, trajectory_anomalies,
                    anomalies=dataset.anomalies)

    plot_trajectory(f"Reconstruction of test corrupted trajectory: {t_name}",
                    {'original': dataset.dataset, 'reconstruct': dataset.dataset_processed},
                    None,
                    {'original': 'black', 'reconstruct': 'orange'})

    if ANOMALY_METHOD == 'oneclass':
        continue

    # Confusion matrix
    trajectory_predicted_anomalies = dataset.dataset.copy()
    if ANOMALY_METHOD in ['score', 'mahalanobis']:
        trajectory_predicted_anomalies.iloc[:, :] = 0
        trajectory_predicted_anomalies.loc[anomalies.loc[anomalies['score'] == True].index.to_numpy(), :] = 1
        trajectory_predicted_anomalies = pd.DataFrame(trajectory_predicted_anomalies.iloc[:, 0])
    else:
        trajectory_predicted_anomalies.iloc[:, :] = 1
        trajectory_predicted_anomalies = trajectory_predicted_anomalies[anomalies].fillna(0)

    trajectory_true_anomalies = dataset.dataset.copy()
    if ANOMALY_METHOD in ['score', 'mahalanobis']:
        trajectory_true_anomalies = pd.DataFrame(trajectory_true_anomalies.iloc[:, 0])
    trajectory_true_anomalies.iloc[:, :] = 0

    count = 0
    for key, indexes in dataset.anomalies.items():
        if ANOMALY_METHOD in ['score', 'mahalanobis']:
            trajectory_true_anomalies.loc[indexes, :] = 1
        else:
            trajectory_true_anomalies.loc[indexes, key] = 1
        count += len(indexes)

    trajectory_true_anomalies = trajectory_true_anomalies.to_numpy().flatten()
    trajectory_predicted_anomalies = trajectory_predicted_anomalies.to_numpy().flatten()

    print(f"Anomalies: {count}")
    print(f"\t* True anomalies:{np.count_nonzero(trajectory_true_anomalies == 1)}")
    print(f"\t* Predicted anomalies: {np.count_nonzero(trajectory_predicted_anomalies == 1)}")

    count_anomalies = np.count_nonzero(trajectory_true_anomalies == 1)
    count_non_anomalies = np.count_nonzero(trajectory_true_anomalies == 0)

    count_matrix = np.array([[count_non_anomalies, count_non_anomalies], [count_anomalies, count_anomalies]])
    cf_matrix = confusion_matrix(trajectory_true_anomalies, trajectory_predicted_anomalies, labels=[0, 1])
    cf_matrix_perc = np.around((cf_matrix * 100 / count_matrix), decimals=2)

    #########################################################################
    #	 					PLOT CONFUSION MATRIX							#
    #########################################################################
    plt.figure()
    group_names = ['True Neg', 'False Pos', 'False Neg', 'True Pos']
    group_counts = ["{0:0.0f}".format(value) for value in cf_matrix.flatten()]
    group_percentages = [f"{value}%" for value in cf_matrix_perc.flatten()]

    labels = [f"{l1}\n{l2}\n{l3}" for l1, l2, l3 in zip(group_names, group_counts, group_percentages)]
    labels = np.asarray(labels).reshape(2, 2)

    ax = sns.heatmap(cf_matrix_perc, annot=labels, fmt='')

    ax.set_title(f'Anomalies Confusion Matrix: {t_name}\n\n')
    ax.set_xlabel('\nPredicted Values')
    ax.set_ylabel('Actual Values ')

    ## Ticket labels - List must be in alphabetical order
    ax.xaxis.set_ticklabels(['False', 'True'])
    ax.yaxis.set_ticklabels(['False', 'True'])
    #########################################################################
    #	 					END PLOT CONFUSION MATRIX						#
    #########################################################################
    plt.show()

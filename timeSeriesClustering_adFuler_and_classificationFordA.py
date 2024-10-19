# conda activate iTF2

# Python 3.10.14
# pandas == 2.2.2
# sklearn == 1.5.2
# statsmodels == 0.14.4
# matplotlib == 3.9.2
# numpy == 1.26.4
# keras == 3.5.0
# xgboost == 2.1.1
# lightgbm == 4.5.0
# catboost == 1.2.7
# sktime == 0.33.1


import pandas as pd
from sklearn.cluster import (KMeans,
                            AffinityPropagation,
                            SpectralClustering,
                            Birch)
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import (MaxAbsScaler,
                                   StandardScaler, 
                                   normalize)

from sklearn.metrics import (silhouette_score,
                            accuracy_score, 
                            confusion_matrix, 
                            classification_report, 
                            matthews_corrcoef,
                            roc_curve, 
                            roc_auc_score,
                            auc,
                            cohen_kappa_score)
from sklearn.decomposition import PCA
from statsmodels.tsa.stattools import adfuller
import matplotlib.pyplot as plt
import numpy as np
import keras

from sklearn.ensemble import (AdaBoostClassifier,
                            HistGradientBoostingClassifier,
                            RandomForestClassifier,
                            ExtraTreesClassifier,
                            GradientBoostingClassifier)

from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
import xgboost as xgb
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.calibration import CalibratedClassifierCV
import lightgbm as lgb
from sklearn.svm import SVC
from catboost import CatBoostClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

from sklearn.pipeline import make_pipeline
from sktime.transformations.panel.reduce import Tabularizer

import warnings
warnings.filterwarnings("ignore")

# data = pd.read_csv('lacity.org-website-traffic.csv', 
#                    parse_dates=['Date']).loc[:, ['Date', 'Device Category', 'Browser', 'Sessions']]

# print(data.head())
# print(data.info(), '\n')
# print(data.describe(), '\n')

# data = data.drop(['Device Category'], axis=1)
# wide_df = data.groupby(['Date', 'Browser']).sum().unstack().T.fillna(0).reset_index(level=0, drop=True)
# print(wide_df.head())


def readucr(filename):
    data = np.loadtxt(filename, delimiter="\t")
    y = data[:, 0]
    x = data[:, 1:]
    return x, y.astype(int)


x_train, y_train = readucr("FordA_TRAIN.tsv")
x_test, y_test = readucr("FordA_TEST.tsv")

classes = np.unique(np.concatenate((y_train, y_test), axis=0))

wide_df = pd.DataFrame(x_test)

print(wide_df.head())
print(wide_df.info(), '\n')
print(wide_df.describe(), '\n')

'''
Автокорреляционный тест Дики-Фуллера
'''
stat_or_notstat = []
p_value = []
for i in range(wide_df.shape[0]):
    adf_test = adfuller(wide_df.iloc[i])
 
    # выведем p-value
    # print('p-value = ' + str(format(adf_test[1], '.3e'))) 

    if adf_test[1] < 0.05: # Используем пороговое значение, равное 0,05 (5%).
        stat_or_notstat.append('стационарен')
        p_value.append(format(adf_test[1], '.3e'))
    else:
        stat_or_notstat.append('нестационарен')
        p_value.append(format(adf_test[1], '.3e'))

'''
Кластеризация
'''
prep = MaxAbsScaler()
kmeans = KMeans(n_clusters=2, init='k-means++', max_iter=300, n_init=3, random_state=0)
affin = AffinityPropagation(max_iter=300, random_state=42)
brc = Birch(n_clusters=2)
gauss = GaussianMixture(n_components=2, random_state=0)

scaled_data = prep.fit_transform(wide_df)
kmeans.fit(scaled_data)
affin.fit(scaled_data)
brc.fit(scaled_data)


# Scaling the Data 
scaler = StandardScaler() 
X_scaled = scaler.fit_transform(wide_df) 
# Normalizing the Data 
X_normalized = normalize(X_scaled) 
# Converting the numpy array into a pandas DataFrame 
X_normalized = pd.DataFrame(X_normalized) 
# Reducing the dimensions of the data 
pca = PCA(n_components = 2) 
X_principal = pca.fit_transform(X_normalized) 
X_principal = pd.DataFrame(X_principal) 
X_principal.columns = ['P1', 'P2'] 
print(X_principal.head())
  
gauss_labels = gauss.fit_predict(X_principal)

for n_cluster in range(2, 10):
    kmeans = KMeans(n_clusters=n_cluster, random_state=0)
    kmeans.fit(scaled_data)
    print('kmeans n_cluster =', n_cluster, silhouette_score(scaled_data, kmeans.labels_))

print("silhouette_score for affin's clusters", silhouette_score(scaled_data, affin.labels_))
print("silhouette_score for brc's clusters", silhouette_score(scaled_data, brc.labels_))
print("silhouette_score for gauss's clusters", silhouette_score(scaled_data, gauss_labels))

# Time Series Clustering With PCA 

prep = MaxAbsScaler()
scaled_data = prep.fit_transform(wide_df)

pca = PCA(n_components=2)
pca_data = pca.fit_transform(scaled_data)

kmeans_pca = KMeans(n_clusters=2, random_state=0)
kmeans_pca.fit(pca_data)

print(f'kmeans_pca.n_clusters = {kmeans_pca.n_clusters}')

clusterslist = []
for cluster in range(kmeans_pca.n_clusters):
    cluster_data = wide_df[kmeans_pca.labels_ == cluster]
    for i in range(cluster_data.shape[0]):
        clusterslist.append((cluster,cluster_data.index[i]))

# формирование списка кластерных прогнозовых меток для записи в файл
clusterslistorder = np.zeros((len(clusterslist)), dtype=int)
for i in range(len(clusterslist)):
    if clusterslist[i][0] == 1:
        clusterslistorder[clusterslist[i][1]] = 1

print("silhouette_score for PCA's clusters", silhouette_score(pca_data, kmeans_pca.labels_))

'''
Подготовка данных для простой 1-размерной-свёрточной нейросети-классификатора
'''
x_train = x_train.reshape((x_train.shape[0], x_train.shape[1], 1))
x_test = x_test.reshape((x_test.shape[0], x_test.shape[1], 1))
num_classes = len(np.unique(y_train))
idx = np.random.permutation(len(x_train))
x_train = x_train[idx]
y_train = y_train[idx]
y_train[y_train == -1] = 0
y_test[y_test == -1] = 0

# '''
# A train part
# '''
def make_model(input_shape):
    input_layer = keras.layers.Input(input_shape)

    conv1 = keras.layers.Conv1D(filters=64, kernel_size=3, padding="same")(input_layer)
    conv1 = keras.layers.BatchNormalization()(conv1)
    conv1 = keras.layers.ReLU()(conv1)

    conv2 = keras.layers.Conv1D(filters=64, kernel_size=3, padding="same")(conv1)
    conv2 = keras.layers.BatchNormalization()(conv2)
    conv2 = keras.layers.ReLU()(conv2)

    conv3 = keras.layers.Conv1D(filters=64, kernel_size=3, padding="same")(conv2)
    conv3 = keras.layers.BatchNormalization()(conv3)
    conv3 = keras.layers.ReLU()(conv3)

    gap = keras.layers.GlobalAveragePooling1D()(conv3)

    output_layer = keras.layers.Dense(num_classes, activation="softmax")(gap)

    return keras.models.Model(inputs=input_layer, outputs=output_layer)

model = make_model(input_shape=x_train.shape[1:])
keras.utils.plot_model(
    model,
    to_file="model.png",
    show_shapes=True,
    show_dtype=False,
    show_layer_names=True,
    rankdir="TB",
    expand_nested=False,
    dpi=200,
    show_layer_activations=True,
    show_trainable=False,
)


# """
# ## Train the model

# """
epochs = 500
batch_size = 32

callbacks = [
    keras.callbacks.ModelCheckpoint(
        "best_model.keras", save_best_only=True, monitor="val_loss" # best_model.tf для старых версий keras-2.14.0
    ),
    keras.callbacks.ReduceLROnPlateau(
        monitor="val_loss", factor=0.5, patience=20, min_lr=0.0001
    ),
    keras.callbacks.EarlyStopping(monitor="val_loss", patience=50, verbose=1),
]


model.compile(
    optimizer="adam",
    loss="sparse_categorical_crossentropy",
    metrics=["sparse_categorical_accuracy"],
)

model.summary()

history = model.fit(
    x_train,
    y_train,
    batch_size=batch_size,
    epochs=epochs,
    callbacks=callbacks,
    validation_split=0.2,
    verbose=1,
)
"""
## Evaluate model on test data
"""
model = keras.models.load_model("best_model.keras")
test_loss, test_acc = model.evaluate(x_test, y_test)
print("\nTest accuracy = ", test_acc)
print("Test loss = ", test_loss, '\n')
"""
## Plot the model's training and validation loss
"""
metric = "sparse_categorical_accuracy"
plt.figure()
plt.plot(history.history[metric])
plt.plot(history.history["val_" + metric])
plt.title("model " + metric)
plt.ylabel(metric, fontsize="large")
plt.xlabel("epoch", fontsize="large")
plt.legend(["train", "val"], loc="best")
# plt.show()
plt.grid()
plt.savefig('sparse_categorical_accuracy.png')
plt.close()


####################################################
## Transformer
####################################################
def transformer_encoder(inputs, head_size, num_heads, ff_dim, dropout=0):
    # Attention and Normalization
    x = keras.layers.MultiHeadAttention(
        key_dim=head_size, num_heads=num_heads, dropout=dropout
    )(inputs, inputs)
    x = keras.layers.Dropout(dropout)(x)
    x = keras.layers.LayerNormalization(epsilon=1e-6)(x)
    res = x + inputs

    # Feed Forward Part
    x = keras.layers.Conv1D(filters=ff_dim, kernel_size=1, activation="relu")(res)
    x = keras.layers.Dropout(dropout)(x)
    x = keras.layers.Conv1D(filters=inputs.shape[-1], kernel_size=1)(x)
    x = keras.layers.LayerNormalization(epsilon=1e-6)(x)
    return x + res

def build_model(
    input_shape,
    head_size,
    num_heads,
    ff_dim,
    num_transformer_blocks,
    mlp_units,
    dropout=0,
    mlp_dropout=0,
):
    inputs = keras.Input(shape=input_shape)
    x = inputs
    for _ in range(num_transformer_blocks):
        x = transformer_encoder(x, head_size, num_heads, ff_dim, dropout)

    x = keras.layers.GlobalAveragePooling1D(data_format="channels_last")(x)
    for dim in mlp_units:
        x = keras.layers.Dense(dim, activation="relu")(x)
        x = keras.layers.Dropout(mlp_dropout)(x)
    outputs = keras.layers.Dense(num_classes, activation="softmax")(x)
    return keras.Model(inputs, outputs)

transmodel = build_model(                         # 2 вариант -- сеть-трансформер
    input_shape=x_train.shape[1:],
    head_size=256,
    num_heads=4,
    ff_dim=4,
    num_transformer_blocks=4,
    mlp_units=[128],
    mlp_dropout=0.4,
    dropout=0.25,
)

keras.utils.plot_model(
    transmodel,
    to_file="model-transformer.png",
    show_shapes=True,
    show_dtype=False,
    show_layer_names=True,
    rankdir="TB",
    expand_nested=False,
    dpi=200,
    show_layer_activations=True,
    show_trainable=False,
)

transcallbacks = [
    keras.callbacks.ModelCheckpoint(
        "best_transmodel.keras", save_best_only=True, monitor="val_loss"
    ),
    keras.callbacks.ReduceLROnPlateau(
        monitor="val_loss", factor=0.5, patience=20, min_lr=0.0001
    ),
    keras.callbacks.EarlyStopping(monitor="val_loss", patience=1, verbose=1),
]

transmodel.compile(
    optimizer="adam",
    loss="sparse_categorical_crossentropy",
    metrics=["sparse_categorical_accuracy"],
)

transmodel.summary()

transhistory = transmodel.fit(
    x_train,
    y_train,
    batch_size=batch_size,
    epochs=epochs,
    callbacks=transcallbacks,
    validation_split=0.2,
    verbose=1,
)

# """
# ## Evaluate model on test data
# """

transmodel = keras.models.load_model("best_transmodel.keras")
transtest_loss, transtest_acc = transmodel.evaluate(x_test, y_test)
print("\nTest transformer accuracy = ", transtest_acc)
print("Test transformer loss = ", transtest_loss, '\n')

plt.figure()
plt.plot(transhistory.history[metric])
plt.plot(transhistory.history["val_" + metric])
plt.title("transmodel " + metric)
plt.ylabel(metric, fontsize="large")
plt.xlabel("epoch", fontsize="large")
plt.legend(["train", "val"], loc="best")
# plt.show()
plt.grid()
plt.savefig('trans_sparse_categorical_accuracy.png')
plt.close()

'''
A predict part
'''
model = keras.models.load_model("best_model.keras")

y_pred = model.predict(x_test)
print(f'y_pred = {y_pred}')

y_predict = []
y_predict_probability = []
for i in range(len(y_pred)):
    if y_pred[i][0] > y_pred[i][1]:
        y_predict.append(0)
        y_predict_probability.append(y_pred[i][0])
    else:
       y_predict.append(1)
       y_predict_probability.append(y_pred[i][1]) 

print(f'\ny_test = {y_test},\ny_predict = {np.array(y_predict)},\ny_predict_probability = {np.array(y_predict_probability)}\n')

confusion = confusion_matrix(y_test, y_predict)
print(confusion)

print(f"\nКоэффициент корреляции Мэтьюса (sklearn.metrics) = {matthews_corrcoef(y_test, y_predict):.2f}\n")

print("Classification Report:")
print(classification_report(y_test, y_predict))


plt.figure(figsize=(8, 6))
all_fpr, all_tpr, _ = roc_curve(y_test, y_predict)
plt.plot(all_fpr, all_tpr, 'orange')
plt.xlabel(r'False Positive Rate ($FPR = \frac{FP}{FP + TN}$)')
plt.ylabel(r'True Positive Rate ($TPR = \frac{TP}{TP + FN}$)')
roc_auc = auc(all_fpr, all_tpr)
plt.title(f'ROC curves for Time-Series-classification (area = {roc_auc:.2f})')
plt.grid()
plt.xlim(0, 1)
plt.ylim(0, 1.05)
# plt.show()
plt.savefig('ROC-simplemodel.png')
plt.close()
# https://www.geeksforgeeks.org/how-to-plot-roc-curve-in-python/
print(f'ROC-AUC area = {roc_auc:.2f}')

Kappa = cohen_kappa_score(y_test, y_predict)
print(f'\nKappa = {Kappa:.2f}')

####################################################
## Transformer
####################################################

transmodel = keras.models.load_model("best_transmodel.keras")
y_pred_trans = transmodel.predict(x_test)
print(f"y_pred_trans = {y_pred_trans}")

y_predict_trans = []

for i in range(len(y_pred)):
    if y_pred_trans[i][0] > y_pred_trans[i][1]:
        y_predict_trans.append(0)
    else:
       y_predict_trans.append(1)

print(f'\ny_test = {y_test},\ny_predict_trans = {np.array(y_predict_trans)}\n')

confusion = confusion_matrix(y_test, y_predict_trans)
print(confusion)

print(f"\nКоэффициент корреляции Мэтьюса (transformer) = {matthews_corrcoef(y_test, y_predict_trans):.2f}\n")

print("Classification Report for transformer:")
print(classification_report(y_test, y_predict_trans))


plt.figure(figsize=(8, 6))
all_fpr_trans, all_tpr_trans, _ = roc_curve(y_test, y_predict_trans)
plt.plot(all_fpr, all_tpr, 'orange')
plt.xlabel(r'False Positive Rate ($FPR = \frac{FP}{FP + TN}$)')
plt.ylabel(r'True Positive Rate ($TPR = \frac{TP}{TP + FN}$)')
roc_auc_trans = auc(all_fpr_trans, all_tpr_trans)
plt.title(f'ROC curves for Time-Series-classification (area = {roc_auc:.2f})')
plt.grid()
plt.xlim(0, 1)
plt.ylim(0, 1.05)
# plt.show()
plt.savefig('ROC-simplemodel.png')
plt.close()
# https://www.geeksforgeeks.org/how-to-plot-roc-curve-in-python/
print(f'ROC-AUC area = {roc_auc_trans:.2f}')

Kappa = cohen_kappa_score(y_test, y_predict_trans)
print(f'\nKappa = {Kappa:.2f}')

'''
Классификация пакетными классификаторами
'''
classifier1 = make_pipeline(Tabularizer(), AdaBoostClassifier(n_estimators=300,
                                                            learning_rate=0.01, # требуется "ручной" подбор скорости обучения для максимизации точности
                                                            random_state=42))
classifier2 = make_pipeline(Tabularizer(), HistGradientBoostingClassifier(max_iter=100))
classifier3 = make_pipeline(Tabularizer(), RandomForestClassifier(random_state=0, n_jobs=-1))
classifier4 = make_pipeline(Tabularizer(), ExtraTreesClassifier(n_estimators=10, max_depth=None, min_samples_split=2, random_state=0))
classifier5 = make_pipeline(Tabularizer(), GradientBoostingClassifier())
classifier6 = make_pipeline(Tabularizer(), DecisionTreeClassifier(random_state=0))
classifier7 = make_pipeline(Tabularizer(), GaussianNB())
classifier8 = make_pipeline(Tabularizer(), xgb.XGBClassifier())
classifier9 = make_pipeline(Tabularizer(), MLPClassifier(random_state=1, max_iter=300))
classifier10 = make_pipeline(Tabularizer(), KNeighborsClassifier(n_neighbors=5))
classifier11 = make_pipeline(Tabularizer(), CalibratedClassifierCV())
classifier12 = make_pipeline(Tabularizer(), lgb.LGBMClassifier(metric='auc'))
classifier13 = make_pipeline(Tabularizer(), SVC(class_weight='balanced'))
classifier14 = make_pipeline(Tabularizer(), CatBoostClassifier(auto_class_weights='Balanced'))
classifier15 = make_pipeline(Tabularizer(), LinearDiscriminantAnalysis(solver='eigen'))
classifier16 = make_pipeline(Tabularizer(), 
                                        CatBoostClassifier(iterations=1100,  # Number of boosting iterations
                                                           learning_rate=0.1, ## ROC-AUC area = 0.84 # 0.01,  ## ROC-AUC area = 0.83
                                                           depth=6, ## ROC-AUC area = 0.85  # =8 Depth of the tree - Глубина деревьев. Регулирует сложность модели. Глубокие деревья могут лучше выявлять сложные зависимости, но также рискуют переобучиться.
                                                           verbose=100,  # Print training progress every 50 iterations
                                                           early_stopping_rounds=10,  # stops training if no improvement in 10 consequtive rounds
                                                           loss_function='Logloss',
                                                           custom_metric=['Accuracy', 'AUC'],
                                                           use_best_model=False, # you must have a eval_set for =True
                                                           random_seed=42,
                                                           auto_class_weights='Balanced')) # fine-tuning

classifiers = [classifier1,
               classifier2,
               classifier3,
               classifier4,
               classifier5,
               classifier6,
               classifier7,
               classifier8,
               classifier9,
               classifier10,
               classifier11,
               classifier12,
               classifier13,
               classifier14,
               classifier15,
               classifier16,]

predicts_of_classifiers = [np.array(y_predict), 
                           np.array(y_predict_trans),]
weights_roc_auc_for_predict = [roc_auc, 
                               roc_auc_trans,]

for classifier in classifiers:
    classifier.fit(x_train, y_train)

    y_pred = classifier.predict(x_test)
    predicts_of_classifiers.append(y_pred)
    
    print(f"\naccuracy = {accuracy_score(y_test, y_pred)}")
    print(confusion_matrix(y_test, y_pred))
    print(f"\nКоэффициент корреляции Мэтьюса = {matthews_corrcoef(y_test, y_pred):.2f}\n")

    # print("Classification Report:")
    # print(classification_report(y_test, y_pred))

    all_fpr, all_tpr, _ = roc_curve(y_test, y_pred)
    roc_auc = auc(all_fpr, all_tpr)
    print(f'ROC-AUC area = {roc_auc:.2f}\n')
    weights_roc_auc_for_predict.append(roc_auc)

print(predicts_of_classifiers)
print(weights_roc_auc_for_predict, sum(weights_roc_auc_for_predict))

final_classifiers_predict = np.transpose(predicts_of_classifiers).dot(np.array(weights_roc_auc_for_predict))

print(final_classifiers_predict/sum(weights_roc_auc_for_predict))
print(np.round(final_classifiers_predict/sum(weights_roc_auc_for_predict)))

'''
Запись прогнозов в файл
'''
wide_df['PCAcluster'] = clusterslistorder
wide_df['p-value'] = p_value
wide_df['stattesttime'] = stat_or_notstat
wide_df['classification'] = np.array(y_predict)
wide_df['probability,%'] = np.round(np.array(y_predict_probability)*100, decimals = 2)
wide_df['final_classifiers_predict'] = np.round(final_classifiers_predict/sum(weights_roc_auc_for_predict))
wide_df['realclass'] = y_test

wide_df.to_csv(f'ford_adfuller_and_clusters_and_classification.csv')
'''
Визуализация кластеров в пространстве принципиальных компонент 
'''
for cluster in range(kmeans_pca.n_clusters):
    cluster_points = pca_data[kmeans_pca.labels_ == cluster]
    plt.scatter(cluster_points[:, 0], cluster_points[:, 1], label=f"Cluster {cluster}")
# Plot the cluster centroids as black 'X' markers
plt.scatter(kmeans_pca.cluster_centers_[:, 0], kmeans_pca.cluster_centers_[:, 1],
            color='black', marker='x', label='Centroids')
plt.title("Clusters in PCA Space")
plt.xlabel("Principal Component 1")
plt.ylabel("Principal Component 2")
plt.legend()
# Display the plot
plt.show()

'''
accuracy = 0.8393939393939394
[[582  99]
 [113 526]]

Коэффициент корреляции Мэтьюса = 0.68

ROC-AUC area = 0.84


accuracy = 0.5189393939393939
[[286 395]
 [240 399]]

Коэффициент корреляции Мэтьюса = 0.05

ROC-AUC area = 0.52

0:      learn: 0.6862676        total: 35.5ms   remaining: 39s
100:    learn: 0.3539041        total: 3.49s    remaining: 34.5s
200:    learn: 0.1725610        total: 6.91s    remaining: 30.9s
300:    learn: 0.0941598        total: 10.3s    remaining: 27.5s
400:    learn: 0.0563607        total: 13.8s    remaining: 24s
500:    learn: 0.0351928        total: 17.2s    remaining: 20.6s
600:    learn: 0.0230704        total: 20.6s    remaining: 17.1s
700:    learn: 0.0160182        total: 24.1s    remaining: 13.7s
800:    learn: 0.0116548        total: 27.5s    remaining: 10.3s
900:    learn: 0.0087401        total: 30.9s    remaining: 6.83s
1000:   learn: 0.0068768        total: 34.3s    remaining: 3.4s
1099:   learn: 0.0057125        total: 37.7s    remaining: 0us

accuracy = 0.8378787878787879
[[576 105]
 [109 530]]

Коэффициент корреляции Мэтьюса = 0.68

ROC-AUC area = 0.84

[array([0, 0, 0, ..., 1, 1, 1]), 
array([0, 0, 0, ..., 0, 0, 0]), 
array([0, 1, 1, ..., 0, 1, 1]), 
array([1, 0, 1, ..., 1, 1, 0]), 
array([0, 1, 1, ..., 1, 1, 0]), 
array([1, 0, 1, ..., 1, 0, 1]), 
array([1, 1, 0, ..., 1, 1, 1]), 
array([0, 0, 1, ..., 1, 0, 0]), 
array([0, 1, 1, ..., 1, 1, 1]), 
array([0, 0, 1, ..., 1, 0, 0]), 
array([0, 0, 1, ..., 1, 0, 0]), 
array([0, 0, 0, ..., 1, 1, 0]), 
array([0, 0, 0, ..., 0, 0, 0]), 
array([1, 0, 1, ..., 1, 1, 0]), 
array([0, 0, 1, ..., 1, 0, 1]), 
array([0, 0, 1, ..., 1, 0, 1], dtype=int64), 
array([0, 1, 1, ..., 0, 1, 1]), 
array([0, 0, 0, ..., 1, 0, 1], dtype=int64)]

[0.9712794633685617,
0.5,
0.49324040178417544, 
0.7840030885262629, 
0.7409567537382888, 
0.6372636208834013, 
0.661761563014898, 
0.5660344839472469, 
0.4916582214776668, 
0.7838583138576933, 
0.7618801403624882, 
0.7174825753345329, 
0.5, 
0.7851716269225731, 
0.8296967315395063, 
0.8388933700095826, 
0.522191888482141, 
0.8376179741198044] sum = 12.422990217368824

[0.23087838 0.23422773 0.6628717  ... 0.83776593 0.49647834 0.50580441]
[0. 0. 1. ... 1. 0. 1.]
'''
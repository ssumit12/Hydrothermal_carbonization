# Time series analysis of htc (dataHtc)

import os

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.ticker import FormatStrFormatter
from sklearn.compose import make_column_transformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.model_selection import cross_val_score
from sklearn.neighbors import KNeighborsRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.svm import SVR
from sklearn.utils import shuffle
from sklearn.inspection import permutation_importance

matplotlib.use('Agg')
matplotlib.rcParams['font.family'] = "Times New Roman"
matplotlib.rcParams['axes.linewidth'] = 1.5
pd.options.mode.chained_assignment = None  # default='warn'

# ----------------------------------------------------------------------------------------------------------------------
# Variables
# ----------------------------------------------------------------------------------------------------------------------
target = 'carbon'
removed_species = 'oxygen'
sample_in = False
param = dict(test_size=0.25, cv=5, scoring='mse', replicas=20, summary=False, grid_search=False)


# ----------------------------------------------------------------------------------------------------------------------
# Implemented Functions
# ----------------------------------------------------------------------------------------------------------------------

def read_data(file_name):
    df = pd.read_excel('{}'.format(file_name), sheet_name='data')
    return df


def columns_stats(df):
    _root = 'regression/dataSummary'
    if not os.path.exists(_root):
        os.makedirs(_root)
    # ---------------------------------
    statistics = pd.DataFrame()
    for column in df.columns:
        if column == df.columns[0]:
            statistics = pd.DataFrame(df[column].value_counts()).reset_index(drop=False)
            statistics.rename(columns={'index': column, column: 'Num_samples'}, inplace=True)
        else:
            temp = pd.DataFrame(df[column].value_counts()).reset_index(drop=False)
            temp.rename(columns={'index': column, column: 'Num_samples'}, inplace=True)
            statistics = pd.concat([statistics, temp], axis=1)
    excel_output(statistics, _root=_root, file_name='columnsStats', csv=False)


def view_data_twin_x(df):
    _root = 'regression/dataSummary/twinX'
    if not os.path.exists(_root):
        os.makedirs(_root)
    # ---------------------------------
    df1 = df.copy(deep=True)
    df1 = df1.loc[df1['type'] == 'static']
    _samples = df1['sample'].unique()
    for _sample in _samples:
        df2 = df1.loc[df1['sample'] == _sample]
        temperatures = df2['temperature'].unique()
        color_left = {200: 'blue', 230: 'red', 260: 'green'}
        color_right = {200: 'steelblue', 230: 'indianred', 260: 'limegreen'}
        columns = ['carbon', 'oxygen']
        fig, ax1 = plt.subplots(1, figsize=(12, 10))
        ax2 = ax1.twinx()
        for temp in temperatures:
            df3 = df2.loc[df2['temperature'] == temp]
            _X = df3['time']
            _y = df3[columns[0]]
            ax1.plot(_X, _y, '-o', linewidth=2.0, color=color_left[temp], label='{} $^o$C ({})'.format(temp, target))
            _y = df3[columns[1]]
            ax2.plot(_X, _y, ':o', linewidth=2.0, color=color_right[temp], label='{} $^o$C (oxygen)'.format(temp))
        # ---------------------------------
        ax1.text(0.03, 0.95, '{}'.format(_sample),
                 ha='left', va='center', transform=ax1.transAxes,
                 fontdict={'color': 'k', 'weight': 'bold', 'size': 32})
        # ---------------------------------
        plt.grid(axis='both', linewidth=0.5)
        x_axis_index = np.linspace(0, 500, num=6)
        ax1.set_xticks(x_axis_index)
        ax1.set_xlim(0, 500)
        ax1.set_xticklabels(x_axis_index, fontsize=20)
        ax1.xaxis.set_major_formatter(FormatStrFormatter('%.0f'))
        ax1.set_xlabel('Time (min)', fontsize=32)
        # ---------------------------------
        y_axis_index = np.linspace(0, 100, num=6)
        ax1.set_yticks(y_axis_index)
        ax1.set_ylim(0, 100)
        ax1.set_yticklabels(y_axis_index, fontsize=20)
        ax1.yaxis.set_major_formatter(FormatStrFormatter('%.0f'))
        ax1.set_ylabel('elemental {} [%]'.format(columns[0]), fontsize=32)
        # ---------------------------------
        ax2.set_yticks(y_axis_index)
        ax2.set_ylim(0, 100)
        ax2.set_yticklabels(y_axis_index, fontsize=20)
        ax2.yaxis.set_major_formatter(FormatStrFormatter('%.0f'))
        ax2.set_ylabel('elemental {} [%]'.format(columns[1]), fontsize=32)
        # ---------------------------------
        ax1.legend(loc='lower right', bbox_to_anchor=(1.0, 1.0), ncol=3, fontsize=20)
        ax2.legend(loc='lower right', bbox_to_anchor=(1.0, 1.09), ncol=3, fontsize=19.6)
        plt.tight_layout()
        plt.savefig('{}/{}_{}_{}.png'.format(_root, _sample, columns[0], columns[1]))
        plt.close()


def view_data_dynamic(df):
    _root = 'regression/dataSummary/dynamic'
    if not os.path.exists(_root):
        os.makedirs(_root)
    # ---------------------------------
    df1 = df.copy(deep=True)
    df1 = df1.loc[df1['type'] == 'dynamic']
    _samples = df1['sample'].unique()
    color_left = {'cellulose': 'blue', 'straw': 'red', 'polar': 'green'}
    color_right = {'cellulose': 'steelblue', 'straw': 'indianred', 'polar': 'limegreen'}
    columns = ['carbon', 'oxygen']
    fig, ax1 = plt.subplots(1, figsize=(12, 10))
    ax2 = ax1.twinx()
    for _sample in _samples:
        df2 = df1.loc[df1['sample'] == _sample]
        _X = df2['temperature']
        _y = df2[columns[0]]
        ax1.plot(_X, _y, '-o', linewidth=2.0, color=color_left[_sample], label='{} ({})'.format(_sample, target))
        _y = df2[columns[1]]
        ax2.plot(_X, _y, ':o', linewidth=2.0, color=color_right[_sample], label='{} (oxygen)'.format(_sample))
    # ---------------------------------
    ax1.text(0.03, 0.95, '{}'.format('Dynamic'),
             ha='left', va='center', transform=ax1.transAxes,
             fontdict={'color': 'k', 'weight': 'bold', 'size': 32})
    # ---------------------------------
    plt.grid(axis='y', linewidth=0.5)
    x_axis_index = np.linspace(160, 260, num=11)
    ax1.set_xticks(x_axis_index)
    ax1.set_xlim(160, 260)
    ax1.set_xticklabels(x_axis_index, fontsize=20)
    ax1.xaxis.set_major_formatter(FormatStrFormatter('%.0f'))
    ax1.set_xlabel('Temperature ($^o$C)', fontsize=32)
    # ---------------------------------
    y_axis_index = np.linspace(30, 80, num=6)
    ax1.set_yticks(y_axis_index)
    ax1.set_ylim(30, 80)
    ax1.set_yticklabels(y_axis_index, fontsize=20)
    ax1.yaxis.set_major_formatter(FormatStrFormatter('%.0f'))
    ax1.set_ylabel('elemental {} [%]'.format(columns[0]), fontsize=32)
    # ---------------------------------
    ax2.set_yticks(y_axis_index)
    ax2.set_ylim(30, 80)
    ax2.set_yticklabels(y_axis_index, fontsize=20)
    ax2.yaxis.set_major_formatter(FormatStrFormatter('%.0f'))
    ax2.set_ylabel('elemental {} [%]'.format(columns[1]), fontsize=32)
    # ---------------------------------
    ax1.legend(loc='lower right', bbox_to_anchor=(1.0, 1.0), ncol=3, fontsize=20)
    ax2.legend(loc='lower right', bbox_to_anchor=(1.0, 1.09), ncol=3, fontsize=19.6)
    plt.tight_layout()
    plt.savefig('{}/dynamic.png'.format(_root))
    plt.close()


def view_data_individual(df):
    _root = 'regression/dataSummary/individual'
    if not os.path.exists(_root):
        os.makedirs(_root)
    # ---------------------------------
    df1 = df.copy(deep=True)
    df1 = df1.loc[df1['type'] == 'static']
    _samples = df1['sample'].unique()
    for _sample in _samples:
        df2 = df1.loc[df1['sample'] == _sample]
        temperatures = df2['temperature'].unique()
        color = {200: 'blue', 230: 'red', 260: 'green'}
        for column in ['nitrogen', 'sulfur', 'hydrogen', 'oxygen', 'carbon']:
            fig, ax = plt.subplots(1, figsize=(9, 9))
            for temp in temperatures:
                df3 = df2.loc[df2['temperature'] == temp]
                _X = df3['time']
                _y = df3[column]
                plt.plot(_X, _y, '-o', color=color[temp], label='{} $^o$C'.format(temp))
            # ---------------------------------
            plt.text(0.03, 0.95, '{}'.format(_sample),
                     ha='left', va='center', transform=ax.transAxes,
                     fontdict={'color': 'k', 'weight': 'bold', 'size': 32})
            # ---------------------------------
            plt.grid(axis='both', linewidth=0.5)
            x_axis_index = np.linspace(0, 500, num=6)
            ax.set_xticks(x_axis_index)
            ax.set_xlim(0, 500)
            ax.set_xticklabels(x_axis_index, fontsize=20)
            ax.xaxis.set_major_formatter(FormatStrFormatter('%.0f'))
            ax.set_xlabel('Time (min)', fontsize=32)
            # ---------------------------------
            y_axis_index = np.linspace(0, 100, num=6)
            ax.set_yticks(y_axis_index)
            ax.set_ylim(0, 100)
            ax.set_yticklabels(y_axis_index, fontsize=20)
            ax.yaxis.set_major_formatter(FormatStrFormatter('%.0f'))
            ax.set_ylabel('elemental {} [%]'.format(column), fontsize=32)
            # ---------------------------------
            plt.legend(loc='upper right', fontsize=23, ncol=1, fancybox=True, shadow=True)
            plt.tight_layout()
            plt.savefig('{}/{}_{}.png'.format(_root, _sample, column))
            plt.close()


def summary_data(df):
    columns_stats(df)
    view_data_twin_x(df)
    view_data_dynamic(df)
    view_data_individual(df)


def excel_output(_object, _root, file_name, csv):
    if csv:
        if _root != '':
            _object.to_csv('{}/{}.csv'.format(_root, file_name))
        else:
            _object.to_csv('{}.csv'.format(file_name))
    else:
        if _root != '':
            _object.to_excel('{}/{}.xls'.format(_root, file_name))
        else:
            _object.to_csv('{}.xls'.format(file_name))


# ----------------------------------------------------------------------------------------------------------------------
def select_features(df):
    df1 = df.copy(deep=True)
    df1 = df1.drop([removed_species, 'temperature'], axis=1)
    if not sample_in:
        df1 = df1.drop(['sample_cellulose', 'sample_straw', 'sample_polar'], axis=1)
    return df1


def recursive_features(df):
    df1 = df.copy(deep=True)
    df1 = df1.drop(['temperature'], axis=1)
    species = ['nitrogen', 'sulfur', 'hydrogen', 'oxygen', 'carbon']
    df1 = df1.drop(species, axis=1)
    # for s in species[0:2]:
    #     first, second, third = df.loc[0, s], df.loc[1, s], df.loc[2, s]
    #     df1['{}_0'.format(s)] = first.repeat(len(df1))
    #     df1['{}_1'.format(s)] = second.repeat(len(df1))
    #     df1['{}_2'.format(s)] = third.repeat(len(df1))
    df1['{}_3dt'.format(target)] = df.loc[:, target].shift(3)
    df1['{}_2dt'.format(target)] = df.loc[:, target].shift(2)
    df1['{}_1dt'.format(target)] = df.loc[:, target].shift(1)
    df1[target] = df.loc[:, target]
    df1 = df1.dropna().reset_index(drop=True)
    return df1


def interpolate_features(df):
    df1 = df.copy(deep=True)
    df1 = df1.drop(['temperature'], axis=1)
    species = ['nitrogen', 'sulfur', 'hydrogen', 'oxygen', 'carbon']
    df1 = df1.drop(species, axis=1)
    for s in species:
        initial, middle, final = df.loc[0, s], df.loc[5, s], df.loc[10, s]
        df1['{}_i'.format(s)] = initial.repeat(len(df1))
        df1['{}_m'.format(s)] = middle.repeat(len(df1))
        df1['{}_f'.format(s)] = final.repeat(len(df1))
    df1[target] = df.loc[:, target]
    return df1


def encode_data(df):
    cat_index = df.columns[(df.dtypes == 'object').values]
    num_index = df.columns[(df.dtypes != 'object').values]
    ohe = OneHotEncoder(sparse=False, handle_unknown='ignore')
    sc = StandardScaler()
    ct = make_column_transformer((ohe, cat_index), (sc, num_index))
    ct.fit_transform(df)
    df2 = ct.transform(df)
    # ---------------------------------
    names = []
    for cat in cat_index:
        unique = df[cat].value_counts().sort_index()
        for name in unique.index:
            names.append('{}_{}'.format(cat, name))
    for num in num_index:
        names.append(num)
    # ---------------------------------
    df2 = pd.DataFrame(df2)
    df2.columns = names
    return df2


def split_data_random(df):
    df = df.copy(deep=True)
    df = shuffle(df)
    head = int((1 - param['test_size']) * len(df))
    tail = len(df) - head
    df_train = df.head(head).reset_index(drop=True)
    df_test = df.tail(tail).reset_index(drop=True)
    return df_train, df_test


def split_xy(df, _shuffle):
    if _shuffle:
        df = shuffle(df)
    _X = df.iloc[:, 0:-1].reset_index(drop=True)
    _y = df.iloc[:, -1].to_numpy()
    return _X, _y


def grid_search(model):
    models = []
    hp1 = {'MLP': [(2,), (4,), (6,), (8,), (10,),
                   (2, 2), (4, 4), (6, 6), (8, 8), (10, 10),
                   (2, 2, 2), (4, 4, 4), (6, 6, 6), (8, 8, 8), (10, 10, 10),
                   (2, 2, 2, 2), (4, 4, 4, 4), (6, 6, 6, 6), (8, 8, 8, 8), (10, 10, 10, 10),
                   (2, 2, 2, 2, 2), (4, 4, 4, 4, 4), (6, 6, 6, 6, 6), (8, 8, 8, 8, 8), (10, 10, 10, 10, 10)],
           'SVM': [1, 0.1, 0.01, 0.001, 0.0001],
           'RF': [10, 50, 100, 200, 500],
           'KNN': [1, 2, 3, 4, 5, 6, 7]}
    hp2 = {'MLP': ['constant'],
           'SVM': [1, 5, 10, 100, 1000],
           'RF': [0.6, 0.7, 0.8, 0.9, 1.0],
           'KNN': ['uniform', 'distance']}
    for n in hp1[model]:
        for m in hp2[model]:
            if model == 'MLP':
                models.append(('MLP_{}_{}'.format(n, m), MLPRegressor(max_iter=10000,
                                                                      hidden_layer_sizes=n, learning_rate=m)))
            elif model == 'SVM':
                models.append(('SVM_{}_{}'.format(n, m), SVR(gamma=n, C=m)))
            elif model == 'RF':
                models.append(('RF_{}_{}'.format(n, m), RandomForestRegressor(n_estimators=n, max_features=m)))
            elif model == 'KNN':
                models.append(('KNN_{}_{}'.format(n, m), KNeighborsRegressor(n_neighbors=n, weights=m)))
    return models


def compare_models(df, models):
    scoring, cv, replicas = 'neg_mean_squared_error', param['cv'], param['replicas']
    if param['scoring'] == 'r2':
        scoring = 'r2'
    # ---------------------------------
    results = pd.DataFrame()
    for i in range(replicas):
        _X_train, _y_train = split_xy(df, True)
        temp = []
        for name, model in models:
            print(name)
            cv_results = cross_val_score(model, _X_train, _y_train, cv=cv, scoring=scoring)
            cv_results = np.mean(cv_results)
            temp.append(cv_results)
        if i == 0:
            results = pd.DataFrame(temp)
        else:
            results = pd.concat([results, pd.DataFrame(temp)], axis=1, ignore_index=True)
    results['mean'] = results.mean(axis=1)
    results['std'] = results.std(axis=1)
    # ---------------------------------
    _names, _models = [], []
    for name, model in models:
        _names.append(name)
        _models.append(model)
    results['name'] = pd.Series(_names)
    results['model'] = pd.Series(_models)
    # ---------------------------------
    id_best = results['mean'].idxmax()
    _best = results.loc[id_best, 'model']
    return results, _best


def prediction(df, estimator):
    replicas = param['replicas']
    errors = pd.DataFrame()
    for i in range(replicas):
        df_training, df_testing = split_data_random(df)
        _X_train, _y_train = split_xy(df_training, True)
        estimator.fit(_X_train, _y_train)
        _X_test, _y_test = split_xy(df_testing, True)
        _y_pred = estimator.predict(_X_test)
        errors.loc[i, 'r2'] = r2_score(_y_test, _y_pred)
        errors.loc[i, 'mse'] = mean_squared_error(_y_test, _y_pred)
        errors.loc[i, 'mae'] = mean_absolute_error(_y_test, _y_pred)
        errors.loc[i, 'rmse'] = np.sqrt(mean_squared_error(_y_test, _y_pred))
    _scores = [('R2', np.mean(errors['r2']), np.std(errors['r2'])),
               ('MSE', np.mean(errors['mse']), np.std(errors['mse'])),
               ('MAE', np.mean(errors['mae']), np.std(errors['mae'])),
               ('RMSE', np.mean(errors['rmse']), np.std(errors['rmse']))]
    return _scores


def sensitivity(df_original, df, _experiment):
    df_time = df_original.copy(deep=True)
    df_time = df_time.loc[df_time['Experiment'] == _experiment].reset_index(drop=True)
    replicas_time = df_time['initial_corrosion_mm_yr'].unique()
    df_time = df_time.loc[df_time['initial_corrosion_mm_yr'] == replicas_time[0]]
    time_hrs_sens = df_time['time_hrs_original']
    # ---------------------------------
    replicas = df['initial_corrosion_mm_yr'].unique()
    df = df.loc[df['initial_corrosion_mm_yr'] == replicas[0]]
    return df, time_hrs_sens


# ----------------------------------------------------------------------------------------------------------------------

def compare_models_plot(df):
    _root = 'regression/gridSearchModels'
    if not os.path.exists(_root):
        os.makedirs(_root)
    # ---------------------------------
    for model_name in ['MLP', 'SVM', 'RF', 'KNN']:
        x_axis_index = [i + 1 for i in np.arange(len(df))]
        _y = [-i for i in df['{}_mean'.format(model_name)]]
        _y_err = df['{}_std'.format(model_name)].tolist()
        bar_width = 0.45
        colors = {'MLP': 'mistyrose', 'SVM': 'cornsilk', 'RF': 'lightgray', 'KNN': 'lightcyan'}
        fig, ax = plt.subplots(1, figsize=(12, 9))
        ax.bar(x_axis_index, _y, width=bar_width, color=colors[model_name], edgecolor='black', zorder=3,
               yerr=_y_err, capsize=5, align='center', ecolor='black', alpha=0.5, label=model_name)
        # ---------------------------------
        letter = {'MLP': 'A', 'RF': 'B', 'KNN': 'C', 'SVM': 'D'}
        plt.text(0.02, 0.98, '{}'.format(letter[model_name]),
                 ha='left', va='top', transform=ax.transAxes,
                 fontdict={'color': 'k', 'weight': 'bold', 'size': 50})
        # ---------------------------------
        ax.grid(axis='y', linewidth=0.35, zorder=0)
        ax.set_xticks(x_axis_index)
        ax.set_xticklabels(x_axis_index, fontsize=20, rotation=45)
        ax.set_xlabel('Grid serach combination', fontsize=30)
        y_axis_max = {'MLP': [0.7, 0.1], 'SVM': [0.6, 0.1], 'RF': [0.3, 0.05], 'KNN': [0.35, 0.05]}
        y_axis_index = np.arange(0, y_axis_max[model_name][0], y_axis_max[model_name][1])
        ax.set_yticks(y_axis_index)
        ax.set_yticklabels(['{:.2f}'.format(i) for i in y_axis_index], fontsize=20)
        ax.set_ylabel('MSE', fontsize=30)
        plt.legend(loc='upper right', fontsize=20, fancybox=True, shadow=True)
        plt.tight_layout()
        plt.savefig('{}/{}.png'.format(_root, model_name))
        plt.close()


def compare_models_box_plot(df):
    _root = 'regression/gridSearchModels'
    if not os.path.exists(_root):
        os.makedirs(_root)
    # ---------------------------------
    cv, replicas = param['cv'], param['replicas']
    # ---------------------------------
    x_axis_labels = [name for name in df['name']]
    df = df.drop(['name', 'mean', 'std', 'model'], axis=1)
    df = df.transform(lambda x: -x)
    _y_matrix = df.values.tolist()
    fig, ax = plt.subplots(1, figsize=(12, 9))
    plt.boxplot(_y_matrix, labels=x_axis_labels, sym='',
                medianprops=dict(color='lightgrey', linewidth=1.0),
                meanprops=dict(linestyle='-', color='black', linewidth=1.5), meanline=True, showmeans=True)
    # ---------------------------------
    info = '{}-fold cross validation analysis \n{} replications per algorithm'.format(cv, replicas)
    plt.text(0.03, 0.96, info,
             ha='left', va='top', transform=ax.transAxes,
             fontdict={'color': 'k', 'size': 23},
             bbox={'boxstyle': 'round', 'fc': 'snow', 'ec': 'gray', 'pad': 0.5})
    # ---------------------------------
    ax.grid(axis='y', linewidth=0.35, zorder=0)
    x_axis_index = [i + 1 for i in np.arange(len(x_axis_labels))]
    ax.set_xticks(x_axis_index)
    ax.set_xticklabels(x_axis_labels, fontsize=30)
    y_axis_index = np.arange(0, 0.12, 0.02)
    ax.set_yticks(y_axis_index)
    ax.set_yticklabels(['{:.2f}'.format(i) for i in y_axis_index], fontsize=20)
    ax.set_ylabel('Mean Squared Error (MSE)', fontsize=28)
    # plt.tight_layout()
    plt.savefig('{}/comparison.png'.format(_root))
    plt.close()


def correlation_plot(df):
    _root = 'regression/bestModelPerformance'
    if not os.path.exists(_root):
        os.makedirs(_root)
    # ---------------------------------
    corr = df.corr()
    plt.subplots(figsize=(12, 12))
    sns.heatmap(corr, vmin=-1, vmax=1, center=0, cmap='coolwarm', square=True)
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    plt.tight_layout()
    plt.savefig('{}/corrMatrix.png'.format(_root))
    plt.close()
    excel_output(pd.DataFrame(corr), _root, file_name='correlation', csv=False)


def importance_plot(df, estimator, _x, _y):
    _root = 'regression/bestModelPerformance'
    if not os.path.exists(_root):
        os.makedirs(_root)
    # ---------------------------------
    names = df.columns
    best_reg[0][1].fit(_x, _y)
    imp = best_reg[0][1].feature_importances_
    indices = np.argsort(imp)
    fig, ax = plt.subplots(1, figsize=(12, 9))
    plt.barh(range(len(indices)), imp[indices], color='black', align='center')
    x_axis_index = np.arange(0, 0.6, 0.1)
    ax.set_xticks(x_axis_index)
    ax.set_xticklabels(x_axis_index, fontsize=20)
    ax.set_xticklabels(['{:.2f}'.format(i) for i in x_axis_index], fontsize=20)
    ax.set_xlabel('Relative Importance', fontsize=30)
    plt.yticks(range(len(indices)), [names[i] for i in indices], fontsize=14)
    plt.tight_layout()
    plt.savefig('{}/featuresImp.png'.format(_root))
    plt.close()
    excel_output(pd.DataFrame(imp), _root, file_name='rf_feature_imp', csv=False)
    # ---------------------------------
    estimator.fit(_x, _y)
    permute_imp_results = permutation_importance(estimator, _x, _y, scoring='neg_mean_squared_error')
    permute_imp = permute_imp_results.importances_mean
    excel_output(pd.DataFrame(permute_imp), _root, file_name='permutation_imp', csv=False)
    return permute_imp_results


def parity_plot(_y_test, _y_pred, _scores):
    _root = 'regression/bestModelPerformance'
    if not os.path.exists(_root):
        os.makedirs(_root)
    # ---------------------------------
    info = '{} = {:.3f} +/- {:.3f}\n{} = {:.3f} +/- {:.3f}\n{} = {:.3f} +/- {:.3f}\n{} = {:.3f} +/- {:.3f}'. \
        format(_scores[0][0], _scores[0][1], _scores[0][2],
               _scores[1][0], _scores[1][1], _scores[1][2],
               _scores[2][0], _scores[2][1], _scores[2][2],
               _scores[3][0], _scores[3][1], _scores[3][2])
    # ---------------------------------
    fig, ax = plt.subplots(1, figsize=(9, 9))
    _mean, _std = dataHTC[target].mean(), dataHTC[target].std()
    _y_test = [i * _std + _mean for i in _y_test]
    _y_pred = [i * _std + _mean for i in _y_pred]
    plt.scatter(_y_pred, _y_test, c='black', label='Testing set')
    a, b = min(min(_y_test), min(_y_pred)), max(max(_y_test), max(_y_pred))
    plt.plot([a, b], [a, b], '-', c='lightgray', linewidth=7.0, label='y = x')
    # ---------------------------------
    plt.text(0.05, 0.97, info,
             ha='left', va='top', transform=ax.transAxes,
             fontdict={'color': 'k', 'size': 21},
             bbox={'boxstyle': 'round', 'fc': 'snow', 'ec': 'gray', 'pad': 0.5})
    # ---------------------------------
    x_axis_index = np.linspace(40, 80, num=5)
    ax.set_xticks(x_axis_index)
    ax.set_xlim(40, 80)
    ax.set_xticklabels(x_axis_index, fontsize=20)
    ax.xaxis.set_major_formatter(FormatStrFormatter('%.0f'))
    # ---------------------------------
    y_axis_index = np.linspace(40, 80, num=5)
    ax.set_yticks(y_axis_index)
    ax.set_ylim(40, 80)
    ax.set_yticklabels(y_axis_index, fontsize=20)
    ax.yaxis.set_major_formatter(FormatStrFormatter('%.0f'))
    # ---------------------------------
    ax.set_xlabel('elemental {} [%] - predicted'.format(target), fontsize=32)
    ax.set_ylabel('elemental {} [%] - true'.format(target), fontsize=32)
    plt.legend(loc='upper right', fontsize=23, fancybox=True, shadow=True)
    plt.tight_layout()
    plt.savefig('{}/parityPlot.png'.format(_root))
    plt.close()
    # ---------------------------------
    df = pd.DataFrame(columns=['True_value', 'Predicted_value'])
    df['True_value'] = _y_test
    df['Predicted_value'] = _y_pred
    excel_output(df, _root, file_name='parityPlotData', csv=False)


def prediction_static_plot(df, estimator):
    exp_num = 0
    df1 = df.copy(deep=True)
    df1 = df1.loc[df1['type_static'] == 1]
    _samples = dataHTC['sample'].unique()
    color = {'cellulose': 'blue', 'straw': 'red', 'polar': 'green',
             'method 1': 'grey', 'method 2': 'grey', 'method 3': 'grey'}
    for _sample in _samples:
        df2 = df1.loc[df1['sample_{}'.format(_sample)] == 1]
        temperatures = df2['temperature'].unique()
        n = 1
        for temp in temperatures:
            fig, ax = plt.subplots(1, figsize=(9, 9))
            _mean, _std = dataHTC[target].mean(), dataHTC[target].std()
            _mean_time, _std_time = dataHTC['time'].mean(), dataHTC['time'].std()
            _mean_temp, _std_temp = dataHTC['temperature'].mean(), dataHTC['temperature'].std()
            df3 = df2.loc[df2['temperature'] == temp]
            _X = [i * _std_time + _mean_time for i in df3['time']]
            _y = [i * _std + _mean for i in df3[target]]
            plt.plot(_X, _y, lw=2.0, marker='o', ms=8, color=color[_sample], label='Experiment', zorder=7)
            # ---------------------------------
            _training_original = pd.concat([df, df3, df3]).drop_duplicates(keep=False)
            _testing_original = df3
            n_init = 0 if sample_in else 3
            mse_summary.loc[exp_num, 'exp_name'] = 'static_{}_{}'.format(_sample, n)
            n_replicas = param['replicas']
            _approach_count = 0
            for _approach in ['method 1', 'method 2', 'method 3']:
                # kinetics approach
                if _approach == 'method 1':

                    _training = select_features(_training_original)
                    _testing = select_features(_testing_original)
                    _X_train, _y_train = split_xy(_training, False)
                    _X_test, _y_test = split_xy(_testing, False)
                    _y_pred = []
                    for replica in range(n_replicas):
                        estimator[0][1].fit(_X_train, _y_train)
                        _y_pred_temp = estimator[0][1].predict(_X_test)
                        _y_pred = _y_pred_temp if replica == 0 else [p + q for p, q in zip(_y_pred, _y_pred_temp)]
                    _y_pred = [p / n_replicas for p in _y_pred]
                    mse = mean_squared_error(_y_test, _y_pred)
                    _y_pred = [i * _std + _mean for i in _y_pred]
                    plt.plot(_X, _y_pred, lw=2.0, ls='--', marker='D', ms=8, color=color[_approach],
                             label='Prediction ({})'.format(_approach))
                # interpolation approach
                elif _approach == 'method 2':
                    _testing = _testing_original.iloc[:, n_init:].reset_index(drop=True)
                    _testing = interpolate_features(_testing)
                    _training = pd.DataFrame(columns=_testing.columns)
                    for s in _samples:
                        df4 = _training_original.loc[_training_original['sample_{}'.format(s)] == 1]
                        for p in ['static', 'dynamic']:
                            df5 = df4.loc[df4['type_{}'.format(p)] == 1]
                            if p == 'static':
                                for t in df5['temperature'].unique():
                                    df6 = df5.loc[df5['temperature'] == t]
                                    df7 = df6.iloc[:, n_init:].reset_index(drop=True)
                                    df7 = interpolate_features(df7)
                                    _training = pd.concat([_training, df7])
                            else:
                                df6 = df5.iloc[:, n_init:].reset_index(drop=True)
                                df6 = interpolate_features(df6)
                                _training = pd.concat([_training, df6])
                    _training = _training.reset_index(drop=True)
                    _X_train, _y_train = split_xy(_training, True)
                    _X_test, _y_test = split_xy(_testing, False)
                    _y_pred = []
                    for replica in range(n_replicas):
                        estimator[1][1].fit(_X_train, _y_train)
                        _y_pred_temp = estimator[1][1].predict(_X_test)
                        _y_pred = _y_pred_temp if replica == 0 else [p + q for p, q in zip(_y_pred, _y_pred_temp)]
                    _y_pred = [p / n_replicas for p in _y_pred]
                    mse = mean_squared_error(_y_test, _y_pred)
                    _y_pred = [i * _std + _mean for i in _y_pred]
                    plt.plot(_X, _y_pred, lw=2.0, ls='-.', marker='s', ms=8, color=color[_approach],
                             label='Prediction ({})'.format(_approach))
                # recursive approach
                else:
                    _y_pred = []
                    _testing = _testing_original.iloc[:, n_init:].reset_index(drop=True)
                    _testing = recursive_features(_testing)
                    _training = pd.DataFrame(columns=_testing.columns)
                    for s in _samples:
                        df4 = _training_original.loc[_training_original['sample_{}'.format(s)] == 1]
                        for p in ['static', 'dynamic']:
                            df5 = df4.loc[df4['type_{}'.format(p)] == 1]
                            if p == 'static':
                                for t in df5['temperature'].unique():
                                    df6 = df5.loc[df5['temperature'] == t]
                                    df7 = df6.iloc[:, n_init:].reset_index(drop=True)
                                    df7 = recursive_features(df7)
                                    _training = pd.concat([_training, df7])
                            else:
                                df6 = df5.iloc[:, n_init:].reset_index(drop=True)
                                df6 = recursive_features(df6)
                                _training = pd.concat([_training, df6])
                    _training = _training.reset_index(drop=True)
                    _X_train, _y_train = split_xy(_training, True)
                    x_hist = [i for i in _testing.iloc[0, -4:-1]]
                    _y_test = _testing.iloc[:, -1]
                    for k in range(len(_testing)):
                        _X_test = _testing.iloc[k:k+1, :-1]
                        _X_test.iloc[0, -3:] = x_hist
                        _y_hat = 0
                        for replica in range(n_replicas):
                            estimator[2][1].fit(_X_train, _y_train)
                            _y_hat_temp = estimator[2][1].predict(_X_test)
                            _y_hat += _y_hat_temp[0]
                        _y_hat = _y_hat / n_replicas
                        x_hist.append(_y_hat)
                        del x_hist[0]
                        _y_pred.append(_y_hat)
                    mse = mean_squared_error(_y_test, _y_pred)
                    _y_pred = [i * _std + _mean for i in _y_pred]
                    plt.plot(_X[3:], _y_pred, lw=2.0, ls=':', marker='x', mew=3, ms=8, color=color[_approach],
                             label='Prediction ({})'.format(_approach))
                mse_summary.loc[exp_num, '{}_{}'.format(estimator[_approach_count][0], _approach)] = mse
                _approach_count += 1
            exp_num += 1
            # ---------------------------------
            temp_real = temp * _std_temp + _mean_temp
            # plt.text(0.03, 0.97, '{}  ({:.0f} $^o$C)'.format(_sample, temp_real),
            #          ha='left', va='top', transform=ax.transAxes,
            #          fontdict={'color': 'k', 'weight': 'bold', 'size': 24})
            # ---------------------------------
            plt.grid(axis='both', linewidth=0.5)
            x_axis_index = np.linspace(0, 500, num=6)
            ax.set_xticks(x_axis_index)
            ax.set_xlim(0, 500)
            ax.set_xticklabels(x_axis_index, fontsize=24)
            ax.xaxis.set_major_formatter(FormatStrFormatter('%.0f'))
            ax.set_xlabel('Time (min)', fontsize=35)
            # ---------------------------------
            y_axis_index = np.linspace(30, 80, num=6)
            ax.set_yticks(y_axis_index)
            ax.set_ylim(30, 80)
            ax.set_yticklabels(y_axis_index, fontsize=24)
            ax.yaxis.set_major_formatter(FormatStrFormatter('%.0f'))
            ax.set_ylabel('elemental {} [%]'.format(target), fontsize=35)
            # ---------------------------------
            plt.legend(loc='best', fontsize=20, ncol=2, fancybox=True, shadow=True)
            # ax.legend(loc='lower right', bbox_to_anchor=(1.0, 1.0), ncol=2, fancybox=True, shadow=True, fontsize=20)
            plt.tight_layout()
            if not param['grid_search']:
                _root = 'regression/predictions'
                if not os.path.exists(_root):
                    os.makedirs(_root)
                plt.savefig('{}/{}_{:.0f}_{}.png'.format(_root, _sample, temp_real, target))
            plt.close()
            n += 1
    return mse_summary


def prediction_dynamic_plot(df, estimator):
    exp_num = 9
    df1 = df.copy(deep=True)
    df1 = df1.loc[df1['type_dynamic'] == 1]
    _samples = dataHTC['sample'].unique()
    color = {'cellulose': 'blue', 'straw': 'red', 'polar': 'green',
             'method 1': 'grey', 'method 2': 'grey', 'method 3': 'grey'}
    for _sample in _samples:
        df2 = df1.loc[df1['sample_{}'.format(_sample)] == 1]
        fig, ax = plt.subplots(1, figsize=(9, 9))
        _mean, _std = dataHTC[target].mean(), dataHTC[target].std()
        _mean_temp, _std_temp = dataHTC['temperature'].mean(), dataHTC['temperature'].std()
        _X = [i * _std_temp + _mean_temp for i in df2['temperature']]
        _y = [i * _std + _mean for i in df2[target]]
        plt.plot(_X, _y, lw=2.0, marker='o', ms=8, color=color[_sample], label='Experiment', zorder=7)
        # ---------------------------------
        _testing_original = df2
        _training_original = pd.concat([df, df2, df2]).drop_duplicates(keep=False)
        n_init = 0 if sample_in else 3
        mse_summary.loc[exp_num, 'exp_name'] = 'dynamic_{}'.format(_sample)
        n_replicas = param['replicas']
        _approach_count = 0
        for _approach in ['method 1', 'method 2', 'method 3']:
            # kinetics approach
            if _approach == 'method 1':
                _training = select_features(_training_original)
                _testing = select_features(_testing_original)
                _X_train, _y_train = split_xy(_training, True)
                _X_test, _y_test = split_xy(_testing, False)
                _y_pred = []
                for replica in range(n_replicas):
                    estimator[0][1].fit(_X_train, _y_train)
                    _y_pred_temp = estimator[0][1].predict(_X_test)
                    _y_pred = _y_pred_temp if replica == 0 else [p + q for p, q in zip(_y_pred, _y_pred_temp)]
                _y_pred = [p / n_replicas for p in _y_pred]
                mse = mean_squared_error(_y_test, _y_pred)
                _y_pred = [i * _std + _mean for i in _y_pred]
                plt.plot(_X, _y_pred, lw=2.0, ls='--', marker='D', ms=8, color=color[_approach],
                         label='Prediction ({})'.format(_approach))
            # interpolation approach
            elif _approach == 'method 2':
                _testing = _testing_original.iloc[:, n_init:].reset_index(drop=True)
                _testing = interpolate_features(_testing)
                _training = pd.DataFrame(columns=_testing.columns)
                for s in _samples:
                    df4 = _training_original.loc[_training_original['sample_{}'.format(s)] == 1]
                    for p in ['static', 'dynamic']:
                        df5 = df4.loc[df4['type_{}'.format(p)] == 1]
                        if p == 'static':
                            for t in df5['temperature'].unique():
                                df6 = df5.loc[df5['temperature'] == t]
                                df7 = df6.iloc[:, n_init:].reset_index(drop=True)
                                df7 = interpolate_features(df7)
                                _training = pd.concat([_training, df7])
                        else:
                            df6 = df5.iloc[:, n_init:].reset_index(drop=True)
                            if len(df6) > 0:
                                df6 = interpolate_features(df6)
                                _training = pd.concat([_training, df6])
                _training = _training.reset_index(drop=True)
                _X_train, _y_train = split_xy(_training, True)
                _X_test, _y_test = split_xy(_testing, False)
                _y_pred = []
                for replica in range(n_replicas):
                    estimator[1][1].fit(_X_train, _y_train)
                    _y_pred_temp = estimator[1][1].predict(_X_test)
                    _y_pred = _y_pred_temp if replica == 0 else [p + q for p, q in zip(_y_pred, _y_pred_temp)]
                _y_pred = [p / n_replicas for p in _y_pred]
                mse = mean_squared_error(_y_test, _y_pred)
                _y_pred = [i * _std + _mean for i in _y_pred]
                plt.plot(_X, _y_pred, lw=2.0, ls='-.', marker='s', ms=8, color=color[_approach],
                         label='Prediction ({})'.format(_approach))
            # recursive approach
            else:
                _y_pred = []
                _testing = _testing_original.iloc[:, n_init:].reset_index(drop=True)
                _testing = recursive_features(_testing)
                _training = pd.DataFrame(columns=_testing.columns)
                for s in _samples:
                    df4 = _training_original.loc[_training_original['sample_{}'.format(s)] == 1]
                    for p in ['static', 'dynamic']:
                        df5 = df4.loc[df4['type_{}'.format(p)] == 1]
                        if p == 'static':
                            for t in df5['temperature'].unique():
                                df6 = df5.loc[df5['temperature'] == t]
                                df7 = df6.iloc[:, n_init:].reset_index(drop=True)
                                df7 = recursive_features(df7)
                                _training = pd.concat([_training, df7])
                        else:
                            df6 = df5.iloc[:, n_init:].reset_index(drop=True)
                            if len(df6) > 0:
                                df6 = recursive_features(df6)
                                _training = pd.concat([_training, df6])
                _training = _training.reset_index(drop=True)
                _X_train, _y_train = split_xy(_training, True)
                x_hist = [i for i in _testing.iloc[0, -4:-1]]
                _y_test = _testing.iloc[:, -1]
                for k in range(len(_testing)):
                    _X_test = _testing.iloc[k:k + 1, :-1]
                    _X_test.iloc[0, -3:] = x_hist
                    _y_hat = 0
                    for replica in range(n_replicas):
                        estimator[2][1].fit(_X_train, _y_train)
                        _y_hat_temp = estimator[2][1].predict(_X_test)
                        _y_hat = _y_hat + _y_hat_temp[0]
                    _y_hat = _y_hat / n_replicas
                    x_hist.append(_y_hat)
                    del x_hist[0]
                    _y_pred.append(_y_hat)
                mse = mean_squared_error(_y_test, _y_pred)
                _y_pred = [i * _std + _mean for i in _y_pred]
                plt.plot(_X[3:], _y_pred, lw=2.0, ls=':', marker='x', mew=3, ms=8, color=color[_approach],
                         label='Prediction ({})'.format(_approach))
            mse_summary.loc[exp_num, '{}_{}'.format(estimator[_approach_count][0], _approach)] = mse
            _approach_count += 1
        exp_num += 1
        # ---------------------------------
        # plt.text(0.03, 0.97, '{}'.format(_sample),
        #          ha='left', va='top', transform=ax.transAxes,
        #          fontdict={'color': 'k', 'weight': 'bold', 'size': 24})
        # ---------------------------------
        plt.grid(axis='both', linewidth=0.5)
        x_axis_index = np.linspace(160, 260, num=6)
        ax.set_xticks(x_axis_index)
        ax.set_xlim(160, 260)
        ax.set_xticklabels(x_axis_index, fontsize=24)
        ax.xaxis.set_major_formatter(FormatStrFormatter('%.0f'))
        ax.set_xlabel('Temperature ($^o$C)', fontsize=35)
        # ---------------------------------
        y_axis_index = np.linspace(30, 80, num=6)
        ax.set_yticks(y_axis_index)
        ax.set_ylim(30, 80)
        ax.set_yticklabels(y_axis_index, fontsize=24)
        ax.yaxis.set_major_formatter(FormatStrFormatter('%.0f'))
        ax.set_ylabel('elemental {} [%]'.format(target), fontsize=35)
        # ---------------------------------
        plt.legend(loc='upper right', fontsize=23, ncol=1, fancybox=True, shadow=True)
        # ax.legend(loc='lower right', bbox_to_anchor=(1.0, 1.0), ncol=2, fancybox=True, shadow=True, fontsize=20)
        plt.tight_layout()
        if not param['grid_search']:
            _root = 'regression/predictions'
            if not os.path.exists(_root):
                os.makedirs(_root)
            plt.savefig('{}/{}_{}.png'.format(_root, _sample, target))
        plt.close()
    return mse_summary


# --------------------------------------------------------------------------------------------------------------------
# BEGIN
# --------------------------------------------------------------------------------------------------------------------

# reading data
dataHTC = read_data('dataHtcCleaned.xlsx')

# data summary (one-time output)
if param['summary']:
    summary_data(dataHTC)
    correlation_plot(dataHTC)

# --------------------------------------------------------------------------------------------------------------------
# REGRESSION PROBLEM
# --------------------------------------------------------------------------------------------------------------------

# pre-processing data
htc = encode_data(dataHTC)

# grid-search to find the best model of each algorithm (one-time output)
mse_summary = pd.DataFrame(index=range(12))
if param['grid_search']:
    root = 'regression/gridSearchModels'
    if not os.path.exists(root):
        os.makedirs(root)
    # ---------------------------------
    for algorithm in ['MLP', 'SVM', 'RF', 'KNN']:
        print(algorithm)
        mse_summary = pd.DataFrame(index=range(12))
        models_regs = grid_search(algorithm)
        for ml_model in range(len(models_regs)):
            print('model {}'.format(ml_model))
            regressor = models_regs[ml_model]
            prediction_static_plot(htc, regressor)
            prediction_dynamic_plot(htc, regressor)
        excel_output(mse_summary, root, algorithm, True)
else:
    # models_reg = [('MLP', MLPRegressor(hidden_layer_sizes=(8,), max_iter=10000)),
    #               ('SVM', SVR(C=100, gamma=0.0001)),
    #               ('RF', RandomForestRegressor(max_features=0.6, n_estimators=500)),
    #               ('KNN', KNeighborsRegressor(n_neighbors=3, weights='distance'))]
    best_reg = [('RF1', RandomForestRegressor(max_features=1.0, n_estimators=100)),
                ('RF2', RandomForestRegressor(max_features=0.8, n_estimators=10)),
                ('SVM3', SVR(C=10, gamma=0.001))]

    # features importance
    X, y = split_xy(select_features(htc), True)
    importance = importance_plot(select_features(htc), best_reg[0][1], X, y)

    # prediction when the experiment is entirely out from the training set
    prediction_static_plot(htc, best_reg)
    prediction_dynamic_plot(htc, best_reg)
    excel_output(mse_summary, 'regression/predictions', 'mse_summary', True)

# ----------------------------------------------------------------------------------------------------------------------
# The End
# ----------------------------------------------------------------------------------------------------------------------

print('DONE!')

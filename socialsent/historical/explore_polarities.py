from socialsent import constants
from socialsent import util
from collections import defaultdict
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from operator import itemgetter
import statsmodels.api as sm


def build_timeseries(raw=False, suffix="-test", years=constants.YEARS):
    timeseries = defaultdict(list)
    for year in years:
        polarities = util.load_pickle(constants.POLARITIES + year + suffix + '.pkl')
        for i, (w, p) in enumerate(sorted(polarities.items(), key=itemgetter(1))):
            if not raw:
                polarities[w] = i / float(len(polarities))
            else:
                polarities[w] = p

        for w, p in polarities.iteritems():
            timeseries[w].append(p)
    for w in timeseries.keys():
        if len(timeseries[w]) < 5:
            del timeseries[w]
    return timeseries

def build_boot_timeseries(suffix="-test", years=constants.YEARS):
    mean_timeseries = defaultdict(list)
    stderr_timeseries = defaultdict(list)
    for year in years:
        polarities_list = util.load_pickle(constants.POLARITIES + year + suffix + '.pkl')
        for w in polarities_list[0]:
            n = float(len(polarities_list))
            mean_timeseries[w].append(np.mean([polarities[w] for polarities in polarities_list]))
            stderr_timeseries[w].append(np.std([polarities[w] for polarities in polarities_list])/n)
    for w in mean_timeseries.keys():
        if len(mean_timeseries[w]) < 5:
            del mean_timeseries[w]
            del stderr_timeseries[w]
    return mean_timeseries, stderr_timeseries

def get_boot_meanseries(suffix="-test", years=constants.YEARS):
    mean_timeseries = []
    stderr_timeseries = []
    for year in years:
        polarities_list = util.load_pickle(constants.POLARITIES + year + suffix + '.pkl')
        year_means = []
        for polarities in polarities_list:
            year_means.append(np.mean(polarities.values()))
        mean_timeseries.append(np.mean(year_means))
        stderr_timeseries.append(np.std(year_means))
    return mean_timeseries, stderr_timeseries


def get_boot_meanseriess(suffix="-test", years=constants.YEARS):
    mean_timeseriess = defaultdict(list)
    for year in years:
        polarities_list = util.load_pickle(constants.POLARITIES + year + suffix + '.pkl')
        for i, polarities in enumerate(polarities_list):
            mean_timeseriess[i].append(np.mean(polarities.values()))
    return mean_timeseriess


def build_boot_ztimeseries(suffix="-test", years=constants.YEARS):
    mean_timeseries = defaultdict(list)
    stderr_timeseries = defaultdict(list)
    for year in years:
        polarities_list = util.load_pickle(constants.POLARITIES + year + suffix + '.pkl')
        means = [np.mean(polarities.values()) for polarities in polarities_list]
        stds = [np.std(polarities.values()) for polarities in polarities_list]
        zscore = lambda i, val : (val - means[i])/stds[i]
        for w in polarities_list[0]:
            mean_timeseries[w].append(np.mean([zscore(i, polarities[w]) for i, polarities in enumerate(polarities_list)]))
            stderr_timeseries[w].append(np.std([zscore(i, polarities[w]) for i, polarities in enumerate(polarities_list)]))
    for w in mean_timeseries.keys():
        if len(mean_timeseries[w]) < 5:
#            #print w + " is not present in all decades"
            del mean_timeseries[w]
            del stderr_timeseries[w]
    return mean_timeseries, stderr_timeseries

def build_boot_zdictseries(suffix="-test", years=constants.YEARS):
    mean_timeseries = defaultdict(lambda : defaultdict(lambda : float('nan')))
    stderr_timeseries = defaultdict(lambda : defaultdict(lambda : float('nan')))
    for year in years:
        polarities_list = util.load_pickle(constants.POLARITIES + year + suffix + '.pkl')
        means = [np.mean(polarities.values()) for polarities in polarities_list]
        stds = [np.std(polarities.values()) for polarities in polarities_list]
        zscore = lambda i, val : (val - means[i])/stds[i]
        for w in polarities_list[0]:
            mean_timeseries[w][year] = (np.mean([zscore(i, polarities[w]) for i, polarities in enumerate(polarities_list)]))
            stderr_timeseries[w][year] = (np.std([zscore(i, polarities[w]) for i, polarities in enumerate(polarities_list)]))
    return mean_timeseries, stderr_timeseries


def build_dictseries(raw=True, suffix="-test",  years=constants.YEARS):
    timeseries = defaultdict(lambda : defaultdict(lambda : float('nan')))
    for year in years:
        polarities = util.load_pickle(constants.POLARITIES + year + suffix + '.pkl')
        for i, (w, p) in enumerate(sorted(polarities.items(), key=itemgetter(1))):
            if not raw:
                polarities[w] = i / float(len(polarities))
            else:
                polarities[w] = p

        for w, p in polarities.iteritems():
            timeseries[w][int(year)] = p
    return timeseries

def zscore(dictseries, years=range(1850, 2000, 10)):
    timeseries = defaultdict(list)
    yearseries = get_yearseries(dictseries, years)
    for year in years:
        year_mean = np.mean(yearseries[year].values())
        year_std = np.std(yearseries[year].values())
        for word in yearseries[year]:
            timeseries[word].append((dictseries[word][year] - year_mean)
                    / year_std)
    return timeseries

def get_yearseries(dictseries, years=range(1850, 2000, 10)):
    yearseries = {}
    for year in years:
        year_series = {word:dictseries[word][year] for word in dictseries if
                not np.isnan(dictseries[word][year])}
        yearseries[year] = year_series
    return yearseries


def slope(ys):
    return LinearRegression().fit(np.matrix(np.arange(len(ys))).T, ys).coef_[0]

def trend_estimate(y):
    X = np.arange(len(y))
    X = sm.add_constant(X)
    mod = sm.OLS(y, X)
    res = mod.fit()
    return res


def plot_timeseries(timeseries, w):
    plt.figure()
    plt.plot(map(int, constants.YEARS), timeseries[w])
    plt.title(w)
    plt.xlabel('date')
    plt.ylabel('polarity')
    plt.show()


def main():
    timeseries = build_timeseries(years=constants.YEARS)
    #plot_timeseries(timeseries, "awesome")

    ws = []
    p_vs_dp = []
    ordering = util.load_pickle(constants.DATA + 'word_ordering.pkl')
    for w, polarities in timeseries.iteritems():
        if ordering.index(w) < 50000:
            ws.append(w)
            p_vs_dp.append((np.mean(polarities[:5]), slope(polarities)))

    xs, ys = zip(*p_vs_dp)
    pred = LinearRegression().fit(np.matrix(xs).T, ys).predict(np.matrix(xs).T)
    print "R2 score", r2_score(ys, pred)

    def onpick(event):
        for i in event.ind:
            w = ws[i]
            print w, p_vs_dp[i]
            plot_timeseries(timeseries, w)
            break
    figure = plt.figure()
    plt.scatter(xs, ys, marker='o', linewidths=0, picker=True, alpha=1)
    plt.xlabel('polarity in 1800')
    plt.ylabel('rate of polarity increase')
    figure.canvas.mpl_connect('pick_event', onpick)
    plt.show()

if __name__ == '__main__':
    main()

import sys
import copy
import json
from operator import add
from collections import defaultdict, Counter
from matplotlib import pyplot as plt
import seaborn as sns
from sklearn import linear_model, model_selection, metrics
import pandas as pd
import numpy as np

import util

#sns.palplot(sns.color_palette("muted"))

#fns = ['sum_mag', 'sum_sq_mag', 'sum_sqrt_mag', 'sum_log_mag', 'KT_distance', 'true_min_actions']
fns = ['sum_mag']
#ks = range(20)
ks = [0,1,3,5,7,9]


# compute the moves in all votes
def compute_moves(votes):
    #print('computing moves')
    aug_votes = list()
    for v in votes:
        aug_v = compute_moves_in_vote(v)
        aug_votes.append(aug_v)
    #print('done computing moves')
    return aug_votes


# compute the moves in one vote
# Comments:
# iterates over the data field which contain an entry for each move.
# each move is encoded as a tuple start_rank, end_rank
# start_rank is the ranking immediately before the move
# end_rank is the ranking immediately after the move
# adds the following entries to data entry:
# start_pos: start position of item moved
# end_pos: end position of item moved
# move_dist: # positions item was moved; +ve if moved up, -ve if moved down
# move_time: # time taken for the move
# Returns a vote with the augmented data field which has the above move information
def compute_moves_in_vote(v):
    if len(v['data']) == 0:
        v['data'] = copy.deepcopy(v['user_data'])
    aug_data = list()
    for d in v['data']:
        aug_d = copy.deepcopy(d)
        aug_d['start_pos'], aug_d['end_pos'], aug_d['move_dist'], aug_d['move_time'] = compute_move(d)
        aug_d['move_dir'] = int(np.sign(aug_d['move_dist']))
        aug_d['move_mag'] = int(np.abs(aug_d['move_dist']))
        aug_d['move_sq_mag'] = int(np.power(aug_d['move_mag'], 2))
        aug_d['move_sqrt_mag'] = float(np.sqrt(aug_d['move_mag']))
        #aug_d['move_log_mag'] = float(np.log(aug_d['move_mag']))
        #if not np.isfinite(aug_d['move_log_mag']):
        #    aug_d['move_log_mag'] = 0
        aug_data.append(aug_d)
    aug_vote = copy.deepcopy(v)
    aug_vote['data'] = copy.deepcopy(aug_data)
    try:
        aug_d = aug_vote['data'][0]
    except:
        print(aug_vote['data'])
        print(v)
        aug_vote['data'][0]
    #aug_d['op_time'] = aug_d['move_time']
    #aug_d['op_time'] = np.nan
    aug_d['op_time'] = aug_d['move_time'] + aug_vote['time1']
    aug_data = [aug_d]
    for i in range(1,len(aug_vote['data']),1):
        d = aug_vote['data'][i]
        dprev = aug_vote['data'][i-1]
        aug_d = copy.deepcopy(d)
        aug_d['op_time'] = d['move_time'] + (d['time'][0] - dprev['time'][1])
        aug_data.append(aug_d)
    aug_vote['data'] = copy.deepcopy(aug_data)
    aug_vote['num_moves'] = len(aug_vote['data'])
    #for fn in ['_', '_sq_', '_sqrt_', '_log_']:
    for fn in ['_', '_sq_', '_sqrt_']:
        aug_vote['sum'+fn+'mag'] = float(np.sum([d['move'+fn+'mag'] for d in aug_vote['data']]))
    return aug_vote


# compute the move for one drag and drop
# Comments:
# Input is an entry from the data field of a vote. Must include the start_rank and end_rank
# start_rank is the ranking immediately before the move
# end_rank is the ranking immediately after the move
# Output is the move: start position, end position of the item moved, and time taken.
def compute_move(d):
    moved_item = d['item']
    start_pos = d['start_rank'].index(moved_item)
    end_pos = d['end_rank'].index(moved_item)
    move_dist = start_pos - end_pos
    move_time = d['time'][1] - d['time'][0]
    return start_pos, end_pos, move_dist, move_time


def compute_f_moves(vote, k):
    initial_ranking = vote['initial_ranking']
    sorted_ranking = sorted(initial_ranking)
    selections = 0
    insertions = 0
    data = list()
    current_ranking = initial_ranking
    while not current_ranking == sorted_ranking:
        data_item = {'time': [0,0], 'start_rank': copy.deepcopy(current_ranking)}
        previous_ranking = current_ranking
        moved_item = None
        if selections < k:
            current_ranking, moved_item = util.doSelectionMove(current_ranking)
            selections += 1
        else:
            current_ranking, moved_item = util.doInsertionMove(current_ranking)
            insertions += 1
        assert(current_ranking != previous_ranking)
        """
        moved_item = None
        for item in initial_ranking:
            woitem_previous = [i for i in previous_ranking if i != item]
            woitem_current = [i for i in current_ranking if i != item]
            if woitem_previous == woitem_current:
                moved_item = item
                break
        """
        assert(moved_item != None)
        data_item['item'] = moved_item
        data_item['end_rank'] = copy.deepcopy(current_ranking)
        data.append(data_item)
    aug_vote = copy.deepcopy(vote)
    aug_vote['data'] = data
    return aug_vote


def stat_votes(votes, field='move_time'):
    moves = [d for v in votes for d in v['data']]
    df = pd.DataFrame.from_dict(moves)
    df_move_time = df[['move_dist', field]]
    df_move_time.plot.scatter('move_dist', field)
    df_move_time.boxplot(column=[field], by='move_dist', showfliers=False)
    plt.savefig(util.loc+'time_v_dist_boxplot.png')
    df_move_time.hist(column='move_dist',bins=range(-10,10,1))
    plt.savefig(util.loc+'dist_hist.png')
    plt.xticks(range(-10,10,1))
    grouped_by_dist = df_move_time.groupby('move_dist', as_index=False)
    averaged_time = grouped_by_dist.aggregate(np.mean)
    #print(list(averaged_time))
    averaged_time.plot(x='move_dist', y=field)
    plt.savefig(util.loc+'time_v_dist_plot.png')


def report(all_scores, best, all_data, name, field, loc):
    plt.boxplot(all_data)
    plt.title('box_{}_{}'.format(name, field))
    plt.savefig(loc+'box_{}_{}.png'.format(name, field))
    plt.close('all')
    with open(loc+'report_{}_{}.txt'.format(name, field), 'w') as fo:
        bestkeysrep = best.keys()
        fo.write(','.join(bestkeysrep)+'\n')
        fo.write(','.join([str(best[k]) for k in bestkeysrep])+'\n')
        l = range(len(best.keys()))
        plt.bar(l, best.values(), align='center')
        plt.xticks(l, best.keys(), rotation='vertical')
        plt.tight_layout()
        plt.savefig(loc+'{}_{}_bestfts.png'.format(name, field))
        fo.write('key, avg, med\n')
        fo.write('{}, {}, {}\n'.format(field, np.mean(all_data), np.median(all_data)))
        for fn in list(fts.keys())+['best']:
            scoresdict = dict()
            alldict = dict()
            for i in range(len(all_scores[fn])):
                scoresdict[i] = all_scores[fn][i]
                alldict[i] = all_data[i]
            plotCVMSE(scoresdict, alldict, name, field, fn, loc)
            scores = all_scores[fn]
            fo.write('{}, {}, {}\n'.format(fn, np.mean(scores), np.median(scores)))
            plt.boxplot(scores)
            plt.title('box_{}_{}_{}_cv_scores'.format(name, field, fn))
            plt.savefig(loc+'box_{}_{}_{}_cv_scores.png'.format(name, field, fn))
            plt.close('all')
            plt.boxplot(scores, showfliers=False)
            plt.title('box_{}_{}_{}_cv_scores_nofliers'.format(name, field, fn))
            plt.savefig(loc+'box_{}_{}_{}_cv_scores_nofliers.png'.format(name, field, fn))
            plt.close('all')
    return None


def plotCVMSE(dic_cv, dic_avg_time, name, field, fn, loc):
    sorted_uid = sorted(dic_cv, key=dic_cv.get)
    sorted_cv = sorted(dic_cv.values())
    sorted_avg_time = [dic_avg_time[uid] for uid in sorted_uid]
    assert sorted_cv == [dic_cv[uid] for uid in sorted_uid]
    list_avg_cv = []
    list_avg_avg_time = []
    for i in range(len(sorted_cv)):
        sublist_cv = sorted_cv[:i+1]
        sublist_time = sorted_avg_time[:i+1]
        list_avg_cv.append(np.mean(sublist_cv))
        list_avg_avg_time.append(sum(sublist_time)/len(sublist_time))

    xs = np.array([i+1 for i in range(len(list_avg_cv))])/float(len(list_avg_cv))

    fig, ax1 = plt.subplots()
    ax1.set_xlabel('# of users included')
    ax1.set_ylabel('Time^2 (sec^2)')
    lns1 = ax1.plot(xs, list_avg_cv, '-', label='Avg(CV MSE)', color='r')

    ax2 = ax1.twinx()

    ax2.set_ylabel('Time (sec)')
    lns2 = ax2.plot(xs, list_avg_avg_time, '-', label='Avg(Time)', color='b')
    lns = lns1+lns2
    labs = [l.get_label() for l in lns]
    ax2.legend(lns, labs)
    ax1.grid()
    ax2.grid()
    fig.tight_layout()  # otherwise the right y-label is slightly clipped
    plt.title('Performance of {} as regressor'.format(fn))
    plt.savefig(loc+'cvmse_{}_{}_{}.png'.format(name, field, fn))
    plt.show()


def count_ops(votes, loc = util.loc + util.sfx):
    ops = 0
    ins = 0
    sel = 0
    either = 0
    time_per_at_n = defaultdict(list)
    time_at_n = defaultdict(list)
    kt_at_n = defaultdict(list)
    per_sel_ins = list()
    either_at_n = defaultdict(list)
    moves_at_n = defaultdict(list)
    ops_at_n = dict()
    per_sel_ins_users = list()
    u2v = util.users_votes(votes)
    for u in u2v:
        n = 0
        time_in_user = list()
        either_in_user = 0
        moves_user = 0
        for v in u2v[u]:
            either_in_vote = 0
            num_moves = len(v['user_data'])
            moves_user += num_moves
            time_in_user.append(v['time12'])
            for d in v['user_data']:
                start_rank = d['start_rank']
                end_rank = d['end_rank']
                ins_rank = util.doInsertion(start_rank)
                sel_rank = util.doSelection(start_rank)
                ops += 1
                if ins_rank == end_rank:
                    ins += 1
                if sel_rank == end_rank:
                    sel += 1
                if ins_rank == end_rank or sel_rank == end_rank:
                    either += 1
                    either_in_vote += 1
            either_in_user += either_in_vote
            either_at_n[n].append(either_in_vote)
            moves_at_n[n].append(num_moves)
            kt_at_n[n].append(v['KT_distance'])
            time_at_n[n].append(v['time12'])
            time_per_at_n[n].append(100.*v['time12']/time_in_user[0])
            either_in_vote = either_in_vote/num_moves
            per_sel_ins.append(either_in_vote)
            n += 1
        per_sel_ins_users.append(either_in_user/moves_user)
    for n in either_at_n.keys():
        either_at_n[n] = sum(either_at_n[n])
        ops_at_n[n] = np.mean(moves_at_n[n])
        moves_at_n[n] = sum(moves_at_n[n])
        either_at_n[n] = 100*either_at_n[n]/moves_at_n[n]
        time_at_n[n] = np.mean(time_at_n[n])
        time_per_at_n[n] = np.mean(time_per_at_n[n])
        kt_at_n[n] = np.mean(kt_at_n[n])
    ns = sorted(either_at_n.keys())
    time_at_n = [time_at_n[n] for n in ns]
    time_per_at_n = [time_per_at_n[n] for n in ns]
    either_at_n = [either_at_n[n] for n in ns]
    ops_at_n = [ops_at_n[n] for n in ns]
    kt_at_n = [kt_at_n[n] for n in ns]
    print(time_at_n)
    print(either_at_n)
    plt.bar(ns, either_at_n, color='b')
    plt.xticks(ns)
    plt.xlabel('poll number in order of participation by user')
    plt.ylabel('%age insertion or selection operations')
    plt.savefig(loc+'selins_progress.png')
    plt.show()
    plt.bar(ns, time_at_n, color='b')
    plt.xticks(ns)
    plt.xlabel('poll number in order of participation by user')
    plt.ylabel('average sorting time')
    plt.savefig(loc+'time_progress.png')
    plt.show()
    plt.bar(ns, time_per_at_n, color='b')
    plt.xticks(ns)
    plt.xlabel('poll number in order of participation by user')
    plt.ylabel('average % time of first poll')
    plt.savefig(loc+'time_per_progress.png')
    plt.show()
    plt.bar(ns, ops_at_n, color='b')
    plt.xticks(ns)
    plt.xlabel('poll number in order of participation by user')
    plt.ylabel('average # drag-and-drop operations')
    plt.savefig(loc+'ops_progress.png')
    plt.show()
    plt.bar(ns, kt_at_n, color='b')
    plt.xticks(ns)
    plt.xlabel('poll number in order of participation by user')
    plt.ylabel('average KT distance')
    plt.savefig(loc+'kt_progress.png')
    plt.show()
    bins = 10*(np.arange(11))
    per_sel_ins = 100*np.array(per_sel_ins)
    sns.distplot(per_sel_ins, bins=bins, kde=False, rug=False)
    plt.xticks(10*np.arange(11))
    plt.xlabel('%age of insertion or selection operations in vote')
    plt.ylabel('frequency')
    plt.savefig(loc + 'selins_vote.png')
    plt.show()
    bins = 10 * (np.arange(11))
    per_sel_ins_users = 100 * np.array(per_sel_ins_users)
    sns.distplot(per_sel_ins_users, bins=bins, kde=False, rug=False)
    plt.xticks(10 * np.arange(11))
    plt.xlabel('%age of insertion or selection operations by voter')
    plt.ylabel('frequency')
    plt.savefig(loc + 'selins_user.png')
    plt.show()
    vdata = [{'Ulam_distance': v['true_min_actions'], 'KT_distance': v['KT_distance'], 'actions': len(v['user_data']),
              'selins_ops': v['num_steps_insertion'], 't12': v['time12'], 'mag': v['sum_mag'], 'si_ops': v['si_ops']}
             for v in votes]
    df = pd.DataFrame(vdata)
    df.boxplot(column=['t12'], by='mag', showfliers=False, showmeans=True)
    plt.title('sorting time vs. Total distance moved')
    plt.suptitle('')
    locs = list()
    labels = list()
    l = 0
    for x in sorted(list(set(df['mag']))):
        if x % 5 == 0:
            labels.append(x)
            locs.append(l)
        l += 1
    plt.xticks(locs, labels)
    plt.xlabel('Total distance moved')
    plt.ylabel('sorting time')
    plt.savefig(loc + 'time_dist.png')
    plt.show()
    plt.close('all')
    gp = df.groupby(by='mag')
    means = gp.mean()
    means = means['t12']
    stds = gp.std()
    stds = stds['t12']
    means.plot(yerr=stds, kind='line', capsize=3)
    #plt.title('sorting time vs. Total distance moved')
    plt.suptitle('')
    plt.xlabel('total distance moved')
    plt.ylabel('average sorting time')
    plt.savefig(loc + 'time_dist_ebar.png')
    plt.show()
    plt.close('all')
    df.boxplot(column=['t12'], by='Ulam_distance', showfliers=False, showmeans=True)
    plt.title('Time spent vs. Ulam distance')
    plt.suptitle('')
    plt.xlabel('Total # Positions moved in vote')
    plt.ylabel('Time spent')
    plt.savefig(loc + 'time_ulam.png')
    df.boxplot(column=['t12'], by='KT_distance', showfliers=False, showmeans=True)
    plt.title('Time spent vs. KT distance')
    plt.suptitle('')
    plt.ylabel('Time spent')
    plt.savefig(loc + 'time_kt.png')
    df.boxplot(column=['t12'], by='selins_ops', showfliers=False, showmeans=True)
    plt.title('Time spent vs. # Selection/Insertion operations')
    plt.suptitle('')
    plt.ylabel('Time spent')
    plt.savefig(loc + 'time_selins.png')
    df.boxplot(column=['t12'], by='actions', showfliers=False, showmeans=True)
    plt.title('Time spent vs. # Voter operations')
    plt.suptitle('')
    plt.ylabel('Time spent')
    plt.savefig(loc + 'time_ops.png')
    df.boxplot(column=['actions'], by='Ulam_distance', showfliers=False, showmeans=True)
    plt.title('# Voter operations vs. Ulam distance')
    plt.suptitle('')
    plt.ylabel('Voter operations')
    plt.savefig(loc + 'ops_ulam.png')
    df.boxplot(column=['actions'], by='KT_distance', showfliers=False, showmeans=True)
    plt.title('# Voter operations vs. KT distance')
    plt.suptitle('')
    plt.ylabel('Voter operations')
    plt.tight_layout()
    plt.savefig(loc + 'ops_kt.png')
    df.boxplot(column='KT_distance', by='Ulam_distance', showfliers=False, showmeans=True)
    plt.title('KT distance vs. Ulam distance')
    plt.suptitle('')
    plt.ylabel('KT distance')
    plt.tight_layout()
    plt.savefig(loc + 'kt_ulam.png')
    df.boxplot(column='KT_distance', by='si_ops', showfliers=False, showmeans=True)
    plt.title('KT distance vs. Selection/Insertion Operations')
    plt.suptitle('')
    plt.ylabel('KT distance')
    plt.tight_layout()
    plt.savefig(loc + 'kt_si.png')
    df.boxplot(column='Ulam_distance', by='si_ops', showfliers=False, showmeans=True)
    plt.title('Ulam distance vs. Selection/Insertion Operations')
    plt.suptitle('')
    plt.ylabel('Ulam distance')
    plt.tight_layout()
    plt.savefig(loc + 'ulam_si.png')
    df.hist(column='actions')
    plt.title('# Voter operations')
    plt.suptitle('')
    plt.tight_layout()
    plt.savefig(loc + 'ops.png')
    plt.show()
    df.hist(column='KT_distance')
    plt.title('KT distance')
    plt.suptitle('')
    plt.tight_layout()
    plt.savefig(loc + 'kt.png')
    plt.show()
    df.hist(column='Ulam_distance')
    plt.title('Ulam distance')
    plt.suptitle('')
    plt.tight_layout()
    plt.savefig(loc + 'ulam.png')
    plt.show()
    print(ops, ins, sel, either)
    with open(loc + 'stats.txt', 'w') as fo:
        fo.write('key, mean, med, std\n')
        for k in list(df):
            fo.write('{}, {}, {}, {}\n'.format(k, np.mean(df[k]), np.median(df[k]), np.std(df[k])))
    return None



def selins(vote):
    vote['user_data'] = vote['data']
    vote = compute_f_moves(vote, 0)
    vote['si_ops'] = len(vote['data'])
    vote = compute_moves_in_vote(vote)
    return vote


fts = {#'sqrt_moves': ['sum_sqrt_mag', 'num_moves'], 'sq_moves': ['sum_sq_mag', 'num_moves'],
       #'log_moves': ['sum_log_mag', 'num_moves'], 'linear_moves': ['sum_mag', 'num_moves'],
       #'sqrt_si': ['sum_sqrt_mag', 'si_ops'], 'sq_si': ['sum_sq_mag', 'si_ops'],
       #'log_si': ['sum_log_mag', 'si_ops'],
       # 'linear_si': ['sum_mag', 'si_ops'],
       #'kt_ulam': ['KT_distance', 'true_min_actions'],
        'kt_si': ['KT_distance', 'si_ops'],
        'si': ['si_ops'],
       #'sqrt': ['sum_sqrt_mag'], 'sq': ['sum_sq_mag'], 'log': ['sum_log_mag'],
       #'linear': ['sum_mag'],
         'KT': ['KT_distance'],
        #'Ulam': ['true_min_actions']
    }


def fit_votes_times_actual(votes, field, fn):
    df = pd.DataFrame.from_dict(votes)
    df = df.dropna()
    xcols = fn
    ycol = field
    dX = df[xcols]
    dy = df[ycol]
    intercept = True
    if 'si_ops' in xcols and len(xcols) > 1:
        intercept = False
    reg = linear_model.LinearRegression(fit_intercept=intercept)
    scores = model_selection.cross_val_score(reg, dX, dy, scoring='neg_mean_squared_error', cv=5)
    score = -1.*np.mean(scores)
    return score


def fit_votes_times_runner(collection, field):
    all_scores = defaultdict(list)
    field_data = list()
    best = list()
    for votes in collection:
        scores = defaultdict(float)
        for fn in fts:
            score = fit_votes_times_actual(votes, field, fts[fn])
            scores[fn] = score
            all_scores[fn].append(score)
        fdata = np.mean([v[field] for v in votes])
        field_data.append(fdata)
        bestft = min(scores, key=scores.get)
        best.append(bestft)
        all_scores['best'].append(scores[bestft])
    best = Counter(best)
    for fn in fts:
        scores = all_scores[fn]
        #print('{},{},{},{},{},{}'.format(field, fn, np.mean(scores), np.median(scores), np.mean(field_data), np.median(field_data)))
    return all_scores, best, field_data


def main2():
    loc = util.loc+util.sfx
    name = 'users-si-linear'
    field = 'time12'

    votes = util.read_dataset(loc=loc)
    votes = [selins(v) for v in votes]
    u2v = util.users_votes(votes)

    count_ops(votes)
    #all_scores, best, all_data = fit_votes_times_runner([u2v[u] for u in u2v], field)
    #report(all_scores, best, all_data, name, field, loc)


def compare_rec(votedict, loc, names):
    data = defaultdict(dict)
    for name in names:
        ops = 0
        ins = 0
        sel = 0
        either = 0
        time_per_at_n = defaultdict(list)
        time_at_n = defaultdict(list)
        sd_time_at_n = defaultdict(list)
        kt_at_n = defaultdict(list)
        per_sel_ins = list()
        either_at_n = defaultdict(list)
        moves_at_n = defaultdict(list)
        si_at_n = defaultdict(list)
        ops_at_n = dict()
        per_sel_ins_users = list()
        votes = votedict[name]
        u2v = util.users_votes(votes)
        for u in u2v:
            n = 0
            time_in_user = list()
            either_in_user = 0
            moves_user = 0
            for v in u2v[u]:
                either_in_vote = 0
                num_moves = len(v['user_data'])
                moves_user += num_moves
                time_in_user.append(v['time12'])
                for d in v['user_data']:
                    start_rank = d['start_rank']
                    end_rank = d['end_rank']
                    ins_rank = util.doInsertion(start_rank)
                    sel_rank = util.doSelection(start_rank)
                    ops += 1
                    if ins_rank == end_rank:
                        ins += 1
                    if sel_rank == end_rank:
                        sel += 1
                    if ins_rank == end_rank or sel_rank == end_rank:
                        either += 1
                        either_in_vote += 1
                either_in_user += either_in_vote
                either_at_n[n].append(either_in_vote)
                moves_at_n[n].append(num_moves)
                kt_at_n[n].append(v['KT_distance'])
                si_at_n[n].append(v['si_ops'])
                time_at_n[n].append(v['time12'])
                time_per_at_n[n].append(100.*v['time12']/time_in_user[0])
                either_in_vote = either_in_vote/num_moves
                per_sel_ins.append(either_in_vote)
                n += 1
            per_sel_ins_users.append(either_in_user/moves_user)
        ns = sorted(either_at_n.keys())
        data_time_at_n = [time_at_n[n] for n in ns]
        for n in either_at_n.keys():
            either_at_n[n] = sum(either_at_n[n])
            ops_at_n[n] = np.mean(moves_at_n[n])
            moves_at_n[n] = sum(moves_at_n[n])
            either_at_n[n] = 100*either_at_n[n]/moves_at_n[n]
            sd_time_at_n[n] = np.std(time_at_n[n])
            time_at_n[n] = np.mean(time_at_n[n])
            time_per_at_n[n] = np.mean(time_per_at_n[n])
            kt_at_n[n] = np.mean(kt_at_n[n])
            si_at_n[n] = np.mean(si_at_n[n])
        sd_time_at_n = [1*sd_time_at_n[n] for n in ns]
        time_at_n = [time_at_n[n] for n in ns]
        time_per_at_n = [time_per_at_n[n] for n in ns]
        either_at_n = [either_at_n[n] for n in ns]
        ops_at_n = [ops_at_n[n] for n in ns]
        kt_at_n = [kt_at_n[n] for n in ns]
        si_at_n = [si_at_n[n] for n in ns]
        data[name]['ns'] = ns
        data[name]['data_time_at_n'] = data_time_at_n
        data[name]['sd_time_at_n'] = sd_time_at_n
        data[name]['time_at_n'] = time_at_n
        data[name]['time_per_at_n'] = time_per_at_n
        data[name]['either_at_n'] = either_at_n
        data[name]['ops_at_n'] = ops_at_n
        data[name]['kt_at_n'] = kt_at_n
        data[name]['si_at_n'] = si_at_n
    plt.boxplot(data['random']['data_time_at_n'], showfliers=False)
    plt.boxplot(data['borda']['data_time_at_n'], showfliers=False)
    plt.xticks(range(util.num_polls))
    plt.xlabel('poll number in order of participation by user')
    plt.ylabel('average sorting time')
    plt.legend(('random', 'Borda'), loc='upper right')
    plt.savefig(loc + 'time_progress_box.png')
    plt.show()
    plt.errorbar([e+1 for e in data['random']['ns']], data['random']['time_at_n'], yerr=data['random']['sd_time_at_n'], linestyle='-', fmt='o', markersize=8, capsize=5)
    plt.errorbar([e+1 for e in data['borda']['ns']], data['borda']['time_at_n'], yerr=data['borda']['sd_time_at_n'], linestyle='-', fmt='^', markersize=8, capsize=5)
    plt.xlabel('poll number in order of participation by user')
    plt.ylabel('average sorting time')
    plt.legend(('random', 'Borda'), loc='upper right')
    plt.savefig(loc + 'time_progress_ebar.png')
    plt.show()
    plt.plot(data['random']['ns'], data['random']['time_at_n'], 'b--', data['borda']['ns'], data['borda']['time_at_n'], 'r^')
    plt.xlabel('poll number in order of participation by user')
    plt.ylabel('average sorting time')
    plt.legend(('random', 'Borda'), loc='upper right')
    plt.savefig(loc+'time_progress.png')
    plt.show()
    plt.plot(data['random']['ns'], data['random']['time_per_at_n'], 'b--', data['borda']['ns'], data['borda']['time_per_at_n'], 'r^')
    plt.xlabel('poll number in order of participation by user')
    plt.ylabel('average % time of first poll')
    plt.legend(('random', 'Borda'), loc='upper right')
    plt.savefig(loc+'time_per_progress.png')
    plt.show()
    plt.plot(data['random']['ns'], data['random']['ops_at_n'], 'b--', data['borda']['ns'], data['borda']['ops_at_n'], 'r^')
    plt.xlabel('poll number in order of participation by user')
    plt.ylabel('average # drag-and-drop operations')
    plt.legend(('random', 'Borda'), loc='upper right')
    plt.savefig(loc+'ops_progress.png')
    plt.show()
    plt.plot(data['random']['ns'], data['random']['kt_at_n'], 'b--', data['borda']['ns'], data['borda']['kt_at_n'], 'r^')
    plt.xlabel('poll number in order of participation by user')
    plt.ylabel('average KT distance')
    plt.legend(('random', 'Borda'), loc='upper right')
    plt.savefig(loc+'kt_progress.png')
    plt.show()
    plt.plot(data['random']['ns'], data['random']['si_at_n'], 'b--', data['borda']['ns'], data['borda']['si_at_n'], 'r^')
    plt.xlabel('poll number in order of participation by user')
    plt.ylabel('average # selection/insertion operations')
    plt.legend(('random', 'Borda'), loc='upper right')
    plt.savefig(loc + 'si_progress.png')
    plt.show()
    return None


def main():
    field = 'time12'
    name = 'random_rec'
    loc = util.loc + util.sfx + name + '/'
    random_votes = util.read_dataset_reco(loc, name)
    random_votes = [selins(v) for v in random_votes]
    #count_ops(random_votes, loc=loc)

    name = 'borda_rec'
    loc = util.loc + util.sfx + name + '/'
    borda_votes = util.read_dataset_reco(loc, name)
    borda_votes = [selins(v) for v in borda_votes]
    #count_ops(borda_votes, loc=loc)

    loc = util.loc + util.sfx
    votes = {'random': random_votes, 'borda': borda_votes}
    compare_rec(votes, loc, ['random', 'borda'])


if __name__ == '__main__':
    main()
    #util.write_dataset(loc=util.loc)

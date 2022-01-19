import sys
import copy
import glob
import json
import itertools
import numpy as np
import dateutil
import datetime
from matplotlib import pyplot as plt
from dateutil.tz import tzutc
from collections import defaultdict


#loc = '040918 (Among Us)/'
#loc, num_polls = '042018 (MTurk)/', 30
#loc, num_polls = '012118 (MTurk)/', 20
#sfx = 'new_exp/'

loc, num_polls = 'reco/', 10
sfx = 'sigma10/'
#name = 'random_rec'

start_time = datetime.datetime(2018, 1, 20, 1, 40, 0, tzinfo=tzutc())


def printUserDeviation(dic_all_correct_votes, typ="mean-std", perc=0.8):
    dic_mean_std = {}
    for u in dic_all_correct_votes:
        votes_time = [v['time12'] for v in dic_all_correct_votes[u]]
        mean = float(np.mean(votes_time))
        std = float(np.std(votes_time))
        if typ == "mean-std":
            dic_mean_std[u] = mean/std
        else:
            dic_mean_std[u] = std/mean

    plt.clf()
    plt.hist(dic_mean_std.values(),bins=20)
    plt.title('A Histogram of '+str(len(dic_all_correct_votes))+' Users')
    plt.ylabel('Count')
    # plt.ylim(0,25)
    plt.xlabel(typ)
    plt.savefig(typ+'.png')
    print(typ+'.png saved')
    plt.clf()

    sorted_mean_std = sorted(dic_mean_std, key=dic_mean_std.get)
    if typ == "mean-std":
        best = sorted_mean_std[int(np.floor(len(sorted_mean_std)*(1-perc))):]
    else:
        best = sorted_mean_std[:int(np.ceil(len(sorted_mean_std)*(perc)))]

    dic_all_correct_votes = {u: dic_all_correct_votes[u] for u in dic_all_correct_votes if u in best}
    return dic_all_correct_votes

def checkVote(v):
    try:
        if len(v['data']) == 0:
            return False
        for d in v['data']:
            if ('item' not in d):# or (len(d['item']) == 0):
                return False
            if len(d['start_rank']) == 0 or len(d['end_rank']) == 0:
                return False
        if dateutil.parser.parse(v["timestamp_submission"]) < start_time:
            return False
        if len(v['submitted_ranking']) == 0:
            return False
    except:
        print('Failed to check ', v)
        return False
    #submitted_names = [int(i['name'][4:]) for i in list(itertools.chain.from_iterable(v['submitted_ranking']))]
    #if sorted(submitted_names) != submitted_names:
    #    return False
    return True


def checkVote0(v):
    try:
        if len(v['data']) == 0:
            return False
        for d in v['data']:
            if ('item' not in d):# or (len(d['item']) == 0):
                return False
        if dateutil.parser.parse(v["timestamp_submission"]) < start_time:
            return False
        if len(v['submitted_ranking']) == 0:
            return False
    except:
        print('Failed to check ', v)
        checkVote(v)
        return False
    #submitted_names = [int(i['name'][4:]) for i in list(itertools.chain.from_iterable(v['submitted_ranking']))]
    #if sorted(submitted_names) != submitted_names:
    #    return False
    return True


def checkUser(u, votes, poll_ids):
    vote_poll_ids = [v['poll_id'] for v in votes if v['user_id']==u]
    if len(vote_poll_ids) != len(poll_ids):
        return False
    if set(vote_poll_ids) != poll_ids:
        return False
    return True


def flatten(inlist):
    outlist = list()
    for e in inlist:
        if isinstance(e, list):
            outlist+=e
        else:
            outlist.append(e)
    return outlist


def reformat(u):
    v = copy.deepcopy(u)
    v['time_submission'] = float(v['time_submission'])/1000
    initial_ranking = v['initial_ranking']
    submitted_ranking = v['submitted_ranking']
    initial_ranking = flatten(initial_ranking)
    submitted_ranking = flatten(submitted_ranking)
    initial_ranking = {int(i['name'][4:]): i['tier'] for i in initial_ranking}
    submitted_ranking = {int(i['name'][4:]): i['tier'] for i in submitted_ranking}
    v['initial_ranking'] = sorted(initial_ranking, key=initial_ranking.get)
    v['submitted_ranking'] = sorted(submitted_ranking, key=submitted_ranking.get)
    v = add_info(v)
    return v


def add_info(v):
    lis = len(LIS(v['initial_ranking']))
    v['true_min_actions'] = len(v['initial_ranking'])-lis
    v['actual_min_actions'] = getActualMinActions(v['initial_ranking'], v['submitted_ranking'])
    v['num_steps_insertion'] = numStepsInsertion(v['initial_ranking'])
    v['KT_distance'] = int(rank_distance(v['initial_ranking'], v['submitted_ranking'], method="kendalltau"))
    list_sort_alg = []
    for d in v['data']:
        try:
            start_rank, end_rank = d['rank']
            start_rank = list(itertools.chain.from_iterable(start_rank))
            end_rank = list(itertools.chain.from_iterable(end_rank))
            start_rank = {int(i['name'][4:]): i['tier'] for i in start_rank}
            end_rank = {int(i['name'][4:]): i['tier'] for i in end_rank}
            d['start_rank'] = sorted(start_rank, key=start_rank.get)
            d['end_rank'] = sorted(end_rank, key=end_rank.get)
            del d['rank']
        except:
            start_rank = d['start_rank']
            end_rank = d['end_rank']
        try:
            d['item'] = int(d['item'][4:])
        except:
            try:
                d['item'] = int(d['item'])
            except:
                moved_item = None
                for item in start_rank:
                    woitem_start = [i for i in start_rank if i != item]
                    woitem_end = [i for i in end_rank if i != item]
                    if woitem_start == woitem_end:
                        moved_item = item
                        break
                try:
                    assert (moved_item != None)
                except:
                    return None
                d['item'] = moved_item
        d['time'] = (float(d['time'][0]), float(d['time'][1]))
        list_sort_alg.append(checkSelectionInsertion(d))

    """
    dic_closeness = {}
    for i in range(len(v['data'])+1):
        dic_closeness[i] = closenessToF(list_sort_alg, i)
    max_closeness = max(dic_closeness.values())
    closest = [f for f in dic_closeness if dic_closeness[f] == max_closeness]
    closest_closeness = dic_closeness[closest[0]]
    if closest_closeness < 0.01:
        v['closest'] = [-1]
        # v['closest_closeness'] = closest_closeness
    else:
        v['closest'] = closest
    v['closest_closeness'] = closest_closeness
    """

    try:
        assert v['submitted_ranking'] == v['data'][-1]['end_rank'] and v['initial_ranking'] == v['data'][0]['start_rank']
    except:
        print(v)
    v['time1'] = v['data'][0]['time'][0]
    v['time2'] = v['data'][-1]['time'][1] - v['data'][0]['time'][0]
    v['time3'] = v['time_submission'] - v['data'][-1]['time'][1]
    v['time12'] = v['time1']+v['time2']
    v['num_actions'] = len(v['data'])
    return v


def getTiers(start_rank, end_rank):
    tiers = {}
    t = 1
    for i in end_rank:
        tiers[i] = t
        t += 1
    start_tiers = [tiers[i] for i in start_rank]
    end_tiers = [tiers[i] for i in end_rank]
    assert end_tiers == sorted(end_tiers)
    return start_tiers, end_tiers


def doInsertion(start_rank):
    sorted_list = sorted(start_rank)
    if sorted_list == start_rank:
        return None
    to_move = None
    for i in range(1, len(start_rank)):
        if start_rank[i] >= start_rank[i-1]:
            continue
        to_move = start_rank[i]
        to_move_pos = i
        break
    assert to_move != None
    end_rank = sorted(start_rank[:to_move_pos+1]) + start_rank[to_move_pos+1:]
    assert len(end_rank) == len(start_rank)
    return end_rank


def doInsertionMove(start_rank):
    sorted_list = sorted(start_rank)
    if sorted_list == start_rank:
        return None
    to_move = None
    for i in range(1, len(start_rank)):
        if start_rank[i] >= start_rank[i-1]:
            continue
        to_move = start_rank[i]
        to_move_pos = i
        break
    assert to_move != None
    end_rank = sorted(start_rank[:to_move_pos+1]) + start_rank[to_move_pos+1:]
    assert len(end_rank) == len(start_rank)
    return end_rank, to_move


def checkSelectionInsertion(d):
    start_rank = d['start_rank']
    end_rank = d['end_rank']
    ret = []
    if end_rank == doSelection(start_rank):
        ret.append("selection")
    if end_rank == doInsertion(start_rank):
        ret.append("insertion")
    if end_rank == doReversedSelection(start_rank):
        ret.append("r-selection")
    if end_rank == doReversedInsertion(start_rank):
        ret.append("r-insertion")
    if len(ret) == 0:
        ret.append("N_A")
    return ret


def doSelection(start_rank):
    sorted_list = sorted(start_rank)
    if sorted_list == start_rank:
        return None
    # print start_rank
    # print sorted_list
    to_move = sorted_list[0]
    for i in range(len(sorted_list)):
        if sorted_list[i] == start_rank[i]:
            continue
        to_move = sorted_list[i]
        break
    # print to_move
    end_rank = [i for i in start_rank if i < to_move]+[to_move]+[i for i in start_rank if i > to_move]
    assert len(end_rank) == len(start_rank)
    return end_rank


def doSelectionMove(start_rank):
    sorted_list = sorted(start_rank)
    if sorted_list == start_rank:
        return None
    # print start_rank
    # print sorted_list
    to_move = sorted_list[0]
    for i in range(len(sorted_list)):
        if sorted_list[i] == start_rank[i]:
            continue
        to_move = sorted_list[i]
        break
    # print to_move
    end_rank = [i for i in start_rank if i < to_move]+[to_move]+[i for i in start_rank if i > to_move]
    assert len(end_rank) == len(start_rank)
    return end_rank, to_move


def doReversedSelection(start_rank):
    sorted_list = sorted(start_rank)
    if sorted_list == start_rank:
        return None
    # print start_rank
    # print sorted_list
    to_move = sorted_list[0]
    for i in reversed(range(len(sorted_list))):
        if sorted_list[i] == start_rank[i]:
            continue
        to_move = sorted_list[i]
        break
    # print to_move
    end_rank = [i for i in start_rank if i < to_move]+[to_move]+[i for i in start_rank if i > to_move]
    assert len(end_rank) == len(start_rank)
    return end_rank


def doReversedInsertion(start_rank):
    sorted_list = sorted(start_rank)
    if sorted_list == start_rank:
        return None
    for i in reversed(range(len(start_rank)-1)):
        if start_rank[i] <= start_rank[i+1]:
            continue
        to_move = start_rank[i]
        to_move_pos = i
        break
    assert to_move != None
    end_rank = start_rank[:to_move_pos] + sorted(start_rank[to_move_pos:])
    assert len(end_rank) == len(start_rank)
    return end_rank


def numStepsInsertion(start_rank):
    truth = sorted(start_rank)
    count = 0
    while start_rank != truth:
        end_rank = doInsertion(start_rank)
        start_rank = end_rank
        count += 1
    return count


def rank_distance(x, y, weights=None, method='spearman'):
    """
    Distance measure between rankings.
    Parameters
    ----------
    x, y: array-like
        1-D permutations of [1..N] vector
    weights: array-like, optional
        1-D array of weights. Default None equals to unit weights.
    method: {'spearman'm 'kendalltau'}, optional
        Defines the method to find distance:
        'spearman' - sum of absolute distance between same elements
        in x and y.
        'kendalltau' - number of inverions needed to bring x to y.
        Default is 'spearman'.
    Return
    ------
    distance: float
        Distance between x and y.
    Example
    -------
    >>> from scipy.stats import rank_distance
    >>> rank_distance([1,3,4,2],[2,3,1,4])
    6.0
    >>> rank_distance([1,3,4,2],[2,3,1,4], method='kendalltau')
    4.0
    """
    x = np.asarray(x)
    y = np.asarray(y)
    if np.unique(x).size != x.size or np.unique(y).size != y.size:
        raise ValueError("x and y must have only unique elements")
    if x.size != y.size:
        raise ValueError("x and y have different size")
    if weights is None:
        weights = np.ones(x.size - 1)
    else:
        weights = np.asarray(weights)
        if weights.size < (x.size - 1):
            raise ValueError("weights vector have a small size")
    if method == 'spearman':
        return _spearman_footrule(x, y, weights)
    elif method == 'kendalltau':
        return _kendalltau_distance(x, y, weights)
    else:
        raise ValueError("unknown value for method parameter.")


def _spearman_footrule(x, y, weights):
    distance = 0
    for i in range(x.size):
        x_index = np.where(x == x[i])[0][0]
        y_index = np.where(y == x[i])[0][0]
        pair = np.abs(x_index - y_index)
        min_index = np.minimum(x_index, y_index)
        for j in range(pair):
            distance += weights[min_index + j]
    return distance


def _kendalltau_distance(x, y, weights):
    distance = 0
    n = x.size - 1
    for i in range(n - 1, -1, -1):
        key = x[i]
        j = i + 1
        while j <= n and np.where(y == key)[0] > np.where(y == x[j])[0]:
            x[j - 1] = x[j]
            distance += weights[j - 1]
            j += 1
        x[j - 1] = key
    return distance


def getActualMinActions(start_rank, end_rank):
    start_tiers, end_tiers = getTiers(start_rank, end_rank)
    lis = len(LIS(start_tiers))
    min_actions = len(start_tiers)-lis
    return min_actions


def LIS(X):
    """Returns the Longest Increasing Subsequence in the Given List/Array"""
    N = len(X)
    P = [0] * N
    M = [0] * (N + 1)
    L = 0
    for i in range(N):
        lo = 1
        hi = L
        while lo <= hi:
            mid = (lo + hi) // 2
            if (X[M[mid]] < X[i]):
                lo = mid + 1
            else:
                hi = mid - 1

        newL = lo
        P[i] = M[newL - 1]
        M[newL] = i

        if (newL > L):
            L = newL

    S = []
    k = M[L]
    for i in range(L - 1, -1, -1):
        S.append(X[k])
        k = P[k]
    return S[::-1]


def closenessToF(list_sort_alg, f):
    pt1 = list_sort_alg[:f]
    pt2 = list_sort_alg[f:]
    pt1_len = len(pt1)
    pt2_len = len(pt2)
    assert pt1_len == f
    pt1 = list(itertools.chain.from_iterable(pt1))
    pt2 = list(itertools.chain.from_iterable(pt2))
    pt1_closeness = pt1.count('selection')
    pt2_closeness = pt2.count('insertion')
    assert pt1_len >= pt1_closeness
    assert pt2_len >= pt2_closeness
    ovr_closeness = float(pt1_closeness+pt2_closeness)/(pt1_len+pt2_len)
    return ovr_closeness


def users_votes(votes):
    u2v = defaultdict(list)
    for v in votes:
        user = v['user_id']
        u2v[user].append(v)
    return u2v


def write_dataset_old(loc=loc):
    all_votes = json.load(open(loc+'vote.txt'))
    all_users = json.load(open(loc+'user.txt'))
    poll_ids = set([v['poll_id'] for v in all_votes])
    assert len(poll_ids) == 20
    list_all_correct_votes = [v for v in all_votes if checkVote(v)]
    #good_user_ids = [u['user_id'] for u in all_users if checkUser(u['user_id'], list_all_correct_votes, poll_ids)]
    #print([v['user_id'] for v in list_all_correct_votes])
    with open(loc+sfx+'ulist.json') as f:
        good_user_ids = json.load(f)
    print('initial_good_users', len(good_user_ids))
    list_all_correct_votes = [v for v in list_all_correct_votes if v['user_id'] in good_user_ids]
    print(len(list_all_correct_votes))
    u2v = users_votes(list_all_correct_votes)
    user_num_votes = [len(u2v[u]) for u in u2v]
    plt.hist(user_num_votes, bins=20)
    plt.show()
    #assert len(list_all_correct_votes)%len(poll_ids) == 0
    for v in list_all_correct_votes:
        reformat(v)

    dic_all_correct_votes = {}
    for v in list_all_correct_votes:
        if v['user_id'] not in dic_all_correct_votes:
            dic_all_correct_votes[v['user_id']] = []
        dic_all_correct_votes[v['user_id']].append(v)

    dic_all_correct_votes = printUserDeviation(dic_all_correct_votes, typ="mean-std")
    list_all_correct_votes = list(itertools.chain.from_iterable(dic_all_correct_votes.values()))
    all_votes = list_all_correct_votes

    u2v = users_votes(all_votes)
    print('#votes: ',len(all_votes), '#users: ', len(u2v.keys()))
    for u in u2v:
        with open(loc+sfx+str(u)+'_votes.json','w') as fo:
            #print(u2v[u])
            user_votes = json.dumps(u2v[u], indent=4, separators=(',', ': '))
            fo.write(user_votes)


def write_dataset_old_2(loc=loc):
    with open(loc+sfx+'vlist.txt') as f:
        u2v = eval(f.read())
    good_votes = [v for u in u2v for v in u2v[u]]
    good_users = u2v.keys()
    print('#votes: ',len(good_votes), '#users: ', len(u2v.keys()))
    for u in u2v:
        with open(loc+sfx+str(u)+'_votes.json','w') as fo:
            #print(u2v[u])
            user_votes = json.dumps(u2v[u], indent=4, separators=(',', ': '))
            fo.write(user_votes)


def write_dataset(loc=loc):
    all_votes = json.load(open(loc+'vote.txt'))
    print(all_votes[0])
    good_users = json.load(open(loc+sfx+'ulist.txt'))
    print('# good users ', len(good_users))
    good_votes = [v for v in all_votes if v['user_id'] in good_users]
    print('# votes good users ', len(good_votes))
    good_votes = [v for v in good_votes if checkVote0(v)]
    print('# votes checked ', len(good_votes))
    good_votes = [reformat(v) for v in good_votes]
    print('# votes reformatter ', len(good_votes))
    good_votes = [v for v in good_votes if checkVote(v)]
    print('# votes checked ', len(good_votes))
    u2v = users_votes(good_votes)
    print('#votes: ',len(good_votes), '#users: ', len(u2v.keys()))
    for u in u2v:
        with open(loc+sfx+str(u)+'_votes.json','w') as fo:
            #print(u2v[u])
            user_votes = json.dumps(u2v[u], indent=4, separators=(',', ': '))
            fo.write(user_votes)


def read_dataset(loc=loc+sfx):
    print('reading votes from {}'.format(loc))
    all_votes = list()
    for fname in glob.glob(loc+'*_votes.json'):
        with open(fname) as f:
            uvotes = json.loads(f.read())
            uvotes = [add_info(v) for v in uvotes if len(v['data']) > 0]
            if len(uvotes) == num_polls:
                all_votes += uvotes
            else:
                print(len(uvotes))
        #break
    print('done reading votes {}'.format(len(all_votes)))
    #print(all_votes[0])
    return all_votes


def read_dataset_reco(loc, name):
    print('reading votes from {}'.format(loc))
    raw_votes = list()
    all_votes = list()
    users = list()
    for fname in glob.glob(loc + '*users*' + name + '*.txt'):
        print('reading users from {}'.format(fname))
        with open(fname) as f:
            users += json.loads(f.read())
    print('# users read ',len(users))
    for fname in glob.glob(loc+'*'+name+'*.json'):
        print('reading votes from {}'.format(fname))
        with open(fname) as f:
            fvotes = json.loads(f.read())
            votes = [reformat(v) for v in fvotes if len(v['data']) > 0 if reformat(v)]
            votes = [v for v in votes if v != None]
            raw_votes += votes
    print('raw votes read ', len(raw_votes))
    u2v = users_votes(raw_votes)
    for u in u2v:
        if len(u2v[u]) == num_polls and u in users:
            all_votes += u2v[u]
    print('done reading votes {}'.format(len(all_votes)))
    #print(all_votes[0])
    return all_votes

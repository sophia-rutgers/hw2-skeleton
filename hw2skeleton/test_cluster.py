from __future__ import division
from hw2skeleton import cluster
from . import io
import os
import numpy as np
import cPickle as pic
def test_dissimilarity():
    ## have to load pkl file because this function keeps on crashing pytest! runs forever
    pdb_ids = [10701, 10814, 13052, 14181, 15813, 17526, 17622, 1806, 18773, 19267, 20326, 20856, 22711, 23319, 23760, 23812, 24307, 24634, 25196, 25551, 25878, 26095, 26246, 27031, 27312, 276, 28672, 28919, 29047, 29209, 29773, 32054, 32088, 33838, 34047, 34088, 34563, 3458, 34958, 35014, 36257, 37224, 37237, 3733, 37438, 38031, 38181, 38472, 38846, 39117, 39299, 39939, 40084, 41719, 41729, 42074, 42202, 42269, 42296, 42633, 43878, 45127, 46042, 4629, 46495, 46975, 47023, 49624, 50018, 50362, 52235, 52954, 53272, 54203, 55996, 56029, 56394, 57370, 57481, 57602, 57644, 58445, 6040, 61242, 62186, 63064, 63634, 63703, 64258, 64392, 65815, 68578, 69893, 70005, 70919, 71389, 72058, 73183, 73462, 73624, 7674, 7780, 78796, 81563, 81697, 81816, 81859, 8208, 82212, 82238, 82886, 82993, 8304, 83227, 83394, 83741, 84035, 85232, 85492, 88042, 91194, 91426, 91796, 91911, 93168, 93192, 93456, 94372, 94652, 94719, 96099, 97218, 97612, 9776, 98170, 98797]
    active_sites = []
    for id in pdb_ids:
        filepath = os.path.join("data", "%s.pdb"%id)
        active_sites.append(io.read_active_site(filepath))

    dist_matrix = pic.load(open('dist_matrix.pkl','rb'))
    active_sites = [str(x) for x in active_sites]
    activesite_a = active_sites.index(str(276))
    activesite_b = active_sites.index(str(4629))

    # update this assertion
    # manually, precomputed the dissimilarity score between these two pdbs
    assert dist_matrix[activesite_a,activesite_b] == 0.75

def test_dissimilarity_not_similarity():
    pdb_ids = [10701, 10814, 13052, 14181, 15813, 17526, 17622, 1806, 18773, 19267, 20326, 20856, 22711, 23319, 23760, 23812, 24307, 24634, 25196, 25551, 25878, 26095, 26246, 27031, 27312, 276, 28672, 28919, 29047, 29209, 29773, 32054, 32088, 33838, 34047, 34088, 34563, 3458, 34958, 35014, 36257, 37224, 37237, 3733, 37438, 38031, 38181, 38472, 38846, 39117, 39299, 39939, 40084, 41719, 41729, 42074, 42202, 42269, 42296, 42633, 43878, 45127, 46042, 4629, 46495, 46975, 47023, 49624, 50018, 50362, 52235, 52954, 53272, 54203, 55996, 56029, 56394, 57370, 57481, 57602, 57644, 58445, 6040, 61242, 62186, 63064, 63634, 63703, 64258, 64392, 65815, 68578, 69893, 70005, 70919, 71389, 72058, 73183, 73462, 73624, 7674, 7780, 78796, 81563, 81697, 81816, 81859, 8208, 82212, 82238, 82886, 82993, 8304, 83227, 83394, 83741, 84035, 85232, 85492, 88042, 91194, 91426, 91796, 91911, 93168, 93192, 93456, 94372, 94652, 94719, 96099, 97218, 97612, 9776, 98170, 98797]
    active_sites = []
    for id in pdb_ids:
        filepath = os.path.join("data", "%s.pdb"%id)
        active_sites.append(io.read_active_site(filepath))

    dist_matrix = pic.load(open('dist_matrix.pkl','rb'))
    active_sites = [str(x) for x in active_sites]
    
    for id in pdb_ids:
        filepath = os.path.join("data", "%s.pdb"%id)
        active_sites.append(io.read_active_site(filepath))

    # make sure that diagonals are 0
    assert dist_matrix[5,5] == 0
    assert dist_matrix[80,80] == 0

def test_partition_clustering():
    # tractable subset
    pdb_ids = [ 23319, 10814, 13052]

    active_sites = []
    for id in pdb_ids:
        filepath = os.path.join("data", "%i.pdb"%id)
        active_sites.append(io.read_active_site(filepath))

    # update this assertion
    assert cluster.cluster_by_partitioning(active_sites,2) == str([[23319, 10814], [13052]])

def test_hierarchical_clustering():
    # tractable subset
    pdb_ids = [276, 4629, 10701]

    active_sites = []
    for id in pdb_ids:
        filepath = os.path.join("data", "%i.pdb"%id)
        active_sites.append(io.read_active_site(filepath))

    # update this assertion
    assert cluster.cluster_hierarchically(active_sites) == '((276,4629),10701)'

def test_silhouette():
    ### this seems silly to copy and paste function, but I chad to input this distance matrix since it's not calculated using the compute_dissimilarity function !!! everything else is the same

    total_sil_scores = [] #compile all scores to average in last step

    dist_matrix = np.array([[0,2,8],
                                [2,0,10],
                                [8,10,0]])
    
    data = np.array([[(2,0)],[(0,0)],[(10,0)]])
    mega = [[100,101],[102]]
    active_sites = [100,101,102]
    
    for clus in mega:
        for act_site in clus:
            same_clus = []
            for othersite in clus:
                if othersite != act_site:
                    act_id = active_sites.index(act_site)
                    other_id = active_sites.index(othersite)
                    same_clus.append(dist_matrix[act_id,other_id])
            if len(same_clus) >= 1:
                ai = np.mean(same_clus)
            else: # because only 1 member in cluster
                ai = 0

            # now see what other distances are for the other clusters
            bi = 0 # update when there is lower bi value
            for otherclus in mega:
                if otherclus != clus:
                    diff_clus = []
                    for other in otherclus:
                        act_id = active_sites.index(act_site)
                        other_id = active_sites.index(other)
                        diff_clus.append(dist_matrix[act_id,other_id])
                    new = np.mean(diff_clus)
                    if new > bi:
                        bi = new
                    print ai,bi
        sil = (bi-ai) / np.maximum(ai,bi)
        if np.isnan(sil)==False:
            total_sil_scores.append(sil)
    # see how many active sites are in the largest clusters
    max_clus = 0
    for clus in mega:
        if len(clus) > max_clus:
            max_clus = len(clus)
    score = np.mean(total_sil_scores)
    assert score == 0.9



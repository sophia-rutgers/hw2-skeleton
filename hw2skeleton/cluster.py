from __future__ import division
from .utils import Atom, Residue, ActiveSite
import numpy as np
from prody import *
import re
import cPickle as pic

def jaccard(active_sites):
    ''' Takes active_sites as input and returns a jaccard DISSIMILARITY matrix'''

    x = len(active_sites)
    jaccard_matrix = np.zeros((x,x))
    for i in range(x):
        a_residues = []
        for j in range(x):
            b_residues = []
            for pdb in [active_sites[i], active_sites[j]]:
                for res in pdb.residues:
                    if str(pdb) == str(active_sites[i]):
                        a_residues.append(res.type)
                    if str(pdb) == str(active_sites[j]):
                        b_residues.append(res.type)
            ins = set.intersection(set(a_residues),set(b_residues))
            union = set.union(set(a_residues),set(b_residues))
            jaccard = len(ins)/len(union)
            jaccard_matrix[i,j] = jaccard
            
            # convert jaccard similarity into jaccard DISSIMILARITY
    for m in range(x):
        for n in range(x):
            jaccard_matrix[m,n] = 1 - jaccard_matrix[m,n]

    return jaccard_matrix

def rmsd_matrix(active_sites):
    ''' Takes active_sites as input and returns rmsd distance matrix'''
    x = len(active_sites)
    direc = '/Users/student/Algo203/hw2-skeleton/data'
    # making rmsd matrix to determine max rmsd for scaling
    rmsd_matrix = np.zeros((x,x)) 
    for i in range(x):
        namei = '%s/%s.pdb' % (direc,active_sites[i])
        pdb_i = parsePDB(namei)['A']# takes only chain A 
        try:
            pdb_i = pdb_i.select('calpha')
        except:
            pass
        for j in range(x):
            namej = '%s/%s.pdb' % (direc,active_sites[j])
            pdb_j = parsePDB(namej)['A']# takes only chain A 
            try:
                pdb_j = pdb_j.select('calpha')
            except:
                pass
            try:
                rmsd_matrix[i,j] = calcRMSD(pdb_i,pdb_j)
            except:
                pass

    # normalize/scale everything
    max = np.max(rmsd_matrix)
    for m in range(x):
        for n in range(x):
            rmsd_matrix[m,n] = (rmsd_matrix[m,n]/max)
    
    # replace all the pairs that you can't calculate rmsd for with 1, because for pairs that don't have the same number of c-alphas (and can't calculate rmsd for) are the MOST dissimilar in my opinion.  keep the diagonals as 0. 
    for i in range(x):
        for j in range(x):
            if rmsd_matrix[i,j] == 0:
                if i != j:
                    rmsd_matrix[i,j] = 1

    return rmsd_matrix

def charge_matrix(active_sites):
    # this is a DISSIMILARITY matrix
    x = len(active_sites)
    charge_matrix = np.zeros((x,x))
    charge = {}
    for res in ['HIS', 'LYS', 'ARG']:
        charge[res] = 1
    for res in ['GLU', 'ASP']:
        charge[res] = -1
    
    for i in range(x):
        i_charge = 0
        pdb_i = active_sites[i].residues
        for res in pdb_i:
            if res.type in charge:
                i_charge += charge[res.type]

        for j in range(x):
            j_charge = 0
            pdb_j = active_sites[j].residues
            for  res in pdb_j:
                if res.type in charge:
                    j_charge += charge[res.type]
        
            charge_matrix[i,j] = abs(i_charge-j_charge)
    # SCALE IT TO 1!
    max = np.max(charge_matrix)
    for m in range(x):
        for n in range(x):
            charge_matrix[m,n] = charge_matrix[m,n]/max 
    
    return charge_matrix 

def compute_dissimilarity(active_sites):
    """
    Compute the DISsimilarity between two given ActiveSite instances. THIS RETURNS A DISTANCE MATRIX.

    Input: two ActiveSite instances
    Output: distance matrix
    """
    # Fill in your code here!
    '''Generates distance matrix'''
    jaccard_matrix = pic.load(open('jaccard_matrix.pkl'))
    charge_matrix = pic.load(open('charge_matrix.pkl'))
    rmsd_matrix = pic.load(open('rmsd_matrix.pkl'))


    x = len(active_sites)
    dist_matrix = np.zeros((x,x))
    for i in range(x):
        for j in range(x):
            jaccard = jaccard_matrix[i,j]
            rmsd_dist = rmsd_matrix[i,j]
            charge_dist = charge_matrix[i,j]
            dist_matrix[i,j] = np.mean([jaccard,rmsd_dist,charge_dist])
    return dist_matrix


######### PARTITIONING CLUS ADAPTED FROM GITHUB USER "letiantian" #############

def cluster_by_partitioning(active_sites,k):
    """
    Cluster a given set of ActiveSite instances using a partitioning method.

    Input: a list of ActiveSite instances and k number of clusters
    Output: a clustering of ActiveSite instances
            (this is really a list of clusters, each of which is list of
            ActiveSite instances)
    """
    # Fill in your code here!

    # D is distance matrix 
    D = compute_dissimilarity(active_sites)
    m, n = D.shape

    if k > n:
        raise Exception('too many medoids')

    # this randomly initializes an array of k medoid indices. but I fixed it for reproducible testing results
    M = np.arange(n)
    np.random.seed(seed=0)
    np.random.shuffle(M)
    M = np.sort(M[:k])

    # create a copy of the array of medoid indices so M is not overwritten
    Mnew = np.copy(M)

    # author uses a dictionary to represent clusters
    C = {}
    for t in xrange(10000):
        # determine clusters (arrays of data indices)
        J = np.argmin(D[:,M], axis=1)
        for kappa in range(k):
            C[kappa] = np.where(J==kappa)[0]
        # update cluster medoids
        for kappa in range(k):
            J = np.mean(D[np.ix_(C[kappa],C[kappa])],axis=1)
            j = np.argmin(J)
            Mnew[kappa] = C[kappa][j]
        np.sort(Mnew)
        # check for convergence, stop code if M and Mnew are the same
        if np.array_equal(M, Mnew):
            break
        M = np.copy(Mnew)
    else:
        # final update of cluster memberships if it hasn't converged by 100 iterations
        J = np.argmin(D[:,M], axis=1)
        for kappa in range(k):
            C[kappa] = np.where(J==kappa)[0]
    
    # store the clusters in a megacluster (list of clusters that contain active sites) 
    megaclusters = []
    labels = []
    for key,val in C.iteritems():
        clus = []
        labels_clus = []
        for ix in val:
            clus.append(active_sites[ix])
            labels_clus.append(ix)
        megaclusters.append(clus)
        labels.append(labels_clus)
    
    return str(megaclusters)

######### HIER CLUS ADAPTED FROM R. FLYNN #############
"""
Agglomerative Clustering Algorithm

Iteratively build hierarchical cluster between all data points.
O(n^2) complexity

Author: Ryan Flynn <parseerror+agglomerative-clustering@gmail.com>
"""

class Cluster:
	def __init__(self):
		pass
	def __repr__(self):
		return '(%s,%s)' % (self.left, self.right)
	def add(self, clusters, grid, lefti, righti):
        # lefti is row (i),  righti is column (j)
		self.left = clusters[lefti]
		self.right = clusters[righti]
		# merge columns grid[row][righti] and row grid[righti] into corresponding lefti
		for r in grid:
        # this is where the code actually takes the labels that have the smallest distance and makes a new cluster
			r[lefti] = min(r[lefti], r.pop(righti))
		grid[lefti] = map(min, zip(grid[lefti], grid.pop(righti)))
		clusters.pop(righti)
		return (clusters, grid)

def cluster_hierarchically(active_sites):
    """
    Cluster the given set of ActiveSite instances using a hierarchical algorithm.                                                                  #
    Input: a list of ActiveSite instances
    Output: a list of clusterings
            (each clustering is a list of lists of Sequence objects)
    """
    # Fill in your code here!
    dist_matrix = compute_dissimilarity(active_sites)
	
    clusters = active_sites[:]
    # makes a copy, otherwise this changes active_sites

    grid = []
    for i in dist_matrix:
        grid.append(list(i))

    # i made a megalist to store clusters at each iteration so i could determine the ideal cluster number. 
    x = len(active_sites)
    megaclusters = []
    while len(clusters) > 1:
        # save clusters to dictionary
        megaclusters.append(list(clusters))
        
        
        distances = [(1, 0, grid[1][0])]
        # distances is a tuple representing [row,column,dist value]. it starts at row 1, col 0, because 0,0 (and all diagionals) are 0
        # now, make distances a huge list that contains all index information (row, col) as well as the distance
	
        for i,row in enumerate(grid[2:]):
			distances += [(i+2, j, c) for j,c in enumerate(row[:i+2])]
        # the above 2 lines are basically updating the row and column indices so that it makes the 'distances' variable a huge list that tacks on row and column indices to the distance value. together, grid[2:] and row[:i+2] restrict this new giant list to only values in the upper triangle half of the input redundant matrix.  i+2 is row, j is col, c is distance
        j,i,_ = min(distances, key=lambda x:x[2])
        # this identifies the minimum distance and itsindex (row,col). j = col, i=row, _ = dist
        # use that information to put it into the class created above, and make a new cluster!
        c = Cluster()
        clusters, grid = c.add(clusters, grid, i, j)
        clusters[i] = c
        # clusters[i] is the cluster formed from the smallest distance! now, repeat until there is only one cluster
        
    return str(clusters.pop())
    #return megaclusters

def label_hier_clus(active_sites,clusnum):
    #requires active_sites and number of clusters as input, returns a list denoting which cluster that active site belongs to
    x = len(active_sites)
    mega= cluster_hierarchically(active_sites)
    clus_dict = {}
    # dictionary to store labels. key = cluster number, value = ative site. this resets for every iteration of clustering
    for n in range(clusnum):
        clus_dict[n] = []
        # pull out clusnum iteration, access all the clusters with [n]
        clus = list(mega[x-clusnum])[n]
        clus = str(clus)
        delimited= re.findall(r"[\w']+", clus)
        for i in delimited:
            clus_dict[n].append(i)
    labels = []
    # make a list of cluster number for the active sites
    for act in active_sites:
        for key,value in clus_dict.iteritems():
            if str(act) in value:
                labels.append(key)
    return labels
#print cluster_hierarchically(active_sites)

def hierarchical_silhouette(active_sites,clusnum):
    ''' Input: list of clusters that active sites belonging to that cluster'''
    
    mega = cluster_hierarchically(active_sites)

    dist_matrix = compute_dissimilarity(active_sites)
    # for all clusters
    active_sites = [str(ac) for ac in active_sites] #convert so i can pull out index later
    x = len(active_sites)
    
    total_sil_scores = [] # compile all scores to average in last step
    
    for n in range(clusnum):
        # pull out clusnum iteration, access all the clusters with [n]
        clus = list(mega[x-clusnum])[n]
        clus = str(clus)
        delimited= re.findall(r"[\w']+", clus)
        # for each data point to calculate ai and bi for silhouette
        for actsite in delimited:
            # gets dist with other active sites of same cluster
            same_clus = []
            for othersite in delimited:
                if actsite != othersite:
                    # get index of these active sites
                    act_id = active_sites.index(actsite)
                    other_id = active_sites.index(othersite)
                    same_clus.append(dist_matrix[act_id,other_id])
            if len(same_clus) >= 1:
                ai = np.mean(same_clus) 
            else: # if list is empty because only one cluster member!!
                ai = 0 # because that means only 1 member in cluster
            #now see what the other distances are for the other clusters
            bi = 100000 #random value. update this every time we find a cluster with a lower bi

            # for all the othe clusters excluding this one
            for h in range(clusnum):
                diff_clus = [] #must reset for every new cluster
                clus = list(mega[x-clusnum])[h]
                clus = str(clus)
                delimited= re.findall(r"[\w']+", clus)
                if h != n:
                    for other in delimited:
                        act_id = active_sites.index(actsite)
                        other_id = active_sites.index(other)
                        diff_clus.append(dist_matrix[act_id,other_id])
                    new = np.mean(diff_clus)
                    if new < bi:
                        bi = new
            sil = (bi-ai)/np.maximum(ai,bi)
            if np.isnan(sil) == False:
                total_sil_scores.append(sil)
    # report on how many active sites are in the largest cluster, to see if that can explain the poor silhouette scores
    max_clus = 0
    for n in range(clusnum):
        clus = str(list(mega[x-clusnum])[n])
        delimited= re.findall(r"[\w']+", clus)
        if len(delimited) > max_clus:
            max_clus = len(delimited)

    print n, np.mean(total_sil_scores), max_clus


def partitioning_silhouette(active_sites,k):
    mega = cluster_by_partitioning(active_sites,k)
    dist_matrix = compute_dissimilarity(active_sites)

    total_sil_scores = [] #compile all scores to average in last step

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
            bi = 1000000000 #random. update when there is lower bi value
            for otherclus in mega:
                if otherclus != clus:
                    diff_clus = []
                    for other in otherclus:
                        act_id = active_sites.index(act_site)
                        other_id = active_sites.index(other)
                        diff_clus.append(dist_matrix[act_id,other_id])
                    new = np.mean(diff_clus)
                    if new < bi:
                        bi = new
        sil = (bi-ai) / np.maximum(ai,bi)
        if np.isnan(sil)==False:
            total_sil_scores.append(sil)
    # see how many active sites are in the largest clusters
    max_clus = 0
    for clus in mega:
        if len(clus) > max_clus:
            max_clus = len(clus)
    print np.mean(total_sil_scores), max_clus


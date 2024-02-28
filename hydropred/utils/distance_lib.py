import numpy as np


def min_distance_arr(rows,cols,lattice_spacing):
    """
    Calculates minimum distance between nodes of hexagonal array, respecting periodic
        boundary conditions.

    Parameters
    ----------
    rows : int
        Number of rows in array
    cols : int
        Number of columns in array
    lattice_spacing : float 
        Distance between nodes when in a hexagonal pattern and three nodes
        form an equilateral triangle (this is the length of the side of the triangle).

    Returns
    -------
    numpy array
        Upper triangular array of distances between each node.

    """
    # creates upper triangular matrix of distances between nodes
    
    # calculate the minimum distance between points and return array containing values
    dist_arr = np.zeros((rows*cols,rows*cols))
    dist_btw_cols = lattice_spacing/2 #(aka x of 30-60-90 triangle)
    placement_coords = np.zeros((rows,cols,2))
    for i in range(rows):
        for j in range(cols):
            placement_coords[i,j,0] =  i * dist_btw_cols * np.sqrt(3)
            if i%2 == 0:    #even row
                placement_coords[i,j,1] =  lattice_spacing*j 
            else:
                placement_coords[i,j,1] =  lattice_spacing*j+dist_btw_cols
            
    # rearrange to mimic periodic boundary condition
    placement_arr = np.arange(rows*cols).reshape((rows,cols))
    placement_arr = np.concatenate((placement_arr, placement_arr))
    placement_arr = np.concatenate((placement_arr, placement_arr),axis=1)
    
    for num in range(rows*cols):
        row,col = np.where(placement_arr==num)
        # each number is in there 4 times, but we just need one that will give us a good box
        for pos in range(len(row)):
            if (row[pos] >= rows//2) & (row[pos] <= (rows*2-rows//2)) & (col[pos] >= cols//2) & (col[pos]<= (cols*2-cols//2)):
                # this is a good one)
                break
        
        # cut part out of the array 
        cut_row = row[pos]-int(rows/2)
        # first row needs to be an even row
        if cut_row%2 == 1:
            cut_row += 1

        cut_col = col[pos]-int(cols/2)
        sub_array = placement_arr[cut_row:cut_row+rows, cut_col:cut_col+cols]

        # each value is in the sub_array one time, so when we use np.where, we will get a single row,col pair
        #    use that pair in the placement_coords array and then compute distance to each of the other points
        #    with greater values, store in dist_arr
        
        # get location of num:
        num_row, num_col = np.where(sub_array == num)
        # coordinates for num
        num_coord = np.array([placement_coords[num_row[0],num_col[0],0],placement_coords[num_row[0],num_col[0],1]])
        for next_num in range(num+1,rows*cols):
            next_num_row, next_num_col = np.where(sub_array == next_num)
            dist_arr[num,next_num] = np.linalg.norm(np.array([placement_coords[next_num_row[0],next_num_col[0],0],placement_coords[next_num_row[0],next_num_col[0],1]])-
                                                    num_coord)    
    
    return np.round(dist_arr,10)
    
def d0_features(arr, lattice_spacing, polar_np_means = True, make_hist = False, max_dist = 0, make_mean_hist = True):
    """
    Creates features based on distance between atoms(nodes). 

    Parameters
    ----------
    arr : numpy array
        Array containing 0/1 polarity values.
    lattice spacing : float
        Distance between nodes when in a hexagonal pattern and three nodes
        form an equilateral triangle (this is the length of the side of the triangle).
    polar_np_means : bool
        True by default, will create mean of inverse min distance from each node to polar nodes 
        and mean to nonpolar nodes.
    make_hist : bool
        False by default, will create histogram of polar and non-polar distances as features.
    max_dist : float
        Optional. If greater than 0, count the number of polar nodes within the max distance of 
        the polar node. Returns -1 if node is non-polar.
        If used, max_distance should be lattice_spacing or greater.
    mean_hist : bool
        True by default, returns mean values across all nodes for 22 total features per sample
        If false, returns 22 features per node per sample

    Returns
    -------
    numpy array(s)
        List of three features for modeling: 
        mean min distance array: 
            inverse of minimum mean distance from each node to other polar and nonpolar nodes in the array, obeys periodic boundary conditions
        histogram pdf array: 
            values of PDF of histogram distances to other polar and nonpolar nodes, based on 11 bins for histogram
        polar counts: number of polar nodes within the specified distance of the node, including the node itself

    """
    # arr = array of polar/non-polar
    num_rows = arr.shape[0]
    num_cols = arr.shape[1]
    if polar_np_means:
        attributes = np.zeros((num_rows, num_cols,2))
    else:
        attributes = []

    dist_arr = min_distance_arr(num_rows, num_cols, lattice_spacing)
    
    # use distance array to find max possible number of polar nodes within a given distance
    # if total number of polar nodes-1 in the sample is less than max possible value, use total
    # number of polar nodes-1 as max

    # for histogram features
    if make_hist:
        num_bins = 11
        bin_max = np.max(dist_arr)
        bin_min = lattice_spacing

        # try just looking at polar (change 2 to 1 and comment out ~line 147 where np are added to attributes)
        hist_attr = np.zeros((num_cols*num_rows,2,num_bins))
    else:
        mean_hist = []
    
    # polar counts
    if max_dist > 0:
        polar_count_attr = np.zeros((num_cols*num_rows,1))
    else:
        polar_count_attr = []

    for i in range(num_rows*num_cols):
        polars = []
        nonpolars = []
        polar_count = 0
        # see if this is a polar node
        is_polar = (arr[i//num_cols, i%num_cols] > 0)
        for j in range(num_rows*num_cols):
            if (i != j):
                # get distance from i to j
                # distance array is upper triangular
                if i<j:
                    d = dist_arr[i,j]
                else:
                    d = dist_arr[j,i]
                # see if j is polar or non-polar
                if arr[j//num_cols, j%num_cols] > 0:
                    polars.append(d)
                    #if max_dist > 0 and is_polar:
                    #    if d <= max_dist:
                    #        polar_count +=1
                    # try max_dist for both polar and non-polar nodes
                    if max_dist > 0 and d <= max_dist: 
                        polar_count += 1

                else:
                    nonpolars.append(d)
        
        if polar_np_means:
            # 2023-12-01 modification: use 1/d as distance feature, this way 
            #            will have closest with largest value
            # if polar=[] or nonpolar=[], assign 0
            if len(polars)==0:
                attributes[i//num_cols, i%num_cols, 0] = 0
            else:
                attributes[i//num_cols, i%num_cols, 0] = np.mean(1/np.asarray(polars))
            
            if len(nonpolars)==0:
                attributes[i//num_cols, i%num_cols, 1] = 0
            else:
                attributes[i//num_cols, i%num_cols, 1] = np.mean(1/np.asarray(nonpolars))

        if make_hist:
            # need to see if end up with empty set in polars or nonpolars to avoid
            # getting nan in histogram
            if len(polars)==0:
                hist_attr[i,0] = np.zeros((num_bins))
            else:
                hist_attr[i,0] = np.histogram(polars, bins=num_bins,range=(bin_min,bin_max), density=True)[0]
            if len(nonpolars)==0:
                hist_attr[i,1] = np.zeros((num_bins))
            else:
                hist_attr[i,1] = np.histogram(nonpolars, bins=num_bins, range=(bin_min,bin_max), density=True)[0]
        
        if max_dist > 0:
            if is_polar:
                # add one for itself
                polar_count_attr[i] = polar_count + 1
            else:
                polar_count_attr[i] = polar_count

    if make_hist:
        if make_mean_hist:
            hist_return = np.mean(hist_attr, axis=0)
            hist_return = hist_return.ravel()
        else:
            hist_return = hist_attr.ravel()
    else:
        hist_return=[]
        
    
    if polar_np_means:
        attributes = attributes.ravel()
        #if not make_hist:
        #    return np.asarray(attributes)
    
    if max_dist > 0:
        polar_count_attr = np.asarray(polar_count_attr.ravel())

    # return mean min dist array, hist pdf array and polar counts
    return np.asarray(attributes), np.asarray(hist_return), polar_count_attr


# polar body counts
def two_bodies(arr):
    """
    Counts number of two body (n_oo) in array, respecting periodic boundary condition. 

    Parameters
    ----------
    arr : numpy array
        Array containing 0/1 polarity values. 0=nonpolar, 1=polar

    Returns
    -------
    int : number of two body pairs in array

    """

    polar = 1

    # add top row to bottom and left column to right to handle periodic boundary conditions
    # but only count verticals with bottom row and horizontals with right most column or will be duplicating
    arr = np.concatenate((arr,[arr[0,:]]), axis=0)
    arr = np.concatenate((arr, arr[:,:1]), axis=1)

    # count
    num_rows, num_cols = arr.shape
    count = 0 

    for i in range(num_rows-1):
        for j in range(num_cols-1):
            if arr[i,j]==polar and arr[i,j+1]==polar:
                count += 1
            if arr[i,j]==polar and arr[i+1,j]==polar:
                count += 1
    
    return count

def three_straight_bodies(arr):
    """
    Counts number of three straight body (n_ooo) in array, respecting periodic boundary condition. 

    Parameters
    ----------
    arr : numpy array
        Array containing 0/1 polarity values. 0=nonpolar, 1=polar

    Returns
    -------
    int : number of three straight body polars in array

    """

    polar = 1

    # add top row to bottom and left 2 columns to right to handle periodic boundary conditions
    arr1 = np.concatenate((arr,[arr[0,:]]), axis=0)
    arr1 = np.concatenate((arr1, arr1[:,:2]), axis=1)

    num_rows, num_cols = arr1.shape
    count = 0 
    #print('new array:',arr1)
    for i in range(num_rows):
        for j in range(num_cols-1):
            if (arr1[i,j] == polar) and (arr1[i,j+1] == polar):
                if (j%2 == 0) :    #even
                    #print('Even. checking at i,j=('+str(i)+','+str(j)+'), (i,j+1)=('+str(i)+','+str(j+1)+')')
                    #print('considering arr1[i+1,j+2], arr1['+str(i+1)+','+str(j+1)+']=',arr1[i+1,j+1])
                    if (i+1 < num_rows) and (j+2<num_cols) and (arr1[i+1,j+2] == polar):
                        #print('found: at i,j=',i,',',j,'i,j+1=',i,',',j+1, 'and i+1,j+2=', i+1,',',j+2)
                        count += 1
                    if (i-1 >= 0) and (j-1>=0) and (arr1[i-1,j-1] == polar):
                        #print('found: at i,j=',i,',',j,'i,j+1=',i,',',j+1, 'and i-1,j-1=',i-1,',',j-1)
                        count += 1
                else:  #odd
                    #print('Odd. checking at i,j=('+str(i)+','+str(j)+'), (i,j+1)=('+str(i)+','+str(j+1)+')')
                    if (i+1 < num_rows) and (j-1 >= 0) and (arr1[i+1,j-1] == polar):
                        #print('found: at i,j=',i,',',j,'i,j+1=',i,',',j+1, 'and i+1,j-1=', i+1,',',j-1)
                        count += 1
                    if (i-1 >= 0) and (j+2<num_cols) and (arr1[i-1,j+2] == polar):
                        #print('found: at i,j=',i,',',j,'i,j+1=',i,',',j+1, 'and i-1,j21=', i-1,',',j+2)
                        count += 1

    # also need those in a straight column
    # add two rows from top to bottom
    arr2 = np.concatenate((arr, arr[:2,:]), axis=0)
    for i in range(arr2.shape[0]-2):
        for j in range(arr2.shape[1]):
            if (arr2[i,j] == polar) and (arr2[i+1,j] == polar) and (arr2[i+2,j] == polar):
                count += 1
                
    
    return count

def three_bent_bodies(arr):
    """
    Counts number of three bent body (n_ooo) in array, respecting periodic boundary condition. 

    Parameters
    ----------
    arr : numpy array
        Array containing 0/1 polarity values. 0=nonpolar, 1=polar

    Returns
    -------
    int : number of three bent body polars in array

    """

    polar = 1

    # add left 2 columns to right to handle periodic boundary conditions
    arr = np.concatenate((arr, arr[:,:2]), axis=1)

    num_rows, num_cols = arr.shape
    count = 0 
   
    for i in range(num_rows):
        for j in range(num_cols-2):
            if (arr[i,j] == polar) and (arr[i,j+1] == polar) and (arr[i,j+2] == polar):
                count += 1
            if j%2 == 0:
                if (i-1>=0) and (arr[i,j] == polar) and (arr[i-1,j+1] == polar) and (arr[i,j+2] == polar):
                    count += 1
            else:  #odd
                if (i+1<num_rows) and (arr[i,j] == polar) and (arr[i+1,j+1] == polar) and (arr[i,j+2] == polar):
                    count += 1
    
    return count

def three_compact_bodies(arr):
    """
    Counts number of three compact body (n_ooo) in array, respecting periodic boundary condition. 

    Parameters
    ----------
    arr : numpy array
        Array containing 0/1 polarity values. 0=nonpolar, 1=polar

    Returns
    -------
    int : number of three compact bodies in array

    """

    polar = 1

    # add top row to bottom and left column to right to handle periodic boundary conditions
    arr = np.concatenate((arr,[arr[0,:]]), axis=0)
    arr = np.concatenate((arr, arr[:,:1]), axis=1)

    # count
    num_rows, num_cols = arr.shape
    count = 0 

    for i in range(num_rows-1):
        for j in range(num_cols):
            if arr[i,j]==polar and arr[i+1,j]==polar:
                #print('found col at (i,j)=('+str(i)+','+str(j)+') and (i+1,j)=('+str(i+1)+','+str(j)+')')
                if j%2==0:
                    #print('even, checking i,j+1:('+str(i)+','+str(j+1)+')')
                    if (j+1<num_cols) and (arr[i,j+1]==polar):
                        count += 1
                    #print('even, checking i,j-1:('+str(i)+','+str(j-1)+')')
                    if (j-1>=0) and (arr[i,j-1]==polar):
                        count += 1    
                else:
                    #print('odd, checking i+1,j+1:('+str(i+1)+','+str(j+1)+')')
                    if (j+1<num_cols) and (arr[i+1,j+1]==polar):
                        count += 1
                    #print('odd, checking i+1,j-1:('+str(i+1)+','+str(j-1)+')')
                    if (j-1>=0) and (arr[i+1,j-1]==polar):
                        count += 1 
    
    return count
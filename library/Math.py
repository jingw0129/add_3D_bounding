import numpy as np
import math

# using this math: https://en.wikipedia.org/wiki/Rotation_matrix
def rotation_matrix(yaw, pitch=0, roll=0):
    tx = roll
    ty = yaw
    tz = pitch

    Rx = np.array([[1,0,0], [0, np.cos(tx), -np.sin(tx)], [0, np.sin(tx), np.cos(tx)]])
    Ry = np.array([[np.cos(ty), 0, np.sin(ty)], [0, 1, 0], [-np.sin(ty), 0, np.cos(ty)]])
    Rz = np.array([[np.cos(tz), -np.sin(tz), 0], [np.sin(tz), np.cos(tz), 0], [0,0,1]])


    return Ry.reshape([3,3])
    # return np.dot(np.dot(Rz,Ry), Rx)

# option to rotate and shift (for label info)
def create_corners(dimension, location=None, R=None):
    dx = dimension[2] / 2
    dy = dimension[0] / 2
    dz = dimension[1] / 2

    x_corners = []
    y_corners = []
    z_corners = []

    for i in [1, -1]:
        for j in [1,-1]:
            for k in [1,-1]:
                x_corners.append(dx*i)
                y_corners.append(dy*j)
                z_corners.append(dz*k)

    corners = [x_corners, y_corners, z_corners]

    # rotate if R is passed in

    if R is not None:
        corners = np.dot(R, corners)

    # shift if location is passed in
    if location is not None:
        for i,loc in enumerate(location):
            corners[i,:] = corners[i,:] + loc

    final_corners = []
    for i in range(8):
        final_corners.append([corners[0][i], corners[1][i], corners[2][i]])


    return final_corners

# this is based on the paper. Math!
# calib is a 3x4 matrix, box_2d is [(xmin, ymin), (xmax, ymax)]
# Math help: http://ywpkwon.github.io/pdf/bbox3d-study.pdf
def calc_location(dimension, proj_matrix, box_2d, alpha, theta_ray):
    #global orientation
    orient = alpha + theta_ray
    R = rotation_matrix(orient)

    # format 2d corners
    xmin = box_2d[0][0]
    ymin = box_2d[0][1]
    xmax = box_2d[1][0]
    ymax = box_2d[1][1]

    # left top right bottom
    box_corners = [xmin, ymin, xmax, ymax]

    # get the point constraints
    constraints = []

    left_constraints = []
    right_constraints = []
    top_constraints = []
    bottom_constraints = []

    # using a different coord system
    dx = dimension[2] / 2
    dy = dimension[0] / 2
    dz = dimension[1] / 2
    if math.isnan(dx) is False and math.isnan(dy) is False and math.isnan(dz) is False:

        # below is very much based on trial and error

        # based on the relative angle, a different configuration occurs
        # negative is back of car, positive is front
        left_mult = 1
        right_mult = -1

        # about straight on but opposite way
        if alpha < np.deg2rad(92) and alpha > np.deg2rad(88):
            left_mult = 1
            right_mult = 1
        # about straight on and same way
        elif alpha < np.deg2rad(-88) and alpha > np.deg2rad(-92):
            left_mult = -1
            right_mult = -1
        # this works but doesnt make much sense
        elif alpha < np.deg2rad(90) and alpha > -np.deg2rad(90):
            left_mult = -1
            right_mult = 1
        # if the car is facing the oppositeway, switch left and right
        switch_mult = -1
        if alpha > 0:
            switch_mult = 1

        # left and right could either be the front of the car ot the back of the car
        # careful to use left and right based on image, no of actual car's left and right
        for i in (-1,1):
            left_constraints.append([left_mult * dx, i*dy, -switch_mult * dz])
        for i in (-1,1):
            right_constraints.append([right_mult * dx, i*dy, switch_mult * dz])

        # top and bottom are easy, just the top and bottom of car
        for i in (-1,1):
            for j in (-1,1):
                top_constraints.append([i*dx, -dy, j*dz])
        for i in (-1,1):
            for j in (-1,1):
                bottom_constraints.append([i*dx, dy, j*dz])

        # now, 64 combinations
        for left in left_constraints:
            for top in top_constraints:
                for right in right_constraints:
                    for bottom in bottom_constraints:
                        constraints.append([left, top, right, bottom])
        # here is gonna  print 64
        # print('constraints', len(constraints),len(constraints[0]), len(constraints[0][0]))
        # filter out the ones with repeats
        constraints = filter(lambda x: len(x) == len(set(tuple(i) for i in x)), constraints)

        # create pre M (the term with I and the R*X)
        pre_M = np.zeros([4,4])
        # 1's down diagonal
        for i in range(0,4):
            pre_M[i][i] = 1

        best_loc = None
        best_error = [1e09]
        best_X = None

        # loop through each possible constraint, hold on to the best guess
        # constraint will be 64 sets of 4 corners
        count = 0
        for constraint in constraints:

            # each corner
            Xa = constraint[0]
            Xb = constraint[1]
            Xc = constraint[2]
            Xd = constraint[3]

            X_array = [Xa, Xb, Xc, Xd]
            # print('X_array in for loop',X_array)
            # M: all 1's down diagonal, and upper 3x1 is Rotation_matrix * [x, y, z]
            Ma = np.copy(pre_M)
            Mb = np.copy(pre_M)
            Mc = np.copy(pre_M)
            Md = np.copy(pre_M)
            M_array = [Ma, Mb, Mc, Md]

            # create A, b
            A = np.zeros([4,3], dtype=np.float)
            b = np.zeros([4,1])

            indicies = [0,1,0,1]
            for row, index in enumerate(indicies):
                # print('row and index',row, index)
                X = X_array[row]
                M = M_array[row]
                # print(M, M[:3,2])
                # print(M[:3, 3])
                # create M for corner Xx
                RX = np.dot(R, X)
                M[:3,3] = RX.reshape(3)
                # print(M[:3,3])
                M = np.dot(proj_matrix, M)
                A[row, :] = M[index,:3] - box_corners[row] * M[2,:3]
                b[row] = box_corners[row] * M[2,3] - M[index,3]

            # solve here with least squares, since over fit will get some error
            loc, error, rank, s = np.linalg.lstsq(A, b, rcond=None)
            # print('least square', loc, error, rank, s)
            # found a better estimation
            if error < best_error:
                count += 1 # for debugging
                best_loc = loc
                best_error = error
                best_X = X_array
        # print(best_loc[0][0], best_loc[1][0], best_loc[2][0])
        # return best_loc, [left_constraints, right_constraints] # for debugging
        best_loc = [best_loc[0][0], best_loc[1][0], best_loc[2][0]]
        # print('bestloc: ',best_loc, 'best_x: ',best_X)
        return best_loc, best_X

# # Instrincs of the camera
# K = np.array([0,637.85,-61.51,22.2,-42.45,27.73])
# k1 = K[1]
# k3 = K[2]
# k5 = K[3]
# k7 = K[4]
# k9 = K[5]
# Cx = 648.33
# Cy = 357.42
# centre = np.array([Cx,Cy])
# asp = 1
# def power(x, n): #如def power (x,n=2) 设置了n的默认值为2
#     s = 1
#     while n > 0:
#         n = n - 1
#         s = s * x
#     return s
# for gr_i in range(1280):
#     for gr_j in range(720):
#         u = gr_i - 720 / 2
#         v = gr_j - 1280 / 2
#         phi = np.arctan2(v, u)
#         r = np.sqrt(pow(u, 2) + pow(v, 2))
#         thita1 = math.pi / 2
#         R1 = (k1) * thita1 + (k3) * power(thita1, 3) + (k5) * power(thita1, 5) + (k7) * power(thita1, 7) + (k9) * power(
#             thita1, 9)
#         f = R1 // thita1
# proj_matrix[0][0] = f

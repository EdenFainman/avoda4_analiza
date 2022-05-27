import sympy as sp

x = sp.symbols('x')


def calc_av(A, b):
    inverseMat = inverse(A)
    av = mul_matrix_wVector(inverseMat, b)
    return av


def identity_matrix(size):
    I = list(range(size))
    for i in range(size):
        I[i] = list(range(size))
        for j in range(size):
            if i == j:
                I[i][j] = 1
            else:
                I[i][j] = 0
    return I


def mul_matrix(m1, m2):
    len_m1 = len(m1)
    cols_m1 = len(m1[0])
    rows_m2 = len(m2)
    if cols_m1 != rows_m2:  # Checks if it is valid to multiply between matrix
        print("Cannot multiply between matrix (incorrect size)")
        return
    new_mat = list(range(len_m1))
    val = 0
    for i in range(len_m1):
        new_mat[i] = list(range(rows_m2))
        for j in range(len(m2[0])):
            for k in range(cols_m1):
                val += m1[i][k] * m2[k][j]
            new_mat[i][j] = val
            val = 0
    return new_mat


def inverse(mat):
    size = len(mat)
    invert_mat = identity_matrix(size)
    for col in range(size):
        elem_mat = identity_matrix(size)
        max_row = max_val_index(mat, col)  # Returns the index of the row with the maximum value in the column
        invert_mat = mul_matrix(eMatForSwap(size, col, max_row), invert_mat)  # Elementary matrix for swap rows
        mat = mul_matrix(eMatForSwap(size, col, max_row), mat)  # swap between rows in case the pivot is 0
        pivot = mat[col][col]
        for row in range(size):
            if row != col and mat[row][col] != 0:
                elem_mat[row][col] = (-1) * (mat[row][col] / pivot)
        mat = mul_matrix(elem_mat, mat)
        invert_mat = mul_matrix(elem_mat, invert_mat)
    # check diagonal numbers
    for i in range(size):
        pivot = mat[i][i]
        if pivot != 1:
            for col in range(size):
                invert_mat[i][col] /= float(pivot)
            mat[i][i] = 1
    return invert_mat


def mul_matrix_wVector(m, v):
    len_m = len(m)
    cols_m = len(m[0])
    rows_v = len(v)
    if cols_m != rows_v:  # Checks if it is valid to multiply between matrix
        print("Cannot multiply between matrix (incorrect size)")
        return
    new_mat = list(range(len_m))
    val = 0
    for i in range(len_m):
        for k in range(len(m[0])):
            val += m[i][k] * v[k]
        new_mat[i] = val
        val = 0
    return new_mat


def max_val_index(mat, col):
    max = abs(mat[col][col])
    index = col
    for row in range(col, len(mat)):
        if abs(mat[row][col]) > max:
            max = abs(mat[row][col])
            index = row
    return index


def eMatForSwap(size, index1, index2):
    mat = identity_matrix(size)
    # swap rows
    tmp = mat[index1]
    mat[index1] = mat[index2]
    mat[index2] = tmp
    return mat


def linear_interpolation(plist, x):
    x1 = 0
    x2 = 0
    y1 = 0
    y2 = 0
    for i in range(len(plist)):
        if plist[i][0] < x < plist[i + 1][0]:
            x1 = plist[i][0]
            y1 = plist[i][1]
            x2 = plist[i + 1][0]
            y2 = plist[i + 1][1]
    fx = ((y1 - y2) / (x1 - x2)) * x + ((y2 * x1 - y1 * x2) / (x1 - x2))
    return fx


def polynomial_interpolation(plist, x):
    n = len(plist)
    A = [[0 for i in range(n)] for j in range(n)]
    for i in range(n):
        for j in range(n):
            A[i][j] = plist[i][0] ** j
    b = [0 for i in range(n)]
    for i in range(n):
        b[i] = plist[i][1]
    a = calc_av(A, b)
    p = 0

    for i in range(n):
        p = p + a[i] * x ** i
    return p


def lagrange_interpolation(pointlist, x):
    yvp = 0

    for i in range(len(pointlist)):
        p = 1
        for j in range(len(pointlist)):
            if i != j:
                p = p * (x - pointlist[j][0]) / (pointlist[i][0] - pointlist[j][0])
        yvp = yvp + p * pointlist[i][1]

    return yvp


def neville_interpolation(plist, xf):
    n = len(plist)
    res = 0
    for m in range(1, n):
        for k in range(n - 1, m - 1, -1):
            plist[k][1] = ((xf - plist[k - m][0]) * plist[k][1] - (xf - plist[k][0]) *
                               plist[k - 1][1]) / (plist[k][0] - plist[k - m][0])
    res = plist[n - 1][1]
    return res



def main():
    x = input("please enter the point you want to calculate")
    x = float(x)
    list_inter = [[0, 0], [1, 0.8415], [2, 0.9093], [3, 0.1411], [4, -0.7568], [5, -0.9589],
                  [6, -0.2794]]  # table for linear interpolation

    fx = linear_interpolation(list_inter, x)
    print("The value of the point", x, "by linear interpolation method is: %.4f" % fx)
    po_inter_t = [[1, 0.8415], [2, 0.9093], [3, 0.1411]]  # table for polynomial interpolation
    fx = polynomial_interpolation(po_inter_t, x)
    print("The value of the point", x, "by linear interpolation method is: %.4f" % fx)

    la_inter_t = [[1, 1], [2, 0], [4, 1.5]]  # table for lagrange interpolation
    fx = lagrange_interpolation(list_inter, x)
    print("The value of the point", x, "by lagrange interpolation method is: %.4f" % fx)

    ne_inter_t = [[1, 0.7651], [1.3, 0.6200], [1.6, 0.4554], [1.9, 0.2818],
                  [2.2, 0.1103]]  # table for neville interpolation
    fx = neville_interpolation(list_inter, x)
    print("The value of the point", x, "by neville interpolation method is: %.4f" % fx)


main()

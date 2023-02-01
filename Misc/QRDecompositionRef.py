import pprint
import scipy
import scipy.linalg

A = scipy.array([[1, 2, -1, 0], [2, 1, 1, 1], [-1, -2, 1, 1],[0, 1, 1,2]]);
Q, R = scipy.linalg.qr(A);

print("A:")
pprint.pprint(A);

print("Q:")
pprint.pprint(Q);

print("R:")
pprint.pprint(R);

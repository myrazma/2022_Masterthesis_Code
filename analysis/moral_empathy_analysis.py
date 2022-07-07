import pickle
import sklearn
print (sklearn.__version__)
from sklearn.decomposition import PCA



file_path = '../data/MoRT_projection/projection_model.p'
file = open(file_path, 'rb')

# dump information to that file
data = pickle.load(file)

# close the file
file.close()

print('Showing the pickled data:')

cnt = 0
for item in data:
    print('The data ', cnt, ' is : ', item)
    print(type(item))
    print()
    cnt += 1

print(type(data))
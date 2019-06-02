import csv
import numpy as np
import matplotlib.pyplot as plt

def initial_weight(n, m):
   return np.random.rand(n, m) 

def normalize(x):
    result =[]
    for i in range(len(x)):
        tmp = x[i][:]
        tmp = normalize_row(tmp)
        result.append(tmp)
    result = np.asarray(result)
    return result

def normalize_row(x):
    norm = np.sqrt(np.dot(x, x.T))
    return x / norm

def distance(x, w):
    diff = (w - x)
    square = np.sum(np.square(diff), axis=1)
    return np.squeeze(np.sqrt(square))

def update_wta(w, x):
    w = w + 0.1 * (x - w)
    return w

# w = normalize(w)
x_axis = []
y_axis = []
w = []

num_clusters = 5

# initial weight
with open("iris.data") as csv_file:
    csv_reader = csv.reader(csv_file, delimiter=',')
    counter = 0
    for row in csv_reader:
        x = [float(num) for num in row[:4]]
        w.append(x)
        counter += 1
        if counter == num_clusters:
            w = np.asarray(w)
            break

num_epochs = 20

for i in range(num_epochs):
    if i == num_epochs - 1:
        clusters = []
        labels = []

    with open("iris.data") as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        for row in csv_reader:
            if len(row) > 0:
                x = [float(num) for num in row[:4]]
                x = np.asarray(x)

                if i == num_epochs - 1:
                    x_axis.append(x[2])
                    y_axis.append(x[3])

                x = np.resize(x, (1, 4))
                # x = normalize_row(x)
                distances = distance(x, w)
                winner_index = np.argmin(distances, axis=0)
                
                if i == num_epochs - 1:
                    clusters.append(winner_index)
                    labels.append(row[4])

                result = []
                winner = w[winner_index][:]
                winner = update_wta(winner, x)
                # print("winner index = ", winner_index)

                for k in range(winner_index):
                    result.append(w[k][:])
                result.append(np.squeeze(winner))
                for k in range(winner_index+1, len(w)):
                    result.append(w[k][:])
                
                w = np.asarray(result)
    
    


# print("Final weight: ", w, "\n\n")
# print("clusters: ", len(clusters), len(x_axis), len(y_axis))

x_axis = np.asarray(x_axis)
y_axis = np.asarray(y_axis)
# x_weight = np.asarray(w[:, 2])
# y_weight = np.asarray(w[:, 3])

plt.grid(color='black', linestyle='-', linewidth=0.1)

colors = ['blue', 'green', 'red', 'purple', 'pink']
cluster_numbers = []

for j in range(num_clusters):
    x_cluster = []
    y_cluster = []
    decision_classes = {}
    for i in range(len(x_axis)):
        if clusters[i] == j:
            x_cluster.append(x_axis[i])
            y_cluster.append(y_axis[i])
            if labels[i] not in decision_classes:
                decision_classes[labels[i]] = 1
            else:
                decision_classes[labels[i]] += 1

    # print('Decision classes: ', decision_classes)
    dominated_class = ''
    max_num_items = 0
    total_items = 0
    for k, v in decision_classes.items():
        total_items += v
        if v > max_num_items:
            max_num_items = v
            dominated_class = k
    print('Dominated class: ', dominated_class, " - fraction: ", max_num_items/total_items)

    x_cluster = np.array(x_cluster)
    y_cluster = np.array(y_cluster)

    cluster_name = 'Cluster ' + str(j+1)
    cluster = plt.scatter(x_cluster, y_cluster, c=colors[j], marker='o', label=cluster_name)
    cluster_numbers.append(cluster)

x_weight = np.asarray(w[:, 2])
y_weight = np.asarray(w[:, 3])
neurons = plt.scatter(x_weight, y_weight, c='orange', marker='^', label='Neurons')

if num_clusters == 5:
    plt.legend(handles=[neurons, cluster_numbers[0], cluster_numbers[1], cluster_numbers[2], cluster_numbers[3], cluster_numbers[4] ])
elif num_clusters == 4:
    plt.legend(handles=[neurons, cluster_numbers[0], cluster_numbers[1], cluster_numbers[2], cluster_numbers[3] ])
elif num_clusters == 3:
    plt.legend(handles=[neurons, cluster_numbers[0], cluster_numbers[1], cluster_numbers[2]])

plt.title('Iris - Self Organizing Map')
plt.xlabel('Petal length in cm')
plt.ylabel('Petal width in cm')

plt.show()



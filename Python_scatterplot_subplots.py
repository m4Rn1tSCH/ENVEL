#Scatterplot with subplots
#Plot the kmeans labels
cl_cen = pd.DataFrame(kmeans.cluster_centers_, index = None)
print(cl_cen)

data = (cl_cen)
colors = ("red", "green", "blue", "yellow", "black")
groups = ("Cluster_1", "Cluster_2", "Cluster_3", "Cluster_4", "Cluster_5")

# Create plot
fig = plt.figure()
ax = fig.add_subplot(1, 1, 1)

for data, color, group in zip(data, colors, groups):
    x, y = data
    ax.scatter(x, y, alpha=0.8, c=color, edgecolors='none', s=30, label=group)

plt.title('Matplot scatter plot')
plt.legend(loc=2)
plt.show()

def show_sample_mnist(d,c):
    perm = np.random.permutation(range(d.shape[0]))[0:50]
    random_imgs   = d[perm]
    random_labels = c[perm] 
    fig = plt.figure(figsize=(10,6))
    for i in range(random_imgs.shape[0]):
        ax=fig.add_subplot(5,10,i+1)
        plt.imshow(random_imgs[i].reshape(28,28), interpolation="nearest", cmap = plt.cm.Greys_r)
        ax.set_title(int(random_labels[i]))
        ax.set_xticklabels([])
        ax.set_yticklabels([])

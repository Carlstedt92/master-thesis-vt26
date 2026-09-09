import matplotlib.pyplot as plt
import matplotlib.image as mpimg

image_name = "loss_curves_ssl_knn_lipo.png"
model_names = ["GAT_MASK_9M_1", "GAT_MASK_9M_2", "GAT_MASK_9M_3",
                "GINE_MASK_9M_1", "GINE_MASK_9M_2", "GINE_MASK_9M_3"]
images = []

for i in range(len(model_names)):
    model_name = model_names[i]
    path = f"models/{model_name}/{image_name}"
    img = mpimg.imread(path)
    images.append(img)

fig, axes = plt.subplots(2, 3, figsize=(15, 10))
for i, ax in enumerate(axes.flat):
    ax.imshow(images[i])
    ax.set_title(model_names[i])
    ax.axis('off')

plt.tight_layout()
plt.savefig("loss_curves_ssl_knn_lipo.png")
plt.show()
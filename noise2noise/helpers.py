# AUTOGENERATED! DO NOT EDIT! File to edit: 00_helpers.ipynb (unless otherwise specified).

__all__ = ['show_img']

# Cell

def show_img(tensor_img):
    plt.imshow(tensor_img.permute(1, 2, 0) )
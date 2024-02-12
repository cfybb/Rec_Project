from data.pretrain import generate_gaussian_heatmap


def decode_gaussian_heatmap(heatmap, scale_factor, sigma):

    # pixel level maximum
    height, width = heatmap.shape
    indices = heatmap.argmax()
    Y = indices // width
    X = indices % width

    # subpixel level maximum
    tl = (max(X - 3 * sigma, 0), max(Y - 3 * sigma, 0))
    br = (min(X + 3 * sigma, width - 1), min(Y + 3 * sigma, height - 1))
    sub_heatmap = heatmap[tl[1]:(br[1] + 1), tl[0]:(br[0] + 1)]





if __name__ == "__main__":
    image_shape = (1280, 720)
    x, y = 302, 502
    scale_factor = 4
    sigma = 2
    heatmap = generate_gaussian_heatmap(x, y, image_shape, scale_factor, sigma)
    x, y = decode_gaussian_heatmap(heatmap, scale_factor, sigma)

import numpy as np
import cv2
from edge_padding import edge_padding

I_WIDTH = 2048
I_HEIGHT = 2048

def test_edge_padding_custom_mask(input_filename, output_filename):

    img = cv2.imread(input_filename, cv2.IMREAD_UNCHANGED)
    if img is None:
        print("Failed to load image")
        return

    if img.ndim == 3 and img.shape[2] == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2BGRA)

    img = cv2.resize(img, (I_WIDTH, I_HEIGHT))
    mask = np.zeros(img.shape[:2], np.uint8)

    # triangle = np.array([(512, 512), (1536, 512), (1024, 1536)])
    triangle = np.zeros((3, 2), np.int32)
    triangle[:, 0] = np.random.randint(0, I_WIDTH, (3), np.int32)
    triangle[:, 1] = np.random.randint(0, I_HEIGHT, (3), np.int32)
    #
    cv2.fillConvexPoly(mask, triangle, 255)

    img = edge_padding.edge_padding_uint8_custom_mask(img, mask)

    show_img = cv2.resize(img, (1024, 1024))
    cv2.imshow("img", show_img)
    cv2.waitKey(0)

    # cv2.imwrite(output_filename, img)


def test_edge_padding(input_filename, output_filename):
    img = cv2.imread(input_filename, cv2.IMREAD_UNCHANGED)
    if img is None:
        print("Failed to load image")
        return

    if img.ndim == 3 and img.shape[2] == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2BGRA)

    img = cv2.resize(img, (I_WIDTH, I_HEIGHT))
    mask = np.zeros(img.shape[:2], np.uint8)

    triangle = np.array([(512, 512), (1536, 512), (1024, 1536)])
    # triangle = np.zeros((3, 2), np.int32)
    # triangle[:, 0] = np.random.randint(0, I_WIDTH, (3), np.int32)
    # triangle[:, 1] = np.random.randint(0, I_HEIGHT, (3), np.int32)

    cv2.fillConvexPoly(mask, triangle, 255)

    img[mask == 0] = 0
    img = edge_padding.edge_padding_uint8(img)

    show_img = cv2.resize(img, (1024, 1024))
    cv2.imshow("img", show_img)
    cv2.waitKey(0)

    # cv2.imwrite(output_filename, img)

if __name__ == '__main__':
    input_filename = r"D:\Assets\EdgePadding\pve01_xht_chexiang01_01_d_2.png"
    output_filename = r"D:\Assets\EdgePadding\pve01_xht_chexiang01_01_d_3.png"

    # test_edge_padding_custom_mask(input_filename, output_filename)
    test_edge_padding(input_filename, output_filename)
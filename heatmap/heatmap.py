import pandas as pd
import cv2
import numpy as np
from PIL import Image
from heatmappy import Heatmapper

# img = cv2.imread('C:/Users/seobosung/Desktop/hackathon/data/picture/map.png')

example_img_path = './data/picture/map_topview_hackathon_5.png'
position_path = '../hackathon_5.xlsx'
save_filename = 'hackathon_5'


example_img = Image.open(example_img_path)

df = pd.read_excel(position_path, usecols=[0, 1, 2])
df_list = df.values.tolist()

def id_heatmap(df_list):
    id_list = []
    for i in df_list:
        id_list.append(i[2])
    id_list = set(id_list)
    id_list = list(id_list)
    if len(id_list)==1:
        print("아이디 하나만 발견 및 종료")
    else:
        total_list = []
        for i in id_list:
            i_list = []
            for j in df_list:
                if i == j[2]:
                    tuple_pose = (j[0], j[1])
                    i_list.append(tuple_pose)

            heatmapper = Heatmapper(
                point_diameter=70,  # the size of each point to be drawn
                point_strength=0.1,  # the strength, between 0 and 1, of each point to be drawn
                opacity=0.6,  # the opacity of the heatmap layer
                colours='default',  # 'default' or 'reveal'
                                    # OR a matplotlib LinearSegmentedColorMap object
                                    # OR the path to a horizontal scale image
                grey_heatmapper='PIL'  # The object responsible for drawing the points
                                    # Pillow used by default, 'PySide' option available if installed
            )

            heatmap = heatmapper.heatmap_on_img(i_list, example_img)
            heatmap.save(f'id_{i}.png')
            print(f'{save_filename}_id_{int(i)} done!')

    # print(total_list)


# print(df_list)

def total_heatmap(df_list):
    total_tuple = []
    for i in df_list:
        # print("i : ",i[0],i[1])
        a = tuple([i[0],i[1]])
        total_tuple.append(a)
    # print(total_tuple)

    heatmapper = Heatmapper(
        point_diameter=70,  # the size of each point to be drawn
        point_strength=0.1,  # the strength, between 0 and 1, of each point to be drawn
        opacity=0.6,  # the opacity of the heatmap layer
        colours='default',  # 'default' or 'reveal'
                            # OR a matplotlib LinearSegmentedColorMap object
                            # OR the path to a horizontal scale image
        grey_heatmapper='PIL'  # The object responsible for drawing the points
                            # Pillow used by default, 'PySide' option available if installed
    )

    heatmap = heatmapper.heatmap_on_img(total_tuple, example_img)
    heatmap.save(f'total_{save_filename}_heatmap.png')
    print("total heamap done")

id_heatmap(df_list)
total_heatmap(df_list)
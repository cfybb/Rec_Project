import os
import cv2
import numpy as np
folder_path_FG = "/Users/shuangliu/Downloads/data/MPIIFaceGaze"
folder_path_G = "/Users/shuangliu/Downloads/data/MPIIGaze\Annotation Subset"
#保存annotation 地址：
# save_file_path = "C:\prdue\job_preperation_general\support_company\project\MPIIFaceGaze"
# save_file_name = "annotationOverall.txt"
save_file_path_FG = "/MPIIFaceGaze/annotationOverall.txt"
'''
#遍历Gaze
for i in range(15):
    G_filename = f'p{i:02}.txt'
    file_path_G = os.path.join(folder_path_G,G_filename)
    with open(file_path_G,'r') as file:
        lines_all = file.readlines()

    FG_filename = f'p{i:02}\\p{i:02}.txt'
    file_path_FG = os.path.join(folder_path_FG,FG_filename)
    with open(file_path_FG,'r') as file:
        lines_FG_all = file.readlines()



    #遍历FG
    for p in range(len(lines_FG_all)):
        #print("下一个文件")
        check = 0
        for k in range(len(lines_all)):

            #这个是对应Pxx txt文件中的每行

            line_list = lines_all[k].split()
            #print(lines_all) #确定输出正确
            line_list_pre_FG = lines_FG_all[p].split()

            line_list_FG = [line_list_pre_FG[0]]
            line_list_FG.extend(j for j in line_list_pre_FG[3:15])
            #print("line_list_FG:",line_list_FG)  #check
            #start to deal with the data.
            if line_list_FG[0] == line_list[0]:
                #print("I am here!")
                line_list_FG.extend(line_list[-4:])
                check = 1
                break
            else:
                continue

        if check == 0:
            line_list_FG.extend(['-1','-1','-1','-1'])
        #存
        #FG_filename_new = f'P{i:02}\\P{i:02}_new.txt'
        #file_path_FG_new = os.path.join(folder_path_FG,FG_filename_new)
        #最后处理FG annotation的第一位位置
        line_list_FG[0] = f'p{i:02}/'+line_list_FG[0]
        line_list_FG_str = ' '.join(line_list_FG)

        with open(save_file_path_FG,'a') as file:
            file.write('\n'+line_list_FG_str)
'''
for w in range(15):
    #FG_filename_plt = f'p{w:02}\\p{w:02}_new.txt'
    #FG_addname_plt = f'p{w:02}'
    #file_path_FG_plt = os.path.join(folder_path_FG,FG_filename_plt)
    #一个文件就不用到处找文件在哪了。
    with open(save_file_path_FG,'r') as file:
        lines_FG_all_plt = file.readlines()
    for o in range(len(lines_FG_all_plt)):
        #this is hard code, will check this again
        if o == 0:
            continue
        line_list_FG_plt = lines_FG_all_plt[o].split()
        #print(line_list_FG_plt)
        plot_dir = os.path.join(folder_path_FG,line_list_FG_plt[0])
        image = cv2.imread(plot_dir)
        points = line_list_FG_plt[1:]
        #print(len(points))
        point_pairs = [(points[i], points[i + 1]) for i in range(0, len(points), 2)]
        for point_pair in point_pairs:
            x, y = point_pair
            #print(type(x),type(y))
            cv2.circle(image, (int(x), int(y)), 5, (0, 255, 0), -1)
        # new_plt_dir = line_list_FG_plt[0]+"_new.jpg"
        # output_image_path = os.path.join(folder_path_FG,new_plt_dir)
        # cv2.imwrite(output_image_path,image)
        cv2.imshow("", image)
        key = cv2.waitKey(-1)
        if key == ord("q"):
            break


        #cv2.imshow("Original Image", cv2.imread(plot_dir))
        #cv2.imshow("Image with Points", image)
        #cv2.waitKey(0)
        #cv2.destroyAllWindows()
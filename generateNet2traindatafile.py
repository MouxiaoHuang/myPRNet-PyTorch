import os
import glob

trainset_file = 'Data/Net2traindata/trainsetFile.txt'

p = 'C:/Users/MSI/Desktop/300WLP'
sub_path_list = os.listdir(p)
img_path_list = []

for item in sub_path_list:
    sub_path = os.path.join(p, item, '*.jpg')
    img_path = glob.glob(sub_path)
    for i in img_path:
        i = i.replace('\\', '/')
        img_path_list.append(i)


modify_trainset_file_list = []
with open('Data/300wlp_all.txt') as fp:
    temp = fp.readlines()
    for item in temp:
        item = item.strip().split()
        if(len(item)==4):
            item[0] = item[0] + ' ' + item[1]
            item[1] = item[2] + ' ' + item[3]
            del item[2:]
        modify_trainset_file_list.append(item)
print(len(modify_trainset_file_list))
print(len(img_path_list))


f = open(trainset_file, 'w')
for i in range(len(img_path_list)):
    #save_img_path = img_path_list[i]
    #save_mat_path = img_path_list[i].replace('.jpg', '.mat')
    #f.writelines(save_img_path + ' ' + save_mat_path + '\n')
    save_img_path = modify_trainset_file_list[i][0]
    save_mat_path = img_path_list[i].replace('.jpg', '.mat')
    f.writelines(save_img_path + ' ' + save_mat_path + '\n')
f.close()


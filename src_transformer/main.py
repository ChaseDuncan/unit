import torch
import torch.autograd as autograd
import math
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Transformer
from torch.nn import TransformerEncoder, TransformerEncoderLayer
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from VIT import ViT
from UNIT import UNIT

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def chunk_slice(chunk_x, chunk_y, image, segmented_image):
    assert(image.shape[0] == segmented_image.shape[0])
    assert(image.shape[1] == segmented_image.shape[1])

    columns = int(image.shape[0] / chunk_x)
    rows = int(image.shape[1] / chunk_y)
 
    chunked_image = np.zeros((columns * rows, chunk_x, chunk_y, image.shape[2]), dtype=np.float32)
    chunked_segmented_image = np.zeros((columns * rows, chunk_x, chunk_y), dtype=np.float32)
    has_tumor = np.zeros((columns * rows,), dtype=np.float32)

    chunked_image_arr = []
    chunked_segmented_image_arr = []
    has_tumor_arr = []

    temp_vector = np.zeros((chunk_x, chunk_y, image.shape[2]))

    for i in range(0, (columns-1)):
        for j in range(0, (rows-1)):
            '''
            if np.all((segmented_image[i * chunk_x:(chunk_x + i*chunk_x), j*chunk_y:(chunk_y + j*chunk_y)] == 0)):
                has_tumor_arr.append(0)
            else:
                has_tumor_arr.append(1)

            temp_vector = np.copy(image[i * chunk_x:(chunk_x + i*chunk_x), j*chunk_y:(chunk_y + j*chunk_y), :]) 
            
            chunked_image_arr.append(temp_vector)

            temp_vector = np.copy(segmented_image[i * chunk_x:(chunk_x + i*chunk_x), j*chunk_y:(chunk_y + j*chunk_y)])

            chunked_segmented_image_arr.append(temp_vector)
            '''
            
            if np.all((segmented_image[i * chunk_x:(chunk_x + i*chunk_x), j*chunk_y:(chunk_y + j*chunk_y)] == 0)):
                pass
                #has_tumor_arr.append(0)
                #temp_vector = np.copy(image[i * chunk_x:(chunk_x + i*chunk_x), j*chunk_y:(chunk_y + j*chunk_y), :]) 
            
                #chunked_image_arr.append(temp_vector)

                #temp_vector = np.copy(segmented_image[i * chunk_x:(chunk_x + i*chunk_x), j*chunk_y:(chunk_y + j*chunk_y)])

                #chunked_segmented_image_arr.append(temp_vector)
            else:
                has_tumor_arr.append(1)
                temp_vector = np.copy(image[i * chunk_x:(chunk_x + i*chunk_x), j*chunk_y:(chunk_y + j*chunk_y), :]) 
            
                chunked_image_arr.append(temp_vector)

                temp_vector = np.copy(segmented_image[i * chunk_x:(chunk_x + i*chunk_x), j*chunk_y:(chunk_y + j*chunk_y)])

                chunked_segmented_image_arr.append(temp_vector)

            

    return (chunked_image_arr, chunked_segmented_image_arr, has_tumor_arr)


def load_data(range_min, range_max):
    chunk_x = 40 #240#
    chunk_y = 40 # 155#

    chunked_images_arr = []
    chunked_segmented_images_arr = []
    images_have_tumor_arr = []


    for i in range (range_min, range_max):
        index = str(i)
        if len(index) == 1:
            index = '00' + index
        elif len(index) == 2:
            index = '0' + index
        
        data_filename = './dataset/brats2020-training/crop/BraTS20_Training_' + index + '_imgs.npy'
        segmented_filename = './dataset/brats2020-training/crop/BraTS20_Training_' + index + '_seg.npy'
        
        data = np.load(data_filename)
        segmented = np.load(segmented_filename)

        data = data.transpose((2, 1, 3, 0))
        segmented = segmented.transpose((1, 0, 2))
        segmented = np.clip(segmented, 0.0, 1.0)


        for j in range(0, data.shape[2]):

            #if you want to chunk use this
            chunked_image, chunked_segmented_image, image_has_tumor = chunk_slice(chunk_x, chunk_y, data[:, :, j], segmented[:, :, j])
            #chunked_image = [data[:,:,j]]
            #chunked_segmented_image = [segmented[:,:,j]]
            
            #print(data[:,:,j].shape)
            #plt.imshow(data[:,:,100, 1:4])
            #plt.show()
            #chunked_image = data[:, :, j]
            #chunked_segmented_image = segmented[:,:,j]
            #print(chunked_image.shape)
            #print(chunked_segmented_image.shape)

            chunked_images_arr = chunked_images_arr + chunked_image
            chunked_segmented_images_arr = chunked_segmented_images_arr + chunked_segmented_image
            images_have_tumor_arr = images_have_tumor_arr + image_has_tumor
            #chunked_images = np.vstack((chunked_images, chunked_image))
            #chunked_segmented_images = np.vstack((chunked_segmented_images, chunked_segmented_image))

            #print(images_have_tumor.shape)
            #print(image_has_tumor.shape)
            #images_have_tumor = np.concatenate((images_have_tumor, image_has_tumor), axis=0)

        #chunked_segmented_images = 

        print('loaded file' + data_filename)
    

    print("copying from list to array")
    chunked_images = np.zeros((len(chunked_images_arr), chunk_x, chunk_y, 4), dtype=np.float32)
    chunked_segmented_images = np.zeros((len(chunked_images_arr), chunk_x, chunk_y), dtype=np.float32)
    images_have_tumor = np.zeros((len(chunked_images_arr),), dtype=np.float32)

    for i in range(0, len(chunked_images_arr)):
        chunked_images[i] = chunked_images_arr[i]
        chunked_segmented_images[i] = chunked_segmented_images_arr[i]
        #images_have_tumor[i] = images_have_tumor_arr[i]
    print('finished copying')

    # i = 0
    # while True: 
    #     while(images_have_tumor[i] == 0.0):
    #         i = i + 1 
    #     print(i)
    #     #print(images_have_tumor[i])
    #     #print(images_have_tumor[i])

    #     f, ax = plt.subplots(2, 2)
    #     ax[0, 0].imshow(segmented[:, :, 40])
    #     ax[0, 1].imshow(data[:, :, 100, 1:4])
    #     ax[1, 0].imshow(chunked_segmented_images[i, :, :])
    #     ax[1, 1].imshow(chunked_images[i, :, :, 1:4])
    #     plt.show()
    #     i = i + 1 


    #f, ax = plt.subplots(2, 2)
    ##ax[0,0].imshow(data[:, :, 30, 0])
    #ax[0, 0].imshow(segmented[:, :, 100])
    #ax[0, 1].imshow(data[:, :, 100, 1:4])
    #ax[1, 0].imshow(chunked_image[50, :, :, 1:4])
    #ax[1, 1].imshow(chunked_image[60, :, :, 1:4])
    # plt.show()
    
    return (chunked_images, chunked_segmented_images, images_have_tumor)

# 0 goes to [1, 0], 1 goes to [0, 1]


def to_one_hot_2d(n):
    result = np.array((2,))
    if n == 0.0:
        result = np.array([1, 0])
    else:
        result = np.array([0, 1])

    result = torch.from_numpy(result)

    return result


# def train():
#     chunked_images, chunked_segmented_images, images_have_tumors = load_data()

#     model = ViT(image_size=30, patch_size=5, num_classes=2, dim=20, depth=10, heads=10, mlp_dim=10, channels=4)

#     criterion = nn.CrossEntropyLoss()
#     optimizer = optim.SGD(model.parameters(), lr=0.001)

#     for epoch in range(0, 700):
#         running_loss = 0.0
        
#         model_input = np.copy(chunked_images)
#         model_output = torch.LongTensor(images_have_tumors)
        
#         model_input = model_input.transpose((0, 3, 1, 2))
#         model_input = torch.from_numpy(model_input)

#         outputs = model(model_input)

#         loss = criterion(outputs, model_output)
#         loss.backward()
#         optimizer.step()

#         running_loss += loss.item()

#         print('[%d] loss: %.3f' % (epoch + 1, running_loss))


def train_unit():
    chunked_images, chunked_segmented_images, images_have_tumor = load_data(12, 15)

    model = UNIT(image_size=40, patch_size=5, dim=30, depth=10, heads=30, mlp_dim=100, channels=4).to(device)

    #criterion = nn.MSELoss()
    #criterion = nn.CrossEntropyLoss()
    criterion = nn.CrossEntropyLoss()
    #optimizer = optim.Adam(model.parameters(), lr=0.001)
    optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.1)
    for epoch in range(0, 10000):
        running_loss = 0.0
        

        total_size = chunked_images.shape[0]
        frac = int(total_size / 10)
        current_index = epoch % 10
        

        model_input = np.copy(chunked_images[current_index*frac:((1 + current_index)*frac)])
        model_output = chunked_segmented_images[current_index*frac:((1 + current_index)*frac)]
        
        model_input= np.copy(chunked_images)
        model_output = np.copy(chunked_segmented_images)

        model_input = model_input.transpose((0, 3, 1, 2))
        model_input = torch.from_numpy(model_input).to(device)

        model_output = model_output.transpose((0, 1, 2))
        model_output = torch.LongTensor(model_output.astype(np.long)).to(device)

        outputs = model(model_input)

        #print(outputs.shape)
        #print(model_output.shape)
        #print(torch.max(model_output))
        loss = criterion(outputs, model_output)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

        print('[%d] loss: %.3f' % (epoch + 1, running_loss))

        #if(epoch % 4000 == 3900):
        #    f, ax = plt.subplots(1, 2)
        #    ax[0].imshow(outputs.cpu().detach().numpy()[23, :, :], cmap='gray')
        #    ax[1].imshow(model_output.cpu().detach().numpy()[23, :, :], cmap='gray')
        #    plt.show()
        if(epoch % 5000 == 2):
            evaluate(model, False)
            model.train()
    return model

def evaluate(model, show_it):
    model.eval()
    chunked_images, chunked_segmented_images, images_have_tumor = load_data(12,13) 

    model_input = np.copy(chunked_images)
    model_output = chunked_segmented_images
        
    model_input = model_input.transpose((0, 3, 1, 2))
    model_input = torch.from_numpy(model_input).to(device)

    print(model_input.shape)

    model_output = model_output.transpose((0, 1, 2))
    model_output = torch.LongTensor(model_output).to(device)



    outputs = model(model_input)

    #criterion = nn.MSELoss()
    criterion = nn.CrossEntropyLoss()
    loss = criterion(outputs, model_output)
    print(loss.item())

    show_it = True
    if show_it:
        f, ax = plt.subplots(2, 2)
        ax[0, 0].imshow(model_input.cpu().detach().numpy()[11, 1:4, :, :].transpose((1, 2, 0)))
        ax[0, 1].imshow(chunked_images[12, :, :, 1:4])
        ax[1, 0].imshow(np.argmax(outputs.cpu().detach().numpy()[11, :, :], axis=0))
        ax[1, 1].imshow(model_output.cpu().detach().numpy()[11, :, :])
        plt.show()

def main():
    model = train_unit()
    evaluate(model, True)

main()

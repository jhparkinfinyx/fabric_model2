import torch
from torch.nn.modules.distance import PairwiseDistance
import torchvision.transforms as transforms
import model.similarity.dyetec_utils as dyetec_utils
import io
import datetime
from PIL import Image
import numpy as np
import cv2

def run(anchor_image, img_rows):
    '''
    :param anchor_image: 검색할 이미지
    :param comparison_vectors: DB에 저장된 내용 ( (), (), ... )
    :return: 변환된 anchor image, [(similar_score, id_num, id_name, path, info, link), ...] 상위 5가지
    '''

    ##### byte image -> PIL Image
    anchor_image = Image.open(io.BytesIO(anchor_image)).convert('RGB')
    # w, h = anchor_image.size
    # center_w, center_h = w // 2, h // 2
    # if w > h:
    #     anchor_image = anchor_image.crop((center_w - center_h, 0, center_w + center_h, h))
    # else:
    #     anchor_image = anchor_image.crop((0, center_h - center_w, w, center_h + center_w))

    image_trans = image_resize(anchor_image)
    image_trans = np.array(image_trans)
    image_trans = cv2.cvtColor(image_trans, cv2.COLOR_RGB2BGR)
    image_trans = cv2.imencode('.png', image_trans)[1].tostring()

    cuda = True if torch.cuda.is_available() else False

    data_transforms = transforms.Compose([
        transforms.Resize(size=(224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.6071, 0.4609, 0.3944],
            std=[0.2457, 0.2175, 0.2129]
        )
    ])

    ##### model 경로 지정
    if not cuda:
        model_info = torch.load('./model/similarity/model_1.pt', map_location='cpu')
    else:
        model_info = torch.load('./model/similarity/model_1.pt')
    model_architecture = model_info['model_architecture']
    embedding_dimension = model_info['embedding_dimension']
    # Instantiate model
    model = dyetec_utils.set_model_architecture(
        model_architecture=model_architecture,
        pretrained=False,
        embedding_dimension=embedding_dimension
    )
    # Load model to GPU or multiple GPUs if available
    # model, flag_train_multi_gpu = dyetec_utils.set_model_gpu_mode(model)
    model.load_state_dict(model_info['model_state_dict'])
    if cuda:
        model.cuda()
    model.eval()

    if not cuda:
        model_info = torch.load('./model/similarity/model_2.pt', map_location='cpu')
    else:
        model_info = torch.load('./model/similarity/model_2.pt')
    # Instantiate model
    model_ = dyetec_utils.set_model_architecture(
        model_architecture=model_architecture,
        pretrained=False,
        embedding_dimension=embedding_dimension
    )
    # Load model to GPU or multiple GPUs if available
    # model, flag_train_multi_gpu = dyetec_utils.set_model_gpu_mode(model)
    model_.load_state_dict(model_info['model_state_dict'])
    if cuda:
        model_.cuda()
    model_.eval()


    with torch.no_grad():
        # 어떤 한 이미지가 입력됐을 때, 저장된 유사도 벡터 중 비슷한거 찾아서 (유사도, 섬유 이름, 이미지) 보여주기
        l2_distance = PairwiseDistance(p=2)

        # 입력된 이미지
        anchor_image = data_transforms(anchor_image).unsqueeze(0)
        if cuda:
            anchor_image = anchor_image.cuda()
        output1 = model(anchor_image)
        output2 = model_(anchor_image)
        print("output1:", output1)
        print("output2:", output2)
        distance_list = []
        for item in img_rows:
            fileID, filename, filepath, comparison_vector1, comparison_vector2 = item[0], item[1], item[2], str2list(item[3]), str2list(item[4])
            file_info, link = item[5], item[6]
            if cuda:
                comparison_vector1 = torch.as_tensor(comparison_vector1).cuda()
                comparison_vector2 = torch.as_tensor(comparison_vector2).cuda()
            else:
                comparison_vector1 = torch.as_tensor(comparison_vector1)
                comparison_vector2 = torch.as_tensor(comparison_vector2)
            distance1 = l2_distance.forward(output1, comparison_vector1)
            distance2 = l2_distance.forward(output2, comparison_vector2)

            # 유사도 계산 값과 함께, 출력될 값들을 tuple로 저장
            dist = np.mean([distance1.item(), distance2.item()])
            comparison_result = (dist.item(), fileID, filename, filepath, file_info, link)
            distance_list.append(comparison_result)

        distance_list.sort()
        results = []
        for file in distance_list[:5]:
            file = list(file)
            score, info = file[0], file[1:]
            if score > 0.55:
                score = 0.083-score*0.01
                if score < 0:
                    score = 0
            elif score > 0.4:
                score = 100 - 100 * (score / 0.6)
            else:
                score = 100 - 100 * (score / 1.2)
            results.append(tuple([score] + info))

        # [(100.00, 440, 'OK-21_COL', 'static/images/OK-21_COL.jpg', 90.0), (), ...]
        return image_trans, results


def image_resize(image):
    new_width, new_height = 500, 810
    hw_rate = new_height / new_width

    w, h = image.size
    center_w, center_h = w // 2, h // 2
    if w > h:
        new_w_half = int((1 / hw_rate * h) // 2)
        if center_w+new_w_half > w:
            new_h_half = int((hw_rate * w) // 2)
            image_crop = image.crop((0, center_h - new_h_half, w, center_h + new_h_half))
        else:
            image_crop = image.crop((center_w-new_w_half, 0, center_w+new_w_half, h))
    else:
        new_h_half = int((hw_rate * w) // 2)
        if center_h+new_h_half > h:
            new_w_half = int((1 / hw_rate * h) // 2)
            image_crop = image.crop((center_w-new_w_half, 0, center_w+new_w_half, h))
        else:
            image_crop = image.crop((0, center_h - new_h_half, w, center_h + new_h_half))

    image = image_crop.resize((new_width, new_height))
    return image

def str2list(vector_str):
    vector_list = [[]]
    vector_str = vector_str[2:-2].split(', ')

    for value_str in vector_str:
        vector_list[0].append(float(value_str))

    return vector_list

# if __name__ == '__main__':
#     main()
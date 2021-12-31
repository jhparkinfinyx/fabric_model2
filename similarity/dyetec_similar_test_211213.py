import torch
from torch.nn.modules.distance import PairwiseDistance
import torchvision.transforms as transforms
import model.similarity.dyetec_utils as dyetec_utils
import io
import datetime
from PIL import Image


def run(anchor_image, img_rows):
    '''
    :param anchor_image: 검색할 이미지
    :param comparison_vectors: DB에 저장된 내용 ( (), (), ... )
    :return: [(similar_score, id_num, id_name, path), ...] 상위 5가지
    '''

    ##### byte image -> PIL Image
    anchor_image = Image.open(io.BytesIO(anchor_image)).convert('RGB')

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
    model_info = torch.load('./model/similarity/model_resnet34_triplet_epoch_26.pt', map_location=torch.device('cpu'))
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

    with torch.no_grad():
        # 어떤 한 이미지가 입력됐을 때, 저장된 유사도 벡터 중 비슷한거 찾아서 (유사도, 섬유 이름, 이미지) 보여주기
        l2_distance = PairwiseDistance(p=2)

        # 입력된 이미지
        anchor_image = data_transforms(anchor_image).unsqueeze(0)
        if cuda:
            anchor_image = anchor_image.cuda()
        output = model(anchor_image)

        distances = []
        distance_list = []
        for item in img_rows:
            fileID, filename, filepath, comparison_vector = item[0], item[1], item[2], str2list(item[3])
            if cuda:
                comparison_vector = torch.as_tensor(comparison_vector).cuda()
            else:
                comparison_vector = torch.as_tensor(comparison_vector)
            distance = l2_distance.forward(output, comparison_vector)

            # 유사도 계산 값과 함께, 출력될 값들을 tuple로 저장
            print(distance.item())
            comparison_result = (distance.item(), fileID, filename, filepath)
            distance_list.append(comparison_result)
            distances.append(distance.item())

        distance_list_score = []
        lowest_score = max(distances)
        for idx, line in enumerate(distance_list):
            line = list(line)
            dist = line[0]
            score = 100 - 100 * (dist / lowest_score)
            line.append(score)
            line = tuple(line)
            distance_list_score.append(line)
        # 유사도 정렬
        distance_list_score.sort()

        # [(1.5460909605026245, 440, 'OK-21_COL', 'static/images/OK-21_COL.jpg', 90.0), (), ...]
        return distance_list_score[:5]


def str2list(vector_str):
    vector_list = [[]]
    vector_str = vector_str[2:-2].split(', ')

    for value_str in vector_str:
        vector_list[0].append(float(value_str))

    return vector_list

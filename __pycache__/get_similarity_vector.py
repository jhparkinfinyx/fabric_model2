import torch
import torchvision.transforms as transforms
import dyetec_utils
import io
import datetime
from PIL import Image

# def main():
#     # anchor_image:
#     image_path = '/home/bong04/data/dyetec_fabric/classification_data/augmented_images/AK-27_COL/AK-27_COL_1.png'
#     anchor_image = Image.open(image_path).convert('RGB')
#
#     print(run(anchor_image))

def run(anchor_image):
    '''
    :param anchor_image: 검색할 이미지
    :return: (similar_score, id_num, id_name, path), ...] 상위 5가지
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
    model_info = torch.load('./model_resnet34_triplet_epoch_26.pt')
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
        # 이미지 등록을 위해, 해당 이미지의 이미지 유사도 벡터 생성하기
        anchor_image = data_transforms(anchor_image)
        if cuda:
            anchor_image = anchor_image.unsqueeze(0).cuda()
        output = model(anchor_image)

        similarity_vector = output.cpu().detach().tolist()

    return similarity_vector


# if __name__ == '__main__':
#     main()
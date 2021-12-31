import torch
from torch.nn.modules.distance import PairwiseDistance
import torchvision.transforms as transforms
import model.similarity.dyetec_utils as dyetec_utils
import io
import datetime
from PIL import Image

# def main():
#     # anchor_image:
#     image_path = '/home/bong04/data/dyetec_fabric/classification_data/augmented_images/AK-27_COL/AK-27_COL_1.png'
#     anchor_image = Image.open(image_path).convert('RGB')
#
#     # json_path = './similarity_vector.json'
#     # with open(json_path, 'r') as j:
#     #     comparison_vectors = json.load(j)
#     comparison_vectors = (
#         (440, 'OK-21_COL', 'static/images/OK-21_COL.jpg',
#          '[[0.0048230672255158424, -0.012627394869923592, 0.03860443830490112, 0.004468015395104885, -0.03822838515043259, 0.010294136591255665, -0.019504422321915627, 0.01904328353703022, -0.015035426244139671, -0.03578915819525719, 0.18606771528720856, -0.03507915511727333, -0.05492023378610611, 0.048078130930662155, 0.03968840464949608, 0.01819724217057228, -0.014535905793309212, 0.3742207884788513, -0.0020925169810652733, -0.007170516066253185, -0.0006099263555370271, 0.003101867623627186, 0.06249600648880005, 0.009347670711576939, -0.008970260620117188, 0.04866574704647064, 0.026480989530682564, 0.0017779010813683271, 0.016018060967326164, 0.008903669193387032, 0.0015569243114441633, -0.021124690771102905, 0.03792467713356018, -0.025298813357949257, 0.04055051878094673, -0.0026232521049678326, -0.04276950657367706, -0.045399010181427, 0.006626849062740803, 0.01575431600213051, -0.003832218237221241, -0.0319354273378849, -0.022571632638573647, -0.01566407084465027, 0.028426464647054672, 0.02353118732571602, 0.017250068485736847, -0.03640612214803696, 0.044924210757017136, -0.04830709472298622, -0.0016788160428404808, 0.0005436250939965248, -0.03277761861681938, -0.03478293493390083, 0.022643988952040672, -0.015819361433386803, -0.0011510525364428759, -0.022841190919280052, 0.0023405086249113083, -0.008657700382173061, 0.027166545391082764, 0.006270390935242176, 0.01837538182735443, 0.009261268191039562, 0.037037912756204605, 0.004736634902656078, -0.016025163233280182, -0.05865154787898064, -0.018597222864627838, 0.022720560431480408, -0.019245430827140808, -0.03553560748696327, 0.03060668334364891, -0.002471435582265258, 0.022662093862891197, 0.013070889748632908, 0.006758429110050201, -0.010429966263473034, 0.010115047916769981, 0.02361879125237465, -0.006870244164019823, 0.02379077486693859, -0.004910806659609079, -0.006210526451468468, -0.017173562198877335, -0.014276993460953236, 0.01753263548016548, 0.004708179738372564, 0.026201829314231873, 0.033670805394649506, 0.014930768869817257, 0.00501614436507225, 0.1846662163734436, -0.006563256960362196, -0.0036352858878672123, 0.039019372314214706, -0.012138720601797104, 0.028326891362667084, 0.006167789921164513, 0.02711559645831585, -0.0076072933152318, 0.007840652950108051, 0.0017271636752411723, 0.03777695447206497, 0.03743470087647438, 0.018913140520453453, -0.07737717777490616, -0.012389915063977242, -0.00963757373392582, 0.053431231528520584, -0.005920048337429762, -0.023364175111055374, -0.009570380672812462, 0.007534594740718603, 0.023069212213158607, 0.00034469985985197127, 0.0017712521366775036, -0.023578280583024025, 0.023566260933876038, 0.004447247367352247, -0.0605677030980587, 0.007283647544682026, -0.024067256599664688, -0.029603520408272743, 0.011836959980428219, 0.016116956248879433, 0.026169706135988235, 0.05742475762963295, 0.033049628138542175, -0.0393628254532814, 0.004590874072164297, -0.013917994685471058, 0.016131924465298653, 5.6411758123431355e-05, 0.020589111372828484, 0.013368409126996994, 0.03654154762625694, -0.022991972044110298, -0.02227967418730259, 0.03980369120836258, -0.043727707117795944, 0.01732022687792778, -0.030737629160284996, 0.00031849948572926223, 0.029619213193655014, -0.04769473522901535, -0.010658004321157932, 0.02515692450106144, 0.0334099605679512, 0.01665440760552883, -0.006578734144568443, -0.003225200343877077, 0.035536330193281174, -0.056155815720558167, -0.011282263323664665, 0.018463628366589546, 0.031022144481539726, -0.007436680607497692, 0.007370037492364645, 0.01684877835214138, -0.015965107828378677, -0.03380491957068443, -0.021960683166980743, 0.0024036092218011618, -0.012040357105433941, 0.026439134031534195, -0.006494651548564434, 0.006880633533000946, 0.01836145855486393, -0.020609037950634956, 0.033282846212387085, -0.0028049086686223745, -0.012974880635738373, 0.001834532362408936, 0.019928259775042534, -0.014286690391600132, -0.035816024988889694, 0.010634003207087517, 0.004187723621726036, -0.003727420000359416, -0.005187278147786856, -0.020458290353417397, -0.009464303031563759, 0.015676843002438545, -0.005147983320057392, -0.028454767540097237, -0.017643911764025688, 0.009723019786179066, 0.047221213579177856, 0.00403759628534317, -0.005528458394110203, -0.008117768913507462, -0.002946970984339714, -0.00014325625670608133, -0.012101849541068077, -0.0069358330219984055, 0.0044467151165008545, -0.003292707959190011, 0.01693004183471203, 0.003994602710008621, -0.0005451341858133674, -0.007566757500171661, 0.03636561706662178, 0.0007999415975064039, -0.04999365285038948, -0.024136601015925407, 0.015325305052101612, 0.02959228679537773, -0.05163972079753876, 0.03546446934342384, 0.00402837572619319, 0.0010207812301814556, 0.015054944902658463, -0.033944252878427505, 0.010487856343388557, 0.005846772342920303, 0.013830289244651794, 0.029181161895394325, -0.03452465683221817, 0.03755543753504753, -0.027599040418863297, 0.019618716090917587, 0.034822456538677216, 0.004160291515290737, 0.0357193760573864, -0.026046423241496086, -0.004763907752931118, 0.023634381592273712, 0.00837996881455183, -0.023241672664880753, 0.06140849366784096, 0.004064662382006645, 0.013078413903713226, -0.03375828266143799, -0.010910874232649803, 0.00931475032120943, -0.02064300887286663, -0.02594696916639805, -0.015484880656003952, 0.02297688089311123, 0.003916766028851271, 0.008845959790050983, 0.007163407746702433, 0.02929312363266945, -0.004870978184044361, 0.019780084490776062, 0.010678868740797043, 0.0316653773188591, 0.01109339576214552, 0.02553967386484146, -0.048375267535448074, 0.02122480608522892, -0.003237128257751465, 0.04267247021198273, 0.018414288759231567, -0.037401989102363586, -0.013888413086533546, 0.008026054129004478, 0.03524850681424141, -0.011093338951468468, -0.013575481250882149, -0.027121135964989662, 0.03788959980010986, -0.017626041546463966, 0.008210225030779839, -0.01817329227924347, 0.043945468962192535, 0.03545662388205528, -0.015202284790575504, -0.00832133088260889, -0.045112818479537964, 0.010366960428655148, 0.010466205887496471, 0.0185563825070858, -0.027704322710633278, 0.010252706706523895, -0.005463873036205769, -0.024433890357613564, -0.01625404693186283, 0.020118609070777893, -0.01203898899257183, -0.03424438089132309, 0.008672114461660385, -0.006366165354847908, 0.018799135461449623, 0.01119394414126873, 0.04044485464692116, 0.00042523309821262956, -0.01229654811322689, -0.028967788442969322, 0.01453687809407711, -0.024907929822802544, -0.0270085372030735, -0.035404957830905914, -0.010761851444840431, 0.012981368228793144, -0.0033071618527173996, -0.017704494297504425, -0.035406965762376785, -0.015561748296022415, 0.04159663990139961, -0.01815754361450672, -0.023695020005106926, 0.038334764540195465, 0.024294253438711166, -0.0323103629052639, 0.032724034041166306, 0.019275056198239326, 0.020927000790834427, 0.027666425332427025, 0.0010418880265206099, 0.036580655723810196, -0.010012274608016014, -0.04162364825606346, -0.030498608946800232, 0.000796377717051655, 0.02326701395213604, -0.008582077920436859, -0.20033463835716248, 0.008904780261218548, -0.008565142750740051, 0.022844435647130013, -0.014962858520448208, 0.038293637335300446, -0.006705217529088259, -0.05003807693719864, 0.0272307600826025, 0.0037386640906333923, -0.6677421927452087, 0.01323652546852827, -0.02598833292722702, -0.003086007898673415, -0.0011840215884149075, -0.026136020198464394, -0.013162493705749512, 0.05392637476325035, -0.0021861293353140354, 0.029249846935272217, 0.002431767527014017, -0.02124159410595894, -0.03948136791586876, -0.021923216059803963, -0.014970927499234676, -0.0017049408052116632, 0.029605839401483536, -0.020857280120253563, 0.022663820534944534, 0.02389393374323845, 0.025586821138858795, 0.03253698721528053, -0.006201412994414568, 0.027256431058049202, 0.02081921510398388, 0.03796883672475815, 0.029210973531007767, 0.006493078079074621, 0.02209429256618023, -0.03221758082509041, -0.022969171404838562, 0.010004518553614616, -0.0407232791185379, 0.030662156641483307, 0.023078452795743942, -0.016504408791661263, 0.020596284419298172, -0.033443886786699295, -0.0036827081348747015, -0.02871040441095829, 0.025761015713214874, -0.027521725744009018, -0.0032131816260516644, 0.03861971199512482, 0.0038778167217969894, 0.021351249888539314, -0.0005178876454010606, 0.014937106519937515, 0.033591218292713165, 0.020342502743005753, -0.013277602382004261, 0.035948313772678375, 0.0233322586864233, -0.022260015830397606, -0.028966709971427917, -0.01466155331581831, -0.03529554605484009, 0.00621913792565465, 0.02534189634025097, -0.01651366613805294, -0.017074668779969215, 0.05787869170308113, 0.006218395661562681, 0.015202231705188751, 0.029879389330744743, 0.03817460685968399, 0.01092924177646637, -0.023398730903863907, 0.023633720353245735, -0.0399119108915329, 0.059012413024902344, 0.006491244770586491, -0.023627188056707382, 0.00956464372575283, -0.002961781108751893, -0.008876239880919456, -0.03365432843565941, -0.008617980405688286, 0.015872085466980934, 0.010149889625608921, 0.009114868007600307, -0.035891737788915634, 0.018952421844005585, 0.030362416058778763, 0.019985729828476906, 0.029016884043812752, 0.013186599127948284, -0.014336076565086842, 0.022741882130503654, -0.011816194280982018, 0.015201601199805737, -0.022479748353362083, 0.0032849875278770924, -0.01593896560370922, 0.0081724151968956, 0.0287568811327219, 0.06848219782114029, 0.002818155102431774, 0.02984004281461239, 0.05141565203666687, -0.004079962149262428, 0.012769586406648159, 0.02524154633283615, -0.012220933102071285, -0.014232520014047623, -0.011992912739515305, -0.0298000518232584, 0.0076166908256709576, -0.015209016390144825, -0.04239501431584358, -0.008212175220251083, -0.04140901565551758, -0.001850405242294073, 0.002434222027659416, 0.015895606949925423, 0.004946987610310316, -0.008907732553780079, 0.0264538936316967, -1.781487480911892e-05, 0.009787827730178833, 0.03627683222293854, 0.010729675181210041, 0.014037840068340302, 0.03445473313331604, -0.038614120334386826, -0.008796044625341892, -0.016392985358834267, -0.002874145284295082, -0.02494732476770878, -0.03925241902470589, 0.005621033720672131, -0.010592005215585232, -0.012913105078041553, -8.637057908345014e-05, -0.03707653284072876, -0.03318606689572334, -0.020651446655392647, -0.03008124977350235, 0.008848666213452816, -0.024880874902009964, 0.016286401078104973, -0.02820851467549801, 0.03223331272602081, -0.01846848800778389, -0.020214051008224487, 0.003726540133357048, 0.02592819184064865, 0.010768999345600605, 0.02328258380293846, -0.03541572391986847, 0.006600126624107361, 0.016002142801880836, 0.01984008215367794, 0.021014433354139328, 0.0018480460857972503, -0.05981333553791046, -0.016721535474061966, -0.008431192487478256, -0.023726578801870346, -0.024544164538383484, -0.0050295209512114525, -0.02194165624678135, 0.024636058136820793, -0.022798195481300354, 0.0012387542519718409, -0.033299770206213, -0.009852073155343533, -0.031008511781692505, -0.021048488095402718, 0.023432500660419464, 0.007386175915598869, 0.012331649661064148, 0.009301121346652508, 0.017089325934648514, 0.004510717466473579, 0.009453699924051762, 0.01767655462026596, -0.031883616000413895, 0.041781820356845856, 0.01799965649843216, -0.007797375787049532, -0.032563209533691406, 0.04704979807138443, 0.00445464625954628]]',
#          datetime.datetime(2021, 11, 23, 16, 20, 22), datetime.datetime(2021, 11, 19, 11, 20, 40)),
#     )
#
#     print(run(anchor_image, comparison_vectors))

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
    model_info = torch.load('./model/similarity/model_resnet34_triplet_epoch_26.pt')
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

        distance_list = []
        for item in img_rows:
            fileID, filename, filepath, comparison_vector = item[0], item[1], item[2], str2list(item[3])
            if cuda:
                comparison_vector = torch.as_tensor(comparison_vector).cuda()
            else:
                comparison_vector = torch.as_tensor(comparison_vector)
            distance = l2_distance.forward(output, comparison_vector)

            # 유사도 계산 값과 함께, 출력될 값들을 tuple로 저장
            comparison_result = (distance.item(), fileID, filename, filepath)
            distance_list.append(comparison_result)

        # 유사도 정렬
        distance_list.sort()

        # [(1.5460909605026245, 440, 'OK-21_COL', 'static/images/OK-21_COL.jpg'), (), ...]
        return distance_list[:5]


def str2list(vector_str):
    vector_list = [[]]
    vector_str = vector_str[2:-2].split(', ')

    for value_str in vector_str:
        vector_list[0].append(float(value_str))

    return vector_list

# if __name__ == '__main__':
#     main()
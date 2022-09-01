import cv2
# from PIL import Image
import argparse
import torchvision.utils as vutils
import torchvision.transforms as transforms
import torch.optim as optim
from models import *

parser = argparse.ArgumentParser()
parser.add_argument('--epoch', type=int, default=1, help='number of epochs to train for')
parser.add_argument('--bs', type=int, default=1, help='number of batch size')
parser.add_argument('--cuda', type=bool, default=True, help='enables cuda')
parser.add_argument('--nms_thresh', type=float, default=0.4, help='NMS thresh')
parser.add_argument('--max_p', type=int, default=4700, help='max number of pixels that can change')
parser.add_argument('--minN', type=int, default=0, help='min idx of images')
parser.add_argument('--maxN', type=int, default=999, help='max idx of images')
parser.add_argument('--save', default='results', help='folder to output images and model checkpoints')
parser.add_argument('--seed', type=int, default=0, help='random seed')

# important argument
parser.add_argument('--max_iter', type=int, default=1000, help='max number of iterations to find adversarial example')
parser.add_argument('--max_loss', type=int, default=1.5, help='max value of loss to stop the training')
# parser.add_argument('--imgsz', type=list, default=[608, 608], help='input image size')
# parser.add_argument('--patchsz', type=list, default=[345, 198], help='w and h of patch size')
parser.add_argument('--imgsz', type=list, default=[960, 960], help='input image size')
parser.add_argument('--patchsz', type=list, default=[206, 175], help='w and h of patch size')
parser.add_argument('--conf_thresh', type=float, default=0.5, help='conf thresh')

args = parser.parse_args()

torch.manual_seed(args.seed)
torch.cuda.manual_seed(args.seed)
np.random.seed(args.seed)
os.makedirs(args.save, exist_ok=True)

pre = transforms.Compose([transforms.ToTensor()])
nor = transforms.Normalize([123.675 / 255., 116.28 / 255., 103.53 / 255.], [58.395 / 255., 57.12 / 255., 57.375 / 255.])

model1 = Yolov4(yolov4conv137weight=None, n_classes=80, inference=True)
pretrained_dict = torch.load('checkpoints/yolov4.pth', map_location=torch.device('cuda'))
model1.load_state_dict(pretrained_dict)
model1.eval().cuda()


# cross mask
# def get_mask(image, patch):
#     image = cv2.imread(image)
#     image = cv2.resize(image, (args.imgsz, args.imgsz))
#     mask = torch.zeros_like(patch).cuda()  # mask initiation
#     results = inference_detector(model2, image)
#
#     box = []
#     for cls, b in enumerate(results):
#         if cls in [2, 5, 7] and len(b) != 0:  # 2 5 7means car bus truck
#             box.append(b)
#     bbox = np.asarray(box).reshape(-1, 5)
#     bbox = bbox[bbox[:, 4] > args.conf_thresh]
#
#     num = bbox.shape[0]
#
#     if num > 5: num = 5
#     if num == 0: return mask.float().cuda()
#     for i in range(num):
#         xc = int(args.patchsz[0] / 2)
#         yc = int(args.patchsz[1] / 2)
#         mask[:, :, yc - 1:yc + 2, 0:args.patchsz[0]] = 1
#         mask[:, :, 0:args.patchsz[1], xc - 1:xc + 2] = 1
#     mask = mask.float().cuda()
#
#     # for i in range(num):
#     #     xc = int((bbox[i,0]+bbox[i,2])/2)
#     #     yc = int((bbox[i,1]+bbox[i,3])/2)
#     #     #w = int(bbox[i,2]-bbox[i,0])
#     #     #h = int(bbox[i,3]-bbox[i,1])
#     #     x1 = int(bbox[i, 0])
#     #     x2 = int(bbox[i, 2])
#     #     y1 = int(bbox[i, 1])
#     #     y2 = int(bbox[i, 3])
#     #     mask[:,:,yc-1:yc+2,x1:x2] = 1
#     #     mask[:,:,y1:y2,xc-1:xc+2] = 1
#     # mask = mask.float().cuda()
#
#     return mask
#
#
# # feng mask
# def get_mask2(image, patch):
#     image = cv2.imread(image)
#     image = cv2.resize(image, (args.imgsz, args.imgsz))
#     mask = torch.zeros_like(patch).cuda()  # mask initiation
#     results = inference_detector(model2, image)
#
#     box = []
#     for cls, b in enumerate(results):
#         if cls in [2, 5, 7] and len(b) != 0:  # 2 5 7means car bus truck
#             box.append(b)
#     bbox = np.asarray(box).reshape(-1, 5)
#     bbox = bbox[bbox[:, 4] > args.conf_thresh]
#
#     num = bbox.shape[0]
#
#     if num > 5: num = 5
#     if num == 0: return mask.float().cuda()
#     for i in range(num):
#         xc = int(args.patchsz[0] / 2)
#         yc = int(args.patchsz[1] / 2)
#         mask[:, :, yc - int(args.patchsz[1] / 4) - 1: yc - int(args.patchsz[1] / 4) + 2, 0:args.patchsz[0]] = 1
#         mask[:, :, yc + int(args.patchsz[1] / 4) - 1:yc + int(args.patchsz[1] / 4) + 2, 0:args.patchsz[0]] = 1
#         mask[:, :, yc - 1:yc + 2, 0:args.patchsz[0]] = 1
#         mask[:, :, 0:args.patchsz[1], xc - 1:xc + 2] = 1
#     mask = mask.float().cuda()
#     return mask
#
#
# # heng feng
# def get_mask3(image, patch):
#     image = cv2.imread(image)
#     image = cv2.resize(image, (args.imgsz, args.imgsz))
#     mask = torch.zeros_like(patch).cuda()  # mask initiation
#     results = inference_detector(model2, image)
#
#     box = []
#     for cls, b in enumerate(results):
#         if cls in [2, 5, 7] and len(b) != 0:  # 2 5 7means car bus truck
#             box.append(b)
#     bbox = np.asarray(box).reshape(-1, 5)
#     bbox = bbox[bbox[:, 4] > args.conf_thresh]
#
#     num = bbox.shape[0]
#
#     if num > 5: num = 5
#     if num == 0: return mask.float().cuda()
#     for i in range(num):
#         xc = int(args.patchsz[0] / 2)
#         yc = int(args.patchsz[1] / 2)
#         mask[:, :, yc - 1:yc + 2, 0:args.patchsz[0]] = 1
#         mask[:, :, 0:args.patchsz[1], xc - 1:xc + 2] = 1
#         mask[:, :, 0:args.patchsz[1], xc - int(args.patchsz[0] / 4) - 1:xc - int(args.patchsz[0] / 4) + 2] = 1
#         mask[:, :, 0:args.patchsz[1], xc + int(args.patchsz[0] / 4) - 1:xc + int(args.patchsz[0] / 4) + 2] = 1
#
#     mask = mask.float().cuda()
#     return mask


# mi mask

def get_mask4(image, patch):
    width = 100
    mask = np.zeros((1, 1260, 2790, 3))
    #mask = np.zeros((1, args.patchsz[1], args.patchsz[0], 3))
    xc = int(2790 / 2)
    yc = int(1260 / 2)

    mask[:, yc - width:yc + width, 0: 2790, :] = 1
    mask[:, 0: 1260, xc - width: xc + width, :] = 1

    cv2.line(mask[0], (xc, 0), (xc, 1260), (1, 1, 1), width)
    cv2.line(mask[0], (0, yc), (2790, yc), (1, 1, 1), width)
    cv2.line(mask[0], (0, 0), (2790, 1260), (1, 1, 1), width)
    cv2.line(mask[0], (0, 1260), (2790, 0), (1, 1, 1), width)

    mask = torch.from_numpy(mask).permute(0, 3, 1, 2)
    mask = mask.float().cuda()

    return mask


files = os.listdir('./images')
files.sort()
# files = files[100:120]


count = 0  # number of images use for train
for file in files:
    flag = 0
    if count < args.minN:
        count += 1
        continue
    if count > args.maxN:
        break
    print(file)


    img_pil = cv2.imread('./images/' + file)
    img_pil = cv2.cvtColor(img_pil, cv2.COLOR_BGR2RGB)
    img_pil = cv2.resize(img_pil, (args.imgsz[0], args.imgsz[1]))
    img_pil = np.transpose(img_pil, (2, 0, 1))
    img = torch.from_numpy(img_pil / 255.).float()
    img = img.unsqueeze(0).cuda()


    patch = torch.zeros((1, 3, 1260, 2790)).cuda()
    patch.requires_grad = True
    img_path = './images/' + file

    # mask = get_mask(img_path, patch)
    # mask = get_mask2(img_path, patch)
    # mask = get_mask3(img_path, patch)
    mask = get_mask4(img_path, patch)

    optimizer = optim.SGD([patch], lr=32 / 255.)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.max_iter, eta_min=0.001)
    for i in range(args.max_iter):
        # img[:, :, 216:612, 76:766] = torch.mul(F.interpolate(patch, size=(args.patchsz[0], args.patchsz[1])), mask)  # 608 :  108 306 38 383
        img[::, :, 375:550, 244:450] = torch.mul(F.interpolate(patch, size=(args.patchsz[1], args.patchsz[0])),
                                               F.interpolate(mask, size=(args.patchsz[1], args.patchsz[0])))
        # img[:, :, 108:306, 38:383] = torch.mul(F.interpolate(patch, size=(args.patchsz[1], args.patchsz[0])),
        #                                        F.interpolate(mask, size=(args.patchsz[1], args.patchsz[0])))
        imgp = img
        out1 = model1(imgp)

        # print(out1[2][:,:,2])
        # print(out1[2][:,:,5])
        # print(out1[2][:,:,7])
        # print(out1[2][:,:,2] + out1[2][:,:,5]  + out1[2][:,:,7])
        # loss = torch.max(out1[2][:,:,2] + out1[2][:,:,5]  + out1[2][:,:,7])

        # out1[2][0] = out1[2][0][out1[1][0, :] > 0.5]
        # print((out1[1][0][:, 0] > 0.5).shape)
        # print((out1[2][0][:, 1] > 0).shape)
        # out1[2][0]  = out1[2][0][out1[2][0][:, 1] > 0]

        loss = torch.max(out1[2][:, :, 2]) + torch.max(out1[2][:, :, 5]) + torch.max(out1[2][:, :, 7])
        print('Num:{:4d}, Iter:{:4d}, Loss:{:.4f}'.format(count, i, loss.item()))
        if loss.item() < args.max_loss:
            flag = 1
            break

        optimizer.zero_grad()
        loss.backward(retain_graph=True)
        patch.grad = torch.sign(patch.grad)
        optimizer.step()
        scheduler.step()
        patch.data.clamp_(0, 1)  # patch values belongs to (0, 1)



    count += 1

    # save patch and mask
    patch_save = patch.clone().detach()
    vutils.save_image(patch_save, args.save + '/' + 'patch.png')
    mask_save = mask.clone().detach()
    vutils.save_image(mask_save, args.save + '/' + 'mask_ori.png')
    imgp_save = imgp.clone().detach()
    vutils.save_image(imgp_save, args.save + '/' + 'img.png')

torch.cuda.empty_cache()
print('Job is not finish')

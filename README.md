### Simple implementation of PRNet (torch version)

- *Notice:* This project is based on ***PyTorch***. And ***tensorflow version*** is ***[here](https://github.com/MouxiaoHuang/myPRNet)***.

---

##### 1. Environment required

> - torch == 1.3.1 (other versions would be Okay)
> - numpy
> - matplotlib
> - opencv-python
> - scipy

##### 2. Introductions of some files

> - ***trainNet1.py*** is the implementations of training PRNet model, and model will be saved in  *model/*.
> - [landmarks detector tool](https://1drv.ms/u/s!AsZLFh2eAFyhgYks-jmES1KYEdtqxA?e=VhE96X) should be downloaded and put into *Data/face_detector/*.
> - **Generate training data:**
>   - Same as [tensorflow version](https://github.com/MouxiaoHuang/myPRNet).

***NOTICE:*** Be careful of **all the PATHS** used in these files, you **MUST** modify them by yourself.

---

Thanks for these contributers and their excellent works:

- [YadiraF/PRNet](https://github.com/YadiraF/PRNet)
- [YadiraF/face3d](https://github.com/YadiraF/face3d)
- [jnulzl/PRNet-Train](https://github.com/jnulzl/PRNet-Train)
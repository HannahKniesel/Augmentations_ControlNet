
from mmseg.apis import init_model
import argparse
import torch
from pathlib import Path


if __name__ == "__main__":

    print("******************************")
    print("AUGMENTATIONS")
    print("******************************")

    # Args Parser
    parser = argparse.ArgumentParser(description='Augmentations')
    parser.add_argument('--config_file', type = str, default="./configs/sem_fpn/fpn_r50_4xb4-160k_ade20k-512x512_noaug.py")
    parser.add_argument('--ckpt_file', type = str, default="./work_dirs/fpn_r50_4xb4-160k_ade20k-512x512_noaug/20240127_201404/best_mIoU_epoch_136.pth")
    parser.add_argument('--train', action='store_true')
    parser.add_argument('--eval', action='store_true')
    parser.add_argument('--input_size', type = int, nargs=4, default=[1,3,512,512])


    args = parser.parse_args()
    print(f"Parameters: {args}")

    
    model = init_model(args.config_file, args.ckpt_file) # datapreprocessor is saved in model (check model.data_preprocessor.mean)


    save_path = str(Path(args.ckpt_file).parent)
    example_forward_input = torch.rand(args.input_size)
    if(args.train):
        module = torch.jit.trace(model.cpu().train(), example_forward_input, check_trace=False)
        module.save(save_path+'/train_model_scripted.pt') # Save
        print(f"INFO::Save train model to {save_path}/train_model_scripted.pt")

    if(args.eval):
        module = torch.jit.trace(model.cpu().eval(), example_forward_input, check_trace=False)
        module.save(save_path+'/eval_model_scripted.pt') # Save
        print(f"INFO::Save eval model to {save_path}/eval_model_scripted.pt")
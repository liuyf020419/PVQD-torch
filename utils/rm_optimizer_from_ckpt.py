import torch


def run(old_ckpt, new_ckpt):
    state = torch.load(old_ckpt, map_location='cpu')
    new_state = {}
    for k, v in state.items():
        if k not in ['optimizer', 'data', 'update_num']:
            new_state[k] = v
        # if k in ['args', 'cfg', 'model', 'optimizer_history']:
        #     new_state[k] = v

    with open(new_ckpt, 'wb') as f:
        torch.save(new_state, f)


if __name__ == '__main__':
    old_ckpt = '/home/liuyf/proteins/PVQD-git/ckpt/uncond_gen/checkpoint/checkpoint_last.pt'
    new_ckpt = '/home/liuyf/proteins/PVQD-git/ckpt/uncond_gen/checkpoint/checkpoint_last_.pt'
    run(old_ckpt, new_ckpt)